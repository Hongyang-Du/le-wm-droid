"""
LeWM DROID MPC Policy Server
=============================
Loads a trained MultiViewJEPA checkpoint and runs CEM-based goal-image MPC.

Start on this server:
    cd /workspace/le-wm-droid
    python -m eval_droid.policy_server \
        --ckpt /root/.stable_worldmodel/lewm_droid_epoch_10_object.ckpt \
        --port 8000

Simulator (on another server) calls:
    POST /reset  {"goal_images": {"pixels_0": <b64-JPEG>, "pixels_1": ..., "pixels_2": ...}}
    POST /act    {"images": {"pixels_0": <b64-JPEG>, ...}, "proprio": [14 floats]}
    <- {"action": [7 floats],           # first sub-step Δcartesian(6)+Δgripper(1)
        "action_block": [[7 floats]×5]} # full 35D block (5 sub-steps)
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from torchvision.transforms import v2 as T

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── constants matching training ────────────────────────────────────────────────
HISTORY_SIZE = 3
FRAMESKIP    = 5            # action block = 5 × 7D deltas
ACTION_DIM   = FRAMESKIP * 7  # 35

_DELTA_MEAN = torch.tensor(
    [0.0035, -0.0001, -0.0033, 0.0081, -0.0022, -0.0018, 0.005] * FRAMESKIP
)  # (35,)
_DELTA_STD = torch.tensor(
    [0.019, 0.0242, 0.0261, 1.5526, 0.0588, 0.1768, 0.112] * FRAMESKIP
).clamp(min=1e-4)  # (35,)

_IMG_TRANSFORM = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.Resize((224, 224)),
])

# ── global state ───────────────────────────────────────────────────────────────
app    = FastAPI()
model  = None     # MultiViewJEPA (loaded from ckpt)
device = None

# per-episode state
_goal_emb:      torch.Tensor | None = None   # (1, 1, D)
_emb_history:   deque = deque(maxlen=HISTORY_SIZE)         # each: (1, D)
_act_norm_hist: deque = deque(maxlen=HISTORY_SIZE - 1)     # each: (1, 35) normalised

# ── helpers ────────────────────────────────────────────────────────────────────

def _decode_img(b64: str) -> torch.Tensor:
    """base64-encoded JPEG/PNG → (3, 224, 224) float tensor."""
    data = base64.b64decode(b64)
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    return _IMG_TRANSFORM(np.array(img))   # (3, 224, 224)


@torch.no_grad()
def _encode_obs(images: dict[str, str]) -> torch.Tensor:
    """3-view images → (1, D) embedding via MultiViewJEPA encoder+projector."""
    cls_tokens = []
    for key in [f"pixels_{i}" for i in range(3)]:
        img = _decode_img(images[key]).unsqueeze(0).to(device)   # (1, 3, 224, 224)
        out = model.encoder(img, interpolate_pos_encoding=True)
        cls_tokens.append(out.last_hidden_state[:, 0])            # (1, D_hidden)
    multi = torch.cat(cls_tokens, dim=-1)                         # (1, 3*D_hidden)
    return model.projector(multi)                                  # (1, D_embed)


def _build_ctx_emb() -> torch.Tensor:
    """Stack obs history → (1, HISTORY_SIZE, D); left-pad if history short."""
    hist = list(_emb_history)
    while len(hist) < HISTORY_SIZE:
        hist.insert(0, hist[0])          # repeat earliest frame
    return torch.stack(hist, dim=1)      # (1, 3, D)


def _build_ctx_act_norm() -> torch.Tensor:
    """Stack past action history → (1, HISTORY_SIZE-1, 35); left-pad with zeros."""
    hist = list(_act_norm_hist)
    while len(hist) < HISTORY_SIZE - 1:
        hist.insert(0, torch.zeros(1, ACTION_DIM, device=device))
    return torch.stack(hist, dim=1)      # (1, 2, 35)


def _project_l1_per_substep(action_raw: torch.Tensor, radius: float = 0.075) -> torch.Tensor:
    """
    Project each 7D sub-action onto the L1-ball of given radius.
    Matches V-JEPA 2 AC constraint (≈7.5cm max end-effector displacement per sub-step).
    action_raw: (N, 35) raw denormalised actions
    returns:    (N, 35) projected
    """
    x = action_raw.view(-1, FRAMESKIP, 7)           # (N, 5, 7)
    l1 = x.abs().sum(dim=-1, keepdim=True)          # (N, 5, 1)
    scale = (radius / l1.clamp(min=radius))         # clamp: no-op when already inside ball
    return (x * scale).view(-1, ACTION_DIM)         # (N, 35)


@torch.no_grad()
def _cem_plan(
    ctx_emb:   torch.Tensor,   # (1, 3, D)
    past_act:  torch.Tensor,   # (1, 2, 35) normalised
    goal_emb:  torch.Tensor,   # (1, 1, D)
    horizon:   int   = 4,
    n_samples: int   = 800,
    n_elites:  int   = 64,
    n_iters:   int   = 10,
    l1_radius: float = 0.075,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-step CEM planner (horizon=4 obs-steps ≈ 1.33s lookahead).
    Samples (N, H, 35) action sequences, rolls out H steps autoregressively,
    minimises MSE between final predicted embedding and goal embedding.

    Returns:
        action_norm (35,) – normalised first-step action (stored in history)
        action_raw  (35,) – L1-projected denormalised first-step action for robot
    """
    # CEM distribution over (horizon, 35) action sequences
    mu    = torch.zeros(horizon, ACTION_DIM, device=device)
    sigma = torch.ones(horizon, ACTION_DIM, device=device)

    goal_exp     = goal_emb.expand(n_samples, -1, -1)              # (N, 1, D)
    past_act_exp = past_act.expand(n_samples, -1, -1)              # (N, 2, 35)

    for _ in range(n_iters):
        # ── 1. Sample & constrain action sequences ──────────────────────────
        # (N, H, 35) in normalised space
        cands_norm = mu + sigma * torch.randn(n_samples, horizon, ACTION_DIM, device=device)

        # Denormalise → L1-project each sub-step → re-normalise
        cands_raw = cands_norm * _DELTA_STD + _DELTA_MEAN          # (N, H, 35)
        cands_raw = _project_l1_per_substep(
            cands_raw.reshape(n_samples * horizon, ACTION_DIM), l1_radius
        ).reshape(n_samples, horizon, ACTION_DIM)
        cands_norm = (cands_raw - _DELTA_MEAN) / _DELTA_STD        # (N, H, 35)

        # ── 2. Encode all H steps at once ───────────────────────────────────
        # (N, H, 35) → (N, H, D)
        all_act_emb = model.action_encoder(cands_norm)

        # ── 3. Autoregressive rollout over H steps ───────────────────────────
        # emb_seq starts as obs history; act_seq starts as past executed actions
        emb_seq = ctx_emb.expand(n_samples, -1, -1).clone()        # (N, 3, D)
        act_seq = model.action_encoder(past_act_exp)                # (N, 2, D)

        for h in range(horizon):
            ctx_e = emb_seq[:, -HISTORY_SIZE:]                      # (N, 3, D)
            ctx_a = torch.cat(
                [act_seq[:, -(HISTORY_SIZE - 1):], all_act_emb[:, h:h+1]], dim=1
            )                                                        # (N, 3, D)

            pred     = model.predict(ctx_e, ctx_a)                  # (N, 3, D)
            next_emb = pred[:, -1:]                                  # (N, 1, D)

            emb_seq = torch.cat([emb_seq, next_emb], dim=1)         # (N, 3+h+1, D)
            act_seq = torch.cat([act_seq, all_act_emb[:, h:h+1]], dim=1)  # (N, 2+h+1, D)

        # ── 4. Cost at final predicted embedding ─────────────────────────────
        final_emb = emb_seq[:, -1:]                                 # (N, 1, D)
        costs = F.mse_loss(final_emb, goal_exp.detach(), reduction="none") \
                  .sum(dim=-1).squeeze(-1)                          # (N,)

        # ── 5. CEM update ────────────────────────────────────────────────────
        elite_idx   = costs.argsort()[:n_elites]
        elites_norm = cands_norm[elite_idx]                         # (E, H, 35)
        mu    = elites_norm.mean(0)                                  # (H, 35)
        sigma = elites_norm.std(0).clamp(min=0.01)

    # Return first-step action (denormalised + projected)
    first_raw  = (mu[0] * _DELTA_STD + _DELTA_MEAN)
    first_raw  = _project_l1_per_substep(first_raw.unsqueeze(0), l1_radius).squeeze(0)
    first_norm = (first_raw - _DELTA_MEAN) / _DELTA_STD

    return first_norm.cpu().numpy(), first_raw.cpu().numpy()


# ── API ────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    goal_images: dict[str, str]   # "pixels_0/1/2" → base64-encoded image


class ActRequest(BaseModel):
    images: dict[str, str]        # "pixels_0/1/2" → base64-encoded image
    proprio: list[float] = []     # 14D proprio (available for future use)


@app.post("/reset")
def reset(req: ResetRequest):
    global _goal_emb, _emb_history, _act_norm_hist
    _emb_history.clear()
    _act_norm_hist.clear()
    goal = _encode_obs(req.goal_images)     # (1, D)
    _goal_emb = goal.unsqueeze(1)           # (1, 1, D)
    return {"status": "ok"}


@app.post("/act")
def act(req: ActRequest):
    """One MPC step: encode obs, plan, return 7D delta action."""
    assert _goal_emb is not None, "Call /reset before /act"

    # 1. Encode current observation
    curr_emb = _encode_obs(req.images)       # (1, D)
    _emb_history.append(curr_emb)

    # 2. Build context tensors
    ctx_emb  = _build_ctx_emb()             # (1, 3, D)
    past_act = _build_ctx_act_norm()         # (1, 2, 35)

    # 3. CEM planning
    action_norm, action_raw = _cem_plan(ctx_emb, past_act, _goal_emb)

    # 4. Store this step's action for next step's context
    _act_norm_hist.append(
        torch.tensor(action_norm, device=device).unsqueeze(0)   # (1, 35)
    )

    # 5. Return first sub-step (7D) and full block
    return {
        "action":       action_raw[:7].tolist(),           # first 7D delta
        "action_block": action_raw.reshape(5, 7).tolist(), # all 5 sub-steps
    }


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    global model, device, _DELTA_MEAN, _DELTA_STD

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/root/.stable_worldmodel/lewm_droid_epoch_10_object.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--horizon",   type=int,   default=4)
    parser.add_argument("--n_samples", type=int,   default=800)
    parser.add_argument("--n_elites",  type=int,   default=64)
    parser.add_argument("--n_iters",   type=int,   default=10)
    parser.add_argument("--l1_radius", type=float, default=0.075)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _DELTA_MEAN = _DELTA_MEAN.to(device)
    _DELTA_STD  = _DELTA_STD.to(device)

    print(f"Loading checkpoint: {args.ckpt}")
    model = torch.load(args.ckpt, map_location=device, weights_only=False)

    # Fix transformers version mismatch: rebuild ViTConfig from saved arch params.
    # Newer transformers added/renamed many attributes; safest fix is a fresh config.
    from transformers import ViTConfig
    old = model.encoder.config
    new_cfg = ViTConfig(
        hidden_size=old.hidden_size,
        num_hidden_layers=old.num_hidden_layers,
        num_attention_heads=old.num_attention_heads,
        intermediate_size=old.intermediate_size,
        hidden_act=old.hidden_act,
        hidden_dropout_prob=old.hidden_dropout_prob,
        attention_probs_dropout_prob=old.attention_probs_dropout_prob,
        initializer_range=old.initializer_range,
        layer_norm_eps=old.layer_norm_eps,
        image_size=old.image_size,
        patch_size=old.patch_size,
        num_channels=old.num_channels,
        qkv_bias=old.qkv_bias,
        encoder_stride=old.encoder_stride,
    )
    model.encoder.config = new_cfg

    model.eval()
    model.requires_grad_(False)
    print(f"Model loaded on {device}. Starting server at {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
