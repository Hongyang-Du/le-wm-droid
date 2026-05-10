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


@torch.no_grad()
def _cem_plan(
    ctx_emb:    torch.Tensor,   # (1, 3, D)
    past_act:   torch.Tensor,   # (1, 2, 35) normalised
    goal_emb:   torch.Tensor,   # (1, 1, D)
    n_samples:  int = 300,
    n_elites:   int = 10,
    n_iters:    int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        action_norm (35,) – normalised action (for storing in history)
        action_raw  (35,) – denormalised deltas to send to robot
    """
    mu    = torch.zeros(ACTION_DIM, device=device)
    sigma = torch.ones(ACTION_DIM, device=device)

    goal_exp = goal_emb.expand(n_samples, -1, -1)   # (N, 1, D)

    for _ in range(n_iters):
        # (N, 35) candidate actions in normalised space
        cands = mu + sigma * torch.randn(n_samples, ACTION_DIM, device=device)

        # Encode candidate actions:  (N, 1, 35) → (N, 1, D)
        cand_emb = model.action_encoder(cands.unsqueeze(1))   # (N, 1, D)

        # Build full action context [a_{t-2}, a_{t-1}, a_candidate]: (N, 3, D)
        past_emb = model.action_encoder(past_act.expand(n_samples, -1, -1))  # (N, 2, D)
        ctx_act  = torch.cat([past_emb, cand_emb], dim=1)                    # (N, 3, D)

        # Predict next embedding
        ctx_exp  = ctx_emb.expand(n_samples, -1, -1)   # (N, 3, D)
        pred     = model.predict(ctx_exp, ctx_act)      # (N, 3, D)
        next_emb = pred[:, -1:]                         # (N, 1, D)

        # Cost = MSE to goal
        costs = F.mse_loss(next_emb, goal_exp.detach(), reduction="none") \
                  .sum(dim=-1).squeeze(-1)              # (N,)

        # CEM update
        elite_idx = costs.argsort()[:n_elites]
        elites    = cands[elite_idx]
        mu        = elites.mean(0)
        sigma     = elites.std(0).clamp(min=0.01)

    action_norm = mu.cpu().numpy()
    action_raw  = (mu * _DELTA_STD.to(device) + _DELTA_MEAN.to(device)).cpu().numpy()
    return action_norm, action_raw


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
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--n_elites",  type=int, default=10)
    parser.add_argument("--n_iters",   type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _DELTA_MEAN = _DELTA_MEAN.to(device)
    _DELTA_STD  = _DELTA_STD.to(device)

    print(f"Loading checkpoint: {args.ckpt}")
    model = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.eval()
    model.requires_grad_(False)
    print(f"Model loaded on {device}. Starting server at {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
