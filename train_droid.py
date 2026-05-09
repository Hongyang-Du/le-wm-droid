"""LeWM end-to-end training on the DROID dataset (LeRobot parquet+video format)."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from data import DroidStreamDataset, DroidJpegDataset
from jepa import JEPA, MultiViewJEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import ModelObjectCallBack


def get_droid_normalizer(dataset, source: str, target: str):
    """Build a z-score normalizer using pre-computed dataset stats."""
    mean, std = dataset.get_norm_stats(source)
    mean = mean.unsqueeze(0)
    std = std.unsqueeze(0).clamp(min=1e-6)

    def norm_fn(x: torch.Tensor) -> torch.Tensor:
        return ((x - mean) / std).float()

    return spt.data.transforms.WrapTorchTransform(norm_fn, source=source, target=target)


def _all_gather_emb(emb: torch.Tensor) -> torch.Tensor:
    """Gather emb across all DDP ranks; keep local slice grad-connected."""
    try:
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return emb
        world = dist.get_world_size()
        rank  = dist.get_rank()
        gathered = [torch.zeros_like(emb) for _ in range(world)]
        dist.all_gather(gathered, emb.contiguous())
        gathered[rank] = emb          # keep local grad path alive
        return torch.cat(gathered, dim=0)   # (B*world, T, D)
    except Exception:
        return emb


def lejepa_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses."""
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]       # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    # SIGReg on the global batch (gathered from all GPUs) so stats are correct
    emb_global = _all_gather_emb(emb)           # (B*world, T, D)
    output["sigreg_loss"] = self.sigreg(emb_global.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


class EpochSyncCallback(pl.Callback):
    """Tells DroidStreamDataset which epoch we're in so episode order re-shuffles."""
    def __init__(self, train_loader):
        self._loader = train_loader

    def on_train_epoch_start(self, trainer, pl_module):
        ds = self._loader.dataset
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(trainer.current_epoch)


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm_droid")
def run(cfg):
    # ------------------------------------------------------------------
    # Dataset  (stream = one video decode per episode per epoch)
    # ------------------------------------------------------------------
    import stable_pretraining.data as dt

    import json
    frames_root = cfg.data.dataset.get("frames_root", None)
    use_jpeg = frames_root is not None

    # Proto for norm stats only (5 episodes, no frames needed)
    proto = DroidJpegDataset(
        root=cfg.data.dataset.root,
        frames_root=frames_root if use_jpeg else cfg.data.dataset.root,
        frameskip=cfg.data.dataset.frameskip,
        num_steps=cfg.data.dataset.num_steps,
        camera_keys=list(cfg.data.dataset.camera_keys),
        max_episodes=5,
    ) if use_jpeg else DroidStreamDataset(
        root=cfg.data.dataset.root,
        frameskip=cfg.data.dataset.frameskip,
        num_steps=cfg.data.dataset.num_steps,
        camera_keys=list(cfg.data.dataset.camera_keys),
        max_episodes=5,
    )

    imagenet_stats = dt.dataset_stats.ImageNet
    transforms = []
    pixel_keys = [k for k in cfg.data.dataset.keys_to_load if k.startswith("pixels")]
    for pk in pixel_keys:
        transforms.append(spt.data.transforms.Compose(
            dt.transforms.ToImage(**imagenet_stats, source=pk, target=pk),
            dt.transforms.Resize((cfg.img_size, cfg.img_size), source=pk, target=pk),
        ))

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_droid_normalizer(proto, col, col))
            setattr(cfg.wm, f"{col}_dim", proto.get_dim(col))

    full_transform = spt.data.transforms.Compose(*transforms)

    # Episode count for train/val split
    max_ep = cfg.data.dataset.get("max_episodes", None)
    if max_ep is None:
        with open(cfg.data.dataset.root + "/meta/episodes.jsonl") as f:
            n_ep = sum(1 for _ in f)
    else:
        n_ep = int(max_ep)

    ds_kwargs = dict(
        root=cfg.data.dataset.root,
        frameskip=cfg.data.dataset.frameskip,
        num_steps=cfg.data.dataset.num_steps,
        keys_to_load=list(cfg.data.dataset.keys_to_load),
        camera_keys=list(cfg.data.dataset.camera_keys),
        clip_stride=cfg.data.dataset.clip_stride,
        max_episodes=n_ep,
        transform=full_transform,
    )

    if use_jpeg:
        # Map-style: true random shuffle, standard random_split
        full_ds = DroidJpegDataset(**ds_kwargs, frames_root=frames_root)
        rnd_gen = torch.Generator().manual_seed(cfg.seed)
        train_ds, val_ds = spt.data.random_split(
            full_ds, [cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
        )
        loader_cfg = dict(cfg.loader)
        train = torch.utils.data.DataLoader(
            train_ds, **loader_cfg, shuffle=True, drop_last=True, generator=rnd_gen
        )
        val = torch.utils.data.DataLoader(val_ds, **loader_cfg, shuffle=False)
    else:
        # Iterable fallback (MP4 stream)
        n_val_ep  = max(1, int(n_ep * (1 - cfg.train_split)))
        n_train_ep = n_ep - n_val_ep
        train_ds = DroidStreamDataset(**{**ds_kwargs, "max_episodes": n_train_ep}, seed=cfg.seed)
        val_ds   = DroidStreamDataset(**ds_kwargs, seed=cfg.seed + 1)
        val_ds._episode_abs_indices = val_ds._episode_abs_indices[n_train_ep:]
        val_ds.lengths = val_ds.lengths[n_train_ep:]
        loader_cfg = dict(cfg.loader)
        train = torch.utils.data.DataLoader(train_ds, **loader_cfg, shuffle=False)
        val   = torch.utils.data.DataLoader(val_ds,   **loader_cfg, shuffle=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    camera_keys = cfg.data.dataset.get("camera_keys", None)
    n_views = len(camera_keys) if camera_keys else 1
    pixel_keys = ["pixels"] if n_views == 1 else [f"pixels_{i}" for i in range(n_views)]

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    # projector input = n_views × hidden_dim (concat of CLS tokens)
    projector = MLP(
        input_dim=n_views * hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    if n_views > 1:
        world_model = MultiViewJEPA(
            camera_keys=pixel_keys,
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=predictor_proj,
        )
    else:
        world_model = JEPA(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=predictor_proj,
        )

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            ModelObjectCallBack(dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1),
            EpochSyncCallback(train),
        ],
        num_sanity_val_steps=0,   # IterableDataset val sanity causes issues
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()


if __name__ == "__main__":
    run()
