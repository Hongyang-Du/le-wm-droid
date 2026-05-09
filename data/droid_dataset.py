"""DROID dataset adapter for the LeRobot parquet + MP4/JPEG format.

Layout on disk:
  <root>/meta/episodes.jsonl           — episode lengths, indices, tasks
  <root>/meta/stats.json               — pre-computed action/state mean+std
  <root>/data/chunk-{C:03d}/episode_{E:06d}.parquet  — action, state
  <root>/videos/chunk-{C:03d}/observation.images.{cam}/episode_{E:06d}.mp4
  <frames_root>/chunk-{C:03d}/observation.images.{cam}/episode_{E:06d}/{t:06d}.jpg

DroidJpegDataset   — map-style random-access using pre-extracted JPEG frames (fast).
DroidStreamDataset — iterable-style, decodes each MP4 once per epoch (fallback).
"""

from __future__ import annotations

import json
import math
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision.io as tvio
import warnings

warnings.filterwarnings("ignore", message="The pts_unit 'pts' gives wrong results")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_full_video(vid_path: str) -> torch.Tensor:
    """Decode an entire MP4 and return (T, C, H, W) uint8."""
    vframes, _, _ = tvio.read_video(vid_path, pts_unit="sec", output_format="TCHW")
    return vframes


def _load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["action", "observation.state"])


def _make_clip(
    frames: dict[str, torch.Tensor],   # key → (T_ep, C, H, W)
    df: pd.DataFrame,
    start: int,
    span: int,
    frameskip: int,
    num_steps: int,
    keys: list[str],
    transform,
) -> dict:
    obs_indices = list(range(start, start + span, frameskip))
    result: dict = {}

    # support both single-camera ('pixels') and multi-camera ('pixels_N') keys
    for key, vid in frames.items():
        if key in keys or key == "pixels":
            obs_idx_clamped = [min(i, len(vid) - 1) for i in obs_indices]
            result[key] = vid[obs_idx_clamped]

    if "action" in keys:
        actions = np.stack(df["action"].iloc[start : start + span].tolist()).astype(np.float32)
        result["action"] = torch.from_numpy(actions)   # (span, 28) — transform before reshape

    if "proprio" in keys:
        states = np.stack(df["observation.state"].iloc[obs_indices].tolist()).astype(np.float32)
        result["proprio"] = torch.from_numpy(states)

    return transform(result) if transform else result


DROID_CAMERAS = [
    "exterior_image_1_left",
    "exterior_image_2_left",
    "wrist_image_left",
]


def _vid_path(root: Path, camera_key: str, ep_abs: int) -> str:
    chunk_id = ep_abs // 1000
    return str(
        root
        / f"videos/chunk-{chunk_id:03d}"
        / f"observation.images.{camera_key}"
        / f"episode_{ep_abs:06d}.mp4"
    )


def _parquet_path(root: Path, ep_abs: int) -> str:
    chunk_id = ep_abs // 1000
    return str(root / f"data/chunk-{chunk_id:03d}" / f"episode_{ep_abs:06d}.parquet")


def _load_meta(root: str | Path, max_episodes: int | None) -> tuple[list[int], np.ndarray, dict]:
    root = Path(root)
    episodes_path = root / "meta" / "episodes.jsonl"
    episodes: list[dict] = []
    with open(episodes_path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    abs_indices = [e["episode_index"] for e in episodes]
    lengths = np.array([e["length"] for e in episodes], dtype=np.int64)

    with open(root / "meta" / "stats.json") as f:
        stats = json.load(f)

    return abs_indices, lengths, stats


# ---------------------------------------------------------------------------
# Map-style dataset (debugging / small subsets)
# ---------------------------------------------------------------------------

class DroidLerobotDataset:
    """Random-access dataset. Slow on full DROID — use DroidStreamDataset instead."""

    ACTION_DIM = 28
    STATE_DIM = 14
    FPS = 15.0

    def __init__(
        self,
        root: str | Path,
        frameskip: int = 5,
        num_steps: int = 4,
        transform=None,
        keys_to_load: list[str] | None = None,
        camera_key: str = "exterior_image_1_left",
        max_episodes: int | None = None,
        clip_stride: int = 1,
        parquet_cache_size: int = 512,
    ) -> None:
        self.root = Path(root)
        self.camera_key = camera_key
        self.frameskip = frameskip
        self.num_steps = num_steps
        self.transform = transform
        self.span = num_steps * frameskip
        self._parquet_cache: OrderedDict[int, pd.DataFrame] = OrderedDict()
        self._parquet_cache_size = parquet_cache_size
        self._keys = keys_to_load or ["pixels", "action", "proprio"]

        self._episode_abs_indices, self.lengths, self._stats = _load_meta(root, max_episodes)

        self.offsets = np.zeros(len(self.lengths), dtype=np.int64)
        if len(self.lengths) > 1:
            self.offsets[1:] = np.cumsum(self.lengths[:-1])

        self.clip_indices: list[tuple[int, int]] = [
            (ep, start)
            for ep, length in enumerate(self.lengths)
            if length >= self.span
            for start in range(0, length - self.span + 1, clip_stride)
        ]

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.clip_indices[idx]
        return self._load_slice(ep_idx, start)

    def get_dim(self, col: str) -> int:
        return {"action": self.ACTION_DIM, "proprio": self.STATE_DIM}.get(col, 0)

    def get_col_data(self, col: str) -> np.ndarray:
        key_map = {"action": "action", "proprio": "observation.state"}
        parquet_col = key_map.get(col)
        if parquet_col is None:
            raise KeyError(f"get_col_data: unknown column '{col}'")
        rows: list[np.ndarray] = []
        budget = 10_000
        for ep_idx in range(len(self._episode_abs_indices)):
            df = self._get_parquet(ep_idx)
            chunk = np.stack(df[parquet_col].tolist()).astype(np.float32)
            rows.append(chunk)
            budget -= len(chunk)
            if budget <= 0:
                break
        return np.concatenate(rows, axis=0)

    def get_norm_stats(self, col: str) -> tuple[torch.Tensor, torch.Tensor]:
        key_map = {"action": "action", "proprio": "observation.state"}
        stats_key = key_map.get(col, col)
        if stats_key not in self._stats:
            raise KeyError(f"No stats for '{col}' in stats.json")
        mean = torch.tensor(self._stats[stats_key]["mean"], dtype=torch.float32)
        std = torch.tensor(self._stats[stats_key]["std"], dtype=torch.float32)
        return mean, std

    def _get_parquet(self, ep_idx: int) -> pd.DataFrame:
        if ep_idx not in self._parquet_cache:
            if len(self._parquet_cache) >= self._parquet_cache_size:
                self._parquet_cache.popitem(last=False)
            self._parquet_cache[ep_idx] = _load_parquet(
                _parquet_path(self.root, self._episode_abs_indices[ep_idx])
            )
        return self._parquet_cache[ep_idx]

    def _load_slice(self, ep_idx: int, start: int) -> dict:
        ep_abs = self._episode_abs_indices[ep_idx]
        t_start = max(0.0, start / self.FPS - 1.0 / self.FPS)
        t_end = (start + self.span - 1) / self.FPS + 2.0 / self.FPS
        vframes, _, _ = tvio.read_video(
            _vid_path(self.root, self.camera_key, ep_abs),
            start_pts=t_start, end_pts=t_end, pts_unit="sec", output_format="TCHW",
        )
        obs_indices = list(range(0, self.span, self.frameskip))
        obs_idx_clamped = [min(i, len(vframes) - 1) for i in obs_indices]

        result: dict = {}
        if "pixels" in self._keys:
            result["pixels"] = vframes[obs_idx_clamped]

        df = self._get_parquet(ep_idx)
        if "action" in self._keys:
            actions = np.stack(df["action"].iloc[start : start + self.span].tolist()).astype(np.float32)
            result["action"] = torch.from_numpy(actions).reshape(self.num_steps, -1)
        if "proprio" in self._keys:
            states = np.stack(df["observation.state"].iloc[obs_idx_clamped].tolist()).astype(np.float32)
            result["proprio"] = torch.from_numpy(states)

        return self.transform(result) if self.transform else result


# ---------------------------------------------------------------------------
# Iterable dataset — decodes each video ONCE, fast for full training
# ---------------------------------------------------------------------------

class DroidStreamDataset(torch.utils.data.IterableDataset):
    """Streams DROID episodes; each video is decoded exactly once per epoch.

    Workers each handle a disjoint shard of episodes. Episode order is
    shuffled each epoch so the model sees diverse batches.

    clip_stride: sample every N-th clip start within each episode.
                 clip_stride=5 gives ~46 clips/ep (vs 231 at stride=1),
                 reducing I/O by 5× while keeping good coverage.
    """

    ACTION_DIM = 28
    STATE_DIM = 14

    def __init__(
        self,
        root: str | Path,
        frameskip: int = 5,
        num_steps: int = 4,
        transform=None,
        keys_to_load: list[str] | None = None,
        camera_key: str = "exterior_image_1_left",
        camera_keys: list[str] | None = None,
        max_episodes: int | None = None,
        clip_stride: int = 5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        # camera_keys overrides camera_key when provided
        self.camera_keys: list[str] = camera_keys or [camera_key]
        self.camera_key = self.camera_keys[0]
        self.frameskip = frameskip
        self.num_steps = num_steps
        self.transform = transform
        self.span = num_steps * frameskip
        self.clip_stride = clip_stride
        self.seed = seed
        n_cams = len(self.camera_keys)
        default_pixel_keys = ["pixels"] if n_cams == 1 else [f"pixels_{i}" for i in range(n_cams)]
        self._keys = keys_to_load or (default_pixel_keys + ["action", "proprio"])
        self._epoch = 0

        self._episode_abs_indices, self.lengths, self._stats = _load_meta(root, max_episodes)

        # Pre-compute per-episode clip counts for __len__ estimate
        self._clips_per_ep = np.array([
            math.ceil(max(0, length - self.span + 1) / clip_stride)
            for length in self.lengths
        ], dtype=np.int64)

    # -- SWM-compatible helpers (called before training starts) --

    def get_dim(self, col: str) -> int:
        return {"action": self.ACTION_DIM, "proprio": self.STATE_DIM}.get(col, 0)

    def get_norm_stats(self, col: str) -> tuple[torch.Tensor, torch.Tensor]:
        key_map = {"action": "action", "proprio": "observation.state"}
        stats_key = key_map.get(col, col)
        if stats_key not in self._stats:
            raise KeyError(f"No stats for '{col}' in stats.json")
        mean = torch.tensor(self._stats[stats_key]["mean"], dtype=torch.float32)
        std = torch.tensor(self._stats[stats_key]["std"], dtype=torch.float32)
        return mean, std

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return int(self._clips_per_ep.sum())

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        n_episodes = len(self._episode_abs_indices)

        # Shuffle episode order deterministically per epoch
        rng = random.Random(self.seed + self._epoch)
        ep_order = list(range(n_episodes))
        rng.shuffle(ep_order)

        # Split episodes across workers
        if worker_info is not None:
            per_worker = math.ceil(n_episodes / worker_info.num_workers)
            start_ep = worker_info.id * per_worker
            end_ep = min(start_ep + per_worker, n_episodes)
            ep_order = ep_order[start_ep:end_ep]

        for ep_idx in ep_order:
            length = int(self.lengths[ep_idx])
            if length < self.span:
                continue

            ep_abs = self._episode_abs_indices[ep_idx]

            # Decode all camera videos ONCE for this episode
            try:
                frames: dict[str, torch.Tensor] = {}
                for i, cam in enumerate(self.camera_keys):
                    key = "pixels" if len(self.camera_keys) == 1 else f"pixels_{i}"
                    frames[key] = _load_full_video(_vid_path(self.root, cam, ep_abs))
            except Exception:
                continue

            # Load parquet ONCE
            try:
                df = _load_parquet(_parquet_path(self.root, ep_abs))
            except Exception:
                continue

            # Yield all clips with clip_stride
            starts = range(0, length - self.span + 1, self.clip_stride)
            for start in starts:
                try:
                    item = _make_clip(
                        frames, df, start,
                        self.span, self.frameskip, self.num_steps,
                        self._keys, self.transform,
                    )
                    # reshape action AFTER transform (normalizer expects raw 28-dim)
                    if "action" in item:
                        item["action"] = item["action"].reshape(self.num_steps, -1)
                    yield item
                except Exception:
                    continue


# ---------------------------------------------------------------------------
# JPEG random-access dataset (fast, requires pre-extracted frames)
# ---------------------------------------------------------------------------

class DroidJpegDataset(torch.utils.data.Dataset):
    """Map-style dataset using pre-extracted JPEG frames.

    Each __getitem__ loads only the 4 observation frames needed per clip
    (T×3 cameras = 12 JPEGs) — no full video decode, true random access.
    Enable with frames_root pointing to the extracted directory.
    """

    ACTION_DIM = 28
    STATE_DIM  = 14

    def __init__(
        self,
        root: str | Path,
        frames_root: str | Path,
        frameskip: int = 5,
        num_steps: int = 4,
        transform=None,
        keys_to_load: list[str] | None = None,
        camera_keys: list[str] | None = None,
        max_episodes: int | None = None,
        clip_stride: int = 1,
        parquet_cache_size: int = 1024,
    ) -> None:
        self.root        = Path(root)
        self.frames_root = Path(frames_root)
        self.frameskip   = frameskip
        self.num_steps   = num_steps
        self.transform   = transform
        self.span        = num_steps * frameskip
        self.camera_keys = camera_keys or ["exterior_image_1_left"]
        n_cams = len(self.camera_keys)
        default_pixel_keys = ["pixels"] if n_cams == 1 else [f"pixels_{i}" for i in range(n_cams)]
        self._keys = keys_to_load or (default_pixel_keys + ["action", "proprio"])
        self._parquet_cache: OrderedDict[int, pd.DataFrame] = OrderedDict()
        self._parquet_cache_size = parquet_cache_size

        self._episode_abs_indices, self.lengths, self._stats = _load_meta(root, max_episodes)

        self.clip_indices: list[tuple[int, int]] = [
            (ep, start)
            for ep, length in enumerate(self.lengths)
            if length >= self.span
            for start in range(0, length - self.span + 1, clip_stride)
        ]

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.clip_indices[idx]
        ep_abs  = self._episode_abs_indices[ep_idx]
        obs_idx = list(range(start, start + self.span, self.frameskip))
        result: dict = {}

        for i, cam in enumerate(self.camera_keys):
            key = "pixels" if len(self.camera_keys) == 1 else f"pixels_{i}"
            if key not in self._keys:
                continue
            ep_dir = (self.frames_root
                      / f"chunk-{ep_abs//1000:03d}"
                      / f"observation.images.{cam}"
                      / f"episode_{ep_abs:06d}")
            result[key] = torch.stack([
                tvio.read_image(str(ep_dir / f"{fi:06d}.jpg")) for fi in obs_idx
            ])  # (T, C, H, W) uint8

        df = self._get_parquet(ep_idx, ep_abs)
        if "action" in self._keys:
            actions = np.stack(df["action"].iloc[start:start+self.span].tolist()).astype(np.float32)
            result["action"] = torch.from_numpy(actions)
        if "proprio" in self._keys:
            states = np.stack(df["observation.state"].iloc[obs_idx].tolist()).astype(np.float32)
            result["proprio"] = torch.from_numpy(states)

        if self.transform:
            result = self.transform(result)

        if "action" in result:
            result["action"] = result["action"].reshape(self.num_steps, -1)

        return result

    def _get_parquet(self, ep_idx: int, ep_abs: int) -> pd.DataFrame:
        if ep_idx not in self._parquet_cache:
            if len(self._parquet_cache) >= self._parquet_cache_size:
                self._parquet_cache.popitem(last=False)
            self._parquet_cache[ep_idx] = _load_parquet(_parquet_path(self.root, ep_abs))
        return self._parquet_cache[ep_idx]

    def get_dim(self, col: str) -> int:
        return {"action": self.ACTION_DIM, "proprio": self.STATE_DIM}.get(col, 0)

    def get_norm_stats(self, col: str) -> tuple[torch.Tensor, torch.Tensor]:
        key_map = {"action": "action", "proprio": "observation.state"}
        stats_key = key_map.get(col, col)
        mean = torch.tensor(self._stats[stats_key]["mean"], dtype=torch.float32)
        std  = torch.tensor(self._stats[stats_key]["std"],  dtype=torch.float32)
        return mean, std
