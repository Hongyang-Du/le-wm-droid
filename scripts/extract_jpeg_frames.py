"""Pre-extract all DROID video frames to JPEG for fast random-access loading.

Output layout mirrors the video layout:
  <frames_root>/chunk-{C:03d}/observation.images.{cam}/episode_{E:06d}/{t:06d}.jpg

Run:
  python scripts/extract_jpeg_frames.py \
      --root /datasets/droid_lerobot \
      --frames_root /datasets/droid_lerobot/frames \
      --workers 16
"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torchvision.io as tvio
from PIL import Image


CAMERAS = [
    "exterior_image_1_left",
    "exterior_image_2_left",
    "wrist_image_left",
]


def extract_episode(args):
    root, frames_root, ep_abs, cameras, quality = args
    chunk = ep_abs // 1000
    errors = []

    for cam in cameras:
        vid = root / f"videos/chunk-{chunk:03d}/observation.images.{cam}/episode_{ep_abs:06d}.mp4"
        out_dir = frames_root / f"chunk-{chunk:03d}/observation.images.{cam}/episode_{ep_abs:06d}"

        if not vid.exists():
            continue
        # skip if already fully extracted
        if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) > 0:
            continue

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            vframes, _, _ = tvio.read_video(str(vid), pts_unit="sec", output_format="TCHW")
            for t, frame in enumerate(vframes):
                jpg = out_dir / f"{t:06d}.jpg"
                if not jpg.exists():
                    img = Image.fromarray(frame.permute(1, 2, 0).numpy())
                    img.save(jpg, quality=quality, optimize=False)
        except Exception as e:
            errors.append(f"ep {ep_abs} cam {cam}: {e}")

    return ep_abs, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/datasets/droid_lerobot")
    parser.add_argument("--frames_root", default="/datasets/droid_lerobot/frames")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--cameras", nargs="+", default=CAMERAS)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    frames_root = Path(args.frames_root)

    # load episode list
    episodes = []
    with open(root / "meta/episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line.strip())["episode_index"])
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]

    print(f"Extracting {len(episodes)} episodes × {len(args.cameras)} cameras → {frames_root}")
    print(f"Workers: {args.workers}  JPEG quality: {args.quality}")

    tasks = [(root, frames_root, ep, args.cameras, args.quality) for ep in episodes]

    done, total = 0, len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(extract_episode, t): t for t in tasks}
        for fut in as_completed(futs):
            ep_abs, errors = fut.result()
            done += 1
            if errors:
                print(f"  WARN ep {ep_abs}: {errors}")
            if done % 500 == 0 or done == total:
                print(f"  {done}/{total} episodes done")

    print("Done.")


if __name__ == "__main__":
    main()
