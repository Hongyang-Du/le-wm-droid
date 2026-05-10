"""
Example client for the LeWM DROID policy server.
Run this on the simulator server.

Usage:
    python client_example.py --server http://<policy-server-ip>:8000
"""

from __future__ import annotations

import argparse
import base64
import io

import numpy as np
import requests
from PIL import Image


def encode_image(img: np.ndarray) -> str:
    """(H, W, 3) uint8 → base64-encoded JPEG string."""
    pil = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def reset_episode(server: str, goal_images: dict[str, np.ndarray]) -> None:
    """Send goal images at the start of an episode."""
    payload = {
        "goal_images": {k: encode_image(v) for k, v in goal_images.items()}
    }
    r = requests.post(f"{server}/reset", json=payload, timeout=30)
    r.raise_for_status()


def get_action(
    server: str,
    images: dict[str, np.ndarray],
    proprio: list[float] | None = None,
) -> tuple[list[float], list[list[float]]]:
    """
    Send current observation, get back 7D delta action.

    Returns:
        action       – first 7D delta [Δx,Δy,Δz,Δrx,Δry,Δrz,Δgripper]
        action_block – full 5×7 block for higher-frequency execution
    """
    payload = {
        "images": {k: encode_image(v) for k, v in images.items()},
        "proprio": proprio or [],
    }
    r = requests.post(f"{server}/act", json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["action"], data["action_block"]


# ── demo loop (replace with your sim env) ─────────────────────────────────────

def run_demo(server: str):
    import time

    print(f"Connecting to policy server: {server}")

    # Fake goal images (replace with real goal obs from your sim)
    goal_imgs = {
        "pixels_0": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
        "pixels_1": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
        "pixels_2": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
    }
    reset_episode(server, goal_imgs)
    print("Episode reset OK")

    for step in range(20):
        # Fake current observation (replace with real obs from your sim)
        obs_imgs = {
            "pixels_0": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
            "pixels_1": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
            "pixels_2": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
        }
        proprio = [0.0] * 14

        t0 = time.time()
        action, block = get_action(server, obs_imgs, proprio)
        dt = time.time() - t0

        print(f"  step {step:3d}  action={[f'{a:.4f}' for a in action]}  ({dt*1000:.0f}ms)")

        # Execute action_block in your sim at ~15Hz (5 sub-steps × 3Hz obs)
        # for sub_action in block:
        #     sim.step(sub_action)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8000")
    args = parser.parse_args()
    run_demo(args.server)
