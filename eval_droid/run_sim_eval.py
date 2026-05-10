"""
Goal-image MPC sim evaluation for LeWM DROID.
Mirrors V-JEPA 2 AC evaluation protocol in robosuite.

Difficulty ladder:
  reach      → EEF reaches within 5cm of object  (easiest, start here)
  lift       → grasp + lift 15cm
  pick_place → lift + move to target bin          (hardest)

Usage:
    # Terminal 1 – policy server:
    cd /workspace/le-wm-droid
    python -m eval_droid.policy_server \\
        --ckpt /root/.stable_worldmodel/lewm_droid_epoch_10_object.ckpt

    # Terminal 2 – sim eval (start with reach):
    python -m eval_droid.run_sim_eval --task reach --episodes 50
    python -m eval_droid.run_sim_eval --task lift  --episodes 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from eval_droid.sim_env import DroidSimEnv, make_env, TASKS
from eval_droid.client_example import reset_episode, get_action

VIDEO_FPS = 15   # 3Hz obs × 5 sub-steps


def _make_video_writer(path: Path, h: int, w: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, VIDEO_FPS, (w, h))


# ── eval loop ─────────────────────────────────────────────────────────────────

def run_episode(
    env:       DroidSimEnv,
    server:    str,
    max_steps: int,
    video_dir: Path | None,
    ep_idx:    int,
    verbose:   bool,
) -> dict:
    """Run one episode. Returns dict with success, steps, total_time."""

    # ── 1. Move to goal state, capture goal images ─────────────────────────────
    env.reset()
    t0 = time.time()
    goal_imgs = env.generate_goal_images()

    # ── 2. Reset to start state ────────────────────────────────────────────────
    obs = env.reset()

    # ── 3. Send goal to policy server ─────────────────────────────────────────
    reset_episode(server, goal_imgs)

    # ── 4. Set up video writer (front-view camera, pixels_0) ──────────────────
    writer = None
    if video_dir is not None:
        h, w = obs["pixels_0"].shape[:2]
        video_path = video_dir / f"ep{ep_idx:03d}.mp4"
        writer = _make_video_writer(video_path, h, w)
        # Write goal image as first 15 frames so viewer knows the target
        goal_bgr = cv2.cvtColor(goal_imgs["pixels_0"], cv2.COLOR_RGB2BGR)
        for _ in range(15):
            writer.write(goal_bgr)

    # ── 5. MPC eval loop ───────────────────────────────────────────────────────
    success = False
    for step in range(max_steps):
        images = {k: obs[k] for k in ["pixels_0", "pixels_1", "pixels_2"]}
        proprio = obs["proprio"].tolist()

        t_act = time.time()
        action, action_block = get_action(server, images, proprio)
        latency_ms = (time.time() - t_act) * 1000

        done = False
        for sub_action in action_block:
            obs, reward, done, info = env.step(np.array(sub_action))
            if writer is not None:
                frame_bgr = cv2.cvtColor(obs["pixels_0"], cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            if env.success():
                success = True
                done = True
            if done:
                break

        if verbose:
            print(f"  step {step:3d}  latency={latency_ms:.0f}ms  success={success}")

        if done or success:
            break

    if writer is not None:
        writer.release()

    total_time = time.time() - t0
    return {"success": success, "steps": step + 1, "total_time": total_time}


def run_eval(args):
    out_dir = Path(args.output_dir) / f"{args.task}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_dir = out_dir / "videos" if args.save_video else None
    if video_dir is not None:
        video_dir.mkdir()

    env = make_env(
        task=args.task,
        has_renderer=not args.headless,
        horizon=args.max_steps * 5 + 50,
    )

    results = []
    for ep in range(args.episodes):
        print(f"\n── Episode {ep + 1}/{args.episodes} ──")
        result = run_episode(
            env=env,
            server=args.server,
            max_steps=args.max_steps,
            video_dir=video_dir,
            ep_idx=ep,
            verbose=args.verbose,
        )
        results.append(result)
        status = "✓ SUCCESS" if result["success"] else "✗ fail"
        print(f"  → {status}  steps={result['steps']}  time={result['total_time']:.1f}s")

    env.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    n_success  = sum(r["success"] for r in results)
    success_rate = n_success / len(results) * 100
    avg_steps  = np.mean([r["steps"] for r in results])

    print(f"\n{'='*50}")
    print(f"Task:         {args.task}")
    print(f"Episodes:     {args.episodes}")
    print(f"Success rate: {n_success}/{args.episodes} = {success_rate:.1f}%")
    print(f"Avg steps:    {avg_steps:.1f}")
    if video_dir:
        print(f"Videos:       {video_dir}/")
    print(f"{'='*50}")

    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "task": args.task,
            "episodes": args.episodes,
            "success_rate": success_rate,
            "avg_steps": float(avg_steps),
            "per_episode": results,
        }, f, indent=2)
    print(f"Results saved to {out_dir}/results.json")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       default="reach",
                        choices=["reach", "lift", "pick_place"],
                        help="difficulty: reach < lift < pick_place")
    parser.add_argument("--server",     default="http://localhost:8000")
    parser.add_argument("--episodes",   type=int,  default=50)
    parser.add_argument("--max_steps",  type=int,  default=100,
                        help="Max obs-steps per episode (each = 5 sub-steps)")
    parser.add_argument("--save_video", action="store_true", default=True,
                        help="Save MP4 per episode (front-view camera)")
    parser.add_argument("--headless",   action="store_true", default=True)
    parser.add_argument("--verbose",    action="store_true", default=False)
    parser.add_argument("--output_dir", default="eval_results")
    args = parser.parse_args()

    run_eval(args)
