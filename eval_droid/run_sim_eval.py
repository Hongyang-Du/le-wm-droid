"""
Goal-image MPC sim evaluation for LeWM DROID.
Mirrors V-JEPA 2 AC evaluation protocol in robosuite.

Usage (policy server must be running on this machine):
    # Terminal 1 – policy server:
    cd /workspace/le-wm-droid
    python -m eval_droid.policy_server \
        --ckpt /root/.stable_worldmodel/lewm_droid_epoch_10_object.ckpt

    # Terminal 2 – sim eval:
    python -m eval_droid.run_sim_eval \
        --task Lift --episodes 50 --server http://localhost:8000

Goal image protocol (same as V-JEPA 2 AC):
    1. At episode start, run scripted policy to goal state → capture goal images
    2. Reset environment to start state
    3. Run MPC policy (POST /reset with goal images, then POST /act per step)
    4. Episode succeeds if task reward is positive within max_steps
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eval_droid.sim_env import DroidSimEnv, make_env
from eval_droid.client_example import encode_image, reset_episode, get_action


# ── scripted goal generators per task ─────────────────────────────────────────

def get_goal_images(env: DroidSimEnv, task: str) -> dict[str, np.ndarray]:
    """
    Generate goal images by running a scripted policy to the goal state.
    Then the caller resets the env before running the MPC policy.
    """
    if task == "Lift":
        return env.generate_goal_state_lift(lift_height=0.15)
    else:
        # For other tasks: just capture the current (reset) state as a
        # placeholder — replace with task-specific scripted goal generators.
        raise NotImplementedError(f"Goal generator not implemented for task={task}. "
                                   "Add one in sim_env.py or provide goal images manually.")


# ── eval loop ─────────────────────────────────────────────────────────────────

def run_episode(
    env:        DroidSimEnv,
    server:     str,
    task:       str,
    max_steps:  int,
    frameskip:  int,
    verbose:    bool,
) -> dict:
    """Run one episode. Returns dict with success, steps, total_time."""

    # ── 1. Move to goal state and capture goal images ──────────────────────────
    env.reset()
    t0 = time.time()
    goal_imgs = get_goal_images(env, task)

    # ── 2. Reset environment to initial state ──────────────────────────────────
    obs = env.reset()

    # ── 3. Send goal images to policy server ──────────────────────────────────
    reset_episode(server, goal_imgs)

    # ── 4. MPC eval loop ──────────────────────────────────────────────────────
    success = False
    for step in range(max_steps):
        # Policy server expects dict with pixels_0/1/2 keys
        images = {k: obs[k] for k in ["pixels_0", "pixels_1", "pixels_2"]}
        proprio = obs["proprio"].tolist()

        t_act = time.time()
        action, action_block = get_action(server, images, proprio)
        latency_ms = (time.time() - t_act) * 1000

        # Execute action block at frameskip=5 sub-steps
        # action_block: [[7D], [7D], [7D], [7D], [7D]]
        done = False
        for sub_action in action_block:
            obs, reward, done, info = env.step(np.array(sub_action))
            if env.success():
                success = True
                done = True
            if done:
                break

        if verbose:
            print(f"  step {step:3d}  action={[f'{a:.3f}' for a in action]}"
                  f"  latency={latency_ms:.0f}ms  success={success}")

        if done or success:
            break

    total_time = time.time() - t0
    return {"success": success, "steps": step + 1, "total_time": total_time}


def run_eval(args):
    env = make_env(
        task_name=args.task,
        has_renderer=not args.headless,
        horizon=args.max_steps * 5 + 50,   # enough for frameskip sub-steps
    )

    results = []
    for ep in range(args.episodes):
        print(f"\n── Episode {ep + 1}/{args.episodes} ──")
        result = run_episode(
            env=env,
            server=args.server,
            task=args.task,
            max_steps=args.max_steps,
            frameskip=5,
            verbose=args.verbose,
        )
        results.append(result)
        print(f"  → success={result['success']}  steps={result['steps']}"
              f"  time={result['total_time']:.1f}s")

    env.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    n_success = sum(r["success"] for r in results)
    success_rate = n_success / len(results) * 100
    avg_steps = np.mean([r["steps"] for r in results])

    print(f"\n{'='*50}")
    print(f"Task:         {args.task}")
    print(f"Episodes:     {args.episodes}")
    print(f"Success rate: {n_success}/{args.episodes} = {success_rate:.1f}%")
    print(f"Avg steps:    {avg_steps:.1f}")
    print(f"{'='*50}")

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_{args.task}_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump({
            "task": args.task,
            "episodes": args.episodes,
            "success_rate": success_rate,
            "avg_steps": float(avg_steps),
            "per_episode": results,
        }, f, indent=2)
    print(f"Results saved to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       default="Lift",
                        choices=["Lift", "PickPlace", "Stack"],
                        help="robosuite task name")
    parser.add_argument("--server",     default="http://localhost:8000",
                        help="Policy server URL")
    parser.add_argument("--episodes",   type=int, default=50)
    parser.add_argument("--max_steps",  type=int, default=100,
                        help="Max obs-steps per episode (each = 5 sub-steps)")
    parser.add_argument("--headless",   action="store_true", default=True)
    parser.add_argument("--verbose",    action="store_true", default=False)
    parser.add_argument("--output_dir", default="eval_results")
    args = parser.parse_args()

    run_eval(args)
