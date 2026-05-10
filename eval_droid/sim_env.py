"""
robosuite environment wrapper for LeWM DROID goal-image MPC evaluation.

Franka robot + OSC_POSE (7D Cartesian delta) + 3 cameras matching DROID layout.
Goal image: step to goal state, capture images, reset to start state.
"""

from __future__ import annotations

import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config

# Camera names used in this sim (mapped to DROID camera key names)
CAM_SIM  = ["frontview", "sideview", "robot0_eye_in_hand"]  # robosuite names
CAM_DROID = ["pixels_0",  "pixels_1",  "pixels_2"]          # keys sent to policy server

IMG_H, IMG_W = 240, 320   # raw capture resolution (policy server resizes to 224×224)


def make_env(
    task_name: str = "Lift",       # "Lift", "PickPlace", "Stack"
    camera_height: int = IMG_H,
    camera_width:  int = IMG_W,
    has_renderer:  bool = False,
    horizon:       int  = 300,
) -> "DroidSimEnv":
    return DroidSimEnv(
        task_name=task_name,
        camera_height=camera_height,
        camera_width=camera_width,
        has_renderer=has_renderer,
        horizon=horizon,
    )


class DroidSimEnv:
    """
    Wraps a robosuite environment with:
      - 7D OSC_POSE Cartesian delta action space
      - 3 cameras (front, side, wrist) → same keys as DROID dataset
      - goal_image capture: step env to goal state, record cameras, reset

    Action space: 7D = [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper_open]
      - Δpos in meters (scale ≈ 0.05m max per step at 3Hz)
      - Δrot in radians
      - gripper: -1 = close, +1 = open
    """

    def __init__(
        self,
        task_name:     str  = "Lift",
        camera_height: int  = IMG_H,
        camera_width:  int  = IMG_W,
        has_renderer:  bool = False,
        horizon:       int  = 300,
    ):
        controller_cfg = load_controller_config(default_controller="OSC_POSE")
        # Scale down position/rotation sensitivity to match DROID L1 ≤ 0.075 constraint
        controller_cfg["output_max"] = [0.075] * 3 + [0.075] * 3
        controller_cfg["output_min"] = [-0.075] * 3 + [-0.075] * 3

        self._env = suite.make(
            env_name=task_name,
            robots="Franka",
            controller_configs=controller_cfg,
            has_renderer=has_renderer,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=CAM_SIM,
            camera_heights=camera_height,
            camera_widths=camera_width,
            horizon=horizon,
            reward_shaping=False,
            ignore_done=False,
        )

        self.task_name = task_name
        self._goal_images: dict[str, np.ndarray] | None = None

    # ── core interface ─────────────────────────────────────────────────────────

    def reset(self) -> dict[str, np.ndarray]:
        """Reset and return first observation."""
        raw = self._env.reset()
        return self._extract_obs(raw)

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """
        action: (7,) = [Δx,Δy,Δz,Δrx,Δry,Δrz, gripper]
        gripper: -1=close, +1=open  (robosuite convention)
        """
        raw, reward, done, info = self._env.step(action.astype(np.float64))
        obs = self._extract_obs(raw)
        return obs, float(reward), bool(done), info

    def success(self) -> bool:
        return bool(self._env._check_success())

    # ── goal image ─────────────────────────────────────────────────────────────

    def capture_goal_images(self) -> dict[str, np.ndarray]:
        """
        Capture images of the CURRENT state as goal images.
        Call this after manually moving the environment to its goal state
        (e.g. after running a scripted policy or resetting to a target layout).
        Returns dict with keys pixels_0/1/2 → (H, W, 3) uint8 arrays.
        """
        raw = self._env._get_observations()
        return self._extract_obs(raw)

    def set_goal_from_current(self) -> dict[str, np.ndarray]:
        """Convenience: alias for capture_goal_images."""
        imgs = self.capture_goal_images()
        self._goal_images = imgs
        return imgs

    # ── observations ───────────────────────────────────────────────────────────

    def _extract_obs(self, raw: dict) -> dict[str, np.ndarray]:
        """Extract camera images and proprio from robosuite obs dict."""
        obs = {}
        for sim_key, droid_key in zip(CAM_SIM, CAM_DROID):
            img = raw[f"{sim_key}_image"]             # (H, W, 3) uint8, RGB
            obs[droid_key] = img
        obs["proprio"] = self._get_proprio(raw)       # (14,) float
        return obs

    def _get_proprio(self, raw: dict) -> np.ndarray:
        """14D proprioception: joint_pos(7) + joint_vel(7)."""
        return np.concatenate([
            raw.get("robot0_joint_pos", np.zeros(7)),
            raw.get("robot0_joint_vel", np.zeros(7)),
        ]).astype(np.float32)

    # ── task-specific goal state generators ───────────────────────────────────

    def generate_goal_state_lift(self, lift_height: float = 0.15) -> dict[str, np.ndarray]:
        """
        For Lift task: run a scripted lift to height, capture goal images.
        Object must already be grasped or we use the object's target pos directly.
        """
        # Move end-effector above object position
        obj_pos = self._env.sim.data.body_xpos[
            self._env.sim.model.body_name2id("cube_main")
        ].copy()

        # Open gripper, move above object, close, lift
        actions = []
        # 1. open gripper, move above
        for _ in range(10):
            delta = obj_pos - self._env._eef_xpos
            delta[2] += 0.10   # approach from above
            a = np.zeros(7); a[:3] = np.clip(delta * 5, -0.075, 0.075); a[6] = 1.0
            actions.append(a)
        # 2. descend
        for _ in range(8):
            delta = obj_pos - self._env._eef_xpos
            a = np.zeros(7); a[:3] = np.clip(delta * 5, -0.075, 0.075); a[6] = 1.0
            actions.append(a)
        # 3. close gripper
        for _ in range(5):
            a = np.zeros(7); a[6] = -1.0
            actions.append(a)
        # 4. lift
        for _ in range(12):
            a = np.zeros(7); a[2] = 0.05; a[6] = -1.0
            actions.append(a)

        for a in actions:
            raw, _, done, _ = self._env.step(a)
            if done:
                break

        return self.capture_goal_images()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()
