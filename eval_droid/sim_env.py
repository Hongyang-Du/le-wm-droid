"""
robosuite environment wrapper for LeWM DROID goal-image MPC evaluation.

Franka robot + OSC_POSE (7D Cartesian delta) + 3 cameras matching DROID layout.
Goal image: step to goal state, capture images, reset to start state.

Difficulty ladder (same as V-JEPA 2 AC):
  reach  → EEF reaches within 5cm of object (no grasp needed)
  lift   → grasp + lift object 15cm above table
  pick_place → lift + move to target zone
"""

from __future__ import annotations

import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config

# Camera names used in this sim (mapped to DROID camera key names)
CAM_SIM   = ["frontview", "sideview", "robot0_eye_in_hand"]  # robosuite names
CAM_DROID = ["pixels_0",  "pixels_1",  "pixels_2"]           # keys sent to policy server

IMG_H, IMG_W = 240, 320   # raw capture resolution (policy server resizes to 224×224)

# Difficulty ladder
TASKS = {
    "reach":      "Lift",        # robosuite env; success = EEF within 5cm of object
    "lift":       "Lift",        # success = object lifted 15cm
    "pick_place": "PickPlace",   # success = object in target bin
}
REACH_THRESHOLD = 0.05   # metres


def make_env(
    task: str = "reach",
    camera_height: int = IMG_H,
    camera_width:  int = IMG_W,
    has_renderer:  bool = False,
    horizon:       int  = 300,
) -> "DroidSimEnv":
    assert task in TASKS, f"task must be one of {list(TASKS)}"
    return DroidSimEnv(
        task=task,
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

    task options: "reach", "lift", "pick_place"

    Action space: 7D = [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
      - Δpos in metres, Δrot in radians
      - gripper: -1 = close, +1 = open
    """

    def __init__(
        self,
        task:          str  = "reach",
        camera_height: int  = IMG_H,
        camera_width:  int  = IMG_W,
        has_renderer:  bool = False,
        horizon:       int  = 300,
    ):
        self.task = task
        controller_cfg = load_controller_config(default_controller="OSC_POSE")
        # Match DROID L1 ≤ 0.075 constraint per sub-step
        controller_cfg["output_max"] = [0.075] * 3 + [0.075] * 3
        controller_cfg["output_min"] = [-0.075] * 3 + [-0.075] * 3

        self._env = suite.make(
            env_name=TASKS[task],
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
        """Task-specific success criterion."""
        if self.task == "reach":
            # Success = EEF within REACH_THRESHOLD of object (no grasp needed)
            obj_pos = self._env.sim.data.body_xpos[
                self._env.sim.model.body_name2id("cube_main")
            ]
            eef_pos = self._env._eef_xpos
            return float(np.linalg.norm(obj_pos - eef_pos)) < REACH_THRESHOLD
        else:
            # lift / pick_place: use robosuite's built-in check
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

    # ── goal state generators (scripted) ──────────────────────────────────────

    def generate_goal_images(self) -> dict[str, np.ndarray]:
        """
        Run scripted policy to goal state for current task, capture images.
        Caller should reset() before running the MPC policy afterwards.
        """
        if self.task == "reach":
            return self._goal_reach()
        elif self.task == "lift":
            return self._goal_lift()
        elif self.task == "pick_place":
            return self._goal_lift()   # lift part is sufficient for goal image
        else:
            raise NotImplementedError(self.task)

    def _goal_reach(self) -> dict[str, np.ndarray]:
        """Goal: EEF touching/at the object. Simple reach, no grasp."""
        obj_pos = self._env.sim.data.body_xpos[
            self._env.sim.model.body_name2id("cube_main")
        ].copy()

        # Move EEF to object position (open gripper, approach from above, descend)
        for _ in range(15):
            delta = obj_pos - self._env._eef_xpos
            delta[2] += 0.08
            a = np.zeros(7); a[:3] = np.clip(delta * 8, -0.075, 0.075); a[6] = 1.0
            self._env.step(a)
        for _ in range(10):
            delta = obj_pos - self._env._eef_xpos
            a = np.zeros(7); a[:3] = np.clip(delta * 8, -0.075, 0.075); a[6] = 1.0
            self._env.step(a)

        return self.capture_goal_images()

    def _goal_lift(self, lift_height: float = 0.15) -> dict[str, np.ndarray]:
        """Goal: object grasped and lifted lift_height above table."""
        obj_pos = self._env.sim.data.body_xpos[
            self._env.sim.model.body_name2id("cube_main")
        ].copy()

        # 1. Open gripper, approach from above
        for _ in range(12):
            delta = obj_pos - self._env._eef_xpos
            delta[2] += 0.10
            a = np.zeros(7); a[:3] = np.clip(delta * 6, -0.075, 0.075); a[6] = 1.0
            self._env.step(a)
        # 2. Descend to object
        for _ in range(10):
            delta = obj_pos - self._env._eef_xpos
            a = np.zeros(7); a[:3] = np.clip(delta * 6, -0.075, 0.075); a[6] = 1.0
            self._env.step(a)
        # 3. Close gripper
        for _ in range(8):
            a = np.zeros(7); a[6] = -1.0
            self._env.step(a)
        # 4. Lift
        for _ in range(15):
            a = np.zeros(7); a[2] = 0.05; a[6] = -1.0
            self._env.step(a)

        return self.capture_goal_images()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()
