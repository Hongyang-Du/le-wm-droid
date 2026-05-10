"""
PyBullet sim environment for LeWM DROID goal-image MPC evaluation.

Uses PyBullet TinyRenderer → zero OpenGL / display requirement.
Franka Panda robot with Cartesian delta control (IK-based), 3 cameras.

Difficulty ladder:
  reach      → EEF within 5cm of object (no grasp)
  lift       → grasp + lift 15cm above table
  pick_place → lift + carry to target zone
"""

from __future__ import annotations

import numpy as np
import pybullet as p
import pybullet_data

CAM_DROID = ["pixels_0", "pixels_1", "pixels_2"]
IMG_H, IMG_W = 240, 320

REACH_THRESHOLD = 0.05   # metres
LIFT_HEIGHT     = 0.15   # metres above table
GRIPPER_OPEN    = 0.04   # metres (each finger)
GRIPPER_CLOSED  = 0.0

TASKS = {"reach": "Lift", "lift": "Lift", "pick_place": "Lift"}

# Panda link indices
EEF_LINK      = 11   # panda_hand
FINGER_JOINT1 = 9
FINGER_JOINT2 = 10
ARM_JOINTS    = list(range(7))


def make_env(
    task:          str  = "reach",
    camera_height: int  = IMG_H,
    camera_width:  int  = IMG_W,
    has_renderer:  bool = False,   # ignored (TinyRenderer always used)
    horizon:       int  = 300,
) -> "DroidSimEnv":
    assert task in TASKS, f"task must be one of {list(TASKS)}"
    return DroidSimEnv(task=task, camera_height=camera_height,
                       camera_width=camera_width, horizon=horizon)


class DroidSimEnv:
    """
    Franka Panda in PyBullet (TinyRenderer, headless).
    Action: 7D = [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
      - Δpos in metres, Δrot in radians
      - gripper: > 0 → open, < 0 → close
    """

    def __init__(self, task="reach", camera_height=IMG_H,
                 camera_width=IMG_W, horizon=300):
        self.task   = task
        self.cam_h  = camera_height
        self.cam_w  = camera_width
        self.horizon = horizon
        self._step_count = 0
        self._gripper_state = GRIPPER_OPEN

        self._client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)

        self._robot = None
        self._cube  = None
        self._target_pos = None

    # ── core interface ─────────────────────────────────────────────────────────

    def reset(self) -> dict[str, np.ndarray]:
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)

        # Table and plane
        p.loadURDF("plane.urdf", physicsClientId=self._client)

        # Franka Panda
        self._robot = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self._client,
        )
        self._reset_arm_home()

        # Cube (object to manipulate)
        cube_x = np.random.uniform(0.3, 0.5)
        cube_y = np.random.uniform(-0.15, 0.15)
        self._cube = p.loadURDF(
            "cube_small.urdf",
            basePosition=[cube_x, cube_y, 0.025],
            physicsClientId=self._client,
        )
        # Target zone for pick_place
        self._target_pos = np.array([0.3, -0.3, 0.025])

        self._step_count = 0
        self._gripper_state = GRIPPER_OPEN
        self._settle(50)
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """action: (7,) Cartesian delta + gripper command."""
        action = np.clip(action, -0.075, 0.075)
        self._apply_cartesian_delta(action[:6])
        self._apply_gripper(action[6])
        for _ in range(5):   # 5 sim steps per control step
            p.stepSimulation(physicsClientId=self._client)
        self._step_count += 1

        obs  = self._get_obs()
        done = self._step_count >= self.horizon
        return obs, 0.0, done, {}

    def success(self) -> bool:
        if self.task == "reach":
            eef = np.array(p.getLinkState(self._robot, EEF_LINK,
                           physicsClientId=self._client)[0])
            obj = np.array(p.getBasePositionAndOrientation(
                self._cube, physicsClientId=self._client)[0])
            return float(np.linalg.norm(eef - obj)) < REACH_THRESHOLD
        elif self.task == "lift":
            obj_z = p.getBasePositionAndOrientation(
                self._cube, physicsClientId=self._client)[0][2]
            return float(obj_z) > LIFT_HEIGHT
        elif self.task == "pick_place":
            obj = np.array(p.getBasePositionAndOrientation(
                self._cube, physicsClientId=self._client)[0])
            return float(np.linalg.norm(obj[:2] - self._target_pos[:2])) < 0.05
        return False

    # ── goal image ─────────────────────────────────────────────────────────────

    def generate_goal_images(self) -> dict[str, np.ndarray]:
        """Run scripted policy to goal state, capture images."""
        if self.task == "reach":
            return self._goal_reach()
        elif self.task in ("lift", "pick_place"):
            return self._goal_lift()
        raise NotImplementedError(self.task)

    def capture_goal_images(self) -> dict[str, np.ndarray]:
        return self._get_obs()

    # ── camera ─────────────────────────────────────────────────────────────────

    def _get_obs(self) -> dict[str, np.ndarray]:
        imgs = self._render_cameras()
        eef_state = p.getLinkState(self._robot, EEF_LINK,
                                   physicsClientId=self._client)
        joint_states = p.getJointStates(self._robot, ARM_JOINTS,
                                        physicsClientId=self._client)
        joint_pos = np.array([s[0] for s in joint_states], dtype=np.float32)
        joint_vel = np.array([s[1] for s in joint_states], dtype=np.float32)
        proprio    = np.concatenate([joint_pos, joint_vel,
                                     np.zeros(14 - 14)]).astype(np.float32)
        return {**imgs, "proprio": proprio}

    def _render_cameras(self) -> dict[str, np.ndarray]:
        eef_pos = np.array(p.getLinkState(self._robot, EEF_LINK,
                           physicsClientId=self._client)[0])

        configs = [
            # Front external camera
            dict(eye=[1.2, 0.0, 0.8], target=[0.4, 0.0, 0.3], up=[0, 0, 1]),
            # Side external camera
            dict(eye=[0.4, -1.0, 0.7], target=[0.4, 0.0, 0.3], up=[0, 0, 1]),
            # Wrist camera (attached to EEF, looking forward-down)
            dict(eye=(eef_pos + np.array([0, 0, 0.05])).tolist(),
                 target=(eef_pos + np.array([0.3, 0, -0.1])).tolist(),
                 up=[0, 0, 1]),
        ]
        imgs = {}
        proj = p.computeProjectionMatrixFOV(
            60, self.cam_w / self.cam_h, 0.01, 10)
        for key, cfg in zip(CAM_DROID, configs):
            view = p.computeViewMatrix(cfg["eye"], cfg["target"], cfg["up"])
            _, _, rgb, _, _ = p.getCameraImage(
                self.cam_w, self.cam_h, view, proj,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=self._client,
            )
            imgs[key] = np.array(rgb, dtype=np.uint8
                                 ).reshape(self.cam_h, self.cam_w, 4)[:, :, :3]
        return imgs

    # ── robot control ──────────────────────────────────────────────────────────

    def _reset_arm_home(self):
        home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i, q in zip(ARM_JOINTS, home):
            p.resetJointState(self._robot, i, q,
                              physicsClientId=self._client)
        self._set_gripper(GRIPPER_OPEN)

    def _apply_cartesian_delta(self, delta6: np.ndarray):
        """Convert Cartesian delta to joint targets via IK."""
        state = p.getLinkState(self._robot, EEF_LINK,
                               computeForwardKinematics=True,
                               physicsClientId=self._client)
        curr_pos = np.array(state[0])
        curr_orn = np.array(state[1])

        new_pos = curr_pos + delta6[:3]
        # Apply rotation delta as axis-angle → new quaternion (small angle approx)
        dorn = delta6[3:]
        orn_delta = p.getQuaternionFromEuler(dorn.tolist())
        new_orn = p.multiplyTransforms([0,0,0], curr_orn,
                                       [0,0,0], orn_delta)[1]

        joints = p.calculateInverseKinematics(
            self._robot, EEF_LINK, new_pos, new_orn,
            physicsClientId=self._client,
        )
        for i, q in zip(ARM_JOINTS, joints[:7]):
            p.setJointMotorControl2(
                self._robot, i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=q, force=500,
                physicsClientId=self._client,
            )

    def _apply_gripper(self, cmd: float):
        if cmd > 0:
            self._gripper_state = min(self._gripper_state + 0.005, GRIPPER_OPEN)
        elif cmd < 0:
            self._gripper_state = max(self._gripper_state - 0.005, GRIPPER_CLOSED)
        self._set_gripper(self._gripper_state)

    def _set_gripper(self, width: float):
        for joint in [FINGER_JOINT1, FINGER_JOINT2]:
            p.setJointMotorControl2(
                self._robot, joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=width, force=100,
                physicsClientId=self._client,
            )

    def _settle(self, steps: int):
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self._client)

    # ── scripted goal policies ─────────────────────────────────────────────────

    def _goal_reach(self) -> dict[str, np.ndarray]:
        """Reach: move EEF to object position."""
        obj_pos = np.array(p.getBasePositionAndOrientation(
            self._cube, physicsClientId=self._client)[0])
        for _ in range(40):
            eef_pos = np.array(p.getLinkState(self._robot, EEF_LINK,
                               physicsClientId=self._client)[0])
            delta = obj_pos - eef_pos
            delta[2] += 0.05
            action = np.zeros(7)
            action[:3] = np.clip(delta * 5, -0.075, 0.075)
            action[6] = 1.0
            self._apply_cartesian_delta(action[:6])
            self._apply_gripper(action[6])
            self._settle(5)
        return self.capture_goal_images()

    def _goal_lift(self) -> dict[str, np.ndarray]:
        """Lift: reach → grasp → lift."""
        obj_pos = np.array(p.getBasePositionAndOrientation(
            self._cube, physicsClientId=self._client)[0])
        # 1. Open gripper, move above
        for _ in range(20):
            eef = np.array(p.getLinkState(self._robot, EEF_LINK,
                           physicsClientId=self._client)[0])
            d = obj_pos - eef; d[2] += 0.12
            self._apply_cartesian_delta(np.clip(d*5,-0.075,0.075))
            self._apply_gripper(1.0); self._settle(5)
        # 2. Descend
        for _ in range(15):
            eef = np.array(p.getLinkState(self._robot, EEF_LINK,
                           physicsClientId=self._client)[0])
            d = obj_pos - eef
            self._apply_cartesian_delta(np.clip(d*5,-0.075,0.075))
            self._apply_gripper(1.0); self._settle(5)
        # 3. Close gripper
        for _ in range(15):
            self._apply_gripper(-1.0); self._settle(5)
        # 4. Lift
        for _ in range(20):
            action = np.zeros(6); action[2] = 0.05
            self._apply_cartesian_delta(action)
            self._apply_gripper(-1.0); self._settle(5)
        return self.capture_goal_images()

    def render(self): pass

    def close(self):
        p.disconnect(physicsClientId=self._client)
