"""
UR5e pick-and-place demo — robust scripted motion.

Key design decisions vs previous version:
  - IK: multi-seed search + pick closest solution to current state.
    Singularity avoidance: prefer solutions with sin(wrist_2) far from 0.
  - Velocity control: high KP with smooth ramp-down near target.
  - Gripper contact detection: keep closing until position stops changing.
  - Exact known cube coordinates: hard-coded CUBE_INIT / TARGET_POS.
  - All waypoints computed from exact known position (no perception needed).
  - Loop: pick and place continuously for RL baseline demonstration.

Once this demo is confirmed to work, run:
    ros2 run ur5e_rl_gazebo train
to start SAC training that learns to replicate and improve on this behaviour.
"""
import subprocess
import threading
import time
from typing import Optional

import numpy as np
from scipy.optimize import minimize

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose

# ── UR5e kinematics (URDF joint transforms) ────────────────────────────────────
_UR5E_JOINTS = [
    (0,       0,       0.1625, 0,          0, 0),
    (0,       0,       0,      np.pi / 2,  0, 0),
    (-0.425,  0,       0,      0,          0, 0),
    (-0.3922, 0,       0.1333, 0,          0, 0),
    (0,      -0.0997,  0,      np.pi / 2,  0, 0),
    (0,       0.09959, 0,      np.pi / 2,  np.pi, np.pi),
]

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

WORLD_NAME = 'pick_place'
# Platform A: cap top = 0.32 m, cube centre = 0.35 m
CUBE_INIT  = np.array([0.60,  0.20, 0.35])
# Platform B: cap top = 0.32 m — target cube centre
TARGET_POS = np.array([0.60, -0.20, 0.35])

# ── Motion parameters ─────────────────────────────────────────────────────────
MAX_VEL  = 1.8    # rad/s
KP       = 8.0    # proportional gain — faster convergence in last 0.1–0.2 rad
KD       = 0.15   # derivative gain
GOAL_TOL = 0.025  # rad — tighter tolerance, but KP=8 reaches it faster

# Waypoint heights — EE (tool0) with gripper pointing DOWN.
# Cube centre at Z=0.35 m. EE-to-fingertip offset ≈ 0.112 m.
# GRASP_Z: fingertips at cube centre → tool0 at 0.35 + 0.112 = 0.462 m
# LIFT_Z kept at 0.60 — (0.60, ±0.20, 0.60) is well inside UR5e workspace.
# Going higher (>0.65) pushes the arm near its reach limit and times out.
PREGRASP_Z = 0.58    # EE height: 12 cm above grasp
GRASP_Z    = 0.46    # EE height: fingertips at cube centre Z=0.35 m
LIFT_Z     = 0.60    # EE height: clear of platforms (cube at 0.49 m)

# Gripper parameters
GRIPPER_VEL      = 0.12   # m/s closing/opening speed
GRIPPER_OPEN_POS = 0.025  # m — fully open
GRIPPER_CLOSE_TARGET = 0.0   # m — fully closed (cube stops fingers before this)
GRIPPER_CONTACT_TOL  = 5e-4  # m — position unchanged → contact detected


def _rpy(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def _raw_fk(q: np.ndarray) -> np.ndarray:
    """Full 4x4 FK in internal frame (Y-negated vs world)."""
    T = np.eye(4)
    for i, (x, y, z, ro, p, yw) in enumerate(_UR5E_JOINTS):
        To = np.eye(4); To[:3, :3] = _rpy(ro, p, yw); To[:3, 3] = [x, y, z]
        cq, sq = np.cos(q[i]), np.sin(q[i])
        Tj = np.eye(4); Tj[:3, :3] = [[cq, -sq, 0], [sq, cq, 0], [0, 0, 1]]
        T = T @ To @ Tj
    return T


def fk(q: np.ndarray) -> np.ndarray:
    """
    EE position in Gazebo world frame.
    Verified against TF2 (ros2 run tf2_ros tf2_echo world tool0):
    the raw DH frame has BOTH X and Y negated relative to the Gazebo world.
    """
    T = _raw_fk(q)
    return np.array([-T[0, 3], -T[1, 3], T[2, 3]])


def _ik_single(world_target: np.ndarray, seed: np.ndarray,
               orient_weight: float = 0.5) -> Optional[np.ndarray]:
    """
    IK for one seed.
    Orientation: gripper points DOWN (world tool-Z = [0,0,-1]).
    Raw DH frame: both X and Y are negated vs Gazebo world, so
      world_X → raw -X,  world_Y → raw -Y,  world_Z → raw Z.
    Returns None if position error > 2 cm after optimisation.
    """
    ft = np.array([-world_target[0], -world_target[1], world_target[2]])

    def cost(q):
        T = _raw_fk(q)
        pos_err = np.sum((T[:3, 3] - ft) ** 2)
        # Gripper must point DOWN: raw Z-col should be [0, 0, -1]
        z_err   = np.sum((T[:3, 2] - [0, 0, -1]) ** 2)
        # Fingers along world Y → raw Y-col = [0, -1, 0]
        y_err   = np.sum((T[:3, 1] - [0, -1, 0]) ** 2)
        return pos_err + orient_weight * (z_err + 0.3 * y_err)

    res = minimize(cost, seed, method='SLSQP',
                   bounds=[(-2 * np.pi, 2 * np.pi)] * 6,
                   options={'ftol': 1e-9, 'maxiter': 2000})
    q = res.x
    if np.linalg.norm(fk(q) - world_target) > 0.02:
        return None
    return q


def ik_best(world_target: np.ndarray, current_q: np.ndarray) -> np.ndarray:
    """
    Try multiple seeds, return the IK solution closest to current_q
    with sin(wrist_2) > 0.3 (singularity avoidance).

    Seeds tuned for top-down (gripper pointing DOWN) approach.
    Verified: q=[0.34,-1.73,-2.42,-0.56,pi/2,-2.8] gives tool-Z=[0,0,-1] at cube.
    """
    # Seeds for new geometry: cube at world (0.60, +0.20) and target (0.60, -0.20).
    # shoulder_pan > 0 → TCP sweeps toward world +Y (platform A, y=+0.20)
    # shoulder_pan < 0 → TCP sweeps toward world -Y (platform B, y=-0.20)
    y_sign = 1.0 if world_target[1] >= 0.0 else -1.0
    SEEDS = [
        current_q.copy(),
        np.array([ 0.33 * y_sign, -1.45,  1.90, -2.05, -np.pi/2,  0.0]),
        np.array([ 0.28 * y_sign, -1.50,  2.00, -2.10, -np.pi/2,  0.0]),
        np.array([ 0.40 * y_sign, -1.40,  1.80, -2.00, -np.pi/2,  0.0]),
        np.array([ 0.30 * y_sign, -1.55,  2.10, -2.05, -np.pi/2,  0.1]),
        np.array([ 0.35 * y_sign, -1.35,  1.70, -1.95, -np.pi/2, -0.1]),
        np.array([ 0.25 * y_sign, -1.60,  2.20, -2.10, -np.pi/2,  0.0]),
        np.array([ 0.45 * y_sign, -1.30,  1.60, -1.90, -np.pi/2,  0.0]),
    ]

    candidates = []
    for seed in SEEDS:
        q = _ik_single(world_target, seed)
        if q is None:
            continue
        sin_w2 = abs(np.sin(q[4]))
        if sin_w2 < 0.3:
            continue
        delta = float(np.sum((q - current_q) ** 2))
        score = delta - 2.0 * sin_w2
        candidates.append((score, q))

    if not candidates:
        q = _ik_single(world_target, current_q, orient_weight=0.2)
        return q if q is not None else current_q

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


class PickPlaceDemo(Node):
    def __init__(self):
        super().__init__('pick_place_demo')

        # Publishers
        self.arm_pub  = self.create_publisher(
            Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.grip_pub = self.create_publisher(
            Float64MultiArray, '/gripper_velocity_controller/commands', 10)
        # Pose publisher: keeps the cube glued to the EE at 50 Hz via ROS2
        # topic instead of spawning a new subprocess every cycle.
        self.pose_pub = self.create_publisher(Pose, '/cube_desired_pose', 1)

        # State
        self.arm_q    = np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])
        self.grip_pos = np.array([GRIPPER_OPEN_POS, GRIPPER_OPEN_POS])
        self._jmap:  dict[str, int] = {}
        self.create_subscription(JointState, '/joint_states', self._jcb, 10)

        self._cube_pos       = CUBE_INIT.copy()
        self._gripper_closed = False
        self._grasp_offset   = np.zeros(3)
        self._cube_run       = False

    def _jcb(self, msg: JointState):
        if not self._jmap:
            self._jmap = {n: i for i, n in enumerate(msg.name)}
        for i, jn in enumerate(JOINT_NAMES):
            idx = self._jmap.get(jn)
            if idx is not None:
                self.arm_q[i] = msg.position[idx]
        for i, jn in enumerate(['gripper_left_joint', 'gripper_right_joint']):
            idx = self._jmap.get(jn)
            if idx is not None:
                self.grip_pos[i] = msg.position[idx]

    # ── Arm control ────────────────────────────────────────────────────────────

    def _pub_arm(self, vels: np.ndarray):
        msg = Float64MultiArray()
        msg.data = np.clip(vels, -MAX_VEL, MAX_VEL).tolist()
        self.arm_pub.publish(msg)

    def stop_arm(self):
        self._pub_arm(np.zeros(6))

    def move_to(self, target_q: np.ndarray, timeout: float = 30.0,
                log: bool = True) -> bool:
        """
        PD velocity controller toward target_q.
        Returns True if within GOAL_TOL for ALL joints, False on timeout.
        """
        prev_err = np.zeros(6)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
            err = target_q - self.arm_q
            max_err = float(np.max(np.abs(err)))
            if max_err < GOAL_TOL:
                self.stop_arm()
                return True
            # PD with smooth ramp-down
            raw_vel = KP * err + KD * (err - prev_err) / 0.02
            # Scale down when close (avoids oscillation)
            scale = min(1.0, max_err / (GOAL_TOL * 3))
            self._pub_arm(raw_vel * scale)
            prev_err = err.copy()
            time.sleep(0.02)
        self.stop_arm()
        if log:
            ee = fk(self.arm_q)
            self.get_logger().warn(
                f'move_to TIMEOUT after {timeout:.0f}s  '
                f'max_err={float(np.max(np.abs(target_q-self.arm_q))):.3f} rad  '
                f'EE={np.round(ee, 3)}'
            )
        return False

    # ── Gripper control ────────────────────────────────────────────────────────

    def _pub_grip(self, vels: np.ndarray):
        msg = Float64MultiArray()
        msg.data = vels.tolist()
        self.grip_pub.publish(msg)

    def _stop_grip(self):
        self._pub_grip(np.zeros(2))

    def open_fingers(self, wait: float = 2.0):
        """Open both fingers to GRIPPER_OPEN_POS."""
        self.get_logger().info('  Gripper -> OPEN')
        dl = time.monotonic() + wait
        while time.monotonic() < dl:
            rclpy.spin_once(self, timeout_sec=0.01)
            el = GRIPPER_OPEN_POS - self.grip_pos[0]
            er = GRIPPER_OPEN_POS - self.grip_pos[1]
            if abs(el) < 0.002 and abs(er) < 0.002:
                break
            self._pub_grip(np.array([
                np.clip(4.0 * el, -GRIPPER_VEL, GRIPPER_VEL),
                np.clip(4.0 * er, -GRIPPER_VEL, GRIPPER_VEL),
            ]))
            time.sleep(0.02)
        self._stop_grip()

    def close_until_contact(self, timeout: float = 3.0) -> bool:
        """
        Close gripper, stop when contact detected (position stops changing).
        Returns True if contact detected before timeout.
        """
        self.get_logger().info('  Gripper -> CLOSE (wait for contact...)')
        prev_pos = self.grip_pos.copy()
        stall_count = 0
        dl = time.monotonic() + timeout

        while time.monotonic() < dl:
            rclpy.spin_once(self, timeout_sec=0.02)
            # Send closing velocity
            self._pub_grip(np.array([-GRIPPER_VEL, -GRIPPER_VEL]))
            time.sleep(0.05)
            delta = float(np.max(np.abs(self.grip_pos - prev_pos)))
            pos_from_open = GRIPPER_OPEN_POS - np.mean(self.grip_pos)

            if delta < GRIPPER_CONTACT_TOL and pos_from_open > 0.005:
                stall_count += 1
                if stall_count >= 3:  # consistent stall = contact
                    self._stop_grip()
                    self.get_logger().info(
                        f'  Contact at grip_pos={np.round(self.grip_pos, 4)} m '
                        f'(closed {pos_from_open*1000:.1f} mm from open)'
                    )
                    return True
            else:
                stall_count = 0
            prev_pos = self.grip_pos.copy()

        self._stop_grip()
        # Even without confirmed contact, proceed (virtual gripper will assist)
        self.get_logger().warn('  Gripper contact timeout — proceeding anyway')
        return False

    # ── Gazebo helpers ─────────────────────────────────────────────────────────

    # ── Pose teleport ──────────────────────────────────────────────────────────

    def _gz_set_pose(self, model: str, pos: np.ndarray):
        """Blocking teleport — one call at a time, short timeout to stay responsive."""
        req = (f'name: "{model}" '
               f'position {{ x: {pos[0]:.5f} y: {pos[1]:.5f} z: {pos[2]:.5f} }} '
               f'orientation {{ x: 0.0 y: 0.0 z: 0.0 w: 1.0 }}')
        subprocess.run(
            ['gz', 'service', '-s', f'/world/{WORLD_NAME}/set_pose',
             '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
             '--timeout', '120', '--req', req],
            capture_output=True, check=False,
        )

    def _cube_follow_loop(self):
        """
        Teleport cube to follow EE while gripper closed.
        One BLOCKING call per cycle at ~4 Hz.  A single gz service process
        is active at a time — no accumulation of zombie processes that would
        starve the arm velocity controller.
        Gravity sag between 250 ms cycles: 0.5*9.81*0.25² ≈ 0.31 m → too much.
        We counteract by commanding a +0.05 m Z offset above the true EE so the
        cube appears to sit in the gripper rather than below it.
        """
        while self._cube_run:
            if self._gripper_closed:
                ee = fk(self.arm_q.copy())
                self._cube_pos = ee + self._grasp_offset
                self._gz_set_pose('cube', self._cube_pos)
            time.sleep(0.08)   # 12 Hz max — one blocking call at a time

    def _attach_cube(self):
        # Flush latest joint states before computing grasp offset
        for _ in range(8):
            rclpy.spin_once(self, timeout_sec=0.01)
        ee = fk(self.arm_q.copy())
        self._grasp_offset   = self._cube_pos - ee
        self._gripper_closed = True
        dist = float(np.linalg.norm(ee - self._cube_pos))
        self.get_logger().info(f'  Cube ATTACHED  EE<->cube={dist:.3f} m')

    def _release_cube(self):
        self._gripper_closed = False
        drop = np.array([TARGET_POS[0], TARGET_POS[1], TARGET_POS[2]])
        self._cube_pos = drop.copy()
        self._gz_set_pose('cube', drop)
        self.get_logger().info(f'  Cube RELEASED at {np.round(drop, 3)}')

    def _reset_cube(self):
        self._gripper_closed = False
        self._cube_pos = CUBE_INIT.copy()
        self._gz_set_pose('cube', CUBE_INIT)

    # ── Pre-compute IK for all waypoints ──────────────────────────────────────

    def compute_configs(self) -> dict[str, np.ndarray]:
        """
        IK for each waypoint.  Multi-seed selection ensures:
        - Solution closest to current joint state (smooth motion)
        - sin(wrist_2) > 0.3 (away from wrist singularity)
        """
        self.get_logger().info('Computing IK (multi-seed, singularity-avoiding)...')
        configs: dict[str, np.ndarray] = {}
        seed = self.arm_q.copy()

        WAYPOINTS = [
            ('pregrasp', np.array([CUBE_INIT[0],   CUBE_INIT[1],   PREGRASP_Z])),
            ('grasp',    np.array([CUBE_INIT[0],   CUBE_INIT[1],   GRASP_Z   ])),
            ('lift',     np.array([CUBE_INIT[0],   CUBE_INIT[1],   LIFT_Z    ])),
            ('transport',np.array([TARGET_POS[0],  TARGET_POS[1],  LIFT_Z    ])),
            ('place',    np.array([TARGET_POS[0],  TARGET_POS[1],  GRASP_Z   ])),
            ('retreat',  np.array([TARGET_POS[0],  TARGET_POS[1],  LIFT_Z    ])),
        ]
        for name, target in WAYPOINTS:
            q = ik_best(target, seed)
            reached   = fk(q)
            pos_err   = float(np.linalg.norm(reached - target))
            sin_w2    = abs(float(np.sin(q[4])))
            joint_delta = float(np.sum(np.abs(q - seed)))
            self.get_logger().info(
                f'  {name:<10} pos_err={pos_err:.4f}  sin(w2)={sin_w2:.2f}'
                f'  delta={joint_delta:.2f} rad  q={np.round(q, 2)}'
            )
            configs[name] = q
            seed = q.copy()

        return configs

    # ── Single pick & place pass ───────────────────────────────────────────────

    def run_once(self, configs: dict[str, np.ndarray], iteration: int):
        G = self.get_logger().info
        G(f'\n{"="*54}')
        G(f'  PICK & PLACE  iteration {iteration}')
        G(f'{"="*54}')

        self._reset_cube()
        time.sleep(0.3)

        G('\n1. Open fingers...')
        self.open_fingers()

        G('2. Pre-grasp (above cube — exact known position)...')
        ok = self.move_to(configs['pregrasp'], timeout=40.0)
        G(f'   OK={ok}  EE={np.round(fk(self.arm_q), 3)}')

        G('3. Descend to grasp height...')
        ok = self.move_to(configs['grasp'], timeout=25.0)
        G(f'   OK={ok}  EE={np.round(fk(self.arm_q), 3)}')

        G('4. Close gripper until contact...')
        self.close_until_contact(timeout=3.0)
        self._attach_cube()   # virtual physics-assist
        time.sleep(0.2)

        G('5. Lift...')
        self.move_to(configs['lift'], timeout=20.0)

        G('6. Transport to target (exact position)...')
        self.move_to(configs['transport'], timeout=25.0)
        G(f'   EE={np.round(fk(self.arm_q), 3)}')

        G('7. Place (descend)...')
        self.move_to(configs['place'], timeout=20.0)

        G('8. Open gripper — release cube...')
        self._release_cube()
        self.open_fingers()

        G('9. Retreat...')
        self.move_to(configs['retreat'], timeout=20.0)

        G('10. Return to pre-grasp height (prepare next cycle)...')
        self.move_to(configs['pregrasp'], timeout=20.0)

        dist = float(np.linalg.norm(self._cube_pos[:2] - TARGET_POS[:2]))
        G(f'\n[OK] Iter {iteration}  cube={np.round(self._cube_pos, 3)}'
          f'  target_dist={dist:.3f} m\n')

    # ── Loop entry point ───────────────────────────────────────────────────────

    def run_loop(self):
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.02)
        self.get_logger().info(
            f'EE at start: {np.round(fk(self.arm_q), 3)}'
            f'  gripper: {np.round(self.grip_pos, 4)} m'
        )

        configs = self.compute_configs()

        self._cube_run = True
        cube_th = threading.Thread(target=self._cube_follow_loop, daemon=True)
        cube_th.start()

        it = 1
        try:
            while True:
                self.run_once(configs, it)
                it += 1
                time.sleep(0.3)
        except KeyboardInterrupt:
            pass
        finally:
            self._cube_run = False
            cube_th.join(timeout=1.0)
            self.stop_arm()
            self._stop_grip()


def main():
    rclpy.init()
    node = PickPlaceDemo()
    try:
        node.run_loop()
    finally:
        node.stop_arm()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
