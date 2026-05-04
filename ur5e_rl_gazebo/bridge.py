"""
ROS2 bridge Gazebo ↔ gymnasium pour UR5e RL.

FK et Jacobien analytiques (pas de TF2) — <1ms par appel.
FK vérifiée contre TF2 : X et Y sont négativés par rapport au frame DH interne.
"""
import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
GRIPPER_JOINT_NAMES = ['gripper_left_joint', 'gripper_right_joint']
GRIPPER_OPEN_POS    = 0.025   # butée haute des doigts (m)

HOME_POSITIONS = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

DISK_INIT_POS  = np.array([0.60,  0.20, 0.35])
DISK_INIT_QUAT = np.array([0.0, 0.0, 0.0, 1.0])
PEG_B_TOP      = np.array([0.60, -0.20, 0.35])
TARGET_POS     = PEG_B_TOP
CUBE_INIT_POS  = DISK_INIT_POS
WORLD_NAME     = 'pick_place'

# Limites joints UR5e (soft limits conservateurs pour éviter auto-collision)
JOINT_LIMITS_LOW  = np.array([-2*np.pi, -np.pi,      0.0,      -2*np.pi, -2*np.pi, -2*np.pi])
JOINT_LIMITS_HIGH = np.array([ 2*np.pi,  0.0,    np.pi,        2*np.pi,  2*np.pi,  2*np.pi])

# ── Cinématique UR5e (joints transforms URDF) ─────────────────────────────────
_UR5E_JOINTS = [
    (0,       0,       0.1625,  0,          0,       0),
    (0,       0,       0,       np.pi/2,    0,       0),
    (-0.425,  0,       0,       0,          0,       0),
    (-0.3922, 0,       0.1333,  0,          0,       0),
    (0,      -0.0997,  0,       np.pi/2,    0,       0),
    (0,       0.09959, 0,       np.pi/2,    np.pi,   np.pi),
]

def _rpy(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return (np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
          @ np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
          @ np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]]))

def _raw_fk(q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for i, (x, y, z, ro, p, yw) in enumerate(_UR5E_JOINTS):
        To = np.eye(4); To[:3,:3] = _rpy(ro,p,yw); To[:3,3] = [x,y,z]
        cq, sq = np.cos(q[i]), np.sin(q[i])
        Tj = np.eye(4); Tj[:3,:3] = [[cq,-sq,0],[sq,cq,0],[0,0,1]]
        T = T @ To @ Tj
    return T

def fk_ur5e(q: np.ndarray) -> np.ndarray:
    """TCP en world frame [x,y,z] — vérifié contre TF2 (X et Y négativés)."""
    T = _raw_fk(q)
    return np.array([-T[0,3], -T[1,3], T[2,3]])

def fk_ur5e_pose(q: np.ndarray) -> np.ndarray:
    """TCP [x,y,z,qx,qy,qz,qw] en world frame."""
    T = _raw_fk(q)
    pos = np.array([-T[0,3], -T[1,3], T[2,3]])
    # Miroir X et Y sur la rotation aussi
    R = T[:3,:3].copy()
    R[0,:] = -R[0,:]
    R[1,:] = -R[1,:]
    # Quaternion depuis R
    trace = R[0,0]+R[1,1]+R[2,2]
    if trace > 0:
        s = 0.5/np.sqrt(trace+1); qw=0.25/s
        qx=(R[2,1]-R[1,2])*s; qy=(R[0,2]-R[2,0])*s; qz=(R[1,0]-R[0,1])*s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2]); qw=(R[2,1]-R[1,2])/s
        qx=0.25*s; qy=(R[0,1]+R[1,0])/s; qz=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:
        s=2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2]); qw=(R[0,2]-R[2,0])/s
        qx=(R[0,1]+R[1,0])/s; qy=0.25*s; qz=(R[1,2]+R[2,1])/s
    else:
        s=2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1]); qw=(R[1,0]-R[0,1])/s
        qx=(R[0,2]+R[2,0])/s; qy=(R[1,2]+R[2,1])/s; qz=0.25*s
    return np.array([pos[0],pos[1],pos[2],qx,qy,qz,qw])

def jacobian(q: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Jacobien numérique 3×6 (position TCP en world frame)."""
    J = np.zeros((3, 6))
    p0 = fk_ur5e(q)
    for i in range(6):
        qp = q.copy(); qp[i] += eps
        J[:, i] = (fk_ur5e(qp) - p0) / eps
    return J

def cartesian_to_joint_vel(q: np.ndarray, tcp_vel: np.ndarray,
                            max_jvel: float = 1.5) -> np.ndarray:
    """Jacobien pseudo-inverse : vitesse TCP → vitesses joints."""
    J = jacobian(q)
    # Pseudo-inverse avec damping (évite singularités)
    lam = 1e-4
    JJT = J @ J.T + lam * np.eye(3)
    J_pinv = J.T @ np.linalg.inv(JJT)   # 6×3
    dq = J_pinv @ tcp_vel
    # Clip aux limites de vitesse
    norm = np.linalg.norm(dq)
    if norm > max_jvel:
        dq = dq * max_jvel / norm
    return dq


class UR5eBridge(Node):

    def __init__(self):
        super().__init__('ur5e_rl_bridge')

        self.joint_pos   = HOME_POSITIONS.copy()
        self.joint_vel   = np.zeros(6)
        self.gripper_pos = np.full(2, GRIPPER_OPEN_POS)
        self.object_pos  = DISK_INIT_POS.copy()
        self.object_quat = DISK_INIT_QUAT.copy()
        self.cube_pos    = self.object_pos

        self.step_count = 0
        self._new_state_event = threading.Event()
        self._joint_name_map: dict[str, int] = {}

        self.vel_pub = self.create_publisher(
            Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.grip_pub = self.create_publisher(
            Float64MultiArray, '/gripper_velocity_controller/commands', 10)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.create_subscription(TFMessage, '/world_dynamic_poses', self._poses_cb, 10)

        # Spinner dédié dans thread daemon — ne bloque pas le thread training
        self._spin_executor = MultiThreadedExecutor(num_threads=2)
        self._spin_executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._spin_executor.spin, daemon=True)
        self._spin_thread.start()

        # Le plugin DetachableJoint démarre ATTACHÉ : détacher immédiatement.
        # sleep 1.5 s : laisse la découverte DDS s'établir avant de publier.
        self.cube_attached = False
        time.sleep(1.5)
        self.detach_cube()
        self.open_gripper_verified()

        self.get_logger().info('UR5eBridge ready')

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _joint_cb(self, msg: JointState):
        if not self._joint_name_map:
            self._joint_name_map = {name: i for i, name in enumerate(msg.name)}
        for out_idx, jname in enumerate(JOINT_NAMES):
            src_idx = self._joint_name_map.get(jname)
            if src_idx is not None:
                self.joint_pos[out_idx] = msg.position[src_idx]
                self.joint_vel[out_idx] = msg.velocity[src_idx]
        for out_idx, jname in enumerate(GRIPPER_JOINT_NAMES):
            src_idx = self._joint_name_map.get(jname)
            if src_idx is not None:
                self.gripper_pos[out_idx] = msg.position[src_idx]
        self.step_count += 1
        self._new_state_event.set()

    def _poses_cb(self, msg: TFMessage):
        # Ignorer les messages pendant la fenêtre de reset (évite race condition)
        if getattr(self, '_reset_guard', False):
            return
        for tf in msg.transforms:
            if tf.child_frame_id in ('cube', 'cube::link'):
                t = tf.transform.translation; q = tf.transform.rotation
                self.object_pos  = np.array([t.x, t.y, t.z])
                self.object_quat = np.array([q.x, q.y, q.z, q.w])
                self.cube_pos    = self.object_pos
                return
        if msg.transforms:
            t = msg.transforms[0].transform.translation
            q = msg.transforms[0].transform.rotation
            pos = np.array([t.x, t.y, t.z])
            if -1.5 <= pos[0] <= 1.5 and -1.5 <= pos[1] <= 1.5 and 0.0 <= pos[2] <= 1.0:
                self.object_pos  = pos
                self.object_quat = np.array([q.x, q.y, q.z, q.w])
                self.cube_pos    = self.object_pos

    # ── API publique ───────────────────────────────────────────────────────────

    def publish_velocity(self, joint_vels: np.ndarray):
        msg = Float64MultiArray()
        msg.data = joint_vels.tolist()
        self.vel_pub.publish(msg)

    def stop(self):
        self.publish_velocity(np.zeros(6))

    # ── Pince physique ─────────────────────────────────────────────────────────

    def close_gripper(self):
        """Fermeture SANS contact : course limitée à ~19 mm → faces internes à
        ±36 mm pour un cube de 60 mm (≈6 mm de jeu par côté). Le cube est soudé
        (DetachableJoint) : serrer à 40 N ne fait que pénétrer le cube et
        coincer les doigts (constaté : doigt droit prisonnier)."""
        msg = Float64MultiArray(); msg.data = [-0.15, -0.15]
        self.grip_pub.publish(msg)
        time.sleep(0.13)
        self.stop_gripper()

    def open_gripper(self):
        """Vitesse positive : les doigts s'ouvrent jusqu'à la butée (25 mm)."""
        msg = Float64MultiArray(); msg.data = [0.15, 0.15]
        for _ in range(3):
            self.grip_pub.publish(msg)
            time.sleep(0.02)

    def stop_gripper(self):
        """Coupe le serrage (vitesse nulle) — les doigts restent en place.
        Évite la pénétration de contact du serrage permanent 40 N qui
        coince les doigts (le cube est soudé, serrer ne sert à rien)."""
        msg = Float64MultiArray(); msg.data = [0.0, 0.0]
        self.grip_pub.publish(msg)

    def open_gripper_verified(self, retries: int = 4) -> bool:
        """Ouvre et VÉRIFIE via joint_states que les doigts atteignent la butée.

        Indispensable : si un trainer meurt pince fermée, le contrôleur conserve
        la commande de fermeture — le cube se coince alors entre les doigts et
        se fait traîner par friction (constaté run 26)."""
        for attempt in range(retries):
            self.open_gripper()
            time.sleep(0.35)   # course 25 mm à 0.15 m/s ≈ 0.17 s
            if np.all(self.gripper_pos > 0.020):
                return True
            # Doigt coincé (pénétration de contact) : wiggle fermer→ouvrir
            if attempt < retries - 1:
                self.close_gripper()
                time.sleep(0.10)
        self.get_logger().warn(
            f'open_gripper: doigts toujours fermés ({self.gripper_pos}) après retries')
        return False

    def _gz_pub_empty(self, topic: str):
        """Publie gz.msgs.Empty — la découverte gz transport peut dépasser 2 s
        sous charge : timeout long + retry, sans jamais lever d'exception."""
        for _ in range(2):
            try:
                subprocess.run(
                    ['gz', 'topic', '-t', topic, '-m', 'gz.msgs.Empty', '-p', ' '],
                    check=False, capture_output=True, timeout=5.0)
                return
            except subprocess.TimeoutExpired:
                continue
        self.get_logger().warn(f'gz topic pub {topic} : timeout ×2 (message peut-être perdu)')

    def attach_cube(self):
        """Soude cube ↔ palm via DetachableJoint (joint fixe à l'offset courant)."""
        self._gz_pub_empty('/cube/attach')
        self.cube_attached = True

    def detach_cube(self):
        self._gz_pub_empty('/cube/detach')
        self.cube_attached = False

    def force_cube_to(self, pos: np.ndarray, retries: int = 4) -> bool:
        """Détache + téléporte le cube et VÉRIFIE qu'il y reste.

        Un message gz topic publié par subprocess peut se perdre sous charge :
        si le detach est perdu, le cube reste soudé au poignet et la soudure le
        ramène après téléportation → on détecte (position ≠ cible) et on retry."""
        for _ in range(retries):
            self.detach_cube()
            self.reset_object(pos)
            time.sleep(0.15)   # la physique + poses_cb rafraîchissent object_pos
            if np.linalg.norm(self.object_pos - pos) < 0.05:
                return True
        return False

    def get_ee_pos(self) -> np.ndarray:
        return fk_ur5e(self.joint_pos)

    def get_ee_pose(self) -> np.ndarray:
        return fk_ur5e_pose(self.joint_pos)

    def wait_for_n_steps(self, executor=None, n_steps: int = 10, timeout: float = 2.0):
        target   = self.step_count + n_steps
        deadline = time.monotonic() + timeout
        while self.step_count < target:
            if time.monotonic() > deadline:
                break
            self._new_state_event.clear()
            self._new_state_event.wait(timeout=0.02)

    # ── Reset ──────────────────────────────────────────────────────────────────

    def go_home(self, timeout: float = 2.0):
        """Ramène les joints à HOME_POSITIONS par contrôle P bloquant."""
        KP = 3.0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            err = HOME_POSITIONS - self.joint_pos
            if np.max(np.abs(err)) < 0.08:
                break
            vel = np.clip(KP * err, -2.0, 2.0)
            self.publish_velocity(vel)
            time.sleep(0.02)
        self.publish_velocity(np.zeros(6))

    def reset_world(self):
        # 0. ORDRE CRITIQUE : éloigner le cube AVANT d'ouvrir — un doigt ayant
        #    pénétré le cube (serrage) est prisonnier tant que le cube est là.
        self.stop()
        self.force_cube_to(DISK_INIT_POS)
        self.open_gripper_verified()
        # 1. Retour home (joints physiques dans Gazebo)
        self.go_home()
        # 2. Re-vérification après le mouvement (le bras a pu frôler le cube)
        if not self.force_cube_to(DISK_INIT_POS):
            self.get_logger().warn('reset_world: cube toujours hors position après retries')
        self.joint_vel  = np.zeros(6)
        self.step_count = 0

    def reset_object(self, pos: np.ndarray | None = None):
        if pos is None:
            pos = DISK_INIT_POS
        req = (f'name: "cube" '
               f'position {{ x: {pos[0]:.4f} y: {pos[1]:.4f} z: {pos[2]:.4f} }} '
               f'orientation {{ x: 0.0 y: 0.0 z: 0.0 w: 1.0 }}')
        try:
            subprocess.run(
                ['gz', 'service', '-s', f'/world/{WORLD_NAME}/set_pose',
                 '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                 '--timeout', '500', '--req', req],
                check=False, capture_output=True, timeout=5.0)
        except subprocess.TimeoutExpired:
            self.get_logger().warn('gz service set_pose : timeout')
        self.object_pos  = pos.copy()
        self.object_quat = DISK_INIT_QUAT.copy()
        self.cube_pos    = self.object_pos

    def reset_cube(self, pos=None):
        self.reset_object(pos)
