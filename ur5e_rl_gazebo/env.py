"""
UR5e pick-and-place — GoalEnv HER-pur, saisie physique DetachableJoint.

Design (aligné sur FetchPickAndPlace / rl-baselines3-zoo) :
  - Reward SPARSE uniquement dans compute_reward : 0 si cube à <5 cm du but, -1 sinon.
    AUCUN shaping hors compute_reward → cohérence totale avec le relabeling HER.
  - Saisie physique : DetachableJoint soude cube↔palm dans Gazebo, les doigts se
    ferment réellement. Le cube suit le bras par la physique, tombe au lâcher.
  - Contrôle pulsé : vitesse pendant PHYSICS_STEPS ticks puis arrêt → la dynamique
    par step est déterministe, indépendante du temps de calcul SAC.
  - L'exploration de la saisie est fournie par démonstrations scriptées (train.py)
    + curriculum d'approche guidée au reset.

Action (4-dim) :
    [0:3]  delta TCP xyz  (±MAX_CART_STEP m par step)
    [3]    pince : >0.3 tenter saisie ; <-0.3 lâcher ; sinon conserver l'état

Observation dict (GoalEnv pour HER) :
    observation   : [6 joint_pos | 3 tcp_xyz | 3 tcp→cube | 1 pince | 3 platA | 3 platB]
    achieved_goal : [3 cube_xyz]   (position réelle Gazebo)
    desired_goal  : [3 peg_B_top]
"""
import time
import numpy as np
import gymnasium
from gymnasium import spaces

import rclpy

from .bridge import (
    UR5eBridge, HOME_POSITIONS, DISK_INIT_POS, PEG_B_TOP,
    JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH,
    fk_ur5e, cartesian_to_joint_vel,
)

MAX_CART_STEP   = 0.05    # déplacement TCP demandé par step (m)
MAX_JVEL        = 1.5     # norme max vitesses joints (rad/s)
PHYSICS_STEPS   = 10      # ticks joint_states par pulse (250 Hz → 40 ms de mouvement)
SETTLE_STEPS    = 1       # ticks d'arrêt après le pulse
MAX_STEPS       = 60
SUCCESS_THRESH  = 0.05    # standard Fetch
CUBE_FALL_Z     = 0.15

# Fenêtre d'alignement pour la saisie — calculée depuis la géométrie des doigts :
#   doigts : tool0+40 mm (palm) → centre à tool0+76 mm, demi-hauteur 36 mm
#   → les doigts couvrent tool0+[40, 112] mm ; faces internes ±31 mm fermé / ±56 mm ouvert
#   cube 60 mm posé sur le cap (top du cap = bas du cube)
#   anti-collision doigts/cap : (vert - 0.112) ≥ -0.030  →  vert ≥ 0.082
GRASP_HORIZ     = 0.02    # cube à <2 cm de l'axe pince (sinon il ne tient pas → gravité)
GRASP_VERT_LO   = 0.083   # tcp_z - cube_z min (bouts de doigts ≥1 mm au-dessus du cap)
GRASP_VERT_HI   = 0.110   # tcp_z - cube_z max (doigts couvrent encore le haut du cube)
HOVER_OFFSET    = 0.090   # hauteur TCP idéale : doigts autour du centre-haut du cube
HIGH_HOVER      = 0.20    # survol haut avant la descente verticale (évite de percuter le cube)

PLATFORM_A = np.array([0.60,  0.20, 0.32])
PLATFORM_B = np.array([0.60, -0.20, 0.32])

# Curriculum de randomisation de la position du cube.
# On commence petit (±2 cm) pour ne pas désapprendre brutalement la politique de base,
# puis on augmente progressivement jusqu'à ±8 cm sur la durée totale d'entraînement.
# Testé : sauter directement à ±8 cm depuis un modèle entraîné en fixe → régression
# immédiate (24% → 10%) car le modèle désapprend avant de réapprendre.
CUBE_RAND_XY_START = 0.02   # ±2 cm au début
CUBE_RAND_XY_END   = 0.08   # ±8 cm à la fin

TCP_LOW  = np.array([0.2, -0.7, 0.1])
TCP_HIGH = np.array([0.9,  0.7, 0.8])

GOAL_DIM = 3

# Curriculum : (seuil de progression, steps guidés au reset)
CURRICULUM = [
    (0.20, 15),
    (0.40,  8),
    (0.60,  4),
    (1.00,  0),
]
TOTAL_TIMESTEPS = 500_000


class UR5ePickPlaceEnv(gymnasium.Env):
    metadata = {'render_modes': []}

    def __init__(self, randomize_cube: bool = False, sensor_noise: bool = True):
        """
        randomize_cube : si True, la position initiale du cube est tirée aléatoirement
        dans ±CUBE_RAND_XY m (xy) autour de DISK_INIT_POS à chaque reset. Force le RL
        à généraliser plutôt que mémoriser la position fixe. À activer en entraînement
        une fois que la politique de base est acquise (sinon trop difficile à amorcer).

        sensor_noise : si True, ajoute du bruit gaussien aux observations pour simuler
        l'incertitude des capteurs réels (encodeurs joints, vision). Désactiver pour
        l'évaluation déterministe.
          - Joints   : σ = 2 mrad  (encodeur 17-bit UR5e → résolution ~0.006 deg)
          - Cube pos : σ = 3 mm    (estimation de pose par vision)
          - Action   : σ = 1 mm    (bruit d'actionneur / lag contrôleur)
        """
        super().__init__()
        self._randomize_cube = randomize_cube
        self._sensor_noise   = sensor_noise

        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low =np.concatenate([JOINT_LIMITS_LOW, TCP_LOW,  np.full(3,-2.), [0.],
                                     np.full(3,-2.), np.full(3,-2.)]).astype(np.float32),
                high=np.concatenate([JOINT_LIMITS_HIGH, TCP_HIGH, np.full(3, 2.), [1.],
                                     np.full(3, 2.), np.full(3, 2.)]).astype(np.float32),
                dtype=np.float32,
            ),
            'achieved_goal': spaces.Box(
                low=np.full(GOAL_DIM, -2., dtype=np.float32),
                high=np.full(GOAL_DIM, 2., dtype=np.float32), dtype=np.float32),
            'desired_goal': spaces.Box(
                low=np.full(GOAL_DIM, -2., dtype=np.float32),
                high=np.full(GOAL_DIM, 2., dtype=np.float32), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1., high=1., shape=(4,), dtype=np.float32)

        if not rclpy.ok():
            rclpy.init()
        self.node = UR5eBridge()

        self._step           = 0
        self._total_steps    = 0     # compteur global pour le curriculum
        self._gripper_closed = False
        self._grasp_age      = 0     # steps depuis la fermeture (serrage limité)

        # Paramètres de bruit (sim-to-real domain randomization)
        self._joint_noise_sigma = 0.002   # rad  — encodeur 17-bit UR5e
        self._cube_noise_sigma  = 0.003   # m    — incertitude estimation pose vision
        self._action_noise_sigma= 0.001   # m    — bruit actionneur / lag contrôleur

        time.sleep(0.5)

    # ── Curriculum ────────────────────────────────────────────────────────────

    def _curriculum_guided_steps(self) -> int:
        progress = min(self._total_steps / TOTAL_TIMESTEPS, 1.0)
        for threshold, n_steps in CURRICULUM:
            if progress < threshold:
                return n_steps
        return 0

    def _pulse(self, jvel: np.ndarray):
        """Vitesse pendant PHYSICS_STEPS ticks, puis arrêt → dynamique déterministe."""
        self.node.publish_velocity(jvel)
        self.node.wait_for_n_steps(n_steps=PHYSICS_STEPS)
        self.node.stop()
        self.node.wait_for_n_steps(n_steps=SETTLE_STEPS)

    def _cartesian_pulse(self, delta_tcp: np.ndarray):
        """Un pulse de mouvement cartésien avec garde des limites joints."""
        q = self.node.joint_pos.copy()
        tcp_vel = delta_tcp / (PHYSICS_STEPS * 0.004)
        jvel = cartesian_to_joint_vel(q, tcp_vel, MAX_JVEL)
        for i in range(6):
            if q[i] < JOINT_LIMITS_LOW[i]  + 0.1 and jvel[i] < 0: jvel[i] = 0.
            if q[i] > JOINT_LIMITS_HIGH[i] - 0.1 and jvel[i] > 0: jvel[i] = 0.
        self._pulse(jvel)

    def _guided_approach(self, n_steps: int):
        """Amène le TCP vers le cube en 2 phases : survol haut puis descente verticale
        (évite de percuter le cube latéralement avec les doigts)."""
        for _ in range(n_steps):
            tcp   = fk_ur5e(self.node.joint_pos)
            cube  = self.node.object_pos
            horiz = np.linalg.norm(tcp[:2] - cube[:2])
            if horiz > GRASP_HORIZ:
                target = cube + np.array([0., 0., HIGH_HOVER])
            else:
                target = cube + np.array([0., 0., HOVER_OFFSET])
            delta = target - tcp
            norm  = np.linalg.norm(delta)
            if norm < 0.015:
                break
            self._cartesian_pulse(delta / norm * min(norm, MAX_CART_STEP))

    # ── Saisie physique ───────────────────────────────────────────────────────

    def _grasp_aligned(self, tcp_pos, cube_pos) -> bool:
        horiz = np.linalg.norm(tcp_pos[:2] - cube_pos[:2])
        vert  = tcp_pos[2] - cube_pos[2]
        return horiz < GRASP_HORIZ and GRASP_VERT_LO < vert < GRASP_VERT_HI

    def _try_grasp(self, tcp_pos, cube_pos):
        """Saisie : exige l'alignement précis pince↔cube, sinon rien ne se passe
        (le cube reste soumis à la gravité, comme en réalité).

        Si aligné : les deux mâchoires en se fermant CENTRENT le cube sur l'axe
        de la pince (effet auto-centrant d'une pince parallèle) → on déplace le
        cube de ≤2 cm sur l'axe, puis joint fixe + serrage des doigts."""
        if self._gripper_closed or not self._grasp_aligned(tcp_pos, cube_pos):
            return
        horiz = np.linalg.norm(tcp_pos[:2] - cube_pos[:2])
        vert  = tcp_pos[2] - cube_pos[2]
        # Auto-centrage : cube ramené sur l'axe vertical de la pince (z inchangé)
        centered = np.array([tcp_pos[0], tcp_pos[1], cube_pos[2]])
        self.node.reset_object(centered)
        self.node.wait_for_n_steps(n_steps=2)
        self.node.attach_cube()
        self.node.close_gripper()
        self._gripper_closed = True
        self._grasp_age      = 0
        print(f'[grasp] tcp=({tcp_pos[0]:.3f},{tcp_pos[1]:.3f},{tcp_pos[2]:.3f}) '
              f'cube=({cube_pos[0]:.3f},{cube_pos[1]:.3f},{cube_pos[2]:.3f}) '
              f'horiz={horiz*1000:.0f}mm vert={vert*1000:.0f}mm → centré+serré')

    def _release(self):
        if not self._gripper_closed:
            return
        self.node.detach_cube()
        self.node.open_gripper()
        self._gripper_closed = False

    def _retreat(self, n_pulses: int = 4):
        """Dégagement après dépose : remontée verticale pure (~10 cm) pour
        sortir les doigts du cube posé sans le faire tomber."""
        self.node.wait_for_n_steps(n_steps=10)   # le cube se pose (~40 ms)
        for _ in range(n_pulses):
            self._cartesian_pulse(np.array([0., 0., MAX_CART_STEP]))

    # ── GoalEnv ───────────────────────────────────────────────────────────────

    def compute_reward(self, achieved_goal, desired_goal, info):
        """SPARSE — seule source de reward, cohérente avec le relabeling HER."""
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d > SUCCESS_THRESH).astype(np.float32)

    # ── Gymnasium ─────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.node.stop()
        self._gripper_closed = False
        self.node.reset_world()          # détache + ouvre + home + cube sur DISK_INIT_POS
        self.node.wait_for_n_steps(n_steps=15, timeout=3.)

        if self._randomize_cube:
            # Plage courante selon le curriculum : croît linéairement de START à END
            progress = min(self._total_steps / TOTAL_TIMESTEPS, 1.0)
            rand_range = CUBE_RAND_XY_START + progress * (CUBE_RAND_XY_END - CUBE_RAND_XY_START)
            rng = self.np_random  # fourni par gymnasium après super().reset(seed=seed)
            dx, dy = rng.uniform(-rand_range, rand_range, size=2)
            cube_start = DISK_INIT_POS + np.array([dx, dy, 0.0])
            self.node.reset_object(cube_start)
            self.node.wait_for_n_steps(n_steps=8, timeout=1.)

        n_guided = self._curriculum_guided_steps()
        if n_guided > 0:
            # _guided_approach lit self.node.object_pos dynamiquement → suit la nouvelle pos
            self._guided_approach(n_guided)

        self._step = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        delta_tcp   = np.clip(action[:3], -1., 1.) * MAX_CART_STEP
        gripper_cmd = float(action[3])

        # Bruit d'actionneur : simule le lag du contrôleur vitesse + bruit moteur.
        # Appliqué AVANT la commande → la politique apprend à être robuste.
        if self._sensor_noise:
            delta_tcp = delta_tcp + self.np_random.normal(
                0., self._action_noise_sigma, size=3).astype(np.float32)

        self._cartesian_pulse(delta_tcp)
        self._total_steps += 1
        self._step        += 1

        tcp_pos  = fk_ur5e(self.node.joint_pos)
        cube_pos = self.node.object_pos.copy()

        # Pince toggle : >0.3 saisir ; lâcher volontaire à -0.85 seulement.
        # À -0.3, le bruit résiduel de la politique (ent~0.03) lâchait le cube
        # en plein transport ~9 fois sur 12 (un seul sample <-0.3 suffit sur
        # ~20 steps). La dépose au but est scriptée → le lâcher volontaire est
        # quasi inutile ; -0.85 le garde possible mais jamais accidentel.
        if gripper_cmd > 0.3:
            self._try_grasp(tcp_pos, cube_pos)
        elif gripper_cmd < -0.85:
            self._release()

        # Serrage limité : couper la vitesse des doigts 3 steps après fermeture
        # (le cube est soudé — serrer en continu coince les doigts par pénétration)
        if self._gripper_closed:
            self._grasp_age += 1
            if self._grasp_age == 3:
                self.node.stop_gripper()

        cube_pos = self.node.object_pos.copy()
        achieved = cube_pos.astype(np.float32)
        desired  = PEG_B_TOP.astype(np.float32)

        d_goal     = np.linalg.norm(achieved - desired)
        terminated = bool(d_goal <= SUCCESS_THRESH)

        info   = {'grasped': self._gripper_closed, 'is_success': terminated}
        reward = float(self.compute_reward(achieved, desired, info))

        # Cube tombé/éjecté : on le remet sur A et l'épisode CONTINUE.
        # NE PAS tronquer — sinon éjecter coûte ~-20 contre -60 pour un échec
        # complet, et la politique APPREND à percuter le cube (observé run -2.5 :
        # ep_len 19, succès 22%). Le coût devient organique : re-approcher = steps.
        cube_lost = bool(
            (not self._gripper_closed and cube_pos[2] < CUBE_FALL_Z)
            or abs(cube_pos[0]) > 1.2 or abs(cube_pos[1]) > 1.2
        )
        if cube_lost:
            print(f'[cube_lost] step={self._step} '
                  f'cube=({cube_pos[0]:.2f},{cube_pos[1]:.2f},{cube_pos[2]:.2f})')
            self.node.reset_object()

        truncated = self._step >= MAX_STEPS

        if terminated:
            # Dépose complète : lâcher, laisser poser, puis dégager la pince
            # verticalement sans percuter le cube (cycle pick-and-place réaliste)
            self._release()
            self._retreat()
            self.node.stop()
        elif truncated:
            self.node.stop()

        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        self.node.stop()
        if self._gripper_closed:
            self.node.detach_cube()
        try:
            self.node._spin_executor.shutdown(wait_for_completion=False)
        except Exception:
            pass
        self.node.destroy_node()
        rclpy.shutdown()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        q        = self.node.joint_pos.copy()
        tcp_pos  = fk_ur5e(q)
        cube_pos = self.node.object_pos.copy()
        gripper  = np.array([1. if self._gripper_closed else 0.])

        # Bruit capteurs (domain randomization sim-to-real) :
        #   • Joints   : encodeur 17-bit → résolution ~0.006 deg, σ=2 mrad
        #   • Cube pos : estimation de pose par vision monoculaire, σ=3 mm
        # Désactivé en évaluation (sensor_noise=False) pour des métriques propres.
        if self._sensor_noise:
            q        = q + self.np_random.normal(
                0., self._joint_noise_sigma, size=q.shape).astype(np.float32)
            tcp_pos  = fk_ur5e(q)  # propagation FK du bruit joints → TCP
            cube_pos = cube_pos + self.np_random.normal(
                0., self._cube_noise_sigma, size=3).astype(np.float32)

        obs = np.concatenate([q, tcp_pos, cube_pos - tcp_pos, gripper,
                              PLATFORM_A, PLATFORM_B]).astype(np.float32)
        return {
            'observation':   np.clip(obs,
                                     self.observation_space['observation'].low,
                                     self.observation_space['observation'].high),
            'achieved_goal': cube_pos.astype(np.float32),
            'desired_goal':  PEG_B_TOP.astype(np.float32),
        }
