"""
SAC + HER + démonstrations pour UR5e pick-and-place.

Hyperparamètres alignés sur rl-baselines3-zoo FetchPickAndPlace (TQC+HER),
réduits pour CPU 4 cœurs : gamma=0.95, tau=0.05, n_sampled_goal=4, reward sparse.
Les démonstrations scriptées (contrôle cartésien + saisie) sont injectées dans le
replay buffer HER avant l'entraînement (Nair et al. 2018 — "Overcoming
Exploration in RL with Demonstrations").

Run:
    ros2 run ur5e_rl_gazebo train
"""
import os, time, pickle
import faulthandler
faulthandler.enable()   # trace les segfaults C (DDS/gz) dans stderr
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from .env import (
    UR5ePickPlaceEnv, MAX_CART_STEP, HOVER_OFFSET, HIGH_HOVER,
    SUCCESS_THRESH, MAX_STEPS,
)
from .bridge import PEG_B_TOP, DISK_INIT_POS, fk_ur5e

# ── Hyperparamètres (zoo FetchPickAndPlace, adaptés CPU) ───────────────────────
TOTAL_TIMESTEPS = 500_000
LEARNING_RATE   = 5e-4
BATCH_SIZE      = 256       # CPU 4 cœurs : 512 + train_freq=1 → fps=1
BUFFER_SIZE     = 1_000_000
LEARNING_STARTS = 1_000
GAMMA           = 0.95
TAU             = 0.05
TRAIN_FREQ      = 2         # une update tous les 2 steps
GRADIENT_STEPS  = 1
N_SAMPLED_GOALS = 4
GOAL_STRATEGY   = 'future'
# FIXE : l'auto-tuning a collapsé 2×/2 sur cette tâche (runs 13 et 24 :
# ent_coef → <0.01 au plateau de succès → plus d'exploration → spirale d'échec)
ENT_COEF        = 0.05
NET_ARCH        = [256, 256]

N_DEMOS         = 60
DEMO_NOISE      = 0.03
DEMO_PATH       = './demos_ur5e.pkl'

SAVE_FREQ       = 5_000
CHECKPOINT_DIR  = './checkpoints'
TB_LOG_DIR      = './tb_logs'
LOG_FREQ        = 400


# ── Démonstrations scriptées ───────────────────────────────────────────────────

def _scripted_action(env: UR5ePickPlaceEnv, rng: np.random.Generator) -> np.ndarray:
    """Politique experte : approche → saisie → levée → transport → dépose."""
    tcp  = fk_ur5e(env.node.joint_pos)
    cube = env.node.object_pos.copy()
    goal = PEG_B_TOP

    if not env._gripper_closed:
        horiz = np.linalg.norm(tcp[:2] - cube[:2])
        vert  = tcp[2] - cube[2]
        if horiz > 0.015:
            # Phase 1 : alignement horizontal en survol haut
            target = cube + np.array([0., 0., HIGH_HOVER])
        else:
            # Phase 2 : descente verticale, doigts autour du cube
            target = cube + np.array([0., 0., HOVER_OFFSET])
        delta = target - tcp
        grip  = 1.0 if (horiz < 0.015 and 0.085 < vert < 0.105) else 0.0
    else:
        horiz_goal = np.linalg.norm(cube[:2] - goal[:2])
        if cube[2] < 0.45 and horiz_goal > 0.10:
            delta = np.array([0., 0., 0.06])                  # levée
        elif horiz_goal > 0.03:
            delta = np.array([goal[0]-cube[0], goal[1]-cube[1], 0.45-cube[2]])
        else:
            delta = goal - cube                               # descente finale
        grip = 1.0

    a_xyz = np.clip(delta / MAX_CART_STEP, -1., 1.)
    a_xyz = np.clip(a_xyz + rng.normal(0., DEMO_NOISE, 3), -1., 1.)
    return np.concatenate([a_xyz, [grip]]).astype(np.float32)


def generate_demos(env, n_demos: int = N_DEMOS, seed: int = 0,
                   save_path: str = DEMO_PATH, existing: list | None = None) -> list:
    """Episodes experts complets : liste d'épisodes de transitions SB3-ready.
    Sauvegarde incrémentale tous les 10 épisodes (protège des crashs)."""
    rng = np.random.default_rng(seed)
    episodes = list(existing) if existing else []
    n_ok = len(episodes)
    while n_ok < n_demos:
        obs, _ = env.reset()
        transitions, success = [], False
        for _ in range(MAX_STEPS):
            action = _scripted_action(env.unwrapped, rng)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transitions.append((
                obs, action, float(reward), next_obs, bool(done),
                {'is_success': bool(info.get('is_success', False)),
                 'TimeLimit.truncated': bool(truncated and not terminated)},
            ))
            obs = next_obs
            if done:
                success = terminated
                break
        if success:
            episodes.append(transitions)
            n_ok += 1
            print(f'[demos] {n_ok}/{n_demos} ✓ ({len(transitions)} steps)')
            if save_path and n_ok % 10 == 0:
                with open(save_path, 'wb') as f:
                    pickle.dump(episodes, f)
                print(f'[demos] checkpoint : {n_ok} épisodes sauvés')
        else:
            print(f'[demos] échec ({len(transitions)} steps) — rejouée')
    return episodes


def inject_demos(model: SAC, episodes: list):
    """Remplit le HerReplayBuffer avec les transitions expertes."""
    buf, n = model.replay_buffer, 0
    for ep in episodes:
        for obs, action, reward, next_obs, done, info in ep:
            buf.add(
                obs={k: np.asarray(v, dtype=np.float32)[None] for k, v in obs.items()},
                next_obs={k: np.asarray(v, dtype=np.float32)[None] for k, v in next_obs.items()},
                action=np.asarray(action, dtype=np.float32)[None],
                reward=np.array([reward], dtype=np.float32),
                done=np.array([done]),
                infos=[info],
            )
            n += 1
    print(f'[demos] {n} transitions expertes injectées dans le buffer HER')


# ── Logging saisies ────────────────────────────────────────────────────────────

class GraspLogger(BaseCallback):
    """Compte saisies, lâchers et succès par fenêtre de LOG_FREQ steps."""

    def __init__(self, log_freq: int = LOG_FREQ):
        super().__init__()
        self.log_freq = log_freq
        self.grasps = self.releases = self.successes = 0
        self._was_grasped = False
        self._last_log    = 0

    def _on_step(self) -> bool:
        info = self.locals.get('infos', [{}])[0]
        grasped_now = info.get('grasped', False)

        if grasped_now and not self._was_grasped:
            self.grasps += 1
        if not grasped_now and self._was_grasped:
            self.releases += 1
        self._was_grasped = grasped_now

        if self.locals.get('dones', [False])[0] and info.get('is_success', False):
            self.successes += 1

        if self.num_timesteps - self._last_log >= self.log_freq:
            print(
                f'[saisies] step={self.num_timesteps:>7,} | '
                f'saisies={self.grasps:>4} | '
                f'lâchers={self.releases:>4} | '
                f'succès={self.successes:>3} | '
                f'curriculum={self.training_env.envs[0].unwrapped._curriculum_guided_steps()} guidés'
            )
            self.grasps = self.releases = self.successes = 0
            self._last_log = self.num_timesteps

        return True


# ── Entraînement ───────────────────────────────────────────────────────────────

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)

    print('[train] Création de l\'environnement (randomize_cube=True, sensor_noise=True)...')
    env = Monitor(UR5ePickPlaceEnv(randomize_cube=True, sensor_noise=True))

    resume = os.environ.get('RESUME_FROM', '')
    if resume and os.path.exists(resume):
        # NB : impossible de passer ent_coef auto→fixe au load (state_dict de
        # l'optimiseur d'entropie présent dans le zip → AttributeError).
        # On reste en auto avec cible -2.5 : -1.0 maintenait trop de bruit
        # (saisies à 17-19 mm de l'axe → plafond ~55%) ; -4 (défaut) collapse.
        print(f'[train] Reprise depuis {resume} (target_entropy=-2.5)')
        model = SAC.load(
            resume, env=env,
            custom_objects={'target_entropy': -2.5},
            tensorboard_log=TB_LOG_DIR,
        )
    else:
        print('[train] Construction SAC + HER...')
        model = SAC(
            policy='MultiInputPolicy',
            env=env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=N_SAMPLED_GOALS,
                goal_selection_strategy=GOAL_STRATEGY,
            ),
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            learning_starts=LEARNING_STARTS,
            batch_size=BATCH_SIZE,
            tau=TAU,
            gamma=GAMMA,
            train_freq=TRAIN_FREQ,
            gradient_steps=GRADIENT_STEPS,
            ent_coef=ENT_COEF,
            policy_kwargs=dict(net_arch=NET_ARCH),
            verbose=1,
            tensorboard_log=TB_LOG_DIR,
        )

    # ── Démonstrations : cache disque (éventuellement partiel) + complément ───
    episodes = []
    if os.path.exists(DEMO_PATH):
        with open(DEMO_PATH, 'rb') as f:
            episodes = pickle.load(f)
        print(f'[demos] {len(episodes)} épisodes chargés depuis {DEMO_PATH}')
    if len(episodes) < N_DEMOS:
        print(f'[demos] Génération de {N_DEMOS - len(episodes)} démonstrations...')
        episodes = generate_demos(env, N_DEMOS, seed=len(episodes), existing=episodes)
        with open(DEMO_PATH, 'wb') as f:
            pickle.dump(episodes, f)
        lens = [len(e) for e in episodes]
        print(f'[demos] Sauvées — longueur moyenne {np.mean(lens):.0f} steps')
    # Injection ×2 : renforce le poids des transitions expertes dans le buffer
    # (à 40k+ steps de buffer, une seule injection se dilue à <5 %)
    inject_demos(model, episodes + episodes)

    # Préfixe horodaté : les reprises repartent de step 0 → sans ça, leurs
    # checkpoints écrasent ceux du run précédent (perdu : le pic à 86 %)
    run_tag = time.strftime('%m%d_%H%M')
    callbacks = [
        CheckpointCallback(save_freq=SAVE_FREQ, save_path=CHECKPOINT_DIR,
                           name_prefix=f'sac_{run_tag}', verbose=1),
        GraspLogger(log_freq=LOG_FREQ),
    ]

    print(f'[train] Démarrage — {TOTAL_TIMESTEPS:,} timesteps (SAC+HER+demos, sparse)')
    t0 = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        tb_log_name='sac_her_ur5e',
        reset_num_timesteps=True,
    )
    elapsed = time.time() - t0

    path = os.path.join(CHECKPOINT_DIR, 'sac_her_ur5e_final')
    model.save(path)
    print(f'[train] Terminé en {elapsed/3600:.1f}h — sauvegardé : {path}')
    env.close()


def evaluate(model_path: str, n_episodes: int = 10, randomize: bool = False,
             rand_xy_range: float = 0.08, seed: int = 42):
    """Évalue le modèle en mode déterministe (sans bruit d'exploration).

    randomize=True : position initiale du cube tirée aléatoirement dans un carré
    de ±rand_xy_range autour de DISK_INIT_POS (x,y). La hauteur z reste fixe
    (cube posé sur la plate-forme). Cela teste la généralisation du modèle au-delà
    des positions mémorisées en entraînement.
    rand_xy_range=0.08 → ±8 cm, ce qui couvre les variations raisonnables sur la
    plate-forme A sans risquer de placer le cube hors de portée du bras.
    """
    env = UR5ePickPlaceEnv(sensor_noise=False)  # déterministe pour métriques propres
    model = SAC.load(model_path, env=env)
    rng = np.random.default_rng(seed)
    rewards, successes = [], []

    print(f'\n[eval] {"positions RANDOMISÉES" if randomize else "position FIXE"}'
          f' — {n_episodes} épisodes, déterministe')
    if randomize:
        print(f'[eval] plage xy : ±{rand_xy_range*100:.0f} cm autour de '
              f'({DISK_INIT_POS[0]:.2f}, {DISK_INIT_POS[1]:.2f})\n')

    for ep in range(n_episodes):
        if randomize:
            dx, dy = rng.uniform(-rand_xy_range, rand_xy_range, size=2)
            cube_start = DISK_INIT_POS + np.array([dx, dy, 0.0])
            # Téléporte le cube avant le reset (reset_world le reposera sur A,
            # donc on le bouge APRÈS reset)
            obs, _ = env.reset()
            env.node.reset_object(cube_start)
            env.node.wait_for_n_steps(n_steps=10, timeout=1.)
            obs = env._get_obs()
            print(f'  Ep {ep+1:2d} cube=({cube_start[0]:.3f},{cube_start[1]:.3f})', end='  ')
        else:
            obs, _ = env.reset()

        ep_reward, done, terminated = 0.0, False, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        successes.append(terminated)
        suffix = '✓' if terminated else '✗'
        print(f'reward={ep_reward:.1f}  succès={terminated} {suffix}')

    rate = sum(successes) / len(successes) * 100
    print(f'\nReward moyen : {sum(rewards)/len(rewards):.2f}')
    print(f'Taux de succès : {rate:.0f}%')
    env.close()
    return rate


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(CHECKPOINT_DIR, 'sac_her_ur5e_final')
        n    = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        rand = len(sys.argv) > 4 and sys.argv[4] == 'randomize'
        evaluate(path, n_episodes=n, randomize=rand)
    else:
        train()


if __name__ == '__main__':
    main()
