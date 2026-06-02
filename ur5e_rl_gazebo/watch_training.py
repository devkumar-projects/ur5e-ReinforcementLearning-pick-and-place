#!/usr/bin/env python3
"""Terminal dashboard for SAC training — reads TensorBoard event files."""
import glob
import os
import sys
import time
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("pip install tensorboard")
    sys.exit(1)

LOGDIR = os.path.expanduser("~/ros2_ws/tb_logs")
REFRESH = 10  # seconds

TAGS = [
    ("rollout/ep_rew_mean",   "Reward moyen (ep)"),
    ("rollout/ep_len_mean",   "Longueur ep (steps)"),
    ("train/actor_loss",      "Actor loss"),
    ("train/critic_loss",     "Critic loss"),
    ("train/ent_coef",        "Entropie coef"),
]

BAR_WIDTH = 40

def sparkline(vals, width=30):
    if len(vals) < 2:
        return " " * width
    mn, mx = min(vals), max(vals)
    rng = mx - mn or 1
    blocks = " ▁▂▃▄▅▆▇█"
    line = ""
    sample = vals[max(0, len(vals)-width):]
    for v in sample:
        idx = int((v - mn) / rng * (len(blocks) - 1))
        line += blocks[idx]
    return line.ljust(width)

def color(text, code):
    return f"\033[{code}m{text}\033[0m"

def find_latest_run(logdir):
    runs = glob.glob(os.path.join(logdir, "*"))
    if not runs:
        return None
    return max(runs, key=os.path.getmtime)

def load_scalars(run_dir):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 500})
    ea.Reload()
    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]
    return data

def render(data, run_dir):
    os.system("clear")
    run_name = os.path.basename(run_dir)
    print(color(f"╔══ SAC UR5e Pick-Place — {run_name} ══╗", "1;36"))
    print(color(f"  Logs: {run_dir}", "90"))
    print(color(f"  TensorBoard: http://localhost:6006", "90"))
    print()

    # Timestep global
    ep_rew = data.get("rollout/ep_rew_mean", [])
    total_steps = ep_rew[-1][0] if ep_rew else 0
    pct = total_steps / 1_000_000 * 100
    filled = int(pct / 100 * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    print(color(f"  Timesteps : {total_steps:>8,} / 1,000,000  [{bar}] {pct:.1f}%", "1;33"))
    print()

    print(color("  MÉTRIQUE                    VALEUR ACTUEL   SPARKLINE (historique)", "1;37"))
    print(color("  " + "─" * 72, "90"))

    for tag, label in TAGS:
        vals_raw = data.get(tag, [])
        if not vals_raw:
            print(f"  {label:<28} {'—':>12}   {'(pas encore de données)'}")
            continue
        vals = [v for _, v in vals_raw]
        cur = vals[-1]
        step = vals_raw[-1][0]
        spark = sparkline(vals, width=28)

        # Coloration selon tag
        if "rew" in tag:
            cur_str = color(f"{cur:+.2f}", "1;32" if cur > 0 else "1;31")
        elif "loss" in tag:
            cur_str = color(f"{cur:.4f}", "1;33")
        else:
            cur_str = f"{cur:.4f}"

        print(f"  {label:<28} {cur_str:>12}   {color(spark, '36')}")

    print()

    # Dernières lignes du log SB3 (verbose)
    log_path = "/tmp/train.log"
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        recent = [l.rstrip() for l in lines if l.strip() and not l.startswith("[INFO]") and not l.startswith("[WARN")][-8:]
        if recent:
            print(color("  LOG STDOUT (SB3):", "1;37"))
            for l in recent:
                print(color(f"    {l}", "90"))

    print()
    print(color(f"  Rafraîchissement toutes les {REFRESH}s  —  Ctrl+C pour quitter", "90"))


def main():
    print("Recherche du run TensorBoard...")
    while True:
        run_dir = find_latest_run(LOGDIR)
        if not run_dir:
            print(f"Aucun run dans {LOGDIR}, attente...")
            time.sleep(5)
            continue
        try:
            data = load_scalars(run_dir)
            render(data, run_dir)
        except Exception as e:
            print(f"Erreur lecture events: {e}")
        time.sleep(REFRESH)


if __name__ == "__main__":
    main()
