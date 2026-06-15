#!/bin/bash
# Garde SAC+HER UR5e : relance la stack (Gazebo + entraînement) si l'entraînement
# est tombé. Premier lancement = from scratch (checkpoints/ vide après archivage) ;
# relances après crash = reprise depuis le checkpoint le plus récent (RESUME_FROM).
# Installé en cron (toutes les minutes). flock garantit une seule instance.

WS=/home/dev-kumar/ros2_ws
LOG=$WS/guard.log
LOCK=$WS/guard.lock

exec 9>"$LOCK"; flock -n 9 || exit 0
ts(){ date '+%F %T'; }

# Entraînement déjà vivant ? rien à faire.
# Match l'exécutable réel (pas une ligne de commande qui contient juste ce texte,
# ex. un moniteur) pour éviter les faux positifs qui bloqueraient la relance.
if pgrep -ax 'python3' | xargs -I{} sh -c 'cat /proc/{}/cmdline 2>/dev/null | tr "\0" " "' 2>/dev/null | grep -q 'lib/ur5e_rl_gazebo/train'; then
    exit 0
fi

echo "[$(ts)] entraînement absent — redémarrage de la stack" >>"$LOG"

# Nettoyage des restes de simulation
pkill -9 -f 'gz sim'              2>/dev/null
pkill -9 -f 'ruby.*gz'           2>/dev/null
pkill -9 -f 'parameter_bridge'   2>/dev/null
pkill -9 -f 'robot_state_publisher' 2>/dev/null
pkill -9 -f 'sim.launch.py'      2>/dev/null
pkill -9 -f 'spawner'            2>/dev/null
sleep 6

source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
export DISPLAY=:0
export ROS_DOMAIN_ID=0
# Headless : serveur gz seul, pas de GUI → plus de débit pour l'entraînement et
# aucune dépendance X (donc plus de crash Qt/xcb depuis cron). Mettre à 0 pour
# revoir la visu 3D.
export GZ_HEADLESS=1
# X authority pour atteindre :0 depuis cron. Sans ça, le plugin Qt xcb de gz
# échoue ("Invalid MIT-MAGIC-COOKIE-1 / could not connect to display :0") et le
# serveur Gazebo crashe au démarrage → simu morte, train figé. (cause des morts du 16/06)
export XAUTHORITY="$(ls -t /run/user/1000/.mutter-Xwaylandauth.* 2>/dev/null | head -1)"
[ -z "$XAUTHORITY" ] && export XAUTHORITY="$HOME/.Xauthority"
cd "$WS"

# Génère l'URDF que le launch spawne depuis /tmp (sinon "Error finding file
# /tmp/ur5e_robot.urdf" → robot non spawné → pas de controller_manager → train figé)
SHARE="$WS/install/ur5e_rl_gazebo/share/ur5e_rl_gazebo"
xacro "$SHARE/urdf/ur5e_with_gripper.urdf.xacro" \
      ur_type:=ur5e \
      simulation_controllers:="$SHARE/config/ur5e_controllers.yaml" \
      safety_limits:=true > /tmp/ur5e_robot.urdf 2>>"$WS/sim.log"

# Lancement de Gazebo + contrôleurs
nohup ros2 launch ur5e_rl_gazebo sim.launch.py >>"$WS/sim.log" 2>&1 &
sleep 35

# Gazebo réellement vivant ? On exige que /clock publie (donc le serveur gz tourne
# et la simu avance). Sinon, on abandonne ce cycle : le cron retentera dans 1 min
# plutôt que de lancer un train qui se figerait sur une simu morte.
if ! timeout 10 ros2 topic echo /clock --once >/dev/null 2>&1; then
    echo "[$(ts)] Gazebo pas prêt (/clock muet) — abandon, retry au prochain cycle" >>"$LOG"
    pkill -9 -f 'sim.launch.py' 2>/dev/null; pkill -9 -f 'gz sim' 2>/dev/null
    exit 0
fi

# Reprise depuis le checkpoint avec le plus grand nombre de steps (pas le plus récent
# par date — un 5k récent ne doit pas l'emporter sur un 45k plus ancien).
LATEST=$(ls "$WS"/checkpoints/*_steps.zip 2>/dev/null \
  | awk -F'_' '{n=$(NF-1)+0; print n, $0}' \
  | sort -n | tail -1 | awk '{print $2}')
export RESUME_FROM="$LATEST"
nohup ros2 run ur5e_rl_gazebo train >>"$WS/train.log" 2>&1 &

echo "[$(ts)] stack relancée (resume=${LATEST:-scratch})" >>"$LOG"
exit 0
