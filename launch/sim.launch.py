"""
Standalone launch: Gazebo + UR5e + parallel-jaw gripper + ros2_control.

Avoids ur_sim_control.launch.py which does not wrap robot_description
in ParameterValue(value_type=str), causing YAML parse failures with
larger URDF files (our custom xacro with the gripper).

Sequence:
  1. robot_state_publisher with gripper URDF
  2. Gazebo Harmonic with pick_place.sdf world
  3. gz_sim create -> spawns the robot (activates gz_ros2_control plugin)
  4. joint_state_broadcaster   (after spawn)
  5. forward_velocity_controller   (arm RL controller)
  6. gripper_velocity_controller   (finger controller)
  7. ros_gz_bridge (clock + cube pose tracking)

Usage:
    ros2 launch ur5e_rl_gazebo sim.launch.py
"""
import os

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg            = FindPackageShare('ur5e_rl_gazebo')
    ur_desc_pkg    = FindPackageShare('ur_description')
    ros_gz_sim_pkg = FindPackageShare('ros_gz_sim')

    xacro_file       = PathJoinSubstitution([pkg, 'urdf', 'ur5e_with_gripper.urdf.xacro'])
    controllers_yaml = PathJoinSubstitution([pkg, 'config', 'ur5e_controllers.yaml'])
    world_sdf        = PathJoinSubstitution([pkg, 'worlds', 'pick_place.sdf'])

    # ── Robot description (URDF via xacro) ────────────────────────────────────
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ', xacro_file,
        ' ur_type:=ur5e',
        ' simulation_controllers:=', controllers_yaml,
        ' safety_limits:=true',
    ])
    # ParameterValue(value_type=str) avoids YAML mis-parsing of the XML string
    robot_description = {
        'robot_description': ParameterValue(robot_description_content, value_type=str)
    }

    # ── robot_state_publisher ─────────────────────────────────────────────────
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': True}],
    )

    # ── Gazebo Harmonic ────────────────────────────────────────────────────────
    # GZ_HEADLESS=1 → serveur seul (-s), pas de client GUI : libère du CPU (débit
    # plus élevé pour l'entraînement) et supprime toute dépendance à X (donc plus
    # de crash Qt/xcb quand la stack est relancée depuis cron). L'env RL n'utilise
    # pas de caméra → aucun rendu nécessaire.
    _gz_flags = ' -s -r -v 2' if os.environ.get('GZ_HEADLESS') == '1' else ' -r -v 2'
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py'])
        ]),
        launch_arguments=[('gz_args', [world_sdf, _gz_flags])],
    )

    # ── Spawn UR5e + gripper into Gazebo ──────────────────────────────────────
    # 3 s timer lets Gazebo finish world loading before the spawn request.
    _spawn_node = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-file', '/tmp/ur5e_robot.urdf',
            '-name', 'ur',
            '-allow_renaming', 'true',
            '-x', '0', '-y', '0', '-z', '0',
        ],
    )
    spawn_robot = TimerAction(period=3.0, actions=[_spawn_node])

    # ── ros_gz_bridge: /clock ────────────────────────────────────────────────
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # ── Controllers — spawned after robot is in Gazebo ────────────────────────
    jsb_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '-c', '/controller_manager'],
        output='screen',
    )
    arm_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['forward_velocity_controller', '-c', '/controller_manager'],
        output='screen',
    )
    grip_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_velocity_controller', '-c', '/controller_manager'],
        output='screen',
    )

    # Sequence: _spawn_node exits -> wait 6s (gz_ros2_control plugin needs time) -> JSB -> arm -> gripper
    load_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=_spawn_node,
            on_exit=[TimerAction(period=6.0, actions=[jsb_spawner])],
        )
    )
    load_arm = RegisterEventHandler(
        OnProcessExit(
            target_action=jsb_spawner,
            on_exit=[arm_spawner],
        )
    )
    load_grip = RegisterEventHandler(
        OnProcessExit(
            target_action=arm_spawner,
            on_exit=[grip_spawner],
        )
    )

    # ── Pose bridge for cube tracking ─────────────────────────────────────────
    pose_bridge = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                name='pose_bridge',
                arguments=[
                    '/world/pick_place/dynamic_pose/info'
                    '@tf2_msgs/msg/TFMessage'
                    '[gz.msgs.Pose_V',
                ],
                remappings=[
                    ('/world/pick_place/dynamic_pose/info', '/world_dynamic_poses'),
                ],
                parameters=[{'use_sim_time': True}],
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        robot_state_pub,
        gz_sim,
        spawn_robot,
        clock_bridge,
        load_jsb,
        load_arm,
        load_grip,
        pose_bridge,
    ])
