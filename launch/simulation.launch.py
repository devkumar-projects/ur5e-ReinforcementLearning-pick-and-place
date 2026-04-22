"""
Standalone launch: Gazebo + UR5e + parallel-jaw gripper + ros2_control.
Peg-to-peg pick & place scene (pick_place.sdf).

Sequence:
  1. OpaqueFunction: xacro -> /tmp/ur5e_robot.urdf  (synchronous, before any node)
  2. robot_state_publisher
  3. Gazebo Harmonic
  4. gz_sim create (5 s timer) -> spawns robot at origin, TCP faces +X toward pegs
  5. joint_state_broadcaster  (after spawn + 6 s settle)
  6. forward_velocity_controller
  7. gripper_velocity_controller
  8. pose_bridge for disk tracking (14 s timer)

Usage:
    ros2 launch ur5e_rl_gazebo simulation.launch.py
"""
import subprocess

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def _write_urdf(context, *args, **kwargs):
    """Synchronously run xacro and write /tmp/ur5e_robot.urdf."""
    pkg_share      = get_package_share_directory('ur5e_rl_gazebo')
    xacro_file     = os.path.join(pkg_share, 'urdf', 'ur5e_with_gripper.urdf.xacro')
    controllers    = os.path.join(pkg_share, 'config', 'ur5e_controllers.yaml')

    result = subprocess.run(
        [
            'xacro', xacro_file,
            'ur_type:=ur5e',
            f'simulation_controllers:={controllers}',
            'safety_limits:=true',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    with open('/tmp/ur5e_robot.urdf', 'w') as f:
        f.write(result.stdout)
    return []


def generate_launch_description():
    pkg            = FindPackageShare('ur5e_rl_gazebo')
    ros_gz_sim_pkg = FindPackageShare('ros_gz_sim')

    xacro_file       = PathJoinSubstitution([pkg, 'urdf', 'ur5e_with_gripper.urdf.xacro'])
    controllers_yaml = PathJoinSubstitution([pkg, 'config', 'ur5e_controllers.yaml'])
    world_sdf        = PathJoinSubstitution([pkg, 'worlds', 'pick_place.sdf'])

    # ── Robot description for robot_state_publisher ─────────────────────────────
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ', xacro_file,
        ' ur_type:=ur5e',
        ' simulation_controllers:=', controllers_yaml,
        ' safety_limits:=true',
    ])
    robot_description = {
        'robot_description': ParameterValue(robot_description_content, value_type=str)
    }

    # ── Write URDF to /tmp synchronously before gz create fires ────────────────
    write_urdf = OpaqueFunction(function=_write_urdf)

    # ── robot_state_publisher ───────────────────────────────────────────────────
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': True}],
    )

    # ── Gazebo Harmonic ─────────────────────────────────────────────────────────
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py'])
        ]),
        launch_arguments=[('gz_args', [world_sdf, ' -r -v 2'])],
    )

    # ── Spawn UR5e + gripper into Gazebo ────────────────────────────────────────
    # 5 s lets Gazebo finish world loading; /tmp/ur5e_robot.urdf is ready by then.
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
    spawn_robot = TimerAction(period=5.0, actions=[_spawn_node])

    # ── ros_gz_bridge: /clock ───────────────────────────────────────────────────
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # ── Controllers ─────────────────────────────────────────────────────────────
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

    load_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=_spawn_node,
            on_exit=[TimerAction(period=6.0, actions=[jsb_spawner])],
        )
    )
    load_arm  = RegisterEventHandler(OnProcessExit(target_action=jsb_spawner, on_exit=[arm_spawner]))
    load_grip = RegisterEventHandler(OnProcessExit(target_action=arm_spawner,  on_exit=[grip_spawner]))

    # ── Pose bridge for disk tracking ───────────────────────────────────────────
    pose_bridge = TimerAction(
        period=14.0,
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
                remappings=[('/world/pick_place/dynamic_pose/info', '/world_dynamic_poses')],
                parameters=[{'use_sim_time': True}],
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        write_urdf,
        robot_state_pub,
        gz_sim,
        spawn_robot,
        clock_bridge,
        load_jsb,
        load_arm,
        load_grip,
        pose_bridge,
    ])
