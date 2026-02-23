"""
slam_ai_explore.launch.py
==========================
Integrated launch: SLAM + AI-guided exploration.
Combines jetson_bot_slam (hardware + RTAB-Map) with robot_control (AI navigator).

Data flow
---------
USB camera → usb_camera_node ──► /image_raw  ──► RTAB-Map (SLAM)
                              ──► /camera/usb_raw ──► ai_navigator (Ollama VLM)

Arduino ──► arduino_bridge_node ──► /wheel_ticks ──► mecanum_odometry ──► /odom ──► RTAB-Map
                                ──► /ultrasonic_range ──► Nav2 costmap

ai_navigator (Ollama VLM) ──► /ai/direction ──► exploration_controller
exploration_controller    ──► /cmd_vel      ──► motor_driver_node
motor_driver_node         ──► /arduino_cmd  ──► arduino_bridge_node ──► Arduino

Usage
-----
    ros2 launch robot_control slam_ai_explore.launch.py
    ros2 launch robot_control slam_ai_explore.launch.py serial_port:=/dev/ttyUSB0 camera_device:=/dev/video1

Requirements
------------
- Ollama running at localhost:8080 with gemma3:4b pulled
  Start with:  docker compose --profile ollama up -d ollama
               docker compose run --rm ollama ollama pull gemma3:4b

- jetson_bot_slam package built alongside robot_control in the same workspace
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    slam_pkg = get_package_share_directory('jetson_bot_slam')

    # ── Launch arguments ──────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('serial_port',    default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('camera_device',  default_value='/dev/video1',
                              description='OpenCV camera device path'),
        DeclareLaunchArgument('ollama_host',    default_value='http://localhost:8080'),
        DeclareLaunchArgument('ollama_model',   default_value='gemma3:4b'),
        DeclareLaunchArgument('slam_goal',
            default_value='explore the bedroom and build a complete map',
            description='Natural-language goal passed to the Ollama VLM'),
        DeclareLaunchArgument('move_duration',  default_value='3.0',
                              description='Seconds per exploration step'),
        DeclareLaunchArgument('move_speed',     default_value='0.25',
                              description='Forward speed in m/s'),
        # Set use_slam:=true only if rtabmap_ros is installed on the image
        DeclareLaunchArgument('use_slam',       default_value='false',
                              description='Start RTAB-Map SLAM node (requires ros-iron-rtabmap-ros)'),
    ]

    # ── URDF / robot_state_publisher ──────────────────────────────────────────
    urdf_file = os.path.join(slam_pkg, 'urdf', 'jetson_bot.urdf.xacro')
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', urdf_file]),
                     'use_sim_time': False}],
    )

    # ── Hardware layer (from jetson_bot_slam) ─────────────────────────────────
    arduino_bridge = Node(
        package='jetson_bot_slam',
        executable='arduino_bridge',
        name='arduino_bridge',
        parameters=[{'serial_port': LaunchConfiguration('serial_port'),
                     'baud_rate': 115200}],
        output='screen',
    )

    mecanum_odom = Node(
        package='jetson_bot_slam',
        executable='mecanum_odometry',
        name='mecanum_odometry',
        parameters=[{
            'wheel_radius':     0.0508,
            'ticks_per_rev':    1440,
            'half_wheelbase':   0.1270,
            'half_track_width': 0.2172,
            'publish_tf':       True,
        }],
        output='screen',
    )

    motor_driver = Node(
        package='jetson_bot_slam',
        executable='motor_driver',
        name='motor_driver',
        parameters=[{
            'wheel_radius':     0.0508,
            'half_wheelbase':   0.1270,
            'half_track_width': 0.2172,
            'control_hz':       10.0,
            'cmd_vel_timeout':  0.6,   # slightly > exploration_controller's 0.5 s publish period
        }],
        output='screen',
    )

    # ── Camera (from robot_control) ───────────────────────────────────────────
    usb_camera = Node(
        package='robot_control',
        executable='usb_camera_node',
        name='usb_camera_node',
        parameters=[{
            'device':   LaunchConfiguration('camera_device'),
            'width':    320,
            'height':   240,
            'slam_fps': 5.0,    # 5 fps is plenty for RTAB-Map mono SLAM
        }],
        output='screen',
    )

    # ── RTAB-Map SLAM (optional — only starts if use_slam:=true) ─────────────
    rtabmap = Node(
        package='rtabmap_ros',
        executable='rtabmap',
        name='rtabmap',
        condition=IfCondition(LaunchConfiguration('use_slam')),
        output='screen',
        arguments=['--delete_db_on_start'],
        parameters=[{
            'subscribe_depth':          False,
            'subscribe_rgb':            True,
            'subscribe_odom':           True,
            'approx_sync':              True,
            'approx_sync_max_interval': 0.2,
            'Reg/Force3DoF':            'true',
            'Grid/3D':                  'false',
            'Grid/CellSize':            '0.05',
            'Grid/RayTracing':          'true',
            'Vis/FeatureType':          '6',       # ORB — fast on Jetson
            'Vis/MaxFeatures':          '500',
            'Vis/MinInliers':           '10',
            'RGBD/AngularUpdate':       '0.05',
            'RGBD/LinearUpdate':        '0.05',
            'Optimizer/Strategy':       '1',       # g2o
            'database_path':            '/root/maps/bedroom.db',
        }],
        remappings=[
            ('rgb/image',        '/image_raw'),
            ('rgb/camera_info',  '/camera_info'),
            ('odom',             '/odom'),
        ],
    )

    # ── AI navigator (Ollama VLM) ─────────────────────────────────────────────
    ai_navigator = Node(
        package='robot_control',
        executable='ai_navigator',
        name='ai_navigator',
        parameters=[{
            'ollama_host':         LaunchConfiguration('ollama_host'),
            'model':               LaunchConfiguration('ollama_model'),
            'goal':                LaunchConfiguration('slam_goal'),
            'stabilization_delay': 1.0,
        }],
        output='screen',
    )

    # ── Exploration controller ────────────────────────────────────────────────
    exploration_ctrl = Node(
        package='robot_control',
        executable='exploration_controller',
        name='exploration_controller',
        parameters=[{
            'move_speed':    LaunchConfiguration('move_speed'),
            'turn_speed':    0.5,
            'move_duration': LaunchConfiguration('move_duration'),
        }],
        output='screen',
    )

    return LaunchDescription(args + [
        robot_state_pub,
        arduino_bridge,
        mecanum_odom,
        motor_driver,
        usb_camera,
        rtabmap,
        ai_navigator,
        exploration_ctrl,
    ])
    