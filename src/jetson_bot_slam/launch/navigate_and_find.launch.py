#!/usr/bin/env python3
"""
navigate_and_find.launch.py — Unified Open-Vocabulary Mobile Manipulation (OVMM) Launch
=======================================================================================
Launches the complete 5-layer autonomous execution framework:
  - Mobile Base & Odometry (Dual RoboClaw + Mecanum Odometry)
  - STL-27L LiDAR (/dev/ttyTHS1 UART) + Dynamic Arm LiDAR Masking (/scan_filtered)
  - SLAM Toolbox Localization / Lifelong Mode (/root/maps/latest)
  - Nav2 Navigation Stack (DWB Local Planner + Global Costmaps)
  - Layer 1: Topological Partitioning & SQLite Semantic Scene Graph
  - Layer 2: Semantic Waypoint Server + Visual Preemption Supervisor
  - Layer 3: Kinematic Docking Controller
  - Layer 4: SO-ARM101 Driver + LeRobot Skill Execution Engine
  - RealSense D405 Wrist Camera + ChArUco Hand-Eye TF Broadcaster
  - Dell Front Camera (On-Demand Snapshot Tagging)

Usage:
  ros2 launch jetson_bot_slam navigate_and_find.launch.py

Triggering a Task:
  ros2 topic pub /ovmm/task_prompt std_msgs/String "data: 'Grab the blue bottle in the kitchen and throw it in the bin'" --once
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    slam_pkg = get_package_share_directory('jetson_bot_slam')
    arm_pkg  = get_package_share_directory('arm_manipulation')
    nav2_pkg = get_package_share_directory('nav2_bringup')
    slam_toolbox_pkg = get_package_share_directory('slam_toolbox')

    # ── Launch arguments ──────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('left_port',          default_value='/dev/roboclaw_left'),
        DeclareLaunchArgument('right_port',         default_value='/dev/roboclaw_right'),
        DeclareLaunchArgument('lidar_port',         default_value='/dev/ttyTHS1'),
        DeclareLaunchArgument('arm_port',           default_value='/dev/arm_controller'),
        DeclareLaunchArgument('front_camera',       default_value='/dev/video0'),
        DeclareLaunchArgument('maps_dir',           default_value='/root/maps'),
        DeclareLaunchArgument('rviz',               default_value='false'),
        DeclareLaunchArgument('map_mode',           default_value='localization',
                              description='SLAM mode: localization (known map) or lifelong (expanding)'),
    ]

    # ── 1. Robot State Publisher & URDF Model ──────────────────────────────
    urdf_file = os.path.join(slam_pkg, 'urdf', 'jetson_bot.urdf.xacro')
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': ParameterValue(Command(['xacro ', urdf_file]), value_type=str),
            'use_sim_time': False,
        }],
    )

    # ── 2. Dual RoboClaw Controller + Mecanum Odometry ────────────────────
    roboclaw = Node(
        package='jetson_bot_slam',
        executable='roboclaw_node',
        name='roboclaw_node',
        parameters=[{
            'left_port':         LaunchConfiguration('left_port'),
            'right_port':        LaunchConfiguration('right_port'),
            'address':           0x80,
            'baudrate':          115200,
            'wheel_radius':      0.0508,
            'half_wheelbase':    0.1270,
            'half_track_width':  0.2172,
            'ticks_per_rev':     1440,
            'max_qpps':          2300,
            'control_hz':        20.0,
            'cmd_vel_timeout':   0.6,
        }],
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

    # ── 3. STL-27L LiDAR + Arm Masking ────────────────────────────────────
    lidar = Node(
        package='ldlidar_stl_ros2',
        executable='ldlidar_stl_ros2_node',
        name='ldlidar',
        output='screen',
        parameters=[{
            'product_name':          'LDLiDAR_STL27L',
            'topic_name':            '/scan',
            'frame_id':              'laser_frame',
            'port_name':             LaunchConfiguration('lidar_port'),
            'port_baudrate':         921600,
            'laser_scan_dir':        True,
            'enable_angle_crop_func': False,
        }],
    )

    arm_mask_params = os.path.join(slam_pkg, 'config', 'arm_mask_params.yaml')
    arm_lidar_mask = Node(
        package='robot_control',
        executable='arm_lidar_mask_node',
        name='arm_lidar_mask',
        parameters=[arm_mask_params],
        output='screen',
    )

    # ── 4. SLAM Toolbox (Localization / Lifelong Mode) ─────────────────────
    slam_params = os.path.join(slam_pkg, 'config', 'mapper_params_localization.yaml')
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_toolbox_pkg, 'launch', 'online_async_launch.py')),
        launch_arguments={
            'use_sim_time':     'false',
            'slam_params_file': slam_params,
            'log_level':        'warn',
        }.items(),
    )

    # ── 5. Nav2 Navigation Stack ───────────────────────────────────────────
    nav2_params = os.path.join(slam_pkg, 'config', 'nav2_params.yaml')
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_pkg, 'launch', 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time': 'false',
            'params_file':  nav2_params,
            'autostart':    'true',
        }.items(),
    )

    # ── 6. Layer 1: Topological Scene Graph & Waypoint Server ──────────────
    topological_scene_graph = Node(
        package='robot_control',
        executable='topological_scene_graph_node',
        name='topological_scene_graph_node',
        parameters=[{
            'maps_dir': LaunchConfiguration('maps_dir'),
            'camera_topic': '/camera/usb_raw',
        }],
        output='screen',
    )

    semantic_waypoint_server = Node(
        package='robot_control',
        executable='semantic_waypoint_server',
        name='semantic_waypoint_server',
        parameters=[{
            'maps_dir': LaunchConfiguration('maps_dir'),
        }],
        output='screen',
    )

    # ── 7. Arm Hardware Driver & ChArUco Hand-Eye TF ───────────────────────
    arm_driver = Node(
        package='arm_manipulation',
        executable='arm_driver_node',
        name='arm_driver_node',
        parameters=[{
            'port': LaunchConfiguration('arm_port'),
            'baudrate': 1000000,
        }],
        output='screen',
    )

    hand_eye_calib = '/root/ros2_ws/calibration/hand_eye_calibration.yaml'
    hand_eye_tf = Node(
        package='arm_manipulation',
        executable='hand_eye_tf_broadcaster',
        name='hand_eye_tf_broadcaster',
        parameters=[{
            'calibration_file': hand_eye_calib,
            'parent_frame': 'wrist_flex_link',
            'child_frame': 'd405_color_optical_frame',
        }],
        output='screen',
    )

    # ── 8. RealSense D405 Wrist Camera & Detection Node ────────────────────
    d405_camera = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='d405_camera',
        parameters=[{
            'camera_name': 'd405',
            'enable_color': True,
            'enable_depth': True,
            'rgb_camera.color_profile': '848x480x30',
            'depth_module.depth_profile': '848x480x30',
            'align_depth.enable': False,
            'pointcloud.enable': False,
        }],
        output='screen',
    )

    detection = Node(
        package='arm_manipulation',
        executable='detection_node',
        name='detection_node',
        parameters=[{
            'model_path': '/root/ros2_ws/models/yolo11n-seg.engine',
            'target_class_id': 39,
            'target_label': 'bottle',
            'confidence_threshold': 0.50,
        }],
        output='screen',
    )

    # ── 9. Layer 2-5: OVMM Supervisor Node ────────────────────────────────
    ovmm_supervisor = Node(
        package='robot_control',
        executable='ovmm_supervisor_node',
        name='ovmm_supervisor_node',
        parameters=[{
            'maps_dir': LaunchConfiguration('maps_dir'),
            'preemption_confidence_thresh': 0.75,
            'docking_standoff_m': 0.40,
        }],
        output='screen',
    )

    # ── 10. RViz2 (optional) ──────────────────────────────────────────────
    rviz_cfg = os.path.join(slam_pkg, 'rviz', 'slam_view.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription(args + [
        robot_state_pub,
        roboclaw,
        mecanum_odom,
        lidar,
        arm_lidar_mask,
        slam_toolbox,
        nav2_bringup,
        topological_scene_graph,
        semantic_waypoint_server,
        arm_driver,
        hand_eye_tf,
        d405_camera,
        detection,
        ovmm_supervisor,
        rviz,
    ])
