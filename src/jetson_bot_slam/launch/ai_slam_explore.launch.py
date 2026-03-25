"""
ai_slam_explore.launch.py
==========================
Full autonomy stack: SLAM + AI-enhanced exploration using VILA 2.7B.

Data flow
---------
USB camera (usb_cam) ──► /image_raw  ──► RTAB-Map (SLAM)
                                      ──► vila_scene_labeller (VILA 2.7B)

Arduino ──► arduino_bridge ──► /wheel_ticks ──► mecanum_odometry ──► /odom ──► RTAB-Map
                            ──► /ultrasonic_range ──► exploration_controller

exploration_controller ──► /cmd_vel             ──► motor_driver ──► Arduino
                       ──► /robot/movement_complete ──► vila_scene_labeller
vila_scene_labeller    ──► /ai/direction        ──► exploration_controller
                       ──► /ai/semantic_label   (map annotation)

Usage (inside dustynv/nano_llm container)
-----------------------------------------
    ros2 launch jetson_bot_slam ai_slam_explore.launch.py
    ros2 launch jetson_bot_slam ai_slam_explore.launch.py rviz:=false
    ros2 launch jetson_bot_slam ai_slam_explore.launch.py use_slam:=false  # AI + exploration only

Requirements
------------
- Running inside dustynv/nano_llm:humble container with GPU passthrough
- VILA model: Efficient-Large-Model/VILA-2.7b (auto-downloaded on first run)
- jetson_bot_slam + robot_control packages built in same workspace
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    slam_pkg = get_package_share_directory('jetson_bot_slam')

    # ── Launch arguments ──────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('serial_port',        default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('camera_device',      default_value='/dev/video0'),
        DeclareLaunchArgument('rviz',               default_value='false',
                              description='Launch RViz2 (needs display)'),
        # VILA model overrides
        DeclareLaunchArgument('vila_model',
                              default_value='Efficient-Large-Model/VILA-2.7b'),
        DeclareLaunchArgument('vila_api',            default_value='awq'),
        DeclareLaunchArgument('vila_quantization',   default_value='q4f16_ft'),
        # Exploration (ultrasonic-driven)
        DeclareLaunchArgument('move_speed',          default_value='0.20'),
        DeclareLaunchArgument('turn_speed',          default_value='0.55'),
        DeclareLaunchArgument('obstacle_distance',   default_value='0.30'),
        DeclareLaunchArgument('emergency_stop_dist', default_value='0.08'),
        DeclareLaunchArgument('backup_clear_dist',   default_value='0.30'),
        DeclareLaunchArgument('backup_s',            default_value='2.0'),
        DeclareLaunchArgument('min_turn_s',          default_value='3.0'),
        DeclareLaunchArgument('max_turn_s',          default_value='10.0'),
        DeclareLaunchArgument('label_every',         default_value='5'),
        # Optional RTAB-Map
        DeclareLaunchArgument('use_slam',            default_value='false',
                              description='Start RTAB-Map SLAM (needs rgb camera)'),
    ]

    # ── URDF / robot_state_publisher ──────────────────────────────────────
    urdf_file = os.path.join(slam_pkg, 'urdf', 'jetson_bot.urdf.xacro')
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': ParameterValue(
                Command(['xacro ', urdf_file]), value_type=str),
            'use_sim_time': False,
        }],
    )

    # ── Hardware layer (jetson_bot_slam) ───────────────────────────────────
    arduino_bridge = Node(
        package='jetson_bot_slam',
        executable='arduino_bridge',
        name='arduino_bridge',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baud_rate':   115200,
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

    motor_driver = Node(
        package='jetson_bot_slam',
        executable='motor_driver',
        name='motor_driver',
        parameters=[{
            'wheel_radius':     0.0508,
            'half_wheelbase':   0.1270,
            'half_track_width': 0.2172,
            'control_hz':       10.0,
            'cmd_vel_timeout':  0.6,
        }],
        output='screen',
    )

    # ── USB Camera (usb_cam — shared between SLAM and VILA) ──────────────
    # Publishes full 720p /image_raw for RTAB-Map.
    # VILA node does its own resize to 384x384 internally.
    usb_camera = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        parameters=[{
            'video_device':    LaunchConfiguration('camera_device'),
            'image_width':     640,
            'image_height':    480,
            'framerate':       15.0,       # 15fps balances SLAM quality vs CPU
            'pixel_format':    'yuyv',     # camera hardware format (only YUYV supported)
            'auto_white_balance': True,
            'autoexposure':    True,
            'camera_frame_id': 'camera_optical_link',
            'io_method':       'mmap',
        }],
        remappings=[
            ('/usb_cam/image_raw',   '/image_raw'),
            ('/usb_cam/camera_info', '/camera_info'),
        ],
        output='screen',
    )

    # ── RTAB-Map SLAM (optional) ──────────────────────────────────────────
    rtabmap = Node(
        package='rtabmap_slam',
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
            ('rgb/image',       '/image_raw'),
            ('rgb/camera_info', '/camera_info'),
            ('odom',            '/odom'),
        ],
    )

    # ── VILA scene labeller (replaces Ollama ai_navigator) ────────────────
    vila_params_file = os.path.join(slam_pkg, 'config', 'vila_params.yaml')
    vila_labeller = Node(
        package='jetson_bot_slam',
        executable='vila_scene_labeller',
        name='vila_scene_labeller',
        parameters=[
            vila_params_file,
            {
                'model_name':   LaunchConfiguration('vila_model'),
                'api':          LaunchConfiguration('vila_api'),
                'quantization': LaunchConfiguration('vila_quantization'),
            },
        ],
        output='screen',
    )

    # ── Exploration controller (from robot_control, ultrasonic-reactive) ──
    exploration_ctrl = Node(
        package='robot_control',
        executable='exploration_controller',
        name='exploration_controller',
        parameters=[{
            'move_speed':          LaunchConfiguration('move_speed'),
            'turn_speed':          LaunchConfiguration('turn_speed'),
            'obstacle_distance':   LaunchConfiguration('obstacle_distance'),
            'emergency_stop_dist': LaunchConfiguration('emergency_stop_dist'),
            'backup_clear_dist':   LaunchConfiguration('backup_clear_dist'),
            'backup_s':            LaunchConfiguration('backup_s'),
            'min_turn_s':          LaunchConfiguration('min_turn_s'),
            'max_turn_s':          LaunchConfiguration('max_turn_s'),
            'label_every':         LaunchConfiguration('label_every'),
        }],
        output='screen',
    )

    # ── RViz2 (optional, needs display) ───────────────────────────────────
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
        arduino_bridge,
        mecanum_odom,
        motor_driver,
        usb_camera,
        rtabmap,
        vila_labeller,
        exploration_ctrl,
        rviz,
    ])
