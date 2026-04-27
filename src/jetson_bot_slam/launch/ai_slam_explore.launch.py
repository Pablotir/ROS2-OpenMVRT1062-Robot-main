"""
ai_slam_explore.launch.py
==========================
Minimal autonomous mapping stack:

  STL-27L LiDAR  ──► /scan ──► slam_toolbox (builds /map)
  RoboClaw L+R   ──► roboclaw_node ──► /wheel_ticks ──► mecanum_odometry ──► /odom
  exploration_controller reads /scan and writes /cmd_vel (linear.x + angular.z only)
  roboclaw_node reads /cmd_vel and drives the wheels (HW PID velocity control)

No Nav2. No costmaps. No path planner. No frontier logic.
The robot drives forward, avoids obstacles via LiDAR, turns toward
open space, and slam_toolbox builds the map in the background.

Optional:
  rviz:=true           — launch RViz2 (needs $DISPLAY)
  use_camera:=true     — enable USB camera + VILA AI labeller
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

    # ── Launch arguments ──────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('left_port',          default_value='/dev/roboclaw_left'),
        DeclareLaunchArgument('right_port',         default_value='/dev/roboclaw_right'),
        DeclareLaunchArgument('lidar_port',         default_value='/dev/lidar'),
        DeclareLaunchArgument('camera_device',      default_value='/dev/video0'),
        DeclareLaunchArgument('rviz',               default_value='false'),
        # Motion tuning
        DeclareLaunchArgument('move_speed',          default_value='0.18'),
        DeclareLaunchArgument('turn_speed',          default_value='0.50'),
        DeclareLaunchArgument('obstacle_distance',   default_value='0.50'),
        DeclareLaunchArgument('emergency_stop_dist', default_value='0.20'),
        # Camera / AI (off by default)
        DeclareLaunchArgument('use_camera',          default_value='false'),
        DeclareLaunchArgument('vila_model',
                              default_value='Efficient-Large-Model/VILA-2.7b'),
        DeclareLaunchArgument('vila_api',            default_value='mlc'),
        DeclareLaunchArgument('vila_quantization',   default_value='q4f16_ft'),
        DeclareLaunchArgument('room_hints_enabled',  default_value='true'),
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

    # ── RoboClaw motor controller ─────────────────────────────────────────
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

    # ── Mecanum odometry ──────────────────────────────────────────────────
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

    # ── STL-27L LiDAR ─────────────────────────────────────────────────────
    lidar = Node(
        package='ldlidar_stl_ros2',
        executable='ldlidar_stl_ros2_node',
        name='ldlidar',
        output='screen',
        parameters=[{
            'product_name':    'LDLiDAR_STL27L',
            'topic_name':      '/scan',
            'frame_id':        'laser_frame',
            'port_name':       LaunchConfiguration('lidar_port'),
            'port_baudrate':   921600,
            'laser_scan_dir':  True,
            'enable_angle_crop_func': False,
        }],
    )

    # ── SLAM Toolbox ──────────────────────────────────────────────────────
    slam_toolbox_pkg  = get_package_share_directory('slam_toolbox')
    slam_params_file  = os.path.join(slam_pkg, 'config', 'mapper_params_online_async.yaml')
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_toolbox_pkg, 'launch', 'online_async_launch.py')),
        launch_arguments={
            'use_sim_time':    'false',
            'slam_params_file': slam_params_file,
            # Suppress noisy 'LaserRangeScan contains X readings, expected Y' lines
            'log_level':       'warn',
        }.items(),
    )

    # ── Exploration controller ─────────────────────────────────────────────
    # The ONLY node that publishes /cmd_vel.
    # Smooth reactive: steers toward open space, scales speed by clearance.
    exploration_ctrl = Node(
        package='robot_control',
        executable='exploration_controller',
        name='exploration_controller',
        parameters=[{
            'move_speed':          LaunchConfiguration('move_speed'),
            'turn_speed':          LaunchConfiguration('turn_speed'),
            'obstacle_distance':   LaunchConfiguration('obstacle_distance'),
            'emergency_stop_dist': LaunchConfiguration('emergency_stop_dist'),
        }],
        output='screen',
    )

    # ── USB Camera + VILA AI (opt-in) ─────────────────────────────────────
    usb_camera = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        parameters=[{
            'video_device':    LaunchConfiguration('camera_device'),
            'image_width':     640,
            'image_height':    480,
            'framerate':       15.0,
            'pixel_format':    'yuyv',
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
        condition=IfCondition(LaunchConfiguration('use_camera')),
    )

    image_convert = Node(
        package='image_proc',
        executable='rectify_node',
        name='image_convert',
        remappings=[
            ('image_raw',   '/image_raw'),
            ('camera_info', '/camera_info'),
            ('image',       '/image_rect_color'),
        ],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_camera')),
    )

    vila_params_file = os.path.join(slam_pkg, 'config', 'vila_params.yaml')
    vila_labeller = Node(
        package='jetson_bot_slam',
        executable='vila_scene_labeller',
        name='vila_scene_labeller',
        condition=IfCondition(LaunchConfiguration('use_camera')),
        parameters=[
            vila_params_file,
            {
                'model_name':         LaunchConfiguration('vila_model'),
                'api':                LaunchConfiguration('vila_api'),
                'quantization':       LaunchConfiguration('vila_quantization'),
                'room_hints_enabled': LaunchConfiguration('room_hints_enabled'),
            },
        ],
        output='screen',
    )

    # ── RViz2 (optional) ──────────────────────────────────────────────────
    rviz_cfg = os.path.join(slam_pkg, 'rviz', 'slam_view.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription(args + [
        robot_state_pub,    # TF / URDF
        roboclaw,           # Dual RoboClaw motor control + encoder reads
        mecanum_odom,       # /odom from encoders
        lidar,              # /scan from STL-27L
        slam_toolbox,       # /scan + /odom → /map
        exploration_ctrl,   # /scan → /cmd_vel (only motion publisher)
        usb_camera,         # opt-in: use_camera:=true
        image_convert,      # opt-in
        vila_labeller,      # opt-in
        rviz,               # opt-in: rviz:=true
    ])
