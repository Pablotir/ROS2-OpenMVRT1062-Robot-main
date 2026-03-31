"""
ai_slam_explore.launch.py
==========================
Full autonomy stack: LiDAR SLAM + Nav2 path planning + frontier exploration
+ AI semantic labelling using VILA 2.7B.

Data flow (Nav2 mode — default)
--------------------------------
STL-27L LiDAR ──► /scan  ──► RTAB-Map (SLAM → /map)
                           ──► Nav2 costmaps (obstacle avoidance)

RTAB-Map /map ──► Nav2 global_costmap ──► explore_lite (pick frontier goal)
                                       ──► Nav2 planner (path to frontier)
                                       ──► Nav2 DWB controller ──► /cmd_vel ──► motor_driver

USB camera ──► /image_raw  ──► image_proc (yuv→rgb)
                            ──► vila_scene_labeller (VILA 2.7B scene label)

Arduino ──► arduino_bridge ──► /wheel_ticks ──► mecanum_odometry ──► /odom ──► RTAB-Map
                            ──► /ultrasonic_range ──► Nav2 costmaps (rear backup)

vila_scene_labeller    ──► /ai/semantic_label    (map annotation)
                       ──► /ai/room              (inferred room)

Usage (inside dustynv/nano_llm container)
-----------------------------------------
    ros2 launch jetson_bot_slam ai_slam_explore.launch.py
    ros2 launch jetson_bot_slam ai_slam_explore.launch.py use_nav2:=false    # reactive fallback
    ros2 launch jetson_bot_slam ai_slam_explore.launch.py use_slam:=false
    ros2 launch jetson_bot_slam ai_slam_explore.launch.py room_hints_enabled:=false

Stop/resume frontier exploration at runtime:
    ros2 topic pub /explore/resume std_msgs/Bool "data: false" --once
    ros2 topic pub /explore/resume std_msgs/Bool "data: true"  --once

Requirements
------------
- Running inside dustynv/nano_llm:humble container with GPU passthrough
- VILA model: Efficient-Large-Model/VILA-2.7b (auto-downloaded on first run)
- STL-27L LiDAR on /dev/ttyUSB1
- Arduino Mega on /dev/ttyUSB0
- Nav2 + explore_lite installed (handled by setup_vila.sh)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    slam_pkg = get_package_share_directory('jetson_bot_slam')

    # ── Launch arguments ──────────────────────────────────────────────────
    args = [
        # Hardware ports
        DeclareLaunchArgument('serial_port',        default_value='/dev/ttyUSB0',
                              description='Arduino Mega serial port'),
        DeclareLaunchArgument('lidar_port',         default_value='/dev/ttyUSB1',
                              description='STL-27L LiDAR serial port'),
        DeclareLaunchArgument('camera_device',      default_value='/dev/video0'),
        DeclareLaunchArgument('rviz',               default_value='false',
                              description='Launch RViz2 (needs display)'),
        # VILA model overrides
        DeclareLaunchArgument('vila_model',
                              default_value='Efficient-Large-Model/VILA-2.7b'),
        DeclareLaunchArgument('vila_api',            default_value='awq'),
        DeclareLaunchArgument('vila_quantization',   default_value='q4f16_ft'),
        # Exploration (LiDAR-driven)
        DeclareLaunchArgument('move_speed',          default_value='0.20'),
        DeclareLaunchArgument('turn_speed',          default_value='0.55'),
        DeclareLaunchArgument('obstacle_distance',   default_value='0.30'),
        DeclareLaunchArgument('emergency_stop_dist', default_value='0.08'),
        DeclareLaunchArgument('rear_safety_dist',    default_value='0.15'),
        DeclareLaunchArgument('backup_s',            default_value='2.0'),
        DeclareLaunchArgument('min_turn_s',          default_value='3.0'),
        DeclareLaunchArgument('max_turn_s',          default_value='10.0'),
        DeclareLaunchArgument('label_every',         default_value='5'),
        # SLAM
        DeclareLaunchArgument('use_slam',            default_value='true',
                              description='Start RTAB-Map LiDAR SLAM'),
        # Nav2 + Frontier exploration
        DeclareLaunchArgument('use_nav2',            default_value='true',
                              description='Use Nav2 + explore_lite instead of reactive controller'),
        # Room identification
        DeclareLaunchArgument('room_hints_enabled',  default_value='true',
                              description='Enable AI room identification from labels'),
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

    # ── STL-27L LiDAR ────────────────────────────────────────────────────
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
            'laser_scan_dir':  True,   # CCW positive (ROS convention)
            'enable_angle_crop_func': False,
        }],
    )

    # ── USB Camera ────────────────────────────────────────────────────────
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
    )

    # ── Image format converter (yuv422 → rgb8) ───────────────────────────
    image_convert = Node(
        package='image_proc',
        executable='convert_node',
        name='image_convert',
        remappings=[
            ('image_raw',        '/image_raw'),
            ('camera_info',      '/camera_info'),
            ('image',            '/image_rect_color'),
        ],
        output='screen',
    )

    # ── RTAB-Map SLAM (LiDAR mode) ───────────────────────────────────────
    rtabmap = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        condition=IfCondition(LaunchConfiguration('use_slam')),
        output='screen',
        parameters=[{
            'subscribe_depth':          False,
            'subscribe_rgb':            False,
            'subscribe_scan':           True,
            'subscribe_odom':           True,
            'approx_sync':              True,
            'approx_sync_max_interval': 0.3,
            'Reg/Strategy':             '1',       # ICP for LiDAR
            'Reg/Force3DoF':            'true',
            'RGBD/NeighborLinkRefining': 'true',
            'Icp/PointToPlane':         'false',
            'Icp/VoxelSize':            '0.05',
            'Icp/MaxCorrespondenceDistance': '0.1',
            'Grid/3D':                  'false',
            'Grid/CellSize':            '0.05',
            'Grid/RayTracing':          'true',
            'Grid/RangeMax':            '12.0',
            'RGBD/AngularUpdate':       '0.05',
            'RGBD/LinearUpdate':        '0.05',
            'Optimizer/Strategy':       '1',       # g2o
            'database_path':            '/root/maps/bedroom.db',
        }],
        remappings=[
            ('scan',  '/scan'),
            ('odom',  '/odom'),
        ],
    )

    # ── VILA scene labeller ───────────────────────────────────────────────
    vila_params_file = os.path.join(slam_pkg, 'config', 'vila_params.yaml')
    vila_labeller = Node(
        package='jetson_bot_slam',
        executable='vila_scene_labeller',
        name='vila_scene_labeller',
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

    # ── Nav2 path planning (replaces reactive controller) ─────────────────
    nav2_pkg = get_package_share_directory('nav2_bringup')
    nav2_params = os.path.join(slam_pkg, 'config', 'nav2_params.yaml')

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_pkg, 'launch', 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time':  'false',
            'params_file':    nav2_params,
            'autostart':     'true',
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_nav2')),
    )

    # ── Frontier explorer (Python-native, no C++ build needed) ────────────
    frontier_explorer = Node(
        package='robot_control',
        executable='frontier_explorer',
        name='frontier_explorer',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_nav2')),
        parameters=[{
            'robot_base_frame':    'base_link',
            'planner_frequency':   0.33,
            'progress_timeout':    30.0,
            'potential_scale':     3.0,
            'gain_scale':          1.0,
            'min_frontier_size':   0.75,
        }],
    )

    # ── Reactive exploration controller (FALLBACK — only when Nav2 off) ───
    exploration_ctrl = Node(
        package='robot_control',
        executable='exploration_controller',
        name='exploration_controller',
        condition=UnlessCondition(LaunchConfiguration('use_nav2')),
        parameters=[{
            'move_speed':          LaunchConfiguration('move_speed'),
            'turn_speed':          LaunchConfiguration('turn_speed'),
            'obstacle_distance':   LaunchConfiguration('obstacle_distance'),
            'emergency_stop_dist': LaunchConfiguration('emergency_stop_dist'),
            'rear_safety_dist':    LaunchConfiguration('rear_safety_dist'),
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
        lidar,            # STL-27L → /scan
        usb_camera,
        image_convert,    # yuv422 → rgb8
        rtabmap,          # LiDAR SLAM (on by default)
        vila_labeller,
        nav2,             # Nav2 path planning (on by default)
        frontier_explorer,  # Python frontier exploration (on by default)
        exploration_ctrl,   # Reactive fallback (only when use_nav2:=false)
        rviz,
    ])
