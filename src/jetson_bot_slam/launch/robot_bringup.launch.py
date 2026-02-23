"""
robot_bringup.launch.py
========================
Starts the core robot stack:
  - robot_state_publisher  (URDF / TF)
  - arduino_bridge         (serial ↔ ROS2)
  - mecanum_odometry       (wheel odometry → /odom + TF)
  - motor_driver           (cmd_vel → Arduino B-commands)
  - usb_cam                (USB camera → /image_raw + /camera_info)

Usage:
    ros2 launch jetson_bot_slam robot_bringup.launch.py
    ros2 launch jetson_bot_slam robot_bringup.launch.py serial_port:=/dev/ttyUSB0 camera_index:=1
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('jetson_bot_slam')

    # ── Launch arguments ──────────────────────────────────────────────────────
    serial_port_arg = DeclareLaunchArgument(
        'serial_port', default_value='/dev/ttyUSB0',
        description='Arduino serial port')

    camera_index_arg = DeclareLaunchArgument(
        'camera_index', default_value='1',
        description='OpenCV camera device index (1 = /dev/video1 on Jetson USB port 1)')

    camera_width_arg = DeclareLaunchArgument(
        'camera_width',  default_value='640')
    camera_height_arg = DeclareLaunchArgument(
        'camera_height', default_value='480')
    camera_fps_arg = DeclareLaunchArgument(
        'camera_fps',    default_value='30')

    # ── URDF via xacro ───────────────────────────────────────────────────────
    urdf_file = os.path.join(pkg, 'urdf', 'jetson_bot.urdf.xacro')
    robot_description = Command(['xacro ', urdf_file])

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_description,
                     'use_sim_time': False}],
    )

    # ── Arduino bridge ───────────────────────────────────────────────────────
    arduino_bridge = Node(
        package='jetson_bot_slam',
        executable='arduino_bridge',
        name='arduino_bridge',
        parameters=[{'serial_port': LaunchConfiguration('serial_port'),
                     'baud_rate': 115200}],
        output='screen',
    )

    # ── Mecanum odometry ─────────────────────────────────────────────────────
    mecanum_odom = Node(
        package='jetson_bot_slam',
        executable='mecanum_odometry',
        name='mecanum_odometry',
        parameters=[{
            'wheel_radius':      0.0508,
            'ticks_per_rev':     1440,
            'half_wheelbase':    0.1270,   # Lx – tune if odometry drifts
            'half_track_width':  0.2172,   # Ly – tune if odometry drifts
            'publish_tf':        True,
        }],
        output='screen',
    )

    # ── Motor driver ─────────────────────────────────────────────────────────
    motor_driver = Node(
        package='jetson_bot_slam',
        executable='motor_driver',
        name='motor_driver',
        parameters=[{
            'wheel_radius':      0.0508,
            'half_wheelbase':    0.1270,
            'half_track_width':  0.2172,
            'control_hz':        10.0,
            'cmd_vel_timeout':   0.5,
        }],
        output='screen',
    )

    # ── USB Camera (usb_cam node, ROS2 Humble+) ───────────────────────────────
    #
    # Install:  sudo apt install ros-humble-usb-cam
    #
    # The camera_info_url should point to a calibration file.
    # Run  ros2 run camera_calibration cameracalibrator  to generate one.
    # Until calibration is done, RTAB-Map will use the raw image but loop
    # closure quality will be reduced.
    usb_camera = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        parameters=[{
            'video_device':   '/dev/video' + LaunchConfiguration('camera_index'),
            'image_width':    LaunchConfiguration('camera_width'),
            'image_height':   LaunchConfiguration('camera_height'),
            'framerate':      LaunchConfiguration('camera_fps'),
            'pixel_format':   'yuyv',
            'camera_frame_id':'camera_optical_link',
            'io_method':      'mmap',
        }],
        remappings=[
            ('/usb_cam/image_raw',   '/image_raw'),
            ('/usb_cam/camera_info', '/camera_info'),
        ],
        output='screen',
    )

    return LaunchDescription([
        serial_port_arg,
        camera_index_arg,
        camera_width_arg,
        camera_height_arg,
        camera_fps_arg,
        robot_state_pub,
        arduino_bridge,
        mecanum_odom,
        motor_driver,
        usb_camera,
    ])
