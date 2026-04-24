"""\r
robot_bringup.launch.py\r
========================\r
Starts the core robot stack:\r
  - robot_state_publisher  (URDF / TF)\r
  - roboclaw_node          (dual RoboClaw motor control + encoder reads)\r
  - mecanum_odometry       (wheel odometry → /odom + TF)\r
  - usb_cam                (USB camera → /image_raw + /camera_info)\r
\r
Usage:\r
    ros2 launch jetson_bot_slam robot_bringup.launch.py\r
    ros2 launch jetson_bot_slam robot_bringup.launch.py left_port:=/dev/ttyACM0 right_port:=/dev/ttyACM1\r
"""\r
\r
import os\r
from ament_index_python.packages import get_package_share_directory\r
from launch import LaunchDescription\r
from launch.actions import DeclareLaunchArgument\r
from launch.substitutions import LaunchConfiguration, Command\r
from launch_ros.actions import Node\r
\r
\r
def generate_launch_description():\r
    pkg = get_package_share_directory('jetson_bot_slam')\r
\r
    # ── Launch arguments ──────────────────────────────────────────────────────\r
    left_port_arg = DeclareLaunchArgument(\r
        'left_port', default_value='/dev/roboclaw_left',\r
        description='Left RoboClaw serial port (M1=RL, M2=FL)')\r
\r
    right_port_arg = DeclareLaunchArgument(\r
        'right_port', default_value='/dev/roboclaw_right',\r
        description='Right RoboClaw serial port (M1=RR, M2=FR)')\r
\r
    camera_index_arg = DeclareLaunchArgument(\r
        'camera_index', default_value='1',\r
        description='OpenCV camera device index (1 = /dev/video1 on Jetson USB port 1)')\r
\r
    camera_width_arg = DeclareLaunchArgument(\r
        'camera_width',  default_value='640')\r
    camera_height_arg = DeclareLaunchArgument(\r
        'camera_height', default_value='480')\r
    camera_fps_arg = DeclareLaunchArgument(\r
        'camera_fps',    default_value='30')\r
\r
    # ── URDF via xacro ───────────────────────────────────────────────────────\r
    urdf_file = os.path.join(pkg, 'urdf', 'jetson_bot.urdf.xacro')\r
    robot_description = Command(['xacro ', urdf_file])\r
\r
    robot_state_pub = Node(\r
        package='robot_state_publisher',\r
        executable='robot_state_publisher',\r
        name='robot_state_publisher',\r
        parameters=[{'robot_description': robot_description,\r
                     'use_sim_time': False}],\r
    )\r
\r
    # ── RoboClaw motor controller ────────────────────────────────────────────\r
    roboclaw = Node(\r
        package='jetson_bot_slam',\r
        executable='roboclaw_node',\r
        name='roboclaw_node',\r
        parameters=[{\r
            'left_port':         LaunchConfiguration('left_port'),\r
            'right_port':        LaunchConfiguration('right_port'),\r
            'address':           0x80,\r
            'baudrate':          115200,\r
            'wheel_radius':      0.0508,\r
            'half_wheelbase':    0.1270,\r
            'half_track_width':  0.2172,\r
            'ticks_per_rev':     1440,\r
            'max_qpps':          2300,\r
            'control_hz':        20.0,\r
            'cmd_vel_timeout':   0.5,\r
        }],\r
        output='screen',\r
    )\r
\r
    # ── Mecanum odometry ─────────────────────────────────────────────────────\r
    mecanum_odom = Node(\r
        package='jetson_bot_slam',\r
        executable='mecanum_odometry',\r
        name='mecanum_odometry',\r
        parameters=[{\r
            'wheel_radius':      0.0508,\r
            'ticks_per_rev':     1440,\r
            'half_wheelbase':    0.1270,\r
            'half_track_width':  0.2172,\r
            'publish_tf':        True,\r
        }],\r
        output='screen',\r
    )\r
\r
    # ── USB Camera (usb_cam node, ROS2 Humble+) ───────────────────────────────\r
    #\r
    # Install:  sudo apt install ros-humble-usb-cam\r
    #\r
    # The camera_info_url should point to a calibration file.\r
    # Run  ros2 run camera_calibration cameracalibrator  to generate one.\r
    # Until calibration is done, RTAB-Map will use the raw image but loop\r
    # closure quality will be reduced.\r
    usb_camera = Node(\r
        package='usb_cam',\r
        executable='usb_cam_node_exe',\r
        name='usb_cam',\r
        parameters=[{\r
            'video_device':   '/dev/video' + LaunchConfiguration('camera_index'),\r
            'image_width':    LaunchConfiguration('camera_width'),\r
            'image_height':   LaunchConfiguration('camera_height'),\r
            'framerate':      LaunchConfiguration('camera_fps'),\r
            'pixel_format':   'yuyv',\r
            'camera_frame_id':'camera_optical_link',\r
            'io_method':      'mmap',\r
        }],\r
        remappings=[\r
            ('/usb_cam/image_raw',   '/image_raw'),\r
            ('/usb_cam/camera_info', '/camera_info'),\r
        ],\r
        output='screen',\r
    )\r
\r
    return LaunchDescription([\r
        left_port_arg,\r
        right_port_arg,\r
        camera_index_arg,\r
        camera_width_arg,\r
        camera_height_arg,\r
        camera_fps_arg,\r
        robot_state_pub,\r
        roboclaw,\r
        mecanum_odom,\r
        usb_camera,\r
    ])\r
