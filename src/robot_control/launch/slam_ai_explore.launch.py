"""\r
slam_ai_explore.launch.py\r
==========================\r
Integrated launch: SLAM + AI-guided exploration.\r
Combines jetson_bot_slam (hardware + SLAM Toolbox) with robot_control (AI navigator).\r
\r
Data flow\r
---------\r
RoboClaw L ──► roboclaw_node ──► /wheel_ticks ──► mecanum_odometry ──► /odom ──► SLAM Toolbox\r
RoboClaw R ──►\r
\r
exploration_controller ──► /cmd_vel ──► roboclaw_node ──► RoboClaw (HW PID)\r
ai_navigator (VILA)    ──► /ai/direction ──► exploration_controller\r
\r
Usage\r
-----\r
    ros2 launch robot_control slam_ai_explore.launch.py\r
\r
Requirements\r
------------\r
- VILA running via nano_llm inside the container\r
- jetson_bot_slam package built alongside robot_control in the same workspace\r
"""\r
\r
import os\r
from ament_index_python.packages import get_package_share_directory\r
from launch import LaunchDescription\r
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription\r
from launch.conditions import IfCondition\r
from launch.launch_description_sources import PythonLaunchDescriptionSource\r
from launch.substitutions import LaunchConfiguration, Command, PythonExpression\r
from launch_ros.actions import Node\r
from launch_ros.parameter_descriptions import ParameterValue\r
\r
\r
def generate_launch_description():\r
    slam_pkg = get_package_share_directory('jetson_bot_slam')\r
\r
    # ── Launch arguments ──────────────────────────────────────────────────────\r
    args = [\r
        DeclareLaunchArgument('left_port',          default_value='/dev/roboclaw_left'),\r
        DeclareLaunchArgument('right_port',         default_value='/dev/roboclaw_right'),\r
        DeclareLaunchArgument('camera_device',       default_value='/dev/video0'),\r
        DeclareLaunchArgument('ollama_host',         default_value='http://localhost:11434'),\r
        DeclareLaunchArgument('ollama_model',        default_value='gemma3:4b'),\r
        # Exploration (LiDAR-driven)\r
        DeclareLaunchArgument('move_speed',          default_value='0.20',\r
                              description='Forward speed m/s'),\r
        DeclareLaunchArgument('turn_speed',          default_value='0.55',\r
                              description='Turn speed rad/s'),\r
        DeclareLaunchArgument('obstacle_distance',   default_value='1.00',\r
                              description='Start slowing when LiDAR reads below this (m)'),\r
        DeclareLaunchArgument('emergency_stop_dist', default_value='0.10',\r
                              description='Emergency stop distance (m)'),\r
        DeclareLaunchArgument('label_every',         default_value='5',\r
                              description='AI scene label every N obstacle-turns'),\r
        # Optional SLAM (SLAM Toolbox)\r
        DeclareLaunchArgument('use_slam',            default_value='false',\r
                              description='Start SLAM Toolbox'),\r
    ]\r
\r
    # ── URDF / robot_state_publisher ──────────────────────────────────────────\r
    urdf_file = os.path.join(slam_pkg, 'urdf', 'jetson_bot.urdf.xacro')\r
    robot_state_pub = Node(\r
        package='robot_state_publisher',\r
        executable='robot_state_publisher',\r
        name='robot_state_publisher',\r
        parameters=[{'robot_description': ParameterValue(Command(['xacro ', urdf_file]), value_type=str),\r
                     'use_sim_time': False}],\r
    )\r
\r
    # ── RoboClaw motor controller node ────────────────────────────────────────\r
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
            'cmd_vel_timeout':   0.6,\r
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
            'wheel_radius':     0.0508,\r
            'ticks_per_rev':    1440,\r
            'half_wheelbase':   0.1270,\r
            'half_track_width': 0.2172,\r
            'publish_tf':       True,\r
        }],\r
        output='screen',\r
    )\r
\r
    # ── Camera (from robot_control) ───────────────────────────────────────────\r
    usb_camera = Node(\r
        package='robot_control',\r
        executable='usb_camera_node',\r
        name='usb_camera_node',\r
        parameters=[{\r
            'device':   LaunchConfiguration('camera_device'),\r
            'width':    320,\r
            'height':   240,\r
            'slam_fps': 5.0,\r
        }],\r
        output='screen',\r
    )\r
\r
    # ── AI navigator (scene labeller only — no movement control) ──────────────\r
    ai_navigator = Node(\r
        package='robot_control',\r
        executable='ai_navigator',\r
        name='ai_navigator',\r
        parameters=[{\r
            'ollama_host':  LaunchConfiguration('ollama_host'),\r
            'model':        LaunchConfiguration('ollama_model'),\r
            'infer_width':  160,\r
            'infer_height': 120,\r
            'jpeg_quality': 65,\r
        }],\r
        output='screen',\r
    )\r
\r
    # ── Exploration controller (LiDAR-reactive, pure autonomy) ────────────────\r
    exploration_ctrl = Node(\r
        package='robot_control',\r
        executable='exploration_controller',\r
        name='exploration_controller',\r
        parameters=[{\r
            'move_speed':           LaunchConfiguration('move_speed'),\r
            'turn_speed':           LaunchConfiguration('turn_speed'),\r
            'obstacle_distance':    LaunchConfiguration('obstacle_distance'),\r
            'emergency_stop_dist':  LaunchConfiguration('emergency_stop_dist'),\r
            'label_every':          LaunchConfiguration('label_every'),\r
        }],\r
        output='screen',\r
    )\r
\r
    return LaunchDescription(args + [\r
        robot_state_pub,\r
        roboclaw,\r
        mecanum_odom,\r
        usb_camera,\r
        ai_navigator,\r
        exploration_ctrl,\r
    ])\r