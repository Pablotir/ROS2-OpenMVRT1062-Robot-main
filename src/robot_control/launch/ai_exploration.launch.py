from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # USB Camera for AI (direct connection)
        Node(
            package='robot_control',
            executable='usb_camera_node',
            name='usb_camera',
            output='screen',
            parameters=[
                {'device': '/dev/video0'},
                {'fps': 2},
                {'width': 320},
                {'height': 240}
            ]
        ),
        
        # OpenMV Camera for SLAM (WiFi)
        Node(
            package='robot_control',
            executable='slam_receiver',
            name='slam_receiver',
            output='screen',
            parameters=[
                {'udp_port': 5006}
            ]
        ),
        
        Node(
            package='robot_control',
            executable='robot_control_node',
            name='robot_control_node',
            output='screen',
            parameters=[
                {'serial_port': '/dev/ttyUSB0'},
                {'baud': 115200},
                {'batch_mode': True},
                {'speed': 0.5}
            ]
        ),
        
        Node(
            package='robot_control',
            executable='ai_navigator',
            name='ai_navigator',
            output='screen',
            parameters=[
                {'ollama_host': 'http://localhost:8080'},
                {'model': 'gemma3:4b'},
                {'goal': 'find the kitchen to get a water bottle from the fridge'}
            ]
        ),
        
        Node(
            package='robot_control',
            executable='exploration_controller',
            name='exploration_controller',
            output='screen'
        )
    ])
