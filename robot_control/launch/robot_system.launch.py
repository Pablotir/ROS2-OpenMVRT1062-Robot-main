from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_control',
            executable='camera_receiver',
            name='camera_receiver',
            output='screen',
            parameters=[
                {'udp_port': 5005}
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
            executable='llm_planner',
            name='llm_planner',
            output='screen'
        )
    ])
