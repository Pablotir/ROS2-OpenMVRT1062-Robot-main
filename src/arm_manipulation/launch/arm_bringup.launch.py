import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_name = 'arm_manipulation'
    pkg_share = get_package_share_directory(pkg_name)

    # Declare launch arguments
    arm_port_arg = DeclareLaunchArgument(
        'arm_port',
        default_value='/dev/arm_controller',
        description='Serial port for the arm controller'
    )
    
    calibration_file_arg = DeclareLaunchArgument(
        'calibration_file',
        default_value='/root/ros2_ws/calibration/hand_eye_calibration.yaml',
        description='Path to hand-eye calibration file'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/root/ros2_ws/models/yolo11n-seg.engine',
        description='Path to the YOLO model engine'
    )
    
    config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value=os.path.join(pkg_share, 'config', 'arm_params.yaml'),
        description='Path to arm parameters YAML file'
    )
    
    skip_moondream_arg = DeclareLaunchArgument(
        'skip_moondream',
        default_value='true',
        description='Skip loading moondream vision-language model'
    )

    # 1. robot_state_publisher with xacro processing
    urdf_path = os.path.join(pkg_share, 'urdf', 'so_arm101.urdf.xacro')
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': Command(['xacro ', urdf_path])
        }],
        output='screen'
    )

    # 2. realsense2_camera_node (D405 Jetson specifics)
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='d405_camera',
        parameters=[{
            'camera_name': 'd405',
            'enable_color': True,
            'enable_depth': True,
            'enable_infra': False,
            'enable_infra1': False,
            'enable_infra2': False,
            'enable_sync': False,
            'rgb_camera.profile': '848x480x15',
            'depth_module.profile': '848x480x15',
            'align_depth.enable': False,
            'pointcloud.enable': False,
            'initial_reset': True,
        }],
        output='screen'
    )

    # 3. arm_driver_node
    arm_driver_node = Node(
        package=pkg_name,
        executable='arm_driver_node',
        name='arm_driver_node',
        parameters=[{
            'port': LaunchConfiguration('arm_port')
        }],
        output='screen'
    )

    # 4. hand_eye_tf_broadcaster
    hand_eye_tf_broadcaster_node = Node(
        package=pkg_name,
        executable='hand_eye_tf_broadcaster',
        name='hand_eye_tf_broadcaster',
        parameters=[{
            'calibration_file': LaunchConfiguration('calibration_file')
        }],
        output='screen'
    )

    # 5. detection_node
    detection_node = Node(
        package=pkg_name,
        executable='detection_node',
        name='detection_node',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'skip_moondream': LaunchConfiguration('skip_moondream')
        }],
        output='screen'
    )

    # 6. pick_place_node
    pick_place_node = Node(
        package=pkg_name,
        executable='pick_place_node',
        name='pick_place_node',
        parameters=[{
            'config_path': LaunchConfiguration('config_path'),
        }],
        output='screen'
    )

    return LaunchDescription([
        arm_port_arg,
        calibration_file_arg,
        model_path_arg,
        config_path_arg,
        skip_moondream_arg,
        robot_state_publisher_node,
        realsense_node,
        arm_driver_node,
        hand_eye_tf_broadcaster_node,
        detection_node,
        pick_place_node
    ])
