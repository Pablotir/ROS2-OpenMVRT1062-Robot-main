"""
explore.launch.py
=================
Launches the full autonomy stack for bedroom SLAM exploration:
  - slam.launch.py         (RTAB-Map + robot bringup)
  - nav2_bringup           (path planning + controller)
  - explore_lite           (frontier-based auto-exploration)

The robot will:
  1. Start RTAB-Map building a map from scratch.
  2. Nav2 provides costmap and global / local planning.
  3. explore_lite picks unexplored frontiers and sends goal poses to Nav2.
  4. The robot autonomously explores until the room is mapped.

Usage:
    ros2 launch jetson_bot_slam explore.launch.py

Install deps:
    sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
    sudo apt install ros-humble-explore-lite

Stop exploration:
    ros2 topic pub /explore/resume std_msgs/Bool "data: false" --once
Resume:
    ros2 topic pub /explore/resume std_msgs/Bool "data: true"  --once
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg     = get_package_share_directory('jetson_bot_slam')
    nav2_pkg = get_package_share_directory('nav2_bringup')

    # ── Launch args ───────────────────────────────────────────────────────────
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true')

    # ── Include SLAM (which includes bringup) ─────────────────────────────────
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'slam.launch.py')),
        launch_arguments={'rviz': LaunchConfiguration('rviz')}.items(),
    )

    # ── Nav2 bringup ──────────────────────────────────────────────────────────
    nav2_params = os.path.join(pkg, 'config', 'nav2_params.yaml')

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_pkg, 'launch', 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time':    'false',
            'params_file':      nav2_params,
            'autostart':       'true',
        }.items(),
    )

    # ── explore_lite ──────────────────────────────────────────────────────────
    explore = Node(
        package='explore_lite',
        executable='explore',
        name='explore',
        output='screen',
        parameters=[{
            # How aggressively to explore:
            'robot_base_frame':         'base_link',
            'costmap_topic':            '/global_costmap/costmap',
            'costmap_updates_topic':    '/global_costmap/costmap_updates',
            'visualize':                True,
            'planner_frequency':        0.33,    # re-plan every 3 s
            'progress_timeout':         30.0,    # give up on frontier after 30 s
            'potential_scale':          3.0,
            'gain_scale':               1.0,
            'transform_tolerance':      0.3,
            'min_frontier_size':        0.75,    # ignore tiny frontiers < 75 cm
        }],
    )

    return LaunchDescription([
        rviz_arg,
        slam,
        nav2,
        explore,
    ])
