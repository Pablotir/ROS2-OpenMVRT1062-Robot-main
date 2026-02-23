"""
slam.launch.py
==============
Launches the full SLAM stack on top of the robot bringup:
  - robot_bringup.launch.py      (sensors + odometry)
  - rtabmap_ros/rtabmap           (monocular visual SLAM + loop closure)
  - rviz2                         (visualisation)

RTAB-Map runs in monocular mode using wheel odometry for metric scale.
The published map is a 2-D occupancy grid (/map) + 3-D point cloud.

Usage:
    ros2 launch jetson_bot_slam slam.launch.py
    ros2 launch jetson_bot_slam slam.launch.py rviz:=false  # headless Jetson
    ros2 launch jetson_bot_slam slam.launch.py localization:=true  # after mapping

Install deps:
    sudo apt install ros-humble-rtabmap-ros
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('jetson_bot_slam')

    # ── Launch args ───────────────────────────────────────────────────────────
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Launch RViz2 for visualisation')

    localization_arg = DeclareLaunchArgument(
        'localization', default_value='false',
        description='Run in localization mode (map already built)')

    database_arg = DeclareLaunchArgument(
        'database_path',
        default_value=os.path.expanduser('~/jetson_bot_slam.db'),
        description='Path to the RTAB-Map database file')

    # ── Include bringup ───────────────────────────────────────────────────────
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'robot_bringup.launch.py')),
    )

    # ── Shared RTAB-Map parameters ─────────────────────────────────────────────
    rtabmap_params = {
        # ------- Memory -------------------------------------------------------
        'Mem/IncrementalMemory':       'true',
        'Mem/InitWMWithAllNodes':      'false',
        'Mem/RehearsalSimilarity':     '0.45',

        # ------- Visual odometry (not used – we supply wheel odom) ------------
        # We rely on /odom from mecanum_odometry_node, not visual odometry.
        # RTAB-Map uses the wheel odom for pose tracking and only does visual
        # loop closure to correct accumulated drift.
        'Vis/EstimationType':          '1',      # 1 = Essential matrix (mono)
        'Vis/MinInliers':              '15',
        'Vis/InlierDistance':          '0.1',
        'Vis/MaxDepth':                '4.0',
        'Vis/FeatureType':             '6',       # 6 = ORB (fast on Jetson)
        'Vis/MaxFeatures':             '500',

        # ------- Registration -------------------------------------------------
        'Reg/Strategy':                '0',       # 0 = Visual
        'Reg/Force3DoF':               'true',    # planar robot

        # ------- SLAM vs. localisation ----------------------------------------
        # (overridden per-mode below)

        # ------- Optimisation -------------------------------------------------
        'Optimizer/Strategy':          '1',       # 1 = g2o
        'RGBD/NeighborLinkRefining':   'true',
        'RGBD/OptimizeMaxError':       '3.0',
        'RGBD/ProximityBySpace':       'true',
        'RGBD/AngularUpdate':          '0.05',    # re-detect after 3°
        'RGBD/LinearUpdate':           '0.05',    # re-detect after 5 cm

        # ------- Grid map ─────────────────────────────────────────────────────
        'Grid/RayTracing':             'true',
        'Grid/3D':                     'false',   # 2-D map only (faster)
        'Grid/CellSize':               '0.05',    # 5 cm/cell
        'Grid/MaxGroundAngle':         '45',
        'GridGlobal/MinSize':          '20',      # min 20 m map before publish

        # ------- Topics ───────────────────────────────────────────────────────
        'subscribe_depth':             'false',   # monocular (no depth)
        'subscribe_rgb':               'true',
        'subscribe_odom':              'true',
        'approx_sync':                 'true',
        'approx_sync_max_interval':    '0.1',
    }

    # SLAM mode params (building the map)
    slam_params = {**rtabmap_params,
                   'Mem/IncrementalMemory': 'true',
                   'Mem/InitWMWithAllNodes': 'false'}

    # Localisation mode params (navigating in a known map)
    loc_params  = {**rtabmap_params,
                   'Mem/IncrementalMemory': 'false',
                   'Mem/InitWMWithAllNodes': 'true'}

    # ── RTAB-Map SLAM node ────────────────────────────────────────────────────
    rtabmap_slam = Node(
        package='rtabmap_ros',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=[slam_params,
                    {'database_path': LaunchConfiguration('database_path')}],
        remappings=[
            ('rgb/image',       '/image_raw'),
            ('rgb/camera_info', '/camera_info'),
            ('odom',            '/odom'),
        ],
        arguments=['--delete_db_on_start'],
        condition=UnlessCondition(LaunchConfiguration('localization')),
    )

    # ── RTAB-Map LOCALISATION node ────────────────────────────────────────────
    rtabmap_loc = Node(
        package='rtabmap_ros',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=[loc_params,
                    {'database_path': LaunchConfiguration('database_path')}],
        remappings=[
            ('rgb/image',       '/image_raw'),
            ('rgb/camera_info', '/camera_info'),
            ('odom',            '/odom'),
        ],
        condition=IfCondition(LaunchConfiguration('localization')),
    )

    # ── RViz2 ─────────────────────────────────────────────────────────────────
    rviz_cfg = os.path.join(pkg, 'rviz', 'slam_view.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        rviz_arg,
        localization_arg,
        database_arg,
        bringup,
        rtabmap_slam,
        rtabmap_loc,
        rviz,
    ])
