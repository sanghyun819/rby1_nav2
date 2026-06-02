#!/usr/bin/env python3
"""
Navigation + Height Costmap Launch
====================================
Nav2 bringup + height_costmap_node + RViz2를 함께 실행.
RViz2에서 기존 2D맵 위에 height costmap이 히트맵처럼 오버레이됩니다.

사용법:
  ros2 launch rby1_nav2 navigation_heightmap.launch.py
  ros2 launch rby1_nav2 navigation_heightmap.launch.py map:=/path/to/map.yaml pcd:=/path/to/map.pcd
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('rby1_nav2')

    # ── Launch Arguments ──
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    map_dir = LaunchConfiguration(
        'map',
        default=os.path.join(pkg_dir, 'maps', 'map2.yaml')
    )

    pcd_file = LaunchConfiguration(
        'pcd',
        default=os.path.join(pkg_dir, 'maps', 'map2_aligned.pcd')
    )

    param_file = LaunchConfiguration(
        'params_file',
        default=os.path.join(pkg_dir, 'config', 'param.yaml')
    )

    nav2_launch_dir = os.path.join(
        get_package_share_directory('nav2_bringup'), 'launch'
    )

    return LaunchDescription([
        # ── Declare Arguments ──
        DeclareLaunchArgument('map', default_value=map_dir,
                              description='Full path to map yaml'),
        DeclareLaunchArgument('pcd', default_value=pcd_file,
                              description='Full path to PCD file'),
        DeclareLaunchArgument('params_file', default_value=param_file,
                              description='Full path to Nav2 param file'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Use simulation clock'),

        LogInfo(msg=['Map: ', map_dir]),
        LogInfo(msg=['PCD: ', pcd_file]),

        # ── Nav2 Bringup (map_server + planner + controller 등) ──
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_launch_dir, 'bringup_launch.py')
            ),
            launch_arguments={
                'map': map_dir,
                'use_sim_time': use_sim_time,
                'params_file': param_file,
            }.items(),
        ),

        # ── Height Costmap Node ──
        Node(
            package='rby1_nav2',
            executable='height_costmap_node',
            name='height_costmap_node',
            output='screen',
            parameters=[{
                'pcd_file': pcd_file,
                'map_yaml': map_dir,
                'publish_rate': 1.0,
                'obstacle_z_min': 0.5,
                'obstacle_z_max': 2.5,
                'inflation_radius': 0.4,
                'cost_scaling_factor': 3.0,
                'frame_id': 'map',
            }],
        ),

        # ── Map↔Odom TF Broadcaster ──
        # AMCL 대신 map→odom TF를 발행.
        # RViz2 "2D Pose Estimate"로 로봇 위치를 지정하면 map→odom이 갱신됩니다.
        Node(
            package='rby1_nav2',
            executable='map_pose_broadcaster',
            name='map_pose_broadcaster',
            output='screen',
            parameters=[{
                'map_frame': 'map',
                'odom_frame': 'odom',
                'base_frame': 'base',
                'broadcast_rate': 20.0,
            }],
        ),

        # ── RViz2 ──
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
        ),
    ])
