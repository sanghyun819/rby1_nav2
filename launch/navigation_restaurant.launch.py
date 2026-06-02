#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    slam = LaunchConfiguration("slam")
    map_yaml = LaunchConfiguration("map")
    use_composition = LaunchConfiguration("use_composition")

    custom_rviz_config_path = os.path.join(
        get_package_share_directory("rby1_nav2"), 
        "rviz",
        "nav2_restaurant.rviz"
    )

    default_nav2_params_path = os.path.join(
        get_package_share_directory("rby1_nav2"), "config", "test restaurant.yaml"
    )
    nav2_params = LaunchConfiguration("params_file")

    nav2_launch_file_dir = os.path.join(
        get_package_share_directory("nav2_bringup"), "launch"
    )
    slam_restaurant_launch_path = (
        "/home/nvidia/rby1_ws/src/slam_toolbox/launch/"
        "online_async_restaurant_launch.py"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation (Gazebo) clock if true",
            ),
            DeclareLaunchArgument(
                "slam",
                default_value="True",
                description="Run SLAM (mapless navigation)",
            ),
            DeclareLaunchArgument(
                "map",
                default_value="",
                description="Map yaml file (unused when slam is true)",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=default_nav2_params_path,
                description="Full path to Nav2 param file to load",
            ),
            DeclareLaunchArgument(
                "use_composition",
                default_value="False",
                description="Use composed Nav2 bringup if true",
            ),

            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(slam_restaurant_launch_path),
                condition=IfCondition(slam),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                }.items(),
            ),

            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [nav2_launch_file_dir, "/localization_launch.py"]
                ),
                condition=UnlessCondition(slam),
                launch_arguments={
                    "map": map_yaml,
                    "use_sim_time": use_sim_time,
                    "params_file": nav2_params,
                    "use_composition": use_composition,
                }.items(),
            ),

            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [nav2_launch_file_dir, "/navigation_launch.py"]
                ),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "params_file": nav2_params,
                    "use_composition": use_composition,
                }.items(),
            ),



            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', custom_rviz_config_path], # -d 옵션으로 경로 직접 주입
                parameters=[{'use_sim_time': use_sim_time}],
                output='screen'
            ),

            # Explore Lite (direct node to avoid duplicate DeclareLaunchArgument)
            # Node(
            #     package="explore_lite",
            #     name="explore_node",
            #     executable="explore",
            #     parameters=[
            #         os.path.join(
            #             get_package_share_directory("explore_lite"),
            #             "config",
            #             "params.yaml",
            #         ),
            #         {"use_sim_time": use_sim_time},
            #     ],
            #     output="screen",
            # ),
        ]
    )
