#!/usr/bin/env python3

# Copyright 2019 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")
    slam = LaunchConfiguration("slam", default="True")
    map_yaml = LaunchConfiguration("map", default="")

    nav2_params = LaunchConfiguration(
        "params_file",
        default=os.path.join(
            get_package_share_directory("rby1_nav2"), "config", "test.yaml"
        ),
    )

    nav2_launch_file_dir = os.path.join(
        get_package_share_directory("nav2_bringup"), "launch"
    )
    slam_toolbox_launch_file_dir = os.path.join(
        get_package_share_directory("slam_toolbox"), "launch"
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
                default_value=nav2_params,
                description="Full path to Nav2 param file to load",
            ),

            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [slam_toolbox_launch_file_dir, "/online_async_launch.py"]
                ),
                condition=IfCondition(slam),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                }.items(),
            ),

            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [nav2_launch_file_dir, "/bringup_launch.py"]
                ),
                launch_arguments={
                    "slam": slam,
                    "map": map_yaml,
                    "use_sim_time": use_sim_time,
                    "params_file": nav2_params,
                }.items(),
            ),
        ]
    )
