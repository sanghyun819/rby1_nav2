from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="pointcloud_to_laserscan",
            executable="pointcloud_to_laserscan_node",
            name="pointcloud_to_laserscan",
            output="screen",
            remappings=[
                ("cloud_in", "livox/lidar"),
                ("scan", "livox/scan"),
            ],
            # parameters=[{
            #     "target_frame": "livox_lidar",
            #     "transform_tolerance": 0.01,
            #     "min_height": -0.55,
            #     "max_height": 1.0,
            #     "angle_min": -1.57,
            #     "angle_max": 1.57,
            #     "angle_increment": 0.0087,
            #     "scan_time": 1.0,
            #     "range_min": 0.1,
            #     "range_max": 30.0,
            #     "use_inf": True,
            #     "concurrency_level": 1,
            # }],
            parameters=[{
                "target_frame": "livox_lidar",
                "transform_tolerance": 0.01,
                "min_height": -1.3,
                "max_height": 1.0,

                # "angle_min": -1.5707963267948966,
                # "angle_max":  1.5707963267948966,
                # "angle_increment": 0.008726646259971648,  # pi/360 -> 361 beams
                # "angle_min": -1.5707963267948966,                 # -pi/2
                # "angle_max":   1.5707973267948966,                 # +pi/2 + 1e-6  (여기가 핵심)
                "angle_increment": 0.008726646259971648,           # pi/360


                "scan_time": 0.1,
                "range_min": 0.3,
                "range_max": 30.0,
                "use_inf": True,
                "concurrency_level": 1,
            }],
        )
    ])
