import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    # Define the camera names
    # cameras = ["camera_wrist", "camera_fixed_front", "camera_fixed_back"]
    cameras = ["camera_fixed_front"]
    
    # Get the path to the parameter file
    apriltag_params = PathJoinSubstitution([
        get_package_share_directory("apriltag_ros"),
        "cfg",
        "tags_36h11.yaml",
    ])

    # A list to hold the Node actions
    node_list = []

    # Loop through the camera names to create a Node action for each
    for name in cameras:
        node = Node(
            package="apriltag_ros",
            executable="apriltag_node",
            name="apriltag_node",
            namespace=name,  # The namespace for this camera's node
            parameters=[apriltag_params],
            remappings=[
                ("image_rect", f"/{name}/rgb"),
                ("camera_info", f"/{name}/camera_info"),
                ("detections", f"/{name}/apriltag/detections"),
            ],
            output="log",
        )
        node_list.append(node)

    return LaunchDescription(node_list)