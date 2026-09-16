"""Bring up AprilTag detection and the fusion node over a configurable camera set.

The camera list was hardcoded here as 'camera_1,camera_2' and was passed only to
the detector -- `perception_manager` was never told which frames to fuse, so it
silently fell back to its own two-camera default. Adding a third camera then
meant editing two files in agreement. One list now feeds both, and it is
overridable without editing anything:

    ros2 launch perception_manager perception_manager.launch.py cameras:=camera_1,camera_2,camera_3
    SDL_PERCEPTION_CAMERAS=camera_1,camera_2,camera_3 ros2 launch ...

The frame names must be the TF frames the detector publishes tags relative to,
i.e. the `frame_id` each camera_info carries.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _setup(context, *args, **kwargs):
    cameras = LaunchConfiguration('cameras').perform(context)
    camera_list = [c.strip() for c in cameras.split(',') if c.strip()]

    pkg_dir = get_package_share_directory('perception_manager')
    config_file_path = os.path.join(pkg_dir, 'config', 'object_configs.yaml')

    apriltag_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('apriltag_ros'), 'launch', 'apriltag.launch.py'
        ])),
        launch_arguments={'camera_names': ','.join(camera_list)}.items(),
    )

    perception_manager_node = Node(
        package='perception_manager',
        executable='perception_manager_node',
        name='perception_manager',
        output='screen',
        parameters=[
            config_file_path,
            {"publish_tf": False},
            {"camera_frames": camera_list},
        ],
    )
    return [apriltag_launch, perception_manager_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'cameras',
            default_value=os.environ.get('SDL_PERCEPTION_CAMERAS', 'camera_1,camera_2'),
            description='Comma-separated camera TF frames to detect in and fuse over.',
        ),
        OpaqueFunction(function=_setup),
    ])
