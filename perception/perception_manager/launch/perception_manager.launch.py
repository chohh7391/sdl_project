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

import yaml

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
    with open(os.path.join(pkg_dir, 'config', 'wrist_camera.yaml')) as fh:
        mount = yaml.safe_load(fh)
    with open(os.path.join(pkg_dir, 'config', 'tag_mount.yaml')) as fh:
        tag_heights = yaml.safe_load(fh)

    apriltag_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('apriltag_ros'), 'launch', 'apriltag.launch.py'
        ])),
        launch_arguments={
            'camera_names': ','.join(camera_list),
            'tag_sizes': '%s:%s' % (mount['frame'], mount['tag_size']),
        }.items(),
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
            {"tag_z_%s" % k: float(v) for k, v in tag_heights.items()},
        ],
    )
    nodes = [apriltag_launch, perception_manager_node]

    # A camera on the arm moves with it, so its transform is the arm's forward
    # kinematics plus the mount: robot_state_publisher turns the simulator's
    # joint states into base_link -> ... -> wrist3_link, and the mount is a
    # static wrist3_link -> camera_3. The same file places the simulated camera
    # (task.py), so the view and its transform agree by construction.
    if mount['frame'] in camera_list:
        with open(os.path.join(pkg_dir, 'config', 'fr5_arm.urdf')) as fh:
            arm_urdf = fh.read()
        nodes.append(Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='wrist_camera_kinematics',
            output='log',
            # ignore_timestamp: publish on every joint state. Otherwise the node
            # skips a message stamped before the last one it published, and the
            # simulator's clock restarts at zero when a tool change rebuilds the
            # world, so the arm's transforms stopped until the new clock passed
            # the old one (~50 s). Move's vgc10 hit this: its joint states add no
            # new joint names, which is what had unblocked the other tools, and
            # the wrist scan of 20261001f Move seed 17 found nothing.
            parameters=[{'robot_description': arm_urdf, 'publish_frequency': 60.0,
                         'ignore_timestamp': True}],
            remappings=[('joint_states', '/isaac_joint_states')],
        ))
        x, y, z = (str(v) for v in mount['xyz'])
        roll, pitch, yaw = (str(v) for v in mount['rpy'])
        nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='wrist_camera_mount',
            output='log',
            arguments=['--x', x, '--y', y, '--z', z,
                       '--roll', roll, '--pitch', pitch, '--yaw', yaw,
                       '--frame-id', mount['parent'], '--child-frame-id', mount['frame']],
        ))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'cameras',
            default_value=os.environ.get('SDL_PERCEPTION_CAMERAS', 'camera_1,camera_2,camera_3'),
            description='Comma-separated camera TF frames to detect in and fuse over.',
        ),
        OpaqueFunction(function=_setup),
    ])
