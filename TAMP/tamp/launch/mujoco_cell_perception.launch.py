"""Perception for the MuJoCo lab cell, wired exactly like the physical rig.

The point of this launch is that nothing in it is a simulation shortcut. The
same three stages run here as on the bench:

    rendered camera -> apriltag_ros -> cho_object_pose -> /perception/object_pose/<name>

MuJoCo renders the camera (mujoco_ros2_control registers every camera in the
scene and publishes image + camera_info), the stock detector finds the tags,
and cho_object_pose gates the detections and turns them into one robot-frame
pose per vessel. `tamp_real_server` subscribes to those topics for its World
State, so the planner is fed by perception rather than by the simulator's own
object poses -- which is the whole thing this cell exists to test.

Two differences from the bench, both stated rather than hidden:

* No rectification hop. The rendered image carries no distortion, because
  mujoco_ros2_control derives CameraInfo from the camera's fovy as an ideal
  pinhole. The physical launch (realsense_apriltag) runs image_proc first.
* The camera's TF comes from the scene file instead of the URDF, since the
  camera is a prop in the cell rather than a part of the robot.

Bring the cell up first, then this:

    ros2 launch cho_bringup_fr5 bringup_mujoco_robot.launch.py \
        gripper:=ag95 mujoco_scene:=scene_ag95_sdl.xml \
        controller_name:=joint_trajectory_controller
    ros2 launch <this file>
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

#: This file sits next to `scripts/` both in the source tree and in the
#: installed share directory, so one relative path serves either.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TAMP_ROOT = os.path.dirname(_HERE)

#: Tags in the cell. The ids are the scene's (beaker 0, flask 1) and the size is
#: the black square's edge: the plates are 125 mm and carry a 10-cell image
#: whose black square is 8 cells, so 100 mm.
#:
#: Not the 40 mm of the wrist-camera task: this camera stands 1.95 m off, where
#: a 40 mm tag spans 12.6 px and cho_object_pose rejects anything under a 25 px
#: shortest edge. Measured on the first run: 12.3 px, rejected.
TAG_IDS = [0, 1]
TAG_SIZE_M = 0.100


def generate_launch_description():
    scene_default = os.path.join(
        get_package_share_directory('cho_description_fr5'), 'xml', 'scene_ag95_sdl.xml')
    objects_default = os.path.join(
        _TAMP_ROOT, 'content', 'configs', 'perception', 'mujoco_cell_objects.yaml')
    detector_config = os.path.join(
        get_package_share_directory('realsense_apriltag'), 'config', 'apriltag_36h11.yaml')

    scene = LaunchConfiguration('scene')
    camera = LaunchConfiguration('camera')
    image_topic = LaunchConfiguration('image_topic')
    info_topic = LaunchConfiguration('info_topic')
    objects_config = LaunchConfiguration('objects_config')

    return LaunchDescription([
        DeclareLaunchArgument('scene', default_value=scene_default),
        DeclareLaunchArgument('camera', default_value='sdl_cam'),
        # mujoco_ros2_control's defaults for a camera with no <sensor> block:
        # <name>/color, <name>/camera_info, and the frame <name>_frame.
        DeclareLaunchArgument('image_topic', default_value='/sdl_cam/color'),
        DeclareLaunchArgument('info_topic', default_value='/sdl_cam/camera_info'),
        DeclareLaunchArgument('objects_config', default_value=objects_default),
        DeclareLaunchArgument('robot_type', default_value='fr5'),
        # Rendered images are noise-free, so agreement across samples says
        # little about accuracy here -- but keeping the gate on keeps this path
        # identical to the bench's, where it says a great deal.
        DeclareLaunchArgument('min_samples', default_value='5'),
        # The rendered camera runs at ~3.7 Hz (mujoco_ros2_control asks its
        # render loop for 5 Hz and does not quite get it), so the 0.5 s window
        # the bench uses cannot hold 5 samples and nothing would ever publish.
        # Widening the window rather than dropping min_samples keeps the gate
        # doing what it is for: agreement across independent detections.
        DeclareLaunchArgument('window_sec', default_value='2.0'),
        DeclareLaunchArgument('max_position_spread_m', default_value='0.01'),

        # base_link -> the camera frame the images are stamped with, read out of
        # the scene file so it cannot drift from what the renderer used.
        ExecuteProcess(
            cmd=['python3',
                 os.path.join(_TAMP_ROOT, 'scripts', 'perception', 'mujoco_camera_tf.py'),
                 '--scene', scene, '--camera', camera],
            name='mujoco_camera_tf',
            output='screen',
        ),

        Node(
            package='apriltag_ros',
            executable='apriltag_node',
            name='apriltag_node',
            output='screen',
            parameters=[
                detector_config,
                {
                    # The cell's tags, over the shared detector tuning. The frame
                    # names follow cho_object_pose's tag_<id> convention; the
                    # consumer derives the same string, so both sides agree.
                    'tag.ids': TAG_IDS,
                    'tag.sizes': [TAG_SIZE_M] * len(TAG_IDS),
                    'tag.frames': ['tag_%d' % tag_id for tag_id in TAG_IDS],
                },
            ],
            remappings=[('image_rect', image_topic), ('camera_info', info_topic)],
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('cho_object_pose'),
                'launch', 'object_pose.launch.py')),
            launch_arguments={
                'robot_type': LaunchConfiguration('robot_type'),
                'objects_config': objects_config,
                'min_samples': LaunchConfiguration('min_samples'),
                'window_sec': LaunchConfiguration('window_sec'),
                'max_position_spread_m': LaunchConfiguration('max_position_spread_m'),
            }.items(),
        ),
    ])
