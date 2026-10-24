# Copyright 2026 Hyunho Cho
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

"""The FR5 pour cell in MuJoCo: the core's FR5 bringup, plus pouring_controller.

    ros2 launch sdl_cell_bringup bringup_mujoco_cell.launch.py
    ros2 launch sdl_cell_bringup bringup_mujoco_cell.launch.py moveit:=true

Includes cho_bringup_fr5's bringup_mujoco_robot (or bringup_mujoco_moveit with
moveit:=true) unchanged, with the AG-95 and the SDL lab cell scene
(cho_description_fr5/xml/scene_ag95_sdl.xml) as defaults; every other argument
of that file is passed through. Then it loads pouring_controller inactive, with
its type and parameters given to the spawner, so the core's controller list and
config need no entry for it. It claims the arm like the core's controllers;
the cell's registry (config/robot_config_additions.yaml) is what makes every
exclusive switch take it down again.

MuJoCo has no scale: the controller is loaded and switchable, and refuses a
pour goal for want of readings (scale_timeout) unless something publishes
sdl_cell_interfaces/ScaleReading on its scale_topic.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

POUR_CONTROLLER = 'pouring_controller'
POUR_CONTROLLER_TYPE = 'sdl_pour_controller/PouringController'


def _include_core(context):
    moveit = LaunchConfiguration('moveit').perform(context).strip().lower()
    if moveit not in ('true', 'false'):
        raise RuntimeError("moveit must be 'true' or 'false', got %r" % moveit)
    launch_file = 'bringup_mujoco_moveit.launch.py' if moveit == 'true' else 'bringup_mujoco_robot.launch.py'
    core = os.path.join(get_package_share_directory('cho_bringup_fr5'), 'launch', launch_file)
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(core),
        launch_arguments={
            'gripper': LaunchConfiguration('gripper'),
            'mujoco_scene': LaunchConfiguration('mujoco_scene'),
        }.items(),
    )]


def generate_launch_description():
    share = get_package_share_directory('sdl_cell_bringup')
    return LaunchDescription([
        DeclareLaunchArgument(
            'moveit', default_value='false',
            description='true includes bringup_mujoco_moveit (move_group, the MoveIt bridge) instead of '
                        'bringup_mujoco_robot'),
        DeclareLaunchArgument(
            'gripper', default_value='ag95',
            description="The core bringup's gripper:=; the cell pours from the AG-95's side grasp"),
        DeclareLaunchArgument(
            'mujoco_scene', default_value='scene_ag95_sdl.xml',
            description="The core bringup's mujoco_scene:= (a file in cho_description_fr5/xml)"),
        DeclareLaunchArgument(
            'pour_params', default_value=os.path.join(share, 'config', 'mujoco', 'pouring_controller.yaml'),
            description='pouring_controller parameter file'),
        DeclareLaunchArgument(
            'controller_manager_timeout', default_value='60',
            description='How long the pour spawner waits for the controller_manager [s]'),
        OpaqueFunction(function=_include_core),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                POUR_CONTROLLER,
                '--controller-type', POUR_CONTROLLER_TYPE,
                '--param-file', LaunchConfiguration('pour_params'),
                '--inactive',
                '--controller-manager-timeout', LaunchConfiguration('controller_manager_timeout'),
            ],
            parameters=[{'use_sim_time': True}],
            output='screen',
        ),
    ])
