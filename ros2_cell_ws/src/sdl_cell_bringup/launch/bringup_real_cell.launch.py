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

"""The real FR5 pour cell: the core's FR5 bringup, plus pouring_controller.

    ros2 launch sdl_cell_bringup bringup_real_cell.launch.py
    ros2 launch sdl_cell_bringup bringup_real_cell.launch.py moveit:=true

Includes cho_bringup_fr5's bringup_real_robot (or bringup_real_moveit with
moveit:=true) unchanged; every argument of that file is passed through (the
robot IP and the gripper come from its config/real/fr5.config.yaml). Then it
loads pouring_controller inactive with config/real/pouring_controller.yaml.
It closes the loop on sdl_cell_interfaces/ScaleReading, which the scale driver
and the relay publish:

    ros2 launch hansung_scale_driver scale.launch.py
    ros2 run sdl_cell_tasks scale_relay

It claims the arm like the core's controllers; the cell's registry
(config/robot_config_additions.yaml) is what makes every exclusive switch take
it down again.
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
    launch_file = 'bringup_real_moveit.launch.py' if moveit == 'true' else 'bringup_real_robot.launch.py'
    core = os.path.join(get_package_share_directory('cho_bringup_fr5'), 'launch', launch_file)
    return [IncludeLaunchDescription(PythonLaunchDescriptionSource(core))]


def generate_launch_description():
    share = get_package_share_directory('sdl_cell_bringup')
    return LaunchDescription([
        DeclareLaunchArgument(
            'moveit', default_value='false',
            description='true includes bringup_real_moveit (move_group, the MoveIt bridge) instead of '
                        'bringup_real_robot'),
        DeclareLaunchArgument(
            'pour_params', default_value=os.path.join(share, 'config', 'real', 'pouring_controller.yaml'),
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
            output='screen',
        ),
    ])
