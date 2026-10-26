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
moveit:=true) unchanged; every argument of that file is passed through but
config_file. That one is the core's config/real/fr5.config.yaml (the robot IP,
the gripper registration; ``fr5_config:=`` names another) with this cell's
config/real/fr5_config_overrides.yaml merged over it -- today the AG-95's grip
force, 60 % -- written to a temporary file. Then it loads pouring_controller
inactive with config/real/pouring_controller.yaml.
It closes the loop on sdl_cell_interfaces/ScaleReading, which the scale driver
and the relay publish:

    ros2 launch hansung_scale_driver scale.launch.py
    ros2 run sdl_cell_tasks scale_relay

It claims the arm like the core's controllers; the cell's registry
(config/robot_config_additions.yaml) is what makes every exclusive switch take
it down again.
"""

import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml

POUR_CONTROLLER = 'pouring_controller'
POUR_CONTROLLER_TYPE = 'sdl_pour_controller/PouringController'


def merged_config(base, overrides, where='fr5_config_overrides'):
    """*base* with *overrides* laid over it, key by key; a key *base* lacks is refused."""
    merged = dict(base)
    for key, value in overrides.items():
        if key not in base:
            raise RuntimeError('%s: %r is not in the core fr5.config.yaml, so it would be added '
                               'rather than override anything' % (where, key))
        if isinstance(value, dict):
            if not isinstance(base[key], dict):
                raise RuntimeError('%s: %r is a mapping here and not in the core file' % (where, key))
            merged[key] = merged_config(base[key], value, '%s.%s' % (where, key))
        else:
            merged[key] = value
    return merged


def cell_fr5_config(base_path, overrides_path):
    """Write the core's fr5.config.yaml with the cell's overrides; return the file's path."""
    with open(base_path, encoding='utf-8') as stream:
        base = yaml.safe_load(stream) or {}
    with open(overrides_path, encoding='utf-8') as stream:
        overrides = yaml.safe_load(stream) or {}
    merged = merged_config(base, overrides)
    handle, path = tempfile.mkstemp(prefix='sdl_cell_fr5_config_', suffix='.yaml')
    with os.fdopen(handle, 'w', encoding='utf-8') as stream:
        stream.write('# %s, with %s merged over it (sdl_cell_bringup)\n' % (base_path, overrides_path))
        yaml.safe_dump(merged, stream, sort_keys=False)
    return path


def _include_core(context):
    moveit = LaunchConfiguration('moveit').perform(context).strip().lower()
    if moveit not in ('true', 'false'):
        raise RuntimeError("moveit must be 'true' or 'false', got %r" % moveit)
    launch_file = 'bringup_real_moveit.launch.py' if moveit == 'true' else 'bringup_real_robot.launch.py'
    core = os.path.join(get_package_share_directory('cho_bringup_fr5'), 'launch', launch_file)
    config = cell_fr5_config(LaunchConfiguration('fr5_config').perform(context),
                             LaunchConfiguration('fr5_config_overrides').perform(context))
    print('[sdl_cell_bringup] fr5 config: %s' % config)
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(core),
        launch_arguments={'config_file': config}.items())]


def generate_launch_description():
    share = get_package_share_directory('sdl_cell_bringup')
    core_config = os.path.join(
        get_package_share_directory('cho_bringup_fr5'), 'config', 'real', 'fr5.config.yaml')
    return LaunchDescription([
        DeclareLaunchArgument(
            'moveit', default_value='false',
            description='true includes bringup_real_moveit (move_group, the MoveIt bridge) instead of '
                        'bringup_real_robot'),
        DeclareLaunchArgument(
            'fr5_config', default_value=core_config,
            description="The core bringup's fr5.config.yaml the cell's overrides go over"),
        DeclareLaunchArgument(
            'fr5_config_overrides',
            default_value=os.path.join(share, 'config', 'real', 'fr5_config_overrides.yaml'),
            description='What the cell sets differently there (the grip force); every key must exist '
                        'in fr5_config'),
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
