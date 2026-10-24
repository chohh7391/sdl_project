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

"""The cell's launch files load, and point at files that exist."""

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.launch_description_sources import get_launch_description_from_python_launch_file
import pytest
import yaml

SHARE = get_package_share_directory('sdl_cell_bringup')
CORE_LAUNCH = os.path.join(get_package_share_directory('cho_bringup_fr5'), 'launch')
BACKENDS = {
    'mujoco': ('bringup_mujoco_robot.launch.py', 'bringup_mujoco_moveit.launch.py'),
    'real': ('bringup_real_robot.launch.py', 'bringup_real_moveit.launch.py'),
}


def _arguments(backend):
    path = os.path.join(SHARE, 'launch', 'bringup_%s_cell.launch.py' % backend)
    description = get_launch_description_from_python_launch_file(path)
    return {entity.name: entity for entity in description.entities if isinstance(entity, DeclareLaunchArgument)}


@pytest.mark.parametrize('backend', sorted(BACKENDS))
def test_the_core_bringups_it_includes_exist(backend):
    for launch_file in BACKENDS[backend]:
        assert os.path.isfile(os.path.join(CORE_LAUNCH, launch_file))


@pytest.mark.parametrize('backend', sorted(BACKENDS))
def test_the_default_pour_parameters_are_the_backends_own(backend):
    arguments = _arguments(backend)
    assert {'moveit', 'pour_params', 'controller_manager_timeout'} <= set(arguments)
    default = ''.join(part.text for part in arguments['pour_params'].default_value)
    assert default == os.path.join(SHARE, 'config', backend, 'pouring_controller.yaml')
    with open(default) as stream:
        params = yaml.safe_load(stream)['pouring_controller']['ros__parameters']
    assert params['bringup_type'] == backend
    assert params['robot_type'] == 'fr5'


def test_the_mujoco_cell_defaults_to_the_lab_cell_scene():
    arguments = _arguments('mujoco')
    scene = ''.join(part.text for part in arguments['mujoco_scene'].default_value)
    assert os.path.isfile(os.path.join(get_package_share_directory('cho_description_fr5'), 'xml', scene))
    assert ''.join(part.text for part in arguments['gripper'].default_value) == 'ag95'
