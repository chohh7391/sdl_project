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

"""The cell's launch files load, point at files that exist, and change only what they say."""

import importlib.util
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
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


def _module(path, name):
    """A launch file as a module, for the functions it defines."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the real cell's camera poses ---------------------------------------------

CORE_REAL = os.path.join(get_package_share_directory('cho_bringup_fr5'), 'config', 'real')


def test_the_cells_extrinsics_pass_the_cores_check_and_place_the_same_cameras():
    # Launched by the core's camera_extrinsics.launch.py with extrinsics_config:=
    # this file, which reads it through load_transforms().
    core = _module(os.path.join(CORE_LAUNCH, 'camera_extrinsics.launch.py'), 'core_extrinsics')
    cell = core.load_transforms(os.path.join(SHARE, 'config', 'real', 'camera_extrinsics.yaml'))
    shipped = core.load_transforms(os.path.join(CORE_REAL, 'camera_extrinsics.yaml'))
    frames = [(entry['name'], entry['parent_frame'], entry['child_frame']) for entry in cell]
    assert frames == [(entry['name'], entry['parent_frame'], entry['child_frame'])
                      for entry in shipped]


# --- the real cell's fr5.config.yaml -------------------------------------------

def _real_cell():
    return _module(os.path.join(SHARE, 'launch', 'bringup_real_cell.launch.py'), 'real_cell')


def test_the_cell_holds_beakers_harder_and_changes_nothing_else():
    real = _real_cell()
    with open(os.path.join(CORE_REAL, 'fr5.config.yaml')) as stream:
        core = yaml.safe_load(stream)
    with open(os.path.join(SHARE, 'config', 'real', 'fr5_config_overrides.yaml')) as stream:
        overrides = yaml.safe_load(stream)
    merged = real.merged_config(core, overrides)
    assert merged['fr5']['gripper_config']['force_percent'] == 60
    merged['fr5']['gripper_config']['force_percent'] = core['fr5']['gripper_config']['force_percent']
    assert merged == core


def test_an_override_the_core_file_does_not_have_is_refused():
    real = _real_cell()
    with pytest.raises(RuntimeError, match='force_procent'):
        real.merged_config({'fr5': {'gripper_config': {'force_percent': 30}}},
                           {'fr5': {'gripper_config': {'force_procent': 60}}})


def test_the_merged_config_is_what_the_core_bringup_is_given(tmp_path):
    real = _real_cell()
    arguments = _arguments('real')
    assert {'fr5_config', 'fr5_config_overrides'} <= set(arguments)
    path = real.cell_fr5_config(os.path.join(CORE_REAL, 'fr5.config.yaml'),
                                os.path.join(SHARE, 'config', 'real', 'fr5_config_overrides.yaml'))
    try:
        with open(path) as stream:
            written = yaml.safe_load(stream)
    finally:
        os.unlink(path)
    assert written['fr5']['gripper_config']['force_percent'] == 60


# --- a D435 with colour beside infra1 -------------------------------------------

D435 = os.path.join(SHARE, 'launch', 'd435.launch.py')


def test_rs_launch_still_lets_the_config_file_win():
    # What stream:=both rests on: the core launch passes enable_color as a launch
    # argument, and rs_launch gives the node [launch arguments, config file], the
    # later winning. If realsense2_camera changes that order, colour stays off.
    rs_launch = os.path.join(get_package_share_directory('realsense2_camera'), 'launch',
                             'rs_launch.py')
    with open(rs_launch) as stream:
        assert 'parameters=[params, params_from_file]' in stream.read()


def test_both_turns_colour_on_and_leaves_infra1_as_the_camera_config_has_it():
    d435 = _module(D435, 'cell_d435')
    side = os.path.join(CORE_REAL, 'd435_side.yaml')
    path = d435.color_config(side, '640x480x15')
    try:
        with open(path) as stream:
            written = yaml.safe_load(stream)
    finally:
        os.unlink(path)
    with open(side) as stream:
        given = yaml.safe_load(stream)
    assert written.pop('enable_color') is True
    assert written.pop('rgb_camera.color_profile') == '640x480x15'
    assert written == given


def test_the_d435_launch_takes_the_core_launchs_arguments():
    description = get_launch_description_from_python_launch_file(D435)
    arguments = {entity.name: entity for entity in description.entities
                 if isinstance(entity, DeclareLaunchArgument)}
    core = get_launch_description_from_python_launch_file(
        os.path.join(get_package_share_directory('cho_realsense'), 'launch', 'd435.launch.py'))
    core_arguments = {entity.name for entity in core.entities
                      if isinstance(entity, DeclareLaunchArgument)}
    assert core_arguments <= set(arguments)
    assert ''.join(part.text for part in arguments['stream'].default_value) == 'both'
    assert ''.join(part.text for part in arguments['color_profile'].default_value) == '640x480x15'


@pytest.mark.parametrize('stream, core_stream', [('both', 'infra1'), ('infra1', 'infra1'),
                                                 ('color', 'color')])
def test_the_core_d435_launch_is_given_one_of_its_own_streams(stream, core_stream):
    d435 = _module(D435, 'cell_d435_include')
    context = LaunchContext()
    side = os.path.join(CORE_REAL, 'd435_side.yaml')
    for name, value in {'stream': stream, 'color_profile': '640x480x15', 'profile': '848x480x30',
                        'serial_no': '_332322072253', 'camera_name': 'side_2',
                        'camera_namespace': 'side_2', 'camera_config': side}.items():
        context.launch_configurations[name] = value
    include, = d435._launch_setup(context)
    given = {name: value for name, value in include.launch_arguments}
    assert given['stream'] == core_stream
    assert (given['serial_no'], given['camera_name']) == ('_332322072253', 'side_2')
    if stream == 'both':
        assert given['camera_config'] != side
        os.unlink(given['camera_config'])
    else:
        assert given['camera_config'] == side


def test_a_stream_the_cell_does_not_know_is_refused():
    d435 = _module(D435, 'cell_d435_refuse')
    context = LaunchContext()
    context.launch_configurations.update({'stream': 'depth', 'camera_config': 'x'})
    with pytest.raises(RuntimeError, match="stream must be one of"):
        d435._launch_setup(context)
