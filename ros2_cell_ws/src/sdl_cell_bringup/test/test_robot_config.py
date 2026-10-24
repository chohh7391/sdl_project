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

"""The cell's registry is the core's, plus the arm controllers the cell loads.

Run against the install (colcon test sources it), so CHO_ROBOT_CONFIG_DIR is
what the environment hook set.
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import yaml

from cho_robot_config import load_robot_config
from cho_task_manager.utils.controller_names import (
    exclusive_arm_controllers,
    load_robot_config as task_manager_config,
)

SHARE = Path(get_package_share_directory('sdl_cell_bringup'))
ADDITIONS = yaml.safe_load((SHARE / 'config' / 'robot_config_additions.yaml').read_text())


def _cell_dir():
    return SHARE / 'robot_config'


def _core_dir():
    return Path(get_package_share_directory('cho_robot_config')) / 'config'


def test_sourcing_the_workspace_points_the_registry_at_the_cell():
    assert Path(os.environ['CHO_ROBOT_CONFIG_DIR']).resolve() == _cell_dir().resolve()


def test_the_cell_registry_holds_every_core_robot():
    assert sorted(p.name for p in _cell_dir().glob('*.yaml')) == \
        sorted(p.name for p in _core_dir().glob('*.yaml'))


def test_only_the_additions_differ_from_the_core():
    for core in _core_dir().glob('*.yaml'):
        cell = yaml.safe_load((_cell_dir() / core.name).read_text())
        expected = yaml.safe_load(core.read_text())
        for robot_type, entry in ADDITIONS.items():
            if expected.get('robot_type') == robot_type:
                expected['controllers']['additional_arm'] = entry['additional_arm']
        assert cell == expected, core.name


def test_every_exclusive_switch_takes_the_added_controllers_down():
    for robot_type, entry in ADDITIONS.items():
        assert load_robot_config(robot_type)['controllers']['additional_arm'] == entry['additional_arm']
        exclusive = exclusive_arm_controllers(task_manager_config(robot_type))
        for name in entry['additional_arm']:
            assert name in exclusive, (robot_type, name, exclusive)
