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

"""The package's tasks, as cho_task_manager finds them, and the core rules its trees keep.

Run against the installed package (colcon test): the tasks are found through
the entry points setup.py declares, not imported by hand.
"""

import ast
from pathlib import Path

import pytest
import yaml

from cho_task_manager.tasks import available_tasks, package_tasks
from cho_task_manager.testing import goal_problems, registry_tree

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'sdl_cell_tasks'
TASKS = ['sdl_occlusion_recovery', 'sdl_occlusion_replay', 'sdl_perceived_replay',
         'sdl_trajectory_replay', 'sdl_vessel_detect']

#: What each example task_params file does when built from the registry alone,
#: as the node builds it with no other parameter: None builds, a string is the
#: refusal (both replays need the cell's own layout, replay_layout:=, for the
#: cell's recording, which assumes the bench's layout).
FROM_THE_REGISTRY_ALONE = {
    'sdl_occlusion_recovery': None,
    'sdl_occlusion_replay': "needs 'replay_layout'",
    'sdl_perceived_replay': None,
    'sdl_trajectory_replay': "needs 'replay_layout'",
    'sdl_vessel_detect': None,
}


def _sources():
    return sorted(PACKAGE.rglob('*.py'))


def _example(task):
    with open(ROOT / 'config' / ('%s.yaml' % task), encoding='utf-8') as stream:
        return yaml.safe_load(stream) or {}


def test_the_package_registers_its_five_tasks():
    assert package_tasks('sdl_cell_tasks') == TASKS


@pytest.mark.parametrize('task', TASKS)
def test_each_task_is_the_fr5s_only(task):
    assert task in available_tasks('fr5')
    for robot in ('franka', 'ur5e', 'openarm'):
        assert task not in available_tasks(robot)


@pytest.mark.parametrize('task', TASKS)
def test_each_example_params_file_builds_as_the_node_would_build_it(task):
    refusal = FROM_THE_REGISTRY_ALONE[task]
    if refusal is not None:
        with pytest.raises(ValueError, match=refusal):
            registry_tree(task, 'fr5', task_params=_example(task))
        return
    tree, config = registry_tree(task, 'fr5', task_params=_example(task))
    assert goal_problems(tree, config) == []


def test_no_core_private_name_is_imported():
    # The core's public API only; a private helper the cell needs is copied,
    # saying where from.
    offenders = []
    for path in _sources():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('cho_'):
                offenders += [f'{path.relative_to(ROOT)}:{node.lineno} {node.module}.{alias.name}'
                              for alias in node.names if alias.name.startswith('_')]
                if any(part.startswith('_') for part in node.module.split('.')):
                    offenders.append(f'{path.relative_to(ROOT)}:{node.lineno} {node.module}')
    assert offenders == []


def test_no_wait_counts_wall_time():
    # WaitBehavior runs on the node clock; py_trees' Timer counts wall time.
    offenders = [str(path.relative_to(ROOT)) for path in _sources()
                 if 'py_trees.timers' in path.read_text()]
    assert offenders == []


def test_no_tree_spells_out_a_home_pose():
    # The constants the registry replaced in the core must not come back here.
    for name in ('FRANKA_HOME_POSITION', 'UR5E_HOME_POSITION', 'HOME_POSE_KEY',
                 'FR5_POSITION_LIMITS', 'EXCLUSIVE_ARM_CONTROLLERS'):
        offenders = [str(path.relative_to(ROOT)) for path in _sources() if name in path.read_text()]
        assert offenders == [], name


def test_nothing_reads_app_data_from_the_cores_share():
    # The cell's tables, meshes and layouts are installed under share/sdl_cell_tasks.
    offenders = [str(path.relative_to(ROOT)) for path in _sources()
                 if "get_package_share_directory('cho_task_manager')" in path.read_text()
                 and path.name != 'replay.py']   # the cell's recording stays the core's
    assert offenders == []
