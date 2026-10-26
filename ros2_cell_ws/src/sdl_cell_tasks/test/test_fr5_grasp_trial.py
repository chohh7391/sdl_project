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

"""One grasp from a two-beaker plan, for a perception-to-grasp trial series.

No ROS graph: trees are built, not ticked.
"""

import pytest

from cho_task_manager.behaviors.action import (
    FollowJointTrajectoryBehavior,
    GripperActionBehavior,
)
from cho_task_manager.tasks import available_tasks, build_task_tree
from cho_task_manager.testing import goal_problems
from cho_task_manager.utils.controller_names import load_robot_config
from cho_task_manager.utils.trajectory_recording import Segment
from sdl_cell_tasks.behaviors.topic import GripperHoldCheckBehavior
from sdl_cell_tasks.tasks.fr5.grasp_trial import backwards, DEFAULT_HELD_FROM_M
from test_fr5_beaker_transfer import _plan

TASK = 'sdl_grasp_trial'


def _config(tmp_path, node=None, **task_params):
    config = load_robot_config('fr5')
    config.update({'home_via': 'direct', **(node or {})})
    config['task_params'] = {'transfer_plan': _plan(tmp_path), **task_params}
    return config


def _tree(tmp_path, node=None, **task_params):
    return build_task_tree(TASK, _config(tmp_path, node, **task_params))


def _grasp_names(tree):
    grasp = [n for n in tree.iterate() if n.name == '3_Grasp'][0]
    return [c.name for c in grasp.children]


def test_the_task_is_registered_for_fr5_only():
    assert TASK in available_tasks('fr5')
    for other in ('franka', 'ur5e', 'openarm'):
        assert TASK not in available_tasks(other)


def test_every_goal_names_the_registry_frame_and_joints(tmp_path):
    config = _config(tmp_path)
    assert goal_problems(build_task_tree(TASK, config), config) == []


def test_it_reaches_grasps_checks_releases_and_backs_off(tmp_path):
    assert _grasp_names(_tree(tmp_path)) == [
        '0_Reach', '1_Gripper_Close', '1_Gripper_Settle', '2_Grasp_Result_Logged',
        '3_Gripper_Open', '3_Gripper_Settle', '4_Retreat']


def test_the_way_back_is_the_plans_own_reach_reversed(tmp_path):
    tree = _tree(tmp_path)
    moves = {n.name: n for n in tree.iterate() if isinstance(n, FollowJointTrajectoryBehavior)}
    reach, retreat = moves['0_Reach'], moves['4_Retreat']
    assert retreat.positions == reach.positions[::-1]
    assert retreat.positions[-1] == reach.positions[0]
    assert retreat.time_scale == reach.time_scale
    gripper = [n for n in tree.iterate() if isinstance(n, GripperActionBehavior)]
    assert [g.grasp for g in gripper] == [True, False]
    check = [n for n in tree.iterate() if isinstance(n, GripperHoldCheckBehavior)][0]
    assert check.min_position == DEFAULT_HELD_FROM_M
    # A missed grasp is a trial's result: the jaws still open and back off.
    assert check.parent.name == '2_Grasp_Result_Logged'


def test_the_held_threshold_is_the_callers(tmp_path):
    check = [n for n in _tree(tmp_path, grasp_held_from=0.01).iterate()
             if isinstance(n, GripperHoldCheckBehavior)][0]
    assert check.min_position == 0.01
    with pytest.raises(ValueError, match='grasp_held_from'):
        _tree(tmp_path, grasp_held_from=0.0)


def test_a_move_backwards_runs_on_a_clock_from_zero():
    move = Segment('move', times=[2.0, 2.1, 2.3], positions=[[0.0] * 6, [0.1] * 6, [0.3] * 6],
                   operations=('Pick',))
    back = backwards(move, 'retreat')
    assert back.times == pytest.approx([0.0, 0.2, 0.3])
    assert back.positions == [[0.3] * 6, [0.1] * 6, [0.0] * 6]


def test_the_homes_are_planned_by_moveit_by_default(tmp_path):
    names = [n.name for n in _tree(tmp_path, node={'home_via': ''}).iterate()]
    assert 'Go_Home_MoveIt' in names and 'Go_Home_MoveIt_Final' in names


def test_a_plan_is_required(tmp_path):
    config = _config(tmp_path)
    del config['task_params']['transfer_plan']
    with pytest.raises(ValueError, match='transfer_plan'):
        build_task_tree(TASK, config)


def test_the_recording_the_plan_names_is_not_taken_twice(tmp_path):
    with pytest.raises(ValueError, match='replay_trajectory'):
        _tree(tmp_path, node={'replay_trajectory': '/tmp/other.csv'})
