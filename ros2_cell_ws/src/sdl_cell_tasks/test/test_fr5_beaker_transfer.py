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

"""The two-beaker transfer, with every input but what is poured taken from its plan.

No ROS graph: trees are built, not ticked.
"""

import json

import pytest

from cho_task_manager.tasks import available_tasks, build_task_tree
from cho_task_manager.testing import goal_problems
from cho_task_manager.utils.controller_names import load_robot_config
from sdl_cell_tasks.behaviors.action import PourActionBehavior
from sdl_cell_tasks.behaviors.service import HeldPoseServiceBehavior
from sdl_cell_tasks.behaviors.topic import GraspMarkerSampleBehavior
from sdl_cell_tasks.tasks.fr5.beaker_transfer import plan_files

TASK = 'sdl_beaker_transfer'
HEADER = 't_s,j1,j2,j3,j4,j5,j6,operation\n'
LAYOUT = {'beaker1': {'xy': [0.4099, -0.3014]}, 'beaker2': {'xy': [0.5178, 0.2969]}}
RIM = [0.5178, 0.2969, 0.1486]
MARKER = [0.4099, -0.3014, 0.0645]


def _rows():
    """Pick, carry, pour (j6 out and back), place: 40 rows at 0.1 s."""
    rows = []
    for i in range(40):
        j2, j6 = 0.1, 0.0
        if i < 10:
            op, j2 = 'Pick', 0.01 * i
        elif i < 20:
            op = 'Move_to_Surface'
        elif i < 30:
            op = 'pouring'
            j6 = -1.2 * (1.0 - abs(i - 24.5) / 4.5) if 20 < i < 29 else 0.0
        else:
            op, j2 = 'Place', 0.1 - 0.005 * (i - 29)
        rows.append(([0.1 * i, 0.0, j2, 0.0, 0.0, 0.0, j6], op))
    return rows


def _plan(tmp_path, pour=True, put_back='start'):
    """A plan as plan_remote.sh delivers it: <base>.csv, .meta.json and _cell.yaml."""
    base = tmp_path / 'traj_realcell_seed0'
    rows = _rows()
    (tmp_path / 'traj_realcell_seed0.csv').write_text(HEADER + ''.join(
        '%s,%s\n' % (','.join('%.6f' % v for v in row), op) for row, op in rows))
    (tmp_path / 'traj_realcell_seed0.meta.json').write_text(json.dumps({
        'seed': '0', 'tool': 'fr5_ag95',
        'joint_names': ['j1', 'j2', 'j3', 'j4', 'j5', 'j6'],
        'waypoints': len(rows), 'home_arm_rad': list(rows[0][0][1:]),
        'gripper_events': [{'t_s': 0.95, 'event': 'close'}, {'t_s': 3.45, 'event': 'open'}],
        'layout_the_trajectory_assumes': LAYOUT,
    }))
    cell = 'layout:\n' + ''.join(
        '  %s:\n    xy: [%r, %r]\n' % (name, e['xy'][0], e['xy'][1]) for name, e in LAYOUT.items())
    if pour:
        cell += ('pour:\n  receiver_rim: [%r, %r, %r]\n  grasp_marker: [%r, %r, %r]\n'
                 '  put_back: %s\n' % (tuple(RIM) + tuple(MARKER) + (put_back,)))
    (tmp_path / 'traj_realcell_seed0_cell.yaml').write_text(cell)
    return str(base)


def _config(plan, node=None, **task_params):
    config = load_robot_config('fr5')
    config.update({'home_via': 'direct', **(node or {})})
    config['task_params'] = {'transfer_plan': plan, 'replay_pour_material': 'liquid',
                             'replay_pour_grams': 10.0, **task_params}
    return config


def _tree(tmp_path, node=None, **task_params):
    return build_task_tree(TASK, _config(_plan(tmp_path), node, **task_params))


def test_the_task_is_registered_for_fr5_only():
    assert TASK in available_tasks('fr5')
    for other in ('franka', 'ur5e', 'openarm'):
        assert TASK not in available_tasks(other)


def test_every_goal_names_the_registry_frame_and_joints(tmp_path):
    config = _config(_plan(tmp_path))
    assert goal_problems(build_task_tree(TASK, config), config) == []


def test_the_pour_goes_where_the_plan_was_made_to_pour(tmp_path):
    root = _tree(tmp_path)
    pours = [node for node in root.iterate() if isinstance(node, PourActionBehavior)]
    assert len(pours) == 1
    assert pours[0].receiver_rim_center == pytest.approx(RIM)
    markers = [node for node in root.iterate() if isinstance(node, GraspMarkerSampleBehavior)]
    assert len(markers) == 1 and markers[0].marker == pytest.approx(MARKER)
    assert root.transfer_summary['put_back'] == 'start'
    assert root.replay_summary['layout_checked'] is True
    assert root.name == 'SDL_Beaker_Transfer_Root'


def test_beaker1_is_put_back_where_it_was_taken_from(tmp_path):
    root = _tree(tmp_path)
    verbs = [node.verb for node in root.iterate() if isinstance(node, HeldPoseServiceBehavior)]
    # released when the jaws close, restored when they open over the riser
    assert verbs == ['release', 'restore']


def test_put_back_elsewhere_resumes_perception_instead(tmp_path):
    root = build_task_tree(TASK, _config(_plan(tmp_path, put_back='elsewhere')))
    verbs = [node.verb for node in root.iterate() if isinstance(node, HeldPoseServiceBehavior)]
    assert verbs == ['release', 'resume']


def test_the_plan_may_be_named_by_its_csv(tmp_path):
    plan = _plan(tmp_path)
    assert plan_files(plan + '.csv') == plan_files(plan)


def test_the_start_pose_is_planned_by_moveit_by_default(tmp_path):
    root = _tree(tmp_path, node={'home_via': ''})
    assert any(node.name == 'Go_Home_MoveIt' for node in root.iterate())
    assert any(node.name == 'Wait_For_MoveIt_Gate' for node in root.iterate())


def test_a_plan_without_its_pour_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no 'pour:' block"):
        build_task_tree(TASK, _config(_plan(tmp_path, pour=False)))


def test_a_missing_plan_file_is_refused(tmp_path):
    plan = _plan(tmp_path)
    (tmp_path / 'traj_realcell_seed0.meta.json').unlink()
    with pytest.raises(ValueError, match='missing'):
        build_task_tree(TASK, _config(plan))


def test_a_plan_is_required(tmp_path):
    config = _config(_plan(tmp_path))
    del config['task_params']['transfer_plan']
    with pytest.raises(ValueError, match='transfer_plan'):
        build_task_tree(TASK, config)


@pytest.mark.parametrize('key', ['replay_pour_grams', 'replay_pour_material'])
def test_what_and_how_much_is_poured_is_the_callers_and_required(tmp_path, key):
    config = _config(_plan(tmp_path))
    del config['task_params'][key]
    with pytest.raises(ValueError, match=key):
        build_task_tree(TASK, config)


def test_a_zero_pour_is_refused(tmp_path):
    with pytest.raises(ValueError, match='replay_pour_grams'):
        _tree(tmp_path, replay_pour_grams=0.0)


@pytest.mark.parametrize('key', ['replay_trajectory', 'replay_meta', 'replay_layout'])
def test_the_recording_the_plan_names_is_not_taken_twice(tmp_path, key):
    with pytest.raises(ValueError, match=key):
        _tree(tmp_path, node={key: '/tmp/other'})


@pytest.mark.parametrize('key, value', [
    ('replay_pour_receiver', [0.5, 0.3, 0.15]),
    ('replay_pour_grasp_marker', '0.4,-0.3,0.06'),
    ('replay_pour_put_back', 'elsewhere'),
])
def test_the_pour_target_the_plan_names_is_not_taken_twice(tmp_path, key, value):
    with pytest.raises(ValueError, match=key):
        _tree(tmp_path, **{key: value})


def test_a_task_param_it_does_not_take_is_refused(tmp_path):
    with pytest.raises(ValueError, match="does not take \\['transfer_plann'\\]"):
        _tree(tmp_path, transfer_plann='x')
