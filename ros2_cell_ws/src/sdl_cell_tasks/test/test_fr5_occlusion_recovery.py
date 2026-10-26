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

"""The recovery tree's shape, and the three files it makes agree.

The tree is short; what is worth pinning is the order of its leaves and the
joins between the sweep table, the object table and the camera table, because
each of those is a name typed in two places and a mismatch shows up as an arm
that drives four waypoints and then times out.
"""

import os

from ament_index_python.packages import get_package_share_directory
import py_trees
import pytest
import yaml

from cho_task_manager.behaviors.action import JointSpaceActionBehavior
from cho_task_manager.behaviors.topic import PoseTargetBehavior
from cho_task_manager.tasks import available_tasks, build_task_tree
from cho_task_manager.testing import goal_problems, registry_tree
from cho_task_manager.utils.controller_names import arm_joint_names, load_robot_config
from sdl_cell_tasks.behaviors.action import OcclusionSweepBehavior, SinglePassSweepBehavior
from sdl_cell_tasks.tasks.fr5 import occlusion_recovery
from sdl_cell_tasks.tasks.fr5.common import VESSELS

TASK = 'sdl_occlusion_recovery'
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_PARAMS = os.path.join(PACKAGE_ROOT, 'config', TASK + '.yaml')


def _sweep_table(name='fr5_bench.yaml'):
    try:
        path = os.path.join(get_package_share_directory('sdl_cell_tasks'), 'config', 'sweep', name)
        if os.path.exists(path):
            return path
    except LookupError:
        pass
    return os.path.join(PACKAGE_ROOT, 'config', 'sweep', name)


def _config(**task_params):
    config = load_robot_config('fr5')
    config['task_params'] = {'sweep_config': _sweep_table(), **task_params}
    return config


def _tree(**task_params):
    return build_task_tree(TASK, _config(**task_params))


def _leaves(root):
    return [node for node in root.iterate() if not node.children]


MODES = [occlusion_recovery.SWEEP_MODE_SINGLE_PASS, occlusion_recovery.SWEEP_MODE_PER_OBJECT]


def _sweep_leaves(root):
    return [leaf for leaf in _leaves(root)
            if isinstance(leaf, (OcclusionSweepBehavior, SinglePassSweepBehavior))]


def _sweeps(leaf):
    """The SweepSpecs a sweep leaf of either kind judges by."""
    if isinstance(leaf, SinglePassSweepBehavior):
        return [target.sweep for target in leaf.targets]
    return [leaf.sweep]


def test_the_task_is_registered_for_the_fr5_and_only_the_fr5():
    assert TASK in available_tasks('fr5')
    for other in ('franka', 'ur5e', 'openarm'):
        assert TASK not in available_tasks(other)


def test_it_refuses_to_build_without_a_sweep_table():
    # Where to look is a bench's knowledge. Defaulting to somebody's bench
    # would drive this arm at the last bench's beaker.
    with pytest.raises(ValueError, match='sweep_config'):
        registry_tree(TASK, 'fr5')


def test_a_sweep_table_that_is_not_there_is_named_rather_than_ignored():
    with pytest.raises(ValueError, match='no such file'):
        _tree(sweep_config='/nowhere/fr5_bench.yaml')


def test_a_relative_sweep_table_is_one_this_package_installs():
    # A task_params file is YAML, with no $(find-pkg-share): the shipped tables
    # are named relative to the package's share directory.
    tree = _tree(sweep_config='config/sweep/fr5_mujoco_cell.yaml')
    assert tree.recovery_summary['vessels'] == [v.name for v in VESSELS]


def test_a_task_param_it_does_not_take_is_refused():
    with pytest.raises(ValueError, match="does not take \\['sweep_confg'\\]"):
        registry_tree(TASK, 'fr5', task_params={'sweep_confg': _sweep_table()})


def test_the_example_params_build_and_send_registry_goals():
    with open(EXAMPLE_PARAMS, encoding='utf-8') as stream:
        params = yaml.safe_load(stream)
    tree, config = registry_tree(TASK, 'fr5', task_params=params)
    assert goal_problems(tree, config) == []


def test_every_vessel_is_recovered_before_it_is_latched():
    # THE ORDERING THIS TREE EXISTS FOR. A PoseTargetBehavior that cannot see
    # its object fails by timing out, so the recovery has to come first rather
    # than as a fallback behind a failed detection -- otherwise every occluded
    # run pays the detection timeout before it may begin to recover, and
    # reports 'no message within Ns' for three different faults.
    leaves = _leaves(_tree(sweep_mode='per_object'))
    for vessel in VESSELS:
        sweep = next(i for i, leaf in enumerate(leaves)
                     if isinstance(leaf, OcclusionSweepBehavior)
                     and leaf.sweep.object == vessel.name)
        latch = next(i for i, leaf in enumerate(leaves)
                     if isinstance(leaf, PoseTargetBehavior)
                     and leaf.record_as == vessel.key)
        assert sweep < latch


def test_the_arm_returns_only_after_the_pose_has_been_latched():
    # THE CONTRACT THIS TREE TURNS ON. A recovered pose lives in the pose
    # node's aggregation window and nowhere else, so it expires within
    # window_sec of the arm leaving the viewpoint. Returning before the latch
    # would throw away the thing the sweep went to get.
    leaves = _leaves(_tree(sweep_mode='per_object'))
    for vessel in VESSELS:
        latch = next(i for i, leaf in enumerate(leaves)
                     if isinstance(leaf, PoseTargetBehavior)
                     and leaf.record_as == vessel.key)
        back = next(i for i, leaf in enumerate(leaves)
                    if isinstance(leaf, JointSpaceActionBehavior)
                    and leaf.name == 'Return_%s' % vessel.name.capitalize())
        assert latch < back


@pytest.mark.parametrize('mode', MODES)
def test_the_return_is_slow_enough_for_the_cells_rate_ceiling(mode):
    # The far end of a raster is about 3 rad away from home, and this
    # cell commissions at 0.394 rad/s (cho_moveit_fr5/config/joint_limits.yaml).
    returns = [leaf for leaf in _leaves(_tree(sweep_mode=mode))
               if isinstance(leaf, JointSpaceActionBehavior)
               and leaf.name.startswith('Return_')]
    assert returns
    for leaf in returns:
        assert leaf.duration >= occlusion_recovery.RETURN_DURATION_SEC


@pytest.mark.parametrize('mode', MODES)
def test_the_sweeps_search_wide_before_they_look_close(mode):
    # Other glassware stands on this bench, so the arm stays high and sweeps
    # laterally for a different line of sight, and only drops for pixels when
    # the decode is still not good enough. The order of the waypoints IS that
    # policy, and the score requirement is what stops it early.
    leaves = _sweep_leaves(_tree(sweep_mode=mode))
    assert leaves
    for leaf in leaves:
        for sweep in _sweeps(leaf):
            assert sweep.waypoints[0].name.startswith('survey')
            assert sweep.waypoints[-1].name.startswith('close')
            assert sweep.min_decision_margin > 0.0


@pytest.mark.parametrize('mode', MODES)
def test_a_sweep_can_run_its_whole_raster_inside_its_own_ceiling(mode):
    # The ceiling has to cover the worst case, which is the case it exists for:
    # an object that genuinely cannot be found, with every waypoint driven.
    # A single pass drives the raster once, however many objects are on it.
    leaves = _sweep_leaves(_tree(sweep_mode=mode))
    assert leaves
    for leaf in leaves:
        sweep = leaf.raster if isinstance(leaf, SinglePassSweepBehavior) else leaf.sweep
        worst = sum(point.duration + sweep.dwell_sec for point in sweep.waypoints)
        assert sweep.timeout_sec > worst, (leaf.name, worst, sweep.timeout_sec)


def test_the_latch_records_the_key_and_frame_a_motion_leaf_would_read():
    config = _config()
    latches = [leaf for leaf in _leaves(_tree(sweep_mode='per_object'))
               if isinstance(leaf, PoseTargetBehavior)]
    assert {leaf.record_as for leaf in latches} == {v.key for v in VESSELS}
    assert {leaf.topic for leaf in latches} == {v.topic for v in VESSELS}
    # Nothing in the pipeline transforms frames, so a pose in any other frame
    # would be driven to as if it were a base-frame one.
    for leaf in latches:
        assert leaf.required_frame == config['arm_base_link']


def test_the_arm_is_homed_at_both_ends():
    # It must not end a perception run parked with the camera 300 mm off the
    # glass, and it must not start one from wherever the last task stopped.
    root = _tree()
    names = [node.name for node in root.iterate()]
    assert any(name.startswith('1_Initialize') for name in names)
    assert any(name.endswith('_Finish') for name in names)


def test_a_failure_still_reaches_the_hold_controller():
    # guarded_mission's abort branch. A sweep that fails leaves the arm
    # somewhere over the bench on a controller nobody is streaming at.
    names = [node.name for node in _tree().iterate()]
    assert 'Mission_Or_Safe_Abort' in names
    assert any('Abort_Switch_To_Hold' in name for name in names)
    assert 'Abort_Verify_Hold_Active' in names


@pytest.mark.parametrize('mode', MODES)
def test_the_sweeps_drive_the_controller_the_rest_of_the_tree_uses(mode):
    # A sweep sent to a controller the bringup did not load is a goal that is
    # never accepted, after the switch has already happened.
    config = _config()
    leaves = _sweep_leaves(_tree(sweep_mode=mode))
    assert leaves
    for leaf in leaves:
        assert config['joint_space'] in leaf.action_name


def test_the_sweep_table_and_the_object_table_name_the_same_vessels():
    # The join that decides the topic. A sweep for 'beaker' and an object
    # table calling it 'beaker_100ml' build a tree that recovers nothing.
    with open(_sweep_table(), encoding='utf-8') as stream:
        swept = {entry['object'] for entry in yaml.safe_load(stream)['sweeps']}
    assert swept == {vessel.name for vessel in VESSELS}


def test_the_summary_says_what_was_covered_and_what_was_not():
    # An object with no sweep is simply not located by this tree, which is a
    # decision worth reading in a log rather than discovering.
    root = _tree()
    assert root.recovery_summary['vessels'] == [v.name for v in VESSELS]
    assert root.recovery_summary['uncovered'] == []
    assert set(root.recovery_summary['recovery_cameras'].values()) == {'wrist'}


def test_the_tree_is_a_one_shot_like_every_other_mission():
    assert isinstance(_tree(), py_trees.decorators.OneShot)


# ------------------------------------------------------------ single pass

def test_the_default_looks_for_every_vessel_in_one_pass():
    # Both vessels are planning targets on this bench, so a return home between
    # them bought nothing and the second sweep began again from waypoint 1.
    root = _tree()
    sweeps = _sweep_leaves(root)
    assert len(sweeps) == 1 and isinstance(sweeps[0], SinglePassSweepBehavior)
    assert [target.name for target in sweeps[0].targets] == [v.name for v in VESSELS]
    assert not [leaf for leaf in _leaves(root) if isinstance(leaf, PoseTargetBehavior)]
    assert root.recovery_summary['sweep_mode'] == 'single_pass'


def test_the_single_pass_latches_where_a_motion_leaf_would_read():
    # The same keys, topics and frame the per-object PoseTargetBehaviors use,
    # so nothing downstream can tell which mode located the vessels.
    config = _config()
    leaf = _sweep_leaves(_tree())[0]
    assert {t.record_as for t in leaf.targets} == {v.key for v in VESSELS}
    assert {t.topic for t in leaf.targets} == {v.topic for v in VESSELS}
    assert leaf.required_frame == config['arm_base_link']


def test_the_single_pass_comes_home_once_and_only_after_it_has_latched():
    leaves = _leaves(_tree())
    names = [leaf.name for leaf in leaves]
    returns = [name for name in names if name.startswith('Return_')]
    assert returns == ['Return_Vessels']
    assert names.index('Recover_Vessels') < names.index('Return_Vessels')


def test_the_finish_step_counts_on_from_the_locate_blocks():
    names = [node.name for node in _tree().iterate()]
    assert '2_Locate_Vessels' in names and '3_Finish' in names
    names = [node.name for node in _tree(sweep_mode='per_object').iterate()]
    assert '4_Finish' in names


def test_an_unknown_sweep_mode_is_refused():
    with pytest.raises(ValueError, match='sweep_mode'):
        _tree(sweep_mode='twice')


def test_a_table_with_a_raster_per_object_points_at_per_object(tmp_path):
    with open(_sweep_table(), encoding='utf-8') as stream:
        table = yaml.safe_load(stream)
    table['sweeps'][1]['waypoints'] = table['sweeps'][1]['waypoints'][:-1]
    path = tmp_path / 'split.yaml'
    path.write_text(yaml.safe_dump(table))
    with pytest.raises(ValueError, match='sweep_mode: per_object'):
        _tree(sweep_config=str(path))
    _tree(sweep_config=str(path), sweep_mode='per_object')


@pytest.mark.parametrize('mode', MODES)
def test_every_goal_names_the_fr5_joints(mode):
    # The sweep waypoints and the returns go to the joint server named, so it
    # matches the positions by name (cho_interfaces/CONTRACT.md). The sweep
    # leaves are this package's, so goal_problems() does not see them.
    config = _config(sweep_mode=mode)
    tree = build_task_tree(TASK, config)
    joints = arm_joint_names(config)
    assert joints == ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
    for leaf in _sweep_leaves(tree):
        assert leaf.joint_names == joints
    returns = [leaf for leaf in _leaves(tree) if isinstance(leaf, JointSpaceActionBehavior)]
    assert returns
    assert goal_problems(tree, config) == []


# ------------------------------------------- a bench's own object table (vessel_table)

def _config_file(*parts):
    return os.path.join(PACKAGE_ROOT, 'config', *parts)


def test_a_vessel_table_names_the_vessels_and_their_topics():
    # The two-beaker cell: the recovery runs on the pose node already started on
    # pour_beakers.yaml, so it has to look for THAT table's names on THAT table's
    # topics -- beaker1 on held_beaker, which the pouring controller reads.
    tree = _tree(vessel_table='config/perception/pour_beakers.yaml',
                 sweep_config='config/sweep/fr5_bench_beakers.yaml')
    assert tree.recovery_summary['vessels'] == ['beaker1', 'beaker2']
    assert tree.recovery_summary['uncovered'] == []
    recover = _sweep_leaves(tree)[0]
    assert sorted(target.topic for target in recover.targets) == [
        '/perception/object_pose/beaker2', '/perception/object_pose/held_beaker']
    assert sorted(target.record_as for target in recover.targets) == [
        'beaker1_pose', 'beaker2_pose']


def test_the_two_beaker_sweep_is_the_bench_sweep_under_other_names():
    with open(_sweep_table(), encoding='utf-8') as handle:
        bench = yaml.safe_load(handle)
    with open(_config_file('sweep', 'fr5_bench_beakers.yaml'), encoding='utf-8') as handle:
        beakers = yaml.safe_load(handle)
    assert [s['object'] for s in beakers['sweeps']] == ['beaker1', 'beaker2']
    assert beakers['defaults'] == bench['defaults']
    assert [s['waypoints'] for s in beakers['sweeps']] == [s['waypoints'] for s in bench['sweeps']]


def test_a_vessel_table_that_is_not_there_is_refused():
    with pytest.raises(ValueError, match='vessel_table'):
        _tree(vessel_table='/nonexistent/table.yaml')


def test_a_vessel_table_whose_objects_the_sweep_does_not_cover_is_refused():
    # The bench's sweep table names beaker and flask; pour_beakers.yaml has neither.
    with pytest.raises(ValueError, match='nothing to recover'):
        _tree(vessel_table='config/perception/pour_beakers.yaml')


def test_without_a_vessel_table_the_bench_vessels_are_tracked():
    assert _tree().recovery_summary['vessels'] == [vessel.name for vessel in VESSELS]
    assert _tree(vessel_table=None).recovery_summary['vessels'] == [v.name for v in VESSELS]


# ------------------------------------------------------------ the first home

def _tree_via(home_via, **task_params):
    config = _config(**task_params)
    config['home_via'] = home_via
    return build_task_tree(TASK, config)


def test_the_first_home_is_planned_by_moveit_by_default():
    # It starts from wherever the last task left the arm (author, 2026-09-30).
    leaves = [leaf.name for leaf in _leaves(_tree())]
    first_home = leaves.index('Go_Home_MoveIt')
    # The bridge's readiness gate is waited out after the switch to its controller...
    assert leaves[first_home - 1] == 'Wait_For_MoveIt_Gate'
    # ...and the arm is handed back to the hold controller before the sweep.
    assert leaves[first_home + 1].startswith('Park_On_')
    assert _tree().recovery_summary['home_via'] == 'moveit'


def test_home_via_direct_keeps_the_straight_interpolation():
    tree = _tree_via('direct')
    leaves = [leaf.name for leaf in _leaves(tree)]
    assert 'Go_Home_MoveIt' not in leaves
    assert 'Go_Home' in leaves
    assert tree.recovery_summary['home_via'] == 'direct'


def test_an_unknown_home_via_is_refused():
    with pytest.raises(ValueError, match='home_via'):
        _tree_via('teleport')
