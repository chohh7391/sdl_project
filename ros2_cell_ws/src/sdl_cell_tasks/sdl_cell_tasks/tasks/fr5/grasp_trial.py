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

"""sdl_grasp_trial: one grasp of beaker1 from a two-beaker plan, and nothing else.

What a perception error does to a grasp is measured here without the pour on
top of it. ``transfer_plan`` names a plan as ``sdl_beaker_transfer`` takes it
(its .csv, .meta.json and _cell.yaml, the cell gated at build time); only its
first grasp is used::

    1_Initialize   to the plan's start pose (home_via; MoveIt by default)
    2_Handover     onto the trajectory controller
    3_Grasp        the plan's reach -> close the jaws on beaker1 -> settle ->
                   check the jaws stopped on it -> open -> settle -> back off
                   (the reach backwards)
    4_Finish       home to the plan's start pose, parked on the hold controller

The back-off is the plan's own reach run backwards, so nothing moves anywhere
the plan did not already clear. The check logs ``grasp HELD`` or ``grasp
MISSED`` (``behaviors/topic/gripper_hold.py``); a missed grasp still opens and
backs off. ``grasp_held_from`` is the finger travel [m] from which the jaws
count as holding something.

The beaker is not lifted. The move after the plan's grasp is no lift -- it
slides the jaws ~50 mm sideways before they rise at all, 30 mm up only 43-45 %
of the way in, 170 mm across (2026-10-01 plans) -- and a straight lift needs a
Cartesian move nothing on the real cell has been checked with yet.

Jaws that centre what they grip push a beaker that was not where perception
put it onto their centre line, so measuring it again once it is released
(plan_remote.sh --measure-only) shows how far the perceived position was off
across the jaws.

    ros2 launch cho_task_manager run_task_manager.launch.py robot_type:=fr5 \\
        task:=sdl_grasp_trial task_params:=<a file like config/sdl_grasp_trial.yaml>
"""

import py_trees

from cho_task_manager.behaviors.action import (
    FollowJointTrajectoryBehavior,
    GripperActionBehavior,
)
from cho_task_manager.behaviors.wait import WaitBehavior
from cho_task_manager.subtrees import guarded_mission
from cho_task_manager.utils.robot_description import DescriptionPositionLimits
from cho_task_manager.utils.trajectory_recording import Segment
from sdl_cell_tasks.behaviors.topic import GripperHoldCheckBehavior
from sdl_cell_tasks.tasks.fr5.beaker_transfer import (
    plan_files,
    planned_node_keys,
    TRANSFER_PLAN_KEY,
)
from sdl_cell_tasks.tasks.fr5.replay import (
    handover_sequence,
    home_block,
    replay_plan,
    replayed_time_scale,
)
from sdl_cell_tasks.utils.params import number, task_params

TASK_NAME = 'sdl_grasp_trial'

#: Finger travel from which the jaws count as holding something [m]: on the
#: real cell's 50 mm beaker they stop at 0.024, on nothing they close to 0.
DEFAULT_HELD_FROM_M = 0.005

#: What task_params may set, and its value when it does not.
DEFAULTS = {
    TRANSFER_PLAN_KEY: None,
    'grasp_held_from': DEFAULT_HELD_FROM_M,
}


def first_grasp(segments):
    """``(reach, close, carry)``: the move up to the first grasp, the grasp, the move after."""
    for index, segment in enumerate(segments):
        if segment.kind == 'gripper' and segment.grasp:
            reach = segments[index - 1] if index > 0 else None
            carry = segments[index + 1] if index + 1 < len(segments) else None
            if reach is None or reach.kind != 'move' or carry is None or carry.kind != 'move':
                raise ValueError('the plan does not reach, grasp and then move: %r'
                                 % [repr(s) for s in segments[:index + 2]])
            return reach, segment, carry
    raise ValueError('the plan never closes the jaws on anything: %r'
                     % [repr(s) for s in segments])


def backwards(segment, label):
    """The same move run the other way, on a clock that starts at zero."""
    end = segment.times[-1]
    return Segment('move', times=[end - t for t in reversed(segment.times)],
                   positions=[list(p) for p in reversed(segment.positions)],
                   operations=(label,), source_t0=segment.source_t0)


def create_sdl_grasp_trial_tree(robot_config) -> py_trees.behaviour.Behaviour:
    """Reach, grasp, check, release and back off, as the plan's grasp."""
    params = task_params(robot_config, TASK_NAME, DEFAULTS)
    csv_path, meta_path, cell_path = plan_files(params[TRANSFER_PLAN_KEY], TASK_NAME)
    planned_node_keys(robot_config, TASK_NAME)
    held_from = number(params, 'grasp_held_from', TASK_NAME)
    if not held_from > 0.0:
        raise ValueError('%s: grasp_held_from must be a positive finger travel [m]; got %r'
                         % (TASK_NAME, params['grasp_held_from']))
    gripper = robot_config.get('gripper')
    if gripper is None:
        raise ValueError("%s closes the jaws, and no gripper controller was given "
                         "(robot_config['gripper'])" % TASK_NAME)

    config = dict(robot_config)
    config.update(replay_trajectory=csv_path, replay_meta=meta_path, replay_layout=cell_path)
    # The plan's recording, layout gate, home_via, speed and velocity ceiling,
    # checked as every cell replay checks them.
    plan = replay_plan(config, TASK_NAME)

    reach, _close, _carry = first_grasp(plan.segments)
    retreat = backwards(reach, 'retreat')
    # The retreat is the reach reversed, so the reach alone sets the clock.
    ceiling_scale = replayed_time_scale(plan, [reach])[0]
    time_scale = max(1.0 / plan.speed_scale, ceiling_scale)
    position_limits = DescriptionPositionLimits(plan.joint_names)
    velocity_limits = {joint: value * plan.scaling for joint, value in plan.limits.items()}

    def move(name, segment):
        return FollowJointTrajectoryBehavior(
            name=name, controller=plan.controller, joint_names=plan.joint_names,
            times=segment.times, positions=segment.positions, time_scale=time_scale,
            position_limits=position_limits, velocity_limits=velocity_limits)

    grasp = py_trees.composites.Sequence(name='3_Grasp', memory=True)
    grasp.add_children([
        move('0_Reach', reach),
        GripperActionBehavior(name='1_Gripper_Close', grasp=True, controller_name=gripper),
        WaitBehavior(name='1_Gripper_Settle', duration_sec=plan.gripper_settle_sec),
        # Logged either way; a missed grasp is a trial's result, not a reason to
        # leave the jaws shut over the bench.
        py_trees.decorators.FailureIsSuccess(
            name='2_Grasp_Result_Logged',
            child=GripperHoldCheckBehavior(name='2_Grasp_Check', min_position=held_from)),
        GripperActionBehavior(name='3_Gripper_Open', grasp=False, controller_name=gripper),
        WaitBehavior(name='3_Gripper_Settle', duration_sec=plan.gripper_settle_sec),
        move('4_Retreat', retreat),
    ])

    mission = py_trees.composites.Sequence(name='SDL_Grasp_Trial_Sequence', memory=True)
    mission.add_children([
        home_block(config, plan, name='1_Initialize', suffix=''),
        handover_sequence(config, plan, '2_Handover'),
        grasp,
        home_block(config, plan, name='4_Finish', suffix='_Final', park=True),
    ])
    root = guarded_mission(mission, config, plan.control_mode, name='SDL_Grasp_Trial_Root')
    root.grasp_summary = {
        'plan': csv_path[:-len('.csv')],
        'time_scale': time_scale,
        'reach_s': reach.duration,
        'reach_operation': reach.operation,
        'held_from_m': held_from,
        'home_via': plan.home_via,
        'layout_checked': plan.layout_checked,
    }
    return root


#: The registration setup.py points at: {robot_type: tree builder}.
TASK = {'fr5': create_sdl_grasp_trial_tree}

__all__ = [
    'DEFAULTS',
    'DEFAULT_HELD_FROM_M',
    'TASK',
    'TASK_NAME',
    'backwards',
    'create_sdl_grasp_trial_tree',
    'first_grasp',
]
