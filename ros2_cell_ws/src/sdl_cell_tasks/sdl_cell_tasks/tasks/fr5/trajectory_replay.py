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

"""sdl_trajectory_replay: replay the cell's recording, its pour poured by weight.

The core's ``task:=trajectory_replay`` replays every recorded span as recorded,
the recorded pour included -- a simulated vessel tipped by a planned amount.
This is the same replay (``tasks/fr5/replay.py``: the core's checks, the core's
FR5 defaults) with one difference. Given ``replay_pour_grams``, the run of
waypoints labelled ``pouring`` is not replayed: the arm goes as far as its first
waypoint, the pouring controller measures the held vessel, pours that many grams
by weight and returns the arm to exactly that configuration, and the replay
resumes from the pour's last waypoint -- which the recording puts at the same
place. See ``utils/pour_splice.py`` for what is checked before that is allowed,
and ``subtrees/pour.py`` for the hand-over.

    1_Initialize    the recording's start pose (home_via)
    2_Handover      onto the trajectory controller, verified
    3_Replay        the recorded segments and gripper events; where the pour was,
                    [Read_Empty_Receiver] -> Switch_To_pouring_controller -> verify
                    -> Pour -> Verify_Back_At_Pour_Start -> switch back -> verify
    4_Finish        back to the start pose, parked on the hold controller

The recording, its layout and how it is replayed are the core node's
parameters, read as the core's replay reads them: ``replay_trajectory`` (empty:
the cell's recording, ``traj_level_seed2`` in cho_task_manager's installed
``config/replay/fr5/``), ``replay_meta``, ``replay_layout`` (required, since
that recording assumes the bench's layout), ``home_via``,
``replay_speed_scale``. The pour is the task's own, ``task_params:=``
(:data:`DEFAULTS`)::

    ros2 launch cho_task_manager run_task_manager.launch.py robot_type:=fr5 \\
        task:=sdl_trajectory_replay home_via:=direct replay_layout:=<the cell's layout> \\
        task_params:=$(ros2 pkg prefix --share sdl_cell_tasks)/config/sdl_trajectory_replay.yaml

The pour controller (``pour_controller``, default ``pouring_controller``) must
be in the robot registry's exclusive arm controllers, which the cell
workspace's registry makes it (sdl_cell_bringup); the tree is refused otherwise.
"""

import py_trees

from cho_task_manager.subtrees import guarded_mission
from sdl_cell_tasks.subtrees.pour import (
    grasp_measure_children,
    parse_pour_request,
    POUR_PARAM_DEFAULTS,
    pour_handover_children,
)
from sdl_cell_tasks.tasks.fr5.replay import (
    handover_sequence,
    home_block,
    replay_children,
    replay_plan,
    replay_summary,
)
from sdl_cell_tasks.utils.params import task_params
from sdl_cell_tasks.utils.pour_controller import checked_pour_controller, DEFAULT_POUR_CONTROLLER
from sdl_cell_tasks.utils.pour_splice import splice_pour

TASK_NAME = 'sdl_trajectory_replay'

#: What task_params may set, and its value when it does not. ``replay_pour_grams``
#: 0 replays the recorded pour as recorded.
DEFAULTS = {
    'pour_controller': DEFAULT_POUR_CONTROLLER,
    **POUR_PARAM_DEFAULTS,
}


def trajectory_replay_params(robot_config):
    """The task's parameters: DEFAULTS, overridden by ``robot_config['task_params']``."""
    return task_params(robot_config, TASK_NAME, DEFAULTS)


def create_sdl_trajectory_replay_tree(robot_config) -> py_trees.behaviour.Behaviour:
    """Check the recording and the pour, home, replay with the pour handed over, hand back."""
    params = trajectory_replay_params(robot_config)
    # Before anything else: every switch into and out of the pour controller,
    # and the guarded root's abort, rely on the registry knowing it.
    pour_controller = checked_pour_controller(robot_config, params['pour_controller'])
    pour_request = parse_pour_request(params)

    plan = replay_plan(robot_config, TASK_NAME)
    segments = plan.segments
    if pour_request is not None:
        # Refused here, before a node exists, like the layout: a recording whose
        # pour cannot be handed over cleanly is not replayed with a gap in it.
        segments = splice_pour(plan.recording, segments)

    def pour_children(index, segment):
        return pour_handover_children(
            robot_config, pour_request, plan.controller, pour_controller,
            prefix='%d_' % index, pour_reference=segment.reference, resume_from=segment.start)

    def grasp_children(index):
        return grasp_measure_children(robot_config, pour_request, prefix='%d_' % index)

    replay_seq = py_trees.composites.Sequence(name='3_Replay', memory=True)
    replay_seq.add_children(replay_children(
        plan, robot_config, segments,
        pour_children=pour_children if pour_request is not None else None,
        grasp_children=grasp_children if pour_request is not None else None))

    mission = py_trees.composites.Sequence(name='SDL_Trajectory_Replay_Sequence', memory=True)
    mission.add_children([
        home_block(robot_config, plan, name='1_Initialize', suffix=''),
        handover_sequence(robot_config, plan, '2_Handover'),
        replay_seq,
        # Back to the recording's own start pose, parked on the hold controller.
        home_block(robot_config, plan, name='4_Finish', suffix='_Final', park=True),
    ])

    root = guarded_mission(mission, robot_config, plan.control_mode,
                           name='SDL_Trajectory_Replay_Root')
    root.replay_summary = replay_summary(
        plan, segments,
        pour=pour_request.as_dict() if pour_request is not None else None,
        pour_controller=pour_controller)
    return root


#: The registration setup.py points at: {robot_type: tree builder}.
TASK = {'fr5': create_sdl_trajectory_replay_tree}

__all__ = [
    'DEFAULTS',
    'TASK',
    'TASK_NAME',
    'create_sdl_trajectory_replay_tree',
    'trajectory_replay_params',
]
