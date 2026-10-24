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

"""sdl_occlusion_replay: find the vessels -- sweeping for any the cameras cannot see -- then replay.

The two halves are existing trees, run one after the other in a single mission:

    1_Initialize          home_subtree        -- the recovery tree's home pose
    2_Locate_Vessels      one sweep for every vessel, then home  (occlusion_recovery)
    3_Replay_Start_Pose   the recording's own start pose  (trajectory_replay)
    4_Handover            onto the trajectory controller
    5_Replay              the recorded segments and gripper events
    6_Finish              back to the start pose, parked on the hold controller

``sweep_mode: per_object`` puts one Recover / Detect / Return per vessel in
place of step 2, and the later steps count on from there.

Both halves are imported, not restated: the locate blocks come from
``sdl_occlusion_recovery`` and the motion blocks from ``tasks/fr5/replay.py``
(the core replay's checks and parts), with every number measured on this arm
(sweep ceilings, return durations, velocity ceiling, gripper settle) carried
along unchanged. The recovery's parameters are this task's task_params
(``sweep_config``, ``sweep_mode``, ``visibility_topic``); the replay's are the
core node's (``replay_trajectory`` -- empty: the cell's recording,
``traj_level_seed2`` --, ``replay_meta``, ``replay_layout``, ``home_via``,
``replay_speed_scale``).

**The perception here does not steer the replay.** The latched poses land on
the blackboard exactly as they do in ``sdl_occlusion_recovery``, and nothing reads
them: the waypoints are replayed as recorded, and whether they may be is still
the declared layout gate's call, which runs at BUILD time as it does in
``trajectory_replay`` -- ``replay_layout`` is required for a recording that
assumes a layout for that reason. To gate the replay on what the cameras
measured instead, use ``sdl_perceived_replay``.

What that buys is a run that SHOWS the perception working: with
``cho_object_pose``'s ``display.launch.py`` up, rviz draws each vessel as the
sweep finds it, keeps it there dimming while the replay runs, and draws each
camera's line of sight while it is looking.

The recovery's own closing home move is left out: the start-pose move follows
it directly, from the recovery's home pose, and a third pose in between would
only lengthen the run. From that home pose to ``traj_level_seed2``'s start, a
straight joint-space line keeps the gripper's envelope 193 mm or more above the
bench, which is what ``home_via:=direct`` drives; ``moveit`` plans it instead.

Run it against the plain bringup and the camera stack::

    ros2 launch cho_object_pose display.launch.py
    ros2 launch cho_task_manager run_task_manager.launch.py task:=sdl_occlusion_replay
        robot_type:=fr5 home_via:=direct
        object_pose_config:=$TABLE object_pose_cameras_config:=$CAMERAS
        task_params:=$SHARE/config/sdl_occlusion_replay.yaml
        replay_trajectory:=<csv> replay_meta:=<meta.json> replay_layout:=<cell.yaml>

(one command; wrapped here only to fit.) ``$TABLE``, ``$CAMERAS`` and ``$SHARE``
are the ones ``sdl_occlusion_recovery`` documents.
"""

import py_trees

from cho_task_manager.subtrees import guarded_mission, home_subtree
from cho_task_manager.utils.controller_names import arm_joint_names
from sdl_cell_tasks.tasks.fr5.common import home_joint_state
from sdl_cell_tasks.tasks.fr5.occlusion_recovery import (
    HOME_DURATION_SEC as RECOVERY_HOME_DURATION_SEC,
    locate_children,
    RECOVERY_DEFAULTS,
    recovery_plan,
    recovery_summary,
    resolve_sweep_mode,
)
from sdl_cell_tasks.tasks.fr5.replay import (
    handover_sequence,
    home_block,
    replay_children,
    replay_plan,
    replay_summary,
)
from sdl_cell_tasks.utils.params import task_params

TASK_NAME = 'sdl_occlusion_replay'

#: What task_params may set, and its value when it does not: the recovery's.
DEFAULTS = dict(RECOVERY_DEFAULTS)


def create_sdl_occlusion_replay_tree(robot_config) -> py_trees.behaviour.Behaviour:
    """Home, locate every vessel (sweeping when blind), then replay the recording."""
    params = task_params(robot_config, TASK_NAME, DEFAULTS)

    # Everything that can refuse, refuses here, before a node is spun up and
    # before the arm has moved for the recovery half.
    plan = replay_plan(robot_config, TASK_NAME)
    sweeps, vessels, visibility_topic = recovery_plan(
        params, arm_joint_names(robot_config), TASK_NAME)
    mode = resolve_sweep_mode(params)

    mission = py_trees.composites.Sequence(
        name='SDL_Occlusion_Replay_Sequence', memory=True)

    # --- the recovery half, as sdl_occlusion_recovery runs it --------------
    mission.add_child(home_subtree(
        robot_config, home_joint_state(robot_config), plan.hold,
        duration=RECOVERY_HOME_DURATION_SEC, name='1_Initialize',
        # Never opened here: the replay's own gripper events decide the jaws.
        open_gripper=False))
    locate = locate_children(robot_config, mode, sweeps, vessels, visibility_topic)
    mission.add_children(locate)

    # --- the replay half, as the core's trajectory_replay runs it ----------
    step = len(locate) + 2
    mission.add_child(home_block(
        robot_config, plan, name='%d_Replay_Start_Pose' % step, suffix='_Replay_Start'))
    mission.add_child(handover_sequence(robot_config, plan, '%d_Handover' % (step + 1)))

    replay_seq = py_trees.composites.Sequence(name='%d_Replay' % (step + 2), memory=True)
    replay_seq.add_children(replay_children(plan, robot_config))
    mission.add_child(replay_seq)

    mission.add_child(home_block(
        robot_config, plan, name='%d_Finish' % (step + 3), suffix='_Final', park=True))

    root = guarded_mission(
        mission, robot_config, plan.control_mode, name='SDL_Occlusion_Replay_Root')
    root.recovery_summary = recovery_summary(sweeps, vessels, visibility_topic, mode)
    root.replay_summary = replay_summary(plan)
    return root


#: The registration setup.py points at: {robot_type: tree builder}.
TASK = {'fr5': create_sdl_occlusion_replay_tree}

__all__ = ['DEFAULTS', 'TASK', 'TASK_NAME', 'create_sdl_occlusion_replay_tree']
