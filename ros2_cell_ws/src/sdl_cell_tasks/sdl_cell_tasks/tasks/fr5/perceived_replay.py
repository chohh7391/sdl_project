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

"""sdl_perceived_replay: replay a recorded trajectory, with the cameras checking the cell it assumes.

``trajectory_replay`` is position control that senses nothing, and its whole
design follows from that: the recording carries the object placement it was
planned against, the arm goes there whether or not anything is, so a replay is
gated on a layout file someone keeps in step with the bench BY HAND. This tree
is that tree with the cameras wired in, so the gate is a measurement rather
than a declaration:

1. **The cell is measured before the arm is handed over.** ``cho_object_pose``
   publishes each tagged vessel's pose in the robot's base frame, and
   ``ObjectLayoutCheckBehavior`` compares those against the recording's own
   ``layout_the_trajectory_assumes``. A bench that has drifted refuses the
   replay with the offset in millimetres, instead of matching a YAML nobody
   re-measured.
2. **The declared layout is now optional, and still honoured.** Given
   ``replay_layout``, the build-time gate runs exactly as it does in
   ``trajectory_replay`` -- BEFORE a node is spun up, which is a strictly better
   place to refuse than after the arm has homed. Keep passing it when you have
   one: the measured check covers only the objects a camera tracks, and it can
   only run once the tree is up.
3. **Optionally, the cell is watched while the replay runs.** ``replay_watch``
   (task_params) names the vessels that are supposed to STAY PUT; a vessel
   that moves more than the tolerance fails the mission branch, which the
   standard abort then takes to the hold controller. It is off by default and has to name objects
   explicitly, because a transfer recording moves a vessel on purpose and a
   watchdog that did not know which would abort the run it exists to protect.

Everything about the motion itself -- the start-pose move, the handover, the
recorded segments, the gripper settle, the position and velocity ceilings -- is
the core ``trajectory_replay``'s, assembled in ``tasks/fr5/replay.py`` rather
than restated here. Those numbers were measured on this arm and two copies of
them would be two things to keep in step. This file adds perception and nothing
else. The replay's parameters are the core node's (``replay_trajectory`` --
empty: the cell's recording, ``traj_level_seed2`` --, ``replay_meta``,
``replay_layout``, ``home_via``, ``replay_speed_scale``); this task's own,
``task_params:=`` (:data:`DEFAULTS`), are ``replay_watch`` and
``replay_drift_tolerance`` (metres; empty: the layout gate's position
tolerance).

What this does NOT do is re-aim the trajectory at where the vessel actually is.
The waypoints are replayed exactly as recorded; the measurement decides whether
they may be replayed at all. Correcting a recorded pour for a moved vessel is a
different job, and it needs a planner rather than a gate.

Run it against a bringup started on the trajectory controller, with the camera
stack up::

    ros2 launch cho_oak       oak.launch.py  name:=side
    ros2 launch cho_realsense d435.launch.py camera_namespace:=rs_left  camera_name:=rs_left  serial_no:=_<serial>
    ros2 launch cho_realsense d435.launch.py camera_namespace:=rs_right camera_name:=rs_right serial_no:=_<serial>
    CAMERAS=$(ros2 pkg prefix --share cho_object_pose)/config/cameras.yaml
    SHARE=$(ros2 pkg prefix --share sdl_cell_tasks)
    TABLE=$SHARE/config/perception/vessel_detect.yaml

    # the detectors get the same table as the pose node: it carries the tag ids
    # and the printed tag size as well as the offsets
    ros2 launch cho_object_pose detectors.launch.py objects_config:=$TABLE
    ros2 launch cho_task_manager run_task_manager.launch.py task:=sdl_perceived_replay
        robot_type:=fr5 replay_trajectory:=/path/to/transfer_seed5_waypoints.csv
        object_pose_config:=$TABLE object_pose_cameras_config:=$CAMERAS
        task_params:=$SHARE/config/sdl_perceived_replay.yaml

(one line; wrapped here only to fit.)

The object table is the one ``sdl_vessel_detect`` uses. It is the same bench
and the same two vessels, and a second copy would be two files that have to
agree about which tag is the beaker. Commission with ``sdl_vessel_detect``
first: it moves nothing, so a wrong offset shows up in rviz instead of in a
replay.
"""

import py_trees

from cho_task_manager.behaviors.topic import PoseTargetBehavior
from cho_task_manager.subtrees import guarded_mission
from sdl_cell_tasks.behaviors.topic import (
    ObjectLayoutCheckBehavior,
    ObjectLayoutMonitorBehavior,
)
from sdl_cell_tasks.tasks.fr5.common import VESSELS
from sdl_cell_tasks.tasks.fr5.replay import (
    handover_sequence,
    home_block,
    position_tolerance,
    replay_children,
    replay_plan,
    replay_summary,
)
from sdl_cell_tasks.utils.params import number, task_params

TASK_NAME = 'sdl_perceived_replay'

#: What task_params may set, and its value when it does not. ``replay_watch``:
#: the vessels to keep a camera on while the arm runs (a list, or one string
#: separated by commas or spaces); none by default. ``replay_drift_tolerance``:
#: how far [m] a watched vessel may move; None is the layout gate's tolerance.
DEFAULTS = {
    'replay_watch': [],
    'replay_drift_tolerance': None,
}

#: Seconds each vessel gets to be detected. Generous on purpose:
#: ``cho_object_pose`` publishes nothing until it has a full agreement window,
#: and on a cold start that also waits on the TF buffer filling and three
#: cameras' exposure settling.
DETECT_TIMEOUT_SEC = 20.0

#: How often the drift monitor logs what it measures, so a known-good run
#: produces the number ``replay_drift_tolerance`` should be set from.
MONITOR_REPORT_PERIOD_SEC = 5.0


def watched_names(params, tracked):
    """Vessels ``replay_watch`` asks to be monitored during the replay.

    Empty by default. A name that is not tracked raises rather than being
    ignored: an operator who asked for a watchdog and silently got none is
    worse off than one who is told the name is wrong.
    """
    raw = params.get('replay_watch') or []
    if isinstance(raw, str):
        names = [part.strip() for part in raw.replace(',', ' ').split() if part.strip()]
    elif isinstance(raw, (list, tuple)) and all(isinstance(name, str) for name in raw):
        names = [name.strip() for name in raw if name.strip()]
    else:
        raise ValueError(
            'replay_watch must be a list of vessel names or one string of them; got %r' % (raw,))
    unknown = sorted(set(names) - set(tracked))
    if unknown:
        raise ValueError(
            'replay_watch names %s, which no camera tracks. Tracked: %s'
            % (unknown, sorted(tracked)))
    return names


def create_sdl_perceived_replay_tree(robot_config) -> py_trees.behaviour.Behaviour:
    """Home, measure the cell, replay the recording, hand back."""
    params = task_params(robot_config, TASK_NAME, DEFAULTS)

    # The declared gate still runs first when there is a layout file, because
    # refusing at BUILD time beats refusing after the arm has homed. The
    # measured check below is not a replacement for it, it is the half a file
    # cannot do -- and without a file, the only one.
    plan = replay_plan(robot_config, TASK_NAME, layout_required=False)
    recording = plan.recording
    base_frame = robot_config['arm_base_link']
    tolerance = position_tolerance(robot_config)

    # Only the vessels that are BOTH tracked by a camera and assumed by this
    # recording. An object the recording assumes and no camera watches stays the
    # declared layout's business, and the check behaviour says so out loud.
    tracked = {vessel.name: vessel for vessel in VESSELS
               if vessel.name in recording.layout}
    if not tracked:
        raise ValueError(
            'none of the vessels the cameras track (%s) appear in the layout %s '
            'assumes (%s), so there is nothing to perceive. Use task:='
            'sdl_trajectory_replay for a recording this bench cannot measure.'
            % (sorted(vessel.name for vessel in VESSELS),
               recording.source, sorted(recording.layout)))

    watch = watched_names(params, tracked)
    drift_tolerance = (tolerance if params['replay_drift_tolerance'] is None
                       else number(params, 'replay_drift_tolerance', TASK_NAME))
    if not drift_tolerance > 0.0:
        raise ValueError('%s: replay_drift_tolerance must be positive metres; got %r'
                         % (TASK_NAME, params['replay_drift_tolerance']))

    mission = py_trees.composites.Sequence(
        name='SDL_Perceived_Replay_Sequence', memory=True)

    init_seq = home_block(robot_config, plan, name='1_Initialize', suffix='')

    # After the home move, not before: the arm is then at the recording's own
    # start pose, which is a repeatable viewpoint and the one it is about to
    # replay from, rather than wherever the operator left it leaning over the
    # bench. Before the handover, so a refusal happens while the arm is still
    # held by a controller nobody is streaming at.
    verify_seq = py_trees.composites.Sequence(name='2_Verify_Cell', memory=True)
    verify_seq.add_children([
        PoseTargetBehavior(
            name='Detect_%s' % vessel.name.capitalize(),
            record_as=vessel.key,
            topic=vessel.topic,
            required_frame=base_frame,
            timeout_sec=DETECT_TIMEOUT_SEC,
        )
        for vessel in tracked.values()
    ])
    verify_seq.add_child(ObjectLayoutCheckBehavior(
        name='Check_Cell_Matches_Recording',
        expected=recording.layout,
        keys={name: vessel.key for name, vessel in tracked.items()},
        position_tolerance_m=tolerance,
    ))

    replay_seq = py_trees.composites.Sequence(name='4_Replay', memory=True)
    replay_seq.add_children(replay_children(plan, robot_config))

    mission.add_children([
        init_seq,
        verify_seq,
        handover_sequence(robot_config, plan, '3_Handover'),
        replay_seq,
        home_block(robot_config, plan, name='5_Finish', suffix='_Final', park=True),
    ])

    monitor = None
    if watch:
        monitor = ObjectLayoutMonitorBehavior(
            name='Watch_Cell_Layout',
            expected=recording.layout,
            topics={name: tracked[name].topic for name in watch},
            required_frame=base_frame,
            position_tolerance_m=drift_tolerance,
            report_period_sec=MONITOR_REPORT_PERIOD_SEC,
        )

    root = guarded_mission(
        mission, robot_config, plan.control_mode,
        name='SDL_Perceived_Replay_Root', monitor=monitor)
    root.replay_summary = replay_summary(
        plan,
        # The node logs this as 'layout checked' or 'not assumed': the cameras
        # check the layout before the handover, with or without a declared one.
        layout_checked=True,
        # What perception added, so a log says how checked the run actually was.
        measured=sorted(tracked),
        unverified=sorted(set(recording.layout) - set(tracked)),
        watched=sorted(watch),
        position_tolerance_m=tolerance,
        drift_tolerance_m=drift_tolerance,
    )
    return root


#: The registration setup.py points at: {robot_type: tree builder}.
TASK = {'fr5': create_sdl_perceived_replay_tree}

__all__ = [
    'DEFAULTS',
    'DETECT_TIMEOUT_SEC',
    'MONITOR_REPORT_PERIOD_SEC',
    'TASK',
    'TASK_NAME',
    'create_sdl_perceived_replay_tree',
    'watched_names',
]
