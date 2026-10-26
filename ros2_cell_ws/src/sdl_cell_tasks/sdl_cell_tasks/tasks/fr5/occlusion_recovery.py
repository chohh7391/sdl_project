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

"""sdl_occlusion_recovery: find the vessels, going and looking at any the standing cameras cannot see.

The bench has two cameras with two jobs. The OAK stands off and watches the
whole cell; the D435 rides the wrist. While nothing is in the way the OAK is
enough. When the arm is reaching across the bench it occludes exactly the
vessel it is reaching for, and the OAK's answer becomes "not in frame" -- which
until now was a line in a log that nothing could act on.

This tree is the acting-on. For each vessel: ask whether it is being
published, go and look with the wrist camera if it is not, latch the pose the
way every other perception task does, and come back out of the way.

    2_Locate_Beaker
      Recover_Beaker   OcclusionSweepBehavior  -- skips instantly when visible
      Detect_Beaker    PoseTargetBehavior      -- latches whatever is there now
      Return_Beaker    JointSpaceActionBehavior -- back to the home pose

THE ORDER OF THE LAST TWO IS THE WHOLE CONTRACT. The recovered pose lives in
``cho_object_pose``'s aggregation window and nowhere else -- there is no
separate lifetime for it, deliberately -- so it expires within ``window_sec`` of
the arm leaving the viewpoint. Latching first and returning second is what
turns a view that lasts half a second into a target the rest of the task can
use. Returning is not tidiness either: the arm was parked over the bench with
the camera looking down at it, which is the OAK's view of everything else
blocked, so a sweep that stayed would occlude the next vessel it went to look
for.

That is ``sweep_mode: per_object``. The default, ``single_pass``, looks for
every vessel in ONE pass over the raster and comes home once::

    2_Locate_Vessels
      Recover_Vessels  SinglePassSweepBehavior  -- latches each vessel where it is found
      Return_Vessels   JointSpaceActionBehavior -- back to the home pose

Both vessels are planning targets on the FR5 bench, so going home between them
bought nothing: the second sweep had to go out and look anyway, and began again
from the first waypoint. Measured 2026-09-23, that was about 33 s of a 74 s
locate. The latch moves inside the sweep leaf, for the reason the order above
matters -- see ``behaviors/action/single_pass_sweep.py``. It needs every vessel
swept over the same waypoints, which the FR5 table is by construction; a table
that is not is refused at build time.

The sweep itself stays high and sweeps SIDEWAYS before it comes down -- the
raster, and the prior work it follows, are documented in the spec the sweep
table is solved from, ``config/sweep/fr5_bench.raster.yaml``.

The recovery leaf comes FIRST and not as a fallback behind a failed detection,
which is worth saying because the Selector version is the obvious one. A
``PoseTargetBehavior`` that cannot see its object fails by TIMING OUT, so a
fallback arrangement pays the full detection timeout before it may begin to
recover, and then reports 'no message within 20s' -- which is true of an
occluded vessel, a detector that never started and a wrong topic alike. The
sweep leaf decides from one visibility snapshot instead: it returns SUCCESS
without moving when the pose is already there, and when it is not, it says
which camera said what and refuses outright for the reasons a sweep cannot fix.

**Nothing here keeps the recovered pose alive.** It ages out of the pose node's
aggregation window like any other sample as soon as the arm leaves the
viewpoint -- so sweeping for the flask expires the beaker's fused estimate, and
the beaker's POSE survives only because ``PoseTargetBehavior`` already latched
it onto the blackboard. That is the same contract ``sdl_vessel_detect`` and
``sdl_perceived_replay`` run under, and deliberately not a new one.

Where to sweep is this file's config and not this file: which joint
configurations put a wrist camera over a beaker is a fact about one bench and
one arm, and the next bench's is different. The task's parameters
(``task_params:=``, :data:`DEFAULTS`) name the table, ``sweep_config``
(``config/sweep/`` here; a relative path is read against this package's share
directory), the vessels it covers are the vessels this tree locates, and
``cho_object_pose`` goes on knowing nothing about a robot. ``sweep_mode`` and
``visibility_topic`` are two more.

``vessel_table`` is the last: the object table whose objects the recovery looks
for, by the names and topics it gives (``config/perception/pour_beakers.yaml``
for the two-beaker cell, with ``config/sweep/fr5_bench_beakers.yaml``). That
lets it run against a pose node already started on that table rather than on
``vessel_detect.yaml``. Unset, it looks for the bench's own vessels
(``tasks/fr5/common.py`` VESSELS), which are ``vessel_detect.yaml``'s.

The first home goes the way the node's ``home_via`` says, as the replays' do:
planned by MoveIt unless ``home_via:=direct`` (the FR5 default is ``moveit``),
because it is the one move that starts from wherever the last task left the
arm; the arm is then parked back on the hold controller the sweep drives. The
returns from the sweep and the last home stay on the hold controller.

Run it against a bringup and the camera stack::

    ros2 launch cho_bringup_fr5 bringup_real_robot.launch.py
    ros2 launch cho_bringup_fr5 camera_extrinsics.launch.py
    ros2 launch cho_oak       oak.launch.py  name:=side_1
    ros2 launch cho_realsense d435.launch.py camera_namespace:=wrist camera_name:=wrist serial_no:=_<serial>

    CAMERAS=$(ros2 pkg prefix --share cho_object_pose)/config/cameras.yaml
    SHARE=$(ros2 pkg prefix --share sdl_cell_tasks)
    TABLE=$SHARE/config/perception/vessel_detect.yaml

    ros2 launch cho_object_pose detectors.launch.py objects_config:=$TABLE cameras_config:=$CAMERAS
    ros2 launch cho_task_manager run_task_manager.launch.py task:=sdl_occlusion_recovery
        robot_type:=fr5 object_pose_config:=$TABLE object_pose_cameras_config:=$CAMERAS
        task_params:=<a file with sweep_config: config/sweep/fr5_bench.yaml>

(the wrapped lines are one command each; ``config/sdl_occlusion_recovery.yaml``
is the example, for the MuJoCo cell)

``object_pose_cameras_config`` is not optional here the way it is elsewhere.
Without it the pose node runs its single-camera path, there is no wrist camera
to recover with and no ``priority`` to make its view override the OAK's -- so
every sweep would end by failing to find the camera it was told to use.
"""

import py_trees

from cho_task_manager.behaviors.action import JointSpaceActionBehavior
from cho_task_manager.behaviors.topic import PoseTargetBehavior
from cho_task_manager.subtrees import guarded_mission, home_subtree
from cho_task_manager.tasks.trajectory_replay import HOME_VIA_MOVEIT, replay_controller
from cho_task_manager.utils.controller_names import arm_joint_names
from sdl_cell_tasks.behaviors.action import (
    DEFAULT_VISIBILITY_TOPIC,
    OcclusionSweepBehavior,
    SinglePassSweepBehavior,
    SweepTarget,
)
from sdl_cell_tasks.tasks.fr5.common import CONTROL_MODE, home_joint_state, tracked_vessels
from sdl_cell_tasks.tasks.fr5.replay import home_move, resolve_home_via
from sdl_cell_tasks.utils.occlusion import load_sweeps
from sdl_cell_tasks.utils.params import share_path, task_params

TASK_NAME = 'sdl_occlusion_recovery'

#: Seconds each vessel gets to be latched AFTER the sweep leaf has already said
#: it is being published. Short on purpose: the waiting is the sweep's job and
#: it has its own, much longer, ceiling. A timeout here means the pose stopped
#: going out between the two leaves -- the arm drifted off the viewpoint, or
#: the window emptied -- and that is worth hearing about quickly rather than
#: sitting through a 20-second wait for.
LATCH_TIMEOUT_SEC = 5.0

#: How long the arm takes to reach the home pose at either end of the tree.
HOME_DURATION_SEC = 8.0

#: How long the return from a sweep viewpoint takes. Longer than HOME_DURATION,
#: because the far end of a raster is about 3 rad away from home, and this
#: cell's commissioning ceiling is 0.394 rad/s
#: (cho_moveit_fr5/config/joint_limits.yaml).
RETURN_DURATION_SEC = 12.0

#: How the vessels are looked for: ``single_pass`` drives the raster once for
#: all of them and returns home once; ``per_object`` is one Recover -> Detect ->
#: Return per vessel. See the module docstring for why the default changed.
SWEEP_MODE_SINGLE_PASS = 'single_pass'
SWEEP_MODE_PER_OBJECT = 'per_object'
SWEEP_MODES = (SWEEP_MODE_SINGLE_PASS, SWEEP_MODE_PER_OBJECT)
DEFAULT_SWEEP_MODE = SWEEP_MODE_SINGLE_PASS

#: The recovery's parameters, which every task that locates the vessels this way
#: takes in its task_params: the sweep table (required -- no default bench),
#: how to sweep, where cho_object_pose says what each camera sees, and the
#: object table whose objects it looks for (None: the bench's own VESSELS).
RECOVERY_DEFAULTS = {
    'sweep_config': None,
    'sweep_mode': DEFAULT_SWEEP_MODE,
    'visibility_topic': DEFAULT_VISIBILITY_TOPIC,
    'vessel_table': None,
}

#: What this task's task_params may set, and its value when it does not.
DEFAULTS = dict(RECOVERY_DEFAULTS)


def sweep_config_path(params, task=TASK_NAME):
    """The sweep table *params* name, checked to exist.

    A relative path is read against this package's share directory, where the
    tables in ``config/sweep/`` are installed.
    """
    value = params.get('sweep_config')
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(
            "%s needs 'sweep_config' in its task_params: the joint configurations that "
            'put the wrist camera over each vessel. They belong to a bench, not to '
            'this file -- see config/sweep/fr5_bench.yaml in sdl_cell_tasks.' % task)
    return share_path(value, 'sweep_config', task)


def vessel_table_path(params, task=TASK_NAME):
    """The object table ``vessel_table`` names, checked to exist, or None when unset.

    A relative path is read against this package's share directory, where the
    tables in ``config/perception/`` are installed.
    """
    value = params.get('vessel_table')
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return share_path(value, 'vessel_table', task)


def tracked(params, task=TASK_NAME):
    """The vessels this bench tracks: ``vessel_table``'s objects, else VESSELS."""
    return tracked_vessels(vessel_table_path(params, task))


def resolve_sweep_mode(params):
    """``sweep_mode`` from the task's parameters, defaulted and checked."""
    mode = str(params.get('sweep_mode') or DEFAULT_SWEEP_MODE).strip().lower()
    if mode not in SWEEP_MODES:
        raise ValueError('sweep_mode must be one of %s; got %r' % (list(SWEEP_MODES), mode))
    return mode


def recovery_plan(params, joint_names, task=TASK_NAME):
    """``(sweeps, vessels, visibility_topic)`` for this bench, checked at build time.

    *params* are the task's parameters (:data:`RECOVERY_DEFAULTS`' keys). Shared
    with ``sdl_occlusion_replay``, which locates the vessels exactly this way
    before it replays: two copies of the joins between the sweep table and the
    object table would be two places for a vessel name to go stale.
    """
    # The joint count is checked against this robot's own list, so a table
    # written for a 7-axis arm fails here rather than reaching an action server
    # that fills a goal in by position.
    sweeps = load_sweeps(sweep_config_path(params, task), joint_names=joint_names)

    bench = tracked(params, task)
    vessels = [vessel for vessel in bench if vessel.name in sweeps]
    if not vessels:
        raise ValueError(
            'the sweep table covers %s, and the vessels this bench tracks are %s, so '
            'there is nothing to recover. The names have to match the object table '
            '(vessel_table, else config/perception/vessel_detect.yaml), which is what '
            'decides the topic.'
            % (sorted(sweeps), sorted(vessel.name for vessel in bench)))

    visibility_topic = str(params.get('visibility_topic') or DEFAULT_VISIBILITY_TOPIC).strip()
    return sweeps, vessels, visibility_topic


def locate_sequences(robot_config, sweeps, vessels, visibility_topic, first_index=2):
    """One ``Recover -> Detect -> Return`` sequence per vessel, in order.

    ``first_index`` numbers them after whatever the calling tree put first, so
    the step names in a log still count up.
    """
    base_frame = robot_config['arm_base_link']
    controller = robot_config['joint_space']
    sequences = []
    for index, vessel in enumerate(vessels):
        locate = py_trees.composites.Sequence(
            name='%d_Locate_%s' % (index + first_index, vessel.name.capitalize()),
            memory=True)
        locate.add_children([
            OcclusionSweepBehavior(
                name='Recover_%s' % vessel.name.capitalize(),
                sweep=sweeps[vessel.name],
                controller_name=controller,
                visibility_topic=visibility_topic,
                joint_names=arm_joint_names(robot_config),
            ),
            PoseTargetBehavior(
                name='Detect_%s' % vessel.name.capitalize(),
                record_as=vessel.key,
                topic=vessel.topic,
                required_frame=base_frame,
                timeout_sec=LATCH_TIMEOUT_SEC,
            ),
            # AFTER the latch, never before: see the module docstring. The
            # blackboard now holds the pose, so the window is free to expire.
            JointSpaceActionBehavior(
                name='Return_%s' % vessel.name.capitalize(),
                target_joints=home_joint_state(robot_config),
                controller_name=controller,
                duration=RETURN_DURATION_SEC,
                timeout_sec=RETURN_DURATION_SEC + 20.0,
                joint_names=arm_joint_names(robot_config),
            ),
        ])
        sequences.append(locate)
    return sequences


def single_pass_sequence(robot_config, sweeps, vessels, visibility_topic, index=2):
    """One ``Recover -> Return`` over every vessel: the raster driven once."""
    controller = robot_config['joint_space']
    try:
        recover = SinglePassSweepBehavior(
            name='Recover_Vessels',
            targets=[SweepTarget(sweeps[vessel.name], vessel.key, vessel.topic)
                     for vessel in vessels],
            required_frame=robot_config['arm_base_link'],
            controller_name=controller,
            visibility_topic=visibility_topic,
            joint_names=arm_joint_names(robot_config),
        )
    except ValueError as error:
        raise ValueError('%s (sweep_mode: %s)' % (error, SWEEP_MODE_PER_OBJECT)) from error
    locate = py_trees.composites.Sequence(name='%d_Locate_Vessels' % index, memory=True)
    locate.add_children([
        recover,
        # After the leaf has latched every vessel, so the windows may expire.
        JointSpaceActionBehavior(
            name='Return_Vessels',
            target_joints=home_joint_state(robot_config),
            controller_name=controller,
            duration=RETURN_DURATION_SEC,
            timeout_sec=RETURN_DURATION_SEC + 20.0,
            joint_names=arm_joint_names(robot_config),
        ),
    ])
    return locate


def locate_children(robot_config, mode, sweeps, vessels, visibility_topic, first_index=2):
    """The locate blocks for *mode* (``sweep_mode``), numbered from *first_index*."""
    if mode == SWEEP_MODE_PER_OBJECT:
        return locate_sequences(robot_config, sweeps, vessels, visibility_topic,
                                first_index=first_index)
    return [single_pass_sequence(robot_config, sweeps, vessels, visibility_topic,
                                 index=first_index)]


def recovery_summary(sweeps, vessels, visibility_topic, mode=DEFAULT_SWEEP_MODE,
                     bench=None):
    """What the recovery half of a tree covers, for the node to report.

    *bench* is every vessel the bench tracks (:func:`tracked`); None is VESSELS.
    """
    bench = tracked_vessels(None) if bench is None else bench
    return {
        'sweep_mode': mode,
        'vessels': [vessel.name for vessel in vessels],
        'min_decision_margin': {name: sweep.min_decision_margin
                                for name, sweep in sweeps.items()},
        'planning_targets': sorted(name for name, sweep in sweeps.items()
                                   if sweep.planning_target),
        'uncovered': sorted(set(vessel.name for vessel in bench) - set(sweeps)),
        'waypoints': {name: [point.name for point in sweep.waypoints]
                      for name, sweep in sweeps.items()},
        'recovery_cameras': {name: sweep.recovery_camera
                             for name, sweep in sweeps.items()},
        'visibility_topic': visibility_topic,
    }


def create_sdl_occlusion_recovery_tree(robot_config) -> py_trees.behaviour.Behaviour:
    """Home, then locate every vessel the sweep table covers, sweeping when blind."""
    params = task_params(robot_config, TASK_NAME, DEFAULTS)

    controller = robot_config['joint_space']
    joint_names = arm_joint_names(robot_config)

    sweeps, vessels, visibility_topic = recovery_plan(params, joint_names)
    mode = resolve_sweep_mode(params)
    home_via = resolve_home_via(robot_config)

    mission = py_trees.composites.Sequence(
        name='SDL_Occlusion_Recovery_Sequence', memory=True)
    if home_via == HOME_VIA_MOVEIT:
        # The one move that starts from wherever the last task left the arm, so
        # it is planned, then the arm is parked back on the hold controller the
        # sweep drives. The gripper is not opened, as below.
        mission.add_child(home_move(
            robot_config, home_joint_state(robot_config), home_via, controller,
            replay_controller(robot_config), name='1_Initialize', suffix='', park=True))
    else:
        mission.add_child(home_subtree(
            robot_config, home_joint_state(robot_config), controller,
            duration=HOME_DURATION_SEC, name='1_Initialize',
            # The gripper is not part of looking at anything, and opening it is
            # one more piece of hardware that has to be present for a
            # perception run.
            open_gripper=False))

    locate = locate_children(robot_config, mode, sweeps, vessels, visibility_topic)
    mission.add_children(locate)

    # Back to a known pose, so the arm does not end a perception run parked
    # over a beaker with the camera 30 cm off the glass.
    mission.add_child(home_subtree(
        robot_config, home_joint_state(robot_config), controller,
        duration=HOME_DURATION_SEC,
        name='%d_Finish' % (len(locate) + 2), suffix='_Final',
        open_gripper=False))

    root = guarded_mission(
        mission, robot_config, CONTROL_MODE,
        name='SDL_Occlusion_Recovery_Root')
    root.recovery_summary = recovery_summary(sweeps, vessels, visibility_topic, mode,
                                             bench=tracked(params))
    root.recovery_summary['home_via'] = home_via
    return root


#: The registration setup.py points at: {robot_type: tree builder}.
TASK = {'fr5': create_sdl_occlusion_recovery_tree}


__all__ = [
    'DEFAULTS',
    'HOME_DURATION_SEC',
    'LATCH_TIMEOUT_SEC',
    'RECOVERY_DEFAULTS',
    'RETURN_DURATION_SEC',
    'SWEEP_MODES',
    'TASK',
    'TASK_NAME',
    'create_sdl_occlusion_recovery_tree',
    'locate_children',
    'locate_sequences',
    'recovery_plan',
    'recovery_summary',
    'resolve_sweep_mode',
    'single_pass_sequence',
    'sweep_config_path',
    'tracked',
    'vessel_table_path',
]
