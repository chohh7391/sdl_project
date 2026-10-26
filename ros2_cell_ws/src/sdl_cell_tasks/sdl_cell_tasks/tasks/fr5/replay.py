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

"""The motion half every replay of the cell shares, assembled from the core's replay.

The core replays a recording with one robot-independent tree,
``cho_task_manager/tasks/trajectory_replay.py``, which has no place to put
anything between its segments: no pour, no grasp measurement, no locate block
before it, no camera check after its home move. The cell's three replays
(sdl_trajectory_replay, sdl_occlusion_replay, sdl_perceived_replay) need
exactly those, so they build the same tree from the same parts here:

* :func:`replay_plan` is everything the core's ``create_trajectory_replay_tree``
  checks before it builds anything -- the control mode, the one arm, the
  recording, the declared layout gate, ``home_via``, ``replay_speed_scale``,
  the joint velocity ceiling from ``motion_limit_parameters()`` -- through the
  core's public functions, with the FR5's measured defaults
  (``cho_task_manager.tasks.fr5.trajectory_replay.DEFAULTS``);
* :func:`home_block`, :func:`replay_children`, :func:`recording_paths` and
  :func:`check_layout` are copies of the core's private ``_home_block``,
  ``_replay_children``, ``_recording_paths`` and ``_check_layout`` (v0.1.0),
  each saying so; ``replay_children`` takes the pour, grasp and place hooks the
  cell splices in, and the MoveIt home waits out the bridge's readiness gate
  after its controller switch (:data:`MOVEIT_GATE_WAIT_SEC`);
* :func:`retimed` sets the replay's clock from the moves it replays, after a
  pour has been spliced out of them.

The parameters are the core node's own (``replay_trajectory``, ``replay_meta``,
``replay_layout``, ``home_via``, ``replay_speed_scale``, ``control_mode``), read
from ``robot_config`` as the core's replay reads them. One thing differs: with
no ``replay_trajectory`` a cell replay plays the cell's recording,
``traj_level_seed2`` in the core's installed ``config/replay/fr5/``
(:func:`cell_recording`), not the robot's example, which pours nothing and
assumes no layout.
"""

from dataclasses import dataclass, replace
import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
import py_trees

from cho_task_manager.behaviors.action import (
    FollowJointTrajectoryBehavior,
    GripperActionBehavior,
    JointSpaceActionBehavior,
)
from cho_task_manager.behaviors.service import (
    ListControllersServiceBehavior,
    SwitchControllerServiceBehavior,
)
from cho_task_manager.behaviors.wait import WaitBehavior
from cho_task_manager.subtrees import home_subtree
from cho_task_manager.tasks.fr5.trajectory_replay import DEFAULTS as FR5_REPLAY_DEFAULTS
from cho_task_manager.tasks.trajectory_replay import (
    EXAMPLE_PROFILE,
    HOME_DURATION_SEC,
    HOME_VIA_CHOICES,
    HOME_VIA_DIRECT,
    HOME_VIA_MOVEIT,
    MOVEIT_PLANNING_MARGIN_SEC,
    replay_controller,
    REPLAY_CONTROL_MODE,
    REPLAY_DATA_DIR,
    ReplayDefaults,
    ROBOT_DIRS,
    velocity_limits,
    velocity_scaling,
)
from cho_task_manager.utils.controller_names import (
    arm_joint_names,
    hold_controllers,
    moveit_joint_action_name,
    resolve_control_mode,
)
from cho_task_manager.utils.msg_utils import make_joint_state
from cho_task_manager.utils.robot_description import DescriptionPositionLimits
from cho_task_manager.utils.trajectory_recording import (
    DEFAULT_POSITION_TOLERANCE_M,
    DEFAULT_YAW_TOLERANCE_DEG,
    load_cell_layout,
    load_recording,
    plan_segments,
    Recording,
    require_layout,
    required_time_scale,
    Segment,
)

#: The FR5 cell's recording, kept in the core's installed config/replay/fr5/ under
#: the name it was recorded with: traj_level_seed2.csv and traj_level_seed2_meta.json
#: (not the <name>_waypoints.csv pair example_recording() resolves). It assumes
#: the bench's layout and has one ``pouring`` span.
CELL_RECORDING = 'traj_level_seed2'

#: The FR5's replay defaults, measured on the real arm (home_via moveit, speed
#: 0.25, gripper settle 4 s): the core's, unchanged.
CELL_DEFAULTS = FR5_REPLAY_DEFAULTS

#: Seconds between handing the arm to the trajectory controller and asking
#: MoveIt to move it. The core's MoveIt bridge takes a goal only while its own
#: check, run every 0.5 s, has seen the planning scene ready within the last
#: 1.5 s (cho_moveit_common moveit_action_bridge.py); a goal sent straight after
#: the switch got there first and was rejected on the real cell ("static floor
#: planning scene is not ready", 2026-09-30).
MOVEIT_GATE_WAIT_SEC = 1.5


def cell_recording() -> Tuple[str, str]:
    """``(csv, meta)``: the installed paths of the cell's recording, in cho_task_manager's share.

    Raises ValueError when the core's install does not have it.
    """
    try:
        share = get_package_share_directory('cho_task_manager')
    except PackageNotFoundError as error:
        raise ValueError('cho_task_manager is not installed: %s' % error) from error
    directory = os.path.join(share, REPLAY_DATA_DIR, ROBOT_DIRS['fr5'])
    csv_path = os.path.join(directory, CELL_RECORDING + '.csv')
    meta_path = os.path.join(directory, CELL_RECORDING + '_meta.json')
    missing = [path for path in (csv_path, meta_path) if not os.path.isfile(path)]
    if missing:
        raise ValueError("the cell's recording is not installed: %s" % ', '.join(missing))
    return csv_path, meta_path


@dataclass(frozen=True)
class ReplayPlan:
    """Everything a replay tree is built from, checked before any of it is."""

    recording: Recording
    segments: List[Segment]
    control_mode: str
    hold: str
    controller: str
    joint_names: List[str]
    home: object                # sensor_msgs/JointState: the recording's start pose
    home_via: str
    speed_scale: float
    time_scale: float
    ceiling_scale: float
    ceiling_joint: Optional[str]
    ceiling_rate: float
    ceiling: float
    limits: Dict[str, float]
    scaling: float
    gripper_settle_sec: float
    layout_checked: bool
    declared_layout: Optional[str]


def recording_paths(robot_config, task) -> Tuple[str, Optional[str]]:
    """``(csv, meta or None)``: the operator's recording, or the cell's (:func:`cell_recording`).

    Copied from cho_task_manager/tasks/trajectory_replay.py (v0.1.0),
    ``_recording_paths``, which is private there. Two changes: *task* names the
    tree in the messages, and with no ``replay_trajectory`` it plays the cell's
    recording rather than the robot's example.
    """
    csv_path = (robot_config.get('replay_trajectory') or '').strip()
    meta_path = (robot_config.get('replay_meta') or '').strip() or None
    if csv_path:
        if not os.path.exists(csv_path):
            raise ValueError('replay_trajectory: no such file (%s)' % csv_path)
        return csv_path, meta_path
    if meta_path:
        raise ValueError(
            "%s needs 'replay_trajectory' when 'replay_meta' is given (%s): the bundled "
            'recording is replayed only when neither is.' % (task, meta_path))
    profile = robot_config.get('profile', 'single')
    if robot_config.get('robot_type') != 'fr5' or profile != EXAMPLE_PROFILE:
        raise ValueError(
            "%s needs 'replay_trajectory': the cell's recording is the FR5 single arm's, "
            "pass a recording made for %s profile '%s'."
            % (task, robot_config.get('robot_type'), profile))
    return cell_recording()


def check_layout(recording, robot_config, task, required=True) -> Tuple[bool, Optional[str]]:
    """Gate the replay on the declared cell layout; ``(compared, layout path or None)``.

    Copied from cho_task_manager/tasks/trajectory_replay.py (v0.1.0),
    ``_check_layout``, which is private there. ``replay_layout`` is needed
    exactly when the recording assumes a layout and *required* is True; with
    *required* False (sdl_perceived_replay, whose cameras measure the cell) a
    missing file only means no declared gate. A file given is always read, so
    one that is not a cell layout is an error rather than ignored.
    """
    assumed = recording.layout
    layout_path = (robot_config.get('replay_layout') or '').strip()
    if not layout_path:
        if assumed and required:
            raise ValueError(
                "%s needs 'replay_layout': %s assumes a cell layout (%s). Pass the cell's "
                'own layout file to check it against.'
                % (task, os.path.basename(recording.source), ', '.join(sorted(assumed))))
        return False, None
    if not os.path.exists(layout_path):
        raise ValueError('replay_layout: no such file (%s)' % layout_path)
    actual = load_cell_layout(layout_path)
    if not assumed:
        return False, layout_path
    # THE GATE. Raises LayoutMismatch (a ValueError), which task_manager_node
    # reports and exits on, so a mismatched cell never reaches a moving arm.
    require_layout(
        recording,
        actual,
        position_tolerance_m=position_tolerance(robot_config),
        yaw_tolerance_deg=float(robot_config.get(
            'replay_yaw_tolerance', DEFAULT_YAW_TOLERANCE_DEG)),
    )
    return True, layout_path


def position_tolerance(robot_config) -> float:
    """How far [m] a vessel may sit from where the recording assumes it, as the core reads it."""
    return float(robot_config.get('replay_position_tolerance', DEFAULT_POSITION_TOLERANCE_M))


def replay_plan(robot_config, task, defaults: ReplayDefaults = CELL_DEFAULTS,
                layout_required=True) -> ReplayPlan:
    """Check everything the core's ``create_trajectory_replay_tree`` checks, and plan the segments.

    The same checks in the same order, through the core's public functions, so
    a recording this refuses is one ``task:=trajectory_replay`` refuses too.
    Raises ValueError (a ``RecordingRejected`` for the recording itself).
    """
    mode = resolve_control_mode(robot_config, defaults.control_mode)
    if mode != REPLAY_CONTROL_MODE:
        raise ValueError(
            "%s drives the trajectory controller, which claims position command "
            "interfaces; control_mode '%s' has none. Start the bringup with "
            "control_mode:=position (and pass that here)." % (task, mode))

    csv_path, meta_path = recording_paths(robot_config, task)

    holds = hold_controllers(robot_config, mode)
    if len(holds) != 1:
        raise ValueError(
            "%s drives one arm's trajectory controller; profile '%s' holds %d arms (%s)."
            % (task, robot_config.get('profile', 'single'), len(holds), ', '.join(holds)))
    controller = replay_controller(robot_config)
    joint_names = arm_joint_names(robot_config)

    recording = load_recording(csv_path, meta_path, joint_names=joint_names)
    layout_checked, layout_path = check_layout(
        recording, robot_config, task, required=layout_required)

    home_via = resolve_home_via(robot_config, defaults)

    speed_scale = float(robot_config.get('replay_speed_scale', defaults.speed_scale))
    if not 0.0 < speed_scale <= 1.0:
        raise ValueError(
            'replay_speed_scale must be in (0, 1]; got %r. Above 1.0 would '
            'replay the recording FASTER than it was recorded, which is a '
            'different motion.' % speed_scale)

    limits = velocity_limits(robot_config)
    unbounded = [joint for joint in joint_names if joint not in limits]
    if unbounded:
        raise ValueError(
            "robot_type '%s': no MoveIt joint velocity limit for %s (limits cover "
            '%s), so their replay would be unbounded'
            % (robot_config['robot_type'], unbounded, sorted(limits)))
    scaling = velocity_scaling(robot_config)
    ceiling_scale, joint, rate, ceiling = required_time_scale(recording, limits, scaling=scaling)
    # The operator's choice and the arm's ceiling are two separate reasons to
    # stretch the clock, and the slower of the two wins.
    time_scale = max(1.0 / speed_scale, ceiling_scale)

    return ReplayPlan(
        recording=recording,
        segments=plan_segments(recording),
        control_mode=mode,
        hold=holds[0],
        controller=controller,
        joint_names=joint_names,
        home=make_joint_state(recording.home, joint_names),
        home_via=home_via,
        speed_scale=speed_scale,
        time_scale=time_scale,
        ceiling_scale=ceiling_scale,
        ceiling_joint=joint,
        ceiling_rate=rate,
        ceiling=ceiling,
        limits=limits,
        scaling=scaling,
        gripper_settle_sec=defaults.gripper_settle_sec,
        layout_checked=layout_checked,
        declared_layout=layout_path,
    )


def resolve_home_via(robot_config, defaults: ReplayDefaults = CELL_DEFAULTS) -> str:
    """``home_via`` as the core node passed it, else the FR5 default, checked.

    With ``moveit``, a robot whose registry has no MoveIt endpoint is a
    build-time error naming it.
    """
    home_via = (robot_config.get('home_via') or defaults.home_via).strip().lower()
    if home_via not in HOME_VIA_CHOICES:
        raise ValueError(
            'home_via must be one of %s; got %r' % (list(HOME_VIA_CHOICES), home_via))
    if home_via == HOME_VIA_MOVEIT:
        moveit_joint_action_name(robot_config)
    return home_via


def home_block(robot_config, plan, name, suffix, park=False):
    """Take the arm to the recording's start pose, either straight there or planned.

    Copied from cho_task_manager/tasks/trajectory_replay.py (v0.1.0),
    ``_home_block``, which is private there, as :func:`home_move`. The gripper
    is never opened here, in either variant: the recording may begin with the
    arm already holding a vessel, and opening would drop it.
    """
    return home_move(robot_config, plan.home, plan.home_via, plan.hold, plan.controller,
                     name, suffix, park=park)


def home_move(robot_config, home, home_via, hold, controller, name, suffix, park=False):
    """:func:`home_block` for any pose: *home* by *hold* (direct) or MoveIt through *controller*.

    The occlusion recovery's first home and the grasp trial's homes go this way
    too, so every home a cell tree plans waits out the bridge's gate the same.
    """
    if home_via == HOME_VIA_DIRECT:
        return home_subtree(
            robot_config,
            target_joints=home,
            controller=hold,
            duration=HOME_DURATION_SEC,
            name=name,
            suffix=suffix,
            open_gripper=False,
        )

    # MoveIt plans and executes through the trajectory controller, so the arm
    # goes to that controller FIRST and the goal is sent to the bridge, not to a
    # controller's own action server.
    sequence = py_trees.composites.Sequence(name=name, memory=True)
    sequence.add_children([
        SwitchControllerServiceBehavior(
            name='Switch_To_%s%s' % (controller, suffix),
            activate=[controller],
            robot_config=robot_config,
        ),
        # On the node clock, like every wait in the cell's trees.
        WaitBehavior(name='Wait_For_MoveIt_Gate%s' % suffix,
                     duration_sec=MOVEIT_GATE_WAIT_SEC),
        JointSpaceActionBehavior(
            name='Go_Home_MoveIt%s' % suffix,
            target_joints=home,
            duration=HOME_DURATION_SEC,
            action_name=moveit_joint_action_name(robot_config),
            # The bridge plans for the registry's joints, by these names.
            joint_names=arm_joint_names(robot_config),
            # Planning and executing a whole move is slower than the straight
            # interpolation the direct path does, and the bridge only answers
            # once execution finished.
            timeout_sec=HOME_DURATION_SEC + MOVEIT_PLANNING_MARGIN_SEC,
        ),
    ])
    if park:
        # Leave the arm held by its hold controller, as the direct path's final
        # block does: ending a session on the controller an external executor
        # was driving is how an arm is left unattended under a live goal.
        sequence.add_child(SwitchControllerServiceBehavior(
            name='Park_On_%s%s' % (hold, suffix),
            activate=[hold],
            robot_config=robot_config,
        ))
    return sequence


def replayed_time_scale(plan, segments):
    """``(scale, joint, rate, ceiling)`` over the moves in *segments*, as the core measures it.

    The core's ``required_time_scale`` measures the whole recording, and so
    does :func:`replay_plan`. A pour handed to the pouring controller is cut
    out of the moves, and on the real cell (2026-09-30) its recorded samples --
    two 1.6 ms apart with a 1 deg j6 step, 680 deg/s on paper -- stretched every
    move 29.7x. What is not replayed does not set the clock here: each move is
    measured on its own, with the core's function, and the slowest wins.
    """
    worst = (1.0, None, 0.0, 0.0)
    for segment in segments:
        if segment.kind != 'move':
            continue
        span = SimpleNamespace(times=segment.times, positions=segment.positions,
                               joint_names=plan.recording.joint_names)
        measured = required_time_scale(span, plan.limits, scaling=plan.scaling)
        if measured[1] is not None and measured[0] > worst[0]:
            worst = measured
    return worst


def retimed(plan, segments):
    """*plan* with its clock set by the moves *segments* replays (:func:`replayed_time_scale`)."""
    ceiling_scale, joint, rate, ceiling = replayed_time_scale(plan, segments)
    return replace(
        plan,
        time_scale=max(1.0 / plan.speed_scale, ceiling_scale),
        ceiling_scale=ceiling_scale,
        ceiling_joint=joint,
        ceiling_rate=rate,
        ceiling=ceiling,
    )


def handover_sequence(robot_config, plan, name):
    """Onto the trajectory controller, verified -- the core replay's ``2_Handover``."""
    sequence = py_trees.composites.Sequence(name=name, memory=True)
    if plan.home_via == HOME_VIA_DIRECT:
        # The direct home ran on the hold controller, so the arm has to change
        # hands here. The MoveIt home already ran on this one and stays on it.
        sequence.add_child(SwitchControllerServiceBehavior(
            name='Switch_To_%s' % plan.controller,
            activate=[plan.controller],
            robot_config=robot_config,
        ))
    # Verified either way: the MoveIt path has no switch here, and sending a
    # trajectory at an inactive controller is a silent no-op.
    sequence.add_child(ListControllersServiceBehavior(
        name='Verify_Replay_Controller_Active',
        require_active=[plan.controller],
    ))
    return sequence


def replay_children(plan, robot_config, segments=None, pour_children=None, grasp_children=None,
                    place_children=None):
    """One behaviour per segment, in recorded order.

    Copied from cho_task_manager/tasks/trajectory_replay.py (v0.1.0),
    ``_replay_children``, which is private there, with the cell's three hooks.
    *segments* defaults to the plan's; a pour replay passes them through
    ``splice_pour()`` first. *pour_children* builds what replaces a pour segment,
    as ``pour_children(index, segment) -> [behaviour, ...]``. *grasp_children*
    builds what follows the settle of the LAST grasp before it -- the one that
    closed on the vessel being poured -- as ``grasp_children(index)``, and
    *place_children* what follows the settle of the FIRST release after it --
    the one that sets that vessel down -- as ``place_children(index)``.
    """
    segments = plan.segments if segments is None else segments
    gripper = robot_config.get('gripper')
    pours = [i for i, segment in enumerate(segments) if segment.kind == 'pour']
    grasps = [i for i, segment in enumerate(segments)
              if segment.kind == 'gripper' and segment.grasp and pours and i < pours[0]]
    grasp_of_pour = grasps[-1] if grasps else None
    releases = [i for i, segment in enumerate(segments)
                if segment.kind == 'gripper' and not segment.grasp and pours and i > pours[0]]
    place_of_pour = releases[0] if releases else None
    # One source for every segment of this replay: one subscription, one parse.
    position_limits = DescriptionPositionLimits(plan.joint_names)
    if gripper is None and any(segment.kind == 'gripper' for segment in segments):
        raise ValueError(
            'the recording drives the gripper, and no gripper controller was given '
            "(robot_config['gripper'])")
    children = []
    for index, segment in enumerate(segments):
        if segment.kind == 'pour':
            if pour_children is None:
                raise ValueError(
                    'the replay reached a spliced-out pour (%r) with nothing to pour it'
                    % segment)
            children.extend(pour_children(index, segment))
            continue
        if segment.kind == 'gripper':
            children.append(GripperActionBehavior(
                name='%d_Gripper_%s_%s' % (
                    index, 'Close' if segment.grasp else 'Open', segment.operation or 'seg'),
                grasp=segment.grasp,
                controller_name=gripper,
            ))
            # Stand still until the jaws really have, on the node's clock: in
            # simulation the jaws move in sim time. The FR5's settle is the
            # core's measured GRIPPER_SETTLE_SEC (tasks/fr5/trajectory_replay.py).
            children.append(WaitBehavior(
                name='%d_Gripper_Settle' % index, duration_sec=plan.gripper_settle_sec))
            if index == grasp_of_pour and grasp_children is not None:
                # Settled jaws, an arm standing still, and the vessel still on
                # the bench near a camera: the best moment the replay has to
                # measure where it sits in them.
                children.extend(grasp_children(index))
            if index == place_of_pour and place_children is not None:
                # The vessel is down and the jaws are clear of it.
                children.extend(place_children(index))
            continue
        children.append(FollowJointTrajectoryBehavior(
            name='%d_Replay_%s' % (index, segment.operation or 'move'),
            controller=plan.controller,
            joint_names=plan.joint_names,
            times=segment.times,
            positions=segment.positions,
            time_scale=plan.time_scale,
            position_limits=position_limits,
            # The same ceiling the stretch was computed against, so a segment
            # that slipped through it is refused rather than sent.
            velocity_limits={name: value * plan.scaling for name, value in plan.limits.items()},
        ))
    return children


def replay_summary(plan, segments=None, **extra):
    """What the node logs about a replay (task_manager_node reads every key here), plus *extra*."""
    segments = plan.segments if segments is None else segments
    summary = {
        'source': plan.recording.source,
        'waypoints': len(plan.recording.times),
        'dropped': list(plan.recording.dropped),
        'duration_s': plan.recording.duration,
        'time_scale': plan.time_scale,
        'speed_scale': plan.speed_scale,
        'ceiling_scale': plan.ceiling_scale,
        'ceiling_joint': plan.ceiling_joint,
        'ceiling_rate': plan.ceiling_rate,
        'ceiling': plan.ceiling,
        'home_via': plan.home_via,
        'segments': [repr(segment) for segment in segments],
        'layout_checked': plan.layout_checked,
        'declared_layout': plan.declared_layout,
    }
    summary.update(extra)
    return summary


__all__ = [
    'CELL_DEFAULTS',
    'CELL_RECORDING',
    'MOVEIT_GATE_WAIT_SEC',
    'ReplayPlan',
    'cell_recording',
    'check_layout',
    'handover_sequence',
    'home_block',
    'home_move',
    'position_tolerance',
    'recording_paths',
    'replay_children',
    'replay_plan',
    'replay_summary',
    'replayed_time_scale',
    'resolve_home_via',
    'retimed',
]
