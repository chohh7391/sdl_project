"""Build and check the trajectories this workspace sends to the real arm.

Two things live here that did not need to exist while the plan was replayed by
publishing joint states into the simulator.

**Timing became real.** Replay used to be a publish-and-sleep loop, so
``time_from_start`` was decoration; the trajectory controller interpolates
against it instead, and rejects a trajectory whose point times do not strictly
increase. That is also why every point below gets its OWN ``Duration`` object:
a ROS message field assignment stores the reference, so reusing one accumulator
gives every point the final time -- which read fine in a plan message nobody
timed, and is a rejected goal now.

**Nothing downstream clamps.** The stock trajectory controller enforces
tolerances, not limits, and the vendor hardware's ``write()`` only rejects NaN.
So a position outside the URDF limits, or a step that implies a joint velocity
no one intended, reaches the arm. :func:`validate_trajectory` is the gate, and
it is meant to be called on every outgoing goal -- a loud refusal here is the
cheap version of the same discovery.
"""

import math

from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from cho_bridge.contract import (
    ARM_JOINTS,
    JOINT_LIMITS,
    MAX_JOINT_VELOCITY,
    MIN_STREAM_HORIZON,
)

#: Default ceiling on how far the first point may sit from where the arm
#: actually is [rad]. A trajectory planned from a stale state does not fail --
#: it lurches, at whatever rate the controller needs to close the gap.
DEFAULT_MAX_START_JUMP = 0.05


class TrajectoryRejected(ValueError):
    """A trajectory was refused before it reached the arm."""


def seconds_to_duration(seconds: float) -> Duration:
    """A FRESH Duration for *seconds*.

    Fresh on purpose: see the module docstring. Callers must never share one
    Duration across points.
    """
    if seconds < 0.0:
        raise TrajectoryRejected(f'negative time_from_start ({seconds:.6f}s)')
    whole = int(seconds)
    nanos = int(round((seconds - whole) * 1e9))
    if nanos >= 1_000_000_000:          # rounding can carry
        whole += 1
        nanos -= 1_000_000_000
    return Duration(sec=whole, nanosec=nanos)


def _rows(values):
    """Accept a list of rows or a 2-D numpy array, return lists of floats."""
    return [[float(v) for v in row] for row in values]


def build_trajectory(
    positions,
    joint_names=ARM_JOINTS,
    dt=None,
    times=None,
    velocities=None,
    accelerations=None,
    time_scale=1.0,
    start_delay=0.0,
):
    """Assemble a JointTrajectory from planner output.

    Give exactly one of *dt* (uniform spacing, first point at *start_delay*) or
    *times* (explicit seconds from start, one per point).

    ``time_scale`` stretches the trajectory: 2.0 makes it take twice as long.
    Velocities and accelerations are rescaled with it, because the controller
    reads them as the derivatives of the path it is being handed -- stretching
    the clock while leaving them alone hands it a trajectory that describes two
    different motions.
    """
    positions = _rows(positions)
    joint_names = list(joint_names)
    if not positions:
        raise TrajectoryRejected('no trajectory points')
    if time_scale <= 0.0:
        raise TrajectoryRejected(f'time_scale must be positive (got {time_scale})')
    if (dt is None) == (times is None):
        raise TrajectoryRejected('give exactly one of dt or times')

    if dt is not None:
        if dt <= 0.0:
            raise TrajectoryRejected(f'dt must be positive (got {dt})')
        times = [start_delay + i * dt for i in range(len(positions))]
    else:
        times = [start_delay + float(t) for t in times]
    if len(times) != len(positions):
        raise TrajectoryRejected(
            f'{len(times)} times for {len(positions)} points')

    velocities = _rows(velocities) if velocities is not None else None
    accelerations = _rows(accelerations) if accelerations is not None else None

    traj = JointTrajectory()
    traj.joint_names = joint_names
    for i, position in enumerate(positions):
        point = JointTrajectoryPoint()
        point.positions = position
        if velocities is not None:
            point.velocities = [v / time_scale for v in velocities[i]]
        if accelerations is not None:
            point.accelerations = [a / (time_scale * time_scale) for a in accelerations[i]]
        # One Duration per point. Never hoist this out of the loop.
        point.time_from_start = seconds_to_duration(times[i] * time_scale)
        traj.points.append(point)
    return traj


def single_point_trajectory(
    position,
    joint_names=ARM_JOINTS,
    horizon_sec=0.1,
    velocity=None,
):
    """One point *horizon_sec* ahead -- the unit a closed loop streams.

    Published on the controller's trajectory topic it replaces whatever is
    running, so the horizon is the loop's rate limit: the arm is being asked to
    cover ``position - current`` in that much time. Passing *velocity* (the
    commanded rate at the end point) keeps the motion from decelerating into
    every replacement, which is what turns a stream into stop-and-go.
    """
    if horizon_sec < MIN_STREAM_HORIZON:
        raise TrajectoryRejected(
            f'horizon {horizon_sec:.4f}s is below {MIN_STREAM_HORIZON}s: the '
            'controller would have fewer than two cycles to arrive')
    return build_trajectory(
        [position],
        joint_names=joint_names,
        times=[horizon_sec],
        velocities=None if velocity is None else [velocity],
    )


def _duration_seconds(duration) -> float:
    return duration.sec + duration.nanosec * 1e-9


def validate_trajectory(
    traj,
    joint_names=ARM_JOINTS,
    limits=None,
    max_velocity=MAX_JOINT_VELOCITY,
    start_position=None,
    max_start_jump=DEFAULT_MAX_START_JUMP,
):
    """Refuse a trajectory the arm should not be asked to follow.

    *start_position* is a mapping joint -> measured position. Given it, the
    first point is checked against where the arm actually is; that is the check
    that catches a plan built from a stale state, which does not fail on its
    own, it lurches.

    Returns the trajectory so it can be used inline. Raises
    :class:`TrajectoryRejected`, whose message names the joint and the index.
    """
    limits = JOINT_LIMITS if limits is None else limits
    expected = list(joint_names)

    if list(traj.joint_names) != expected:
        raise TrajectoryRejected(
            f'joint names {list(traj.joint_names)} != expected {expected}')
    if not traj.points:
        raise TrajectoryRejected('no trajectory points')

    previous_time = None
    previous_position = None
    for index, point in enumerate(traj.points):
        if len(point.positions) != len(expected):
            raise TrajectoryRejected(
                f'point {index} has {len(point.positions)} positions for '
                f'{len(expected)} joints')

        now = _duration_seconds(point.time_from_start)
        if not math.isfinite(now) or now < 0.0:
            raise TrajectoryRejected(f'point {index} has time_from_start {now}')
        if previous_time is not None and now <= previous_time:
            raise TrajectoryRejected(
                f'point {index} has time_from_start {now:.6f}s, not after '
                f'{previous_time:.6f}s. Point times must strictly increase -- '
                'sharing one Duration object across points is the usual cause')

        for joint_index, name in enumerate(expected):
            value = point.positions[joint_index]
            if not math.isfinite(value):
                raise TrajectoryRejected(f'point {index} {name} is {value}')
            low, high = limits[name]
            if value < low or value > high:
                raise TrajectoryRejected(
                    f'point {index} {name} = {value:+.4f} rad is outside its '
                    f'limits [{low:+.4f}, {high:+.4f}]')
            for series, label in ((point.velocities, 'velocity'),
                                  (point.accelerations, 'acceleration')):
                if series and not math.isfinite(series[joint_index]):
                    raise TrajectoryRejected(
                        f'point {index} {name} {label} is {series[joint_index]}')

            if previous_position is not None:
                span = now - previous_time
                rate = abs(value - previous_position[joint_index]) / span
                if rate > max_velocity + 1e-9:
                    raise TrajectoryRejected(
                        f'point {index} implies {name} at {rate:.3f} rad/s, '
                        f'over the {max_velocity:.3f} rad/s ceiling. Nothing '
                        'downstream clamps this')

        previous_time = now
        previous_position = list(point.positions)

    if start_position is not None:
        first = traj.points[0].positions
        for joint_index, name in enumerate(expected):
            if name not in start_position:
                raise TrajectoryRejected(
                    f'no measured position for {name}; cannot check the start')
            jump = abs(first[joint_index] - start_position[name])
            if jump > max_start_jump:
                raise TrajectoryRejected(
                    f'the trajectory starts {jump:.4f} rad away from the '
                    f'measured {name} (ceiling {max_start_jump:.4f}). The arm '
                    'is not where this plan assumes it is')
    return traj
