"""What must be true of anything this workspace sends to the real arm.

Two failures these cover are not hypothetical. Sharing one Duration object
across trajectory points is what `TAMPServer.process_plan` does today -- it was
invisible while replay was a publish-and-sleep loop and is a rejected goal the
moment a trajectory controller reads the times. And nothing downstream of here
clamps a joint position or a commanded rate, so the validator is the last thing
between a planner bug and a real arm.

No ROS graph is needed: these are message objects and pure checks.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from cho_bridge.contract import ARM_JOINTS, JOINT_LIMITS, MAX_JOINT_VELOCITY  # noqa: E402
from cho_bridge.trajectory import (  # noqa: E402
    TrajectoryRejected,
    build_trajectory,
    seconds_to_duration,
    single_point_trajectory,
    validate_trajectory,
)

HOME = [0.0, -math.pi / 4, -math.pi / 2, math.pi / 4, -math.pi / 2, 0.0]


def _ramp(steps, joint=0, step=0.002):
    """A slow single-joint ramp from the ready pose."""
    rows = []
    for i in range(steps):
        row = list(HOME)
        row[joint] += i * step
        rows.append(row)
    return rows


def _seconds(point):
    return point.time_from_start.sec + point.time_from_start.nanosec * 1e-9


# --- the Duration bug ------------------------------------------------------

def test_every_point_gets_its_own_time():
    traj = build_trajectory(_ramp(5), dt=0.1)
    times = [_seconds(p) for p in traj.points]
    assert times == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    # The aliasing failure produces equal times, so also assert the objects are
    # distinct: that is the actual defect, and equal-looking times could later
    # come from a builder that copies.
    identities = {id(point.time_from_start) for point in traj.points}
    assert len(identities) == len(traj.points)


def test_a_shared_duration_is_rejected_by_the_validator():
    traj = build_trajectory(_ramp(4), dt=0.1)
    shared = traj.points[-1].time_from_start
    for point in traj.points:
        point.time_from_start = shared
    with pytest.raises(TrajectoryRejected, match='strictly increase'):
        validate_trajectory(traj)


# --- building --------------------------------------------------------------

def test_dt_and_times_are_mutually_exclusive():
    with pytest.raises(TrajectoryRejected):
        build_trajectory(_ramp(3), dt=0.1, times=[0.0, 0.1, 0.2])
    with pytest.raises(TrajectoryRejected):
        build_trajectory(_ramp(3))


def test_time_scale_rescales_the_derivatives_too():
    rows = _ramp(3)
    velocities = [[0.2] * 6 for _ in rows]
    accelerations = [[0.4] * 6 for _ in rows]
    traj = build_trajectory(rows, dt=0.1, velocities=velocities,
                            accelerations=accelerations, time_scale=2.0)

    assert [_seconds(p) for p in traj.points] == pytest.approx([0.0, 0.2, 0.4])
    # Twice the time means half the speed and a quarter of the acceleration;
    # leaving them alone would describe a different motion than the times do.
    assert traj.points[0].velocities[0] == pytest.approx(0.1)
    assert traj.points[0].accelerations[0] == pytest.approx(0.1)


def test_seconds_to_duration_carries_on_rounding():
    duration = seconds_to_duration(1.9999999999)
    assert duration.sec == 2 and duration.nanosec == 0
    assert seconds_to_duration(0.25) == seconds_to_duration(0.25)


def test_a_streamed_point_needs_a_usable_horizon():
    with pytest.raises(TrajectoryRejected, match='below'):
        single_point_trajectory(HOME, horizon_sec=0.001)
    traj = single_point_trajectory(HOME, horizon_sec=0.1, velocity=[0.0] * 6)
    assert len(traj.points) == 1
    assert _seconds(traj.points[0]) == pytest.approx(0.1)


# --- validation ------------------------------------------------------------

def test_a_slow_ramp_passes():
    assert validate_trajectory(build_trajectory(_ramp(20), dt=0.1)) is not None


def test_a_position_outside_the_urdf_limits_is_refused():
    rows = _ramp(3)
    rows[2][1] = JOINT_LIMITS['j2'][0] - 0.01
    with pytest.raises(TrajectoryRejected, match='outside its limits'):
        validate_trajectory(build_trajectory(rows, dt=0.1))


def test_a_step_over_the_velocity_ceiling_is_refused():
    rows = _ramp(2)
    rows[1][0] = HOME[0] + MAX_JOINT_VELOCITY * 0.1 * 2.0     # twice the ceiling
    with pytest.raises(TrajectoryRejected, match='rad/s'):
        validate_trajectory(build_trajectory(rows, dt=0.1))


def test_a_non_finite_position_is_refused():
    rows = _ramp(3)
    rows[1][3] = float('nan')
    with pytest.raises(TrajectoryRejected):
        validate_trajectory(build_trajectory(rows, dt=0.1))


def test_the_joint_names_must_be_the_arm_in_order():
    traj = build_trajectory(_ramp(3), dt=0.1)
    traj.joint_names = list(reversed(ARM_JOINTS))
    with pytest.raises(TrajectoryRejected, match='joint names'):
        validate_trajectory(traj)


def test_a_plan_that_starts_away_from_the_arm_is_refused():
    traj = build_trajectory(_ramp(3), dt=0.1)
    measured = {name: HOME[i] for i, name in enumerate(ARM_JOINTS)}
    measured['j3'] += 0.4      # the arm is not where this plan assumes
    with pytest.raises(TrajectoryRejected, match='not where this plan assumes'):
        validate_trajectory(traj, start_position=measured)


def test_the_start_check_passes_when_the_arm_is_there():
    traj = build_trajectory(_ramp(3), dt=0.1)
    measured = {name: HOME[i] for i, name in enumerate(ARM_JOINTS)}
    assert validate_trajectory(traj, start_position=measured) is not None
