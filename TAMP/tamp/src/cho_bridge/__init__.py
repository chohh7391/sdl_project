"""Commanding the real FR5, which another workspace owns.

`contract` names everything borrowed from cho_robot_project, `trajectory`
builds and checks what gets sent, `executor` sends it, `scale_relay` adapts the
RS-232 scale into the plain number a control loop wants.
"""

from cho_bridge.executor import ChoCommandFailed, ChoExecutor
from cho_bridge.trajectory import (
    TrajectoryRejected,
    build_trajectory,
    single_point_trajectory,
    validate_trajectory,
)

__all__ = [
    'ChoCommandFailed',
    'ChoExecutor',
    'TrajectoryRejected',
    'build_trajectory',
    'single_point_trajectory',
    'validate_trajectory',
]
