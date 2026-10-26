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

"""Whether the closed jaws are holding something: the finger joint stopped short of shut.

The AG-95 reports one stroke position and nothing about force, so the only sign
that a grasp found its object is where the jaws stopped: on the beaker they
stop at its diameter (0.024 m of finger travel on the real cell's 50 mm
beaker, 51 % open), on nothing they close to 0.

The result goes to the log as one line a trial series can be scored from:
``grasp HELD: jaws at 24.2 mm`` or ``grasp MISSED: jaws at 0.3 mm``.
"""

import math

import py_trees
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from cho_task_manager.utils.clock import arm, deadline_after

#: The AG-95's finger joint, as the FR5 bringups publish it on /joint_states
#: (cho_bringup_fr5's controllers.yaml gripper_joint).
DEFAULT_FINGER_JOINT = 'gripper_finger_joint'


class GripperHoldCheckBehavior(py_trees.behaviour.Behaviour):
    """SUCCESS when the finger joint reads at least *min_position*, FAILURE when below."""

    def __init__(self, name: str, min_position: float = 0.005,
                 joint: str = DEFAULT_FINGER_JOINT, topic: str = '/joint_states',
                 timeout_sec: float = 5.0):
        super().__init__(name)
        if not (math.isfinite(min_position) and min_position > 0.0):
            raise ValueError(f'[{name}] min_position must be positive (got {min_position})')
        self.min_position = min_position
        self.joint = joint
        self.topic = topic
        self.timeout_sec = timeout_sec
        self.node = None
        self.subscription = None
        self._position = None
        self._deadline = None

    def setup(self, **kwargs):
        self.node = kwargs['node']
        self.subscription = self.node.create_subscription(
            JointState, self.topic, self._on_state, qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup())
        return True

    def _on_state(self, msg):
        if self.joint in msg.name:
            self._position = msg.position[msg.name.index(self.joint)]

    def initialise(self):
        self._position = None
        # None while the node clock reads 0 (utils/clock.py); update() takes it then.
        self._deadline = deadline_after(self.node.get_clock(), self.timeout_sec)

    def update(self):
        if self._position is None:
            clock = self.node.get_clock()
            self._deadline = arm(self._deadline, clock, self.timeout_sec)
            if self._deadline is not None and clock.now() > self._deadline:
                self.node.get_logger().error(
                    f'[{self.name}] grasp UNKNOWN: no {self.joint} on {self.topic} within '
                    f'{self.timeout_sec:.0f} s')
                return py_trees.common.Status.FAILURE
            return py_trees.common.Status.RUNNING
        held = self._position >= self.min_position
        self.node.get_logger().info(
            f'[{self.name}] grasp {"HELD" if held else "MISSED"}: jaws at '
            f'{1000.0 * self._position:.1f} mm of finger travel (held from '
            f'{1000.0 * self.min_position:.1f} mm)')
        return py_trees.common.Status.SUCCESS if held else py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        self._deadline = None


__all__ = ['DEFAULT_FINGER_JOINT', 'GripperHoldCheckBehavior']
