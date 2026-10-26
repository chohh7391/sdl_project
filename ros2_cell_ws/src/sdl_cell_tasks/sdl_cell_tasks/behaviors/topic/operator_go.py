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

"""Wait for the operator to say go: one std_msgs/Empty on a topic, sent after this started.

A trial series runs one step per go: between pours the operator refills or
empties a vessel, resets a camera, or simply decides the cell is ready, and
the tree holds the arm still until they say so. Nothing that arrived before the
wait began counts -- a go from the last trial does not start the next.

    ros2 topic pub --once /real_cell/trial_go std_msgs/msg/Empty
"""

import py_trees
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Empty

from cho_task_manager.utils.clock import arm, deadline_after

DEFAULT_GO_TOPIC = '/real_cell/trial_go'


class OperatorGoBehavior(py_trees.behaviour.Behaviour):
    """RUNNING until a message arrives on *topic* after this behaviour started."""

    def __init__(self, name: str, topic: str = DEFAULT_GO_TOPIC, remind_sec: float = 30.0):
        super().__init__(name)
        if not topic:
            raise ValueError(f'[{name}] topic is required')
        if not remind_sec > 0.0:
            raise ValueError(f'[{name}] remind_sec must be positive (got {remind_sec})')
        self.topic = topic
        self.remind_sec = remind_sec
        self.node = None
        self.subscription = None
        self._armed = False
        self._go = False
        self._next_reminder = None

    def setup(self, **kwargs):
        self.node = kwargs['node']
        self.subscription = self.node.create_subscription(
            Empty, self.topic, self._on_go, 10, callback_group=ReentrantCallbackGroup())
        return True

    def _on_go(self, _msg):
        if self._armed:
            self._go = True

    def initialise(self):
        self._go = False
        self._armed = True
        # The first reminder goes out on the first tick the node clock has started.
        self._next_reminder = None

    def update(self):
        if self._go:
            self.node.get_logger().info(f'[{self.name}] go received on {self.topic}')
            return py_trees.common.Status.SUCCESS
        clock = self.node.get_clock()
        self._next_reminder = arm(self._next_reminder, clock, 0.0)
        if self._next_reminder is not None and clock.now() >= self._next_reminder:
            self.node.get_logger().info(
                f'[{self.name}] holding still until the operator says go: '
                f'ros2 topic pub --once {self.topic} std_msgs/msg/Empty')
            self._next_reminder = deadline_after(clock, self.remind_sec)
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self._armed = False
        self._go = False


__all__ = ['DEFAULT_GO_TOPIC', 'OperatorGoBehavior']
