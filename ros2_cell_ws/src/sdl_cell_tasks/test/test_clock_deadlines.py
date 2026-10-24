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

"""Every deadline of this package's leaves waits for the node clock to start.

Under ``use_sim_time`` the node clock reads 0 until the first /clock message.
A deadline taken from that reading is far in the past once /clock arrives with
the simulator's real time, so the first tick after it would call the deadline
missed. ``cho_task_manager.utils.clock`` holds the rule; these tests pin this
package's leaves to it, as the core's test_clock_deadlines pins the core's.
"""

from pathlib import Path
from unittest.mock import MagicMock

import py_trees
import pytest
from rclpy.time import Time

from sdl_cell_tasks.behaviors.topic import GraspMarkerSampleBehavior, ScaleLatchBehavior

RUNNING = py_trees.common.Status.RUNNING
SUCCESS = py_trees.common.Status.SUCCESS
FAILURE = py_trees.common.Status.FAILURE
PACKAGE = Path(__file__).resolve().parents[1] / 'sdl_cell_tasks'


@pytest.fixture(autouse=True)
def clean_blackboard():
    py_trees.blackboard.Blackboard.clear()
    yield
    py_trees.blackboard.Blackboard.clear()


class NodeClock:
    """A node clock that only moves when the test says so -- like /clock."""

    def __init__(self, seconds):
        self.seconds = seconds

    def now(self):
        return Time(nanoseconds=int(round(self.seconds * 1e9)))


#: (leaf factory, its timeout [s]). Each is left without the input it waits
#: for, so the deadline is the only way out.
LEAVES = {
    'grasp_marker': (lambda: GraspMarkerSampleBehavior(
        'Grasp', marker_topic='/marker', required_frame='base', joint_names=['j1'],
        timeout_sec=5.0), 5.0),
    'scale_latch': (lambda: ScaleLatchBehavior(
        'Latch', record_as='zero', settle_sec=2.0, timeout_sec=6.0), 6.0),
}


def _wire(factory, start):
    leaf = factory()
    clock = NodeClock(start)
    leaf.node = MagicMock()
    leaf.node.get_clock.return_value = clock
    return leaf, clock


@pytest.mark.parametrize('kind', sorted(LEAVES))
def test_a_deadline_from_before_the_clock_started_is_not_missed_when_it_does(kind):
    factory, timeout = LEAVES[kind]
    leaf, clock = _wire(factory, start=0.0)
    leaf.initialise()
    assert leaf.update() == RUNNING

    clock.seconds = 5000.0           # /clock arrives, far past zero
    assert leaf.update() == RUNNING
    clock.seconds += timeout - 0.1
    assert leaf.update() == RUNNING
    clock.seconds += 0.2             # the full timeout after the clock started
    assert leaf.update() == FAILURE


@pytest.mark.parametrize('kind', sorted(LEAVES))
def test_a_running_clock_times_out_exactly_as_before(kind):
    factory, timeout = LEAVES[kind]
    leaf, clock = _wire(factory, start=100.0)
    leaf.initialise()
    clock.seconds += timeout - 0.1
    assert leaf.update() == RUNNING
    clock.seconds += 0.2
    assert leaf.update() == FAILURE


def _reading(grams):
    msg = MagicMock()
    msg.grams, msg.stable = grams, True
    return msg


def test_a_scale_hold_that_began_before_the_clock_is_timed_from_its_first_reading():
    # Stamped 0, the hold would read as 5000 s settled the moment /clock came,
    # and an unsettled pan would become every later pour's zero.
    leaf, clock = _wire(LEAVES['scale_latch'][0], start=0.0)
    leaf.initialise()
    for _ in range(3):
        leaf._on_reading(_reading(139.15))
        assert leaf.update() == RUNNING

    clock.seconds = 5000.0
    leaf._on_reading(_reading(139.15))
    assert leaf.update() == RUNNING
    clock.seconds += 1.9
    leaf._on_reading(_reading(139.15))
    assert leaf.update() == RUNNING
    clock.seconds += 0.2
    leaf._on_reading(_reading(139.15))
    assert leaf.update() == SUCCESS


def test_no_behaviour_takes_a_deadline_off_the_clock_directly():
    offenders = [
        f'{path.relative_to(PACKAGE)}:{number}'
        for path in (PACKAGE / 'behaviors').rglob('*.py')
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if 'Duration(seconds' in line
    ]
    assert offenders == [], 'take deadlines through cho_task_manager.utils.clock.deadline_after / arm'
