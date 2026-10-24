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

"""The pouring controller's registry wiring, which two packages have to agree on.

The controller itself is C++ and is exercised against a running arm. What is
checked here is the part that fails silently: whether the rest of the stack
knows the controller exists. If it does not reach the exclusive set, switching
another controller in never takes the pouring one down, both claim the same
command interfaces, and controller_manager refuses the switch -- leaving the arm
on the controller a failed pour was driving. The core's registry has no pour
role any more; sdl_cell_bringup's copy lists the controller in fr5's
``controllers.additional_arm``, and sourcing the cell workspace selects it.

Run against the install (colcon test sources it), so the registry is the one
the cell's environment hook selected.
"""

from pathlib import Path

import pytest

from cho_task_manager.utils.controller_names import (
    exclusive_arm_controllers,
    load_robot_config,
    valid_controller_action_names,
)
from sdl_cell_tasks.behaviors.action import PourActionBehavior
from sdl_cell_tasks.utils.pour_controller import (
    checked_pour_controller,
    DEFAULT_POUR_CONTROLLER,
    pour_action_name,
)

PACKAGE = Path(__file__).resolve().parents[1] / 'sdl_cell_tasks'


def test_the_cell_registry_lists_the_pour_controller_among_fr5s_arm_controllers():
    assert checked_pour_controller(load_robot_config('fr5'), DEFAULT_POUR_CONTROLLER) == \
        DEFAULT_POUR_CONTROLLER


def test_switching_away_takes_the_pour_controller_down():
    config = load_robot_config('fr5')
    assert DEFAULT_POUR_CONTROLLER in exclusive_arm_controllers(config)


def test_the_pour_action_keeps_its_own_name():
    # The C++ server keeps the name the action had before the core's contract.
    assert pour_action_name(DEFAULT_POUR_CONTROLLER) == '/controller_action_server/pouring_controller'


def test_a_pour_goal_is_addressable_from_a_behaviour():
    # The registry builds no pour name, so the behaviour is checked against the
    # name it does build for the controller, and then opens the pour endpoint.
    assert pour_action_name(DEFAULT_POUR_CONTROLLER) not in valid_controller_action_names()
    leaf = PourActionBehavior('pour', DEFAULT_POUR_CONTROLLER, target_grams=1.0, container_grams=1.0)
    assert leaf.action_name == pour_action_name(DEFAULT_POUR_CONTROLLER)


def test_the_pour_action_is_named_in_one_place():
    offenders = [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob('*.py')
                 if '/controller_action_server' in path.read_text()
                 and path.name != 'pour_controller.py']
    assert offenders == []


@pytest.mark.parametrize('robot', ['franka', 'ur5e', 'openarm'])
def test_robots_without_a_scale_have_no_pour_controller(robot):
    with pytest.raises(ValueError, match='Source the cell workspace'):
        checked_pour_controller(load_robot_config(robot), DEFAULT_POUR_CONTROLLER)


def test_a_pour_controller_the_registry_does_not_list_is_refused():
    with pytest.raises(ValueError, match="'pouring_controler' is not among fr5's arm controllers"):
        checked_pour_controller(load_robot_config('fr5'), 'pouring_controler')
