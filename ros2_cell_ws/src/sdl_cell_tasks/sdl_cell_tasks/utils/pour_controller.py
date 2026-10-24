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

"""The pouring controller: its name, the one place its action is named, and the registry check.

The pour is the cell's, not the core's: the core registry has no pour role and
names no pour action. The cell's bringup loads ``pouring_controller``
(sdl_pour_controller), and sdl_cell_bringup's copy of the registry lists it in
fr5's ``controllers.additional_arm``, so every exclusive switch -- the core's
``guarded_mission()`` abort included -- takes it down. A tree that switches to
it relies on that, so it refuses to be built when the registry it was given does
not list the controller (:func:`checked_pour_controller`).
"""

from cho_task_manager.utils.controller_names import exclusive_arm_controllers

#: The controller sdl_cell_bringup loads, and task_params' default for ``pour_controller``.
DEFAULT_POUR_CONTROLLER = 'pouring_controller'

#: The namespace the pour action server keeps: ``/controller_action_server/<controller>``,
#: the name the action had before the core's action contract, and still has.
POUR_ACTION_NAMESPACE = '/controller_action_server'


def pour_action_name(controller):
    """The Pour action *controller* serves. Every pour client in this package names it here."""
    node = str(controller or '').strip('/')
    if not node:
        raise ValueError('the pour controller name must be non-empty')
    return f'{POUR_ACTION_NAMESPACE}/{node}'


def checked_pour_controller(robot_config, controller):
    """*controller* if the robot registry lists it among the arm's exclusive controllers.

    Raises ValueError otherwise, before anything is built: a tree that switches
    the arm to a controller the registry does not know would leave it active
    through every later exclusive switch, the abort's included.
    """
    if not isinstance(controller, str) or not controller.strip():
        raise ValueError(f'pour_controller must be a controller name; got {controller!r}')
    controller = controller.strip()
    exclusive = exclusive_arm_controllers(robot_config)
    if controller not in exclusive:
        raise ValueError(
            f"the pour controller '{controller}' is not among {robot_config.get('robot_type')}'s "
            f'arm controllers in the robot registry ({exclusive}), so no exclusive switch would '
            'take it down. Source the cell workspace (its install/setup.bash points '
            "CHO_ROBOT_CONFIG_DIR at sdl_cell_bringup's registry, which adds it), or name the "
            'pour controller its bringup loads.')
    return controller


__all__ = [
    'DEFAULT_POUR_CONTROLLER',
    'POUR_ACTION_NAMESPACE',
    'checked_pour_controller',
    'pour_action_name',
]
