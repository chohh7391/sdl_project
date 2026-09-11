"""Every name and number this workspace borrows from cho_robot_project.

The FR5 arm is owned by a different repository (``~/ros2_ws/src/cho_robot_project``):
its controllers, its action servers, its joint limits. This module is the one
place those cross-repo facts are written down, so that when the other side
changes something -- a controller is renamed, the pour controller finally
lands -- exactly one file here has to follow.

Nothing in this module imports ROS. Import it to read a name; import
``cho_bridge.executor`` to use one.
"""

# --- controllers -----------------------------------------------------------
#
# Names from cho_robot_config/config/fr5.yaml, which is that repo's single
# source of truth for controller roles. The task-manager tree reads the same
# entry to build its exclusive-switch set, so a name that disagrees here is a
# switch that leaves two controllers claiming the arm.

#: controllers.moveit_trajectory -- the FollowJointTrajectory controller.
#: MoveIt is its other consumer: only one owner at a time.
TRAJECTORY_CONTROLLER = 'joint_trajectory_controller'

#: controllers.hold / direct_joint. Where the arm is parked between sessions.
HOLD_CONTROLLER = 'joint_space_position_controller'

#: controllers.direct_task. Not driven from here; listed so a switch can take
#: it down if an operator left it active.
TASK_CONTROLLER = 'task_space_ik_controller'

#: The closed-loop pouring controller. NOT BUILT YET on the cho side -- the
#: pour path here is written against this name and fails loudly until the
#: controller and its action exist. See README.md.
POUR_CONTROLLER = 'pouring_controller'

#: Controllers that claim the arm's command interfaces. Activating one of them
#: must deactivate the rest, which is what `executor.switch_to` derives its
#: deactivate list from. The gripper is deliberately absent: it claims the
#: finger interfaces and must survive an arm-controller switch.
ARM_CONTROLLERS = (
    TRAJECTORY_CONTROLLER,
    HOLD_CONTROLLER,
    TASK_CONTROLLER,
    POUR_CONTROLLER,
)

# --- endpoints -------------------------------------------------------------

CONTROLLER_MANAGER_NS = '/controller_manager'
SWITCH_CONTROLLER_SRV = f'{CONTROLLER_MANAGER_NS}/switch_controller'
LIST_CONTROLLERS_SRV = f'{CONTROLLER_MANAGER_NS}/list_controllers'

#: Stock joint_trajectory_controller endpoints. The action is for planned
#: segments (it reports abort reasons); the topic replaces the active
#: trajectory with no handshake, which is what a closed loop streams on.
FJT_ACTION = f'/{TRAJECTORY_CONTROLLER}/follow_joint_trajectory'
FJT_STREAM_TOPIC = f'/{TRAJECTORY_CONTROLLER}/joint_trajectory'

#: cho_task_manager/utils/controller_names.py: ACTION_SERVER_NAMESPACE.
CONTROLLER_ACTION_NS = '/controller_action_server'
GRIPPER_ACTION = f'{CONTROLLER_ACTION_NS}/gripper_controller'
POUR_ACTION = f'{CONTROLLER_ACTION_NS}/{POUR_CONTROLLER}'

#: Published by joint_state_broadcaster. Read it BY JOINT NAME: an AG-95 build
#: puts the gripper joint in the same message, so position[:6] is not the arm.
JOINT_STATES_TOPIC = '/joint_states'

#: Status topic the cho task tree watches to know the session is alive and,
#: on `SESSION_IDLE`, finished. std_msgs/String, latched on the publisher side.
SESSION_TOPIC = '/tamp_current_op'
SESSION_IDLE = 'idle'

# --- the arm ---------------------------------------------------------------

#: Joint order used by every goal built here. It already matches
#: TAMPServer.cmd_js_names, so no remapping happens anywhere.
ARM_JOINTS = ('j1', 'j2', 'j3', 'j4', 'j5', 'j6')

#: Position limits [rad] from cho_description_fr5/urdf/fr5_macro.xacro.
#: Checked before anything is sent: neither the stock trajectory controller nor
#: the vendor hardware write() clamps a commanded position.
JOINT_LIMITS = {
    'j1': (-3.0543, 3.0543),
    'j2': (-4.6251, 1.4835),
    'j3': (-2.8274, 2.8274),
    'j4': (-4.6251, 1.4835),
    'j5': (-3.0543, 3.0543),
    'j6': (-3.0543, 3.0543),
}

#: Joint-velocity ceiling [rad/s] every outgoing trajectory is checked against.
#: This is NOT the URDF limit (3.15 rad/s). It is the envelope the cho
#: controllers enforce on the real arm -- max_delta_q 0.005 rad at the 125 Hz
#: controller_manager rate -- and on the trajectory-controller path this check
#: is the only thing enforcing it, because the stock controller does not and
#: the vendor write() only rejects NaN.
MAX_JOINT_VELOCITY = 0.625

#: Smallest horizon [s] a streamed point may ask for. Below roughly two
#: controller cycles at 125 Hz the controller is being asked to arrive in less
#: time than it has cycles to get there.
MIN_STREAM_HORIZON = 0.016
