"""Send commands to the real FR5, through the controllers cho_robot_project owns.

This is the whole command-out surface of this workspace. Planning stays where
it is; what changes on real hardware is where the motion goes -- not a stream
of joint states into the simulator, but goals to the arm's own controllers:

* planned segments  -> FollowJointTrajectory action (it reports why it aborted)
* a closed loop     -> the controller's trajectory topic (replaces the active
                       trajectory with no handshake, so a 25 Hz loop pays none)
* the gripper       -> the cho Gripper action
* a pour            -> the cho pour controller, with the controller switch
                       around it (NOT BUILT YET -- see README.md)

Two rules hold everywhere here.

**Nothing is sent unvalidated.** The stock trajectory controller enforces
tolerances, not limits, and the vendor hardware only rejects NaN, so
`trajectory.validate_trajectory` is the last gate before a real arm. Refusing
to send is always better than finding out.

**Nothing here spins.** Every call blocks on a future while SOMEBODY ELSE
spins the node -- tamp_server's MultiThreadedExecutor, or the executor thread
the standalone bridge node starts. Passing a node that nothing spins makes
every call here time out, which is the confusing way to discover it.
"""

import time

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory

from cho_bridge.contract import (
    ARM_CONTROLLERS,
    ARM_JOINTS,
    FJT_ACTION,
    FJT_STREAM_TOPIC,
    GRIPPER_ACTION,
    HOLD_CONTROLLER,
    JOINT_STATES_TOPIC,
    LIST_CONTROLLERS_SRV,
    POUR_ACTION,
    POUR_CONTROLLER,
    SESSION_IDLE,
    SESSION_TOPIC,
    SWITCH_CONTROLLER_SRV,
    TRAJECTORY_CONTROLLER,
)
from cho_bridge.trajectory import single_point_trajectory, validate_trajectory

#: How long after a trajectory's own duration a goal is still considered live.
#: The controller's goal_time tolerance is 1.5 s, so anything past that plus a
#: margin is the action itself not answering rather than the arm running late.
GOAL_MARGIN_SEC = 8.0

_FJT_ERRORS = {
    FollowJointTrajectory.Result.SUCCESSFUL: 'SUCCESSFUL',
    FollowJointTrajectory.Result.INVALID_GOAL: 'INVALID_GOAL',
    FollowJointTrajectory.Result.INVALID_JOINTS: 'INVALID_JOINTS',
    FollowJointTrajectory.Result.OLD_HEADER_TIMESTAMP: 'OLD_HEADER_TIMESTAMP',
    FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED: 'PATH_TOLERANCE_VIOLATED',
    FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED: 'GOAL_TOLERANCE_VIOLATED',
}


class ChoCommandFailed(RuntimeError):
    """A command to the arm was refused, aborted, or never answered."""


class ChoExecutor:
    """Command surface for one FR5, over one already-spinning node."""

    def __init__(self, node, joints=ARM_JOINTS, arm_controllers=ARM_CONTROLLERS,
                 callback_group=None):
        self.node = node
        self.joints = list(joints)
        self.arm_controllers = list(arm_controllers)
        # Reentrant so a blocking call made from inside a callback (which is
        # how tamp_server executes a plan) can still be served.
        self._group = callback_group or ReentrantCallbackGroup()

        self._switch_cli = node.create_client(
            SwitchController, SWITCH_CONTROLLER_SRV, callback_group=self._group)
        self._list_cli = node.create_client(
            ListControllers, LIST_CONTROLLERS_SRV, callback_group=self._group)
        self._fjt = ActionClient(
            node, FollowJointTrajectory, FJT_ACTION, callback_group=self._group)
        self._stream_pub = node.create_publisher(JointTrajectory, FJT_STREAM_TOPIC, 1)
        self._session_pub = node.create_publisher(String, SESSION_TOPIC, 1)

        # Built on first use: the gripper and the pour controller carry
        # cho_interfaces types, and the pour action does not exist yet.
        self._gripper = None
        self._pour = None

        self._joint_state = None
        node.create_subscription(
            JointState, JOINT_STATES_TOPIC, self._on_joint_state, 10,
            callback_group=self._group)

        self._goal_handle = None
        # Last point handed to the streaming topic. A stream is validated
        # against this rather than against the measured arm: a position
        # controller always lags a moving command, and that lag is the
        # thing it is busy closing, not a jump in the command.
        self._last_stream_point = None

    # -- state ------------------------------------------------------------

    def _on_joint_state(self, msg):
        self._joint_state = msg

    def joint_positions(self, timeout_sec=5.0):
        """Measured arm positions as ``{joint: rad}``.

        By name, never by slice: an AG-95 build publishes the gripper joint in
        the same message, so ``position[:6]`` is not necessarily the arm.
        """
        deadline = time.monotonic() + timeout_sec
        while self._joint_state is None:
            if time.monotonic() > deadline:
                raise ChoCommandFailed(
                    f'no {JOINT_STATES_TOPIC} within {timeout_sec}s; is the '
                    'bringup running with joint_state_broadcaster active?')
            time.sleep(0.02)

        msg = self._joint_state
        index = {name: i for i, name in enumerate(msg.name)}
        missing = [name for name in self.joints if name not in index]
        if missing:
            raise ChoCommandFailed(
                f'{JOINT_STATES_TOPIC} does not carry {missing}; it has '
                f'{list(msg.name)}')
        return {name: float(msg.position[index[name]]) for name in self.joints}

    def wait_for_stack(self, timeout_sec=10.0):
        """Names of the endpoints that did not come up in time (empty is good)."""
        deadline = time.monotonic() + timeout_sec
        missing = []
        for label, ready in (
            (SWITCH_CONTROLLER_SRV, lambda: self._switch_cli.service_is_ready()),
            (LIST_CONTROLLERS_SRV, lambda: self._list_cli.service_is_ready()),
            (FJT_ACTION, lambda: self._fjt.server_is_ready()),
        ):
            while not ready():
                if time.monotonic() > deadline:
                    missing.append(label)
                    break
                time.sleep(0.05)
        return missing

    # -- controllers ------------------------------------------------------

    def _call(self, client, request, timeout_sec, what):
        if not client.service_is_ready() and not client.wait_for_service(timeout_sec=timeout_sec):
            raise ChoCommandFailed(f'{what}: service {client.srv_name} unavailable')
        future = client.call_async(request)
        return self._wait(future, timeout_sec, what)

    def _wait(self, future, timeout_sec, what):
        deadline = time.monotonic() + timeout_sec
        while not future.done():
            if time.monotonic() > deadline:
                raise ChoCommandFailed(
                    f'{what}: no answer within {timeout_sec}s. Nothing spins '
                    'this node unless tamp_server or the bridge node does.')
            time.sleep(0.01)
        return future.result()

    def controller_states(self, timeout_sec=5.0):
        """``{controller: state}`` as the controller manager reports it."""
        response = self._call(
            self._list_cli, ListControllers.Request(), timeout_sec, 'list_controllers')
        return {entry.name: entry.state for entry in response.controller}

    def switch_to(self, controller, timeout_sec=5.0, verify=True):
        """Give the arm to *controller*, taking every other arm controller down.

        BEST_EFFORT on purpose -- deactivating a controller that was already
        inactive is not an error worth failing on -- and therefore verified
        afterwards: with BEST_EFFORT, activating a controller the bringup never
        loaded still answers ``ok``, and nothing would be holding the arm.
        """
        request = SwitchController.Request()
        request.activate_controllers = [controller]
        request.deactivate_controllers = [
            name for name in self.arm_controllers if name != controller]
        request.strictness = SwitchController.Request.BEST_EFFORT
        response = self._call(self._switch_cli, request, timeout_sec, 'switch_controller')
        if response is None or not response.ok:
            raise ChoCommandFailed(f'switch to {controller} was refused')

        if verify:
            states = self.controller_states(timeout_sec)
            if states.get(controller) != 'active':
                raise ChoCommandFailed(
                    f"switch to {controller} reported ok but it is "
                    f"'{states.get(controller, 'not loaded')}'. Was it spawned "
                    'by the bringup?')
        self.node.get_logger().info(f'arm is on {controller}')
        return controller

    def hold(self, timeout_sec=5.0):
        """Cancel whatever is running and park the arm on its hold controller."""
        self.cancel(timeout_sec=timeout_sec, quiet=True)
        return self.switch_to(HOLD_CONTROLLER, timeout_sec=timeout_sec)

    # -- motion -----------------------------------------------------------

    def execute_trajectory(self, traj, timeout_sec=None, validate=True,
                           check_start=True, max_velocity=None):
        """Run *traj* through the trajectory controller and wait for its result.

        Raises before anything is sent when the trajectory does not pass
        :func:`~cho_bridge.trajectory.validate_trajectory`, and raises with the
        controller's own error name when the goal is rejected or aborted.
        """
        if validate:
            kwargs = {}
            if max_velocity is not None:
                kwargs['max_velocity'] = max_velocity
            if check_start:
                kwargs['start_position'] = self.joint_positions()
            validate_trajectory(traj, joint_names=self.joints, **kwargs)

        if timeout_sec is None:
            last = traj.points[-1].time_from_start
            timeout_sec = last.sec + last.nanosec * 1e-9 + GOAL_MARGIN_SEC

        if not self._fjt.server_is_ready() and not self._fjt.wait_for_server(timeout_sec=5.0):
            raise ChoCommandFailed(f'no action server at {FJT_ACTION}')

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        send = self._fjt.send_goal_async(goal)
        handle = self._wait(send, 10.0, 'follow_joint_trajectory goal')
        if handle is None or not handle.accepted:
            raise ChoCommandFailed('the trajectory goal was rejected')

        self._goal_handle = handle
        # The goal drives the arm away from wherever the stream left it.
        self._last_stream_point = None
        try:
            result = self._wait(
                handle.get_result_async(), timeout_sec, 'follow_joint_trajectory result')
        finally:
            self._goal_handle = None

        if result.status != GoalStatus.STATUS_SUCCEEDED:
            code = _FJT_ERRORS.get(result.result.error_code, result.result.error_code)
            raise ChoCommandFailed(
                f'trajectory did not complete: {code} '
                f'{result.result.error_string}'.strip())
        return result.result

    def stream_point(self, position, horizon_sec=0.1, velocity=None, validate=True):
        """Publish one point on the controller's trajectory topic.

        This is the closed-loop primitive: the published trajectory replaces the
        running one, so the caller's loop rate and *horizon_sec* together are
        the commanded joint rate. Nothing downstream bounds it.
        """
        traj = single_point_trajectory(
            position, joint_names=self.joints, horizon_sec=horizon_sec, velocity=velocity)
        if validate:
            # The rate that matters is the one this point asks for relative to
            # the point before it: at most one horizon's worth of travel at the
            # ceiling. The FIRST point of a stream has no predecessor, so it is
            # checked against the measured arm -- which is right exactly there,
            # because that is the one moment the command has not moved yet.
            reference = self._last_stream_point or self.joint_positions()
            validate_trajectory(
                traj,
                joint_names=self.joints,
                start_position=reference,
                max_start_jump=max_velocity_step(horizon_sec),
            )
        self._stream_pub.publish(traj)
        self._last_stream_point = {
            name: float(value) for name, value in zip(self.joints, traj.points[0].positions)}
        return traj

    def cancel(self, timeout_sec=5.0, quiet=False):
        """Cancel the trajectory goal in flight, if there is one."""
        handle = self._goal_handle
        if handle is None:
            return False
        try:
            self._wait(handle.cancel_goal_async(), timeout_sec, 'cancel')
        except ChoCommandFailed:
            if not quiet:
                raise
            return False
        self.node.get_logger().warn('trajectory goal cancelled')
        return True

    # -- gripper ----------------------------------------------------------

    def set_gripper(self, grasp, width=0.0, speed=0.0, force=0.0, timeout_sec=20.0):
        """Open or close the AG-95 through the cho gripper action.

        Only loaded when the description is expanded with ``gripper:=ag95``; on
        a build without it there is no action server and this says so rather
        than hanging.
        """
        client, action_type = self._gripper_client()
        if not client.server_is_ready() and not client.wait_for_server(timeout_sec=5.0):
            raise ChoCommandFailed(
                f'no action server at {GRIPPER_ACTION}; was the bringup started '
                'with gripper:=ag95?')
        goal = action_type.Goal()
        goal.grasp = bool(grasp)
        goal.width = float(width)
        goal.speed = float(speed)
        goal.force = float(force)
        handle = self._wait(client.send_goal_async(goal), 10.0, 'gripper goal')
        if handle is None or not handle.accepted:
            raise ChoCommandFailed('the gripper goal was rejected')
        result = self._wait(handle.get_result_async(), timeout_sec, 'gripper result')
        if result.status != GoalStatus.STATUS_SUCCEEDED or not result.result.is_completed:
            raise ChoCommandFailed(
                f"gripper {'close' if grasp else 'open'} did not complete")
        return result.result

    def _gripper_client(self):
        if self._gripper is None:
            try:
                from cho_interfaces.action import Gripper
            except ImportError as error:
                raise ChoCommandFailed(
                    'cho_interfaces is not on the path; source the '
                    'cho_robot_project workspace before driving the gripper'
                ) from error
            self._gripper = (
                ActionClient(self.node, Gripper, GRIPPER_ACTION,
                             callback_group=self._group),
                Gripper,
            )
        return self._gripper

    # -- pour -------------------------------------------------------------

    def pour(self, target_grams, timeout_sec=180.0, **goal_fields):
        """Switch to the pour controller, pour, and switch the arm back.

        The pour itself is closed on the scale INSIDE that controller -- this
        only owns the handover around it, because the trajectory controller and
        the pour controller claim the same command interfaces and exactly one
        of them may hold the arm.

        The cho side does not have this controller yet, so this path raises a
        clear error rather than pretending. Everything it needs is named in
        ``contract.py``.
        """
        client, action_type = self._pour_client()
        if not client.server_is_ready() and not client.wait_for_server(timeout_sec=5.0):
            raise ChoCommandFailed(
                f'no action server at {POUR_ACTION}. The {POUR_CONTROLLER} is '
                'not built/spawned on the cho side yet')

        previous = TRAJECTORY_CONTROLLER
        self.switch_to(POUR_CONTROLLER)
        try:
            goal = action_type.Goal()
            goal.target_grams = float(target_grams)
            for field, value in goal_fields.items():
                setattr(goal, field, value)
            handle = self._wait(client.send_goal_async(goal), 10.0, 'pour goal')
            if handle is None or not handle.accepted:
                raise ChoCommandFailed('the pour goal was rejected')
            result = self._wait(handle.get_result_async(), timeout_sec, 'pour result')
            if result.status != GoalStatus.STATUS_SUCCEEDED:
                # The controller says WHICH bound stopped it -- tilt cap, quiet
                # scale, timeout. Dropping that here would turn the one useful
                # thing about a refused pour into "it failed".
                reason = getattr(result.result, 'message', '') or 'no reason reported'
                raise ChoCommandFailed(
                    'the pour did not complete: %s (poured %.2f g, peak tilt %.3f rad)'
                    % (reason, getattr(result.result, 'final_grams', float('nan')),
                       getattr(result.result, 'peak_tilt', float('nan'))))
            return result.result
        finally:
            # Always hand the arm back, including after a failed pour: leaving
            # it on a controller nothing is commanding is how a session ends
            # with the arm unattended.
            self.switch_to(previous)

    def _pour_client(self):
        if self._pour is None:
            try:
                from cho_interfaces.action import Pour
            except ImportError as error:
                raise ChoCommandFailed(
                    'cho_interfaces has no Pour action yet: the pouring '
                    'controller and its interface are still to be built on the '
                    'cho_robot_project side'
                ) from error
            self._pour = (
                ActionClient(self.node, Pour, POUR_ACTION, callback_group=self._group),
                Pour,
            )
        return self._pour

    # -- session status ---------------------------------------------------

    def announce(self, operator):
        """Publish what is running, for the cho task tree that supervises this.

        ``ExternalSessionBehavior`` holds its tree open on this topic and ends
        the session on :data:`~cho_bridge.contract.SESSION_IDLE`, so publish
        that in a ``finally`` -- including after a failure. Silence only ever
        times out.
        """
        message = String()
        message.data = str(operator)
        self._session_pub.publish(message)

    def announce_idle(self):
        self.announce(SESSION_IDLE)


def max_velocity_step(horizon_sec, max_velocity=None):
    """How far a joint may move in *horizon_sec* at the velocity ceiling."""
    from cho_bridge.contract import MAX_JOINT_VELOCITY
    ceiling = MAX_JOINT_VELOCITY if max_velocity is None else max_velocity
    return ceiling * horizon_sec
