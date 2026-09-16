#!/usr/bin/env python3
"""The same TAMP server, driving a real FR5 instead of a simulator.

This is deliberately NOT a copy of `tamp_server.py`. It subclasses it, so the
planner, the plan loop, both pour control laws and every planning service are
the ones the simulated runs use -- literally the same code objects. Only the
plant changes, through the hooks `TAMPServer` exposes for it:

    _start_execution / _finish_execution   session boundaries
    _publish_arm_command                   one arm configuration -> the plant
    _arm_q                                 measured arm configuration
    _execute_trajectory_step               replay one planned segment
    execute_gripper_action                 open/close
    _perception_pose / set_tamp_env_cb     where the World State comes from

The perception RECOVERY ladder is not in that list on purpose: it is inherited
whole from the base (`_localize_with_recovery`), which reaches this plant
through the hooks above. Re-implementing it here would give the physical cell a
different recovery policy from the one the paper's numbers were measured under.

That matters beyond tidiness: the adaptive pour the paper reports has to be one
implementation. A forked file would let the simulated law and the physical law
drift while both were still called "the same controller".

What the plant is here:

* motion goes to `joint_trajectory_controller` in cho_robot_project -- planned
  segments as one FollowJointTrajectory goal, the pour's closed loop as points
  streamed on that controller's trajectory topic. `cho_bridge` owns that, and
  validates everything before it is sent.
* the World State comes from `cho_object_pose`, which publishes one
  `PoseStamped` per tagged object in the arm's base frame. Untagged furniture
  has no tag to detect in either cell, so it is read from a static pose file.
* the arm's controller state is owned by the `fjt_handover` task tree, not by
  this process: two agents switching controllers race each other. By default
  this server only VERIFIES that the trajectory controller is active.

Run it exactly like the simulated one::

    scripts/run_tamp.sh              # tamp_server.py
    python tamp_real_server.py       # this, against a real (or MuJoCo) bringup

Environment:
    SDL_STATE_SOURCE=perception   as in the base; `ground_truth` is refused here
    SDL_STATIC_POSES=<path>       poses for the untagged entities
    SDL_REAL_SWITCH=verify|switch whether this process may switch controllers
    SDL_STREAM_HORIZON=0.10       lookahead of one streamed point [s]
    SDL_GROUND_ON_TABLE=1         stand perceived vessels on the bench (see below)
    SDL_RECOVERY*                 perception recovery ladder; see tamp_server.py
"""

import os

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from cho_bridge.contract import ARM_JOINTS, TRAJECTORY_CONTROLLER
from cho_bridge.executor import ChoCommandFailed, ChoExecutor
from cho_bridge.trajectory import build_trajectory, required_time_scale
from tamp_server import TAMPServer, main as tamp_main


class EStopRequested(RuntimeError):
    """An operator stopped the plan between commands."""


class RealTAMPServer(TAMPServer):
    """TAMPServer whose plant is a real FR5 under cho_robot_project."""

    #: cho_object_pose publishes one topic per object in its table.
    OBJECT_POSE_TOPIC = '/perception/object_pose/%s'

    #: Stand a perceived vessel on the bench instead of believing the camera's
    #: estimate of its height.
    #:
    #: This is not a fudge and it is not optional book-keeping: a single camera
    #: constrains a tag's position across the image well and along its own view
    #: ray poorly, and the ray of a camera looking down at the bench is mostly
    #: vertical. Measured in the MuJoCo cell, the beaker's perceived z was 12.1
    #: mm low - enough that the planner refused the scene outright, because a
    #: 0.135 m vessel centred 12 mm low has its base 7 mm inside the table.
    #:
    #: A vessel standing on a bench of known height has a known height. So x, y
    #: and yaw come from perception, and z comes from the bench. The simulated
    #: path did the same thing (the TF branch of the original set_tamp_env_cb
    #: forced the vessels' z), and this states it instead of hiding it: what is
    #: perception-in-the-loop here is the horizontal placement.
    GROUND_PERCEIVED_ON_TABLE = os.environ.get('SDL_GROUND_ON_TABLE', '1') == '1'

    #: The frame cho_object_pose resolves into for this robot
    #: (cho_robot_config fr5.yaml model.arm_base_link). Nothing here transforms
    #: frames, so a pose in any other frame is refused rather than obeyed.
    BASE_FRAME = 'base_link'

    def __init__(self, tamp):
        super().__init__(tamp)

        self.arm = ChoExecutor(self, joints=ARM_JOINTS,
                               callback_group=self.reentrant_group)
        self.stream_horizon = float(os.environ.get('SDL_STREAM_HORIZON', '0.10'))
        self.may_switch = os.environ.get('SDL_REAL_SWITCH', 'verify') == 'switch'
        self._estop = False

        # The base subscribes to isaac_joint_states, which nothing publishes
        # here. Its callback is reused unchanged: only the topic differs.
        self.create_subscription(
            JointState, '/joint_states', self.joint_states_cb, 10,
            callback_group=self.reentrant_group)

        # One latched pose per tagged object, straight off cho_object_pose.
        # Subscribing per object rather than reading TF on purpose: the gates
        # that decide a detection is good enough (agreement window, spread)
        # live in that node, and a pose only reaches its topic once it passed
        # them. A TF lookup would see poses that did not.
        self._object_poses = {}
        self._newest_object_stamp = None
        for entity in self.PERCEPTION_ENTITIES:
            topic = self.OBJECT_POSE_TOPIC % entity
            self.create_subscription(
                PoseStamped, topic,
                lambda msg, name=entity: self._on_object_pose(name, msg), 10,
                callback_group=self.reentrant_group)
            self.get_logger().info('[state] %s <- %s' % (entity, topic))

        self._static_poses = self._load_static_poses()
        self._table_top_z = self._load_table_top()

        self.estop_srv = self.create_service(
            Trigger, 'tamp_estop', self._estop_cb,
            callback_group=self.reentrant_group)

        self.get_logger().info(
            'RealTAMPServer ready: motion -> %s, controller switching %s'
            % (TRAJECTORY_CONTROLLER,
               'owned here' if self.may_switch else 'left to the task tree'))

    # -- World State ------------------------------------------------------

    def _load_static_poses(self):
        """Poses for entities that carry no tag (table, stirrer, trays).

        The simulated path asks the simulator for these. There is no simulator
        here and no tag on them either, so they come from a file that the cell's
        layout owns. A missing entry is an error at use, not a zero pose.
        """
        path = os.environ.get('SDL_STATIC_POSES', '').strip()
        if not path:
            return {}
        with open(path, encoding='utf-8') as stream:
            loaded = yaml.safe_load(stream) or {}
        poses = {name: [float(v) for v in values]
                 for name, values in (loaded.get('poses') or {}).items()}
        for name, values in poses.items():
            if len(values) != 7:
                raise ValueError(
                    'static pose for %s must be [x, y, z, qw, qx, qy, qz], got %d values'
                    % (name, len(values)))
        self.get_logger().info(
            '[state] static poses for %s from %s' % (sorted(poses), path))
        return poses

    def _load_table_top(self):
        """Bench height [m] in the robot frame, or None if the cell did not say.

        Read from the same file as the static poses so the number that grounds a
        vessel is the cell's, not a constant compiled in here.
        """
        path = os.environ.get('SDL_STATIC_POSES', '').strip()
        if not path or not self.GROUND_PERCEIVED_ON_TABLE:
            return None
        with open(path, encoding='utf-8') as stream:
            loaded = yaml.safe_load(stream) or {}
        top = loaded.get('table_top_z')
        if top is None:
            self.get_logger().warn(
                '[state] SDL_GROUND_ON_TABLE is set but %s declares no '
                'table_top_z, so perceived vessels keep the camera\'s height '
                'estimate' % path)
            return None
        self.get_logger().info('[state] bench top at z = %.4f m' % float(top))
        return float(top)

    def _grounded_z(self, entity):
        """Centre height of *entity* standing on the bench, or None.

        The dimensions are the planner's own (cutamp envs.utils.ENTITIES), so a
        vessel is grounded to exactly the height the collision model expects it
        at - which is the whole point, since the alternative is the planner
        rejecting its own initial state.
        """
        if self._table_top_z is None:
            return None
        try:
            from envs.utils import ENTITIES
        except ImportError:
            return None
        entry = ENTITIES.get(entity)
        dims = getattr(entry, 'dims', None)
        if not dims:
            return None
        return self._table_top_z + float(dims[2]) / 2.0

    def _on_object_pose(self, entity, msg):
        self._object_poses[entity] = msg
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._newest_object_stamp is None or stamp > self._newest_object_stamp:
            self._newest_object_stamp = stamp

    def _perception_pose(self, entity):
        """base_link -> entity from cho_object_pose, or None.

        Same return convention as the base: the object's own pose, w-first
        quaternion, no z correction (the planner lift is applied downstream in
        envs/utils.py, so adding one here would double it).

        The age is measured against the newest stamp seen on ANY object-pose
        topic rather than this node's clock, for the reason the base gives for
        using the camera clock: the perception chain may be stamping from a
        simulation clock whose epoch is nowhere near wall time, and a wall-clock
        age would then reject every pose.
        """
        msg = self._object_poses.get(entity)
        if msg is None:
            self.get_logger().warn(
                '[state] %s has no perception pose yet on %s'
                % (entity, self.OBJECT_POSE_TOPIC % entity))
            return None

        if msg.header.frame_id != self.BASE_FRAME:
            self.get_logger().error(
                "[state] %s was published in frame '%s', expected '%s'. "
                'Nothing here transforms frames, so using it would place the '
                'object somewhere it is not.'
                % (entity, msg.header.frame_id, self.BASE_FRAME))
            return None

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        age = (self._newest_object_stamp or stamp) - stamp
        if age > self.PERCEPTION_MAX_AGE_S:
            self.get_logger().warn(
                '[state] %s perception pose is %.1f s old (limit %.1f s); '
                'treating it as not localized'
                % (entity, age, self.PERCEPTION_MAX_AGE_S))
            return None

        p, q = msg.pose.position, msg.pose.orientation
        z = float(p.z)
        grounded = self._grounded_z(entity)
        if grounded is not None:
            self.get_logger().info(
                '[state] %s from perception at (%.4f, %.4f), standing on the '
                'bench at z %.4f (camera said %.4f, %.1f mm out)'
                % (entity, p.x, p.y, grounded, z, (z - grounded) * 1e3))
            z = grounded
        else:
            self.get_logger().info(
                '[state] %s from perception at (%.4f, %.4f, %.4f), age %.2f s'
                % (entity, p.x, p.y, z, age))
        return [p.x, p.y, z, q.w, q.x, q.y, q.z]

    async def set_tamp_env_cb(self, request, response):
        """World State from perception plus the static table, never a simulator.

        The base blocks on the simulator's GetEntityState service before it does
        anything, which on hardware waits forever, and falls back to it for
        every untagged entity. Both are replaced here; what is NOT replaced is
        the rule that a tagged vessel the pipeline failed to localize fails the
        request, because falling back would report a perception-in-the-loop
        result that was not one.
        """
        state_source = os.environ.get('SDL_STATE_SOURCE', 'perception').strip()
        if state_source != 'perception':
            self.get_logger().error(
                "[state] SDL_STATE_SOURCE=%r: this server has no simulator to "
                'read ground truth from' % state_source)
            response.success = False
            return response

        poses = {}
        sources = {}
        for entity in request.entities:
            if entity in self.PERCEPTION_ENTITIES:
                # The base's recovery ladder, NOT _perception_pose directly.
                # It resolves self._perception_pose, so it polls this class's
                # cho_object_pose source, and its motion rungs go through the
                # plant hooks below -- which is why the ladder lives on the
                # base and both set_tamp_env_cb implementations call into it.
                pose = self._localize_with_recovery(entity)
                if pose is None:
                    self.get_logger().error(
                        '[state] no perception pose for %s; the trial is a '
                        'perception failure, not a planning one' % entity)
                    response.success = False
                    return response
                poses[entity] = pose
                sources[entity] = 'perception'
                continue

            if entity not in self._static_poses:
                self.get_logger().error(
                    '[state] %s carries no tag and has no entry in the static '
                    'pose file; set SDL_STATIC_POSES to a file that has one '
                    'rather than planning around an assumed pose' % entity)
                response.success = False
                return response
            poses[entity] = list(self._static_poses[entity])
            sources[entity] = 'static'

        self.get_logger().info(
            '[state] source=%s %s' % (state_source, ' '.join(
                '%s:%s' % (k, sources[k]) for k in sorted(sources))))

        self.tamp.update_env(
            name=request.env_name,
            poses=poses,
            movables=request.movables,
            statics=request.statics,
            ex_collision=request.ex_collision,
            rearrange_grid=request.rearrange_grid,
        )
        response.success = True
        return response

    # -- plant ------------------------------------------------------------

    def _arm_q(self):
        """Measured arm configuration, indexed by joint name.

        Not a slice: a build with the AG-95 publishes the finger joint in the
        same message, and joint_state_broadcaster gives no ordering guarantee.
        """
        try:
            return [self.arm.joint_positions()[name] for name in ARM_JOINTS]
        except ChoCommandFailed as error:
            self.get_logger().warn('[state] %s' % error)
            return []

    def _start_execution(self):
        if self._estop:
            self.get_logger().error('e-stop is latched; reset it before executing')
            return False
        try:
            if self.may_switch:
                self.arm.switch_to(TRAJECTORY_CONTROLLER)
            else:
                state = self.arm.controller_states().get(TRAJECTORY_CONTROLLER)
                if state != 'active':
                    self.get_logger().error(
                        "%s is '%s', not active. The fjt_handover task tree owns "
                        'the switch; start it, or set SDL_REAL_SWITCH=switch to '
                        'let this process do it.' % (TRAJECTORY_CONTROLLER, state))
                    return False
        except ChoCommandFailed as error:
            self.get_logger().error('cannot take the arm: %s' % error)
            return False
        return True

    def _finish_execution(self, success):
        if not success:
            # Whatever was in flight is no longer wanted. The tree's abort
            # branch parks the arm; this just stops the motion first.
            self.arm.cancel(quiet=True)

    def _check_estop(self):
        if self._estop:
            raise EStopRequested('plan stopped by tamp_estop')

    def _publish_arm_command(self, positions, joint_names=None):
        """Stream one configuration to the trajectory controller.

        The closed-loop primitive: publishing replaces the running trajectory,
        so the caller's loop rate and the horizon together set the commanded
        joint rate. Both pour laws and move_to_target reach the arm through
        here, unchanged from the simulated path.
        """
        self._check_estop()
        self.arm.stream_point(list(positions), horizon_sec=self.stream_horizon)

    def _execute_trajectory_step(self, plan_part):
        """Send one planned segment as a single trajectory goal.

        Not the base's publish-and-sleep replay: `time_from_start` is the
        timing on this path, so the whole segment goes as one goal and the
        controller interpolates it. The goal also reports WHY it stopped, which
        a stream cannot.
        """
        self._check_estop()
        plan_trajectory = plan_part['plan']
        dt = float(os.environ.get('SDL_EXEC_DT', '0.04'))
        positions = plan_trajectory.position.tolist()

        # The planner's own velocities are deliberately NOT forwarded. They
        # belong to cuRobo's interpolation step, and the waypoints are re-timed
        # here at SDL_EXEC_DT, so handing them over would describe a different
        # motion than the point times do. The controller derives them from the
        # path it is given instead.
        scale = required_time_scale(positions, dt)
        if scale > 1.0:
            self.get_logger().warn(
                'stretching this segment by %.2fx: at %.3f s per waypoint it '
                'implies joint speeds above the arm envelope. The shape is '
                'unchanged; the planner is asking for a faster arm than this '
                'one is held to.' % (scale, dt))

        traj = build_trajectory(
            positions,
            joint_names=list(plan_trajectory.joint_names),
            dt=dt,
            time_scale=scale,
        )
        self.arm.execute_trajectory(traj)

    def execute_gripper_action(self, plan_part):
        """Open or close the AG-95 through the cho gripper action.

        The simulated path also publishes the intended grasp target so the
        simulator welds the right object. A real gripper closes on whatever is
        between its fingers, so there is nothing to announce.
        """
        self._check_estop()
        closing = plan_part['action'] == 'close'
        # Same conservative rule as the base: a close counts from the moment it
        # is commanded, and only a command that returned clears it, so a
        # set_gripper that raises mid-open leaves the arm "holding". The
        # recovery ladder reads this before it moves anything, and it has to be
        # maintained on BOTH plants -- maintaining it only on the base would
        # mean the physical arm is the one that sweeps with a vessel in it.
        if closing:
            self._holding = True
        self.arm.set_gripper(grasp=closing)
        if not closing:
            self._holding = False

    # -- e-stop -----------------------------------------------------------

    def _estop_cb(self, request, response):
        """Stop the plan at the next command, and cancel what is in flight.

        The plan loop is a blocking Python loop, so this cannot interrupt it
        mid-step -- it latches, every hook checks before commanding, and the
        goal in flight is cancelled now. Call it again to clear.
        """
        if self._estop:
            self._estop = False
            response.success = True
            response.message = 'e-stop cleared'
        else:
            self._estop = True
            self.arm.cancel(quiet=True)
            response.success = True
            response.message = 'e-stop latched; motion cancelled'
        self.get_logger().warn(response.message)
        return response


if __name__ == '__main__':
    tamp_main(server_factory=RealTAMPServer)
