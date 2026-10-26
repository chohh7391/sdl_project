#!/usr/bin/env python3
"""Pour after pour from one start pose, for a pour's statistics: no pick, no carry.

    pour_series.py to-start PLAN          home -> the plan's pour start, jaws open
    pour_series.py series --material liquid|granular [--flow-index F]
                          [--targets 10,30,50] [--repeat 10] [--timeout S]
                          [--out DIR] [--bringup-log LOG]
    pour_series.py console [--material syrup|water|sugar] [--out DIR] [--bringup-log LOG]
                                          the same pours, one typed command at a time
    pour_series.py to-home PLAN           the pour start -> home, the same way back
    pour_series.py shift PLAN [--receiver X,Y] [--dry-run]
                                          the pour start moved with the receiver
    pour_series.py move-to Q1,..,Q6 [--from Q1,..,Q6]
                                          a checked joint move to another pour start
    pour_series.py move-to --path FILE    along a checked joint path, {"path": [[j1..j6], ..], "times": [s, ..]}
    pour_series.py unshift                back the way shift or move-to came

shift: the receiver stood at the plan's receiver_rim when the plan was made;
when it stands elsewhere now (tag 1, or --receiver), the jaws are carried
there in a straight line at the same height and attitude, so the vessel ends
where it was relative to the receiver in the plan, and every pour aligns the
few centimetres it always has instead of the whole distance. The path is
solved by IK every 2 mm on the URDF from /robot_description, refused if a
joint jumps or leaves its limits, and saved (~/real_cell_runs/pour_series/
shift.json) so unshift retraces it before to-home.

PLAN is an exported replay plan without its extension (<run>/deliver/
traj_realcell_seed<N>). The arm goes to the pour start along the plan's own
reach and carry -- a path the cell has run, which MoveIt's scene (no camera
mounts) would not guarantee -- on the trajectory controller, at the replay's
x4 clock, and back the same way.

A series is one material. It asks for each step on the terminal:
  Enter: close the jaws on the vessel the operator holds between them, record
         where it sits (held_beaker and the joints, as the replay's grasp
         does), tell the pose node the robot has it, switch to the pouring
         controller;
  Enter per pour (after refilling the vessel and emptying the receiver):
         close the jaws again -- they hold already; the gripper call pauses
         the servo stream, which drains the robot's motion queue (it grows
         ~0.13 commands/s, 8 ms of lag each) -- read the receiver from tag 1
         (/perception/object_pose/beaker2) and the empty pan, and pour;
  Enter: open the jaws to hand the vessel back.
Targets run in blocks: --repeat pours of the first, then of the next.

The console does the same, each step when the operator types it:
  close          close the jaws on the vessel held between them, record it, pouring controller on
  open           trajectory controller back on, open the jaws (hold the vessel)
  10, 30, 50 ..  pour that many grams of the current material (Ctrl-C cancels: the vessel is parked)
  steps [T S W]  T g (50) in steps to S, 2S, .. T g (10) in total, the pan read W s (3) after each
  syrup, water, sugar   change the material
  status, help, q
It keeps one summary.csv for every material (the label column tells them
apart); started again with the same --out it carries on the numbering.

Recorded under --out (default ~/real_cell_runs/pour_series/<stamp>_<material>):
summary.csv, one row a pour (what the controller reported, and the pan 10 s
after it finished, against the empty pan read before it), and per pour the
scale and the controller's feedback, scale_NN.csv and feedback_NN.csv. No bag.
"""
import argparse
import csv
import json
import math
import os
import sys
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions

from builtin_interfaces.msg import Duration as DurationMsg
from cho_interfaces.action import Gripper
from cho_interfaces.msg import ObjectVisibilityArray
from sdl_cell_interfaces.action import Pour
from sdl_cell_interfaces.msg import ScaleReading
from controller_manager_msgs.srv import SwitchController
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint
import tf2_ros

JOINTS = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
JTC = 'joint_trajectory_controller'
POUR = 'pouring_controller'
HOLD = 'joint_space_position_controller'
CLOCK = 4.0                      # the replay's time scale
HELD = '/perception/object_pose/held_beaker'
RECEIVER = '/perception/object_pose/beaker2'
BEAKER_H = 0.070                 # the receiver's rim above its base; the controller fixes z anyway
SETTLE_AFTER = 10.0              # pan read this long after a pour ends [s]
JAWS = 'gripper_finger_link'     # the vessel held in the jaws stands ~6 cm from it
JAWS_REACH = 0.15                # a tag 0 pose farther than this from the jaws is not the held vessel [m]
PRESETS = {                      # console material -> (material, flow_index, pour timeout [s], 0: the controller's)
    'syrup': ('liquid', 1.0, 300.0),
    'water': ('liquid', 0.0, 0.0),
    'sugar': ('granular', 0.0, 300.0),
}
SUMMARY = ['trial', 'label', 'material', 'flow_index', 'target', 'rep', 'tolerance', 'empty_pan', 'controller_grams',
           'pan_after_10s', 'error_pan', 'elapsed', 'peak_tilt_deg', 'trims', 'afterflow',
           'completed', 'receiver_x', 'receiver_y', 'started', 'message',
           'settle_s', 'run', 'step', 'cum_target', 'cum_pan']


def read_plan(plan):
    import csv as _csv
    rows = list(_csv.DictReader(open(plan + '.csv')))
    carry = [i for i, r in enumerate(rows) if (r.get('operation') or '') == 'Move_to_Surface']
    if not carry:
        sys.exit('%s.csv has no Move_to_Surface rows' % plan)
    rows = rows[:carry[-1] + 1]
    return [float(r['t_s']) for r in rows], [[float(r[j]) for j in JOINTS] for r in rows]


class Series(Node):
    def __init__(self):
        super().__init__('pour_series')
        self.joints = None
        self.scale = []                  # (stamp, grams, stable)
        self.held = None                 # (recv time, PoseStamped)
        self.receiver = None
        self.in_hand = None              # the vessel's pose while the robot holds it
        self.seen = {}                   # object -> (recv time, cameras that see it now, status)
        self.feedback = []
        self.lock = threading.Lock()
        rel = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(JointState, '/joint_states', self._on_joints, qos_profile_sensor_data)
        self.create_subscription(ScaleReading, '/scale/reading', self._on_scale, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, HELD, lambda m: self._on_pose('held', m), rel)
        self.create_subscription(PoseStamped, RECEIVER, lambda m: self._on_pose('receiver', m), rel)
        self.create_subscription(PoseStamped, HELD + '/in_hand', lambda m: self._on_pose('in_hand', m), rel)
        self.create_subscription(ObjectVisibilityArray, '/perception/object_visibility', self._on_vis, rel)
        self.switch = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.jtc = ActionClient(self, FollowJointTrajectory, '/%s/follow_joint_trajectory' % JTC)
        # The core's action contract (cho_robot_project v0.1.0): /<controller>/gripper.
        self.gripper = ActionClient(self, Gripper, '/gripper_controller/gripper')
        self.pour = ActionClient(self, Pour, '/controller_action_server/pouring_controller')
        self.tf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf, self)

    # ------------------------------------------------------------- topics
    def _on_joints(self, m):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in JOINTS):
            self.joints = [d[j] for j in JOINTS]

    def _on_scale(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 or time.time()
        with self.lock:
            self.scale.append((t, m.grams, m.stable))
            self.scale[:] = self.scale[-20000:]

    def _on_pose(self, which, m):
        setattr(self, which, (time.time(), m))

    def _on_vis(self, m):
        now = time.time()
        for o in m.objects:
            self.seen[o.name] = (now, [c.camera for c in o.cameras if c.state == 1], o.status)

    # ------------------------------------------------------------- calls
    def wait(self, future, timeout):
        end = time.time() + timeout
        while not future.done():
            if time.time() > end:
                return None
            time.sleep(0.02)
        return future.result()

    def switch_to(self, activate, deactivate):
        if not self.switch.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('no /controller_manager/switch_controller')
        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.activate_asap = True
        req.timeout = DurationMsg(sec=5)
        res = self.wait(self.switch.call_async(req), 15.0)
        if res is None or not res.ok:
            raise RuntimeError('switch to %s failed' % activate)

    def grip(self, close):
        if not self.gripper.wait_for_server(timeout_sec=5.0):
            raise RuntimeError('no gripper action server')
        g = Gripper.Goal()
        g.grasp = bool(close)
        h = self.wait(self.gripper.send_goal_async(g), 10.0)
        if h is None or not h.accepted:
            raise RuntimeError('gripper goal refused')
        self.wait(h.get_result_async(), 20.0)

    def trigger(self, service, quiet=False):
        c = self.create_client(Trigger, service)
        if not c.wait_for_service(timeout_sec=3.0):
            print('  (no %s; skipped)' % service)
            return
        r = self.wait(c.call_async(Trigger.Request()), 5.0)
        if not quiet or r is None:
            print('  %s: %s' % (service, r.message if r else 'no answer'))

    def follow(self, times, points):
        if not self.jtc.wait_for_server(timeout_sec=5.0):
            raise RuntimeError('no %s action server' % JTC)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        t0 = times[0]
        for t, q in zip(times, points):
            p = JointTrajectoryPoint()
            p.positions = list(q)
            p.time_from_start = Duration(seconds=(t - t0) * CLOCK + 0.5).to_msg()
            goal.trajectory.points.append(p)
        h = self.wait(self.jtc.send_goal_async(goal), 10.0)
        if h is None or not h.accepted:
            raise RuntimeError('trajectory refused')
        res = self.wait(h.get_result_async(), (times[-1] - t0) * CLOCK + 30.0)
        if res is None or res.result.error_code != 0:
            raise RuntimeError('trajectory did not finish: %s' % (res.result.error_string if res else 'timeout'))

    def fresh(self, which, age=1.5):
        """The pose, only while a camera actually sees the tag now.

        The pose node republishes a pose it is HOLDING with fresh stamps when no
        camera sees the object, so a recent message alone proves nothing: on
        2026-10-03 it held beaker2 where the operator had carried it, 0.32 m
        from the scale, while the jaws hid its tag from side_1.
        """
        v = getattr(self, which)
        name = 'beaker1' if which == 'held' else 'beaker2'
        s = self.seen.get(name)
        if v is None or time.time() - v[0] > age or s is None or time.time() - s[0] > age or not s[1]:
            return None
        return v[1]

    def jaws(self):
        """Where the jaws are now in base_link, or None."""
        try:
            t = self.tf.lookup_transform('base_link', JAWS, rclpy.time.Time()).transform.translation
        except tf2_ros.TransformException:
            return None
        return [t.x, t.y, t.z]

    def params(self, node, names, timeout=5.0):
        """Doubles from another node's parameters, by name; None for one it does not have."""
        c = self.create_client(GetParameters, '/%s/get_parameters' % node)
        if not c.wait_for_service(timeout_sec=timeout):
            raise RuntimeError('no /%s/get_parameters' % node)
        r = self.wait(c.call_async(GetParameters.Request(names=list(names))), timeout)
        if r is None:
            raise RuntimeError('/%s/get_parameters did not answer' % node)
        return {n: (v.double_value if v.type == 3 else None) for n, v in zip(names, r.values)}

    def why_not(self, which):
        s = self.seen.get('beaker1' if which == 'held' else 'beaker2')
        return s[2][:140] if s else 'no /perception/object_visibility'

    def scale_now(self, stable_for=1.0, timeout=8.0):
        """The pan once it has read stable for stable_for seconds, else the last reading."""
        end = time.time() + timeout
        while time.time() < end:
            with self.lock:
                recent = [s for s in self.scale if s[0] > self.scale[-1][0] - stable_for] if self.scale else []
            if recent and all(s[2] for s in recent) and recent[-1][0] - recent[0][0] >= stable_for * 0.8:
                return recent[-1][1]
            time.sleep(0.1)
        with self.lock:
            return self.scale[-1][1] if self.scale else float('nan')


SHIFT_FILE = os.path.expanduser('~/real_cell_runs/pour_series/shift.json')
CAMERAS = {'side_1': (0.621, 0.452, 0.398), 'side_2': (0.534, -0.470, 1.132)}   # optical centres, 10-02


def robot_model(node, timeout=5.0):
    """(model, data, wrist3 frame id) from /robot_description."""
    import tempfile
    import pinocchio as pin
    from rclpy.qos import DurabilityPolicy
    from std_msgs.msg import String
    got = {}
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String, '/robot_description', lambda m: got.setdefault('x', m.data), qos)
    end = time.time() + timeout
    while 'x' not in got and time.time() < end:
        time.sleep(0.05)
    if 'x' not in got:
        sys.exit('no /robot_description')
    with tempfile.NamedTemporaryFile('w', suffix='.urdf', delete=False) as f:
        f.write(got['x'])
    model = pin.buildModelFromUrdf(f.name)
    os.unlink(f.name)
    return model, model.createData(), model.getFrameId('wrist3_link')


def straight_line(model, data, fid, q0, delta, step=0.002, min_j5=0.1):
    """Joint path carrying wrist3 from its pose at q0 by delta (base frame), attitude held.

    Refused where |j5| falls under min_j5: the wrist singularity, where j4 and
    j6 swing through ~3 rad for millimetres of travel. Under the side grasp the
    receiver nearer y = 0 drives j5 to it -- 0 at ~22 cm from the plan's pour
    start on 2026-10-03 -- and the pour itself tilts the wrist further.
    """
    import numpy as np
    import pinocchio as pin
    idx = [model.joints[model.getJointId(j)].idx_q for j in JOINTS]
    vdx = [model.joints[model.getJointId(j)].idx_v for j in JOINTS]

    def fk(qa):
        q = pin.neutral(model)
        q[idx] = qa
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return q, pin.SE3(data.oMf[fid])

    _, T0 = fk(np.array(q0))
    n = max(1, int(math.ceil(np.linalg.norm(delta) / step)))
    path, qa = [list(q0)], np.array(q0, float)
    for k in range(1, n + 1):
        target = pin.SE3(T0.rotation, T0.translation + np.asarray(delta) * k / n)
        for _ in range(200):
            q, T = fk(qa)
            err = pin.log6(T.actInv(target)).vector
            if np.linalg.norm(err) < 1e-6:
                break
            J = pin.computeFrameJacobian(model, data, q, fid, pin.LOCAL)[:, vdx]
            qa = qa + J.T @ np.linalg.solve(J @ J.T + 1e-8 * np.eye(6), err)
        else:
            raise RuntimeError('IK did not converge %d mm along the line' % round(1000 * np.linalg.norm(delta) * k / n))
        lo, hi = model.lowerPositionLimit[idx], model.upperPositionLimit[idx]
        if (qa < lo).any() or (qa > hi).any():
            raise RuntimeError('a joint leaves its limits %d mm along the line' % round(1000 * np.linalg.norm(delta) * k / n))
        if abs(qa[4]) < min_j5:
            raise RuntimeError('j5 reaches %.3f rad %d mm along the line: the wrist singularity'
                               % (qa[4], round(1000 * np.linalg.norm(delta) * k / n)))
        if np.max(np.abs(qa - np.array(path[-1]))) > 0.03:
            raise RuntimeError('a joint jumps %.3f rad in one 2 mm step' % np.max(np.abs(qa - np.array(path[-1]))))
        path.append(list(qa))
    _, T1 = fk(qa)
    return path, T0.translation, T1.translation


def near(a, b, tol):
    return a is not None and max(abs(x - y) for x, y in zip(a, b)) <= tol


def ask(prompt):
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return 'q'


class Log:
    """summary.csv and the per-pour files under out; reopened, it carries on the numbering."""

    def __init__(self, out):
        os.makedirs(out, exist_ok=True)
        self.out = out
        path = os.path.join(out, 'summary.csv')
        rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
        if rows and list(rows[0].keys()) != SUMMARY:
            # Written before a column was added: rewrite it under the current
            # header, the new columns empty in the old rows.
            with open(path, 'w', newline='') as f:
                cw = csv.DictWriter(f, SUMMARY, extrasaction='ignore')
                cw.writeheader()
                cw.writerows(rows)
        self.n = max([int(r['trial']) for r in rows] or [0])
        self.reps = {}
        for r in rows:
            key = (r['label'], float(r['target']), str(r.get('tolerance') or ''))
            self.reps[key] = self.reps.get(key, 0) + 1
        self.f = open(path, 'a', newline='')
        self.w = csv.writer(self.f)
        if self.f.tell() == 0:
            self.w.writerow(SUMMARY)
            self.f.flush()

    def close(self):
        self.f.close()


def close_jaws(node):
    """Close the jaws on the vessel the operator holds between them, and record where it sits.

    Returns (grasp_joints, grasp_marker or None) as the replay's grasp records
    them, with the pouring controller on.
    """
    node.grip(True)
    time.sleep(2.0)
    # Released first: the pose node may still be holding the vessel where it
    # last stood (the riser), over what the cameras see in the jaws now.
    node.trigger(HELD + '/release')
    # Only a pose by the jaws is the vessel in them: on 2026-10-03 side_1 also
    # saw a tag 0 lying on the table 0.40 m away, and the first close took it.
    # Closed again on a vessel the robot already holds, the pose node publishes
    # no held_beaker -- only held_beaker/in_hand, whose position is the same
    # bottom centre -- so that counts too.
    held, far = None, None
    t_end = time.time() + 8.0
    while held is None and time.time() < t_end:
        v = node.fresh('held', 1.0)
        if v is None and node.in_hand is not None and time.time() - node.in_hand[0] < 1.0:
            v = node.in_hand[1]
        jaws = node.jaws()
        if v is not None and v.header.frame_id in ('base_link', '') and jaws is not None:
            p = v.pose.position
            d = math.dist([p.x, p.y, p.z], jaws)
            if d <= JAWS_REACH:
                held = v
            else:
                far = (p, d)
        time.sleep(0.1)
    grasp_joints = list(node.joints)
    grasp_marker = None
    if held is not None:
        p = held.pose.position
        grasp_marker = [p.x, p.y, p.z]
        print('  grasp: vessel at (%.4f, %.4f, %.4f), %.0f mm from the jaws, with the arm at %s'
              % (p.x, p.y, p.z, 1000 * math.dist([p.x, p.y, p.z], node.jaws() or [p.x, p.y, p.z]),
                 ['%.3f' % q for q in grasp_joints]))
    elif far is not None:
        print('  grasp: the only tag 0 seen is at (%.4f, %.4f, %.4f), %.0f mm from the jaws -- not the vessel '
              'in them; not used. The controller measures the vessel at each pour. (close again to retry)'
              % (far[0].x, far[0].y, far[0].z, 1000 * far[1]))
    else:
        print('  grasp: no camera sees tag 0 by the jaws (%s); the controller measures the vessel at each pour'
              % node.why_not('held'))
    node.switch_to([POUR], [JTC, HOLD])
    return grasp_joints, grasp_marker


def open_jaws(node):
    node.switch_to([JTC], [POUR, HOLD])
    node.grip(False)
    node.trigger(HELD + '/resume')


def pour_once(node, log, label, material, flow_index, timeout, tg, grasp, start_pose, bringup_log='',
              tolerance=0.0, effective_tolerance='', settle=SETTLE_AFTER, extra=None):
    """One pour of tg grams into the receiver tag 1 shows now, logged; the summary row, or None.

    tolerance is the goal's (0: the controller's weight_tolerance); the
    controller floors it by the material's dose quantum, which
    effective_tolerance, when known, records. The pan is read settle seconds
    after the pour ends. extra, for a step of a stepped pour, is
    {run, step, cum_target, base}: base is the pan before its first step.
    """
    if bringup_log and os.path.exists(bringup_log):
        lines = [ln for ln in open(bringup_log, errors='ignore') if 'robot queue' in ln]
        if lines:
            print('  ' + lines[-1].split('robot queue', 1)[1][:40].join(['robot queue', '']))
    # Measured now, never held: on 2026-10-03 the pose node kept a wrist view of
    # the receiver from 1.8 h before over the standing cameras' (it outranks
    # them), 22 mm from where they saw it, and two pours went there. Release
    # and resume drop whatever it holds; then only a pose published after that,
    # under a 'publishing' status, counts.
    node.trigger(RECEIVER + '/release', quiet=True)
    node.trigger(RECEIVER + '/resume', quiet=True)
    t_mark = time.time()
    rec = None
    t_end = t_mark + 5.0
    while rec is None and time.time() < t_end:
        v, s = node.receiver, node.seen.get('beaker2')
        if (v is not None and v[0] > t_mark + 0.3 and s is not None and s[1]
                and s[2].startswith('publishing')):
            rec = v[1]
        time.sleep(0.1)
    if rec is None:
        print('  no fresh measurement of the receiver (tag 1) in 5 s (%s); not pouring' % node.why_not('receiver'))
        return None
    if not 0.05 <= rec.pose.position.z <= 0.12:
        print('  the receiver measures at z %.3f m -- not standing on the scale (0.05-0.12); not pouring'
              % rec.pose.position.z)
        return None
    print('  receiver seen by %s' % ', '.join(node.seen['beaker2'][1]))
    # The controller lays its lip path out from the grasp measured at close;
    # without one it waits for the held vessel's pose and gives up after 5 s.
    # The pose node does not publish that pose while the robot holds the vessel
    # (it publishes held_beaker/in_hand, which levels it), so a pour without a
    # measured grasp is checked for it here rather than refused by the
    # controller.
    if grasp is None or grasp[1] is None:
        t_end = time.time() + 5.0
        while (node.held is None or time.time() - node.held[0] > 1.0) and time.time() < t_end:
            time.sleep(0.1)
        if node.held is None or time.time() - node.held[0] > 1.0:
            print('  no grasp measured at close and no pose of the vessel (tag 0) now (%s); '
                  'not pouring -- open and close again where a camera sees tag 0' % node.why_not('held'))
            return None
    if not near(node.joints, start_pose, 0.15):
        print('  the arm is not at the pour start (%s); not pouring' % ['%.3f' % q for q in node.joints])
        return None
    empty = node.scale_now()
    r = rec.pose.position
    goal = Pour.Goal()
    goal.target_grams = float(tg)
    goal.container_grams = empty
    goal.material = Pour.Goal.MATERIAL_GRANULAR if material == 'granular' else Pour.Goal.MATERIAL_LIQUID
    goal.flow_index = float(flow_index)
    goal.timeout = float(timeout)
    goal.tolerance = float(tolerance)
    goal.receiver_rim_center = [r.x, r.y, r.z + BEAKER_H]
    if grasp is not None and grasp[1] is not None:
        goal.grasp_joints = grasp[0]
        goal.grasp_marker.x, goal.grasp_marker.y, goal.grasp_marker.z = grasp[1]
    fb = []

    def on_fb(msg):
        f = msg.feedback
        fb.append((time.time(), f.phase, f.current_grams, math.degrees(f.tilt), f.flow_rate))

    started = time.time()
    print('  pouring %g g of %s%s: receiver (%.4f, %.4f), empty pan %.2f g  (Ctrl-C cancels)'
          % (tg, label, ' +/- %g g' % effective_tolerance if effective_tolerance != '' else '', r.x, r.y, empty))
    h = node.wait(node.pour.send_goal_async(goal, feedback_callback=on_fb), 10.0)
    if h is None or not h.accepted:
        print('  the pour goal was refused')
        return None
    try:
        res = node.wait(h.get_result_async(), (timeout or 300.0) + 60.0)
    except KeyboardInterrupt:
        print('\n  Ctrl-C: cancelling the pour; the controller parks the vessel')
        node.wait(h.cancel_goal_async(), 5.0)
        res = node.wait(h.get_result_async(), 60.0)
    done = time.time()
    print('  pour ended after %.1f s; reading the pan in %.0f s' % (done - started, settle))
    try:
        time.sleep(settle)
    except KeyboardInterrupt:
        print('  Ctrl-C: reading the pan now')
    after = node.scale_now(stable_for=1.0, timeout=5.0)
    result = res.result if res is not None else None
    if result is not None and not result.is_completed and result.peak_tilt == 0.0 and abs(after - empty) < 0.3:
        print('  not poured, not logged: %s' % result.message[:160])
        return None
    with node.lock:
        samples = [s for s in node.scale if started - 2.0 <= s[0] <= time.time()]
    log.n += 1
    n = log.n
    key = (label, float(tg), str(effective_tolerance))   # a tolerance is its own series
    log.reps[key] = rep = log.reps.get(key, 0) + 1
    with open(os.path.join(log.out, 'scale_%03d.csv' % n), 'w', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['t', 'grams', 'stable'])
        cw.writerows([(round(t - started, 3), g, int(st)) for t, g, st in samples])
    with open(os.path.join(log.out, 'feedback_%03d.csv' % n), 'w', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['t', 'phase', 'grams', 'tilt_deg', 'flow'])
        cw.writerows([(round(t - started, 3), p, g, round(tl, 2), fl) for t, p, g, tl, fl in fb])
    row = [n, label, material, flow_index, tg, rep, effective_tolerance, round(empty, 2),
           round(result.final_grams, 2) if result else '', round(after - empty, 2),
           round(after - empty - tg, 2), round(done - started, 1),
           round(math.degrees(result.peak_tilt), 1) if result else '',
           result.trim_pulses if result else '', round(result.measured_afterflow, 2) if result else '',
           int(result.is_completed) if result else 0, round(r.x, 4), round(r.y, 4),
           time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started)),
           (result.message if result else 'no result')[:200],
           settle]
    if extra:
        row += [extra['run'], extra['step'], extra['cum_target'], round(after - extra['base'], 2)]
    else:
        row += ['', '', '', '']
    log.w.writerow(row)
    log.f.flush()
    print('  -> #%d %s %g g +/- %s g (#%d): controller %s g, pan %.2f g (%+.2f) in %.1f s: %s'
          % (n, label, tg, effective_tolerance, rep, row[8], after - empty, after - empty - tg, done - started,
             row[SUMMARY.index('message')][:90]))
    return row


def pour_steps(node, log, name, grasp, start_pose, bringup_log, tol, effective, refresh,
               total=50.0, step=10.0, wait=3.0, base=None):
    """Pour total grams in steps: to step, then 2 x step, ... -- each step to the
    running total against the pan before the first, so a step's error is not
    carried into the next -- reading the pan wait seconds after each.

    base, given, is that first empty pan: a stepped pour that stopped part way
    carries on under the log's last run, its done steps skipped."""
    material, flow_index, timeout = PRESETS[name]
    run = time.strftime('%Y%m%d-%H%M%S')
    if base is None:
        base = node.scale_now()
    else:
        path = os.path.join(log.out, 'summary.csv')
        runs = [r['run'] for r in csv.DictReader(open(path)) if r.get('run')] if os.path.exists(path) else []
        if runs:
            run = runs[-1]
            print('  carrying on stepped pour %s from an empty pan of %.2f g' % (run, base))
    n_steps = int(round(total / step))
    print('  stepped pour %s: %g g in %d steps of %g g, the pan read %g s after each; empty pan %.2f g'
          % (run, total, n_steps, step, wait, base))
    for k in range(1, n_steps + 1):
        cum = step * k
        inc = round(cum - (node.scale_now() - base), 2)
        if inc < 0.5:
            print('  step %d/%d: the pan is already within 0.5 g of %g g; skipped' % (k, n_steps, cum))
            continue
        print('\n  step %d/%d: to %g g, %.2f g more' % (k, n_steps, cum, inc))
        refresh()
        row = pour_once(node, log, name, material, flow_index, timeout, inc, grasp, start_pose, bringup_log,
                        tol, effective(name), settle=wait,
                        extra={'run': run, 'step': k, 'cum_target': cum, 'base': base})
        if row is None:
            print('  step %d did not pour; the stepped pour stops here' % k)
            break
        if 'cancel' in str(row[SUMMARY.index('message')]).lower():
            print('  cancelled; the stepped pour stops here')
            break
    final = node.scale_now() - base
    print('\n  stepped pour %s: %.2f g on the pan for %g g (%+.2f)' % (run, final, total, final - total))


CONSOLE_HELP = '''  close            close the jaws on the vessel held between them (pouring controller on)
  open             open the jaws -- hold the vessel first (trajectory controller back on)
  10, 30, 50, ..   pour that many grams of the current material; Ctrl-C cancels the pour
  steps [T S W]    pour T g (50) in steps of S g (10) -- to S, 2S, .. T in total -- W s (3) apart
  steps T S W B    carry on a stepped pour that stopped, B being its first empty pan (grams)
  syrup, water, sugar   change the material
  tol 0.3          what counts as done, +/- grams (tol 0: the controller's default); floored by the material
  status           material, tolerance, jaws, receiver, pan, pours so far
  q                quit (the jaws and controllers stay as they are)'''


def console(node, a):
    log = Log(os.path.expanduser(a.out or '~/real_cell_runs/pour_series/%s_console' % time.strftime('%Y%m%d-%H%M%S')))
    start_pose = list(node.joints)
    name = a.material
    grasp = None                         # set by close in this console
    try:
        import readline                  # noqa: F401  (arrow keys and history at the prompt)
    except ImportError:
        pass
    names = ['weight_tolerance'] + ['%s.%s.dose_quantum' % (m, e) for m in ('liquid', 'granular')
                                    for e in ('free', 'resistant')]
    cp = {}

    def refresh():
        """The controller's tolerance parameters now: they can be changed under a running console."""
        try:
            cp.update(node.params(POUR, names))
        except RuntimeError as e:
            print('  (%s: the tolerance is logged as last read)' % e)

    refresh()
    tol = 0.0                            # the goal's; 0 is the controller's weight_tolerance

    def effective(name):
        """The tolerance the controller will use: the goal's, floored by the material's dose quantum."""
        m, fi, _ = PRESETS[name]
        free, res = cp.get('%s.free.dose_quantum' % m), cp.get('%s.resistant.dose_quantum' % m)
        base = tol or cp.get('weight_tolerance')
        if None in (free, res, base):
            return ''
        f = min(max(fi, 0.0), 1.0)
        return round(max(base, free + (res - free) * f), 3)

    print('pour console: %s, logging to %s (%d pours there so far)' % (name, log.out, log.n))
    print('pour start %s' % ['%.3f' % q for q in start_pose])
    print(CONSOLE_HELP)
    while True:
        try:
            cmd = input('\n[%s +/-%s g | jaws %s] > ' % (name, effective(name), 'closed' if grasp else 'open/unknown')
                        ).strip().lower()
        except EOFError:
            cmd = 'q'
        except KeyboardInterrupt:
            print()
            continue
        if not cmd:
            continue
        try:
            if cmd in ('q', 'quit', 'exit'):
                break
            elif cmd in ('h', 'help', '?'):
                print(CONSOLE_HELP)
            elif cmd in PRESETS:
                name = cmd
                m, fi, to = PRESETS[name]
                print('  material %s: %s, flow_index %g, timeout %s' % (name, m, fi, '%g s' % to if to else 'the controller\'s'))
            elif cmd == 'tol' or cmd.startswith('tol '):
                refresh()
                if cmd != 'tol':
                    v = float(cmd.split()[1])
                    if not 0.0 <= v <= 5.0:
                        print('  %g g is out of range' % v)
                        continue
                    tol = v
                for nm in PRESETS:
                    e = effective(nm)
                    print('  %s: +/- %s g%s' % (nm, e, '' if e == '' or tol == 0 or e <= tol
                                                else '  (asked %g; floored by its dose quantum)' % tol))
            elif cmd == 'steps' or cmd.startswith('steps '):
                if grasp is None:
                    print('  close the jaws first (close): the pouring controller is not on')
                    continue
                v = [float(x) for x in cmd.split()[1:5]]
                total, step, wait = (v + [50.0, 10.0, 3.0][len(v):])[:3]
                base = v[3] if len(v) > 3 else None
                if not (0.0 < step <= total <= 200.0 and 0.0 <= wait <= 60.0):
                    print('  steps wants 0 < step <= total <= 200 g and a wait of 0-60 s')
                    continue
                pour_steps(node, log, name, grasp, start_pose, a.bringup_log, tol, effective, refresh,
                           total, step, wait, base)
            elif cmd in ('close', 'c'):
                grasp = close_jaws(node)
                print('  jaws closed; pouring controller on')
            elif cmd in ('open', 'o'):
                open_jaws(node)
                grasp = None
                print('  jaws open; trajectory controller on')
            elif cmd in ('status', 's'):
                refresh()
                rec = node.fresh('receiver', 1.0)
                print('  material %s +/- %s g, jaws %s, arm %s the pour start' % (
                    name, effective(name), 'closed' if grasp else 'open/unknown', 'at' if near(node.joints, start_pose, 0.15) else 'NOT at'))
                print('  receiver: %s' % ('(%.4f, %.4f) seen by %s' % (
                    rec.pose.position.x, rec.pose.position.y, ', '.join(node.seen['beaker2'][1])) if rec
                    else 'not seen (%s)' % node.why_not('receiver')))
                with node.lock:
                    last = node.scale[-1] if node.scale else None
                hand = node.held
                print('  vessel in the jaws (tag 0): %s' % (
                    'pose %.1f s old' % (time.time() - hand[0]) if hand else 'no pose yet'))
                print('  pan: %s' % ('%.2f g%s' % (last[1], '' if last[2] else ' (unstable)') if last else 'no reading'))
                for (lb, tg, tl), k in sorted(log.reps.items()):
                    print('  %s %g g +/- %s g: %d pours' % (lb, tg, tl or '?', k))
            else:
                try:
                    tg = float(cmd)
                except ValueError:
                    print('  ? %s  (help lists the commands)' % cmd)
                    continue
                if not 0.0 < tg <= 200.0:
                    print('  %g g is out of range' % tg)
                    continue
                if grasp is None:
                    print('  close the jaws first (close): the pouring controller is not on')
                    continue
                m, fi, to = PRESETS[name]
                refresh()
                pour_once(node, log, name, m, fi, to, tg, grasp, start_pose, a.bringup_log, tol, effective(name))
        except RuntimeError as e:
            print('  failed: %s' % e)
        except KeyboardInterrupt:
            print('\n  interrupted')
    log.close()
    print('console done: %s' % log.out)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('to-start', 'to-home'):
        p = sub.add_parser(name)
        p.add_argument('plan')
    sh = sub.add_parser('shift')
    sh.add_argument('plan')
    sh.add_argument('--receiver', default='', help='x,y of the receiver now (default: tag 1, seen now)')
    sh.add_argument('--speed', type=float, default=0.02, help='[m/s]')
    sh.add_argument('--dry-run', action='store_true')
    sub.add_parser('unshift')
    mv = sub.add_parser('move-to')
    mv.add_argument('target', nargs='?', default='', help='j1..j6 [rad], comma separated')
    mv.add_argument('--path', default='', help='a JSON joint path with times [s], instead of a target')
    mv.add_argument('--from', dest='source', default='', help='where the arm has to be now (j1..j6)')
    mv.add_argument('--rate', type=float, default=0.05, help='largest joint speed [rad/s]')
    s = sub.add_parser('series')
    s.add_argument('--material', required=True, choices=['liquid', 'granular'])
    s.add_argument('--flow-index', type=float, default=0.0)
    s.add_argument('--targets', default='10,30,50')
    s.add_argument('--repeat', type=int, default=10)
    s.add_argument('--timeout', type=float, default=0.0, help='pour goal timeout [s] (0: the controller\'s)')
    s.add_argument('--label', default='', help='name in the output folder, e.g. syrup')
    s.add_argument('--out', default='')
    s.add_argument('--bringup-log', default='', help='print the robot queue from its servo stats')
    c = sub.add_parser('console')
    c.add_argument('--material', default='syrup', choices=sorted(PRESETS))
    c.add_argument('--out', default='', help='summary.csv here; an existing one is carried on')
    c.add_argument('--bringup-log', default='', help='print the robot queue from its servo stats')
    a = ap.parse_args()

    # Ctrl-C as a plain KeyboardInterrupt: rclpy's own handler shuts the
    # context down, and a pour could then no longer be cancelled.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Series()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    t0 = time.time()
    while node.joints is None and time.time() - t0 < 5.0:
        time.sleep(0.05)
    if node.joints is None:
        sys.exit('no /joint_states: is the bringup up?')

    try:
        if a.cmd == 'shift':
            import yaml
            plan = os.path.expanduser(a.plan)
            old = yaml.safe_load(open(plan + '_cell.yaml'))['pour']['receiver_rim'][:2]
            _, points = read_plan(plan)
            if not near(node.joints, points[-1], 0.05):
                sys.exit('the arm is not at this plan\'s pour start; shift moves from there')
            if a.receiver:
                new = [float(v) for v in a.receiver.split(',')]
            else:
                rec = None
                t_end = time.time() + 4.0
                while rec is None and time.time() < t_end:
                    rec = node.fresh('receiver', 1.0)
                    time.sleep(0.1)
                if rec is None:
                    sys.exit('no camera sees tag 1 now (%s)' % node.why_not('receiver'))
                new = [rec.pose.position.x, rec.pose.position.y]
            delta = [new[0] - old[0], new[1] - old[1], 0.0]
            model, data, fid = robot_model(node)
            path, p0, p1 = straight_line(model, data, fid, node.joints, delta)
            dist = math.hypot(delta[0], delta[1])
            print('receiver (%.4f, %.4f) in the plan, (%.4f, %.4f) now: shift (%+.1f, %+.1f) mm, %d steps'
                  % (old[0], old[1], new[0], new[1], 1000 * delta[0], 1000 * delta[1], len(path) - 1))
            moved = [max(abs(x - y) for x, y in zip(path[-1], path[0]))]
            print('wrist3 (%.3f, %.3f, %.3f) -> (%.3f, %.3f, %.3f); largest joint change %.3f rad; joints at the end %s'
                  % (*p0, *p1, moved[0], ['%.3f' % q for q in path[-1]]))
            for cam, c in CAMERAS.items():
                print('  wrist3 to %s optical centre: %.0f -> %.0f mm' % (
                    cam, 1000 * math.dist(p0, c), 1000 * math.dist(p1, c)))
            if a.dry_run:
                return 0
            os.makedirs(os.path.dirname(SHIFT_FILE), exist_ok=True)
            json.dump({'plan': plan, 'receiver_plan': old, 'receiver_now': new, 'path': path},
                      open(SHIFT_FILE, 'w'))
            node.switch_to([JTC], [POUR, HOLD])
            dur = max(3.0, dist / a.speed)
            node.follow([dur * k / (len(path) - 1) / CLOCK for k in range(len(path))], path)
            print('shift: arrived; saved %s' % SHIFT_FILE)
            return 0
        if a.cmd == 'move-to' and a.path:
            new = json.load(open(os.path.expanduser(a.path)))
            path, times = new['path'], new['times']
            if not near(node.joints, path[0], 0.03):
                sys.exit('the arm is not where the path starts (%s); not moving' % ['%.3f' % q for q in node.joints])
            # Chained onto the move that brought the arm here, so unshift goes
            # back through both to the plan's pour start.
            old = json.load(open(SHIFT_FILE)) if os.path.exists(SHIFT_FILE) else None
            saved = {'kind': 'move-to', 'path': path, 'times': times}
            if old is not None and near(old['path'][-1], path[0], 0.03):
                ot = old.get('times') or [old['duration'] * k / (len(old['path']) - 1) for k in range(len(old['path']))]
                saved = {'kind': 'move-to', 'path': old['path'] + path[1:],
                         'times': ot + [ot[-1] + t - times[0] for t in times[1:]]}
            node.switch_to([JTC], [POUR, HOLD])
            print('move-to: %d points, %.1f s' % (len(path), times[-1] - times[0]))
            node.follow([t / CLOCK for t in times], path)
            json.dump(saved, open(SHIFT_FILE, 'w'))
            print('move-to: arrived; saved %s (%d points; unshift retraces them)' % (SHIFT_FILE, len(saved['path'])))
            return 0
        if a.cmd == 'move-to':
            target = [float(v) for v in a.target.split(',')]
            source = [float(v) for v in a.source.split(',')] if a.source else node.joints
            if len(target) != 6 or not near(node.joints, source, 0.03):
                sys.exit('the arm is not at --from (%s); not moving' % ['%.3f' % q for q in node.joints])
            n = 60
            path = [[s0 + (t - s0) * k / n for s0, t in zip(node.joints, target)] for k in range(n + 1)]
            dur = max(5.0, max(abs(t - s0) for s0, t in zip(node.joints, target)) / a.rate)
            os.makedirs(os.path.dirname(SHIFT_FILE), exist_ok=True)
            json.dump({'kind': 'move-to', 'path': path, 'duration': dur}, open(SHIFT_FILE, 'w'))
            node.switch_to([JTC], [POUR, HOLD])
            print('move-to: %.1f s' % dur)
            node.follow([dur * k / n / CLOCK for k in range(n + 1)], path)
            print('move-to: arrived; saved %s' % SHIFT_FILE)
            return 0
        if a.cmd == 'unshift':
            sh = json.load(open(SHIFT_FILE))
            path = list(reversed(sh['path']))
            if not near(node.joints, path[0], 0.08):
                sys.exit('the arm is not at the shifted pour start; not moving')
            node.switch_to([JTC], [POUR, HOLD])
            if sh.get('times'):
                end = sh['times'][-1]
                node.follow([(end - t) / CLOCK for t in reversed(sh['times'])], path)
            else:
                dur = sh.get('duration') or max(
                    3.0, math.hypot(*[a_ - b_ for a_, b_ in zip(sh['receiver_now'], sh['receiver_plan'])]) / 0.02)
                node.follow([dur * k / (len(path) - 1) / CLOCK for k in range(len(path))], path)
            os.replace(SHIFT_FILE, SHIFT_FILE + '.done')
            print('unshift: back at the plan\'s pour start')
            return 0
        if a.cmd in ('to-start', 'to-home'):
            times, points = read_plan(os.path.expanduser(a.plan))
            if a.cmd == 'to-home':
                end = times[-1]
                times = [end - t for t in reversed(times)]
                points = list(reversed(points))
            if not near(node.joints, points[0], 0.08):
                sys.exit('the arm is not where this path starts (%s); not moving. Joints now %s, path starts %s'
                         % (a.cmd, ['%.3f' % q for q in node.joints], ['%.3f' % q for q in points[0]]))
            node.switch_to([JTC], [POUR, HOLD])
            if a.cmd == 'to-start':
                node.grip(False)
            print('%s: %d points, %.1f s' % (a.cmd, len(points), (times[-1] - times[0]) * CLOCK))
            node.follow(times, points)
            print('%s: arrived' % a.cmd)
            return 0

        if a.cmd == 'console':
            return console(node, a)

        targets = [float(v) for v in a.targets.split(',')]
        label = a.label or ('%s%s' % (a.material, '' if a.flow_index == 0 else '_f%g' % a.flow_index))
        out = os.path.expanduser(a.out or '~/real_cell_runs/pour_series/%s_%s'
                                 % (time.strftime('%Y%m%d-%H%M%S'), label))
        os.makedirs(out, exist_ok=True)
        json.dump(vars(a), open(os.path.join(out, 'args.json'), 'w'), indent=1)
        start_pose = list(node.joints)
        print('series %s: %s g x %d, flow_index %g, into %s' % (label, a.targets, a.repeat, a.flow_index, out))

        if ask('\n[jaws] put the vessel between the open jaws, then Enter to close them (q quits): ') == 'q':
            return 1
        grasp = close_jaws(node)
        log = Log(out)
        plan_ = [(tg, k) for tg in targets for k in range(a.repeat)]
        i = 0
        while i < len(plan_):
            tg, k = plan_[i]
            ans = ask('\n[%d/%d] %g g (#%d of %d) -- refill the vessel, empty the receiver, then Enter '
                      '(s skips, q quits): ' % (i + 1, len(plan_), tg, k + 1, a.repeat))
            if ans == 'q':
                break
            if ans == 's':
                i += 1
                continue
            node.grip(True)               # holds already; drains the robot's motion queue
            time.sleep(1.0)
            if pour_once(node, log, label, a.material, a.flow_index, a.timeout, tg, grasp, start_pose,
                         a.bringup_log) is not None:
                i += 1

        log.close()
        if ask('\n[jaws] hold the vessel, then Enter to open the jaws: ') != 'q':
            open_jaws(node)
        print('series done: %s' % out)
        return 0
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
