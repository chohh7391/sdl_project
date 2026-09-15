#!/usr/bin/env python3
"""Drive the real FR5 by hand, through the same path a plan will use.

The first real-hardware session should not be a TAMP plan. This is the small
tool that goes in front of it: one command per thing that can go wrong, in the
order they are worth finding out.

    scripts/bridge/cho_bridge_node.py status
    scripts/bridge/cho_bridge_node.py switch joint_trajectory_controller
    scripts/bridge/cho_bridge_node.py nudge j1 0.05 --duration 4
    scripts/bridge/cho_bridge_node.py gripper open
    scripts/bridge/cho_bridge_node.py hold

`nudge` is the one that matters: it reads where the arm actually is, builds a
two-point trajectory to a few degrees away, validates it and sends it through
the trajectory controller action. If that works, the whole command path works,
and it is one small joint motion rather than a plan.

Everything here is deliberately slow and small. The velocity ceiling every
trajectory is checked against lives in `cho_bridge.contract`.
"""

import argparse
import sys
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor

from cho_bridge.contract import (
    ARM_JOINTS,
    HOLD_CONTROLLER,
    MAX_JOINT_VELOCITY,
    TRAJECTORY_CONTROLLER,
)
from cho_bridge.executor import ChoCommandFailed, ChoExecutor
from cho_bridge.trajectory import (
    TrajectoryRejected,
    build_trajectory,
    required_time_scale,
)


def _spun_node(name='cho_bridge'):
    """A node with an executor spinning it in the background.

    ChoExecutor blocks on futures and never spins, so something has to. In
    tamp_server that is its own MultiThreadedExecutor; here it is this thread.
    """
    node = rclpy.create_node(name)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return node, executor


def cmd_status(arm, args):
    states = arm.controller_states()
    print('controllers:')
    for name, state in sorted(states.items()):
        mark = '*' if state == 'active' else ' '
        print(f'  {mark} {name}: {state}')
    print('joints [rad]:')
    for joint, value in arm.joint_positions().items():
        print(f'    {joint}: {value:+.5f}')
    missing = arm.wait_for_stack(timeout_sec=2.0)
    print(f'endpoints missing: {missing if missing else "none"}')
    return 0


def cmd_switch(arm, args):
    arm.switch_to(args.controller)
    print(f'active: {args.controller}')
    return 0


def cmd_hold(arm, args):
    arm.hold()
    print(f'active: {HOLD_CONTROLLER}')
    return 0


def cmd_nudge(arm, args):
    start = arm.joint_positions()
    if args.joint not in start:
        print(f'unknown joint {args.joint}; known: {list(start)}', file=sys.stderr)
        return 2

    target = dict(start)
    target[args.joint] = start[args.joint] + args.delta
    rate = abs(args.delta) / args.duration
    print(f'{args.joint}: {start[args.joint]:+.5f} -> {target[args.joint]:+.5f} rad '
          f'over {args.duration:.1f}s ({rate:.3f} rad/s, ceiling '
          f'{MAX_JOINT_VELOCITY:.3f})')

    traj = build_trajectory(
        [[start[j] for j in ARM_JOINTS], [target[j] for j in ARM_JOINTS]],
        joint_names=ARM_JOINTS,
        times=[0.0, args.duration],
    )
    arm.execute_trajectory(traj)
    print('done')
    return 0


def cmd_goto(arm, args):
    """Drive every joint to a given configuration, in one trajectory.

    `nudge` moves one joint because that is the safest first motion; getting to
    a named configuration -- the planner's own start pose, say -- needs all six
    to move together, and one trajectory is how the arm gets there without
    passing through whatever pose a sequence of single-joint moves implies.
    """
    if len(args.target) != len(ARM_JOINTS):
        print('need %d joint values, got %d' % (len(ARM_JOINTS), len(args.target)),
              file=sys.stderr)
        return 2

    start = arm.joint_positions()
    rows = [[start[j] for j in ARM_JOINTS], list(args.target)]
    scale = required_time_scale(rows, args.duration)
    duration = args.duration * scale
    if scale > 1.0:
        print('stretching to %.1fs: %.1fs would exceed %.3f rad/s'
              % (duration, args.duration, MAX_JOINT_VELOCITY))
    for index, joint in enumerate(ARM_JOINTS):
        print('  %s: %+.5f -> %+.5f rad' % (joint, rows[0][index], rows[1][index]))

    traj = build_trajectory(rows, joint_names=ARM_JOINTS, times=[0.0, duration])
    arm.execute_trajectory(traj)
    print('done')
    return 0


def cmd_gripper(arm, args):
    arm.set_gripper(grasp=(args.action == 'close'), width=args.width)
    print(f'gripper {args.action}')
    return 0


def cmd_pour(arm, args):
    result = arm.pour(args.grams)
    print(f'pour result: {result}')
    return 0


def cmd_relay(arm, args):
    from cho_bridge.scale_relay import main as relay_main
    relay_main()
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('status', help='controllers, joints, endpoints').set_defaults(
        run=cmd_status, needs_arm=True)

    switch = sub.add_parser('switch', help='give the arm to one controller')
    switch.add_argument('controller', nargs='?', default=TRAJECTORY_CONTROLLER)
    switch.set_defaults(run=cmd_switch, needs_arm=True)

    sub.add_parser('hold', help='cancel and park on the hold controller').set_defaults(
        run=cmd_hold, needs_arm=True)

    nudge = sub.add_parser('nudge', help='one small single-joint move, end to end')
    nudge.add_argument('joint', help=f'one of {", ".join(ARM_JOINTS)}')
    nudge.add_argument('delta', type=float, help='radians, signed')
    nudge.add_argument('--duration', type=float, default=4.0)
    nudge.set_defaults(run=cmd_nudge, needs_arm=True)

    goto = sub.add_parser('goto', help='drive every joint to a configuration')
    goto.add_argument('target', type=float, nargs='+',
                      help='%d joint values in radians, in order' % len(ARM_JOINTS))
    goto.add_argument('--duration', type=float, default=8.0)
    goto.set_defaults(run=cmd_goto, needs_arm=True)

    gripper = sub.add_parser('gripper', help='open or close the AG-95')
    gripper.add_argument('action', choices=('open', 'close'))
    gripper.add_argument('--width', type=float, default=0.0)
    gripper.set_defaults(run=cmd_gripper, needs_arm=True)

    pour = sub.add_parser('pour', help='run the pour controller (not built yet)')
    pour.add_argument('grams', type=float)
    pour.set_defaults(run=cmd_pour, needs_arm=True)

    sub.add_parser('relay', help='run the scale relay in the foreground').set_defaults(
        run=cmd_relay, needs_arm=False)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if not args.needs_arm:
        rclpy.init()
        try:
            return args.run(None, args)
        finally:
            rclpy.try_shutdown()

    rclpy.init()
    node, executor = _spun_node()
    try:
        arm = ChoExecutor(node)
        missing = arm.wait_for_stack(timeout_sec=10.0)
        if missing and args.command != 'status':
            print(f'not ready: {missing}. Is the bringup running?', file=sys.stderr)
            return 1
        return args.run(arm, args)
    except (ChoCommandFailed, TrajectoryRejected) as error:
        print(f'refused: {error}', file=sys.stderr)
        return 1
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
