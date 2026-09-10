#!/usr/bin/env python

from __future__ import print_function

import math
import random
import time

from pddlstream.algorithms.meta import solve, create_parser
from pddlstream.language.constants import print_solution, PDDLProblem
from pddlstream.language.generator import from_gen_fn, from_fn, from_test, universe_test
from pddlstream.utils import read, INF, get_file_path, Profiler, str_from_object, negate_test

from examples.pybullet.utils.pybullet_tools.fr5_primitives import (
    BodyPose, BodyConf, Command, get_grasp_gen, get_stable_gen, get_ik_fn,
    get_free_motion_gen, get_holding_motion_gen, get_movable_collision_test, get_tool_link
)
import examples.pybullet.utils.pybullet_tools.fr5_primitives as fr5_primitives
from examples.pybullet.utils.pybullet_tools.utils import (
    WorldSaver, connect, get_pose, set_pose, Pose, Point, set_default_camera, stable_z, BLOCK_URDF, SINK_URDF,
    load_model, is_placement, disconnect, get_bodies, HideOutput, wait_for_user, LockRenderer, has_gui,
    draw_pose, draw_global_system, FR5_AG95_URDF, BEAKER_URDF, get_movable_joints, set_joint_positions, get_configuration,
    set_color, pairwise_collision,
)
from examples.pybullet.tamp.streams import (
    get_cfree_pose_pose_test, get_cfree_obj_approach_pose_test
)

TIMED_STREAMS = (
    'inverse-kinematics',
    'plan-free-motion',
    'plan-holding-motion',
)
FR5_HOME_ARM = (0.0, -1.05, -2.18, -1.57, 1.57, 0.0)
STACK_Z_OFFSET = 0.03
TARGET_OBJ2_SCALE = 0.4
STIR_MAX_CARRY_TILT = math.pi
OBJ1_MAX_CARRY_TILT = math.radians(5)
STIR_IK_NUM_ATTEMPTS = 30

# Favor IK-friendly top placements for obj2 first: centered, slight positive clearance, limited yaw.
OBJ2_XY_OFFSETS = (
    (0.00, 0.00),
    (0.005, 0.00), (-0.005, 0.00),
    (0.00, 0.005), (0.00, -0.005),
    (0.01, 0.00), (-0.01, 0.00),
    (0.00, 0.01), (0.00, -0.01),
)
OBJ2_Z_OFFSETS = (0.02, 0.03, 0.01, 0.04)
OBJ2_YAW_ANGLES = (0.0, math.pi / 2, math.pi / 4, -math.pi / 4)

# Randomized layout ranges that remain generally reachable for FR5.
OBJ1_X_RANGE = (0.18, 0.36)
OBJ1_Y_RANGE = (-0.52, -0.24)
OBJ2_X_RANGE = (0.32, 0.50)
OBJ2_Y_RANGE = (-0.52, -0.24)
STIRRER_X_RANGE = (0.30, 0.46)
STIRRER_Y_RANGE = (0.24, 0.46)
MIN_OBJ1_STIRRER_DIST = 0.38
MIN_OBJ1_OBJ2_DIST = 0.10


def sample_xy(x_range, y_range):
    return random.uniform(*x_range), random.uniform(*y_range)


def sample_stir_layout(robot, floor, target_obj_1, target_obj_2, stirrer, obstacle=None, max_attempts=300):
    obj1_z = stable_z(target_obj_1, floor)
    obj2_z = stable_z(target_obj_2, floor)
    stirrer_z = stable_z(stirrer, floor)
    obstacle_z = stable_z(obstacle, floor) if obstacle is not None else None

    for _ in range(max_attempts):
        obj1_x, obj1_y = sample_xy(OBJ1_X_RANGE, OBJ1_Y_RANGE)
        obj2_x, obj2_y = sample_xy(OBJ2_X_RANGE, OBJ2_Y_RANGE)
        stirrer_x, stirrer_y = sample_xy(STIRRER_X_RANGE, STIRRER_Y_RANGE)

        if math.hypot(obj1_x - stirrer_x, obj1_y - stirrer_y) < MIN_OBJ1_STIRRER_DIST:
            continue
        if math.hypot(obj1_x - obj2_x, obj1_y - obj2_y) < MIN_OBJ1_OBJ2_DIST:
            continue

        set_pose(target_obj_1, Pose(Point(x=obj1_x, y=obj1_y, z=obj1_z)))
        set_pose(target_obj_2, Pose(Point(x=obj2_x, y=obj2_y, z=obj2_z)))
        set_pose(stirrer, Pose(Point(x=stirrer_x, y=stirrer_y, z=stirrer_z)))

        obstacle_x = None
        obstacle_y = None
        if obstacle is not None:
            # Place obstacle between obj1 and stirrer.
            obstacle_x = 0.8 * (obj1_x + stirrer_x)
            obstacle_y = 0.5 * (obj1_y + stirrer_y)
            set_pose(obstacle, Pose(Point(x=obstacle_x, y=obstacle_y, z=obstacle_z)))

        bodies = [target_obj_1, target_obj_2, stirrer]
        if obstacle is not None:
            bodies.append(obstacle)

        if any(pairwise_collision(b1, b2)
               for i, b1 in enumerate(bodies)
               for b2 in bodies[i + 1:]):
            continue
        if any(pairwise_collision(robot, body) for body in bodies):
            continue

        return {
            'target_obj_1_x': obj1_x,
            'target_obj_1_y': obj1_y,
            'target_obj_2_x': obj2_x,
            'target_obj_2_y': obj2_y,
            'stirrer_x': stirrer_x,
            'stirrer_y': stirrer_y,
            'obstacle_x': obstacle_x,
            'obstacle_y': obstacle_y,
        }

    raise RuntimeError('Failed to sample a collision-free stir layout within max attempts.')


def create_timing_stats():
    return {name: {'calls': 0, 'time': 0.0} for name in TIMED_STREAMS}


def wrap_timed_fn(name, fn, timing_stats):
    def wrapped(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            stats = timing_stats[name]
            stats['calls'] += 1
            stats['time'] += time.perf_counter() - start_time
    return wrapped


def print_timing_breakdown(total_solve_time, timing_stats):
    motion_time = sum(timing_stats[name]['time'] for name in TIMED_STREAMS)
    task_time = max(0.0, total_solve_time - motion_time)
    print('\nPlanning Time Breakdown:')
    print('  Total solve time: {:.3f}s'.format(total_solve_time))
    print('  Task planning time (est.): {:.3f}s'.format(task_time))
    print('  Motion planning time: {:.3f}s'.format(motion_time))
    for name in TIMED_STREAMS:
        stats = timing_stats[name]
        print('    - {}: {:.3f}s (calls={})'.format(name, stats['time'], stats['calls']))


def configure_stir_motion_constraints():
    # Stir task allows tilting during move_holding / IK retreat unlike liquid transfer.
    fr5_primitives.MAX_CARRY_TILT = STIR_MAX_CARRY_TILT


def get_body_tilt_limit(body, target_obj_1):
    # Keep obj1 orientation constrained while allowing looser tilt for other carried objects.
    return OBJ1_MAX_CARRY_TILT if body == target_obj_1 else STIR_MAX_CARRY_TILT


def get_stir_ik_fn(robot, fixed, target_obj_1, teleport=False, num_attempts=10):
    base_ik_fn = get_ik_fn(robot, fixed, teleport, num_attempts=num_attempts)

    def fn(body, pose, grasp):
        prev_tilt = fr5_primitives.MAX_CARRY_TILT
        fr5_primitives.MAX_CARRY_TILT = get_body_tilt_limit(body, target_obj_1)
        try:
            return base_ik_fn(body, pose, grasp)
        finally:
            fr5_primitives.MAX_CARRY_TILT = prev_tilt

    return fn


def get_stir_holding_motion_gen(robot, fixed, target_obj_1, teleport=False):
    base_holding_motion_fn = get_holding_motion_gen(robot, fixed, teleport)

    def fn(conf1, conf2, body, grasp, fluents=[]):
        prev_tilt = fr5_primitives.MAX_CARRY_TILT
        fr5_primitives.MAX_CARRY_TILT = get_body_tilt_limit(body, target_obj_1)
        try:
            return base_holding_motion_fn(conf1, conf2, body, grasp, fluents=fluents)
        finally:
            fr5_primitives.MAX_CARRY_TILT = prev_tilt

    return fn


def get_fixed(robot, movable):
    rigid = [body for body in get_bodies() if body != robot]
    return [body for body in rigid if body not in movable]


def get_stir_pose_gen(fixed=None, target_obj_1=None, target_obj_2=None, stirrer=None,
                      obj1_goal_pose=None, obj2_goal_z=None, obj2_goal_quat=None,
                      z_offset=STACK_Z_OFFSET):
    if fixed is None:
        fixed = []
    base_gen = get_stable_gen(fixed)

    def gen(body, surface):
        # Force target_obj_1 final placement to a predefined stirrer-top pose.
        if (body == target_obj_1) and (surface == stirrer) and (obj1_goal_pose is not None):
            yield (BodyPose(body, obj1_goal_pose),)
            return
        # Sample target_obj_2 poses around target_obj_1's goal pose, not its current pose.
        if (body == target_obj_2) and (surface == target_obj_1) and (obj1_goal_pose is not None) and (obj2_goal_z is not None):
            (gx, gy, _), _ = obj1_goal_pose
            for dz in OBJ2_Z_OFFSETS:
                for dx, dy in OBJ2_XY_OFFSETS:
                    for yaw in OBJ2_YAW_ANGLES:
                        # Z축 기준 회전을 나타내는 Quaternion 생성
                        quat = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
                        pose = ((gx + dx, gy + dy, obj2_goal_z + dz), quat)
                        yield (BodyPose(body, pose),)
            return
        for (body_pose,) in base_gen(body, surface):
            yield (body_pose,)
    return gen


def pddlstream_from_problem(robot, target_obj_1, target_obj_2, stirrer,
                            obj1_goal_pose=None, obj2_goal_z=None, obj2_goal_quat=None, movable=None,
                            teleport=False, grasp_name='top', timing_stats=None):
    if movable is None:
        movable = [target_obj_1, target_obj_2]
    if timing_stats is None:
        timing_stats = create_timing_stats()

    domain_pddl = read(get_file_path(__file__, 'domain_stir.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream_stir.pddl'))
    constant_map = {}

    print('Robot:', robot)
    conf = BodyConf(robot, get_configuration(robot))
    init = [
        ('CanMove',),
        ('Conf', conf),
        ('AtConf', conf),
        ('HandEmpty',),
        ('BaseObject', target_obj_1, stirrer),
        ('TopObject', target_obj_2),
    ]

    fixed = get_fixed(robot, movable)
    print('Movable:', movable)
    print('Fixed:', fixed)

    for body in movable:
        pose = BodyPose(body, get_pose(body))
        init += [
            ('Graspable', body),
            ('Pose', body, pose),
            ('Stable', body, pose),
            ('AtPose', body, pose),
        ]
        # Support relations are facts for current placements.
        for surface in fixed + [target_obj_1]:
            if body == surface:
                continue
            if is_placement(body, surface):
                init += [('Supported', body, pose, surface)]

    # Restrict candidate placement surfaces to the intended stir sequence only.
    init += [
        ('Stackable', target_obj_1, stirrer),
        ('Stackable', target_obj_2, target_obj_1),
    ]

    goal = (
        'and',
        ('HandEmpty',),
        ('On', target_obj_1, stirrer),
        ('On', target_obj_2, target_obj_1),
    )

    stream_map = {
        'sample-pose': from_gen_fn(
            get_stir_pose_gen(
                fixed=fixed,
                target_obj_1=target_obj_1,
                target_obj_2=target_obj_2,
                stirrer=stirrer,
                obj1_goal_pose=obj1_goal_pose,
                obj2_goal_z=obj2_goal_z,
                obj2_goal_quat=obj2_goal_quat,
                z_offset=STACK_Z_OFFSET,
            )
        ),
        'sample-grasp': from_gen_fn(get_grasp_gen(robot, grasp_name)),
        'inverse-kinematics': from_fn(
            wrap_timed_fn(
                'inverse-kinematics',
                get_stir_ik_fn(
                    robot, fixed, target_obj_1, teleport, num_attempts=STIR_IK_NUM_ATTEMPTS
                ),
                timing_stats,
            )
        ),
        'plan-free-motion': from_fn(
            wrap_timed_fn('plan-free-motion', get_free_motion_gen(robot, fixed, teleport), timing_stats)
        ),
        'plan-holding-motion': from_fn(
            wrap_timed_fn(
                'plan-holding-motion',
                get_stir_holding_motion_gen(robot, fixed, target_obj_1, teleport),
                timing_stats,
            )
        ),
        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test()),
        # For stir stacking, allow near-contact approach to support surfaces.
        'test-cfree-approach-pose': from_test(universe_test),
        'test-cfree-traj-pose': from_test(negate_test(get_movable_collision_test())),
        'TrajCollision': get_movable_collision_test(),
    }
    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


def load_world(with_obstacle=False, max_layout_attempts=300):
    set_default_camera()
    draw_global_system()
    with HideOutput():
        robot = load_model(FR5_AG95_URDF, fixed_base=True)
        set_pose(robot, Pose(Point(z=0.05)))
        set_joint_positions(robot, get_movable_joints(robot)[:6], FR5_HOME_ARM)

        floor = load_model('models/short_floor.urdf')
        target_obj_1 = load_model(BEAKER_URDF, fixed_base=False)
        target_obj_2 = load_model(BLOCK_URDF, fixed_base=False, scale=TARGET_OBJ2_SCALE)
        stirrer = load_model(SINK_URDF, fixed_base=True, pose=Pose(Point(x=0.35, y=-0.35)))
        obstacle = None
        if with_obstacle:
            obstacle = load_model(BLOCK_URDF, fixed_base=True)

    draw_pose(Pose(), parent=robot, parent_link=get_tool_link(robot))

    body_names = {
        target_obj_1: 'target_obj_1',
        target_obj_2: 'target_obj_2',
        stirrer: 'stirrer',
    }
    if obstacle is not None:
        body_names[obstacle] = 'obstacle'

    layout = sample_stir_layout(
        robot, floor, target_obj_1, target_obj_2, stirrer, obstacle=obstacle, max_attempts=max_layout_attempts
    )
    if obstacle is not None:
        set_color(obstacle, (0.95, 0.35, 0.1, 1.0))

    obj1_initial_pose = get_pose(target_obj_1)
    stirrer_xyz, _ = get_pose(stirrer)
    obj1_goal_z = stable_z(target_obj_1, stirrer)
    obj1_goal_pose = Pose(
        Point(x=stirrer_xyz[0], y=stirrer_xyz[1], z=obj1_goal_z)
    )
    # Compute obj2 stacking height when obj1 is at its goal pose.
    set_pose(target_obj_1, obj1_goal_pose)
    obj2_goal_z = stable_z(target_obj_2, target_obj_1)
    set_pose(target_obj_1, obj1_initial_pose)
    _, obj2_goal_quat = get_pose(target_obj_2)

    return robot, body_names, [target_obj_1, target_obj_2], stirrer, obj1_goal_pose, obj2_goal_z, obj2_goal_quat, layout


def postprocess_plan(plan):
    paths = []
    for name, args in plan:
        if name in ['place', 'place-base', 'place-top']:
            paths += args[-1].reverse().body_paths
        elif name in ['move', 'move_free', 'move_holding', 'pick']:
            paths += args[-1].body_paths
    return Command(paths)


def main():
    parser = create_parser()
    parser.set_defaults(unit=True)
    parser.add_argument('-enable', action='store_true', help='Enables rendering during planning')
    parser.add_argument('-teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('-simulate', action='store_true', help='Simulates the system')
    parser.add_argument('--obs', dest='with_obstacle', action='store_true', help='Enable obstacle')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max-layout-attempts', type=int, default=300)
    parser.set_defaults(with_obstacle=False)
    args = parser.parse_args()
    print('Arguments:', args)

    if args.seed is not None:
        random.seed(args.seed)

    configure_stir_motion_constraints()
    connect(use_gui=True)
    robot, names, movable, stirrer, obj1_goal_pose, obj2_goal_z, obj2_goal_quat, layout = load_world(
        with_obstacle=args.with_obstacle,
        max_layout_attempts=args.max_layout_attempts,
    )
    target_obj_1, target_obj_2 = movable
    print('Objects:', names)
    print('Layout:', layout)
    saver = WorldSaver()

    # wait_for_user('환경이 로드되었습니다. 확인 후 터미널에서 Enter를 누르면 종료됩니다.')
    # disconnect()
    # return  

    timing_stats = create_timing_stats()
    problem = pddlstream_from_problem(
        robot,
        target_obj_1=target_obj_1,
        target_obj_2=target_obj_2,
        stirrer=stirrer,
        obj1_goal_pose=obj1_goal_pose,
        obj2_goal_z=obj2_goal_z,
        obj2_goal_quat=obj2_goal_quat,
        movable=movable,
        teleport=args.teleport,
        grasp_name='top',
        timing_stats=timing_stats,
    )
    _, _, _, stream_map, init, goal = problem
    print('Init:', init)
    print('Goal:', goal)
    print('Streams:', str_from_object(set(stream_map)))

    with Profiler():
        with LockRenderer(lock=not args.enable):
            solve_start = time.perf_counter()
            solution = solve(problem, algorithm=args.algorithm, unit_costs=args.unit, success_cost=INF)
            solve_elapsed = time.perf_counter() - solve_start
            saver.restore()
    print_solution(solution)
    print_timing_breakdown(solve_elapsed, timing_stats)
    plan, _, _ = solution
    if (plan is None) or not has_gui():
        disconnect()
        return

    command = postprocess_plan(plan)
    if args.simulate:
        wait_for_user('Simulate?')
        command.control()
    else:
        wait_for_user('Execute?')
        command.refine(num_steps=10).execute(time_step=0.001)
    wait_for_user('Finish?')
    disconnect()


if __name__ == '__main__':
    main()
