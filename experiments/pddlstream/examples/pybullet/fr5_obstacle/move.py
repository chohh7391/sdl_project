#!/usr/bin/env python

from __future__ import print_function

import math
import random

from pddlstream.algorithms.meta import solve, create_parser
from pddlstream.language.constants import print_solution, PDDLProblem
from pddlstream.language.generator import from_gen_fn, from_fn, from_test, universe_test
from pddlstream.utils import read, INF, get_file_path, str_from_object, negate_test

from examples.pybullet.fr5_obstacle import stir as move_stir
from examples.pybullet.utils.pybullet_tools.fr5_primitives import (
    BodyPose,
    BodyConf,
    Command,
    get_grasp_gen,
    get_stable_gen,
    get_free_motion_gen,
    get_movable_collision_test,
)
from examples.pybullet.utils.pybullet_tools.utils import (
    WorldSaver,
    connect,
    disconnect,
    load_model,
    set_pose,
    get_pose,
    Pose,
    Point,
    stable_z,
    HideOutput,
    LockRenderer,
    has_gui,
    wait_for_user,
    set_default_camera,
    draw_global_system,
    draw_pose,
    get_configuration,
    is_placement,
    pairwise_collision,
    set_color,
    FR5_AG95_URDF,
    BEAKER_URDF,
    SINK_URDF,
    BLOCK_URDF,
    get_movable_joints,
    set_joint_positions,
)
from examples.pybullet.utils.pybullet_tools.fr5_primitives import get_tool_link
from examples.pybullet.tamp.streams import get_cfree_pose_pose_test

# Keep this aligned with run_move_trials.py difficulty settings.
OBJ_X_RANGE = (0.25, 0.52)
OBJ_Y_RANGE = (-0.34, 0.34)
GOAL_X_RANGE = (0.40, 0.53)
GOAL_Y_RANGE = (-0.42, 0.32)
MIN_OBJ_GOAL_XY_DIST = 0.18

MOVE_GOAL_XY_OFFSETS = (
    (0.00, 0.00),
    (0.01, 0.00), (-0.01, 0.00),
    (0.00, 0.01), (0.00, -0.01),
)
MOVE_GOAL_YAWS = (0.0, math.pi / 2)


def sample_xy(x_range, y_range):
    return random.uniform(*x_range), random.uniform(*y_range)


def load_world(with_obstacle=True):
    set_default_camera()
    draw_global_system()
    with HideOutput():
        robot = load_model(FR5_AG95_URDF, fixed_base=True)
        set_pose(robot, Pose(Point(z=0.05)))
        set_joint_positions(robot, get_movable_joints(robot)[:6], move_stir.FR5_HOME_ARM)

        floor = load_model('models/short_floor.urdf')
        target_obj_1 = load_model(BLOCK_URDF, fixed_base=False)
        goal_surface = load_model(SINK_URDF, fixed_base=True)
        obstacle = None
        if with_obstacle:
            # Same size as target object, fixed as an obstacle.
            obstacle = load_model(BLOCK_URDF, fixed_base=True)

    draw_pose(Pose(), parent=robot, parent_link=get_tool_link(robot))
    if obstacle is not None:
        set_color(obstacle, (0.92, 0.35, 0.12, 1.0))

    body_names = {
        target_obj_1: 'target_obj_1',
        goal_surface: 'goal_surface',
    }
    if obstacle is not None:
        body_names[obstacle] = 'obstacle'
    return robot, floor, target_obj_1, goal_surface, obstacle, body_names


def sample_layout(robot, floor, target_obj_1, goal_surface, obstacle, max_attempts=10):
    goal_z = stable_z(goal_surface, floor)
    obj1_z = stable_z(target_obj_1, floor)
    obstacle_z = stable_z(obstacle, floor) if obstacle is not None else None

    for _ in range(max_attempts):
        obj1_x, obj1_y = sample_xy(OBJ_X_RANGE, OBJ_Y_RANGE)
        goal_x, goal_y = sample_xy(GOAL_X_RANGE, GOAL_Y_RANGE)

        if math.hypot(obj1_x - goal_x, obj1_y - goal_y) < MIN_OBJ_GOAL_XY_DIST:
            continue

        obstacle_x = None
        obstacle_y = None
        if obstacle is not None:
            # Place obstacle midway between start and goal as requested.
            obstacle_x = 0.5 * (obj1_x + goal_x)
            obstacle_y = 0.5 * (obj1_y + goal_y)

        set_pose(target_obj_1, Pose(Point(x=obj1_x, y=obj1_y, z=obj1_z)))
        set_pose(goal_surface, Pose(Point(x=goal_x, y=goal_y, z=goal_z)))
        if obstacle is not None:
            set_pose(obstacle, Pose(Point(x=obstacle_x, y=obstacle_y, z=obstacle_z)))

        # Reject intersecting initial layouts.
        bodies = [target_obj_1, goal_surface]
        if obstacle is not None:
            bodies.append(obstacle)
        if any(pairwise_collision(b1, b2)
               for i, b1 in enumerate(bodies)
               for b2 in bodies[i + 1:]):
            continue

        # Reject robot initial collisions.
        if any(pairwise_collision(robot, body) for body in bodies):
            continue

        return {
            'target_obj_1_x': obj1_x,
            'target_obj_1_y': obj1_y,
            'goal_x': goal_x,
            'goal_y': goal_y,
            'obstacle_x': obstacle_x,
            'obstacle_y': obstacle_y,
        }

    raise RuntimeError('Failed to sample a collision-free obstacle layout within max attempts.')


def get_move_pose_gen(fixed=None, target_obj_1=None, goal_surface=None, obj1_goal_center=None, obj1_goal_z=None):
    if fixed is None:
        fixed = []
    base_gen = get_stable_gen(fixed)

    def gen(body, surface):
        if (
            (body == target_obj_1)
            and (surface == goal_surface)
            and (obj1_goal_center is not None)
            and (obj1_goal_z is not None)
        ):
            gx, gy = obj1_goal_center
            for dx, dy in MOVE_GOAL_XY_OFFSETS:
                for yaw in MOVE_GOAL_YAWS:
                    quat = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
                    pose = ((gx + dx, gy + dy, obj1_goal_z), quat)
                    yield (BodyPose(body, pose),)
        for (body_pose,) in base_gen(body, surface):
            yield (body_pose,)

    return gen


def pddlstream_from_move_problem(
    robot,
    target_obj_1,
    goal_surface,
    obj1_goal_center=None,
    obj1_goal_z=None,
    movable=None,
    teleport=False,
):
    if movable is None:
        movable = [target_obj_1]

    domain_pddl = read(get_file_path(__file__, 'domain_stir.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream_stir.pddl'))
    constant_map = {}

    conf = BodyConf(robot, get_configuration(robot))
    init = [
        ('CanMove',),
        ('Conf', conf),
        ('AtConf', conf),
        ('HandEmpty',),
        ('BaseObject', target_obj_1, goal_surface),
    ]

    fixed = move_stir.get_fixed(robot, movable)
    for body in movable:
        pose = BodyPose(body, get_pose(body))
        init += [
            ('Graspable', body),
            ('Pose', body, pose),
            ('Stable', body, pose),
            ('AtPose', body, pose),
        ]
        for surface in fixed:
            if body == surface:
                continue
            if is_placement(body, surface):
                init += [('Supported', body, pose, surface)]

    init += [('Stackable', target_obj_1, goal_surface)]

    goal = (
        'and',
        ('HandEmpty',),
        ('On', target_obj_1, goal_surface),
    )

    stream_map = {
        'sample-pose': from_gen_fn(
            get_move_pose_gen(
                fixed=fixed,
                target_obj_1=target_obj_1,
                goal_surface=goal_surface,
                obj1_goal_center=obj1_goal_center,
                obj1_goal_z=obj1_goal_z,
            )
        ),
        'sample-grasp': from_gen_fn(get_grasp_gen(robot, 'top')),
        'inverse-kinematics': from_fn(
            move_stir.get_stir_ik_fn(
                robot, fixed, target_obj_1, teleport, num_attempts=move_stir.STIR_IK_NUM_ATTEMPTS
            )
        ),
        'plan-free-motion': from_fn(get_free_motion_gen(robot, fixed, teleport)),
        'plan-holding-motion': from_fn(
            move_stir.get_stir_holding_motion_gen(robot, fixed, target_obj_1, teleport)
        ),
        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test()),
        'test-cfree-approach-pose': from_test(universe_test),
        'test-cfree-traj-pose': from_test(negate_test(get_movable_collision_test())),
        'TrajCollision': get_movable_collision_test(),
    }
    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


def compute_goal_params(target_obj_1, goal_surface):
    (gx, gy, _), _ = get_pose(goal_surface)
    obj1_goal_z = stable_z(target_obj_1, goal_surface)
    return (gx, gy), obj1_goal_z


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
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max-layout-attempts', type=int, default=500)
    obstacle_group = parser.add_mutually_exclusive_group()
    # obstacle_group.add_argument(
    #     '--with-obstacle',
    #     dest='with_obstacle',
    #     action='store_false',
    #     help='Enable midpoint obstacle (default)',
    # )
    obstacle_group.add_argument(
        '--obs',
        dest='with_obstacle',
        action='store_true',
        help='Disable obstacle',
    )
    parser.set_defaults(with_obstacle=False)
    args = parser.parse_args()
    print('Arguments:', args)

    if args.seed is not None:
        random.seed(args.seed)

    move_stir.configure_stir_motion_constraints()
    connect(use_gui=True)
    try:
        robot, floor, target_obj_1, goal_surface, obstacle, names = load_world(
            with_obstacle=args.with_obstacle
        )
        layout = sample_layout(
            robot, floor, target_obj_1, goal_surface, obstacle,
            max_attempts=args.max_layout_attempts,
        )
        print('Objects:', names)
        print('Layout:', layout)

        obj1_goal_center, obj1_goal_z = compute_goal_params(target_obj_1, goal_surface)
        problem = pddlstream_from_move_problem(
            robot,
            target_obj_1=target_obj_1,
            goal_surface=goal_surface,
            obj1_goal_center=obj1_goal_center,
            obj1_goal_z=obj1_goal_z,
            movable=[target_obj_1],
            teleport=args.teleport,
        )
        _, _, _, stream_map, init, goal = problem
        print('Init:', init)
        print('Goal:', goal)
        print('Streams:', str_from_object(set(stream_map)))

        saver = WorldSaver()
        with LockRenderer(lock=not args.enable):
            solution = solve(
                problem,
                algorithm=args.algorithm,
                unit_costs=args.unit,
                success_cost=INF,
            )
            saver.restore()
        print_solution(solution)

        plan, _, _ = solution
        if (plan is None) or not has_gui():
            return

        command = postprocess_plan(plan)
        if args.simulate:
            wait_for_user('Simulate?')
            command.control()
        else:
            wait_for_user('Execute?')
            command.refine(num_steps=10).execute(time_step=0.001)
        wait_for_user('Finish?')
    finally:
        disconnect()


if __name__ == '__main__':
    main()
