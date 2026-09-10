#!/usr/bin/env python

from __future__ import print_function

import argparse
import csv
import math
import os
import random
import time

from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import PDDLProblem
from pddlstream.language.generator import from_gen_fn, from_fn, from_test, universe_test
from pddlstream.utils import read, INF, get_file_path, negate_test

from examples.pybullet.fr5_Move import stir as move_stir
from examples.pybullet.utils.pybullet_tools.fr5_primitives import (
    BodyPose,
    BodyConf,
    get_grasp_gen,
    get_stable_gen,
    get_free_motion_gen,
    get_movable_collision_test,
)
from examples.pybullet.utils.pybullet_tools.utils import (
    connect,
    disconnect,
    load_model,
    set_pose,
    get_pose,
    Pose,
    Point,
    stable_z,
    HideOutput,
    FR5_AG95_URDF,
    BEAKER_URDF,
    SINK_URDF,
    set_joint_positions,
    get_movable_joints,
    pairwise_collision,
    get_configuration,
    is_placement,
)
from examples.pybullet.tamp.streams import get_cfree_pose_pose_test

DEFAULT_TRIALS = 30
DEFAULT_MAX_TIME = 120.0
DEFAULT_MAX_ITERATIONS = 1024
DEFAULT_CSV_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Keep layouts mostly reachable but not overly easy.
OBJ_X_RANGE = (0.25, 0.52)
OBJ_Y_RANGE = (-0.34, 0.34)
GOAL_X_RANGE = (0.27, 0.53)
GOAL_Y_RANGE = (-0.32, 0.32)
MIN_OBJ_GOAL_XY_DIST = 0.18

# Candidate on-goal placements for obj1 (reduced to avoid over-relaxing).
MOVE_GOAL_XY_OFFSETS = (
    (0.00, 0.00),
    (0.01, 0.00), (-0.01, 0.00),
    (0.00, 0.01), (0.00, -0.01),
)
MOVE_GOAL_YAWS = (0.0, math.pi / 2)


def sample_xy(x_range, y_range):
    return random.uniform(*x_range), random.uniform(*y_range)


def load_world():
    with HideOutput():
        robot = load_model(FR5_AG95_URDF, fixed_base=True)
        set_pose(robot, Pose(Point(z=0.05)))
        set_joint_positions(robot, get_movable_joints(robot)[:6], move_stir.FR5_HOME_ARM)

        floor = load_model('models/short_floor.urdf')
        target_obj_1 = load_model(BEAKER_URDF, fixed_base=False)
        goal_surface = load_model(SINK_URDF, fixed_base=True)
    return robot, floor, target_obj_1, goal_surface


def sample_layout(robot, floor, target_obj_1, goal_surface, max_attempts=3):
    goal_z = stable_z(goal_surface, floor)
    for _ in range(max_attempts):
        obj1_x, obj1_y = sample_xy(OBJ_X_RANGE, OBJ_Y_RANGE)
        goal_x, goal_y = sample_xy(GOAL_X_RANGE, GOAL_Y_RANGE)
        if math.hypot(obj1_x - goal_x, obj1_y - goal_y) < MIN_OBJ_GOAL_XY_DIST:
            continue

        set_pose(target_obj_1, Pose(Point(x=obj1_x, y=obj1_y, z=stable_z(target_obj_1, floor))))
        set_pose(goal_surface, Pose(Point(x=goal_x, y=goal_y, z=goal_z)))

        if pairwise_collision(target_obj_1, goal_surface):
            continue
        # Reject layouts where robot starts in collision with scene objects.
        if pairwise_collision(robot, target_obj_1) or pairwise_collision(robot, goal_surface):
            continue
        return {
            'target_obj_1_x': obj1_x,
            'target_obj_1_y': obj1_y,
            'goal_x': goal_x,
            'goal_y': goal_y,
        }
    raise RuntimeError('Failed to sample a collision-free layout within the maximum attempts.')


def get_move_pose_gen(
    fixed=None,
    target_obj_1=None,
    goal_surface=None,
    obj1_goal_center=None,
    obj1_goal_z=None,
):
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
            # Also allow fallback stable samples on the same support.
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


def run_single_trial(trial_idx, max_time, max_iterations):
    result = {
        'trial': trial_idx,
        'success': 0,
        'planning_time_sec': 0.0,
        'target_obj_1_x': float('nan'),
        'target_obj_1_y': float('nan'),
        'goal_x': float('nan'),
        'goal_y': float('nan'),
    }

    connect(use_gui=False)
    trial_start = time.perf_counter()
    try:
        move_stir.configure_stir_motion_constraints()
        robot, floor, target_obj_1, goal_surface = load_world()
        try:
            layout = sample_layout(robot, floor, target_obj_1, goal_surface)
        except RuntimeError as err:
            print('[trial {}] layout sampling failed: {}'.format(trial_idx, err))
            result['success'] = 0
            result['planning_time_sec'] = time.perf_counter() - trial_start
            return result
        result.update(layout)

        obj1_goal_center, obj1_goal_z = compute_goal_params(target_obj_1, goal_surface)
        problem = pddlstream_from_move_problem(
            robot,
            target_obj_1=target_obj_1,
            goal_surface=goal_surface,
            obj1_goal_center=obj1_goal_center,
            obj1_goal_z=obj1_goal_z,
            movable=[target_obj_1],
            teleport=False,
        )

        start_time = time.perf_counter()
        try:
            solution = solve(
                problem,
                algorithm='adaptive',
                unit_costs=True,
                success_cost=INF,
                max_time=max_time,
                max_iterations=max_iterations,
                verbose=False,
            )
            result['success'] = int(solution[0] is not None)
        except Exception:
            result['success'] = 0
        result['planning_time_sec'] = time.perf_counter() - start_time
        return result
    except Exception as err:
        print('[trial {}] unexpected error: {}'.format(trial_idx, err))
        result['success'] = 0
        result['planning_time_sec'] = time.perf_counter() - trial_start
        return result
    finally:
        disconnect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS)
    parser.add_argument('--max-time', type=float, default=DEFAULT_MAX_TIME)
    parser.add_argument('--max-iterations', type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--csv-path', default=None)
    args = parser.parse_args()

    if args.csv_path is None:
        args.csv_path = os.path.join(DEFAULT_CSV_DIR, 'move_fr5.csv')
    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)

    fieldnames = [
        'trial',
        'success',
        'planning_time_sec',
        'target_obj_1_x',
        'target_obj_1_y',
        'goal_x',
        'goal_y',
    ]

    with open(args.csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trial_idx in range(1, args.trials + 1):
            result = run_single_trial(
                trial_idx=trial_idx,
                max_time=args.max_time,
                max_iterations=args.max_iterations,
            )
            writer.writerow(result)
            f.flush()
            print('[{}/{}][fr5-move] success={} time={:.3f}s'.format(
                trial_idx, args.trials, result['success'], result['planning_time_sec']))

    print('Saved:', args.csv_path)


if __name__ == '__main__':
    main()
