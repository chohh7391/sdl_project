#!/usr/bin/env python

from __future__ import print_function

import argparse
import csv
import os
import random
import time

from pddlstream.algorithms.meta import solve
from pddlstream.utils import INF

from examples.pybullet.fr5_Stir.stir import (
    pddlstream_from_problem,
    configure_stir_motion_constraints,
    FR5_HOME_ARM,
    TARGET_OBJ2_SCALE,
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
    BLOCK_URDF,
    BEAKER_URDF,
    SINK_URDF,
    set_joint_positions,
    get_movable_joints,
    pairwise_collision,
)

DEFAULT_TRIALS = 30
DEFAULT_MAX_TIME = 120.0
DEFAULT_MAX_ITERATIONS = 1024
DEFAULT_CSV_DIR = os.path.join(os.path.dirname(__file__), 'data')

X_RANGE = (0.2, 0.55)
Y_RANGE = (-0.55, 0.55)


def sample_xy():
    return random.uniform(*X_RANGE), random.uniform(*Y_RANGE)


def load_world():
    with HideOutput():
        robot = load_model(FR5_AG95_URDF, fixed_base=True)
        set_pose(robot, Pose(Point(z=0.05)))
        set_joint_positions(robot, get_movable_joints(robot)[:6], FR5_HOME_ARM)

        floor = load_model('models/short_floor.urdf')
        target_obj_1 = load_model(BEAKER_URDF, fixed_base=False)
        target_obj_2 = load_model(BLOCK_URDF, fixed_base=False, scale=TARGET_OBJ2_SCALE)
        stirrer = load_model(SINK_URDF, fixed_base=True)

    return robot, floor, target_obj_1, target_obj_2, stirrer


def sample_layout(floor, target_obj_1, target_obj_2, stirrer, max_attempts=200):
    stirrer_z = stable_z(stirrer, floor)
    for _ in range(max_attempts):
        obj1_x, obj1_y = sample_xy()
        obj2_x, obj2_y = sample_xy()
        stirrer_x, stirrer_y = sample_xy()

        set_pose(target_obj_1, Pose(Point(x=obj1_x, y=obj1_y, z=stable_z(target_obj_1, floor))))
        set_pose(target_obj_2, Pose(Point(x=obj2_x, y=obj2_y, z=stable_z(target_obj_2, floor))))
        set_pose(stirrer, Pose(Point(x=stirrer_x, y=stirrer_y, z=stirrer_z)))

        bodies = [target_obj_1, target_obj_2, stirrer]
        if any(pairwise_collision(b1, b2)
               for i, b1 in enumerate(bodies)
               for b2 in bodies[i + 1:]):
            continue

        return {
            'target_obj_1_x': obj1_x,
            'target_obj_1_y': obj1_y,
            'target_obj_2_x': obj2_x,
            'target_obj_2_y': obj2_y,
            'stirrer_x': stirrer_x,
            'stirrer_y': stirrer_y,
        }
    raise RuntimeError('Failed to sample a collision-free layout within the maximum attempts.')


def compute_goal_params(target_obj_1, target_obj_2, stirrer):
    obj1_initial_pose = get_pose(target_obj_1)
    (sx, sy, _), _ = get_pose(stirrer)
    obj1_goal_z = stable_z(target_obj_1, stirrer)
    obj1_goal_pose = Pose(Point(x=sx, y=sy, z=obj1_goal_z))

    set_pose(target_obj_1, obj1_goal_pose)
    obj2_goal_z = stable_z(target_obj_2, target_obj_1)
    set_pose(target_obj_1, obj1_initial_pose)
    _, obj2_goal_quat = get_pose(target_obj_2)
    return obj1_goal_pose, obj2_goal_z, obj2_goal_quat


def run_single_trial(trial_idx, max_time, max_iterations):
    result = {
        'trial': trial_idx,
        'success': 0,
        'planning_time_sec': 0.0,
        'target_obj_1_x': float('nan'),
        'target_obj_1_y': float('nan'),
        'target_obj_2_x': float('nan'),
        'target_obj_2_y': float('nan'),
        'stirrer_x': float('nan'),
        'stirrer_y': float('nan'),
    }

    connect(use_gui=False)
    try:
        configure_stir_motion_constraints()
        robot, floor, target_obj_1, target_obj_2, stirrer = load_world()
        layout = sample_layout(floor, target_obj_1, target_obj_2, stirrer)
        result.update(layout)

        obj1_goal_pose, obj2_goal_z, obj2_goal_quat = compute_goal_params(target_obj_1, target_obj_2, stirrer)
        problem = pddlstream_from_problem(
            robot,
            target_obj_1=target_obj_1,
            target_obj_2=target_obj_2,
            stirrer=stirrer,
            obj1_goal_pose=obj1_goal_pose,
            obj2_goal_z=obj2_goal_z,
            obj2_goal_quat=obj2_goal_quat,
            movable=[target_obj_1, target_obj_2],
            teleport=False,
            grasp_name='top',
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
        args.csv_path = os.path.join(DEFAULT_CSV_DIR, 'stir_fr5.csv')
    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)

    fieldnames = [
        'trial',
        'success',
        'planning_time_sec',
        'target_obj_1_x',
        'target_obj_1_y',
        'target_obj_2_x',
        'target_obj_2_y',
        'stirrer_x',
        'stirrer_y',
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
            print('[{}/{}][fr5-stir] success={} time={:.3f}s'.format(
                trial_idx, args.trials, result['success'], result['planning_time_sec']))

    print('Saved:', args.csv_path)


if __name__ == '__main__':
    main()
