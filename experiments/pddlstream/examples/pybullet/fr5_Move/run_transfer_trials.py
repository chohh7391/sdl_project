#!/usr/bin/env python

from __future__ import print_function

import argparse
import csv
import os
import random
import time

from pddlstream.algorithms.meta import solve
from pddlstream.utils import INF

from examples.pybullet.fr5.transfer import pddlstream_from_problem as fr5_pddlstream_from_problem, FR5_HOME_ARM
from examples.pybullet.fr5.panda_transfer import pddlstream_from_problem as panda_pddlstream_from_problem
from examples.pybullet.utils.pybullet_tools.utils import connect, disconnect, load_model, set_pose, get_pose, Pose, Point, \
    stable_z, HideOutput, FR5_AG95_URDF, BLOCK_URDF, SMALL_BLOCK_URDF, SINK_URDF, \
    set_joint_positions, get_movable_joints, pairwise_collision, add_data_path, load_pybullet

DEFAULT_TRIALS = 30
DEFAULT_MAX_TIME = 120.0
DEFAULT_MAX_ITERATIONS = 1024
DEFAULT_CSV_DIR = '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5/data'

X_RANGE = (0.2, 0.55)
Y_RANGE = (-0.55, 0.55)
PANDA_HOME_ARM = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
PANDA_FIXED_GOAL_XY = (0.35, -0.35)


def sample_xy():
    return random.uniform(*X_RANGE), random.uniform(*Y_RANGE)


def load_world(robot_name):
    with HideOutput():
        if robot_name == 'fr5':
            robot = load_model(FR5_AG95_URDF, fixed_base=True)
            set_pose(robot, Pose(Point(z=0.05)))
            set_joint_positions(robot, get_movable_joints(robot)[:6], FR5_HOME_ARM)
        elif robot_name == 'panda':
            add_data_path()
            robot = load_pybullet('franka_panda/panda.urdf', fixed_base=True)
            set_pose(robot, Pose(Point(z=0.07)))
        else:
            raise ValueError('Unsupported robot: {}'.format(robot_name))

        floor = load_model('models/short_floor.urdf')
        source_obj = load_model(BLOCK_URDF, fixed_base=False)
        pour_target = load_model(SMALL_BLOCK_URDF, fixed_base=True)
        goal_surface = load_model(SINK_URDF, fixed_base=True)
        obstacle = None
        if robot_name != 'panda':
            obstacle = load_model(BLOCK_URDF, fixed_base=True)

    if obstacle is not None:
        set_pose(obstacle, Pose(Point(x=0.25, y=0.25, z=stable_z(obstacle, floor))))
    return robot, floor, source_obj, pour_target, goal_surface, obstacle


def sample_layout(floor, source_obj, pour_target, goal_surface, obstacle=None, random_goal=True, max_attempts=200):
    goal_z = get_pose(goal_surface)[0][2]
    gx, gy = PANDA_FIXED_GOAL_XY
    for _ in range(max_attempts):
        sx, sy = sample_xy()
        px, py = sample_xy()
        if random_goal:
            gx, gy = sample_xy()

        set_pose(source_obj, Pose(Point(x=sx, y=sy, z=stable_z(source_obj, floor))))
        set_pose(pour_target, Pose(Point(x=px, y=py, z=stable_z(pour_target, floor))))
        set_pose(goal_surface, Pose(Point(x=gx, y=gy, z=goal_z)))

        bodies = [source_obj, pour_target, goal_surface]
        if obstacle is not None:
            bodies.append(obstacle)
        if any(pairwise_collision(b1, b2)
               for i, b1 in enumerate(bodies)
               for b2 in bodies[i + 1:]):
            continue

        return {
            'source_obj_x': sx,
            'source_obj_y': sy,
            'pour_target_x': px,
            'pour_target_y': py,
            'goal_surface_x': gx,
            'goal_surface_y': gy,
        }
    raise RuntimeError('Failed to sample a collision-free layout within the maximum attempts.')


def run_single_trial(trial_idx, max_time, max_iterations, robot_name):
    result = {
        'trial': trial_idx,
        'success': 0,
        'planning_time_sec': 0.0,
        'source_obj_x': float('nan'),
        'source_obj_y': float('nan'),
        'pour_target_x': float('nan'),
        'pour_target_y': float('nan'),
        'goal_surface_x': float('nan'),
        'goal_surface_y': float('nan'),
    }
    connect(use_gui=False)
    try:
        robot, floor, source_obj, pour_target, goal_surface, obstacle = load_world(robot_name)
        layout = sample_layout(
            floor, source_obj, pour_target, goal_surface, obstacle,
            random_goal=(robot_name != 'panda'))
        result.update(layout)

        problem_fn = fr5_pddlstream_from_problem if robot_name == 'fr5' else panda_pddlstream_from_problem
        problem_kwargs = dict(
            robot=robot,
            movable=[source_obj],
            pour_target=pour_target,
            goal_surface=goal_surface,
            stackable_surfaces=[goal_surface],
            teleport=False,
        )
        if robot_name == 'fr5':
            problem_kwargs['sample_pour_poses'] = True
        problem = problem_fn(**problem_kwargs)

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
    parser.add_argument('--robot', choices=['fr5', 'panda'], default='fr5')
    args = parser.parse_args()

    if args.csv_path is None:
        args.csv_path = os.path.join(DEFAULT_CSV_DIR, 'transfer_{}_sample_z.csv'.format(args.robot))

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)

    fieldnames = [
        'trial',
        'success',
        'planning_time_sec',
        'source_obj_x',
        'source_obj_y',
        'pour_target_x',
        'pour_target_y',
        'goal_surface_x',
        'goal_surface_y',
    ]

    with open(args.csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trial_idx in range(1, args.trials + 1):
            result = run_single_trial(
                trial_idx=trial_idx,
                max_time=args.max_time,
                max_iterations=args.max_iterations,
                robot_name=args.robot,
            )
            writer.writerow(result)
            f.flush()
            print('[{}/{}][{}] success={} time={:.3f}s'.format(
                trial_idx, args.trials, args.robot, result['success'], result['planning_time_sec']))

    print('Saved:', args.csv_path)


if __name__ == '__main__':
    main()
