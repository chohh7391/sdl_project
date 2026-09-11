#!/usr/bin/env python

from __future__ import print_function

import time

from pddlstream.algorithms.meta import solve, create_parser
from pddlstream.utils import INF, Profiler, str_from_object
from pddlstream.language.constants import print_solution

from examples.pybullet.fr5.panda_transfer import (
    pddlstream_from_problem,
    create_timing_stats,
    print_timing_breakdown,
    postprocess_plan,
)
from examples.pybullet.utils.pybullet_tools.panda_primitives import get_tool_link
from examples.pybullet.utils.pybullet_tools.utils import (
    WorldSaver,
    connect,
    disconnect,
    get_pose,
    set_pose,
    Pose,
    Point,
    set_default_camera,
    stable_z,
    BLOCK_URDF,
    SMALL_BLOCK_URDF,
    SINK_URDF,
    load_model,
    HideOutput,
    wait_for_user,
    add_data_path,
    load_pybullet,
    LockRenderer,
    has_gui,
    draw_pose,
    draw_global_system,
    TOOL_BLOCK_URDF,
    set_color,
    get_movable_joints,
    set_joint_positions,
)

PANDA_HOME_ARM = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)


def load_world():
    set_default_camera()
    draw_global_system()
    with HideOutput():
        add_data_path()
        robot = load_pybullet('franka_panda/panda.urdf', fixed_base=True)
        set_pose(robot, Pose(Point(z=0.07)))
        movable_joints = get_movable_joints(robot)
        # set_joint_positions(robot, movable_joints[:7], PANDA_HOME_ARM)
        # if len(movable_joints) >= 9:
        #     set_joint_positions(robot, movable_joints[7:9], [0.04, 0.04])

        floor = load_model('models/short_floor.urdf')
        source_obj = load_model(BLOCK_URDF, fixed_base=False)
        pour_target = load_model(SMALL_BLOCK_URDF, fixed_base=True)
        # obstacle = load_model(TOOL_BLOCK_URDF, fixed_base=True)
        goal_surface = load_model(SINK_URDF, pose=Pose(Point(x=+0.55, y=-0.25)))

    draw_pose(Pose(), parent=robot, parent_link=get_tool_link(robot))

    body_names = {
        source_obj: 'source_obj',
        pour_target: 'pour_target',
        goal_surface: 'goal_surface',
        # obstacle: 'obstacle',
    }
    movable_bodies = [source_obj]

    set_pose(source_obj, Pose(Point(x=0.51, y=-0.17, z=stable_z(source_obj, floor))))
    set_pose(pour_target, Pose(Point(x=0.23, y=0.4, z=stable_z(pour_target, floor))))
    set_pose(goal_surface, Pose(Point(x=0.35, y=-0.35, z=stable_z(goal_surface, floor))))
    # set_pose(obstacle, Pose(Point(x=-0.8, y=0.0, z=stable_z(obstacle, floor))))
    # set_color(obstacle, (0.95, 0.35, 0.1, 1.0))

    return robot, body_names, movable_bodies, pour_target, goal_surface


def main():
    parser = create_parser()
    parser.set_defaults(unit=True)
    parser.add_argument('-enable', action='store_true', help='Enables rendering during planning')
    parser.add_argument('-teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('-simulate', action='store_true', help='Simulates the system')
    args = parser.parse_args()
    print('Arguments:', args)

    connect(use_gui=True)
    robot, names, movable, pour_target, goal_surface = load_world()
    print('Objects:', names)


    # wait_for_user('환경이 로드되었습니다. 확인 후 터미널에서 Enter를 누르면 종료됩니다.')
    # disconnect()
    # return 

    saver = WorldSaver()

    timing_stats = create_timing_stats()
    problem = pddlstream_from_problem(
        robot,
        movable=movable,
        pour_target=pour_target,
        goal_surface=goal_surface,
        teleport=args.teleport,
        timing_stats=timing_stats,
        grasp_name='side',
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
