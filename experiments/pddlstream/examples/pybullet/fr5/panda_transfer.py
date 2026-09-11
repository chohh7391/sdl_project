#!/usr/bin/env python

from __future__ import print_function

import time

from pddlstream.algorithms.meta import solve, create_parser
from examples.pybullet.utils.pybullet_tools.panda_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
    get_stable_gen, get_ik_fn, get_free_motion_gen, \
    get_holding_motion_gen, get_movable_collision_test, get_tool_link
from examples.pybullet.utils.pybullet_tools.utils import WorldSaver, connect, dump_body, get_pose, set_pose, Pose, \
    Point, set_default_camera, stable_z, \
    BLOCK_URDF, SMALL_BLOCK_URDF, get_configuration, SINK_URDF, load_model, is_placement, get_body_name, \
    disconnect, DRAKE_IIWA_URDF, get_bodies, HideOutput, wait_for_user, KUKA_IIWA_URDF, add_data_path, load_pybullet, \
    LockRenderer, has_gui, draw_pose, draw_global_system, FR5_AG95_URDF,TOOL_BLOCK_URDF, set_color, \
    get_movable_joints, set_joint_positions
from pddlstream.language.generator import from_gen_fn, from_fn, empty_gen, from_test, universe_test
from pddlstream.utils import read, INF, get_file_path, find_unique, Profiler, str_from_object, negate_test
from pddlstream.language.constants import print_solution, PDDLProblem
from examples.pybullet.tamp.streams import get_cfree_approach_pose_test, get_cfree_pose_pose_test, get_cfree_traj_pose_test, \
    move_cost_fn, get_cfree_obj_approach_pose_test

TIMED_STREAMS = (
    'inverse-kinematics',
    'plan-free-motion',
    'plan-holding-motion',
)
POUR_Z_OFFSET = 0.22
FR5_HOME_ARM = (0.0, -1.05, -2.18, -1.57, 1.57, 0.0)

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

def get_fixed(robot, movable):
    rigid = [body for body in get_bodies() if body != robot]
    fixed = [body for body in rigid if body not in movable]
    return fixed

def place_movable(certified):
    placed = []
    for literal in certified:
        if literal[0] == 'not':
            fact = literal[1]
            if fact[0] == 'trajcollision':
                _, b, p = fact[1:]
                set_pose(b, p.pose)
                placed.append(b)
    return placed

def get_free_motion_synth(robot, movable=[], teleport=False):
    fixed = get_fixed(robot, movable)
    def fn(outputs, certified):
        assert(len(outputs) == 1)
        q0, _, q1 = find_unique(lambda f: f[0] == 'freemotion', certified)[1:]
        obstacles = fixed + place_movable(certified)
        free_motion_fn = get_free_motion_gen(robot, obstacles, teleport)
        return free_motion_fn(q0, q1)
    return fn

def get_holding_motion_synth(robot, movable=[], teleport=False):
    fixed = get_fixed(robot, movable)
    def fn(outputs, certified):
        assert(len(outputs) == 1)
        q0, _, q1, o, g = find_unique(lambda f: f[0] == 'holdingmotion', certified)[1:]
        obstacles = fixed + place_movable(certified)
        holding_motion_fn = get_holding_motion_gen(robot, obstacles, teleport)
        return holding_motion_fn(q0, q1, o, g)
    return fn

#######################################################

def get_pour_pose_gen(z_offsets=(0.16, 0.19, POUR_Z_OFFSET, 0.25, 0.28, 0.31),
                      xy_offsets=((0.00, 0.00),
                                  (0.03, 0.00), (-0.03, 0.00),
                                  (0.05, 0.00), (-0.05, 0.00),
                                  (0.00, 0.03), (0.00, -0.03),
                                  (0.00, 0.05), (0.00, -0.05),
                                  (0.02, 0.02), (0.02, -0.02),
                                  (-0.02, 0.02), (-0.02, -0.02),
                                  (0.04, 0.04), (0.04, -0.04),
                                  (-0.04, 0.04), (-0.04, -0.04))):
    def gen(body, target):
        (tx, ty, tz), _ = get_pose(target)
        for dz in z_offsets:
            for dx, dy in xy_offsets:
                pour_pose = BodyPose(body, Pose(Point(x=tx + dx, y=ty + dy, z=tz + dz)))
                yield (pour_pose,)
    return gen

def pddlstream_from_problem(robot, movable=None, pour_target=None, goal_surface=None,
                            stackable_surfaces=None, teleport=False, grasp_name='side', timing_stats=None):
    #assert (not are_colliding(tree, kin_cache))
    if movable is None:
        movable = []
    if (pour_target is None) or (goal_surface is None):
        raise ValueError('pour_target and goal_surface must be specified.')
    if timing_stats is None:
        timing_stats = create_timing_stats()

    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    constant_map = {}

    print('Robot:', robot)
    conf = BodyConf(robot, get_configuration(robot))
    init = [('CanMove',),
            ('Conf', conf),
            ('AtConf', conf),
            ('HandEmpty',),
            ('PourTarget', pour_target)]

    fixed = get_fixed(robot, movable)
    print('Movable:', movable)
    print('Fixed:', fixed)
    if stackable_surfaces is None:
        stackable_surfaces = fixed
    stackable_surfaces = set(stackable_surfaces)
    for body in movable:
        pose = BodyPose(body, get_pose(body))
        init += [('Graspable', body),
                 ('Pose', body, pose),
                 ('Stable', body, pose),
                 ('AtPose', body, pose)]
        for surface in fixed:
            if surface in stackable_surfaces:
                init += [('Stackable', body, surface)]
            if is_placement(body, surface):
                init += [('Supported', body, pose, surface)]

    body = movable[0]
    goal = ('and',
            ('HandEmpty',),
            ('Poured', body),
            ('On', body, goal_surface),
    )

    stream_map = {
        'sample-pose': from_gen_fn(get_stable_gen(fixed)),
        'sample-grasp': from_gen_fn(get_grasp_gen(robot, grasp_name)),
        'sample-pour-pose': from_gen_fn(get_pour_pose_gen()),
        'inverse-kinematics': from_fn(wrap_timed_fn('inverse-kinematics',
                                                    get_ik_fn(robot, fixed, teleport), timing_stats)),
        'plan-free-motion': from_fn(wrap_timed_fn('plan-free-motion',
                                                  get_free_motion_gen(robot, fixed, teleport), timing_stats)),
        'plan-holding-motion': from_fn(wrap_timed_fn('plan-holding-motion',
                                                     get_holding_motion_gen(robot, fixed, teleport), timing_stats)),

        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test()),
        'test-cfree-approach-pose': from_test(get_cfree_obj_approach_pose_test()),
        'test-cfree-traj-pose': from_test(negate_test(get_movable_collision_test())), #get_cfree_traj_pose_test()),

        'TrajCollision': get_movable_collision_test(),
    }

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


#######################################################

def load_world():
    # TODO: store internal world info here to be reloaded
    set_default_camera()
    draw_global_system()
    with HideOutput():
        #add_data_path()
        robot = load_model(FR5_AG95_URDF, fixed_base=True) # DRAKE_IIWA_URDF | KUKA_IIWA_URDF
        set_pose(robot, Pose(Point(z=0.05)))
        set_joint_positions(robot, get_movable_joints(robot)[:6], FR5_HOME_ARM)
        floor = load_model('models/short_floor.urdf')
        source_obj = load_model(BLOCK_URDF, fixed_base=False)
        pour_target = load_model(SMALL_BLOCK_URDF, fixed_base=True)
        obstacle = load_model(TOOL_BLOCK_URDF, fixed_base=True)
        goal_surface = load_model(SINK_URDF, pose=Pose(Point(x=+0.55, y=-0.25)))

    draw_pose(Pose(), parent=robot, parent_link=get_tool_link(robot)) # TODO: not working
    # dump_body(robot)
    # wait_for_user()

    body_names = {
        source_obj: 'source_obj',
        pour_target: 'pour_target',
        goal_surface: 'goal_surface',
        obstacle: 'obstacle',
    }
    movable_bodies = [source_obj]

    set_pose(source_obj, Pose(Point(x=0.2, y=0.55, z=stable_z(source_obj, floor))))
    set_pose(pour_target, Pose(Point(x=-0.45, y=0.15, z=stable_z(pour_target, floor))))
    # set_pose(source_obj, Pose(Point(x=0.51, y=-0.17, z=stable_z(source_obj, floor))))
    # set_pose(pour_target, Pose(Point(x=0.22, y=0.2, z=stable_z(pour_target, floor))))
    set_pose(goal_surface, Pose(Point(x=0.35, y=-0.35, z=stable_z(goal_surface, floor))))
    set_pose(obstacle, Pose(Point(x=-0.8, y=0.0, z=stable_z(obstacle, floor))))
    set_color(obstacle, (0.95, 0.35, 0.1, 1.0))

    return robot, body_names, movable_bodies, pour_target, goal_surface

def postprocess_plan(plan):
    paths = []
    for name, args in plan:
        if name == 'place':
            paths += args[-1].reverse().body_paths
        elif name in ['move', 'move_free', 'move_holding', 'pick']:
            paths += args[-1].body_paths
    return Command(paths)

#######################################################

def main():
    parser = create_parser()
    parser.add_argument('-enable', action='store_true', help='Enables rendering during planning')
    parser.add_argument('-teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('-simulate', action='store_true', help='Simulates the system')
    args = parser.parse_args()
    print('Arguments:', args)

    connect(use_gui=True)
    robot, names, movable, pour_target, goal_surface = load_world()
    print('Objects:', names)
    saver = WorldSaver()

    ####################################################
    # wait_for_user('환경이 로드되었습니다. 확인 후 터미널에서 Enter를 누르면 종료됩니다.')
    # disconnect()
    # return  
    ####################################################


    timing_stats = create_timing_stats()
    problem = pddlstream_from_problem(
        robot, movable=movable, pour_target=pour_target, goal_surface=goal_surface,
        teleport=args.teleport, timing_stats=timing_stats,
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
    plan, cost, evaluations = solution
    if (plan is None) or not has_gui():
        disconnect()
        return

    command = postprocess_plan(plan)
    if args.simulate:
        wait_for_user('Simulate?')
        command.control()
    else:
        wait_for_user('Execute?')
        #command.step()
        command.refine(num_steps=10).execute(time_step=0.001)
    wait_for_user('Finish?')
    disconnect()

if __name__ == '__main__':
    main()
