#!/usr/bin/env python3
"""PDDLStream baseline on the SAME seeded layouts cuTAMP was measured on.

The stock runner (fr5_obstacle/run_transfer_trials.py) samples its own layouts
from its own ranges, so pairing its rows with cuTAMP's would not be pairing at
all and an exact McNemar test on them would be meaningless. This runner instead
reads the layouts exported from Task._randomize_layout (verified against the
simulator's own logs by analysis/export_layouts.py) and reproduces, as far as
PyBullet allows, the scene cuTAMP planned in:

  same object layout   the seeded (x, y, yaw) of beaker, flask, magnet, box
                       and stirrer, plus the seeded six-joint arm home
  same robot           fr5_ag95, the tool the Transfer task is evaluated with,
                       from the same URDF, its base bottom on the table top to
                       match where the Isaac robot stands (see ROBOT_BASE_Z)
  same geometry        boxes with cuTAMP's own extents (see make_models.py),
                       not the stock 0.2 m blocks
  same obstacles       magnet, box and stirrer present as fixed bodies, which
                       get_fixed() then hands the planner
  same goal            HandEmpty and Poured(beaker) and On(beaker,
                       goal_region), the goal_region a 0.1 m square at
                       (0.35, -0.35) -- structurally cuTAMP's transfer goal
  same budget          --max-time, to be set to cuTAMP's planning budget
  same restarts        a failed solve() is restarted with fresh samples
                       until the budget is spent, as cuTAMP's harness restarts
                       a failed round (see RESTART_STREAM_STRIDE)

What still differs is inherent to the baseline and belongs in the write-up:
PyBullet against Isaac Sim for collision and stability, and PDDLStream's
sampling against cuTAMP's batched optimization. Execution is NOT compared --
only planning success and planning time, which is what the table claims.
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PDDL_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if PDDL_ROOT not in sys.path:
    sys.path.insert(0, PDDL_ROOT)

from pddlstream.algorithms.meta import solve
from pddlstream.utils import INF

from examples.pybullet.fr5.transfer import (
    pddlstream_from_problem as transfer_problem)
from examples.pybullet.fr5_obstacle.move import (
    pddlstream_from_move_problem as move_problem)
from examples.pybullet.fr5_obstacle.stir import (
    pddlstream_from_problem as stir_problem)
from examples.pybullet.utils.pybullet_tools.utils import (
    connect, disconnect, load_pybullet, load_model, set_pose, get_pose, Pose,
    Point, Euler, stable_z, HideOutput, FR5_AG95_URDF, BLOCK_URDF,
    SMALL_BLOCK_URDF, SINK_URDF, set_joint_positions, get_movable_joints,
    save_state, restore_state)

# Per-task tool, matching what each task is evaluated with in the Isaac
# harness (content/configs/xdl/tool_map.yml, mirrored in run_trials.sh):
# Transfer the 2-finger ag95, Move the vgc10 suction cup, Stir the dh3.
URDF_DIR = os.path.dirname(FR5_AG95_URDF)
# The vgc10 and dh3 URDFs ship with package:// mesh URIs that PyBullet cannot
# resolve (only the ag95 had been converted), so *_pb.urdf copies with relative
# paths are used; every mesh they name was checked to exist and all three load.
TASK_URDF = {
    "transfer": FR5_AG95_URDF,
    "move": os.path.join(URDF_DIR, "fr5_vgc10_pb.urdf"),
    "stir": os.path.join(URDF_DIR, "fr5_dh3_pb.urdf"),
}

MODELS = os.path.join(HERE, "models")
TABLE_TOP_Z = 0.0          # cuTAMP's table: dims_z 0.02 centred at z = -0.01
# The Isaac USD robot's prim origin is the BOTTOM of its base (fr5_ag95.usd
# spans z 0.0000..0.2174), so at the origin its base rests on the table top.
# The URDF's base_link origin sits 0.0455 m above the base bottom instead, so
# base_link must go to +0.0455 for the two robots to stand at the same height;
# at 0 the base is 45.5 mm inside the table and every configuration collides,
# which is what made the fully matched scene fail on all 30 seeds.
ROBOT_BASE_Z = 0.0455
GOAL_XY = (0.35, -0.35)    # envs/transfer.py goal_region pose
# envs/move.py places the box on box_region, which it resizes to the goal
# TRAY's own 0.15 m footprint; the tray sits at G3, per task.py.
MOVE_TRAY_XY = (0.15, -0.6)
MOVE_TRAY_DIMS = (0.15, 0.15, 0.01)
# Objects present but not acted on, per task. Everything loaded fixed-base ends
# up in get_fixed(), which is how the planner is told about it.
TASK_OBSTACLES = {
    "transfer": ("magnet", "box", "stirrer"),
    "move": ("beaker", "flask", "magnet", "stirrer"),
    "stir": ("flask", "box"),
}
OBSTACLES = TASK_OBSTACLES["transfer"]
# Half-heights, to put each box's centre a half-height above the table top
HALF_Z = {"beaker": 0.0675, "flask": 0.06, "magnet": 0.0175, "box": 0.04,
          "stirrer": 0.045, "goal_region": 0.005}


def model(name):
    return os.path.join(MODELS, "%s.urdf" % name)


def place(body, xy, half_z, yaw):
    set_pose(body, Pose(Point(x=xy[0], y=xy[1], z=TABLE_TOP_Z + half_z),
                        Euler(yaw=yaw)))


NOMINAL_HOME = (0.0, -1.05, -2.18, -1.57, 1.57, 0.0)


def build_world_task(layout, task):
    """Scene for one task, every object at this seed's pose.

    The Isaac harness evaluates each task with its own tool, so the tool
    follows the task here too. Objects the task does not act on are still
    loaded, fixed-base, which is how get_fixed() hands them to the planner.
    """
    objs = layout["objects"]
    with HideOutput():
        robot = load_pybullet(TASK_URDF[task], fixed_base=True)
        set_pose(robot, Pose(Point(z=ROBOT_BASE_Z)))
        set_joint_positions(robot, get_movable_joints(robot)[:6],
                            layout["home_arm_rad"])

        table = load_pybullet(model("table"), fixed_base=True)
        set_pose(table, Pose(Point(z=TABLE_TOP_Z - 0.01)))

        acted, extra = {}, {}
        if task == "transfer":
            acted["beaker"] = load_pybullet(model("beaker"), fixed_base=False)
            goal = load_pybullet(model("goal_region"), fixed_base=True)
            place(goal, GOAL_XY, HALF_Z["goal_region"], 0.0)
            # The pour target. Loaded here, before the obstacles, so the body
            # order -- which is what get_fixed() reports and therefore what the
            # samplers iterate -- is the same on every run.
            extra["flask"] = load_pybullet(model("flask"), fixed_base=True)
            place(extra["flask"], objs["flask"]["xy"], HALF_Z["flask"],
                  objs["flask"]["yaw_rad"])
        elif task == "move":
            acted["box"] = load_pybullet(model("box"), fixed_base=False)
            goal = load_pybullet(model("box_goal"), fixed_base=True)
            place(goal, MOVE_TRAY_XY, MOVE_TRAY_DIMS[2] / 2.0, 0.0)
        elif task == "stir":
            acted["beaker"] = load_pybullet(model("beaker"), fixed_base=False)
            acted["magnet"] = load_pybullet(model("magnet"), fixed_base=False)
            goal = load_pybullet(model("stirrer"), fixed_base=True)
            place(goal, objs["stirrer"]["xy"], HALF_Z["stirrer"],
                  objs["stirrer"]["yaw_rad"])
        else:
            raise ValueError(task)

        for name, body in acted.items():
            place(body, objs[name]["xy"], HALF_Z[name], objs[name]["yaw_rad"])

        for name in TASK_OBSTACLES[task]:
            b = load_pybullet(model(name), fixed_base=True)
            place(b, objs[name]["xy"], HALF_Z[name], objs[name]["yaw_rad"])

    return {"robot": robot, "table": table, "goal": goal, "acted": acted,
            "extra": extra}


def build_problem(task, w, teleport=False):
    """The PDDLStream problem for `task`, goal-matched to cuTAMP's.

    transfer  HandEmpty and Poured(beaker) and On(beaker, goal_region)
    move      HandEmpty and On(box, tray)
    stir      HandEmpty and On(beaker, stirrer) and On(magnet, beaker)

    These are the goals envs/transfer.py, envs/move.py and envs/stir.py state,
    predicate for predicate.
    """
    if task == "transfer":
        raise AssertionError("transfer is built in run_seed, which needs the "
                             "flask as the pour target")
    if task == "move":
        box = w["acted"]["box"]
        centre, z = compute_move_goal(box, w["goal"])
        return move_problem(robot=w["robot"], target_obj_1=box,
                            goal_surface=w["goal"], obj1_goal_center=centre,
                            obj1_goal_z=z, movable=[box], teleport=teleport)
    beaker = w["acted"]["beaker"]
    magnet = w["acted"]["magnet"]
    stirrer = w["goal"]
    (sx, sy, _), _ = get_pose(stirrer)
    beaker_goal = Pose(Point(x=sx, y=sy, z=stable_z(beaker, stirrer)))
    initial = get_pose(beaker)
    set_pose(beaker, beaker_goal)
    magnet_goal_z = stable_z(magnet, beaker)
    set_pose(beaker, initial)
    _, magnet_quat = get_pose(magnet)
    return stir_problem(robot=w["robot"], target_obj_1=beaker,
                        target_obj_2=magnet, stirrer=stirrer,
                        obj1_goal_pose=beaker_goal,
                        obj2_goal_z=magnet_goal_z,
                        obj2_goal_quat=magnet_quat,
                        movable=[beaker, magnet], teleport=teleport)


def compute_move_goal(body, tray):
    (gx, gy, _), _ = get_pose(tray)
    return (gx, gy), stable_z(body, tray)


def build_world(layout, opt):
    """Load the robot and every object at this seed's pose.

    Each difference from the stock runner's scene is switchable, because
    0/30 on the fully matched scene has to be attributed to a specific change
    before any of it can be reported as a planner comparison.
    """
    objs = layout["objects"]
    with HideOutput():
        robot = load_pybullet(FR5_AG95_URDF, fixed_base=True)
        set_pose(robot, Pose(Point(z=opt["robot_z"])))
        set_joint_positions(
            robot, get_movable_joints(robot)[:6],
            layout["home_arm_rad"] if opt["seeded_home"] else NOMINAL_HOME)

        if opt["table"] == "matched":
            table = load_pybullet(model("table"), fixed_base=True)
            set_pose(table, Pose(Point(z=TABLE_TOP_Z - 0.01)))
        else:
            table = load_model("models/short_floor.urdf")

        if opt["goal"] == "matched":
            goal_region = load_pybullet(model("goal_region"), fixed_base=True)
            place(goal_region, GOAL_XY, HALF_Z["goal_region"], 0.0)
        else:
            goal_region = load_model(SINK_URDF, fixed_base=True)
            set_pose(goal_region, Pose(Point(
                x=GOAL_XY[0], y=GOAL_XY[1],
                z=stable_z(goal_region, table))))

        if opt["geometry"] == "matched":
            beaker = load_pybullet(model("beaker"), fixed_base=False)
            place(beaker, objs["beaker"]["xy"], HALF_Z["beaker"],
                  objs["beaker"]["yaw_rad"])
            flask = load_pybullet(model("flask"), fixed_base=True)
            place(flask, objs["flask"]["xy"], HALF_Z["flask"],
                  objs["flask"]["yaw_rad"])
        else:
            beaker = load_model(BLOCK_URDF, fixed_base=False)
            set_pose(beaker, Pose(Point(
                x=objs["beaker"]["xy"][0], y=objs["beaker"]["xy"][1],
                z=stable_z(beaker, table))))
            flask = load_model(SMALL_BLOCK_URDF, fixed_base=True)
            set_pose(flask, Pose(Point(
                x=objs["flask"]["xy"][0], y=objs["flask"]["xy"][1],
                z=stable_z(flask, table))))

        others = []
        if opt["obstacles"] == "matched":
            for name in OBSTACLES:
                b = load_pybullet(model(name), fixed_base=True)
                place(b, objs[name]["xy"], HALF_Z[name], objs[name]["yaw_rad"])
                others.append(b)
        elif opt["obstacles"] == "stock":
            b = load_model(BLOCK_URDF, fixed_base=True)
            set_pose(b, Pose(Point(x=0.25, y=0.25, z=stable_z(b, table))))
            others.append(b)

    return {"robot": robot, "table": table, "goal_region": goal_region,
            "beaker": beaker, "flask": flask, "obstacles": others}


# Restarts. The adaptive algorithm returns "no plan" not only at the budget but
# also early, once the samples it drew are exhausted -- in 20261001e 6 of the
# 10 Transfer failures and all 5 Stir failures ended in under 10 s of a 180 s
# budget. cuTAMP's harness restarts a failed round with fresh samples until the
# budget is spent (tamp_server, SDL_PLAN_BUDGET_S), so the baseline does the
# same: the scene is put back to the seed's initial state, the sampler gets
# the next stream, and solve() gets what is left of the budget. The first
# attempt keeps the stream a single run always had, so first_attempt_success
# is the no-restart verdict. A planner exception is not retried: it is a
# defect, not an unlucky draw, and is recorded as one.
RESTART_STREAM_STRIDE = 1000003     # prime; attempt k of (planner seed, layout)
                                    # never reuses another pair's stream


def _seed_sampler(value):
    # PDDLStream draws from the global `random`; numpy for the pose samplers
    random.seed(value)
    try:
        import numpy as _np
        _np.random.seed(value % (2 ** 32))
    except ImportError:
        pass


def _make_problem(task, layout, opt):
    """(world, a function building the PDDLStream problem from its current state)."""
    if task == "transfer" and opt.get("variant"):
        w = build_world(layout, opt)
        return w, lambda: transfer_problem(
            robot=w["robot"], movable=[w["beaker"]],
            pour_target=w["flask"], goal_surface=w["goal_region"],
            stackable_surfaces=[w["goal_region"]], teleport=False,
            sample_pour_poses=True)
    w = build_world_task(layout, task)
    if task == "transfer":
        return w, lambda: transfer_problem(
            robot=w["robot"], movable=[w["acted"]["beaker"]],
            pour_target=w["extra"]["flask"], goal_surface=w["goal"],
            stackable_surfaces=[w["goal"]], teleport=False,
            sample_pour_poses=True)
    return w, lambda: build_problem(task, w)


def run_seed(layout, max_time, max_iterations, opt):
    task = opt["task"]
    # Seed the sampler from the layout's own seed. PDDLStream draws from the
    # global `random`, so without this the baseline moves by about a seed
    # between runs and a paired McNemar on its rows is not reproducible: a
    # dedicated 60 s run solved 18/30 where the 180 s run censored at 60 s
    # solved 19/30. With it, the same seed gives the same verdict.
    _ps = opt.get("planner_seed", 0) * 100003 + layout["seed"]
    _seed_sampler(_ps)
    row = {"seed": layout["seed"], "plan_success": 0,
           "planning_time_s": float("nan"), "failure_reason": "",
           "beaker_xy": "", "flask_xy": "", "attempts": 0,
           "first_attempt_success": 0,
           "first_attempt_time_s": float("nan")}
    objs = layout["objects"]
    row["beaker_xy"] = "%.4f;%.4f" % tuple(objs["beaker"]["xy"])
    row["flask_xy"] = "%.4f;%.4f" % tuple(objs["flask"]["xy"])
    connect(use_gui=False)
    try:
        w, make = _make_problem(task, layout, opt)
        initial = save_state()
        problem = make()
        t0 = time.perf_counter()
        while True:
            if row["attempts"]:
                restore_state(initial)
                _seed_sampler(_ps + row["attempts"] * RESTART_STREAM_STRIDE)
                problem = make()
            left = max_time - (time.perf_counter() - t0)
            try:
                solution = solve(problem, algorithm="adaptive", unit_costs=True,
                                 success_cost=INF, max_time=left,
                                 max_iterations=max_iterations, verbose=False)
                row["plan_success"] = int(solution[0] is not None)
                row["failure_reason"] = ("" if row["plan_success"]
                                         else "no_plan_within_budget")
                crashed = False
            except Exception as exc:        # a planner crash is a failure, and
                row["plan_success"] = 0      # the reason is recorded, not hidden
                row["failure_reason"] = "planner_error: %s" % (
                    str(exc).replace("\n", " ")[:160],)
                crashed = True
            row["attempts"] += 1
            elapsed = time.perf_counter() - t0
            if row["attempts"] == 1:
                row["first_attempt_success"] = row["plan_success"]
                row["first_attempt_time_s"] = elapsed
            if (row["plan_success"] or crashed or not opt.get("restart", True)
                    or elapsed >= max_time):
                break
        row["planning_time_s"] = elapsed
        return row
    finally:
        disconnect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", default=os.path.join(
        PDDL_ROOT, "..", "..", "_2026__IEEE_Access", "revision", "analysis",
        "data", "seed_layouts.json"))
    ap.add_argument("--seeds", default="")
    ap.add_argument("--max-time", type=float, default=60.0,
                    help="planning budget [s]; match cuTAMP's")
    ap.add_argument("--max-iterations", type=int, default=10000)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--task", choices=("transfer", "move", "stir"),
                    default="transfer")
    ap.add_argument("--planner-seed", type=int, default=0,
                    help="repetition index; changes the sampler's stream while "
                         "keeping the layout fixed")
    ap.add_argument("--no-restart", action="store_true",
                    help="one solve() per layout, as the runner did before "
                         "restarts (20261001c/e); see RESTART_STREAM_STRIDE")
    ap.add_argument("--variant", action="store_true",
                    help="transfer only: use the switchable bisection scene "
                         "below instead of the matched one")
    ap.add_argument("--table", choices=("matched", "stock"), default="matched")
    ap.add_argument("--goal", choices=("matched", "stock"), default="matched")
    ap.add_argument("--geometry", choices=("matched", "stock"),
                    default="matched")
    ap.add_argument("--obstacles", choices=("matched", "stock", "none"),
                    default="matched")
    ap.add_argument("--robot-z", type=float, default=ROBOT_BASE_Z)
    ap.add_argument("--nominal-home", action="store_true",
                    help="use the shared home instead of the seed's home")
    a = ap.parse_args()
    opt = {"task": a.task, "variant": a.variant,
           "planner_seed": a.planner_seed, "restart": not a.no_restart,
           "table": a.table, "goal": a.goal, "geometry": a.geometry,
           "obstacles": a.obstacles, "robot_z": a.robot_z,
           "seeded_home": not a.nominal_home}

    data = json.load(open(os.path.abspath(a.layouts)))
    layouts = data["layouts"]
    if a.seeds:
        want = {int(v) for v in a.seeds.replace(",", " ").split()}
        layouts = [l for l in layouts if l["seed"] in want]
    print("pddlstream baseline: task=%s %d seeds, budget %.0f s, tool=%s, %s"
          % (a.task, len(layouts), a.max_time,
             os.path.basename(TASK_URDF[a.task]),
             "one solve per layout" if a.no_restart
             else "restarts until the budget is spent"))

    fields = ["timestamp", "seed", "task", "planner", "robot", "plan_success",
              "planning_time_s", "failure_reason", "beaker_xy", "flask_xy",
              "max_time_s", "planner_seed", "restart", "attempts",
              "first_attempt_success", "first_attempt_time_s"]
    new = not os.path.exists(a.csv) or os.path.getsize(a.csv) == 0
    if not new:
        # rows are appended; a file started with the other column set would
        # come out misaligned
        have = next(csv.reader(open(a.csv)))
        if have != fields:
            sys.exit("%s has columns %s; write to a new file" % (a.csv, have))
    os.makedirs(os.path.dirname(os.path.abspath(a.csv)) or ".", exist_ok=True)
    with open(a.csv, "a") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        if new:
            wr.writeheader()
        for layout in layouts:
            row = run_seed(layout, a.max_time, a.max_iterations, opt)
            row.update({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "task": a.task, "planner": "pddlstream",
                        "robot": os.path.basename(
                            TASK_URDF[a.task]).replace(".urdf", ""),
                        "max_time_s": a.max_time,
                        "planner_seed": a.planner_seed,
                        "restart": int(opt["restart"])})
            wr.writerow(row)
            fh.flush()
            print("seed %-3d success=%d time=%.2fs attempts=%d %s"
                  % (row["seed"], row["plan_success"], row["planning_time_s"],
                     row["attempts"], row["failure_reason"]))
    print("wrote", a.csv)


if __name__ == "__main__":
    main()
