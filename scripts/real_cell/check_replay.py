#!/usr/bin/env python3
"""Build the FR5 replay tree for a fetched trajectory, without moving, from the task_params it writes.

    check_replay.py BASE [--grams G --material liquid|granular [--flow-index F] [--pour-timeout S]
                          [--receiver X,Y,Z] [--grasp-marker X,Y,Z] [--put-back start|elsewhere]]
                         [--home-via moveit|direct] [--repeat N] [--write-params FILE]
    check_replay.py BASE --transfer --grams G --material M [--flow-index F] [...]
    check_replay.py BASE --grasp-only [...]

The first form builds sdl_trajectory_replay from the pour given here. --transfer
builds sdl_beaker_transfer instead, the two-beaker cell's task, which takes the
pour target from the plan's own BASE_cell.yaml -- the tree plan_remote.sh
launches. --grasp-only builds sdl_grasp_trial. All three are ros2_cell_ws's
(sdl_cell_tasks), run by the core's task manager.

BASE is a deliver/ file stem (BASE.csv, BASE.meta.json, BASE_cell.yaml). The
tree is BUILT and never ticked, so nothing is sent to the robot, but everything
the task refuses at build time is refused here too: a cell layout that does not
match the recording, a pour that cannot be spliced out and handed to the
pouring controller, joint velocities past the arm's limits (the clock is
stretched, and the factor is printed). With --write-params FILE the task's own
parameters are written to FILE and the tree is built from that file, read the
way the task manager node reads ``task_params:=``, so what plan_remote.sh
launches is what was checked. Needs ROS 2 and ros2_cell_ws sourced, not a
running graph.
"""
import argparse
import sys

import yaml

from cho_task_manager.tasks import build_task_tree
from cho_task_manager.utils.controller_names import load_robot_config
from cho_task_manager.utils.task_params import load_task_params


def point(text, what):
    """``x,y,z`` as three floats, for a task_params list; None for an empty *text*."""
    if not text:
        return None
    values = [float(v) for v in text.replace(",", " ").split()]
    if len(values) != 3:
        raise SystemExit("%s must be x,y,z: %r" % (what, text))
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("--grams", type=float, default=0.0)
    ap.add_argument("--material", default="liquid")
    ap.add_argument("--flow-index", type=float, default=0.0)
    ap.add_argument("--pour-timeout", type=float, default=0.0,
                    help="the pour goal's timeout [s]; 0 leaves it to the controller")
    ap.add_argument("--receiver", default="",
                    help="the receiver's rim centre x,y,z (base_link, m)")
    ap.add_argument("--grasp-marker", default="",
                    help="beaker1's perceived bottom centre x,y,z, the grasp's marker")
    ap.add_argument("--put-back", default="elsewhere",
                    help="where the poured vessel is set down: start or elsewhere")
    ap.add_argument("--home-via", default="moveit")
    ap.add_argument("--transfer", action="store_true",
                    help="build sdl_beaker_transfer: the pour target from BASE_cell.yaml")
    ap.add_argument("--grasp-only", action="store_true",
                    help="build sdl_grasp_trial: the plan's grasp of beaker1 and nothing else")
    ap.add_argument("--repeat", type=int, default=1,
                    help="pours made at the pour start, one trial after another")
    ap.add_argument("--write-params", default="",
                    help="write the task's task_params here and build the tree from that file")
    a = ap.parse_args()

    config = load_robot_config("fr5")
    config["home_via"] = a.home_via
    pour = {}
    if a.grams > 0:
        pour = {"replay_pour_grams": a.grams, "replay_pour_material": a.material,
                "replay_pour_flow_index": a.flow_index}
        if a.pour_timeout > 0:
            pour["replay_pour_timeout"] = a.pour_timeout
        if a.repeat != 1:
            pour["replay_pour_repeat"] = a.repeat
    if a.grasp_only:
        task = "sdl_grasp_trial"
        params = {"transfer_plan": a.base}
    elif a.transfer:
        task = "sdl_beaker_transfer"
        params = {"transfer_plan": a.base, **pour}
    else:
        task = "sdl_trajectory_replay"
        config.update({"replay_trajectory": a.base + ".csv",
                       "replay_meta": a.base + ".meta.json",
                       "replay_layout": a.base + "_cell.yaml"})
        params = dict(pour)
        if pour:
            params.update({"replay_pour_receiver": point(a.receiver, "--receiver"),
                           "replay_pour_grasp_marker": point(a.grasp_marker, "--grasp-marker"),
                           "replay_pour_put_back": a.put_back})
    if a.write_params:
        with open(a.write_params, "w", encoding="utf-8") as stream:
            stream.write("# task_params of %s for %s (scripts/real_cell/check_replay.py)\n"
                         % (task, a.base))
            yaml.safe_dump(params, stream, sort_keys=False)
        params = load_task_params(a.write_params)
    config["task_params"] = params
    try:
        tree = build_task_tree(task, config)
    except Exception as err:   # RecordingRejected, LayoutMismatch, ValueError
        print("replay REFUSED: %s: %s" % (type(err).__name__, err), file=sys.stderr)
        return 1
    if a.grasp_only:
        g = tree.grasp_summary
        print("%s tree builds: home via %s, clock x%.2f, reach %.1f s recorded"
              % (task, g["home_via"], g["time_scale"], g["reach_s"]))
        return 0
    summary = getattr(tree, "replay_summary", {}) or {}
    print("%s tree builds: home via %s, clock x%.2f%s"
          % (task, summary.get("home_via", a.home_via), float(summary.get("time_scale", 1.0)),
             ", %d pours" % summary["pour_repeat"] if summary.get("pour_repeat", 1) > 1 else ""))
    for seg in summary.get("segments", []):
        print("   %s" % seg)
    pour = summary.get("pour")
    if pour:
        marker = ("the perceived (%.4f, %.4f, %.4f)" % tuple(pour["grasp_marker"])
                  if pour.get("grasp_marker") else pour["marker_topic"])
        rim = ("(%.4f, %.4f, %.4f)" % tuple(pour["receiver_rim"])
               if pour.get("receiver_rim") else "none given")
        print("   pour: %g g of %s by weight, grasp marker %s, into the receiver rim at %s"
              % (pour["target_grams"], pour["material"], marker, rim))
    else:
        print("   pour: replayed as recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
