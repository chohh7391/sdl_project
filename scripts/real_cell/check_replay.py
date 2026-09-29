#!/usr/bin/env python3
"""Build the FR5 trajectory_replay tree for a fetched trajectory, without moving.

    check_replay.py BASE [--grams G --material liquid|granular [--flow-index F]
                          --receiver X,Y,Z [--grasp-marker X,Y,Z]] [--home-via moveit|direct]

BASE is a deliver/ file stem (BASE.csv, BASE.meta.json, BASE_cell.yaml). The
tree is BUILT and never ticked, so nothing is sent to the robot, but everything
cho_task_manager refuses at build time is refused here too: a cell layout that
does not match the recording, a pour that cannot be spliced out and handed to
the pouring controller, joint velocities past the arm's limits (the clock is
stretched, and the factor is printed). Needs ROS 2 and the cho workspace
sourced, not a running graph.
"""
import argparse
import sys

from cho_task_manager.tasks import build_task_tree
from cho_task_manager.utils.controller_names import load_robot_config


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("--grams", type=float, default=0.0)
    ap.add_argument("--material", default="liquid")
    ap.add_argument("--flow-index", type=float, default=0.0)
    ap.add_argument("--receiver", default="",
                    help="the receiver's rim centre x,y,z (base_link, m); required with a pour")
    ap.add_argument("--grasp-marker", default="",
                    help="beaker1's perceived bottom centre x,y,z, the grasp's marker")
    ap.add_argument("--home-via", default="moveit")
    a = ap.parse_args()

    config = load_robot_config("fr5")
    config.update({"replay_trajectory": a.base + ".csv",
                   "replay_meta": a.base + ".meta.json",
                   "replay_layout": a.base + "_cell.yaml",
                   "home_via": a.home_via})
    if a.grams > 0:
        config.update({"replay_pour_grams": a.grams, "replay_pour_material": a.material,
                       "replay_pour_flow_index": a.flow_index,
                       "replay_pour_receiver": a.receiver,
                       "replay_pour_grasp_marker": a.grasp_marker})
    try:
        tree = build_task_tree("trajectory_replay", config)
    except Exception as err:   # RecordingRejected, LayoutMismatch, ValueError
        print("replay REFUSED: %s: %s" % (type(err).__name__, err), file=sys.stderr)
        return 1
    summary = getattr(tree, "replay_summary", {}) or {}
    print("replay tree builds: home via %s, clock x%.2f"
          % (summary.get("home_via", a.home_via), float(summary.get("time_scale", 1.0))))
    for seg in summary.get("segments", []):
        print("   %s" % seg)
    pour = summary.get("pour")
    if pour:
        marker = ("the perceived (%.4f, %.4f, %.4f)" % tuple(pour["grasp_marker"])
                  if pour.get("grasp_marker") else pour["marker_topic"])
        print("   pour: %g g of %s by weight, grasp marker %s, into the receiver rim at "
              "(%.4f, %.4f, %.4f)" % (pour["target_grams"], pour["material"], marker,
                                      *pour["receiver_rim"]))
    else:
        print("   pour: replayed as recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
