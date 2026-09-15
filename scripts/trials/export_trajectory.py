#!/usr/bin/env python3
"""Turn a recorded trial into a replayable waypoint file.

Writes a CSV of time-stamped arm waypoints plus the gripper events recovered
from the achieved joint states, and a header carrying the layout the trajectory
was planned against. The trajectory is position-controlled and senses nothing,
so it is only valid for that layout.
"""
import argparse, csv, json, sys

GRIPPER_JOINT = "gripper_finger1_joint"   # the driven finger; the rest mimic it


def gripper_events(states, threshold=0.05):
    """(t, 'close'|'open') from the driven finger crossing a threshold."""
    ev, prev = [], None
    for s in states:
        if GRIPPER_JOINT not in s["names"]:
            continue
        v = s["positions"][s["names"].index(GRIPPER_JOINT)]
        closed = v > threshold
        if prev is None:
            prev = closed
            continue
        if closed != prev:
            ev.append({"t": s["t"], "event": "close" if closed else "open",
                       "joint_value": round(v, 4)})
            prev = closed
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-meta", required=True)
    a = ap.parse_args()

    d = json.load(open(a.rec))
    cmds = d["commands"]
    if not cmds:
        print("no arm commands in %s" % a.rec)
        return 1
    t0 = cmds[0]["t"]
    names = cmds[0]["names"]
    ops = d.get("operations", [])
    ev = gripper_events(d.get("states_subsampled", []))

    def op_at(t):
        cur = ""
        for o in ops:
            if o["t"] - t0 <= t:
                cur = o["op"]
        return cur

    with open(a.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s"] + list(names) + ["operation"])
        for c in cmds:
            t = round(c["t"] - t0, 4)
            w.writerow([t] + [round(v, 6) for v in c["positions"]] + [op_at(t)])

    meta = {
        "seed": d.get("seed"), "tool": d.get("tool"),
        "recorded_utc": d.get("recorded_utc"),
        "joint_names": names,
        "waypoints": len(cmds),
        "duration_s": round(cmds[-1]["t"] - t0, 3),
        "home_arm_rad": d.get("home_arm_rad"),
        "layout_the_trajectory_assumes": d.get("layout"),
        "gripper_events": [{"t_s": round(e["t"] - t0, 3), "event": e["event"]}
                           for e in ev],
        "operations": [{"t_s": round(o["t"] - t0, 3), "op": o["op"]}
                       for o in ops if o["t"] - t0 >= 0],
        "replay_notes": [
            "Positions are absolute joint angles in radians for j1..j6, "
            "sampled at the rate the planner published them; t_s is seconds "
            "from the first command.",
            "The arm is position-controlled and senses nothing during replay: "
            "the vessels must be at the layout above or the pour misses.",
            "Gripper events are recovered from the simulated finger joint, not "
            "from the gripper command channel, which is a service rather than a "
            "topic; treat the times as approximate and drive the real gripper "
            "from them rather than replaying finger angles.",
        ],
    }
    json.dump(meta, open(a.out_meta, "w"), indent=1)
    print("wrote %s (%d waypoints, %.1f s) and %s"
          % (a.out_csv, len(cmds), meta["duration_s"], a.out_meta))
    print("gripper events:", [(e["t_s"], e["event"]) for e in meta["gripper_events"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
