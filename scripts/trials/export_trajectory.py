#!/usr/bin/env python3
"""Turn a recorded trial into a replayable waypoint file.

Writes a CSV of time-stamped arm waypoints plus the gripper events recovered
from the achieved joint states, and a header carrying the layout the trajectory
was planned against. The trajectory is position-controlled and senses nothing,
so it is only valid for that layout.
"""
import argparse, csv, datetime, glob, json, os, sys

# Reuse the recorder's own parser so the two cannot disagree about what a
# layout line looks like.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout_log import read_layout  # noqa: E402

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


def _recorded_epoch(rec):
    """When the recording was written, as a unix time, or None."""
    stamp = rec.get("recorded_utc")
    if not stamp:
        return None
    try:
        return datetime.datetime.strptime(
            stamp, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=datetime.timezone.utc).timestamp()
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--sim-log", default=None,
                    help="the trial's sim log, to recover the layout the "
                         "trajectory assumes. Use this when the recorder was "
                         "run without --sim-log: without a layout the file "
                         "cannot be replayed safely, because a position-"
                         "controlled arm that senses nothing will pour wherever "
                         "the vessels are NOT.")
    ap.add_argument("--sim-log-dir", default=None,
                    help="directory to find the newest sim_seed<SEED>_*.log in, "
                         "as an alternative to naming the file")
    a = ap.parse_args()

    d = json.load(open(a.rec))

    # Recover the layout if the recording lacks one.
    if not d.get("layout"):
        log = a.sim_log
        if log is None and a.sim_log_dir and d.get("seed") not in (None, ""):
            cand = glob.glob(os.path.join(
                a.sim_log_dir, "sim_seed%s_*.log" % d["seed"]))
            # The log CLOSEST IN TIME to the recording, not the newest one. A
            # seed is re-run whenever a trajectory is re-recorded, so "newest"
            # silently attaches a later run's layout to this run's waypoints.
            # The layout happens to be seed-deterministic today, which is
            # exactly why that mistake would not announce itself.
            when = _recorded_epoch(d)
            if cand and when is not None:
                log = min(cand, key=lambda f: abs(os.path.getmtime(f) - when))
                skew = abs(os.path.getmtime(log) - when)
                if skew > 900:
                    print("WARNING: nearest sim log for seed %s is %.0f min "
                          "from the recording; name it with --sim-log instead"
                          % (d["seed"], skew / 60.0), file=sys.stderr)
            elif cand:
                log = max(cand, key=os.path.getmtime)
                print("WARNING: recording carries no timestamp; using the "
                      "newest sim log for seed %s" % d["seed"], file=sys.stderr)
        if log:
            objs, home = read_layout(log)
            if objs:
                d["layout"] = objs
                if d.get("home_arm_rad") is None:
                    d["home_arm_rad"] = home
                print("recovered layout from %s: %s"
                      % (log, ", ".join(sorted(objs))))

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
    if not meta["layout_the_trajectory_assumes"]:
        print("\nWARNING: this trajectory carries NO layout. It is position-"
              "controlled and\n         senses nothing, so replaying it without "
              "knowing where the vessels\n         were planned to be will pour "
              "onto the bench. Re-export with\n         --sim-log or "
              "--sim-log-dir before handing it to anyone.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
