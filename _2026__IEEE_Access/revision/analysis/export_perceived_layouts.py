#!/usr/bin/env python3
"""The World State cuTAMP planned from in the perception state, as PDDLStream layouts.

    export_perceived_layouts.py CAMPAIGN_DIR --task transfer [--out FILE]

For each layout of CAMPAIGN_DIR/cutamp/<task>_perception_rep0.csv, the trial's
tamp_server log (the latest logs/<batch>/tamp_seed<N>_<ts>.log that started
before the row was written) holds the perception report the trial planned
from: per tagged vessel the pose the plan used and the simulator's pose at the
same moment. This writes analysis/data/seed_layouts.json with those vessels
moved by exactly that difference -- in x, y and yaw:

  xy   layout xy + (perceived xy - simulator xy)
  yaw  layout yaw + (perceived yaw - simulator yaw)
  z    unchanged: PDDLStream sets every object down on the table from its own
       geometry (run_paired_trials.place) and reads no height in the ground-
       truth state either, so the perceived height error does not enter it

The simulator pose and the exported layout agree to settling (0.1 mm), so the
two PDDLStream states differ by the perception error alone. Untagged objects
keep their layout pose, as in cuTAMP's perception state. Each vessel keeps its
layout xy as gt_xy, which the runner writes to beaker_xy/flask_xy so the
planner comparison can check the pairing against cuTAMP's rows.
"""
import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUTS = os.path.join(HERE, "data", "seed_layouts.json")
VESSELS = ("beaker", "flask")
# The simulator pose the report gives and the exported spawn pose of the same
# layout agree to settling; more than this means the wrong layout or trial.
GT_TOL_M = 0.005


def _ts(path):
    m = re.search(r"_(\d{8}_\d{6})\.log$", path)
    return dt.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S") if m else None


def trial_log(campaign, batch, seed, written):
    logs = [f for f in glob.glob(os.path.join(campaign, "logs", batch, "tamp_seed%s_*.log" % seed))
            if _ts(f) is not None and _ts(f) <= written]
    return max(logs, key=_ts) if logs else None


def perception_report(path):
    """The first perception report the trial logged (the one it planned from)."""
    for line in open(path, errors="replace"):
        i = line.find("[perception] {")
        if i >= 0:
            return json.loads(line[i + len("[perception] "):])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign")
    ap.add_argument("--task", required=True, choices=("transfer", "move", "stir"))
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    base = json.load(open(LAYOUTS))
    by_seed = {l["seed"]: l for l in base["layouts"]}
    batch = "%s_perception_rep%d" % (a.task, a.rep)
    rows = list(csv.DictReader(open(os.path.join(a.campaign, "cutamp", batch + ".csv"))))
    out_layouts, problems = [], []
    for r in sorted(rows, key=lambda r: int(r["seed"])):
        seed = int(r["seed"])
        log = trial_log(a.campaign, batch, seed, dt.datetime.fromisoformat(r["timestamp"]))
        rep = perception_report(log) if log else None
        if rep is None:
            problems.append("seed %d: no perception report (%s)" % (seed, log))
            continue
        lay = json.loads(json.dumps(by_seed[seed]))
        lay["perception"] = {"trial_log": os.path.relpath(log, a.campaign)}
        for v in VESSELS:
            e = rep["entities"].get(v)
            if e is None or "pose" not in e:
                problems.append("seed %d: %s was not localized" % (seed, v))
                continue
            obj = lay["objects"][v]
            gx, gy = obj["xy"]
            if math.hypot(e["gt"][0] - gx, e["gt"][1] - gy) > GT_TOL_M:
                problems.append("seed %d: %s simulator pose %s is not the layout's %s"
                                % (seed, v, e["gt"][:2], obj["xy"]))
                continue
            obj["gt_xy"] = [gx, gy]
            obj["gt_yaw_rad"] = obj["yaw_rad"]
            obj["xy"] = [gx + e["pose"][0] - e["gt"][0], gy + e["pose"][1] - e["gt"][1]]
            obj["yaw_rad"] = obj["yaw_rad"] + math.radians(e["err_yaw_deg"])
            obj["yaw_deg"] = math.degrees(obj["yaw_rad"])
            lay["perception"][v] = {k: e.get(k) for k in ("source", "err_xy_mm", "err_z_mm", "err_yaw_deg")}
        out_layouts.append(lay)
    if problems:
        sys.exit("refusing to export %s:\n  %s" % (batch, "\n  ".join(problems)))
    doc = {"source": "%s, cutamp/%s.csv and its trials' perception reports; layouts from %s"
                     % (os.path.basename(os.path.normpath(a.campaign)), batch, os.path.relpath(LAYOUTS, a.campaign)),
           "state_source": "perception", "layouts": out_layouts}
    text = json.dumps(doc, indent=1) + "\n"
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
        print("wrote %s (%d layouts)" % (a.out, len(out_layouts)))
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
