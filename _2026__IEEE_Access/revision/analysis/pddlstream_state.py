#!/usr/bin/env python3
"""PDDLStream planning from the ground-truth state against the perception state.

    pddlstream_state.py CAMPAIGN_DIR [--budgets 60,120,180] [--out pddlstream_state.md]

Pairs, layout by layout and planner seed by planner seed,
CAMPAIGN_DIR/pddlstream/pddlstream_<task>_5streams.csv (the exported layouts)
with CAMPAIGN_DIR/pddlstream_perception/pddlstream_<task>_5streams.csv (the
vessel poses cuTAMP's perception trial of the same layout planned from,
analysis/export_perceived_layouts.py). The only difference between the two is
the perception error of the beaker and flask in x, y and yaw.

  success   planning success within each budget, Wilson 95%, and the exact
            McNemar test on the paired layouts, with the layouts that differ
  time      restricted mean time to solution over each budget (Kaplan-Meier,
            unsolved runs censored), and the solved runs' median
  input     where each perceived pose came from and how far it moved the
            vessel from the layout

PDDLStream only plans, as in the planner comparison: nothing here is executed.
"""
import argparse
import csv
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_planners import wilson, exact_mcnemar, km_rmst  # noqa: E402


def _ok(r):
    return str(r["plan_success"]).strip() in ("1", "True", "true")


def load(path):
    """{planner_seed: {seed: row}}"""
    out = {}
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            out.setdefault(int(r["planner_seed"]), {})[int(r["seed"])] = r
    return out


def within(r, budget):
    return _ok(r) and float(r["planning_time_s"]) <= budget


def rmst(rows, budget):
    times, ev = [], []
    for r in rows:
        if within(r, budget):
            times.append(float(r["planning_time_s"])); ev.append(1)
        else:
            times.append(budget); ev.append(0)
    return km_rmst(times, ev, budget)[0]


def _rate(k, n):
    lo, hi = wilson(k, n)
    return "%d/%d = %.1f%% [%.1f, %.1f]" % (k, n, 100.0 * k / n, lo, hi) if n else "0/0"


def _xy(v):
    try:
        return tuple(float(x) for x in v.split(";"))
    except (AttributeError, ValueError):
        return None


def report(campaign, budgets):
    lines = ["# PDDLStream: ground-truth state vs perception state\n",
             "Same layouts, same planner seeds; the perception state gives the planner the "
             "beaker and flask poses (x, y, yaw) cuTAMP's perception trial planned from. "
             "Planning only.\n"]
    for task in ("transfer", "move", "stir"):
        gt = load(os.path.join(campaign, "pddlstream", "pddlstream_%s_5streams.csv" % task))
        pe = load(os.path.join(campaign, "pddlstream_perception", "pddlstream_%s_5streams.csv" % task))
        lines.append("\n## %s\n" % task)
        seeds = sorted(set(gt) & set(pe))
        if not seeds:
            lines.append("missing data: ground truth %d planner seed(s), perception %d"
                         % (len(gt), len(pe)))
            continue
        bad = [s for ps in seeds for s in set(gt[ps]) & set(pe[ps])
               if gt[ps][s]["beaker_xy"] != pe[ps][s]["beaker_xy"]
               or gt[ps][s]["flask_xy"] != pe[ps][s]["flask_xy"]]
        if bad:
            raise SystemExit("%s: the two states' rows are not the same layouts: %s" % (task, bad[:10]))

        lines.append("| budget | planner seed | ground truth [Wilson 95%] | RMST | perception [Wilson 95%] "
                     "| RMST | ground truth only | perception only | exact McNemar p |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for budget in budgets:
            for ps in seeds:
                shared = sorted(set(gt[ps]) & set(pe[ps]))
                a = {s: within(gt[ps][s], budget) for s in shared}
                b = {s: within(pe[ps][s], budget) for s in shared}
                only_a = [s for s in shared if a[s] and not b[s]]
                only_b = [s for s in shared if b[s] and not a[s]]
                lines.append("| %.0f s | %d | %s | %.1f s | %s | %.1f s | %s | %s | %.3g |" % (
                    budget, ps, _rate(sum(a.values()), len(shared)),
                    rmst([gt[ps][s] for s in shared], budget),
                    _rate(sum(b.values()), len(shared)),
                    rmst([pe[ps][s] for s in shared], budget),
                    "%d %s" % (len(only_a), only_a) if only_a else "0",
                    "%d %s" % (len(only_b), only_b) if only_b else "0",
                    exact_mcnemar(len(only_a), len(only_b))))
        for name, d in (("ground truth", gt), ("perception", pe)):
            ts = sorted(float(r["planning_time_s"]) for ps in seeds for r in d[ps].values() if _ok(r))
            fails = sorted(r["failure_reason"].split(":")[0] for ps in seeds for r in d[ps].values() if not _ok(r))
            lines.append("\n%s: solved-only planning time median %s s (n=%d)%s" % (
                name, ("%.2f" % statistics.median(ts)) if ts else "--", len(ts),
                ("; not planned: %s" % ", ".join("%s %d" % (k, fails.count(k)) for k in sorted(set(fails))))
                if fails else ""))

        # the input: how far perception moved each vessel, and from which source
        lines.append("\nPerceived input (from the perception rows):\n")
        lines.append("| vessel | from the fixed cameras | from the wrist recovery | shift from the layout [mm] median / max |")
        lines.append("|---|---|---|---|")
        rows = [r for ps in seeds for r in pe[ps].values()]
        for v in ("beaker", "flask"):
            src = [r.get("perc_%s_source" % v, "") for r in rows]
            shift = sorted(1000 * math.hypot(p[0] - g[0], p[1] - g[1])
                           for p, g in ((_xy(r.get("perc_%s_xy" % v)), _xy(r["%s_xy" % v])) for r in rows)
                           if p and g)
            lines.append("| %s | %d | %d | %s |" % (
                v, src.count("fixed"), src.count("recovery"),
                ("%.1f / %.1f" % (statistics.median(shift), shift[-1])) if shift else "--"))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign")
    ap.add_argument("--budgets", default="60,120,180")
    ap.add_argument("--out")
    a = ap.parse_args()
    text = report(a.campaign, [float(b) for b in a.budgets.split(",") if b.strip()])
    sys.stdout.write(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
