#!/usr/bin/env python3
"""Paired cuTAMP vs PDDLStream comparison over REPEATED runs, per budget.

Inputs (one directory each):
  cuTAMP      <task>_<state>_rep<r>.csv   30 layouts per file, one file per
              repetition r = 0, 1, 2, ... (scripts/trials/trial_driver.py rows);
              --state ground_truth (default) or perception
  PDDLStream  pddlstream_<task>_5streams.csv   30 layouts x N planner seeds, the
              seed in the `planner_seed` column (run_paired_trials.py rows); for
              the perception state, the directory 20_pddlstream.sh wrote with
              RERUN_PDDL_STATE=perception (pddlstream_perception/)

Pairing: cuTAMP repetition r is paired with PDDLStream planner seed r on the
same 30 layouts, for every r both planners have. Each exact McNemar test is
therefore on 30 INDEPENDENT pairs; pooling the repetitions into one test would
not be, because the repetitions of one layout share that layout.

Pooled success rates use every trial of each planner (all cuTAMP repetitions,
all PDDLStream seeds) with a Wilson 95% interval. A success that arrives after
the budget is an unsolved, censored observation under that budget, for both
planners alike, and the restricted mean time to solution comes from the
Kaplan-Meier estimate over the budget, as do the times by which 25/50/75% of
the runs were solved (NR: never reached within the budget). The spread of the
solved runs alone (mean, SD, quartiles) is reported beside them, labelled as
such: it describes only the runs that succeeded.

compare_planners.py answers the same question for ONE run per planner; it keys
results by seed and refuses files with several rows per seed, which is why
this script exists.

    planner_comparison.py --cutamp DIR --pddlstream DIR --budgets 60,120,180 \
        [--tasks transfer,move,stir] [--out results.md]
"""
import argparse
import csv
import glob
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_planners import wilson, exact_mcnemar, km_rmst  # noqa: E402


def _truthy(v):
    return str(v).strip() in ("1", "True", "true")


def load_cutamp_reps(d, task, state="ground_truth"):
    """{rep: {seed: (ok, t)}} from <task>_<state>_rep<r>.csv."""
    reps = {}
    for f in sorted(glob.glob(os.path.join(d, "%s_%s_rep*.csv" % (task, state)))):
        m = re.search(r"_rep(\d+)\.csv$", f)
        if not m:
            continue
        rows = {}
        for r in csv.DictReader(open(f)):
            s = int(r["seed"])
            if s in rows:
                raise SystemExit("%s: seed %d appears twice -- a resumed batch "
                                 "appended to a partial one" % (f, s))
            t = r.get("planning_time_s", "")
            rows[s] = (_truthy(r["plan_success"]), float(t) if t else None)
        reps[int(m.group(1))] = rows
    return reps


def load_pddl_streams(d, task):
    """{planner_seed: {seed: (ok, t)}} from pddlstream_<task>_5streams.csv."""
    f = os.path.join(d, "pddlstream_%s_5streams.csv" % task)
    if not os.path.exists(f):
        return {}
    out = {}
    for r in csv.DictReader(open(f)):
        ps, s = int(r["planner_seed"]), int(r["seed"])
        rows = out.setdefault(ps, {})
        if s in rows:
            raise SystemExit("%s: layout %d appears twice for planner seed %d"
                             % (f, s, ps))
        t = r.get("planning_time_s", "")
        rows[s] = (_truthy(r["plan_success"]), float(t) if t else None)
    return out


# cuTAMP records where the vessels settled at trial start; PDDLStream reads the
# spawn poses exported from Isaac Sim. Same layout agrees to settling (0.1 mm
# measured), different layouts differ by centimetres.
LAYOUT_TOL_M = 0.005


def _xy(v):
    try:
        return tuple(float(x) for x in v.split(";"))
    except (AttributeError, ValueError):
        return None


def layouts(paths):
    """{seed: (beaker_xy, flask_xy)} from CSVs; the first file that has a seed wins."""
    out = {}
    for f in paths:
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            out.setdefault(int(r["seed"]), (_xy(r.get("beaker_xy", "")), _xy(r.get("flask_xy", ""))))
    return out


def check_same_layouts(task, cu_dir, pd_dir, state="ground_truth"):
    """Refuse to pair planners that did not see the same scenes.

    The pairing is only meaningful if layout s is the same scene for both: the
    baseline reads the layouts exported from Isaac Sim, and a change to the
    randomizer (or to anything its overlap test depends on) would silently pair
    different scenes. Both runners record where the vessels were, so compare.
    """
    cu = layouts(sorted(glob.glob(os.path.join(cu_dir, "%s_%s_rep*.csv" % (task, state)))))
    pd_file = os.path.join(pd_dir, "pddlstream_%s_5streams.csv" % task)
    pd = layouts([pd_file])
    bad = []
    if state != "ground_truth":
        # ...and the baseline was given the state of THIS cuTAMP state's trials
        for r in csv.DictReader(open(pd_file)):
            if r.get("state_source", "ground_truth") != state:
                bad.append("seed %s: PDDLStream row is from the %s state"
                           % (r["seed"], r.get("state_source") or "ground_truth"))
    for s in sorted(set(cu) & set(pd)):
        for i, name in enumerate(("beaker", "flask")):
            a, b = cu[s][i], pd[s][i]
            if a and b and max(abs(x - y) for x, y in zip(a, b)) > LAYOUT_TOL_M:
                bad.append("seed %d %s %s vs %s" % (s, name, a, b))
    if bad:
        raise SystemExit("%s: cuTAMP and PDDLStream did not see the same layouts -- not paired:\n  %s"
                         % (task, "\n  ".join(bad[:10])))
    return len(set(cu) & set(pd))


def within(res, budget):
    return {s: (ok and t is not None and t <= budget, t) for s, (ok, t) in res.items()}


def _km(trials, budget):
    times, ev = [], []
    for ok, t in trials:
        if ok:
            times.append(min(t, budget)); ev.append(1)
        else:
            times.append(budget); ev.append(0)
    return km_rmst(times, ev, budget)


def rmst(trials, budget):
    return _km(trials, budget)[0]


def km_quartiles(trials, budget):
    """Times by which 25/50/75% of the runs were solved; None if not within the budget."""
    curve = _km(trials, budget)[2]
    return [next((t for t, s in curve if s <= 1.0 - q + 1e-12), None) for q in (0.25, 0.5, 0.75)]


def _q(t):
    return "NR" if t is None else "%.1f" % t


def solved_times(res_list):
    return sorted(t for res in res_list for ok, t in res.values() if ok and t is not None)


def spread(ts):
    """Solved-only spread: mean, SD (n-1), median and quartiles (linear interpolation)."""
    if len(ts) == 1:
        return ts[0], 0.0, ts[0], ts[0], ts[0]
    q1, med, q3 = statistics.quantiles(ts, n=4, method="inclusive")
    return statistics.mean(ts), statistics.stdev(ts), med, q1, q3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutamp", required=True, help="directory of cuTAMP rep CSVs")
    ap.add_argument("--pddlstream", required=True, help="directory of PDDLStream CSVs")
    ap.add_argument("--budgets", default="60,120,180",
                    help="comma-separated wall-clock budgets [s], DECLARED before the run")
    ap.add_argument("--tasks", default="transfer,move,stir")
    ap.add_argument("--state", default="ground_truth", choices=("ground_truth", "perception"),
                    help="which cuTAMP state to pair; PDDLStream rows must be from the same")
    ap.add_argument("--out", default=None, help="also write the report here (markdown)")
    a = ap.parse_args()
    budgets = [float(b) for b in a.budgets.split(",") if b.strip()]

    lines = []
    say = lines.append
    say("# Planner comparison (cuTAMP vs PDDLStream)%s\n"
        % ("" if a.state == "ground_truth" else ", %s state" % a.state))
    say("cuTAMP: `%s`  \nPDDLStream: `%s`  \nbudgets (declared): %s s\n"
        % (a.cutamp, a.pddlstream, ", ".join("%.0f" % b for b in budgets)))

    for task in [t.strip() for t in a.tasks.split(",") if t.strip()]:
        cu = load_cutamp_reps(a.cutamp, task, a.state)
        pd = load_pddl_streams(a.pddlstream, task)
        say("\n## %s\n" % task)
        if not cu or not pd:
            say("missing data: %d cuTAMP repetition file(s), %d PDDLStream seed(s)\n"
                % (len(cu), len(pd)))
            continue
        pairs = sorted(set(cu) & set(pd))
        n_same = check_same_layouts(task, a.cutamp, a.pddlstream, a.state)
        say("cuTAMP repetitions %s, PDDLStream planner seeds %s; paired on %s. "
            "Layouts checked identical on all %d shared seeds.\n"
            % (sorted(cu), sorted(pd), pairs, n_same))

        cs, ps = solved_times(cu.values()), solved_times(pd.values())
        for name, ts in (("cuTAMP", cs), ("PDDLStream", ps)):
            if ts:
                mean, sd, med, q1, q3 = spread(ts)
                say("- %s solved-only planning time (n=%d): mean %.2f s, SD %.2f s, "
                    "median %.2f s [IQR %.2f-%.2f], range %.2f-%.2f s"
                    % (name, len(ts), mean, sd, med, q1, q3, ts[0], ts[-1]))
        say("")
        say("| budget | cuTAMP success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] "
            "| PDDLStream success [Wilson 95%] | RMST | KM 25/50/75% solved by [s] |")
        say("|---|---|---|---|---|---|---|")
        per_budget = []
        for budget in budgets:
            cu_trials = [v for r in cu for v in within(cu[r], budget).values()]
            pd_trials = [v for r in pd for v in within(pd[r], budget).values()]
            kc, nc = sum(v[0] for v in cu_trials), len(cu_trials)
            kp, np_ = sum(v[0] for v in pd_trials), len(pd_trials)
            lc, hc = wilson(kc, nc)
            lp, hp = wilson(kp, np_)
            say("| %.0f s | %d/%d = %.1f%% [%.1f, %.1f] | %.1f s | %s "
                "| %d/%d = %.1f%% [%.1f, %.1f] | %.1f s | %s |"
                % (budget, kc, nc, 100.0 * kc / nc, lc, hc, rmst(cu_trials, budget),
                   " / ".join(_q(t) for t in km_quartiles(cu_trials, budget)),
                   kp, np_, 100.0 * kp / np_, lp, hp, rmst(pd_trials, budget),
                   " / ".join(_q(t) for t in km_quartiles(pd_trials, budget))))
            rows = []
            for r in pairs:
                c, p = within(cu[r], budget), within(pd[r], budget)
                shared = sorted(set(c) & set(p))
                b = sum(1 for s in shared if c[s][0] and not p[s][0])
                cc = sum(1 for s in shared if p[s][0] and not c[s][0])
                rows.append((r, len(shared), sum(c[s][0] for s in shared),
                             sum(p[s][0] for s in shared), b, cc, exact_mcnemar(b, cc)))
            per_budget.append((budget, rows))
        say("")
        say("| budget | pairing | cuTAMP | PDDLStream | cuTAMP-only | PDDLStream-only | exact McNemar p |")
        say("|---|---|---|---|---|---|---|")
        for budget, rows in per_budget:
            for r, n, kc, kp, b, c, p in rows:
                say("| %.0f s | rep %d vs seed %d | %d/%d | %d/%d | %d | %d | %.3g |"
                    % (budget, r, r, kc, n, kp, n, b, c, p))

    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
