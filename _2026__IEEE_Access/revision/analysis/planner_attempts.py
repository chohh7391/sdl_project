#!/usr/bin/env python3
"""How cuTAMP spent each planning call: restarts, cuRobo candidates, time.

    planner_attempts.py CAMPAIGN_DIR [--out attempts.md]

Reads each trial's tamp_server log next to its CSV row (logs/<batch>/
tamp_seed<N>_<ts>.log, the latest one that started before the row was
written) and counts, per task and state source:

  attempts    optimizations run (the first plus restarts with a new seed)
  candidate   which satisfying particle cuRobo planned, counted from the
              first candidate of the successful attempt (1 = the best one,
              i.e. what the planner did before candidates were added)

and, for PDDLStream, from its own CSV (pddlstream/pddlstream_<task>_5streams.csv,
rows written with restarts):

  attempts    solve() calls (the first plus restarts with fresh samples)
  first       whether the first call alone solved it -- the verdict a run
              without restarts would have drawn at that stream

and, when the tag also holds a run without restarts (pddlstream_no_restart/),
how many layouts that run solved.

Planning time is the driver's wall clock from the CSV. Nothing here is a
success rate: planner_comparison.py reports those, censored at the budgets.
"""
import argparse
import collections
import csv
import datetime as dt
import glob
import os
import re
import statistics
import sys


def _ts(path):
    m = re.search(r"_(\d{8}_\d{6})\.log$", path)
    return dt.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S") if m else None


def trial_log(campaign, batch, seed, written):
    logs = [f for f in glob.glob(os.path.join(campaign, "logs", batch, "tamp_seed%s_*.log" % seed))
            if _ts(f) is not None and _ts(f) <= written]
    return max(logs, key=_ts) if logs else None


def parse(path):
    """(attempts, winning candidate index or None, candidates failed in total)."""
    attempts, won, failed = 0, None, 0
    for line in open(path, errors="replace"):
        if "[plan] seeded attempt" in line:
            attempts += 1
        m = re.search(r"\[curobo\] candidate (\d+)/\d+ (failed|gave a full motion plan)", line)
        if m:
            if m.group(2) == "failed":
                failed += 1
            else:
                won = int(m.group(1))
        m = re.search(r"attempts: (\d+), time:", line)
        if m:
            attempts = int(m.group(1))
    return attempts, won, failed


ATTEMPT_BINS = ((1, 1), (2, 2), (3, 9), (10, 10 ** 9))


def _bin(lo, hi):
    return str(lo) if lo == hi else ("%d+" % lo if hi >= 10 ** 9 else "%d-%d" % (lo, hi))


def pddl_section(campaign):
    out = []
    d = os.path.join(campaign, "pddlstream")
    for f in sorted(glob.glob(os.path.join(d, "pddlstream_*_5streams.csv"))):
        task = os.path.basename(f).split("_")[1]
        rows = list(csv.DictReader(open(f)))
        if not rows or "attempts" not in rows[0]:
            continue
        ok = lambda r: str(r["plan_success"]).strip() in ("1", "True", "true")
        restart = sorted({r.get("restart", "") for r in rows})
        out.append("\n## pddlstream, %s (%d trials, restart %s)\n" % (task, len(rows), "/".join(restart)))
        out.append("| attempts | planned | not planned | solved-only time min / median / max [s] |")
        out.append("|---|---|---|---|")
        for lo, hi in ATTEMPT_BINS:
            sub = [r for r in rows if lo <= int(r["attempts"] or 0) <= hi]
            if not sub:
                continue
            ts = sorted(float(r["planning_time_s"]) for r in sub if ok(r))
            span = "%.1f / %.1f / %.1f" % (ts[0], statistics.median(ts), ts[-1]) if ts else "--"
            out.append("| %s | %d | %d | %s |" % (_bin(lo, hi), sum(ok(r) for r in sub),
                                                 sum(not ok(r) for r in sub), span))
        first = sum(str(r.get("first_attempt_success")).strip() == "1" for r in rows)
        out.append("\nsolved by the first solve() alone: %d/%d; solved only after a restart: %d"
                   % (first, len(rows), sum(ok(r) for r in rows) - first))
        reasons = collections.Counter(r["failure_reason"].split(":")[0] for r in rows if not ok(r))
        if reasons:
            out.append("not planned, by reason: " + ", ".join("%s %d" % kv for kv in sorted(reasons.items())))
        ref = os.path.join(campaign, "pddlstream_no_restart", os.path.basename(f))
        if os.path.exists(ref):
            rr = list(csv.DictReader(open(ref)))
            out.append("the same tag's run without restarts (pddlstream_no_restart/): %d/%d planned"
                       % (sum(ok(r) for r in rr), len(rr)))
    if out:
        out.insert(0, "\n# PDDLStream planning calls: restarts\n\nPer trial, from its CSV. "
                      "`attempts` = solve() calls within the limit; each restart puts the scene back "
                      "and draws a fresh sample stream.\n")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    groups = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.campaign, "cutamp", "*.csv"))):
        batch = os.path.basename(f)[:-4]
        task, src = batch.split("_")[0], ("perception" if "_perception_" in batch else "ground_truth")
        for r in csv.DictReader(open(f)):
            log = trial_log(a.campaign, batch, r["seed"], dt.datetime.fromisoformat(r["timestamp"]))
            if log is None:
                continue
            n, won, failed = parse(log)
            ok = str(r["plan_success"]).strip() in ("1", "True", "true")
            t = float(r["planning_time_s"]) if r.get("planning_time_s") else None
            groups[(task, src)].append((ok, n, won, failed, t))

    out = ["# cuTAMP planning calls: restarts and cuRobo candidates\n",
           "Per trial, from its tamp_server log. `attempts` = optimizations run; `candidate` = which "
           "satisfying particle cuRobo planned in the successful attempt (1 = the best, the only one "
           "tried before candidates were added). Times are the driver's wall clock, solved trials only "
           "-- success rates and censored times are in the planner comparison.\n"]
    for (task, src), rows in sorted(groups.items()):
        out.append("\n## %s, %s (%d trials with a log)\n" % (task, src, len(rows)))
        out.append("| attempts | planned | not planned | solved-only time min / median / max [s] |")
        out.append("|---|---|---|---|")
        for n in sorted({x[1] for x in rows}):
            sub = [x for x in rows if x[1] == n]
            ts = sorted(x[4] for x in sub if x[0] and x[4] is not None)
            span = "%.1f / %.1f / %.1f" % (ts[0], statistics.median(ts), ts[-1]) if ts else "--"
            out.append("| %d | %d | %d | %s |" % (n, sum(x[0] for x in sub), sum(not x[0] for x in sub), span))
        won = collections.Counter(x[2] for x in rows if x[0] and x[2] is not None)
        if won:
            out.append("\nsuccessful attempt planned with candidate: " + ", ".join(
                "%d: %d" % (k, v) for k, v in sorted(won.items())))
        out.append("candidates cuRobo could not plan, all trials: %d" % sum(x[3] for x in rows))
    out += pddl_section(a.campaign)
    text = "\n".join(out) + "\n"
    sys.stdout.write(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
