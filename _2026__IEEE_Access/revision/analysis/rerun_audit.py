#!/usr/bin/env python3
"""Audit a re-run campaign directory before any of its numbers are used.

  rerun_audit.py DATA/rerun/<TAG>                 # report; exit 0 only if CLEAN
  rerun_audit.py DATA/rerun/<TAG> --drop-flagged  # remove contaminated trials

Contamination is judged PER TRIAL, not per batch. Every trial row carries the
time it was written; the GPU monitor samples every 30 s. A trial is flagged if
any monitor sample inside its window saw a GPU process that is not part of the
campaign. A cuTAMP trial's window runs from the previous row (trials run one
after another and each row is written as its trial ends), capped at
CUTAMP_TRIAL_WINDOW_S; a PDDLStream row's window is its own planning time.

--drop-flagged rewrites each CSV without its flagged rows (the original is kept
as <name>.csv.flagged-<epoch>), after which re-running the campaign with the
same RERUN_TAG runs exactly those trials again: the stages resume by filling in
missing seeds. Other checks: every batch finished, no real-cell scene marker in
any batch's simulator logs (SCENE_CONTAMINATED), every layout present exactly
once per CSV.
"""
import argparse
import collections
import csv
import datetime
import glob
import os
import shutil
import sys
import time

CUTAMP_TRIAL_WINDOW_S = 15 * 60   # start-up + three planning attempts + execution
PDDL_SLACK_S = 30


def epoch_of(ts):
    # trial rows are written in the machine's local time, ISO format
    return time.mktime(datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S").timetuple())


def load_monitor(d):
    samples = []
    mon = os.path.join(d, "gpu_monitor.csv")
    if os.path.exists(mon):
        for r in csv.DictReader(open(mon)):
            try:
                samples.append((int(r["epoch"]), int(r["n_foreign"] or 0),
                                float(r["loadavg_1m"] or 0), r.get("foreign", "")))
            except ValueError:
                continue
    return samples


def flagged_rows(path, samples, pddl):
    """Row indices whose window saw a foreign GPU process, with what was seen.

    Trials in a batch run one after another and each row is written as its trial
    ends, so a cuTAMP trial occupied the interval since the PREVIOUS row -- capped
    at CUTAMP_TRIAL_WINDOW_S, which is what a gap between two sessions of a
    resumed batch falls back to. A PDDLStream row's window is its own planning
    time.
    """
    out = []
    prev_t1 = None
    for i, r in enumerate(csv.DictReader(open(path))):
        try:
            t1 = epoch_of(r["timestamp"])
        except (KeyError, ValueError):
            continue
        if pddl:
            try:
                t0 = t1 - float(r.get("planning_time_s") or 0) - PDDL_SLACK_S
            except ValueError:
                t0 = t1 - 200
        else:
            t0 = t1 - CUTAMP_TRIAL_WINDOW_S
            if prev_t1 is not None and prev_t1 <= t1:
                t0 = max(t0, prev_t1)
            prev_t1 = t1
        hits = [s for s in samples if t0 <= s[0] <= t1 and s[1] > 0]
        if hits:
            out.append((i, r.get("seed"), r.get("planner_seed", ""), hits[0][3][:100]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--drop-flagged", action="store_true")
    a = ap.parse_args()
    d = a.dir
    out = ["# Re-run audit: `%s`\n" % os.path.basename(os.path.normpath(d))]
    say = out.append
    commit = open(os.path.join(d, "COMMIT")).read().strip() if os.path.exists(os.path.join(d, "COMMIT")) else "?"
    budgets = open(os.path.join(d, "BUDGETS")).read().strip() if os.path.exists(os.path.join(d, "BUDGETS")) else "?"
    say("commit `%s`, declared budgets %s s\n" % (commit, budgets))
    samples = load_monitor(d)
    clean = True
    if not samples:
        say("**No GPU monitor samples** -- exclusivity cannot be shown for any trial.\n")
        clean = False

    # --- batches ---------------------------------------------------------------
    contaminated = set()
    cf = os.path.join(d, "SCENE_CONTAMINATED")
    if os.path.exists(cf):
        contaminated = {os.path.basename(l.strip()) for l in open(cf) if l.strip()}
    last = collections.OrderedDict()
    bf = os.path.join(d, "batches.tsv")
    if os.path.exists(bf):
        for line in open(bf):
            p = line.rstrip("\n").split("\t")
            if len(p) == 4:
                last.setdefault(p[0], []).append((int(p[1]), int(p[2]), p[3]))
    say("## Batches\n")
    say("| batch | attempts | last status | wall time (last) | max load avg (last) | scene |")
    say("|---|---|---|---|---|---|")
    for name, runs in last.items():
        t0, t1, status = runs[-1]
        load = max((s[2] for s in samples if t0 <= s[0] <= t1), default=float("nan"))
        scene = "**CONTAMINATED**" if name in contaminated else "paper"
        clean &= status == "done" and name not in contaminated
        say("| %s | %d | %s | %.1f h | %.1f | %s |"
            % (name, len(runs), status if status == "done" else "**%s**" % status,
               (t1 - t0) / 3600.0, load, scene))
    if not last:
        say("| (no batches recorded) | | | | | |")
        clean = False

    # --- trials ------------------------------------------------------------------
    say("\n## Trials: completeness and GPU exclusivity\n")
    say("| file | planner seed | layouts | duplicates | missing | trials that shared the GPU |")
    say("|---|---|---|---|---|---|")
    to_drop = {}
    for sub, pddl in (("cutamp", False), ("pddlstream", True)):
        for f in sorted(glob.glob(os.path.join(d, sub, "*.csv"))):
            rows = list(csv.DictReader(open(f)))
            groups = collections.defaultdict(list)
            for r in rows:
                groups[r.get("planner_seed", "") if pddl else ""].append(r)
            flags = flagged_rows(f, samples, pddl) if samples else []
            if flags:
                to_drop[f] = {i for i, *_ in flags}
            for ps in sorted(groups, key=lambda k: int(k) if k else -1):
                c = collections.Counter(int(r["seed"]) for r in groups[ps])
                dup = sorted(s for s, k in c.items() if k > 1)
                miss = sorted(set(range(30)) - set(c))
                fl = sorted(int(seed) for _, seed, p, _ in flags if (p or "") == ps)
                clean &= not dup and not miss and not fl
                say("| `%s` | %s | %d | %s | %s | %s |"
                    % (os.path.basename(f), ps or "-", len(c), dup or "-", miss or "-",
                       ("**%s**" % fl) if fl else "-"))
            for _, seed, ps, what in flags[:2]:
                say("|  | foreign process near seed %s: `%s` | | | | |" % (seed, what))

    say("\n**Verdict: %s**" % ("CLEAN" if clean else
        "NOT CLEAN -- numbers from the rows in bold are not results. Trials that shared the GPU "
        "can be removed with --drop-flagged and re-run under the same RERUN_TAG."))

    if a.drop_flagged and to_drop:
        stamp = int(time.time())
        for f, idx in to_drop.items():
            shutil.copy(f, "%s.flagged-%d" % (f, stamp))
            with open(f) as fh:
                rd = csv.DictReader(fh)
                fields, rows = rd.fieldnames, list(rd)
            with open(f, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(r for i, r in enumerate(rows) if i not in idx)
            say("\ndropped %d flagged trial(s) from `%s` (original kept as `.flagged-%d`)"
                % (len(idx), os.path.basename(f), stamp))

    print("\n".join(out))
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
