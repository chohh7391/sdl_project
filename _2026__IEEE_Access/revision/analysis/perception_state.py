#!/usr/bin/env python3
"""What the perception state put into the World State, and what it cost.

    perception_state.py CAMPAIGN_DIR [--out perception_state.md]

For every task with both a ground-truth and a perception batch
(cutamp/<task>_ground_truth_rep*.csv, cutamp/<task>_perception_rep*.csv),
from the trial driver's perc_* columns (tamp_perception_report):

  localization   per tagged vessel, where its pose came from: the fixed cell
                 cameras (fixed), the wrist camera's recovery scan (recovery),
                 or nowhere (failed), with Wilson 95% intervals, and the time
                 localizing took when a scan was needed
  error          the pose the plan used against the simulator's own pose at
                 the same moment: horizontal distance and height [mm], yaw
                 [deg] (yaw of a vessel with a round body is scored as the
                 tag's, i.e. as recorded), by source
  check          what the fixed cameras found when they looked again after
                 planning and before the arm moved: verified (within
                 tolerance), moved (the plan was redone), unseen
  success        task success in each state, paired on the same layouts
                 (exact McNemar per repetition), and in the perception state
                 by whether any vessel needed the scan

Nothing here re-scores a trial: success is the driver's task_success.
"""
import argparse
import csv
import glob
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_planners import wilson, exact_mcnemar  # noqa: E402

VESSELS = ("beaker", "flask")


def _truthy(v):
    return str(v).strip() in ("1", "True", "true")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load(pattern):
    """{rep: {seed: row}}"""
    out = {}
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"_rep(\d+)\.csv$", f)
        if m:
            out[int(m.group(1))] = {int(r["seed"]): r for r in csv.DictReader(open(f))}
    return out


def _rate(k, n):
    lo, hi = wilson(k, n)
    return "%d/%d = %.1f%% [%.1f, %.1f]" % (k, n, 100.0 * k / n, lo, hi) if n else "0/0"


def _stats(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return "--"
    p90 = vals[min(len(vals) - 1, int(round(0.9 * (len(vals) - 1))))]
    return "%.1f / %.1f / %.1f / %.1f (n=%d)" % (
        statistics.median(vals), statistics.mean(vals), p90, vals[-1], len(vals))


def report(campaign):
    lines = ["# Perception state: localization, error, the check before execution, success\n"]
    tasks = sorted({os.path.basename(f).split("_")[0]
                    for f in glob.glob(os.path.join(campaign, "cutamp", "*_perception_rep*.csv"))})
    if not tasks:
        return "\n".join(lines + ["no perception batches"]) + "\n"
    for task in tasks:
        gt = load(os.path.join(campaign, "cutamp", "%s_ground_truth_rep*.csv" % task))
        pe = load(os.path.join(campaign, "cutamp", "%s_perception_rep*.csv" % task))
        rows = [r for rep in pe.values() for r in rep.values()]
        lines.append("\n## %s (%d perception trials)\n" % (task, len(rows)))

        # which vessels this task perceives: those with a source in any row
        vessels = [v for v in VESSELS if any(r.get("perc_%s_source" % v) for r in rows)]
        lines.append("| vessel | fixed cameras | wrist recovery | not localized | "
                     "localization with recovery [Wilson 95%] |")
        lines.append("|---|---|---|---|---|")
        for v in vessels:
            src = [r.get("perc_%s_source" % v, "") for r in rows]
            n = sum(1 for x in src if x)
            k_f, k_r, k_x = src.count("fixed"), src.count("recovery"), src.count("failed")
            lines.append("| %s | %d | %d | %d | %s |" % (v, k_f, k_r, k_x, _rate(k_f + k_r, n)))
        # One sweep serves every vessel the fixed cameras miss, and its time is
        # booked to the vessel it started for, so the time is per trial.
        scan_t = sorted(sum(_f(r.get("perc_%s_recovery_s" % v)) or 0.0 for v in vessels)
                        for r in rows
                        if any(r.get("perc_%s_source" % v) in ("recovery", "failed") for v in vessels))
        lines.append("\ntrials that needed the wrist scan: %d; time localizing in them [s] "
                     "median / max: %s" % (len(scan_t), ("%.1f / %.1f" % (statistics.median(scan_t), scan_t[-1]))
                                           if scan_t else "--"))

        lines.append("\nPose error against the simulator, median / mean / p90 / max:\n")
        lines.append("| vessel | source | horizontal [mm] | height, absolute [mm] | yaw, absolute [deg] |")
        lines.append("|---|---|---|---|---|")
        for v in vessels:
            for source in ("fixed", "recovery"):
                sel = [r for r in rows if r.get("perc_%s_source" % v) == source]
                if not sel:
                    continue
                lines.append("| %s | %s | %s | %s | %s |" % (
                    v, source,
                    _stats([_f(r.get("perc_%s_err_xy_mm" % v)) for r in sel]),
                    _stats([abs(x) for x in (_f(r.get("perc_%s_err_z_mm" % v)) for r in sel) if x is not None]),
                    _stats([abs(x) for x in (_f(r.get("perc_%s_err_yaw_deg" % v)) for r in sel) if x is not None])))

        lines.append("\nThe check before execution (fixed cameras, after planning):\n")
        lines.append("| vessel | verified | moved (planned again) | unseen | disagreement when seen [mm] median / max |")
        lines.append("|---|---|---|---|---|")
        for v in vessels:
            ver = [r.get("perc_%s_verify" % v, "") for r in rows]
            d = sorted(x for x in (_f(r.get("perc_%s_verify_dxy_mm" % v)) for r in rows) if x is not None)
            lines.append("| %s | %d | %d | %d | %s |" % (
                v, ver.count("verified"), ver.count("moved"), ver.count("unseen"),
                ("%.1f / %.1f" % (statistics.median(d), d[-1])) if d else "--"))
        lines.append("\nplans redone after a vessel moved: %d" % sum(_truthy(r.get("perc_replanned")) for r in rows))

        # success, paired
        lines.append("\nTask success, ground-truth vs perception state, same layouts:\n")
        lines.append("| repetition | ground truth | perception | ground truth only | perception only | exact McNemar p |")
        lines.append("|---|---|---|---|---|---|")
        k_gt = n_gt = k_pe = n_pe = 0
        for rep in sorted(set(gt) & set(pe)):
            shared = sorted(set(gt[rep]) & set(pe[rep]))
            a = {s: _truthy(gt[rep][s]["task_success"]) for s in shared}
            b = {s: _truthy(pe[rep][s]["task_success"]) for s in shared}
            only_a = sum(1 for s in shared if a[s] and not b[s])
            only_b = sum(1 for s in shared if b[s] and not a[s])
            lines.append("| %d | %d/%d | %d/%d | %d | %d | %.3g |" % (
                rep, sum(a.values()), len(shared), sum(b.values()), len(shared),
                only_a, only_b, exact_mcnemar(only_a, only_b)))
            k_gt += sum(a.values()); n_gt += len(shared)
            k_pe += sum(b.values()); n_pe += len(shared)
        lines.append("\npooled: ground truth %s, perception %s" % (_rate(k_gt, n_gt), _rate(k_pe, n_pe)))
        if not vessels:
            lines.append("(no perc_* columns in these rows: recorded before the perception report existed)")
            continue
        scanned = [r for r in rows if any(r.get("perc_%s_source" % v) == "recovery" for v in vessels)]
        plain = [r for r in rows if r not in scanned
                 and all(r.get("perc_%s_source" % v) == "fixed" for v in vessels)]
        lines.append("perception, every vessel from the fixed cameras: %s" % _rate(
            sum(_truthy(r["task_success"]) for r in plain), len(plain)))
        lines.append("perception, a vessel recovered by the wrist scan: %s" % _rate(
            sum(_truthy(r["task_success"]) for r in scanned), len(scanned)))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign")
    ap.add_argument("--out")
    a = ap.parse_args()
    text = report(a.campaign)
    sys.stdout.write(text)
    if a.out:
        open(a.out, "w").write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
