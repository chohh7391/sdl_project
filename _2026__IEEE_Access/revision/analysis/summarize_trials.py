#!/usr/bin/env python3
"""Summarise a randomized-trial CSV produced by sdl_project/scripts/trials/trial_driver.py.

Reports the rates the paper needs, each with a Wilson 95% CI (PLAN.md section 4.2),
plus the descriptive statistics and the failure breakdown. Optionally audits the
per-trial logs for events that are recorded only there.

Usage:
  summarize_trials.py <trials.csv> [--logs <logdir>]

Deliberately stdlib-only so it runs in any of the project's environments.
"""
import argparse
import csv
import glob
import math
import os
import statistics as st


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion, in percent."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom * 100.0, (centre + half) / denom * 100.0)


def rate(label, k, n, note=""):
    lo, hi = wilson(k, n)
    pct = 100.0 * k / n if n else float("nan")
    print(f"  {label:<28} {k:>3}/{n:<3} = {pct:5.1f}%  [95% CI {lo:5.1f}, {hi:5.1f}]{note}")


def floats(rows, field):
    out = []
    for r in rows:
        v = (r.get(field) or "").strip()
        if v in ("", "nan"):
            continue
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out


def describe(label, values, unit=""):
    if not values:
        print(f"  {label:<28} (no data)")
        return
    sd = st.stdev(values) if len(values) > 1 else 0.0
    print(f"  {label:<28} n={len(values):<3} min {min(values):.2f}  median {st.median(values):.2f}"
          f"  mean {st.mean(values):.2f}  sd {sd:.2f}  max {max(values):.2f} {unit}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--logs", help="per-trial log directory to audit")
    ap.add_argument("--budget", type=float, default=60.0,
                    help="planning-time budget to report against [s] (paper: 60)")
    ap.add_argument("--theta-max", type=float, default=5.0,
                    help="transport-tilt bound to report against [deg] (paper: 5)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    rows.sort(key=lambda r: int(r["seed"]))
    n = len(rows)
    has_outcome = "task_success" in (rows[0] if rows else {})

    print(f"\n=== {os.path.basename(args.csv)} : {n} trials ===")
    seeds = {int(r["seed"]) for r in rows}
    dup = n - len(seeds)
    print(f"  seeds {min(seeds)}..{max(seeds)}, {len(seeds)} distinct"
          + (f", {dup} duplicate row(s)" if dup else ""))

    # The success criteria differ per task (scripts/trials/trial_driver.py
    # TASK_OUTCOMES), so label the headline with what was actually required
    # instead of describing every batch as if it were Transfer.
    tasks = {r.get("task", "") for r in rows}
    task = tasks.pop() if len(tasks) == 1 else "mixed"
    targets = {r.get("target_obj") or "" for r in rows} - {""}
    target = targets.pop() if len(targets) == 1 else "target object"
    criteria = {
        "transfer": "poured + placed upright in the goal region",
        "move": "placed upright inside the goal region (no pour step)",
        "stir": "vessel upright on the stirrer + stir bar inside it",
    }.get(task, "task terminal condition met")

    print("\nRATES")
    if has_outcome:
        rate("task success", sum(r["task_success"] == "True" for r in rows), n,
             f"   <- headline [{task}]: {criteria}")
    rate("planning success", sum(r["plan_success"] == "True" for r in rows), n)
    rate("trajectory ran to completion", sum(r["execute_success"] == "True" for r in rows), n,
         "   (NOT a task-success criterion)")
    if has_outcome:
        rate(f"{target} upright after release", sum(r["placed_upright"] == "True" for r in rows), n)
        rate(f"{target} inside goal region", sum(r["placed_in_goal"] == "True" for r in rows), n)
        if task != "move":
            rate("pour step executed", sum(r["poured"] == "True" for r in rows), n)
        if any(r.get("aux_check") for r in rows):
            ok = sum("=True" in (r.get("aux_check") or "") for r in rows)
            rate("task-specific extra condition", ok, n)
        errs = [float(r["goal_err_mm"]) for r in rows if (r.get("goal_err_mm") or "").strip()]
        if errs:
            errs.sort()
            med = errs[len(errs) // 2]
            print(f"  placement error from goal centre [mm]: n={len(errs)} "
                  f"min {errs[0]:.1f}  median {med:.1f}  max {errs[-1]:.1f}")

    planned = [r for r in rows if r["plan_success"] == "True"]
    t = floats(planned, "planning_time_s")
    print("\nPLANNING")
    describe("time [s]", t)
    if t:
        rate(f"time <= {args.budget:.0f} s", sum(1 for x in t if x <= args.budget), len(t))
    describe("satisfying particles", floats(planned, "num_satisfying"))

    print("\nCARRIED-VESSEL TILT (executed, simulator measurement)")
    tilt = floats(rows, "max_transport_tilt_deg")
    describe("transport max [deg]", tilt)
    if tilt:
        rate(f"transport <= {args.theta_max:.0f} deg",
             sum(1 for x in tilt if x <= args.theta_max), len(tilt))
    if has_outcome:
        describe("final tilt after release [deg]", floats(rows, "final_tilt_deg"))

    print("\nFAILURES")
    reasons = {}
    for r in rows:
        key = (r["failure_reason"] or "").split("(")[0].strip()
        if key:
            reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {v:>3}x {k}")
    else:
        print("  none recorded")

    if args.logs:
        print("\nLOG AUDIT")
        logs = sorted(glob.glob(os.path.join(args.logs, "tamp_seed*.log")))
        print(f"  planner logs found        : {len(logs)}")
        for label, needle in (
            ("trials that retried", "raised on attempt"),
            ("carry-tilt rejections", "exceeding the"),
            ("go-home fallbacks", "GO_HOME_FALLBACK"),
            ("linear-approach fallbacks", "Linear grasp approach failed"),
        ):
            hit = occ = 0
            for f in logs:
                try:
                    c = open(f, errors="replace").read().count(needle)
                except OSError:
                    continue
                occ += c
                hit += 1 if c else 0
            print(f"  {label:<26} {hit} trial(s), {occ} occurrence(s)")


if __name__ == "__main__":
    main()
