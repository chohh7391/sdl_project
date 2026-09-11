#!/usr/bin/env python3
"""Paired cuTAMP vs PDDLStream comparison on shared seeds.

Proportions get Wilson 95% intervals, the paired planner comparison uses the
EXACT McNemar test on the seeds both planners attempted, and time-to-solution
is censoring-aware: a seed that used its whole budget without a plan is a
censored observation at the budget, not a missing value and not a long time, so
the summary is a Kaplan-Meier curve plus the restricted mean over the budget.
Reporting only the successful trials' mean planning time would compare the easy
subsets of two different planners.
"""
import argparse, csv, math, sys


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h) * 100, min(1.0, c + h) * 100


def binom_cdf_upper(k, n, p=0.5):
    """P(X >= k) for X ~ Bin(n, p), by exact summation."""
    tot = 0.0
    for i in range(k, n + 1):
        tot += math.comb(n, i) * p ** i * (1 - p) ** (n - i)
    return tot


def exact_mcnemar(b, c):
    """Two-sided exact McNemar on discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    p = 2.0 * binom_cdf_upper(k, n, 0.5)
    return min(1.0, p)


def km_rmst(times, events, horizon):
    """Kaplan-Meier survival of 'not yet solved' and the restricted mean.

    times/events are per-seed: time is when a plan was found or the budget ran
    out; event=1 means solved. The restricted mean time to solution over
    [0, horizon] is the area under the survival curve, which is finite even
    when some seeds never solve.
    """
    order = sorted(range(len(times)), key=lambda i: times[i])
    n_at_risk = len(times)
    surv = 1.0
    area = 0.0
    prev_t = 0.0
    curve = [(0.0, 1.0)]
    i = 0
    while i < len(order):
        t = times[order[i]]
        d = 0
        cens = 0
        while i < len(order) and times[order[i]] == t:
            if events[order[i]]:
                d += 1
            else:
                cens += 1
            i += 1
        area += surv * (min(t, horizon) - prev_t)
        prev_t = min(t, horizon)
        if n_at_risk > 0 and d > 0:
            surv *= (1.0 - d / n_at_risk)
            curve.append((t, surv))
        n_at_risk -= (d + cens)
    area += surv * max(0.0, horizon - prev_t)
    return area, surv, curve


def load_cutamp(path):
    out = {}
    for r in csv.DictReader(open(path)):
        s = r.get("seed")
        if s is None or s == "":
            continue
        ok = str(r.get("plan_success", "")).strip().lower() == "true"
        t = r.get("planning_time_s", "")
        out[int(s)] = (ok, float(t) if t not in ("", None) else None,
                       r.get("failure_reason", ""))
    return out


def load_pddl(path):
    out = {}
    for r in csv.DictReader(open(path)):
        ok = str(r.get("plan_success", "")).strip() in ("1", "True", "true")
        t = r.get("planning_time_s", "")
        out[int(r["seed"])] = (ok, float(t) if t else None,
                              r.get("failure_reason", ""))
    return out


def censor(res, budget):
    """Hold every planner to the SAME budget.

    cuTAMP's recorded runs had no wall-clock cutoff -- its slowest transfer
    seed took 167.58 s -- while the baseline was stopped at its --max-time. A
    success that arrived after the budget is not a success under that budget,
    so it becomes an unsolved, censored observation for both planners alike.
    """
    out = {}
    for s, (ok, t, why) in res.items():
        if ok and t is not None and t > budget:
            out[s] = (False, budget, "exceeded_%.0fs_budget" % budget)
        else:
            out[s] = (ok, t, why)
    return out


def report(name, res, budget):
    n = len(res)
    k = sum(1 for v in res.values() if v[0])
    lo, hi = wilson(k, n)
    times, events = [], []
    for ok, t, _ in res.values():
        if ok and t is not None:
            times.append(min(t, budget)); events.append(1)
        else:
            times.append(budget); events.append(0)
    rmts, surv, _ = km_rmst(times, events, budget)
    solved = sorted(t for ok, t, _ in res.values() if ok and t is not None)
    print("%-11s success %2d/%-2d = %5.1f%%  Wilson 95%% CI [%.1f, %.1f]"
          % (name, k, n, 100.0 * k / n, lo, hi))
    print("%-11s restricted mean time to solution over %.0f s: %.2f s  "
          "(unsolved fraction at the budget: %.1f%%)"
          % ("", budget, rmts, 100.0 * surv))
    if solved:
        print("%-11s solved-only planning time: median %.2f s  min %.2f  max %.2f"
              % ("", solved[len(solved) // 2], solved[0], solved[-1]))
    return k, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutamp", required=True)
    ap.add_argument("--pddlstream", required=True)
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--label", default="transfer")
    a = ap.parse_args()

    cu = load_cutamp(a.cutamp)
    pd = load_pddl(a.pddlstream)
    shared = sorted(set(cu) & set(pd))
    print("=== %s: %d cuTAMP seeds, %d PDDLStream seeds, %d shared\n"
          % (a.label, len(cu), len(pd), len(shared)))
    if not shared:
        print("no shared seeds -- the runs are not paired"); return 1

    cu_s = censor({s: cu[s] for s in shared}, a.budget)
    pd_s = censor({s: pd[s] for s in shared}, a.budget)
    late = sum(1 for s in shared
               if cu[s][0] and cu[s][1] is not None and cu[s][1] > a.budget)
    if late:
        print("cuTAMP solved %d of these seeds only AFTER the %.0f s budget; "
              "they count as unsolved here\n" % (late, a.budget))
    report("cuTAMP", cu_s, a.budget)
    print()
    report("PDDLStream", pd_s, a.budget)

    b = sum(1 for s in shared if cu_s[s][0] and not pd_s[s][0])
    c = sum(1 for s in shared if pd_s[s][0] and not cu_s[s][0])
    both = sum(1 for s in shared if cu_s[s][0] and pd_s[s][0])
    neither = sum(1 for s in shared if not cu_s[s][0] and not pd_s[s][0])
    p = exact_mcnemar(b, c)
    print("\npaired on %d seeds: both %d, cuTAMP only %d, PDDLStream only %d, "
          "neither %d" % (len(shared), both, b, c, neither))
    print("exact McNemar (two-sided) on the %d discordant pairs: p = %.6g"
          % (b + c, p))
    only = [s for s in shared if cu_s[s][0] and not pd_s[s][0]]
    if only:
        print("cuTAMP-only seeds: %s" % ", ".join(str(s) for s in only))
    only = [s for s in shared if pd_s[s][0] and not cu_s[s][0]]
    if only:
        print("PDDLStream-only seeds: %s" % ", ".join(str(s) for s in only))
    return 0


if __name__ == "__main__":
    sys.exit(main())
