#!/usr/bin/env python3
"""Paired comparison of one outcome between two conditions over repeated runs.

The question it answers: on the SAME layouts, does condition B change outcome
FIELD relative to condition A? E.g. task success with the rendered-perception
World State (B) against the simulator's ground-truth state (A), R1#1.

  paired_outcome.py --a 'DIR/transfer_ground_truth_rep*.csv' \
                    --b 'DIR/transfer_perception_rep*.csv' --field task_success

Repetition r of A is paired with repetition r of B (the rep number is read from
the file name), each pairing is one exact McNemar test on independent layouts,
and pooled rates get Wilson 95% intervals. A pairing whose two files do not hold
the same layouts is refused rather than compared.
"""
import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_planners import wilson, exact_mcnemar  # noqa: E402


# Two runs of one seed record the SAME layout up to physics settling at trial
# start (measured: 0.1 mm); two different seeds differ by centimetres. 5 mm
# separates the two cases with room on both sides.
LAYOUT_TOL_M = 0.005


def _xy(v):
    try:
        return [float(x) for x in v.split(";")]
    except (AttributeError, ValueError):
        return None


def same_layout(a, b):
    for u, v in zip(a, b):
        u, v = _xy(u), _xy(v)
        if u and v and max(abs(x - y) for x, y in zip(u, v)) > LAYOUT_TOL_M:
            return False
    return True


def load(pattern, field):
    reps = {}
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"rep(\d+)\.csv$", f)
        if not m:
            continue
        rows = {}
        for r in csv.DictReader(open(f)):
            s = int(r["seed"])
            if s in rows:
                raise SystemExit("%s: seed %d twice" % (f, s))
            rows[s] = (str(r.get(field, "")).strip() in ("True", "true", "1"),
                       r.get("beaker_xy", ""), r.get("flask_xy", ""))
        reps[int(m.group(1))] = (f, rows)
    return reps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="glob of condition A rep CSVs")
    ap.add_argument("--b", required=True, help="glob of condition B rep CSVs")
    ap.add_argument("--field", default="task_success")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    a = ap.parse_args()

    A, B = load(a.a, a.field), load(a.b, a.field)
    print("# %s: %s vs %s\n" % (a.field, a.label_a, a.label_b))
    if not A or not B:
        print("missing data: %d file(s) for %s, %d for %s" % (len(A), a.label_a, len(B), a.label_b))
        return 1
    ka = sum(v[0] for _, rows in A.values() for v in rows.values())
    na = sum(len(rows) for _, rows in A.values())
    kb = sum(v[0] for _, rows in B.values() for v in rows.values())
    nb = sum(len(rows) for _, rows in B.values())
    print("| condition | %s [Wilson 95%%] |" % a.field)
    print("|---|---|")
    print("| %s | %d/%d = %.1f%% [%.1f, %.1f] |" % ((a.label_a, ka, na, 100.0 * ka / na) + wilson(ka, na)))
    print("| %s | %d/%d = %.1f%% [%.1f, %.1f] |" % ((a.label_b, kb, nb, 100.0 * kb / nb) + wilson(kb, nb)))
    print("\n| pairing | %s | %s | %s only | %s only | exact McNemar p |"
          % (a.label_a, a.label_b, a.label_a, a.label_b))
    print("|---|---|---|---|---|---|")
    for r in sorted(set(A) & set(B)):
        (fa, ra), (fb, rb) = A[r], B[r]
        shared = sorted(set(ra) & set(rb))
        moved = [s for s in shared if not same_layout(ra[s][1:], rb[s][1:])]
        if moved:
            raise SystemExit("rep %d: layouts differ between %s and %s on seeds %s -- not paired"
                             % (r, fa, fb, moved))
        b = sum(1 for s in shared if ra[s][0] and not rb[s][0])
        c = sum(1 for s in shared if rb[s][0] and not ra[s][0])
        print("| rep %d | %d/%d | %d/%d | %d | %d | %.3g |"
              % (r, sum(ra[s][0] for s in shared), len(shared),
                 sum(rb[s][0] for s in shared), len(shared), b, c, exact_mcnemar(b, c)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
