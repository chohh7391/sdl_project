"""Is the residual pour error a correctable offset or tracking scatter?

The lip error is decomposed in the frame the pour uses: 'along' is the lean
direction normalize(flask_xy), so negative means the lip stopped short on the
near side; 'perp' is lateral. A mean far from zero with a small spread relative
to it is an offset worth correcting; a mean near zero with a large spread is
tracking error, and no constant correction helps.
"""
import csv, math, statistics as st, sys

rows = [r for r in csv.DictReader(open(sys.argv[1]))
        if (r.get("pour_lip_err_along_mm") or "").strip() not in ("", "None")]
a = [float(r["pour_lip_err_along_mm"]) for r in rows]
p = [float(r["pour_lip_err_perp_mm"]) for r in rows]
e = [float(r["pour_lip_err_mm"]) for r in rows]
n = len(rows)
print("pours: %d" % n)


def describe(name, v):
    m, sd = st.mean(v), st.pstdev(v)
    se = sd / math.sqrt(len(v))
    # |mean| / sd: how much of the error is a shift rather than spread
    print("  %-6s mean %+6.1f  sd %5.1f  median %+6.1f  "
          "95%% CI of the mean [%+.1f, %+.1f]  |mean|/sd %.2f  negative %d/%d"
          % (name, m, sd, st.median(v), m - 1.96 * se, m + 1.96 * se,
             abs(m) / sd if sd else float("inf"),
             sum(1 for x in v if x < 0), len(v)))
    return m, sd


ma, sa = describe("along", a)
mp, sp = describe("perp", p)
print("  |err|  median %5.1f  p90 %5.1f  within 17 mm %d/%d = %.0f%%"
      % (st.median(e), sorted(e)[max(0, int(0.9 * n) - 1)],
         sum(1 for x in e if x <= 17), n, 100 * sum(1 for x in e if x <= 17) / n))

print("\ncounterfactual: subtract the mean 'along' offset from every pour")
corr = [math.hypot(ai - ma, pi - mp) for ai, pi in zip(a, p)]
print("  |err| median %5.1f -> %5.1f   within 17 mm %d/%d -> %d/%d"
      % (st.median(e), st.median(corr),
         sum(1 for x in e if x <= 17), n, sum(1 for x in corr if x <= 17), n))
print("\nverdict: %s" % (
    "a correctable offset dominates" if abs(ma) > sa
    else "scatter dominates; no constant correction recovers much"))
