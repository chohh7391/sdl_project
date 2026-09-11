"""Summarise the perception campaign: localization rate (Wilson 95% CI),
per-camera visibility, and the pose-error distribution over localized seeds."""
import csv, math, sys, statistics as st

def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h) * 100, min(1.0, c + h) * 100)

rows = list(csv.DictReader(open(sys.argv[1])))
seeds = sorted({r["seed"] for r in rows}, key=int)
print("seeds: %d   rows: %d\n" % (len(seeds), len(rows)))

for obj in ("beaker", "flask"):
    rs = [r for r in rows if r["object"] == obj]
    n = len(rs)
    pub = [r for r in rs if r["published"] == "1"]
    lo, hi = wilson(len(pub), n)
    two = sum(1 for r in rs if int(r["n_cams"]) == 2)
    one = sum(1 for r in rs if int(r["n_cams"]) == 1)
    zero = sum(1 for r in rs if int(r["n_cams"]) == 0)
    c1 = sum(1 for r in rs if float(r["cam1_rate"]) > 0.1)
    c2 = sum(1 for r in rs if float(r["cam2_rate"]) > 0.1)
    print("== %s  (n=%d)" % (obj, n))
    print("   localized      %d/%d = %.1f%%  Wilson 95%% CI [%.1f, %.1f]"
          % (len(pub), n, 100.0 * len(pub) / n, lo, hi))
    print("   cameras seeing it: 2 -> %d,  1 -> %d,  0 -> %d" % (two, one, zero))
    print("   camera_1 sees it %d/%d,  camera_2 %d/%d" % (c1, n, c2, n))
    for col, unit in (("dxy_mm", "mm"), ("dz_mm", "mm"), ("dyaw_deg", "deg")):
        v = [float(r[col]) for r in pub if r[col] not in ("nan", "")]
        if not v:
            continue
        a = [abs(x) for x in v]
        print("   |%-9s| median %6.2f  mean %6.2f  p90 %6.2f  max %6.2f %s"
              % (col, st.median(a), st.mean(a),
                 sorted(a)[max(0, int(0.9 * len(a)) - 1)], max(a), unit))
    # error split by how many cameras contributed
    for k in (2, 1):
        v = [abs(float(r["dxy_mm"])) for r in pub
             if int(r["n_cams"]) == k and r["dxy_mm"] not in ("nan", "")]
        w = [abs(float(r["dz_mm"])) for r in pub
             if int(r["n_cams"]) == k and r["dz_mm"] not in ("nan", "")]
        if v:
            print("   %d-camera seeds (n=%d): |dxy| median %.2f max %.2f | "
                  "|dz| median %.2f max %.2f mm"
                  % (k, len(v), st.median(v), max(v), st.median(w), max(w)))
    miss = [r["seed"] for r in rs if r["published"] != "1"]
    if miss:
        print("   NOT localized on seeds: %s" % ", ".join(miss))
    print()

both = 0
for s in seeds:
    rs = [r for r in rows if r["seed"] == s]
    if len(rs) == 2 and all(r["published"] == "1" for r in rs):
        both += 1
lo, hi = wilson(both, len(seeds))
print("both vessels localized on %d/%d seeds = %.1f%%  Wilson 95%% CI [%.1f, %.1f]"
      % (both, len(seeds), 100.0 * both / len(seeds), lo, hi))
print("(transfer needs both; move and stir need only the vessel they act on)")
