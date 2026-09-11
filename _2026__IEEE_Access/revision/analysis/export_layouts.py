#!/usr/bin/env python3
"""Reproduce the Isaac Sim seeded layouts offline, and verify the reproduction.

The paired planner comparison needs the PDDLStream baseline to face the SAME
object layouts as cuTAMP; the baseline runner samples its own, which would make
an exact McNemar test meaningless. Task._randomize_layout is a pure function of
SDL_SEED (numpy PCG64, fixed draw order), so the layouts can be regenerated
without launching Isaac Sim -- but only a verified reproduction may be used, so
--verify replays the layouts the simulator actually logged and requires a
bit-level match on every seed before the export is written.

Constants mirror isaacsim/scripts/standalone/task.py; --verify is what catches
any drift between the two.
"""
import argparse, glob, json, os, re, sys
import numpy as np

NOMINAL_XY = {
    "beaker":  (0.49, 0.17475),
    "flask":   (0.48383, 0.33166),
    "magnet":  (-0.3, 0.416),
    "box":     (-0.12621, -0.57484),
    "stirrer": (-0.15, 0.6),
}
NOMINAL_Z = {
    "beaker": 0.0675, "flask": 0.06, "magnet": 0.0175, "box": 0.04,
    "stirrer": 0.045,
}
FOOTPRINT = {
    "beaker": (0.05, 0.05), "flask": (0.07, 0.07), "magnet": (0.045, 0.045),
    "box": (0.108, 0.108), "stirrer": (0.18, 0.18),
}
ORDER = ["beaker", "flask", "magnet", "box", "stirrer"]
NOMINAL_HOME = np.array([0.0, -1.05, -2.18, -1.57, 1.57, 0.0])
POS_JITTER_M = 0.05
YAW_RANGE_RAD = np.pi
ROBOT_JITTER_RAD = np.deg2rad(10.0)
OVERLAP_MARGIN_M = 0.01
BASE_KEEPOUT_M = 0.15
WORKSPACE_HALF_M = 0.70
MAX_RESAMPLE = 300


def bounding_radius(dims_xy):
    return 0.5 * float(np.hypot(dims_xy[0], dims_xy[1]))


def layout(seed):
    """The layout Task._randomize_layout produces for `seed`."""
    rng = np.random.default_rng(seed)
    home = NOMINAL_HOME + rng.uniform(-ROBOT_JITTER_RAD, ROBOT_JITTER_RAD, size=6)
    placed, out = {}, {}
    for name in ORDER:
        nominal = NOMINAL_XY[name]
        radius = bounding_radius(FOOTPRINT[name])
        chosen = None
        for _ in range(MAX_RESAMPLE):
            dx, dy = rng.uniform(-POS_JITTER_M, POS_JITTER_M, size=2)
            x, y = nominal[0] + dx, nominal[1] + dy
            if abs(x) > WORKSPACE_HALF_M or abs(y) > WORKSPACE_HALF_M:
                continue
            if np.hypot(x, y) < BASE_KEEPOUT_M + radius:
                continue
            if any(np.hypot(x - px, y - py) < radius + pr + OVERLAP_MARGIN_M
                   for (px, py, pr) in placed.values()):
                continue
            chosen = (x, y)
            break
        if chosen is None:
            chosen = (float(nominal[0]), float(nominal[1]))
        yaw = float(rng.uniform(-YAW_RANGE_RAD, YAW_RANGE_RAD))
        out[name] = {"xy": [float(chosen[0]), float(chosen[1])],
                     "z": NOMINAL_Z[name], "yaw_rad": yaw,
                     "yaw_deg": float(np.rad2deg(yaw))}
        placed[name] = (chosen[0], chosen[1], radius)
    return {"seed": seed, "home_arm_rad": [float(v) for v in home],
            "objects": out}


LOG_RE = re.compile(
    r"\[Task\]\s+(beaker|flask|magnet|box|stirrer)\s+xy=\(([-\d.]+),([-\d.]+)\)"
    r"\s+yaw=([-\d.]+)deg")
HOME_RE = re.compile(r"\[Task\]\s+home_arm\(rad\) = \[([-\d.,\s]+)\]")


def parse_log(path):
    objs, home = {}, None
    for ln in open(path, errors="ignore"):
        m = LOG_RE.search(ln)
        if m:
            objs[m.group(1)] = (float(m.group(2)), float(m.group(3)),
                                float(m.group(4)))
        m = HOME_RE.search(ln)
        if m and home is None:
            home = [float(v) for v in m.group(1).split(",")]
    return objs, home


def verify(log_glob):
    """Replay every logged layout and compare, to 1e-4 -- the logged precision."""
    paths = sorted(glob.glob(log_glob))
    if not paths:
        print("verify: no logs matched %s" % log_glob)
        return False
    ok, checked = True, 0
    for p in paths:
        m = re.search(r"s(?:eed)?(\d+)", os.path.basename(p))
        if not m:
            continue
        seed = int(m.group(1))
        logged, home = parse_log(p)
        if not logged:
            continue
        mine = layout(seed)
        checked += 1
        for name, (lx, ly, lyaw) in logged.items():
            g = mine["objects"][name]
            if (abs(g["xy"][0] - lx) > 1e-4 or abs(g["xy"][1] - ly) > 1e-4
                    or abs(((g["yaw_deg"] - lyaw + 180) % 360) - 180) > 1e-1):
                print("MISMATCH seed %d %s: logged (%.4f,%.4f,%.1f) "
                      "reproduced (%.4f,%.4f,%.1f)"
                      % (seed, name, lx, ly, lyaw,
                         g["xy"][0], g["xy"][1], g["yaw_deg"]))
                ok = False
        if home is not None:
            for i, v in enumerate(home):
                if abs(mine["home_arm_rad"][i] - v) > 1e-4:
                    print("MISMATCH seed %d home j%d: logged %.4f reproduced %.4f"
                          % (seed, i + 1, v, mine["home_arm_rad"][i]))
                    ok = False
    print("verify: %d logged layouts checked, %s"
          % (checked, "all identical" if ok else "MISMATCHES above"))
    return ok and checked > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-29")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", default=None,
                    help="glob of Isaac Sim logs to replay and compare")
    a = ap.parse_args()

    if a.verify and not verify(a.verify):
        print("refusing to export an unverified reproduction")
        return 1
    if a.seeds.count("-") == 1 and ":" not in a.seeds:
        lo, hi = (int(v) for v in a.seeds.split("-"))
        seeds = list(range(lo, hi + 1))
    else:
        seeds = [int(v) for v in a.seeds.replace(",", " ").split()]
    data = {"source": "isaacsim/scripts/standalone/task.py _randomize_layout",
            "layouts": [layout(s) for s in seeds]}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(data, open(a.out, "w"), indent=1)
        print("wrote %d layouts to %s" % (len(seeds), a.out))
    else:
        print(json.dumps(data, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
