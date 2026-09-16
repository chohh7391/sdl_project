#!/usr/bin/env python3
"""Solve the L2 recovery scan's viewpoints, and score how much of the cell they see.

The recovery ladder's third rung sweeps the arm so a wrist-mounted camera can
find an object the two fixed cameras lost. Its viewpoints were originally
written as joint offsets around the recovery home pose, which is easy to write
and impossible to check by eye: forward kinematics puts the optical axis of
every one of those legs through the bench within 0.27 m of the base, while the
vessels sit at 0.46-0.65 m, so the sweep looked at bare bench.

This script states the requirement geometrically instead -- put the camera at a
chosen height above a chosen bench point with the optical axis vertical -- and
solves the six joint angles for it, then reports what fraction of a set of real
layouts each candidate table actually images. Run it whenever the mount, the
intrinsics or the layout distribution changes, and paste the printed table into
tamp_server.TAMPServer.SCAN_POSES (or export SDL_RECOVERY_SCAN_POSES).

    # score the shipped table and solve a new one over the default region
    scripts/trials/solve_scan_poses.py --layouts logs/

Layouts are read from the sim logs' "beaker xy=(...)" lines, so the score is
against the layouts actually evaluated rather than an assumed distribution.
"""
import argparse
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - scoring still works without a solver
    least_squares = None

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
URDF = os.path.join(PROJ, "TAMP/tamp/content/assets/robot/dcp_description"
                          "/urdf/robot/fr5_ag95.urdf")

# The wrist camera as isaacsim Task mounts it: SDL_WRIST_CAMERA_XYZ in
# wrist3_link's frame, turned 180 deg about X so a USD camera (which images
# along its own -Z) looks down the wrist's +Z tool axis.
MOUNT_XYZ = np.array([0.06, 0.0, 0.05])
# utils/camera.py CameraInfo: D435 colour at 1280x720.
FX, FY, CX, CY, W, H = 923.7, 925.8, 640.0, 360.0, 1280, 720
# tamp_server.TAMPServer.RECOVERY_HOME
HOME = np.array([0.0, -1.05, -2.18, -1.57, 1.57, 0.0])
# Vessel tag heights above the bench; a tag is what actually has to be imaged.
OBJ_Z = {"beaker": 0.036, "flask": 0.080}


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                              np.sin(p), np.cos(y), np.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _T(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _axis_angle(a, th):
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class Arm:
    """base_link -> wrist3_link forward kinematics straight from the URDF."""

    def __init__(self, urdf=URDF):
        root = ET.parse(urdf).getroot()
        joints = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
            ax, lim = j.find("axis"), j.find("limit")
            joints[j.get("name")] = dict(
                parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                type=j.get("type"), origin=_T(_rpy(*rpy), np.array(xyz)),
                axis=(np.array([float(v) for v in ax.get("xyz").split()])
                      if ax is not None else np.array([0., 0., 1.])),
                lo=float(lim.get("lower")) if lim is not None and lim.get("lower") else -np.pi,
                hi=float(lim.get("upper")) if lim is not None and lim.get("upper") else np.pi)
        by_child = {v["child"]: v for v in joints.values()}
        self.chain, cur = [], "wrist3_link"
        while cur in by_child:
            j = by_child[cur]
            self.chain.append(j)
            cur = j["parent"]
        self.chain.reverse()
        self.lo = np.array([j["lo"] for j in self.chain])
        self.hi = np.array([j["hi"] for j in self.chain])
        self.cam_local = _T(np.diag([1., -1., -1.]), MOUNT_XYZ)

    def fk(self, q):
        M, i = np.eye(4), 0
        for j in self.chain:
            M = M @ j["origin"]
            if j["type"] in ("revolute", "continuous"):
                M = M @ _T(_axis_angle(j["axis"], q[i]), np.zeros(3))
                i += 1
        return M

    def camera(self, q):
        return self.fk(np.asarray(q, dtype=float)) @ self.cam_local

    def look_at(self, q):
        """Where this configuration's optical axis meets the bench, or None."""
        M = self.camera(q)
        t, z = M[:3, 3], -M[:3, 2]
        if z[2] > -1e-6:
            return None
        return t + (-t[2] / z[2]) * z

    def sees(self, q, point):
        """Is `point` (world xyz) inside the wrist camera's image at `q`?"""
        M = self.camera(q)
        R, t = M[:3, :3], M[:3, 3]
        v = R.T @ (np.asarray(point, dtype=float) - t)
        depth = -v[2]                      # USD camera images along its own -Z
        if depth <= 1e-6:
            return False
        u = FX * (v[0] / depth) + CX
        w = -FY * (v[1] / depth) + CY
        return 0 <= u < W and 0 <= w < H


def solve_pose(arm, target_xy, height, seeds=24, rng_seed=1):
    """Joints putting the camera `height` above `target_xy`, looking straight down."""
    if least_squares is None:
        raise SystemExit("scipy is required to solve poses (scoring works without it)")

    def residual(q):
        M = arm.camera(q)
        t, z = M[:3, 3], -M[:3, 2]
        geom = np.array([t[2] - height, z[0], z[1],
                         t[0] - target_xy[0], t[1] - target_xy[1], z[2] + 1.0]) * 3.0
        # Weak regularisers: stay near the home pose, and keep the wrist roll
        # small -- j6 turns the camera about its own optical axis, so it changes
        # which way the 16:9 frame lies but not whether a point is reachable by
        # the sweep. Left free it wanders to ~167 deg for no gain.
        reg = np.concatenate([[0.05 * q[5]], 0.02 * (q - HOME)])
        return np.concatenate([geom, reg]), geom

    def f(q):
        return residual(q)[0]

    rng = np.random.default_rng(rng_seed)
    best = None
    for k in range(seeds):
        q0 = np.clip(HOME + (0 if k == 0 else rng.normal(0, 0.6, 6)),
                     arm.lo + 1e-3, arm.hi - 1e-3)
        r = least_squares(f, q0, bounds=(arm.lo, arm.hi), xtol=1e-12, ftol=1e-12)
        err = np.linalg.norm(residual(r.x)[1])
        if err < 0.02:
            cost = np.linalg.norm(r.x - HOME) + abs(r.x[5])
            if best is None or cost < best[0]:
                best = (cost, r.x, err)
    return best


def read_layouts(path):
    """Vessel xy per seed, from the trial sim logs (newest log per seed wins)."""
    files = sorted(glob.glob(os.path.join(path, "sim_seed*.log")))
    best = {}
    for f in files:
        m = re.search(r"sim_seed(\d+)_", os.path.basename(f))
        if not m:
            continue
        seed = int(m.group(1))
        txt = open(f, errors="ignore").read()
        b = re.search(r"beaker\s+xy=\(([-\d.]+),([-\d.]+)\)", txt)
        fl = re.search(r"flask\s+xy=\(([-\d.]+),([-\d.]+)\)", txt)
        if not (b and fl):
            continue
        t = os.path.getmtime(f)
        if seed not in best or t > best[seed][0]:
            best[seed] = (t, (float(b.group(1)), float(b.group(2))),
                          (float(fl.group(1)), float(fl.group(2))))
    return [dict(seed=s, beaker=v[1], flask=v[2]) for s, v in sorted(best.items())]


def score(arm, legs, layouts, label):
    counts = {"beaker": 0, "flask": 0}
    both = 0
    for L in layouts:
        hit = {}
        for name in ("beaker", "flask"):
            p = list(L[name]) + [OBJ_Z[name]]
            hit[name] = any(arm.sees(q, p) for q in legs)
            counts[name] += hit[name]
        both += hit["beaker"] and hit["flask"]
    n = len(layouts)
    print("%-28s beaker %2d/%d  flask %2d/%d  both %2d/%d"
          % (label, counts["beaker"], n, counts["flask"], n, both, n))
    return both


SHIPPED_OFFSET_LEGS = [
    [HOME[0] + dj1, HOME[1] + dj2, HOME[2], HOME[3], HOME[4], HOME[5]]
    for dj2 in (0.0, 0.2) for dj1 in (-1.57, 1.57, 0.0)
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=URDF)
    ap.add_argument("--height", type=float, default=0.45,
                    help="camera height above the bench [m]")
    ap.add_argument("--targets", default="0.48:0.12,0.48:0.28,0.48:0.40",
                    help="bench points to look at, x:y comma separated")
    ap.add_argument("--layouts", default=os.path.join(PROJ, "scripts/trials/logs"),
                    help="directory of sim_seed*.log files to score against")
    args = ap.parse_args()

    arm = Arm(args.urdf)
    targets = [tuple(float(v) for v in t.split(":")) for t in args.targets.split(",")]

    print("solving %d viewpoints at h=%.2f m" % (len(targets), args.height))
    legs = []
    for t in targets:
        got = solve_pose(arm, t, args.height)
        if got is None:
            print("  (%.2f, %.2f): NO SOLUTION" % t)
            continue
        _, q, err = got
        hit = arm.look_at(q)
        print("  (%.2f, %.2f): residual %.5f, travel %.2f rad, axis meets bench at (%.3f, %.3f)"
              % (t[0], t[1], err, np.linalg.norm(q - HOME), hit[0], hit[1]))
        legs.append(q)

    layouts = read_layouts(args.layouts)
    if layouts:
        print("\nscored against %d layouts from %s" % (len(layouts), args.layouts))
        score(arm, [np.array(q) for q in SHIPPED_OFFSET_LEGS], layouts,
              "j1/j2 offsets (superseded)")
        score(arm, legs, layouts, "solved viewpoints")
    else:
        print("\nno sim_seed*.log layouts found under %s; not scored" % args.layouts)

    print("\nSDL_RECOVERY_SCAN_POSES=" + ";".join(
        ",".join("%.4f" % v for v in q) for q in legs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
