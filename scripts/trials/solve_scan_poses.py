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

# The wrist camera's mount: the D435 optical frame (x right, y down, z along the
# view) in wrist3_link, from the one file the simulator and the perception
# launch also read.
MOUNT_FILE = os.path.join(PROJ, "perception/perception_manager/config/wrist_camera.yaml")
# utils/camera.py CameraInfo: D435 colour at 1280x720.
FX, FY, CX, CY, W, H = 923.7, 925.8, 640.0, 360.0, 1280, 720
# tamp_server.TAMPServer.RECOVERY_HOME
HOME = np.array([0.0, -1.05, -2.18, -1.57, 1.57, 0.0])
# Half the tag plate's edge: the whole plate has to be in the image, with a margin.
TAG_HALF_M = 0.05
EDGE_MARGIN_PX = 20


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
        import yaml
        with open(MOUNT_FILE) as fh:
            m = yaml.safe_load(fh)
        self.cam_local = _T(_rpy(*[float(v) for v in m["rpy"]]),
                            np.array([float(v) for v in m["xyz"]]))

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
        t, z = M[:3, 3], M[:3, 2]          # optical frame: +z along the view
        if z[2] > -1e-6:
            return None
        return t + (-t[2] / z[2]) * z

    def sees(self, q, point, margin_px=0.0):
        """Is `point` (world xyz) inside the wrist camera's image at `q`?"""
        M = self.camera(q)
        R, t = M[:3, :3], M[:3, 3]
        v = R.T @ (np.asarray(point, dtype=float) - t)
        depth = v[2]
        if depth <= 1e-6:
            return False
        u = FX * (v[0] / depth) + CX
        w = FY * (v[1] / depth) + CY
        return margin_px <= u < W - margin_px and margin_px <= w < H - margin_px

    def sees_tag(self, q, centre):
        """The whole (level) tag plate centred at `centre` inside the image."""
        cx, cy, cz = centre
        return all(self.sees(q, (cx + dx, cy + dy, cz), EDGE_MARGIN_PX)
                   for dx in (-TAG_HALF_M, TAG_HALF_M) for dy in (-TAG_HALF_M, TAG_HALF_M))


def solve_pose(arm, target_xy, height, seeds=24, rng_seed=1):
    """Joints putting the camera at world z `height` above `target_xy`, looking straight down."""
    if least_squares is None:
        raise SystemExit("scipy is required to solve poses (scoring works without it)")

    def residual(q):
        M = arm.camera(q)
        t, z = M[:3, 3], M[:3, 2]
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


def solve_look(arm, target, dist, max_tilt_deg=50.0, seeds=48, rng_seed=1):
    """Joints aiming the optical axis at `target` (world xyz) from `dist` away.

    The camera sits 0.22 m out from the wrist, so a straight-down view high
    enough to take in the tag region is out of the FR5's reach; an oblique view
    is not. The tilt of the axis from vertical is free up to `max_tilt_deg` (a
    level tag stays readable at that angle: its short side still spans ~150 px
    at 0.35 m), and the solve prefers configurations near the home pose.
    """
    if least_squares is None:
        raise SystemExit("scipy is required to solve poses (scoring works without it)")
    target = np.asarray(target, dtype=float)
    cos_max = np.cos(np.radians(max_tilt_deg))

    def residual(q):
        M = arm.camera(q)
        t, z = M[:3, 3], M[:3, 2]
        d = target - t
        along = float(d @ z)
        perp = d - along * z
        tilt = max(0.0, (-z[2]) * -1.0 + cos_max)   # >0 once the axis is flatter than max
        geom = np.concatenate([perp * 5.0, [(along - dist) * 2.0, tilt * 5.0]])
        reg = np.concatenate([[0.05 * q[5]], 0.02 * (q - HOME)])
        return np.concatenate([geom, reg]), geom

    rng = np.random.default_rng(rng_seed)
    best = None
    for k in range(seeds):
        q0 = np.clip(HOME + (0 if k == 0 else rng.normal(0, 0.7, 6)),
                     arm.lo + 1e-3, arm.hi - 1e-3)
        r = least_squares(lambda q: residual(q)[0], q0, bounds=(arm.lo, arm.hi),
                          xtol=1e-12, ftol=1e-12)
        err = np.linalg.norm(residual(r.x)[1])
        if err < 0.01:
            cost = np.linalg.norm(r.x - HOME) + abs(r.x[5])
            if best is None or cost < best[0]:
                best = (cost, r.x, err)
    return best


def read_layouts(path):
    """Tag plate centres per seed, from the trial sim logs (newest log per seed wins)."""
    files = sorted(glob.glob(os.path.join(path, "sim_seed*.log")))
    best = {}
    for f in files:
        m = re.search(r"sim_seed(\d+)_", os.path.basename(f))
        if not m:
            continue
        seed = int(m.group(1))
        tags = {}
        for t in re.finditer(r"tag /World/(beaker|flask)/visual/apriltag_\d+ "
                             r"world=\(([-\d.]+),([-\d.]+),([-\d.]+)\)",
                             open(f, errors="ignore").read()):
            tags[t.group(1)] = tuple(float(t.group(i)) for i in (2, 3, 4))
        if len(tags) < 2:
            continue
        mt = os.path.getmtime(f)
        if seed not in best or mt > best[seed][0]:
            best[seed] = (mt, tags)
    return [dict(seed=s, **v[1]) for s, v in sorted(best.items())]


def score(arm, legs, layouts, label):
    counts = {"beaker": 0, "flask": 0}
    both = 0
    for L in layouts:
        hit = {}
        for name in ("beaker", "flask"):
            hit[name] = any(arm.sees_tag(q, L[name]) for q in legs)
            counts[name] += hit[name]
        both += hit["beaker"] and hit["flask"]
    n = len(layouts)
    print("%-28s beaker tag %2d/%d  flask tag %2d/%d  both %2d/%d"
          % (label, counts["beaker"], n, counts["flask"], n, both, n))
    return both


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=URDF)
    ap.add_argument("--height", type=float, default=0.62,
                    help="camera height above the bench [m] (the tags are at 0.18 m)")
    ap.add_argument("--targets", default="0.48:0.12,0.48:0.28,0.48:0.40",
                    help="bench points to look at, x:y comma separated")
    ap.add_argument("--layouts", default=os.path.join(PROJ, "scripts/trials/logs"),
                    help="directory of sim_seed*.log files to score against")
    ap.add_argument("--look", default="",
                    help="aim at tag-plane points instead, x:y comma separated (see --dist)")
    ap.add_argument("--dist", type=float, default=0.35, help="camera-to-target distance for --look [m]")
    ap.add_argument("--tag-z", type=float, default=0.18, help="height of the tag plane for --look [m]")
    ap.add_argument("--max-tilt", type=float, default=50.0, help="largest axis tilt from vertical for --look [deg]")
    args = ap.parse_args()

    arm = Arm(args.urdf)
    if args.look:
        looks = [tuple(float(v) for v in t.split(":")) for t in args.look.split(",")]
        print("aiming %d viewpoints from %.2f m at the tag plane z=%.2f (tilt <= %.0f deg)"
              % (len(looks), args.dist, args.tag_z, args.max_tilt))
        legs = []
        for t in looks:
            got = solve_look(arm, (t[0], t[1], args.tag_z), args.dist, args.max_tilt)
            if got is None:
                print("  (%.2f, %.2f): NO SOLUTION" % t)
                continue
            _, q, err = got
            z = arm.camera(q)[:3, 2]
            print("  (%.2f, %.2f): residual %.5f, travel %.2f rad, axis tilt %.1f deg"
                  % (t[0], t[1], err, np.linalg.norm(q - HOME), np.degrees(np.arccos(-z[2]))))
            legs.append(q)
        layouts = read_layouts(args.layouts)
        if layouts:
            print("\nscored against %d layouts from %s" % (len(layouts), args.layouts))
            score(arm, legs, layouts, "aimed viewpoints")
        print("\nSDL_RECOVERY_SCAN_POSES=" + ";".join(
            ",".join("%.4f" % v for v in q) for q in legs))
        return 0

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
        score(arm, legs, layouts, "solved viewpoints")
    else:
        print("\nno sim_seed*.log layouts found under %s; not scored" % args.layouts)

    print("\nSDL_RECOVERY_SCAN_POSES=" + ";".join(
        ",".join("%.4f" % v for v in q) for q in legs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
