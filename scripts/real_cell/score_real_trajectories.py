#!/usr/bin/env python3
"""Score real-cell Transfer recordings and export the ones fit to replay.

    score_real_trajectories.py RUN_DIR [--beaker-h 0.070] [--export]

RUN_DIR is what record_transfer_real.sh wrote: raw_seed<N>.json, transfer_real.csv,
logs/, SCENE_ENV.txt. For every recording it measures, from the commanded joint
stream and the URDF:

  EEF angle      tool axis to horizontal at finger-close, and its maximum while
                 carrying (the pour itself excluded) -- the rig wants it level
  grasp height   where the fingers close, above the BEAKER'S BASE (bench offset
                 and riser taken from SCENE_ENV.txt), and how far that is below
                 the rim of the REAL beaker (--beaker-h; the model is shorter on
                 purpose, the rim that matters is the real one)
  return         beaker back on its box: centre within the driver's tolerance
                 of the box centre, upright
  pour           lip error at peak tilt, and the peak tilt

A recording is ACCEPTED only if all hold: EEF <= --max-eef deg, rim clearance >=
--min-rim mm, returned upright onto the box, lip error < --max-lip mm, peak tilt
>= --min-tilt deg. --export writes accepted ones as trajectory CSV + meta through
export_trajectory.py and checks each meta names beaker, flask, scale and riser.

Real-cell only: these trajectories are for the robot, not for the paper.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "trials"))
from solve_scan_poses import Arm, URDF  # noqa: E402

GRIP = "gripper_finger1_joint"


def grasp_offset_from_urdf(urdf=URDF):
    """The grasp_frame origin in wrist3_link coordinates, composed along the URDF.

    Read from the URDF, not hardcoded, so a change such as the D435 mount's 4 mm
    TCP shift reaches the score as soon as the URDF carries it. The chain has
    rotations (wrist3->flange and flange->tool0 turn and cancel), so the
    transforms are composed in full rather than their offsets summed.
    """
    from solve_scan_poses import _rpy, _T
    root = ET.parse(urdf).getroot()
    parent_of = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        parent_of[j.find("child").get("link")] = (j.find("parent").get("link"),
                                                  _T(_rpy(*rpy), np.array(xyz)), j.get("type"))
    M, cur = np.eye(4), "grasp_frame"
    while cur != "wrist3_link":
        if cur not in parent_of:
            raise SystemExit("no chain from grasp_frame to wrist3_link in %s" % urdf)
        p, T, typ = parent_of[cur]
        if typ != "fixed":
            raise SystemExit("grasp chain has a %s joint at %s" % (typ, cur))
        M = T @ M
        cur = p
    return M @ np.array([0.0, 0.0, 0.0, 1.0])


def scene_env(run_dir):
    env = {}
    p = os.path.join(run_dir, "SCENE_ENV.txt")
    if os.path.exists(p):
        for line in open(p):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                env[k] = v
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--beaker-h", type=float, default=0.070,
                    help="REAL beaker height [m] (measured 70 mm; the model is shorter)")
    ap.add_argument("--max-eef", type=float, default=10.5)
    ap.add_argument("--min-rim", type=float, default=10.0, help="[mm]")
    ap.add_argument("--max-lip", type=float, default=17.0, help="[mm] ~ the flask mouth radius")
    ap.add_argument("--min-tilt", type=float, default=53.0,
                    help="[deg] peak pour tilt; below ~53 deg a half-full beaker does not pour")
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()

    env = scene_env(a.run_dir)
    bench = float(env.get("SDL_TABLE_Z_M", "0") or 0)
    riser = float(env.get("SDL_BEAKER_RISER_M", "0") or 0)
    base_z = bench + riser                       # the beaker's base, world z
    arm, grasp = Arm(), grasp_offset_from_urdf()
    print("scene: bench %+.3f m, riser %.3f m -> beaker base at z=%.3f; real beaker %.0f mm; "
          "grasp_frame at (%.3f, %.3f, %.3f) in wrist3_link"
          % (bench, riser, base_z, a.beaker_h * 1000, grasp[0], grasp[1], grasp[2]))

    rows = {}
    csvp = os.path.join(a.run_dir, "transfer_real.csv")
    if os.path.exists(csvp):
        for r in csv.DictReader(open(csvp)):
            rows[r["seed"]] = r              # last row per seed = the recorded run

    def elev(q):
        ax = arm.fk(np.asarray(q, float))[:3, 2]
        return abs(np.degrees(np.arcsin(np.clip(ax[2] / np.linalg.norm(ax), -1, 1))))

    print("\n%-5s %-7s %-9s %-10s %-9s %-9s %-7s %-8s %-7s %s"
          % ("seed", "EEF", "EEF carry", "grasp", "rim gap", "return", "upright", "lip", "tilt", "verdict"))
    accepted = []
    for f in sorted(glob.glob(os.path.join(a.run_dir, "raw_seed*.json")),
                    key=lambda p: int(os.path.basename(p)[8:-5])):
        seed = os.path.basename(f)[8:-5]
        d = json.load(open(f))
        cmds = d.get("commands", [])
        r = rows.get(seed)
        if not cmds or r is None or r.get("plan_success") != "True":
            print("%-5s %s" % (seed, "no plan / not recorded"))
            continue
        ev, prev = [], None
        for s in d.get("states_subsampled", []):
            if GRIP not in s["names"]:
                continue
            c = s["positions"][s["names"].index(GRIP)] > 0.05
            if prev is None:
                prev = c
                continue
            if c != prev:
                ev.append((s["t"], "close" if c else "open"))
                prev = c
        close = next((t for t, e in ev if e == "close"), None)
        opn = next((t for t, e in ev if e == "open"), None)
        if close is None or opn is None:
            print("%-5s %s" % (seed, "gripper never closed and reopened"))
            continue
        ops = d.get("operations", [])
        pour = [o["t"] for o in ops if o["op"] == "pouring"]
        pend = min((o["t"] for o in ops if pour and o["t"] > pour[0] and o["op"] != "pouring"),
                   default=None)
        carry = [elev(c["positions"]) for c in cmds if close <= c["t"] <= opn
                 and not (pour and pend and pour[0] <= c["t"] <= pend)]
        q_close = min(cmds, key=lambda c: abs(c["t"] - close))["positions"]
        grasp_h = ((arm.fk(np.asarray(q_close, float)) @ grasp)[2] - base_z) * 1000.0
        rim = a.beaker_h * 1000.0 - grasp_h
        eef, eef_carry = elev(q_close), max(carry) if carry else float("nan")
        try:
            lip, tilt = float(r["pour_lip_err_mm"]), float(r["pour_peak_tilt_deg"])
        except (KeyError, ValueError):
            lip, tilt = float("inf"), 0.0
        returned = r.get("placed_in_goal") == "True"
        upright = r.get("placed_upright") == "True"
        why = []
        if not eef_carry <= a.max_eef:
            why.append("EEF")
        if rim < a.min_rim:
            why.append("rim")
        if not (returned and upright):
            why.append("return")
        if not lip < a.max_lip:
            why.append("lip")
        if tilt < a.min_tilt:
            why.append("tilt")
        print("%-5s %5.1f°  %6.1f°   %6.1f mm %6.1f mm %6s mm %-7s %6.1f  %5.1f°  %s"
              % (seed, eef, eef_carry, grasp_h, rim, r.get("goal_err_mm", "?"),
                 "yes" if upright else "no", lip, tilt, "ACCEPT" if not why else "reject: " + ",".join(why)))
        if not why:
            accepted.append((lip, seed, f))

    accepted.sort()
    print("\naccepted %d; best first: %s" % (len(accepted), ", ".join("seed %s (lip %.1f mm)" % (s, l)
                                                                   for l, s, _ in accepted) or "none"))
    if not a.export:
        return 0
    out = os.path.join(a.run_dir, "deliver")
    os.makedirs(out, exist_ok=True)
    for _, seed, f in accepted:
        base = os.path.join(out, "traj_realcell_seed%s" % seed)
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "trials", "export_trajectory.py"),
                        "--rec", f, "--out-csv", base + ".csv", "--out-meta", base + ".meta.json",
                        "--sim-log-dir", os.path.join(a.run_dir, "logs")], check=True,
                       stdout=subprocess.DEVNULL)
        lay = json.load(open(base + ".meta.json")).get("layout_the_trajectory_assumes") or {}
        missing = [k for k in ("beaker", "flask", "scale", "riser") if k not in lay]
        print("exported %s%s" % (base + ".csv", "  META MISSING %s" % missing if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
