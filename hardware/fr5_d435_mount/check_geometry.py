#!/usr/bin/env python3
"""Check the exported mount.stl against the things the design claims.

Numpy only, no CAD kernel: every check below is a statement about where
material is or is not, which is all that matters for "does it fit". Run it
after changing a parameter in fr5_d435_mount.scad and re-exporting.

    python3 check_geometry.py [mount_color.stl] [lens offset, mm]
"""
import re
import struct
import sys

import numpy as np

# Kept in step with fr5_d435_mount.scad by hand -- these are the *claims*.
TILT = 8.0
CAM_W, CAM_D, CAM_H = 90.0, 25.0, 25.0
CAM_CLEAR, CAM_Z = 38.0, 120.0
CAM_FIT, WALL_T, PLATE_T = 0.8, 4.0, 7.0
# The D435's imagers are off the body centreline (realsense2_description):
# depth/infra1 by 17.5 mm, colour by 32.5. The cradle is shifted by this much so
# the imager, not the body, sits in the tool's symmetry plane.
CAM_LENS_DY = 32.5
RING_T, RING_OD, RING_BORE = 4.0, 62.0, 32.0
BOLT_BCD, BOLT_D, BOLT_CLOCK = 50.0, 6.6, 45.0
# AG-95 base envelope, measured from meshes/gripper/ag95/base_link.STL. The arm
# runs along the gripper's narrow axis (the wings' axis), so 33.5 is the half
# the mount has to stay out of.
GRIP_HALF_X, GRIP_HALF_Y, GRIP_TOP = 33.5, 46.5, 134.7
# The adapter's wings, measured on the rig: they reach r = 93/2 and stand 20 mm
# proud of the adapter's underside, on the same axis as the arm.
WING_R, WING_H = 46.5, 20.0
# NOT measured: how much the adapter itself lifts the gripper. The wing height
# is used as an upper bound, which makes the occlusion numbers below pessimistic
# rather than optimistic.
ADAPTER_T = 20.0


AG95 = ("/home/home/sdl_ws/src/sdl_project/TAMP/tamp/content/assets/robot/"
        "dcp_description/meshes/gripper/ag95/base_link.STL")


def load_stl(path):
    """Vertices of every triangle, ASCII or binary STL (OpenSCAD writes ASCII)."""
    blob = open(path, "rb").read()
    if len(blob) > 84:
        n = struct.unpack("<I", blob[80:84])[0]
        if len(blob) == 84 + 50 * n:
            dt = np.dtype([("n", "<3f4"), ("v", "<3,3f4"), ("a", "<u2")])
            tri = np.frombuffer(blob[84 : 84 + 50 * n], dtype=dt)
            return tri["v"].reshape(-1, 3).astype(np.float64), n
    nums = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", blob)
    v = np.array(nums, dtype=np.float64)
    return v, len(v) // 3


def main(path="mount_color.stl", lens_dy=None):
    global CAM_LENS_DY
    if lens_dy is not None:
        CAM_LENS_DY = float(lens_dy)
    v, ntri = load_stl(path)
    s, c = np.sin(np.radians(TILT)), np.cos(np.radians(TILT))
    cam_x = CAM_CLEAR + CAM_H / 2 * c
    # world -> camera local (u = width, v = optical axis, n = height)
    R = np.array([[0.0, -s, -c], [-1.0, 0, 0], [0.0, c, -s]])
    loc = (v - np.array([cam_x, 0.0, CAM_Z])) @ R

    ok = True

    def check(name, good, detail):
        nonlocal ok
        ok &= bool(good)
        print(f"[{'ok ' if good else 'FAIL'}] {name}: {detail}")

    lo, hi = v.min(0), v.max(0)
    check("bounding box", True, f"{np.round(lo,2).tolist()} .. {np.round(hi,2).tolist()}")

    # 1a. The arm passes UNDER the adapter's wings, so nothing may rise into
    #     them. Treated as a full cylinder, which is conservative.
    above = v[v[:, 2] > RING_T + 0.5]
    in_wing = above[(above[:, 2] < RING_T + WING_H)
                    & (np.hypot(above[:, 0], above[:, 1]) < WING_R)]
    check("clears the adapter wings", len(in_wing) == 0,
          f"{len(in_wing)} vertices inside r<{WING_R} below z={RING_T+WING_H}; "
          f"nearest material above the ring sits at x={above[:,0].min():.1f} mm")

    # 1b. And the camera passes OVER them, into the gripper's narrow side.
    top = above[above[:, 2] > RING_T + WING_H]
    in_grip = top[(np.abs(top[:, 0]) < GRIP_HALF_X) & (np.abs(top[:, 1]) < GRIP_HALF_Y)
                  & (top[:, 2] < RING_T + ADAPTER_T + GRIP_TOP)]
    check("clears the AG-95 body", len(in_grip) == 0,
          f"{len(in_grip)} vertices inside the gripper envelope; lowest material "
          f"above the wings is at z={top[:,2].min():.1f} mm "
          f"(wing top {RING_T+WING_H:.0f} mm)")

    # 1c. The camera body itself has to clear the wings too -- it hangs inboard
    #     of the tower, straight over them.
    corners = np.array([[CAM_LENS_DY + a * CAM_W / 2, b * CAM_D / 2, d * CAM_H / 2]
                        for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)])
    cam_world = corners @ R.T + np.array([cam_x, 0.0, CAM_Z])
    check("camera clears the wings", cam_world[:, 2].min() > RING_T + WING_H,
          f"lowest camera corner z={cam_world[:,2].min():.1f} mm vs wing top "
          f"{RING_T + WING_H:.0f} mm")

    # 2. The camera's own volume has to be empty.
    in_cam = ((np.abs(loc[:, 0] - CAM_LENS_DY) < CAM_W / 2 - 0.1) & (np.abs(loc[:, 1]) < CAM_D / 2 - 0.1)
              & (np.abs(loc[:, 2]) < CAM_H / 2 - 0.1))
    check("camera pocket is clear", in_cam.sum() == 0, f"{in_cam.sum()} vertices inside the D435 body")

    # 3. End walls: the camera drops between them with CAM_FIT total slack.
    # the wall faces live between the plate's inner surface and the wall tops
    # the wall tops: strictly above the plate's inner surface, beside the camera
    band = loc[(loc[:, 2] > -CAM_H / 2 + 0.1) & (loc[:, 2] < -CAM_H / 2 + 10.1)
               & (np.abs(loc[:, 1]) < CAM_D) & (np.abs(loc[:, 0] - CAM_LENS_DY) > 20)]
    right = band[band[:, 0] > CAM_LENS_DY][:, 0] - CAM_LENS_DY
    check("end wall spacing", abs(right.min() - (CAM_W + CAM_FIT) / 2) < 0.05,
          f"inner face at u={right.min():.2f} mm from the body centre "
          f"(want {(CAM_W+CAM_FIT)/2:.2f}), outer at u={right.max():.2f}")

    # 3b. And the body centre must sit CAM_LENS_DY off the tool's symmetry plane,
    #     which is what puts the imager on it. Read off the fixing hole.
    hole = v[(np.abs(loc[:, 2] + CAM_H / 2) < 1e-3)
             & (np.hypot(loc[:, 0] - CAM_LENS_DY, loc[:, 1]) < 5)]
    check("imager on the tool's symmetry plane", len(hole) > 0 and abs(hole[:, 1].mean() + CAM_LENS_DY) < 0.05,
          f"fixing hole at y={hole[:, 1].mean():.2f} mm (want {-CAM_LENS_DY:.2f}), "
          f"so the imager lands at y={hole[:, 1].mean() + CAM_LENS_DY:+.2f}")

    # 4. Nothing sticks out in front of the lens plane, i.e. into the FOV.
    front = loc[loc[:, 1] > CAM_D / 2 + 0.01]
    check("nothing in front of the lens face", len(front) == 0,
          f"{len(front)} vertices ahead of v={CAM_D/2} mm")

    # 5. Flange interface: bore, four bolt holes, four pin holes.
    # Holes are cut straight through, so their edges live on the flange face.
    ring = v[np.abs(v[:, 2]) < 1e-3]
    r = np.hypot(ring[:, 0], ring[:, 1])
    check("ring bore", abs(r.min() - RING_BORE / 2) < 0.05,
          f"smallest radius {r.min():.2f} mm (want {RING_BORE/2:.2f})")
    for label, clock in (("bolt", BOLT_CLOCK), ("pin", 0.0)):
        hits = []
        for k in range(4):
            a = np.radians(clock + 90 * k)
            ctr = np.array([BOLT_BCD / 2 * np.cos(a), BOLT_BCD / 2 * np.sin(a)])
            d = np.hypot(*(ring[:, :2] - ctr).T)
            hits.append(int((d < BOLT_D / 2 + 0.05).sum()))
        check(f"{label} holes", all(h > 0 for h in hits), f"vertex counts per hole {hits}")

    # 6. The engraving is a cut into the plate's outboard face, so its floor
    #    shows up 0.7 mm inside that face.
    face = -CAM_H / 2 - PLATE_T
    eng = loc[(loc[:, 2] > face + 0.6) & (loc[:, 2] < face + 0.8)]
    check("label engraved on the plate", len(eng) > 0,
          f"{len(eng)} vertices on the engraved floor")

    # 7. What the frame actually looks like: where the gripper lands in the
    #    image and where the tool axis sits. Same pinhole model that was checked
    #    against a real wrist frame.
    axis_z = CAM_Z + cam_x / np.tan(np.radians(TILT))
    W, H = 480, 270
    fyp = (H / 2) / np.tan(np.radians(21.25))
    t = np.radians(TILT)
    C = np.array([cam_x, 0.0, CAM_Z])
    yo, zo = np.array([np.cos(t), 0, np.sin(t)]), np.array([-np.sin(t), 0, np.cos(t)])
    def img_v(P):                       # image row, 0 = top edge, 1 = bottom
        d = np.asarray(P, float) - C
        return (H / 2 + fyp * (d @ yo) / (d @ zo)) / H
    # the gripper's own mesh, plus a slab standing in for the fingers
    gm = load_stl(AG95)[0] * 1000.0
    gm = gm @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]).T
    fy, fx, fz = (np.linspace(-46, 46, 30), np.linspace(-20, 20, 12),
                  np.linspace(106, 170, 25))
    FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing="ij")
    gm = np.vstack([gm, np.stack([FX.ravel(), FY.ravel(), FZ.ravel()], 1)])
    grips = []
    for lift in (0.0, 10.0, 20.0):
        P = gm + np.array([0, 0, RING_T + lift])
        dd = P - C
        Z = dd @ zo
        front = Z > 1e-3
        vv = (H / 2 + fyp * (dd @ yo)[front] / Z[front]) / H
        vis = vv[(vv > 0) & (vv < 1)]
        grips.append(vis.max() * 100 if len(vis) else 0.0)
    print(f"\ncamera body centre       (x, z) = ({cam_x:.2f}, {CAM_Z:.2f}) mm from the flange face")
    print(f"optical axis meets tool axis at  z = {axis_z:.0f} mm ahead of the flange")
    print("gripper reaches down to  %.0f / %.0f / %.0f %% of the frame"
          " (adapter lift 0/10/20 mm)" % tuple(grips))
    print("tool axis sits at        " + " / ".join(
        f"{img_v([0, 0, D]) * 100:.0f} %" for D in (350, 400, 450)) + "  of frame height"
        " (50 % = centre, at 350/400/450 mm)")
    print(f"\n{ntri} triangles, {len(np.unique(v, axis=0))} unique vertices")
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
