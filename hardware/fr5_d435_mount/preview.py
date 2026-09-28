#!/usr/bin/env python3
"""Flat-shaded orthographic previews of the mount, and of the mount in place.

numpy + matplotlib only: a painter's-algorithm renderer is enough to see
whether the part looks like what the parameters say, and it runs anywhere the
rest of this repo runs.

    python3 preview.py
"""
import re
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

HERE = __file__.rsplit("/", 1)[0]
AG95 = ("/home/home/sdl_ws/src/sdl_project/TAMP/tamp/content/assets/robot/"
        "dcp_description/meshes/gripper/ag95/base_link.STL")

WING_R, WING_H, ADAPTER_T, CAM_LENS_DY = 46.5, 20.0, 20.0, 32.5
TILT, CAM_W, CAM_D, CAM_H, CAM_CLEAR, CAM_Z, RING_T = 8.0, 90.0, 25.0, 25.0, 38.0, 120.0, 4.0


def load_stl(path, scale=1.0):
    blob = open(path, "rb").read()
    if len(blob) > 84:
        n = struct.unpack("<I", blob[80:84])[0]
        if len(blob) == 84 + 50 * n:
            dt = np.dtype([("n", "<3f4"), ("v", "<3,3f4"), ("a", "<u2")])
            tri = np.frombuffer(blob[84 : 84 + 50 * n], dtype=dt)
            return tri["v"].astype(np.float64) * scale
    v = np.array(re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", blob), dtype=np.float64)
    return v.reshape(-1, 3, 3) * scale


def box(size, T=np.eye(4)):
    sx, sy, sz = np.array(size) / 2.0
    p = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    f = [(0,1,3),(0,3,2),(4,7,5),(4,6,7),(0,5,1),(0,4,5),(2,3,7),(2,7,6),
         (0,2,6),(0,6,4),(1,5,7),(1,7,3)]
    p = p @ T[:3, :3].T + T[:3, 3]
    return p[np.array(f)]


def draw(ax, tris, direction, up, color, alpha=1.0, edge=None):
    w = np.array(direction, float); w /= np.linalg.norm(w)
    u = np.cross(up, w); u /= np.linalg.norm(u)
    v = np.cross(w, u)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    key = (w + np.array(up) * 0.45); key /= np.linalg.norm(key)
    lit = 0.62 + 0.38 * np.abs(n @ key)
    pts = np.stack([tris @ u, tris @ v], axis=-1)
    depth = (tris.mean(1) @ w)
    order = np.argsort(depth)
    rgb = np.array(matplotlib.colors.to_rgb(color))
    fc = np.clip(rgb[None, :] * lit[order][:, None], 0, 1)
    ax.add_collection(PolyCollection(pts[order], facecolors=np.c_[fc, np.full(len(fc), alpha)],
                                     edgecolors=edge if edge else "none", linewidths=0.15))
    return pts.reshape(-1, 2)


def frame(ax, extents, title):
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
    ax.set_xlim(extents[0]); ax.set_ylim(extents[1])
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cccccc")


def cam_transform():
    s, c = np.sin(np.radians(TILT)), np.cos(np.radians(TILT))
    T = np.eye(4)
    T[:3, :3] = np.array([[0, -s, -c], [-1, 0, 0], [0, c, -s]])
    T[:3, 3] = [CAM_CLEAR + CAM_H / 2 * c, 0, CAM_Z]
    return T


def predicted_frame(ax, cam_clear, cam_z, tilt, lift, title, lens_dy=0.0):
    """Where the gripper lands in a 480x270 wrist frame. The same pinhole model
    reproduced a real wrist image, so this is a fair picture of the view."""
    W, H = 480, 270
    fxp = (W / 2) / np.tan(np.radians(34.7))
    fyp = (H / 2) / np.tan(np.radians(21.25))
    g = load_stl(AG95, scale=1000.0).reshape(-1, 3)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    fy, fx, fz = (np.linspace(-46, 46, 30), np.linspace(-20, 20, 12),
                  np.linspace(106, 170, 25))
    FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing="ij")
    P = np.vstack([g @ R.T, np.stack([FX.ravel(), FY.ravel(), FZ.ravel()], 1)])
    P = P + np.array([0, 0, RING_T + lift])
    t = np.radians(tilt)
    # the optical origin is the imager, which is lens_dy off the body centre
    C = np.array([cam_clear + CAM_H / 2 * np.cos(t), lens_dy, cam_z])
    xo, yo, zo = (np.array([0, -1, 0.0]), np.array([np.cos(t), 0, np.sin(t)]),
                  np.array([-np.sin(t), 0, np.cos(t)]))
    d = P - C
    Z = d @ zo
    front = Z > 1e-3
    u = W / 2 + fxp * (d @ xo)[front] / Z[front]
    vv = H / 2 + fyp * (d @ yo)[front] / Z[front]
    ax.add_patch(plt.Rectangle((0, 0), W, H, fc="#eef2f6", ec="#334155", lw=1.6, zorder=0))
    ax.scatter(u, vv, s=1.2, c="#c0392b", alpha=0.35, edgecolors="none", zorder=2)
    for D, mk in ((350, "o"), (400, "s"), (450, "^")):
        a = np.array([0, 0, D]) - C
        ax.plot(W / 2 + fxp * (a @ xo) / (a @ zo), H / 2 + fyp * (a @ yo) / (a @ zo),
                mk, ms=7, mfc="none", mec="#0d7c4f", mew=1.8, zorder=3)
    ax.axhline(H / 2, color="#94a3b8", lw=0.8, ls=":", zorder=1)
    ax.axvline(W / 2, color="#94a3b8", lw=0.8, ls=":", zorder=1)
    ax.set_xlim(-30, W + 30); ax.set_ylim(H + 30, -30)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def main():
    mount = load_stl(f"{HERE}/mount_color.stl")
    T = cam_transform()
    T[:3, 3] = T[:3, 3] + T[:3, 0] * CAM_LENS_DY      # body sits off to one side
    cam = box([CAM_W, CAM_D, CAM_H], T)
    grip = load_stl(AG95, scale=1000.0)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])   # fingers open across the arm
    grip = grip @ R.T + np.array([0, 0, RING_T + ADAPTER_T])
    # the adapter's two wings, as measured: the arm goes under, the camera over
    wings = np.vstack([box([WING_R - 18, 30, WING_H],
                           T=np.array([[1., 0, 0, sx * (WING_R + 18) / 2],
                                       [0, 1., 0, 0],
                                       [0, 0, 1., RING_T + WING_H / 2],
                                       [0, 0, 0, 1.]]))
                       for sx in (-1, 1)])

    views = [((-1, -1.6, 0.75), (0, 0, 1), "isometric"),
             ((0, -1, 0), (0, 0, 1), "side: X-Z, tool axis up"),
             ((1, 0, 0), (0, 0, 1), "front: from the tool axis outwards"),
             ((0, 0, -1), (1, 0, 0), "top: flange face")]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (d, up, name) in zip(axes.ravel(), views):
        p = draw(ax, mount, d, up, "#3f96dd")
        draw(ax, cam, d, up, "#ff8a3d", alpha=0.5)
        lo, hi = p.min(0) - 8, p.max(0) + 8
        frame(ax, [(lo[0], hi[0]), (lo[1], hi[1])], name)
    fig.suptitle("FR5 wrist mount for RealSense D435 (camera shown in orange)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{HERE}/preview_mount.png", dpi=110)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (d, up, name) in zip(axes, [views[0], views[1]]):
        p = draw(ax, grip, d, up, "#b9c0c7", alpha=0.85)
        draw(ax, wings, d, up, "#4bb07a", alpha=0.9)
        p2 = draw(ax, mount, d, up, "#3f96dd")
        draw(ax, cam, d, up, "#ff8a3d", alpha=0.6)
        allp = np.vstack([p, p2])
        lo, hi = allp.min(0) - 8, allp.max(0) + 8
        frame(ax, [(lo[0], hi[0]), (lo[1], hi[1])], name)
    fig.suptitle("blue = mount,  orange = D435,  green = adapter wings,  grey = AG-95",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{HERE}/preview_assembly.png", dpi=110)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    predicted_frame(axes[0], 38, 44, 15, 10,
                    "before:  cam_z=44, tilt=15, body centred", lens_dy=CAM_LENS_DY)
    predicted_frame(axes[1], 38, CAM_Z, TILT, 10,
                    f"after:  cam_z={CAM_Z:.0f}, tilt={TILT:.0f}, imager centred", lens_dy=0.0)
    fig.suptitle("predicted 480x270 wrist frame   red = AG-95,   green = tool axis at "
                 "350 / 400 / 450 mm", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{HERE}/preview_view.png", dpi=120)
    print("wrote preview_mount.png, preview_assembly.png and preview_view.png")


if __name__ == "__main__":
    main()
