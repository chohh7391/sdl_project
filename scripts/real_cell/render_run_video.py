#!/usr/bin/env python3
"""Render a real-cell run's replay bag into one video: cameras, perception, weight.

    render_run_video.py RUN_DIR [--out FILE.mp4] [--title TEXT] [--fps 30]
                        [--speed 1.0] [--fast N] [--hold S] [--start S] [--end S]

RUN_DIR is a bag_run.sh directory: its 2_replay bag, plan_remote_2.log (the
task's steps) and bringup.log (the pouring controller's lines; a live_logs
bringup log works too, via --bringup). Laid out 1920 x 1080:

    side_1, rectified, with every tag it detected   | side_2 colour, with the
    outlined and named                               | perceived beakers drawn in
    ------------------------------------------------+---------------------------
    wrist colour                                     | weight and tilt over the run

and a bar along the top naming the step the task is in, the pour's own phase
while it pours. Nothing is computed that the run did not publish: the tags are
the detector's, the beakers the pose node's, the grams the pour's feedback.

--fast N plays everything but the pour N times faster (the pour, from 3 s
before it to 2 s after, stays at --speed) and says which in the bar; --hold S
holds the last frame, with the pour's result, S seconds.
"""
import argparse
import bisect
import math
import os
import re
import subprocess
import sys

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from tf2_ros import Buffer

W, H = 1920, 1080
BAR = 64
PANEL_W, PANEL_H = W // 2, (H - BAR) // 2
BG = (24, 24, 28)
FG = (235, 235, 235)
MUTED = (150, 150, 155)
ACCENT = (80, 200, 255)      # BGR: amber-ish orange for beaker1
B1 = (60, 160, 255)
B2 = (255, 200, 60)
GOOD = (120, 220, 120)
FONT = cv2.FONT_HERSHEY_SIMPLEX
NAMES = {0: 'beaker1', 1: 'beaker2'}

TOPICS = {
    'side_1': '/side_1/left/image_raw/compressed',
    'side_1_info': '/side_1/left/camera_info',
    'side_2': '/side_2/side_2/color/image_raw/compressed',
    'side_2_info': '/side_2/side_2/color/camera_info',
    'wrist': '/wrist/wrist/color/image_raw/compressed',
    'det_1': '/side_1/detections',
    'b1': '/perception/object_pose/held_beaker',
    'b1_hand': '/perception/object_pose/held_beaker/in_hand',
    'b2': '/perception/object_pose/beaker2',
    'fb': '/controller_action_server/pouring_controller/_action/feedback',
    'scale': '/scale/reading',
    'tf': '/tf',
    'tf_static': '/tf_static',
}

STEPS = [  # task behaviour name prefix -> what the bar says
    ('Go_Home_MoveIt_Final', 'Home'),
    ('Go_Home_MoveIt', 'Start pose'),
    ('_Replay_Pick', 'Reach'),
    ('_Gripper_Close', 'Grasp'),
    ('_Replay_Move_to_Surface', 'Carry'),
    ('_Pour_', 'Pour'),
    ('_Gripper_Open', 'Release'),
    ('_Replay_Place', 'Place'),
]
POUR_PHASES = {0: 'level and align', 1: 'seeking the onset', 2: 'pouring', 3: 'stopping',
               4: 'settling', 5: 'topping up', 6: 'returning'}


def text(img, s, org, scale=0.7, color=FG, thick=2):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def fit(img, w, h):
    """Letterbox img into w x h."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    out = np.full((h, w, 3), BG, np.uint8)
    rw, rh = int(iw * s), int(ih * s)
    small = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    x, y = (w - rw) // 2, (h - rh) // 2
    out[y:y + rh, x:x + rw] = small
    return out, s, x, y


def label(panel, s):
    (tw, th), _ = cv2.getTextSize(s, FONT, 0.6, 1)
    cv2.rectangle(panel, (0, 0), (tw + 20, th + 14), (0, 0, 0), -1)
    text(panel, s, (10, th + 7), 0.6, FG, 1)


def task_steps(log_path):
    """[(start, label)] from the task manager's 'Sending Goal' lines."""
    steps = []
    pat = re.compile(r'\[(\d+\.\d+)\] \[task_manager_node\]: \[([^\]]+)\] Sending Goal')
    for line in open(log_path, errors='ignore'):
        m = pat.search(line)
        if not m:
            continue
        name = m.group(2)
        for key, lab in STEPS:
            if key in name:
                steps.append((float(m.group(1)), lab))
                break
    end = re.compile(r'\[(\d+\.\d+)\] \[task_manager_node\]: Task finished')
    for line in open(log_path, errors='ignore'):
        m = end.search(line)
        if m:
            steps.append((float(m.group(1)), 'Done'))
    return steps


def pour_summary(log_path):
    """(target grams, material), and one line on what the pour delivered."""
    target, result, t0 = None, None, None
    stamp = re.compile(r'\[(\d+\.\d+)\] \[pouring_controller\]')
    for line in open(log_path, errors='ignore'):
        ts = stamp.search(line)
        m = re.search(r'Pouring ([0-9.]+) g of (liquid|granular media)', line)
        if m:
            target = (float(m.group(1)), 'sugar' if 'granular' in m.group(2) else 'water')
            t0 = float(ts.group(1)) if ts else None
        m = re.search(r'Poured ([0-9.]+) g in ([0-9.]+) s', line)
        if m and target:
            result = 'poured %s g in %s s (target %g g, %+.2f g)' % (
                m.group(1), m.group(2), target[0], float(m.group(1)) - target[0])
        m = re.search(r'Aborted: (.*)', line)
        if m and 'pouring_controller' in line:
            d = re.search(r'delivered ([0-9.]+) g', m.group(1))
            if d and target:
                took = ' in %.1f s' % (float(ts.group(1)) - t0) if ts and t0 else ''
                result = 'poured %s g%s (target %g g, %+.2f g)' % (
                    d.group(1), took, target[0], float(d.group(1)) - target[0])
            else:
                result = m.group(1)[:90]
    return target, result


def read_bag(path, t_from, t_to):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'),
           rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    wanted = [v for v in TOPICS.values() if v in types]
    r.set_filter(rosbag2_py.StorageFilter(topics=wanted))
    by_topic = {v: k for k, v in TOPICS.items()}
    cls = {t: get_message(types[t]) for t in wanted}
    while r.has_next():
        topic, raw, stamp = r.read_next()
        t = stamp * 1e-9
        key = by_topic[topic]
        if key not in ('tf', 'tf_static', 'side_1_info', 'side_2_info') and not (
                t_from - 1.0 <= t <= t_to):
            continue
        if key in ('side_1', 'side_2', 'wrist'):
            yield t, key, raw      # decoded only when shown
        else:
            yield t, key, deserialize_message(raw, cls[topic])


class Series:
    def __init__(self):
        self.t, self.v = [], []

    def add(self, t, v):
        self.t.append(t)
        self.v.append(v)


def draw_plot(panel, x0, y0, w, h, series, t_now, t_range, v_range, color, title, unit,
              target=None):
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (45, 45, 52), 1)
    t0, t1 = t_range
    v0, v1 = v_range

    def px(t, v):
        return (int(x0 + (t - t0) / (t1 - t0) * w), int(y0 + h - (v - v0) / (v1 - v0) * h))

    for frac in (0.25, 0.5, 0.75):
        y = int(y0 + h * frac)
        cv2.line(panel, (x0, y), (x0 + w, y), (40, 40, 46), 1)
    if target is not None:
        cv2.line(panel, px(t0, target), px(t1, target), (90, 90, 230), 1, cv2.LINE_AA)
        text(panel, 'target %g %s' % (target, unit), (x0 + w - 150, px(t0, target)[1] - 6),
             0.5, (120, 120, 240), 1)
    k = bisect.bisect_right(series.t, t_now)
    j = bisect.bisect_left(series.t, t0)
    if k - j >= 2:
        pts = np.array([px(t, min(max(v, v0), v1))
                        for t, v in zip(series.t[j:k], series.v[j:k])], np.int32)
        cv2.polylines(panel, [pts], False, color, 2, cv2.LINE_AA)
    cx = px(t_now, v0)[0]
    cv2.line(panel, (cx, y0), (cx, y0 + h), (90, 90, 90), 1)
    text(panel, title, (x0, y0 - 10), 0.6, MUTED, 1)
    text(panel, '%.0f' % v1, (x0 - 44, y0 + 12), 0.45, MUTED, 1)
    text(panel, '%.0f' % v0, (x0 - 44, y0 + h), 0.45, MUTED, 1)


def project(info, T_cam_base, pts_base):
    """Base-frame points -> pixels for a pinhole camera_info (rectified K)."""
    K = np.array(info.k).reshape(3, 3)
    P = (T_cam_base[:3, :3] @ pts_base.T).T + T_cam_base[:3, 3]
    ok = P[:, 2] > 0.05
    uv = (K @ P.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv, ok


def transform(tfbuf, target, source, t):
    try:
        tr = tfbuf.lookup_transform(target, source, Time(seconds=t)).transform
    except Exception:
        try:
            tr = tfbuf.lookup_transform(target, source, Time()).transform
        except Exception:
            return None
    q, p = tr.rotation, tr.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def rim_points(bottom, axis=(0.0, 0.0, 1.0), radius=0.025, height=0.070, n=36):
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    u = np.cross(a, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(a, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    th = np.linspace(0, 2 * math.pi, n)
    ring = radius * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))
    return np.vstack([np.asarray(bottom) + ring, np.asarray(bottom) + a * height + ring])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('run_dir')
    ap.add_argument('--out', default='')
    ap.add_argument('--title', default='')
    ap.add_argument('--bringup', default='', help='the bringup log (default RUN_DIR/bringup.log)')
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--speed', type=float, default=1.0, help='playback speed, e.g. 2 for 2x')
    ap.add_argument('--fast', type=float, default=0.0,
                    help='playback speed outside the pour (0: --speed throughout)')
    ap.add_argument('--hold', type=float, default=0.0, help='seconds to hold the last frame')
    ap.add_argument('--start', type=float, default=None, help='seconds into the bag')
    ap.add_argument('--end', type=float, default=None)
    a = ap.parse_args()

    run = os.path.expanduser(a.run_dir.rstrip('/'))
    bag = os.path.join(run, '2_replay')
    steps = task_steps(os.path.join(run, 'plan_remote_2.log'))
    bringup = a.bringup or os.path.join(run, 'bringup.log')
    target, result = pour_summary(bringup) if os.path.exists(bringup) else (None, None)
    if not steps:
        sys.exit('no task steps in %s/plan_remote_2.log' % run)
    t_close = next((s0 for s0, lab in steps if lab == 'Grasp'), None)
    t_open = next((s0 for s0, lab in steps if lab == 'Release'), None)
    marker = None
    for name in sorted(os.listdir(os.path.join(run, 'plan'))) if os.path.isdir(
            os.path.join(run, 'plan')) else []:
        if name.endswith('_cell.yaml'):
            import yaml
            cell = yaml.safe_load(open(os.path.join(run, 'plan', name)))
            marker = np.array(cell.get('pour', {}).get('grasp_marker'), float)
    t_begin = steps[0][0] - 2.0 if a.start is None else None
    t_end = steps[-1][0] + 2.0 if a.end is None else None
    # the bag's own clock for --start/--end
    meta_r = rosbag2_py.SequentialReader()
    meta_r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    bag0 = meta_r.get_metadata().starting_time.nanoseconds * 1e-9
    del meta_r
    if a.start is not None:
        t_begin = bag0 + a.start
    if a.end is not None:
        t_end = bag0 + a.end
    title = a.title or ('%s %g g' % (target[1].capitalize(), target[0]) if target else
                        os.path.basename(run))
    out = a.out or os.path.join(run, re.sub(r'[^A-Za-z0-9]+', '_', title).strip('_').lower()
                                + '.mp4')

    tfbuf = Buffer(cache_time=rclpy_duration(600.0))
    info = {}
    maps1 = None
    latest = {}          # key -> (t, raw or msg)
    decoded = {}         # key -> (t, image)
    dets = []
    scale = Series()
    grams = Series()
    tilt = Series()
    phase = [None, None]
    scale_zero = None
    b1, b1_hand, b2 = None, None, None

    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
                           '-pix_fmt', 'bgr24', '-s', '%dx%d' % (W, H), '-r', str(a.fps),
                           '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
                           '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    t_frame = t_begin
    span = (t_begin, t_end)
    t_pour = next((s0 for s0, lab in steps if lab == 'Pour'), None)
    t_after = next((s0 for s0, lab in steps if t_pour is not None and s0 > t_pour), None)
    fast = a.fast if a.fast > 0.0 and t_pour is not None else a.speed

    def speed_at(t):
        if t_pour is not None and t_pour - 3.0 <= t <= (t_after or t_end) + 2.0:
            return a.speed
        return fast

    def image(key):
        if key not in latest:
            return None
        t, raw = latest[key]
        if decoded.get(key, (None,))[0] != t:
            msg = deserialize_message(raw, get_message('sensor_msgs/msg/CompressedImage'))
            img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_UNCHANGED)
            decoded[key] = (t, img)
        return decoded[key][1]

    def step_at(t):
        cur = None
        for s0, lab in steps:
            if s0 <= t:
                cur = lab
        return cur

    held = {}

    def beaker1_in_jaws(t):
        """(bottom, axis) of beaker1 while held: the grasp at the close, carried by TF."""
        if marker is None or t_close is None or not (t_close + 0.5 <= t < (t_open or 1e18)):
            return None
        if 'ee' not in held:
            Tc = transform(tfbuf, 'base_link', 'wrist3_link', t_close + 0.5)
            if Tc is None:
                return None
            Ti = np.linalg.inv(Tc)
            held['ee'] = (Ti[:3, :3] @ marker + Ti[:3, 3], Tc[:3, :3].T @ np.array([0, 0, 1.0]))
        T = transform(tfbuf, 'base_link', 'wrist3_link', t)
        if T is None:
            return None
        b, a = held['ee']
        return T[:3, :3] @ b + T[:3, 3], T[:3, :3] @ a

    def render(t, badge=True):
        frame = np.full((H, W, 3), BG, np.uint8)
        # --- bar
        cv2.rectangle(frame, (0, 0), (W, BAR), (12, 12, 14), -1)
        text(frame, title, (24, 42), 1.0, FG, 2)
        order = ['Start pose', 'Reach', 'Grasp', 'Carry', 'Pour', 'Place', 'Release', 'Home']
        cur = step_at(t)
        (title_w, _), _ = cv2.getTextSize(title, FONT, 1.0, 2)
        x = max(520, 24 + title_w + 60)
        for lab in order:
            on = lab == cur
            (tw, _), _ = cv2.getTextSize(lab, FONT, 0.62, 2)
            if on:
                cv2.rectangle(frame, (x - 10, 14), (x + tw + 10, 50), (60, 140, 230), -1)
            text(frame, lab, (x, 40), 0.62, FG if on else MUTED, 2 if on else 1)
            x += tw + 34
        text(frame, '%5.1f s' % (t - t_begin), (W - 130, 42), 0.8, MUTED, 2)
        if badge and fast != a.speed:
            sp = speed_at(t)
            badge = '%gx' % sp if sp != 1.0 else 'real time'
            (bw, _), _ = cv2.getTextSize(badge, FONT, 0.62, 2)
            bx = W - 160 - bw
            cv2.rectangle(frame, (bx - 10, 16), (bx + bw + 10, 48),
                          (60, 60, 66) if sp != 1.0 else (40, 110, 40), -1)
            text(frame, badge, (bx, 40), 0.62, FG, 2)

        # --- side_1, rectified, with tags
        img = image('side_1')
        if img is not None and maps1 is not None:
            rect = cv2.remap(img, maps1[0], maps1[1], cv2.INTER_LINEAR)
            rect = cv2.cvtColor(rect, cv2.COLOR_GRAY2BGR)
            shown = [d for td, ds in dets if t - td < 0.4 for d in ds][-4:]
            seen_ids = set()
            for d in reversed(shown):
                if d.id in seen_ids:
                    continue
                seen_ids.add(d.id)
                pts = np.array([[c.x, c.y] for c in d.corners], np.int32)
                col = B1 if d.id == 0 else B2 if d.id == 1 else (180, 180, 180)
                cv2.polylines(rect, [pts], True, col, 3, cv2.LINE_AA)
                cx, cy = int(d.centre.x), int(d.centre.y)
                text(rect, '%s (tag %d)' % (NAMES.get(d.id, 'tag'), d.id), (cx - 70, cy - 28),
                     0.9, col, 2)
            panel, _, _, _ = fit(rect, PANEL_W, PANEL_H)
        else:
            panel = np.full((PANEL_H, PANEL_W, 3), BG, np.uint8)
        label(panel, 'side_1  OAK-D, AprilTag detections')
        frame[BAR:BAR + PANEL_H, 0:PANEL_W] = panel

        # --- side_2 colour, with the perceived beakers
        img = image('side_2')
        if img is not None:
            vis = img.copy()
            T = transform(tfbuf, 'side_2_color_optical_frame', 'base_link', t)
            if T is not None and 'side_2_info' in info:
                jaws = beaker1_in_jaws(t)
                b1_now = (t, jaws[0], jaws[1]) if jaws is not None else b1
                for name, pose, col in (('beaker2', b2, B2),
                                        ('beaker1 (in the jaws)' if jaws is not None
                                         else 'beaker1', b1_now, B1)):
                    if pose is None or (jaws is None or name == 'beaker2') and t - pose[0] > 1.5:
                        continue
                    pts = rim_points(pose[1], pose[2])
                    uv, ok = project(info['side_2_info'], T, pts)
                    if ok.all():
                        n = len(pts) // 2
                        for ring in (uv[:n], uv[n:]):
                            cv2.polylines(vis, [ring.astype(np.int32)], True, col, 2,
                                          cv2.LINE_AA)
                        top = uv[n:].mean(axis=0)
                        text(vis, name, (int(top[0]) - 40, int(top[1]) - 24), 0.55, col, 2)
            panel, _, _, _ = fit(vis, PANEL_W, PANEL_H)
        else:
            panel = np.full((PANEL_H, PANEL_W, 3), BG, np.uint8)
        label(panel, 'side_2  overhead, perceived beakers')
        frame[BAR:BAR + PANEL_H, PANEL_W:W] = panel

        # --- wrist colour
        img = image('wrist')
        panel = fit(img, PANEL_W, PANEL_H)[0] if img is not None else \
            np.full((PANEL_H, PANEL_W, 3), BG, np.uint8)
        label(panel, 'wrist  D435')
        frame[BAR + PANEL_H:H, 0:PANEL_W] = panel

        # --- weight and tilt
        panel = np.full((PANEL_H, PANEL_W, 3), (18, 18, 22), np.uint8)
        g_now = grams.v[bisect.bisect_right(grams.t, t) - 1] if grams.t and t >= grams.t[0] \
            else 0.0
        tgt = target[0] if target else None
        big = '%5.2f g' % g_now
        text(panel, 'poured', (40, 56), 0.7, MUTED, 1)
        text(panel, big, (40, 120), 2.0, GOOD if (tgt and abs(g_now - tgt) <= 0.5) else FG, 4)
        if tgt:
            text(panel, '/ %g g' % tgt, (330, 120), 1.0, MUTED, 2)
        k = bisect.bisect_right(phase[0] or [], t) - 1 if phase[0] else -1
        if cur == 'Pour' and k >= 0:
            text(panel, POUR_PHASES.get(phase[1][k], ''), (40, 160), 0.8, ACCENT, 2)
        elif cur == 'Done' and result:
            text(panel, result, (40, 160), 0.6, GOOD, 1)
        top = max([tgt or 0.0] + scale.v) if scale.v else (tgt or 10.0)
        draw_plot(panel, 80, 210, PANEL_W - 120, 140, scale, t, span,
                  (min(0.0, min(scale.v) if scale.v else 0.0), max(5.0, top * 1.15)), GOOD,
                  'scale [g]', 'g', tgt)
        draw_plot(panel, 80, 395, PANEL_W - 120, 110, tilt, t, span, (0.0, 100.0), ACCENT,
                  'tilt [deg]', 'deg')
        frame[BAR + PANEL_H:H, PANEL_W:W] = panel
        cv2.line(frame, (PANEL_W, BAR), (PANEL_W, H), (60, 60, 66), 2)
        cv2.line(frame, (0, BAR + PANEL_H), (W, BAR + PANEL_H), (60, 60, 66), 2)
        return frame

    n = 0
    for t, key, msg in read_bag(bag, t_begin, t_end):
        while t_frame <= min(t, t_end):
            ff.stdin.write(render(t_frame).tobytes())
            t_frame += speed_at(t_frame) / a.fps
            n += 1
        if t > t_end:
            break
        if key in ('side_1', 'side_2', 'wrist'):
            latest[key] = (t, msg)
        elif key in ('tf', 'tf_static'):
            for tr in msg.transforms:
                if key == 'tf_static':
                    tfbuf.set_transform_static(tr, 'bag')
                else:
                    tfbuf.set_transform(tr, 'bag')
        elif key == 'side_1_info' and maps1 is None:
            K, D = np.array(msg.k).reshape(3, 3), np.array(msg.d)
            Rr, P = np.array(msg.r).reshape(3, 3), np.array(msg.p).reshape(3, 4)
            maps1 = cv2.initUndistortRectifyMap(K, D, Rr, P[:3, :3], (msg.width, msg.height),
                                                cv2.CV_16SC2)
        elif key == 'side_2_info':
            info['side_2_info'] = msg
        elif key == 'det_1':
            if msg.detections:
                dets.append((t, list(msg.detections)))
                dets[:] = dets[-3:]
        elif key in ('b1', 'b1_hand', 'b2'):
            p, q = msg.pose.position, msg.pose.orientation
            axis = (2 * (q.x * q.z + q.y * q.w), 2 * (q.y * q.z - q.x * q.w),
                    1 - 2 * (q.x * q.x + q.y * q.y))
            pose = (t, np.array([p.x, p.y, p.z]), axis if key == 'b1_hand' else (0, 0, 1))
            if key == 'b1':
                b1 = pose
            elif key == 'b2':
                b2 = pose
            else:
                b1_hand = pose
        elif key == 'fb':
            f = msg.feedback
            grams.add(t, f.current_grams)
            tilt.add(t, math.degrees(f.tilt))
            if phase[0] is None:
                phase[0], phase[1] = [], []
            phase[0].append(t)
            phase[1].append(f.phase)
        elif key == 'scale':
            if scale_zero is None:
                scale_zero = msg.grams
            scale.add(t, msg.grams - scale_zero)
        if b1_hand is not None and t - b1_hand[0] > 1.5:
            b1_hand = None
    while t_frame <= t_end:
        ff.stdin.write(render(t_frame).tobytes())
        t_frame += speed_at(t_frame) / a.fps
        n += 1
    if a.hold:
        last = render(min(t_frame, t_end), badge=False)
        for _ in range(int(round(a.hold * a.fps))):
            ff.stdin.write(last.tobytes())
            n += 1
    ff.stdin.close()
    ff.wait()
    print('wrote %s: %d frames, %.1f s (%gx%s%s)' % (
        out, n, n / a.fps, a.speed, ', %gx outside the pour' % fast if fast != a.speed else '',
        ', %g s held' % a.hold if a.hold else ''))


def rclpy_duration(sec):
    from rclpy.duration import Duration
    return Duration(seconds=sec)


if __name__ == '__main__':
    main()
