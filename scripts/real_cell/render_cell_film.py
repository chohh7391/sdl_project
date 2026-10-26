#!/usr/bin/env python3
"""A real-cell run as one film: the robot and what perception saw, the cameras, the pour.

    render_cell_film.py RUN_DIR --wide WIDE.mp4 --wide-offset EPOCH
                        [--close CLOSE.mp4 --close-offset EPOCH]
                        [--out FILE.mp4] [--title TEXT] [--fast 4] [--hold 3]

RUN_DIR is a bag_run.sh directory with both bags: 1_perceive (perception, and
the wrist's occlusion recovery when there was one) and 2_replay (the task),
plan_remote_{1,2}.log, bringup.log and plan/. WIDE and CLOSE are hand-held
recordings of the replay -- the whole cell, and the pour close up -- and their
offsets the bag-clock epoch of their first frame (film_sync.py finds them).

Laid out 1920 x 1080 under a bar naming the step:

    the robot as /joint_states had it, the       | side_1 and side_2, with what
    beakers where perception put them, each      | their detectors found, and
    camera's line of sight to a tag it sees      | each beaker's per-camera state
    ---------------------------------------------+-----------------------------
    the recording: the cell, close up while      | weight and tilt
    it pours                                     |

Played --fast times faster except the pour, which runs in real time, with a
card for the planning between the two bags. Nothing is drawn that the run did
not publish: the joints, the detector's tags and their TF, the pose node's
beakers and its per-camera states, the pour's feedback and the scale.
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
from rosidl_runtime_py.utilities import get_message
from tf2_ros import Buffer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_run_video as rv  # noqa: E402
import twin_render as tw  # noqa: E402

W, H, BAR = 1920, 1080, 64
TOP_H = 540
BOT_H = H - BAR - TOP_H
BG, FG, MUTED = rv.BG, rv.FG, rv.MUTED
B1, B2, GOOD, ACCENT = rv.B1, rv.B2, rv.GOOD, rv.ACCENT
text = rv.text

CAMS = {  # camera -> (optical frame, colour of its line of sight)
    'side_1': ('side_1_left_camera_optical_frame', (200, 200, 120)),
    'side_2': ('side_2_infra1_optical_frame', (200, 140, 220)),
    'wrist': ('wrist_infra1_optical_frame', (120, 220, 200)),
}
TAG_FRAME = re.compile(r'^(side_1|side_2|wrist)_tag_(\d+)$')
TOPICS = dict(rv.TOPICS)
TOPICS.update({'joints': '/joint_states', 'vis': '/perception/object_visibility',
               'status': '/real_cell/status', 'urdf': '/robot_description',
               'side_2_det': '/side_2/detections'})
CAM_STATE = {0: ('no data', (90, 90, 95)), 1: ('ok', GOOD), 2: ('not in view', (110, 110, 115)),
             3: ('rejected', (60, 140, 230)), 4: ('no TF', (60, 140, 230)), 5: ('superseded', (170, 170, 120)),
             6: ('stale', (90, 90, 95))}
ORDER = ['Perceive', 'Recovery', 'Plan', 'Start pose', 'Reach', 'Grasp', 'Carry', 'Pour', 'Place',
         'Release', 'Home']


class Film:
    """Frames of a recording by bag-clock epoch, read forward."""

    def __init__(self, path, offset):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.offset, self.idx, self.frame = offset, -1, None

    def at(self, epoch):
        k = int(round((epoch - self.offset) * self.fps))
        if k < 0 or k >= self.n:
            return None
        if k < self.idx or k - self.idx > 90:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            self.idx = k - 1
        while self.idx < k:
            if not self.cap.grab():
                return self.frame
            self.idx += 1
        ok, f = self.cap.retrieve()
        if ok:
            self.frame = f
        return self.frame


def read(bag, keys):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'),
           rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    wanted = {TOPICS[k]: k for k in keys if TOPICS[k] in types}
    r.set_filter(rosbag2_py.StorageFilter(topics=list(wanted)))
    cls = {t: get_message(types[t]) for t in wanted}
    while r.has_next():
        topic, raw, stamp = r.read_next()
        key = wanted[topic]
        yield stamp * 1e-9, key, raw if key in ('side_1', 'side_2') else deserialize_message(
            raw, cls[topic])


def bag_span(bag):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'),
           rosbag2_py.ConverterOptions('cdr', 'cdr'))
    m = r.get_metadata()
    t0 = m.starting_time.nanoseconds * 1e-9
    return t0, t0 + m.duration.nanoseconds * 1e-9


def planning_card(run):
    """Lines for the card between the bags, from plan_remote_{1,2}.log."""
    lines = []
    log1 = open(os.path.join(run, 'plan_remote_1.log'), errors='ignore').read()
    log2 = open(os.path.join(run, 'plan_remote_2.log'), errors='ignore').read()
    m = re.search(r'=== planning on (\S+), run (\S+), seeds ([0-9 ]+):', log1)
    seeds = re.findall(r'^=== seed (\d+)\s+(\d\d:\d\d:\d\d)', log1, re.M)
    acc = re.search(r'accepted (\d+); best first: ([^\n]*)', log1)
    # plan_remote.sh's 'plan: <run>/deliver/traj_realcell_seed<N>' line; transfer_plan:= in
    # the logs of runs from before the cell's tasks took a task_params file.
    used = re.search(r'(?:^plan: |transfer_plan:=)\S*_seed(\d+)', log2, re.M)
    if m:
        lines.append('run %s: seeds %s planned in Isaac Sim with cuTAMP' % (m.group(2), m.group(3).strip()))
    if seeds:
        lines.append('first seed at %s, last at %s' % (seeds[0][1], seeds[-1][1]))
    if acc:
        lines.append('accepted %s: %s' % (acc.group(1), acc.group(2)[:90]))
    if used:
        lines.append('replaying seed %s' % used.group(1))
    return lines


def tag_frame_pose(tfbuf, frame, t):
    return rv.transform(tfbuf, 'base_link', frame, t)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('run_dir')
    ap.add_argument('--wide', default='')
    ap.add_argument('--wide-offset', type=float, default=None)
    ap.add_argument('--close', default='')
    ap.add_argument('--close-offset', type=float, default=None)
    ap.add_argument('--out', default='')
    ap.add_argument('--title', default='')
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--fast', type=float, default=4.0)
    ap.add_argument('--hold', type=float, default=3.0)
    ap.add_argument('--card', type=float, default=3.0, help='seconds of the planning card')
    ap.add_argument('--max-frames', type=int, default=0, help='stop after this many (a test)')
    a = ap.parse_args()

    run = os.path.expanduser(a.run_dir.rstrip('/'))
    steps = rv.task_steps(os.path.join(run, 'plan_remote_2.log'))
    target, result = rv.pour_summary(os.path.join(run, 'bringup.log'))
    title = a.title or ('FR5 pours %g g of %s' % (target[0], target[1]) if target else
                        os.path.basename(run))
    out = a.out or os.path.join(run, re.sub(r'[^A-Za-z0-9]+', '_', title).strip('_').lower() + '.mp4')
    t_close = next((s0 for s0, lab in steps if lab == 'Grasp'), None)
    t_open = next((s0 for s0, lab in steps if lab == 'Release'), None)
    t_pour = next((s0 for s0, lab in steps if lab == 'Pour'), None)
    t_after = next((s0 for s0, lab in steps if t_pour is not None and s0 > t_pour), None)
    marker = None
    pdir = os.path.join(run, 'plan')
    for name in sorted(os.listdir(pdir)) if os.path.isdir(pdir) else []:
        if name.endswith('_cell.yaml'):
            import yaml
            marker = np.array(yaml.safe_load(open(os.path.join(pdir, name))).get('pour', {})
                              .get('grasp_marker'), float)
    wide = Film(a.wide, a.wide_offset) if a.wide and a.wide_offset is not None else None
    close = Film(a.close, a.close_offset) if a.close and a.close_offset is not None else None

    urdf = None
    for _, _, msg in read(os.path.join(run, '2_replay'), ['urdf']):
        urdf = msg.data
        break
    twin = tw.TwinRenderer(urdf, 960, TOP_H)
    tfbuf = Buffer(cache_time=rv.rclpy_duration(3600.0))

    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                           '-s', '%dx%d' % (W, H), '-r', str(a.fps), '-i', '-', '-c:v', 'libx264',
                           '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', out],
                          stdin=subprocess.PIPE)
    n_frames = [0]

    def emit(frame):
        ff.stdin.write(frame.tobytes())
        n_frames[0] += 1
        if a.max_frames and n_frames[0] >= a.max_frames:
            ff.stdin.close()
            ff.wait()
            print('stopped after %d frames: %s' % (n_frames[0], out))
            sys.exit(0)

    has_recovery = False
    for t, key, msg in read(os.path.join(run, '1_perceive'), ['status']):
        if 'occlusion recovery' in msg.text:
            has_recovery = True
            break
    order = [o for o in ORDER if o != 'Recovery' or has_recovery]

    def bar(frame, cur, clock, speed):
        cv2.rectangle(frame, (0, 0), (W, BAR), (12, 12, 14), -1)
        text(frame, title, (24, 42), 0.9, FG, 2)
        (tw_, _), _ = cv2.getTextSize(title, rv.FONT, 0.9, 2)
        x = max(430, 24 + tw_ + 40)
        for lab in order:
            on = lab == cur
            (lw, _), _ = cv2.getTextSize(lab, rv.FONT, 0.55, 2)
            if on:
                cv2.rectangle(frame, (x - 8, 16), (x + lw + 8, 48), (60, 140, 230), -1)
            text(frame, lab, (x, 39), 0.55, FG if on else MUTED, 2 if on else 1)
            x += lw + 26
        if speed is not None:
            badge = '%gx' % speed if speed != 1.0 else 'real time'
            (bw, _), _ = cv2.getTextSize(badge, rv.FONT, 0.6, 2)
            bx = W - 150 - bw
            cv2.rectangle(frame, (bx - 10, 16), (bx + bw + 10, 48),
                          (60, 60, 66) if speed != 1.0 else (40, 110, 40), -1)
            text(frame, badge, (bx, 40), 0.6, FG, 2)
        if clock:
            text(frame, clock, (W - 120, 42), 0.75, MUTED, 2)

    # ------------------------------------------------------------------ state
    st = dict(latest={}, decoded={}, dets1=[], dets2=[], q={}, b1=None, b2=None, b1_hand=None,
              vis={}, status='', tags={}, pulses=[], last_shown={}, maps1=None, info2=None)
    scale, grams, tilt = rv.Series(), rv.Series(), rv.Series()
    phase = ([], [])
    scale_zero = [None]

    def image(key):
        if key not in st['latest']:
            return None
        t, raw = st['latest'][key]
        if st['decoded'].get(key, (None,))[0] != t:
            m = deserialize_message(raw, get_message('sensor_msgs/msg/CompressedImage'))
            st['decoded'][key] = (t, cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8),
                                                  cv2.IMREAD_UNCHANGED))
        return st['decoded'][key][1]

    held = {}

    def beaker1_pose(t):
        """4x4 of beaker1: in the jaws (grasp at the close, carried by TF), else where perceived."""
        if t_close is not None and t_close + 0.5 <= t < (t_open or 1e18) and st['phase_b']:
            if 'w3_b' not in held:
                Tc = rv.transform(tfbuf, 'base_link', 'wrist3_link', t_close + 0.5)
                b = marker if marker is not None else (st['b1'][1] if st['b1'] else None)
                if Tc is None or b is None:
                    return None, False
                Tb = np.eye(4)
                Tb[:3, 3] = b
                held['w3_b'] = np.linalg.inv(Tc) @ Tb
            T3 = rv.transform(tfbuf, 'base_link', 'wrist3_link', t)
            return (T3 @ held['w3_b'], True) if T3 is not None else (None, True)
        if st['b1'] is None:
            return None, False
        Tb = np.eye(4)
        Tb[:3, 3] = st['b1'][1]
        return Tb, False

    def on_pose(name, t, pos):
        last = st['last_shown'].get(name)
        if last is None or np.linalg.norm(pos - last) > 0.003:
            o = st['vis'].get(name)
            seen = [c.camera for c in o.cameras if c.state == 1] if o else []
            st['pulses'].append((t, name, pos.copy(), (np.linalg.norm(pos - last) * 1000) if last is not None
                                 else None, seen))
            st['pulses'][:] = st['pulses'][-8:]
            st['last_shown'][name] = pos.copy()

    def consume(t, key, msg):
        if key in ('side_1', 'side_2'):
            st['latest'][key] = (t, msg)
        elif key in ('tf', 'tf_static'):
            for tr in msg.transforms:
                if key == 'tf_static':
                    tfbuf.set_transform_static(tr, 'bag')
                else:
                    tfbuf.set_transform(tr, 'bag')
                    m = TAG_FRAME.match(tr.child_frame_id)
                    if m:
                        st['tags'][(m.group(1), int(m.group(2)))] = t
        elif key == 'joints':
            st['q'] = dict(zip(msg.name, msg.position))
        elif key == 'side_1_info' and st['maps1'] is None:
            K, D = np.array(msg.k).reshape(3, 3), np.array(msg.d)
            Rr, P = np.array(msg.r).reshape(3, 3), np.array(msg.p).reshape(3, 4)
            st['maps1'] = cv2.initUndistortRectifyMap(K, D, Rr, P[:3, :3], (msg.width, msg.height),
                                                      cv2.CV_16SC2)
        elif key == 'side_2_info':
            st['info2'] = msg
        elif key == 'det_1':
            if msg.detections:
                st['dets1'].append((t, list(msg.detections)))
                st['dets1'][:] = st['dets1'][-3:]
        elif key in ('b1', 'b2', 'b1_hand'):
            p = msg.pose.position
            pos = np.array([p.x, p.y, p.z])
            if key == 'b1':
                st['b1'] = (t, pos)
                on_pose('beaker1', t, pos)
            elif key == 'b2':
                st['b2'] = (t, pos)
                on_pose('beaker2', t, pos)
        elif key == 'vis':
            for o in msg.objects:
                st['vis'][o.name] = o
        elif key == 'status':
            st['status'] = msg.text
        elif key == 'fb':
            f = msg.feedback
            grams.add(t, f.current_grams)
            tilt.add(t, math.degrees(f.tilt))
            phase[0].append(t)
            phase[1].append(f.phase)
        elif key == 'scale':
            if scale_zero[0] is None:
                scale_zero[0] = msg.grams
            scale.add(t, msg.grams - scale_zero[0])

    # ------------------------------------------------------------------ panels
    def twin_panel(t, view='main'):
        T1, in_jaws = beaker1_pose(t)
        beakers = []
        if T1 is not None:
            beakers.append((T1, B1, 0.7))
        if st['b2'] is not None:
            T2 = np.eye(4)
            T2[:3, 3] = st['b2'][1]
            beakers.append((T2, B2, 0.7))
        img, P = twin.render(st['q'], beakers=beakers, view=view)
        # cameras: a small frustum each, and a line to every tag seen in the last 0.3 s
        for cam, (frame, col) in CAMS.items():
            Tc = rv.transform(tfbuf, 'base_link', frame, t)
            if Tc is None:
                continue
            o = Tc[:3, 3]
            corners = [Tc[:3, :3] @ np.array([x, y, 0.09]) + o for x, y in
                       ((-0.06, -0.035), (0.06, -0.035), (0.06, 0.035), (-0.06, 0.035))]
            uv, ok = tw.project(P, np.vstack([o] + corners))
            if ok.all():
                pts = uv.astype(int)
                for k in range(1, 5):
                    cv2.line(img, tuple(pts[0]), tuple(pts[k]), col, 1, cv2.LINE_AA)
                    cv2.line(img, tuple(pts[k]), tuple(pts[1 + k % 4]), col, 1, cv2.LINE_AA)
                text(img, cam, (pts[0][0] + 6, pts[0][1] - 6), 0.5, col, 1)
            for (c, tag), ts in st['tags'].items():
                if c != cam or t - ts > 0.3 or tag not in (0, 1):
                    continue
                Tt = rv.transform(tfbuf, 'base_link', '%s_tag_%d' % (cam, tag), ts)
                if Tt is None:
                    continue
                uv, ok = tw.project(P, np.vstack([o, Tt[:3, 3]]))
                if ok.all():
                    # solid where the pose node took this camera's sample, faint where the
                    # detector saw the tag but the sample was rejected or superseded
                    o_vis = st['vis'].get('beaker1' if tag == 0 else 'beaker2')
                    used = o_vis is not None and any(c.camera == cam and c.state == 1 for c in o_vis.cameras)
                    col_l = B1 if tag == 0 else B2
                    p0, p1 = uv.astype(int)
                    if used:
                        cv2.line(img, tuple(p0), tuple(p1), col_l, 2, cv2.LINE_AA)
                        cv2.circle(img, tuple(p1), 4, col_l, -1, cv2.LINE_AA)
                    else:
                        cv2.line(img, tuple(p0), tuple(p1), tuple(int(c * 0.45) for c in col_l), 1, cv2.LINE_AA)
        # perception updates: a ring that grows and fades where a beaker was placed anew
        for t0, name, pos, moved, seen in st['pulses']:
            age = t - t0
            if not 0.0 <= age < 1.2:
                continue
            uv, ok = tw.project(P, pos + np.array([0, 0, 0.035]))
            if not ok.all():
                continue
            c = tuple(int(v) for v in uv[0])
            col = B1 if name == 'beaker1' else B2
            cv2.circle(img, c, int(18 + 40 * age), col, 2, cv2.LINE_AA)
        for name, T, col in (('beaker1', T1, B1), ('beaker2', beakers[-1][0] if st['b2'] else None, B2)):
            if T is None:
                continue
            uv, ok = tw.project(P, T[:3, 3] + T[:3, 2] * 0.085)
            if ok.all():
                c = tuple(int(v) for v in uv[0])
                lab = name + (' (in the jaws)' if name == 'beaker1' and in_jaws else '')
                text(img, lab, (c[0] - 40, c[1] - 8), 0.55, col, 2)
        rv.label(img, '3D: /joint_states, perceived beakers; lines: a camera sees the tag (bold: pose taken)'
                 if view == 'main'
                 else '3D from above')
        return img

    def cams_panel(t):
        panel = np.full((TOP_H, 960, 3), BG, np.uint8)
        img = image('side_1')
        if img is not None and st['maps1'] is not None:
            rect = cv2.cvtColor(cv2.remap(img, st['maps1'][0], st['maps1'][1], cv2.INTER_LINEAR),
                                cv2.COLOR_GRAY2BGR)
            shown, seen_ids = [d for td, ds in st['dets1'] if t - td < 0.4 for d in ds][-4:], set()
            for d in reversed(shown):
                if d.id in seen_ids:
                    continue
                seen_ids.add(d.id)
                pts = np.array([[c.x, c.y] for c in d.corners], np.int32)
                col = B1 if d.id == 0 else B2 if d.id == 1 else (180, 180, 180)
                cv2.polylines(rect, [pts], True, col, 4, cv2.LINE_AA)
            p, _, _, _ = rv.fit(rect, 533, 300)
            rv.label(p, 'side_1  OAK-D')
            panel[8:308, 4:537] = p
        img = image('side_2')
        if img is not None:
            vis = img.copy()
            T = rv.transform(tfbuf, 'side_2_color_optical_frame', 'base_link', t)
            if T is not None and st['info2'] is not None:
                T1, in_jaws = beaker1_pose(t)
                for name, pos, axis, col in (('beaker2', st['b2'][1] if st['b2'] else None, (0, 0, 1), B2),
                                             ('beaker1', T1[:3, 3] if T1 is not None else None,
                                              T1[:3, 2] if T1 is not None else None, B1)):
                    if pos is None:
                        continue
                    uv, ok = rv.project(st['info2'], T, rv.rim_points(pos, axis))
                    if ok.all():
                        n = len(uv) // 2
                        for ring in (uv[:n], uv[n:]):
                            cv2.polylines(vis, [ring.astype(np.int32)], True, col, 2, cv2.LINE_AA)
            p, _, _, _ = rv.fit(vis, 400, 300)
            rv.label(p, 'side_2  D435')
            panel[8:308, 552:952] = p
        # each beaker: where the pose node has it, and what each camera says
        y = 340
        for name, col in (('beaker1', B1), ('beaker2', B2)):
            o = st['vis'].get(name)
            pose = st['b1'] if name == 'beaker1' else st['b2']
            where = '(%.3f, %.3f)' % (pose[1][0], pose[1][1]) if pose else '-'
            text(panel, '%s  %s' % (name, where), (16, y), 0.62, col, 2)
            x = 330
            if o:
                for c in o.cameras:
                    s, cc = CAM_STATE.get(c.state, ('?', MUTED))
                    cv2.circle(panel, (x, y - 6), 7, cc, -1, cv2.LINE_AA)
                    text(panel, '%s %s' % (c.camera, s), (x + 14, y), 0.5, FG if c.state == 1 else MUTED, 1)
                    x += 200
                text(panel, o.status.split(' | ')[0][:118], (16, y + 30), 0.45, MUTED, 1)
            y += 82
        return panel

    def film_panel(t, in_pour):
        if not st['phase_b']:
            return rv.fit(twin_panel(t, 'top'), 960, BOT_H)[0]
        panel = np.full((BOT_H, 960, 3), BG, np.uint8)
        f, lab = None, ''
        if in_pour and close is not None:
            f = close.at(t)
            lab = 'recording, close up'
        if f is None and wide is not None:
            f = wide.at(t)
            lab = 'recording, the cell'
        if f is None:
            text(panel, 'the hand-held cameras run from the replay', (210, BOT_H // 2), 0.7, MUTED, 1)
            return panel
        p, _, _, _ = rv.fit(f, 960, BOT_H)
        rv.label(p, lab)
        return p

    def perception_panel(t):
        panel = np.full((BOT_H, 960, 3), (18, 18, 22), np.uint8)
        text(panel, 'perception', (40, 50), 0.65, MUTED, 1)
        s = re.split(r' \| |\n', st['status'])[0]
        s = re.sub(r'^\d\d:\d\d:\d\d\s+', '', s)
        text(panel, s[:70], (40, 100), 0.8, ACCENT, 2)
        y = 160
        for line in measured:
            text(panel, line[:92], (40, y), 0.55, FG, 1)
            y += 34
        return panel

    def plot_panel(t, cur, span):
        if not st['phase_b']:
            return perception_panel(t)
        panel = np.full((BOT_H, 960, 3), (18, 18, 22), np.uint8)
        g_now = grams.v[bisect.bisect_right(grams.t, t) - 1] if grams.t and t >= grams.t[0] else 0.0
        tgt = target[0] if target else None
        text(panel, 'poured', (40, 50), 0.65, MUTED, 1)
        text(panel, '%5.2f g' % g_now, (40, 108), 1.8, GOOD if (tgt and abs(g_now - tgt) <= 0.5) else FG, 4)
        if tgt:
            text(panel, '/ %g g' % tgt, (300, 108), 0.9, MUTED, 2)
        k = bisect.bisect_right(phase[0], t) - 1
        if cur == 'Pour' and k >= 0:
            text(panel, rv.POUR_PHASES.get(phase[1][k], ''), (40, 145), 0.75, ACCENT, 2)
        elif cur in ('Home', 'Done') and result:
            text(panel, result, (40, 145), 0.6, GOOD, 1)
        top = max([tgt or 0.0] + scale.v) if scale.v else (tgt or 10.0)
        rv.draw_plot(panel, 80, 190, 960 - 120, 120, scale, t, span,
                     (min(0.0, min(scale.v) if scale.v else 0.0), max(5.0, top * 1.15)), GOOD,
                     'scale [g]', 'g', tgt)
        rv.draw_plot(panel, 80, 365, 960 - 120, 90, tilt, t, span, (0.0, 100.0), ACCENT,
                     'tilt [deg]', 'deg')
        return panel

    def compose(t, cur, clock, speed, in_pour, span):
        frame = np.full((H, W, 3), BG, np.uint8)
        bar(frame, cur, clock, speed)
        frame[BAR:BAR + TOP_H, 0:960] = twin_panel(t)
        frame[BAR:BAR + TOP_H, 960:W] = cams_panel(t)
        frame[BAR + TOP_H:H, 0:960] = film_panel(t, in_pour)
        frame[BAR + TOP_H:H, 960:W] = plot_panel(t, cur, span)
        cv2.line(frame, (960, BAR), (960, H), (60, 60, 66), 2)
        cv2.line(frame, (0, BAR + TOP_H), (W, BAR + TOP_H), (60, 60, 66), 2)
        return frame

    keys = ['side_1', 'side_1_info', 'side_2', 'side_2_info', 'det_1', 'b1', 'b2', 'b1_hand', 'fb',
            'scale', 'tf', 'tf_static', 'joints', 'vis', 'status']

    # what perception settled on, as plan_remote reported it
    measured = []
    for ln in open(os.path.join(run, 'plan_remote_1.log'), errors='ignore'):
        m = re.match(r'^(beaker[12]): \((-?[0-9.]+), (-?[0-9.]+)\) from (\d+) poses.*spread ([0-9.]+) mm', ln)
        if m:
            measured.append('%s at (%s, %s) from %s poses, spread %s mm' % m.groups())
        elif ln.startswith('=== to recover') or ln.startswith('recovery tree:'):
            measured.append(ln.strip().strip('= ').split(' (log')[0])

    # ------------------------------------------------------- part A: perception
    st['phase_b'] = False
    p0, p1 = bag_span(os.path.join(run, '1_perceive'))
    tf_ = p0
    t_start = p0
    for t, key, msg in read(os.path.join(run, '1_perceive'), keys):
        while tf_ <= t:
            cur = 'Recovery' if 'occlusion recovery' in st['status'] else \
                'Plan' if st['status'].startswith(('planning', 'ready')) else 'Perceive'
            emit(compose(tf_, cur, '%5.1f s' % (tf_ - t_start), a.fast, False, (p0, p1)))
            tf_ += a.fast / a.fps
        consume(t, key, msg)
    last = compose(tf_, 'Plan', '%5.1f s' % (tf_ - t_start), None, False, (p0, p1))

    # ------------------------------------------------------- the planning card
    lines = planning_card(run)
    for k in range(int(a.card * a.fps)):
        card = (last * 0.25).astype(np.uint8)
        bar(card, 'Plan', '', None)
        text(card, 'Planning on rci-pc', (560, 420), 1.6, FG, 3)
        for i, ln in enumerate(lines):
            text(card, ln, (560, 490 + 44 * i), 0.8, MUTED, 2)
        emit(card)

    # ------------------------------------------------------- part B: the replay
    st['phase_b'] = True
    st['pulses'] = []
    scale.t.clear(), scale.v.clear()
    scale_zero[0] = None
    r0 = steps[0][0] - 2.0
    r1 = steps[-1][0] + 2.0
    tf_ = r0

    def speed_at(t):
        return 1.0 if t_pour is not None and t_pour - 3.0 <= t <= (t_after or r1) + 2.0 else a.fast

    def step_at(t):
        cur = None
        for s0, lab in steps:
            if s0 <= t:
                cur = lab
        return cur

    for t, key, msg in read(os.path.join(run, '2_replay'), keys):
        while tf_ <= min(t, r1):
            sp = speed_at(tf_)
            emit(compose(tf_, step_at(tf_), '%5.1f s' % (tf_ - r0), sp, sp == 1.0, (r0, r1)))
            tf_ += sp / a.fps
        if t > r1:
            break
        consume(t, key, msg)
    while tf_ <= r1:
        sp = speed_at(tf_)
        emit(compose(tf_, step_at(tf_), '%5.1f s' % (tf_ - r0), sp, sp == 1.0, (r0, r1)))
        tf_ += sp / a.fps
    end = compose(r1, 'Home', '%5.1f s' % (r1 - r0), None, False, (r0, r1))
    for _ in range(int(a.hold * a.fps)):
        emit(end)
    ff.stdin.close()
    ff.wait()
    print('wrote %s: %d frames, %.1f s' % (out, n_frames[0], n_frames[0] / a.fps))
    return 0


if __name__ == '__main__':
    sys.exit(main())
