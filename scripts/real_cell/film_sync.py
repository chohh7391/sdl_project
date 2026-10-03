#!/usr/bin/env python3
"""When a hand-held recording of a real-cell run started, on the bag's clock.

    film_sync.py VIDEO BAG [--local-tz-hours 9] [--search 30]

The camera's own creation_time puts it within a few seconds (DJI writes UTC;
a camera that writes local time with a Z needs --local-tz-hours). This refines
that by cross-correlating how much the picture changes from frame to frame
with how fast the arm moves (/joint_states in BAG): the lag with the best
correlation, within +-search s of the creation time, is when the video's first
frame was taken. Printed: the offset (bag epoch of video t=0), the correlation
there, and the next-best peak for comparison.
"""
import argparse
import datetime
import json
import subprocess
import sys

import cv2
import numpy as np


def creation_epoch(path, local_tz_hours=0.0):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format_tags=creation_time',
                          '-of', 'json', path], capture_output=True, text=True).stdout
    s = json.loads(out)['format']['tags']['creation_time']
    t = datetime.datetime.strptime(s[:19], '%Y-%m-%dT%H:%M:%S').replace(
        tzinfo=datetime.timezone.utc).timestamp()
    return t - 3600.0 * local_tz_hours


def motion_series(path, rate=10.0, max_s=None):
    """(times from the first frame, mean abs frame difference) at about `rate` Hz."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / rate)))
    ts, vals, prev, i = [], [], None, 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            g = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                ts.append(i / fps)
                vals.append(float(np.mean(np.abs(g - prev))))
            prev = g
            if max_s and i / fps > max_s:
                break
        i += 1
    return np.array(ts), np.array(vals)


def arm_speed(bag, rate=10.0):
    """(bag epoch, sum |joint velocity| of j1..j6) at `rate` Hz, from /joint_states."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import JointState
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'),
           rosbag2_py.ConverterOptions('cdr', 'cdr'))
    r.set_filter(rosbag2_py.StorageFilter(topics=['/joint_states']))
    t, q = [], []
    while r.has_next():
        _, raw, stamp = r.read_next()
        m = deserialize_message(raw, JointState)
        idx = [m.name.index(j) for j in ('j1', 'j2', 'j3', 'j4', 'j5', 'j6') if j in m.name]
        if len(idx) == 6:
            t.append(stamp * 1e-9)
            q.append([m.position[k] for k in idx])
    t, q = np.array(t), np.array(q)
    grid = np.arange(t[0], t[-1], 1.0 / rate)
    qg = np.stack([np.interp(grid, t, q[:, k]) for k in range(6)], axis=1)
    speed = np.r_[0.0, np.abs(np.diff(qg, axis=0)).sum(axis=1) * rate]
    return grid, speed


def best_offset(vt, vm, bt, bs, guess, search, rate=10.0):
    """Video t=0 on the bag clock, maximising the correlation of motion with arm speed."""
    lags = np.arange(guess - search, guess + search, 1.0 / rate)
    grid = np.arange(bt[0], bt[-1], 1.0 / rate)
    s = np.interp(grid, bt, bs)
    s = (s - s.mean()) / (s.std() + 1e-9)
    out = []
    for lag in lags:
        m = np.interp(grid - lag, vt, vm, left=np.nan, right=np.nan)
        ok = np.isfinite(m)
        if ok.sum() < 0.8 * min(len(grid), (vt[-1] - vt[0]) * rate):
            out.append(-1.0)
            continue
        mm = m[ok]
        mm = (mm - mm.mean()) / (mm.std() + 1e-9)
        out.append(float(np.mean(mm * s[ok])))
    out = np.array(out)
    k = int(np.argmax(out))
    away = np.abs(lags - lags[k]) > 2.0
    return lags[k], out[k], (lags[away][np.argmax(out[away])], out[away].max()) if away.any() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('video')
    ap.add_argument('bag')
    ap.add_argument('--local-tz-hours', type=float, default=0.0)
    ap.add_argument('--search', type=float, default=30.0)
    a = ap.parse_args()
    guess = creation_epoch(a.video, a.local_tz_hours)
    vt, vm = motion_series(a.video)
    bt, bs = arm_speed(a.bag)
    lag, c, other = best_offset(vt, vm, bt, bs, guess, a.search)
    print('%s: creation %.1f, best %.2f (%+.2f s from creation), correlation %.3f; next peak %s'
          % (a.video, guess, lag, lag - guess, c,
             '%+.2f s at %.3f' % (other[0] - guess, other[1]) if other else '-'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
