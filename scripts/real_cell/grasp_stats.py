#!/usr/bin/env python3
"""A perception-to-grasp trial series, scored from its bag_run.sh directories.

    grasp_stats.py TRIAL_DIR [TRIAL_DIR ...] [--csv OUT.csv] [--urdf FR5.urdf]

One directory per trial, as the series runs it:

    bag_run.sh grasp_N --perceive --beakers --recover beaker1 --grasp-only 0
    BAG_DIR=<dir> bag_run.sh grasp_N --fetch RUN --grasp-only --replay
    BAG_DIR=<dir> bag_run.sh grasp_N --perceive --beakers --recover beaker1 --measure-only

From plan_remote_1.log: beaker1 as the standing cameras gave it, and as the
wrist re-measured it before planning (the position the plan grasps at). From
plan_remote_2.log: whether the jaws stopped on it (``grasp HELD`` / ``MISSED``,
and where). From plan_remote_3.log: where it stood once the jaws had let go.

Jaws that centre what they grip push a beaker that was not where perception
put it onto their centre line, so the shift between the grasped-at position and
the released one is the perception error across the jaws, as the grasp
corrected it. With --urdf it is split along the jaws' closing axis and their
approach, from the plan's own reach (the plan's CSV, in the run's deliver/).

Printed: one line per trial, the grasp rate with its Wilson 95% interval, and
the shift and the standing-vs-wrist difference as mean, sd and worst.
"""
import argparse
import csv
import math
import os
import re
import statistics

BEAKER = re.compile(r'^beaker1: \((-?[0-9.]+), (-?[0-9.]+)\) from (\d+) poses on .*base at '
                    r'z=(-?[0-9.]+)')
MEASURED = re.compile(r'^=== measured: beaker1 (-?[0-9.]+),(-?[0-9.]+) z (-?[0-9.]+)')
GRASP = re.compile(r'grasp (HELD|MISSED): jaws at ([0-9.]+) mm')
RUN = re.compile(r'=== planning on \S+, run (\S+),')
SEEDS = re.compile(r'=== ready: (\S+\.csv)')


def wilson(k, n, z=1.959964):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def read(path):
    return open(path, errors='ignore').read().splitlines() if os.path.exists(path) else []


def trial(directory):
    t = {'dir': directory}
    lines = read(os.path.join(directory, 'plan_remote_1.log'))
    seen = [BEAKER.match(line) for line in lines]
    seen = [m for m in seen if m]
    recovered = any(line.startswith('recovery tree:') for line in lines)
    if seen:
        t['standing'] = (float(seen[0].group(1)), float(seen[0].group(2)))
        last = seen[-1]
        t['grasped_at'] = (float(last.group(1)), float(last.group(2)))
        t['wrist'] = t['grasped_at'] if recovered and len(seen) > 1 else None
    for line in lines:
        m = RUN.search(line)
        if m:
            t['run'] = m.group(1)
    for line in read(os.path.join(directory, 'plan_remote_2.log')):
        m = GRASP.search(line)
        if m:
            t['held'] = m.group(1) == 'HELD'
            t['jaws_mm'] = float(m.group(2))
        m = SEEDS.search(line)
        if m:
            t['csv'] = m.group(1)
    for line in read(os.path.join(directory, 'plan_remote_3.log')):
        m = MEASURED.match(line)
        if m:
            t['released'] = (float(m.group(1)), float(m.group(2)))
    return t


def jaw_axes(csv_path, urdf):
    """(closing, approach) unit vectors in the base frame at the plan's grasp."""
    import numpy as np
    import pinocchio as pin
    rows = list(csv.DictReader(open(csv_path)))
    pick = [r for r in rows if (r.get('operation') or '') == 'Pick'][-1]
    model = pin.buildModelFromUrdf(urdf)
    data = model.createData()
    q = pin.neutral(model)
    q[:6] = [float(pick[j]) for j in ('j1', 'j2', 'j3', 'j4', 'j5', 'j6')]
    pin.framesForwardKinematics(model, data, q)
    R = data.oMf[model.getFrameId('wrist3_link')].rotation
    closing, approach = R[:, 0].copy(), R[:, 2].copy()
    closing[2] = approach[2] = 0.0
    return closing / np.linalg.norm(closing), approach / np.linalg.norm(approach)


def describe(values):
    if not values:
        return '-'
    return '%.1f +- %.1f mm, worst %.1f' % (
        statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0,
        max(abs(v) for v in values))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--csv', default='')
    ap.add_argument('--urdf', default='', help='FR5 URDF, to split the shift along the jaws')
    a = ap.parse_args()
    rows = [trial(d) for d in a.dirs]

    print('%-3s %-26s %-6s %7s %9s %9s %9s %9s' % (
        '#', 'trial', 'grasp', 'jaws', 'shift', 'across', 'along', 'stand-wr'))
    shifts, across, along, stand = [], [], [], []
    for i, t in enumerate(rows, 1):
        s = c = p = d = None
        if 'grasped_at' in t and 'released' in t:
            dx = t['released'][0] - t['grasped_at'][0]
            dy = t['released'][1] - t['grasped_at'][1]
            s = 1000 * math.hypot(dx, dy)
            shifts.append(s)
            if a.urdf and t.get('csv') and os.path.exists(t['csv']):
                closing, approach = jaw_axes(t['csv'], a.urdf)
                c = 1000 * (dx * closing[0] + dy * closing[1])
                p = 1000 * (dx * approach[0] + dy * approach[1])
                across.append(c)
                along.append(p)
        if t.get('wrist') and t.get('standing'):
            d = 1000 * math.hypot(t['standing'][0] - t['wrist'][0],
                                  t['standing'][1] - t['wrist'][1])
            stand.append(d)
        t.update(shift_mm=s, across_mm=c, along_mm=p, standing_vs_wrist_mm=d)
        fmt = lambda v: '%9.1f' % v if v is not None else '%9s' % '-'  # noqa: E731
        print('%-3d %-26s %-6s %7s %s %s %s %s' % (
            i, os.path.basename(t['dir'].rstrip('/'))[:26],
            {True: 'HELD', False: 'MISSED'}.get(t.get('held'), '?'),
            '%.1f' % t['jaws_mm'] if 'jaws_mm' in t else '-', fmt(s), fmt(c), fmt(p), fmt(d)))

    scored = [t for t in rows if 'held' in t]
    k = sum(1 for t in scored if t['held'])
    lo, hi = wilson(k, len(scored))
    print()
    print('grasped: %d/%d = %s (Wilson 95%% %.0f-%.0f%%)' % (
        k, len(scored), '%.0f%%' % (100.0 * k / len(scored)) if scored else '-', 100 * lo,
        100 * hi))
    print('shift on release      %s  (n=%d)' % (describe(shifts), len(shifts)))
    if across:
        print('   across the jaws    %s' % describe(across))
        print('   along the approach %s' % describe(along))
    print('standing vs wrist     %s  (n=%d)' % (describe(stand), len(stand)))

    if a.csv:
        keys = ['dir', 'run', 'held', 'jaws_mm', 'standing', 'wrist', 'grasped_at', 'released',
                'shift_mm', 'across_mm', 'along_mm', 'standing_vs_wrist_mm', 'csv']
        with open(a.csv, 'w', newline='') as handle:
            w = csv.DictWriter(handle, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            for t in rows:
                w.writerow(t)
        print('wrote %s' % a.csv)


if __name__ == '__main__':
    main()
