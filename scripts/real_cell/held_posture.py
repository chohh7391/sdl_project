#!/usr/bin/env python3
"""The arm's posture while it holds beaker1, against the plans that ran clean.

    held_posture.py build OUT.json PLAN [PLAN ...]   envelope of the plans given
    held_posture.py check ENVELOPE.json PLAN [PLAN ...] [--margin 0.15]

PLAN is an exported replay plan without its extension (<run>/deliver/
traj_realcell_seed<N>): its .csv and .meta.json. The held phase is the plan
from its gripper close to the open after it. The envelope is each joint's
range over the held phases of the plans given; a plan passes when no joint of
its held phase leaves that range by more than --margin.

Why the held phase: on 2026-10-02 the planner returned a seed that picked
beaker1 with the wrist turned over (j5 +2.3..+2.7 rad, j1 from -1.0 where
every plan before it held j5 within -0.8..+0.7 and j1 above +0.1), and carrying
it that way the arm ran into side_1's mount. The camera clearance checked
then was to the lens, 146 mm, which the mount does not obey. A posture the
cell has run before has also been past the mounts before.

score_real_trajectories.py applies the same check to a run's recordings
(held_posture_envelope.json beside this file).
"""
import argparse
import csv
import json
import os
import sys

JOINTS = ('j1', 'j2', 'j3', 'j4', 'j5', 'j6')
DEFAULT_MARGIN = 0.15
ENVELOPE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'held_posture_envelope.json')


def held_rows(plan):
    """Joint rows of a plan between its gripper close and the open after it."""
    rows = list(csv.DictReader(open(plan + '.csv')))
    events = json.load(open(plan + '.meta.json'))['gripper_events']
    close = next(e['t_s'] for e in events if e['event'] == 'close')
    opened = next(e['t_s'] for e in events if e['event'] == 'open' and e['t_s'] > close)
    return [[float(r[j]) for j in JOINTS] for r in rows if close <= float(r['t_s']) <= opened]


def excursions(rows, envelope, margin=DEFAULT_MARGIN):
    """{joint: how far past the envelope plus margin [rad]} for the joints that leave it."""
    out = {}
    for k, joint in enumerate(JOINTS):
        lo, hi = envelope['joints'][joint]
        past = max(max(lo - margin - q[k], q[k] - hi - margin) for q in rows)
        if past > 0.0:
            out[joint] = past
    return out


def load(path=ENVELOPE):
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('out')
    b.add_argument('plans', nargs='+')
    c = sub.add_parser('check')
    c.add_argument('envelope')
    c.add_argument('plans', nargs='+')
    c.add_argument('--margin', type=float, default=DEFAULT_MARGIN)
    a = ap.parse_args()

    if a.cmd == 'build':
        rows = [q for p in a.plans for q in held_rows(p)]
        env = {'note': 'joint range [rad] while beaker1 is held, over plans the real cell ran '
                       'without touching anything (held_posture.py build)',
               'plans': [os.path.relpath(p, os.path.expanduser('~')) for p in a.plans],
               'joints': {j: [round(min(q[k] for q in rows), 3), round(max(q[k] for q in rows), 3)]
                          for k, j in enumerate(JOINTS)}}
        json.dump(env, open(a.out, 'w'), indent=1)
        print('wrote %s from %d plans: %s' % (a.out, len(a.plans), '  '.join(
            '%s [%+.2f, %+.2f]' % (j, *env['joints'][j]) for j in JOINTS)))
        return 0
    env = load(a.envelope)
    bad = 0
    for p in a.plans:
        past = excursions(held_rows(p), env, a.margin)
        bad += bool(past)
        print('%-60s %s' % (p, 'pass' if not past else 'REJECT: ' + ', '.join(
            '%s %.2f rad past' % (j, v) for j, v in past.items())))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
