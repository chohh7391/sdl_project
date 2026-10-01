#!/usr/bin/env python3
"""Every pour the real cell's pouring controller made, from its log, and their statistics.

    pour_stats.py LOG [LOG ...] [--since EPOCH] [--until EPOCH] [--csv OUT.csv]

LOG is a bringup log (~/real_cell_runs/live_logs/bringupN.log, or a run's
bringup.log): every "Pouring X g of ..." line opens a trial and the pour's own
outcome line closes it -- "Poured G g in S s ..." or "Aborted: ...". What a
trial delivered is the controller's number: the poured grams, or for an
aborted pour the one its message gives, else the last settle before it.

Printed per material and target separately -- they are not pooled -- with the
proportion inside the goal's tolerance and its Wilson 95% interval, mean and
standard deviation of what was delivered, the mean and worst absolute error,
and how long the pours took. A trial that ended for any other reason than the
pour (a goal the tree cancelled, a controller switched off mid-pour) is listed
and left out of the statistics, with the reason; so is one the scale saw
jump (something other than the pour moved the pan).
"""
import argparse
import csv
import math
import re
import statistics
import sys

STAMP = re.compile(r'\[(\d+\.\d+)\] \[pouring_controller\]: (.*)$')
START = re.compile(r'Pouring ([0-9.]+) g of (liquid|granular media) \(flow_index ([0-9.]+)\) '
                   r'into a ([0-9.]+) g vessel \(tol ([0-9.]+) g')
POURED = re.compile(r'Poured ([0-9.]+) g in ([0-9.]+) s, peak tilt ([0-9.]+) deg, (\d+) trim')
DELIVERED = re.compile(r'delivered ([0-9.]+) g')
SHORT = re.compile(r'([0-9.]+) g short')
STOP_AT = re.compile(r'Stop (\d+) at ([0-9.]+) g landed: forecast .* = ([0-9.]+) g')
SETTLED = re.compile(r'Stop (\d+) settled at ([0-9.]+) g: ([0-9.]+) g came after it, '
                     r'forecast ([0-9.]+) g')
#: Endings that say nothing about the pour: the tree took the goal away, or
#: something outside it moved the pan (a hand on the receiver reads as a jump).
NOT_THE_POUR = ('cancelled', 'deactivated mid-pour', 'scale reading jumped')


def wilson(k, n, z=1.959964):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def trials(paths, since, until):
    out = []
    for path in paths:
        cur = None
        for line in open(path, errors='ignore'):
            m = STAMP.search(line)
            if not m:
                continue
            t, msg = float(m.group(1)), m.group(2)
            if not since <= t <= until:
                continue
            s = START.search(msg)
            if s:
                if cur is not None:
                    cur['end'] = 'no outcome logged'
                    out.append(cur)
                cur = {'log': path, 't0': t, 'target': float(s.group(1)),
                       'material': 'granular' if s.group(2).startswith('granular') else 'liquid',
                       'flow_index': float(s.group(3)), 'container': float(s.group(4)),
                       'tol': float(s.group(5)), 'stops': 0, 'settles': [],
                       'first_stop': None, 'first_after': None, 'first_forecast': None}
                continue
            if cur is None:
                continue
            a = STOP_AT.search(msg)
            if a:
                cur['stops'] = max(cur['stops'], int(a.group(1)))
                if a.group(1) == '1':
                    cur['first_stop'] = float(a.group(2))
                continue
            st = SETTLED.search(msg)
            if st:
                cur['settles'].append(float(st.group(2)))
                if st.group(1) == '1':
                    cur['first_after'] = float(st.group(3))
                    cur['first_forecast'] = float(st.group(4))
                continue
            p = POURED.search(msg)
            if p and '[/controller_action_server' in msg:
                cur.update(delivered=float(p.group(1)), seconds=float(p.group(2)),
                           peak_tilt=float(p.group(3)), trims=int(p.group(4)), ok=True,
                           end='poured', t1=t)
                out.append(cur)
                cur = None
                continue
            if msg.startswith('[/controller_action_server') and 'Aborted:' in msg:
                reason = msg.split('Aborted:', 1)[1].strip()
                d = DELIVERED.search(reason)
                if d:
                    delivered = float(d.group(1))
                elif SHORT.search(reason):
                    delivered = cur['target'] - float(SHORT.search(reason).group(1))
                else:
                    delivered = cur['settles'][-1] if cur['settles'] else float('nan')
                cur.update(delivered=delivered, seconds=t - cur['t0'], peak_tilt=float('nan'),
                           trims=max(0, cur['stops'] - 1), ok=False, end=reason, t1=t)
                out.append(cur)
                cur = None
        if cur is not None:
            cur['end'] = 'no outcome logged'
            out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--since', type=float, default=0.0, help='epoch seconds')
    ap.add_argument('--until', type=float, default=float('inf'), help='epoch seconds')
    ap.add_argument('--csv', default='', help='write one row per trial here')
    a = ap.parse_args()

    rows = trials(a.logs, a.since, a.until)
    if not rows:
        sys.exit('no pours in %s' % ', '.join(a.logs))
    print('%-3s %-9s %6s %9s %7s %6s %5s %5s  %s' % (
        '#', 'material', 'target', 'delivered', 'error', 'time', 'stops', '1st', 'outcome'))
    for i, r in enumerate(rows, 1):
        delivered = r.get('delivered', float('nan'))
        print('%-3d %-9s %6g %9.2f %+7.2f %6.1f %5d %5s  %s' % (
            i, r['material'], r['target'], delivered, delivered - r['target'],
            r.get('seconds', float('nan')), r['stops'],
            '%.2f' % r['first_stop'] if r['first_stop'] is not None else '-',
            r['end'][:70]))

    print()
    groups = {}
    for r in rows:
        if 'delivered' not in r or any(k in r['end'] for k in NOT_THE_POUR) or \
                r['end'] == 'no outcome logged':
            continue
        groups.setdefault((r['material'], r['flow_index'], r['target']), []).append(r)
    for (material, flow, target), g in sorted(groups.items()):
        d = [r['delivered'] for r in g if math.isfinite(r['delivered'])]
        err = [x - target for x in d]
        k = sum(1 for r in g if math.isfinite(r['delivered'])
                and abs(r['delivered'] - target) <= r['tol'])
        lo, hi = wilson(k, len(g))
        secs = [r['seconds'] for r in g]
        print('%s (flow_index %.2f), %g g: n=%d' % (material, flow, target, len(g)))
        print('   delivered  %.2f +- %.2f g (mean +- sd), error %+.2f g, |error| mean %.2f, '
              'worst %.2f g' % (statistics.mean(d), statistics.stdev(d) if len(d) > 1 else 0.0,
                                 statistics.mean(err), statistics.mean(abs(e) for e in err),
                                 max(abs(e) for e in err)))
        print('   within %.2f g: %d/%d = %.0f%% (Wilson 95%% %.0f-%.0f%%)' % (
            g[0]['tol'], k, len(g), 100.0 * k / len(g), 100 * lo, 100 * hi))
        print('   time %.1f +- %.1f s, trim pulses %.1f on average, %d of %d aborted by the '
              'controller (over, short or out of time)' % (
                  statistics.mean(secs), statistics.stdev(secs) if len(secs) > 1 else 0.0,
                  statistics.mean(r['trims'] for r in g), sum(1 for r in g if not r['ok']),
                  len(g)))
    left = [r for r in rows if r not in [x for g in groups.values() for x in g]]
    if left:
        print('\nleft out (not ended by the pour): %d' % len(left))
        for r in left:
            print('   %g g at %.0f: %s' % (r['target'], r['t0'], r['end'][:90]))

    if a.csv:
        keys = ['log', 't0', 't1', 'material', 'flow_index', 'target', 'tol', 'container',
                'delivered', 'seconds', 'peak_tilt', 'stops', 'trims', 'first_stop',
                'first_after', 'first_forecast', 'ok', 'end']
        with open(a.csv, 'w', newline='') as handle:
            w = csv.DictWriter(handle, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print('\nwrote %s' % a.csv)


if __name__ == '__main__':
    main()
