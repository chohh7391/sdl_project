#!/usr/bin/env python3
"""Measure where vessels stand on the real cell, from this machine's cameras.

    measure_vessels.py beaker1 beaker2 [--seconds 4] [--min-samples 10] [--max-spread-mm 5]
    measure_vessels.py beaker2=/perception/object_pose/some_other_topic

Prints one line per vessel, ``NAME X Y`` in base_link metres, on stdout and
everything else on stderr, so plan_remote.sh --perceive can read the result.

The poses come from cho_object_pose (the AprilTag pose node), which already
applies each tag's mount offset: its xy IS the vessel centre. Only x and y are
taken -- the z it reports is a standoff above the bench, not the vessel -- which
is also all the planner's measured layout uses (SDL_LAYOUT_XY).

Default topics:
  beaker1  /perception/object_pose/held_beaker   pour_vessel.yaml's tag 0: the
           beaker that is held and poured, so the same pose node serves the
           replay's pour marker and this measurement
  beaker2  /perception/object_pose/beaker2

A vessel is reported only if it gave --min-samples poses within --seconds, all
within --max-spread-mm of their median, in base_link. Anything else exits 1 and
names the vessel: a plan against a pose that is still moving, or that one camera
alone jumps between, is a plan against a guess.
"""
import argparse
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy

DEFAULT_TOPICS = {
    "beaker1": "/perception/object_pose/held_beaker",
    "beaker2": "/perception/object_pose/beaker2",
}
FRAME = "base_link"
DISCOVERY_S = 6.0


def median(values):
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("vessels", nargs="+", help="NAME or NAME=TOPIC")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--max-spread-mm", type=float, default=5.0)
    a = ap.parse_args()

    topics = {}
    for item in a.vessels:
        name, _, topic = item.partition("=")
        topic = topic or DEFAULT_TOPICS.get(name)
        if not topic:
            ap.error("no default topic for %r; give it as %s=/topic" % (name, name))
        topics[name] = topic

    rclpy.init()
    node = rclpy.create_node("measure_vessels")
    samples = {name: [] for name in topics}
    frames = {name: set() for name in topics}
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

    def callback(name):
        def on_pose(msg):
            frames[name].add(msg.header.frame_id)
            samples[name].append((msg.pose.position.x, msg.pose.position.y))
        return on_pose

    subs = [node.create_subscription(PoseStamped, topic, callback(name), qos)
            for name, topic in topics.items()]
    # DDS discovery alone can take a couple of seconds, and a window that opens
    # before it finishes reports a vessel as unseen. Open it once every topic has
    # a publisher, or after DISCOVERY_S whatever the state.
    discovery = time.monotonic() + DISCOVERY_S
    while time.monotonic() < discovery and not all(
            node.count_publishers(t) for t in topics.values()):
        rclpy.spin_once(node, timeout_sec=0.05)
    for name in samples:
        samples[name].clear()
        frames[name].clear()
    deadline = time.monotonic() + a.seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    del subs
    node.destroy_node()
    rclpy.shutdown()

    ok, out = True, []
    for name, topic in topics.items():
        pts = samples[name]
        if frames[name] - {FRAME}:
            print("%s: poses on %s are in %s, not %s" % (name, topic, sorted(frames[name]), FRAME),
                  file=sys.stderr)
            ok = False
            continue
        if len(pts) < a.min_samples:
            print("%s: %d pose(s) on %s in %.0f s, need %d -- is the tag seen, and the pose "
                  "node running on the table that names it?"
                  % (name, len(pts), topic, a.seconds, a.min_samples), file=sys.stderr)
            ok = False
            continue
        mx, my = median([p[0] for p in pts]), median([p[1] for p in pts])
        spread = max(math.hypot(x - mx, y - my) for x, y in pts) * 1e3
        print("%s: (%.4f, %.4f) from %d poses on %s, spread %.1f mm"
              % (name, mx, my, len(pts), topic, spread), file=sys.stderr)
        if spread > a.max_spread_mm:
            print("%s: spread %.1f mm exceeds %.1f mm; not reporting a pose that moves"
                  % (name, spread, a.max_spread_mm), file=sys.stderr)
            ok = False
            continue
        out.append("%s %.4f %.4f" % (name, mx, my))
    if not ok:
        return 1
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
