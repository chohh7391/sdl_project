#!/usr/bin/env python3
"""Measure where vessels stand on the real cell, from this machine's cameras.

    measure_vessels.py beaker1 beaker2 [--seconds 4] [--min-samples 10] [--max-spread-mm 5]
                       [--max-bearing-spread-deg 5]
    measure_vessels.py beaker2=/perception/object_pose/some_other_topic

Prints one line per vessel, ``NAME X Y Z BEARING_DEG`` (base_link metres and
degrees; Z is the vessel's bottom), on stdout and everything else on stderr, so
plan_remote.sh --perceive can read the result.

The poses come from cho_object_pose (the AprilTag pose node), which already
applies each tag's mount offset: its xy IS the vessel centre, and on the
two-beaker cell's tables (cho_task_manager config/perception/pour_beakers.yaml)
its z is the tag's height, which is level with the vessel's bottom. BEARING is
the direction from the vessel's axis to its tag, i.e. which way the mount's
plate points: the tables put the vessel along the tag's own -y as printed, so
the tag lies along +y of the tag's heading.

The heading is read from the body the pose node draws for the vessel
(/perception/object_markers), NOT from the published pose. The pose carries a
GRASP orientation, whose yaw is folded into (-90, 90] deg because a gripper
does not care which way round it holds a vessel -- so a tag near +-90 deg of
yaw comes out as either, and the plate bearing flips by 180 deg from one
camera to the next (seen 2026-09-29: side_1 178.5 deg, the wrist 0.5 deg, for
a plate on the robot's side). The drawn body is oriented by the UNFOLDED
heading the offset is applied in. Without markers it falls back to the pose,
and says the bearing is then only known modulo 180 deg.

Default topics:
  beaker1  /perception/object_pose/held_beaker   the beaker that is held and
           poured: the pouring controller reads its marker from this topic
           (controllers.yaml vessel_pose_topic), so the table publishes the
           source beaker's tag there and one pose node serves both
  beaker2  /perception/object_pose/beaker2

A vessel is reported only if it gave --min-samples poses within --seconds, all
within --max-spread-mm of their median, in base_link. Anything else exits 1 and
names the vessel: a plan against a pose that is still moving, or that one camera
alone jumps between, is a plan against a guess. The vessels that DID pass are
printed either way, so a caller can tell which ones to go and look for.

--latch is for an occlusion-recovery sweep, where a tag is in view only while
the wrist camera dwells over it: each vessel is reported as soon as its last
--min-samples poses agree within the same limits, and the wait ends when every
vessel has latched or after --seconds.

--camera CAM counts only the poses published while CAM itself measures the tag
(the pose node's /perception/object_visibility says CAM is OK for that vessel,
at --min-margin or better). The pose node goes on publishing a vessel it cannot
see -- the pose it last measured, held while nothing moves it -- and the standing
cameras may still publish one of their own; a recovery that is re-measuring the
vessel with the wrist camera wants neither, only what the wrist saw.
"""
import argparse
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

MARKER_TOPIC = "/perception/object_markers"
#: A drawn body older than this does not describe the pose just in.
MARKER_FRESH_S = 0.5

VISIBILITY_TOPIC = "/perception/object_visibility"
#: A visibility snapshot older than this says nothing about the pose just in.
VISIBILITY_FRESH_S = 0.5

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


def assess(pts):
    """(x, y, z, spread_mm, bearing_deg, bearing_spread_deg) of a set of poses."""
    mx, my = median([p[0] for p in pts]), median([p[1] for p in pts])
    mz = median([p[2] for p in pts])
    spread = max(math.hypot(p[0] - mx, p[1] - my) for p in pts) * 1e3
    bearing = math.degrees(math.atan2(sum(p[4] for p in pts), sum(p[3] for p in pts)))
    bspread = max(abs((math.degrees(math.atan2(p[4], p[3])) - bearing + 180.0) % 360.0 - 180.0)
                  for p in pts)
    return mx, my, mz, spread, bearing, bspread


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("vessels", nargs="+", help="NAME or NAME=TOPIC")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--max-spread-mm", type=float, default=5.0)
    ap.add_argument("--max-bearing-spread-deg", type=float, default=5.0)
    ap.add_argument("--latch", action="store_true",
                    help="report each vessel once its last --min-samples poses agree; "
                         "wait up to --seconds for all of them")
    ap.add_argument("--camera", default="",
                    help="count only poses published while this camera measures the tag")
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="with --camera: the decode margin that camera must have")
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

    # name -> (monotonic time, whether --camera measured it) from the pose
    # node's visibility snapshots; only read with --camera.
    measured_by = {}
    # name -> (monotonic time, unfolded tag heading [rad]) from the body the
    # pose node draws for the vessel, published with every pose.
    heading = {}
    folded = set()

    def on_markers(msg):
        drawn = [m for m in msg.markers if m.ns == "object_pose"]
        for body in drawn:
            if body.type == Marker.TEXT_VIEW_FACING:
                continue
            label = next((m for m in drawn if m.id == body.id + 1
                          and m.type == Marker.TEXT_VIEW_FACING), None)
            name = label.text.split(" ")[0] if label is not None else None
            if name in topics:
                q = body.pose.orientation
                heading[name] = (time.monotonic(), 2.0 * math.atan2(q.z, q.w))

    def on_visibility(msg):
        for entry in msg.objects:
            if entry.name not in topics:
                continue
            ok = any(view.camera == a.camera and view.state == view.STATE_OK
                     and view.decision_margin >= a.min_margin for view in entry.cameras)
            measured_by[entry.name] = (time.monotonic(), ok and entry.publishing)

    def callback(name):
        def on_pose(msg):
            if a.camera:
                seen = measured_by.get(name)
                if seen is None or time.monotonic() - seen[0] > VISIBILITY_FRESH_S or not seen[1]:
                    return
            frames[name].add(msg.header.frame_id)
            seen = heading.get(name)
            if seen is not None and time.monotonic() - seen[0] <= MARKER_FRESH_S:
                ax, ay = math.cos(seen[1]), math.sin(seen[1])
            else:
                # The grasp orientation: its x axis carries the FOLDED yaw.
                folded.add(name)
                q = msg.pose.orientation
                ax, ay = 1.0 - 2.0 * (q.y * q.y + q.z * q.z), 2.0 * (q.x * q.y + q.w * q.z)
            # The heading's +y (z up), along which the tag lies from the vessel.
            samples[name].append((msg.pose.position.x, msg.pose.position.y,
                                  msg.pose.position.z, -ay, ax))
        return on_pose

    # Subscribed before the poses, so the first pose already has its body.
    subs = [node.create_subscription(MarkerArray, MARKER_TOPIC, on_markers, QoSProfile(depth=10))]
    subs += [node.create_subscription(PoseStamped, topic, callback(name), qos)
             for name, topic in topics.items()]
    if a.camera:
        from cho_interfaces.msg import ObjectVisibilityArray
        subs.append(node.create_subscription(ObjectVisibilityArray, VISIBILITY_TOPIC,
                                             on_visibility, QoSProfile(depth=1)))
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
    latched = {}
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if not a.latch:
            continue
        for name, pts in samples.items():
            if name in latched or len(pts) < a.min_samples:
                continue
            recent = pts[-a.min_samples:]
            if assess(recent)[3] <= a.max_spread_mm and assess(recent)[5] <= a.max_bearing_spread_deg:
                latched[name] = list(recent)
                print("%s: latched after %d poses" % (name, len(pts)), file=sys.stderr)
        if len(latched) == len(topics):
            break
    if a.latch:
        for name in samples:
            samples[name] = latched.get(name, samples[name])
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
        mx, my, mz, spread, bearing, bspread = assess(pts)
        outward = math.degrees(math.atan2(my, mx))
        off = (bearing - outward + 180.0) % 360.0 - 180.0
        print("%s: (%.4f, %.4f) from %d poses on %s, spread %.1f mm; base at z=%.4f; tag plate "
              "at bearing %.1f deg (+-%.1f), %.0f deg from straight away from the robot"
              % (name, mx, my, len(pts), topic, spread, mz, bearing, bspread, off), file=sys.stderr)
        if name in folded:
            print("%s: WARNING no drawn body on %s for some poses; their plate bearing is the "
                  "folded grasp yaw's, known only modulo 180 deg" % (name, MARKER_TOPIC),
                  file=sys.stderr)
        if spread > a.max_spread_mm or bspread > a.max_bearing_spread_deg:
            print("%s: spread %.1f mm / %.1f deg exceeds %.1f mm / %.1f deg; not reporting a pose "
                  "that moves" % (name, spread, bspread, a.max_spread_mm, a.max_bearing_spread_deg),
                  file=sys.stderr)
            ok = False
            continue
        if abs(off) > 90.0:
            print("%s: WARNING its tag plate points back toward the robot, where the arm comes "
                  "from; the planner routes around it, but turn it away if the plan fails"
                  % name, file=sys.stderr)
        out.append("%s %.4f %.4f %.4f %.1f" % (name, mx, my, mz, bearing))
    if out:
        print("\n".join(out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
