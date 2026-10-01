#!/usr/bin/env python3
"""Put the real-cell plan into rviz beside what the robot and cameras see.

    monitor_node.py [--runs-dir ~/real_cell_runs] [--frame base_link]

Run by scripts/real_cell/monitor.sh, which also starts rviz on
scripts/real_cell/rviz/real_cell.rviz. The robot model, the camera images and
the perceived beakers (/perception/object_markers) come from the bringup and
the pose node; this node adds what only the plan knows:

  /real_cell/planned_path  MarkerArray  the grasp frame along the trajectory,
                                        coloured by operation, with the gripper
                                        closing (green) and opening (red), and
                                        the rim the pour goes into (magenta):
                                        the perceived receiver plan_remote.sh
                                        hands the pouring controller
  /real_cell/status        Marker       the pipeline's stage (plan_remote.sh
                                        writes RUNS/STATUS), the balance reading,
                                        and how far each perceived beaker is from
                                        where the plan put it

  /real_cell/fixed_scene   MarkerArray  the cell's FIXED furniture, which is not
                                        perceived: the riser bar under beaker1
                                        and the balance under beaker2, as boxes,
                                        where the planner puts them

No vessel is drawn from the plan or from any preset layout: the beakers in
rviz are the perceived ones only, and the plan is shown as the path it drives.

The trajectory shown is the one plan_remote.sh last selected: the file stem in
RUNS/CURRENT. The node rereads it whenever that file changes, so a new plan
appears in rviz as soon as it has been fetched and checked.

Heights: the meta records each object's xy (and yaw); z comes from the run's
SCENE_ENV.txt -- bench, riser and balance heights -- in the planner's world,
which the real cell treats as base_link (the bench offset was measured from
the robot base's plane). Display only: nothing here is sent to the robot.
"""
import argparse
import csv
import json
import math
import os
import sys

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "trials"))
sys.path.insert(0, HERE)

#: The real glassware, drawn at the size it IS (the planner's source beaker is
#: modelled shorter on purpose; that is a planning device, not the object).
BODY = {"beaker": (0.050, 0.070), "beaker2": (0.050, 0.070), "flask": (0.087, 0.160)}
LABEL = {"beaker": "beaker1", "beaker2": "beaker2", "flask": "flask"}
PERCEIVED = {"beaker": "/perception/object_pose/held_beaker",
             "beaker2": "/perception/object_pose/beaker2"}
OP_COLOUR = {"Pick": (0.3, 0.6, 1.0), "Move_to_Surface": (1.0, 0.85, 0.2),
             "pouring": (1.0, 0.25, 0.25), "Place": (0.3, 1.0, 0.4)}
PLAN_RGBA = (0.3, 0.95, 0.4, 0.35)


def colour(rgb, a=1.0):
    return ColorRGBA(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=float(a))


#: The receiver's outer rim radius, as the pouring controller is configured
#: (controllers.yaml geometry.receiver_radius).
RIM_RADIUS = 0.025


#: The two-beaker cell's furniture as record_transfer_real.sh plans it (author,
#: 2026-09-29). It does not move, so it is drawn whether or not a plan is up.
CELL_ENV = {"SDL_TABLE_Z_M": "-0.013", "SDL_BEAKER_RISER_M": "0.072",
            "SDL_BEAKER_RISER_XY_M": "0.200", "SDL_BEAKER_RISER_Y_M": "0.070",
            "SDL_RISER_XY": "0.360,-0.300", "SDL_SCALE_TOP_M": "0.092",
            "SDL_SCALE_BODY_BEARING_DEG": "180"}
PAN_XY = (0.520, 0.320)
FURNITURE_RGBA = (0.62, 0.62, 0.66, 0.55)


def fixed_scene(frame, stamp):
    """The riser bar and the balance, as the planner's boxes (envs/constants.py)."""
    for key, value in CELL_ENV.items():
        os.environ.setdefault(key, value)
    sys.path.insert(0, os.path.join(HERE, "..", "..", "TAMP", "tamp", "src"))
    from envs import constants as C
    bench = C.TABLE_Z_OFFSET
    (sx, sy), syaw = C.scale_pose_from_pan(PAN_XY)
    boxes = [("riser", C.riser_xy(), 0.0, C.beaker_riser_dims()),
             ("scale", (sx, sy), syaw, C.scale_dims())]
    markers = []
    for k, (name, (x, y), yaw, dims) in enumerate(boxes):
        m = Marker()
        m.header.frame_id, m.header.stamp = frame, stamp
        m.ns, m.id, m.type, m.action = "fixed cell", k, Marker.CUBE, Marker.ADD
        m.pose.position.x, m.pose.position.y = float(x), float(y)
        m.pose.position.z = bench + float(dims[2]) / 2.0
        m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        m.scale.x, m.scale.y, m.scale.z = (float(v) for v in dims)
        m.color = colour(FURNITURE_RGBA[:3], FURNITURE_RGBA[3])
        markers.append(m)
    return MarkerArray(markers=markers)


def receiver_rim(run_dir):
    """The rim centre plan_remote.sh sends with the pour (REQUEST.env), or None."""
    try:
        for line in open(os.path.join(run_dir, "REQUEST.env")):
            if line.startswith("RECEIVER_RIM="):
                xyz = [float(v) for v in line.split("=", 1)[1].strip().split(",")]
                return tuple(xyz) if len(xyz) == 3 else None
    except (OSError, ValueError):
        pass
    return None


def scene_env(run_dir):
    env = {}
    try:
        for line in open(os.path.join(run_dir, "SCENE_ENV.txt")):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                env[k] = v
    except OSError:
        pass
    return env


class Monitor:
    def __init__(self, node, runs_dir, frame):
        self.node, self.runs_dir, self.frame = node, runs_dir, frame
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        # One message per topic: a latched topic keeps only its last message,
        # so the path and its gripper events travel together.
        self.pub_path = node.create_publisher(MarkerArray, "/real_cell/planned_path", latched)
        self.pub_status = node.create_publisher(Marker, "/real_cell/status", 10)
        # Latched: an rviz started later still gets the furniture.
        self.pub_fixed = node.create_publisher(MarkerArray, "/real_cell/fixed_scene", latched)
        self.pub_fixed.publish(fixed_scene(frame, node.get_clock().now().to_msg()))
        self.current, self.current_mtime = None, None
        self.planned_xy, self.plan_label = {}, "no plan selected yet"
        self.receiver_rim = None
        self.perceived, self.scale = {}, None
        for name, topic in PERCEIVED.items():
            node.create_subscription(
                PoseStamped, topic,
                lambda m, n=name: self.perceived.__setitem__(
                    n, (m.pose.position.x, m.pose.position.y, node.get_clock().now())), 10)
        try:
            from cho_interfaces.msg import ScaleReading
            from rclpy.qos import qos_profile_sensor_data
            node.create_subscription(ScaleReading, "/scale/reading",
                                     lambda m: setattr(self, "scale", (m.grams, m.stable)),
                                     qos_profile_sensor_data)
        except ImportError:
            node.get_logger().warn("cho_interfaces not sourced: no balance reading in the status")
        node.create_timer(1.0, self.reload_if_changed)
        node.create_timer(0.5, self.publish_status)

    # --------------------------------------------------------------- the plan
    def reload_if_changed(self):
        path = os.path.join(self.runs_dir, "CURRENT")
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            # No plan selected (plan_remote.sh has not chosen one, or it was
            # cleared): take the last one off the screen rather than leave a
            # stale plan that reads as if something were there.
            if self.current is not None or self.current_mtime is None:
                self.clear()
            return
        if mtime == self.current_mtime:
            return
        self.current_mtime = mtime
        base = open(path).read().strip()
        try:
            self.load(base)
        except Exception as err:   # a half-written or foreign file must not kill the view
            self.plan_label = "could not show %s: %s" % (os.path.basename(base), err)
            self.node.get_logger().error(self.plan_label)

    def clear(self):
        self.pub_path.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
        self.current, self.current_mtime, self.planned_xy = None, -1.0, {}
        self.plan_label = "no plan selected yet"

    def load(self, base):
        meta = json.load(open(base + ".meta.json"))
        lay = meta.get("layout_the_trajectory_assumes") or {}
        run_dir = os.path.dirname(os.path.dirname(base))
        env = scene_env(run_dir)
        bench = float(env.get("SDL_TABLE_Z_M", "0") or 0)
        riser_h = float(lay.get("riser", {}).get("dims", [0, 0, 0])[2]) if "riser" in lay else 0.0
        scale_h = float(lay.get("scale", {}).get("dims", [0, 0, 0])[2]) if "scale" in lay else 0.0
        support = {"beaker": bench + riser_h, "beaker2": bench + scale_h, "flask": bench + scale_h}

        stamp = self.node.get_clock().now().to_msg()
        markers = [Marker(action=Marker.DELETEALL)]

        def add(kind, ns, xyz, dims, yaw_deg, rgba, text=None):
            m = Marker()
            m.header.frame_id, m.header.stamp = self.frame, stamp
            m.ns, m.id, m.type, m.action = ns, len(markers), kind, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in xyz)
            half = math.radians(yaw_deg or 0.0) / 2.0
            m.pose.orientation.z, m.pose.orientation.w = math.sin(half), math.cos(half)
            m.scale.x, m.scale.y, m.scale.z = (float(v) for v in dims)
            m.color = colour(rgba[:3], rgba[3])
            if text is not None:
                m.text = text
            markers.append(m)

        self.planned_xy = {}
        for name, entry in sorted(lay.items()):
            x, y = entry["xy"]
            yaw = entry.get("yaw_deg", 0.0)
            if name in BODY:
                dia, h = BODY[name]
                z0 = support[name]
                self.planned_xy[name] = (x, y)
                add(Marker.CYLINDER, "planned vessels", (x, y, z0 + h / 2.0), (dia, dia, h), 0.0, PLAN_RGBA)
                add(Marker.TEXT_VIEW_FACING, "planned labels", (x, y, z0 + h + 0.04), (0, 0, 0.03),
                    0.0, (0.6, 1.0, 0.6, 1.0), "%s PLAN" % LABEL[name])
            elif name in ("riser", "scale"):
                d = entry["dims"]
                add(Marker.CUBE, "planned furniture", (x, y, bench + d[2] / 2.0), d, yaw, (0.6, 0.6, 0.65, 0.35))
            elif name.startswith("tag_"):
                d = entry["dims"]
                owner = name[len("tag_"):]
                z = support.get(owner, bench) - d[2] / 2.0
                add(Marker.CUBE, "planned tag plates", (x, y, z), d, yaw, (0.9, 0.9, 0.9, 0.8))
        # The planned bodies are kept for the status line's plan-vs-perception
        # distance only; nothing preset is drawn (the author, 2026-09-29).
        del markers
        self.receiver_rim = receiver_rim(run_dir)
        self.publish_path(base, meta, stamp)
        self.current = base
        self.plan_label = "plan: %s / %s" % (
            os.path.basename(run_dir), os.path.basename(base))
        self.node.get_logger().info("showing %s" % base)

    def publish_path(self, base, meta, stamp):
        from solve_scan_poses import Arm
        from score_real_trajectories import flange_of, grasp_offset_from_urdf
        import numpy as np
        # The gripper as the run was planned with it: on the adapter flange if
        # SCENE_ENV.txt says so.
        env = scene_env(os.path.dirname(os.path.dirname(base)))
        arm, grasp = Arm(), grasp_offset_from_urdf(flange=flange_of(env))
        rows = list(csv.DictReader(open(base + ".csv")))
        joints = meta.get("joint_names") or ["j1", "j2", "j3", "j4", "j5", "j6"]
        line = Marker()
        line.header.frame_id, line.header.stamp = self.frame, stamp
        line.ns, line.id, line.type, line.action = "planned path", 0, Marker.LINE_STRIP, Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.004
        times, points = [], []
        for i, r in enumerate(rows):
            if i % 2 and i != len(rows) - 1:
                continue
            q = np.array([float(r[j]) for j in joints])
            p = (arm.fk(q) @ grasp)[:3]
            points.append(p)
            times.append(float(r["t_s"]))
            line.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
            line.colors.append(colour(OP_COLOUR.get(r.get("operation", ""), (1.0, 1.0, 1.0))))
        path = [Marker(action=Marker.DELETEALL), line]
        # The gripper events as spheres on the path: where the jaws close and open.
        for k, ev in enumerate(meta.get("gripper_events", [])):
            i = min(range(len(times)), key=lambda n: abs(times[n] - float(ev["t_s"])))
            s = Marker()
            s.header.frame_id, s.header.stamp = self.frame, stamp
            s.ns, s.id, s.type, s.action = "gripper events", k + 1, Marker.SPHERE, Marker.ADD
            s.pose.position.x, s.pose.position.y, s.pose.position.z = (float(v) for v in points[i])
            s.pose.orientation.w = 1.0
            s.scale.x = s.scale.y = s.scale.z = 0.02
            s.color = colour((0.2, 1.0, 0.2) if ev["event"] == "close" else (1.0, 0.2, 0.2))
            path.append(s)
        if self.receiver_rim is not None:
            # The receiver's rim as the pour goal will carry it: a ring at the
            # rim's height, its outer radius (controllers.yaml receiver_radius).
            ring = Marker()
            ring.header.frame_id, ring.header.stamp = self.frame, stamp
            ring.ns, ring.id, ring.type, ring.action = "pour target", 100, Marker.LINE_STRIP, Marker.ADD
            ring.pose.orientation.w = 1.0
            ring.scale.x = 0.003
            ring.color = colour((1.0, 0.2, 1.0))
            cx, cy, cz = self.receiver_rim
            for k in range(33):
                a = 2.0 * math.pi * k / 32
                ring.points.append(Point(x=cx + RIM_RADIUS * math.cos(a),
                                         y=cy + RIM_RADIUS * math.sin(a), z=cz))
            path.append(ring)
        self.pub_path.publish(MarkerArray(markers=path))

    # ------------------------------------------------------------- the status
    def publish_status(self):
        lines = []
        try:
            lines.append(open(os.path.join(self.runs_dir, "STATUS")).read().strip())
        except OSError:
            lines.append("(no pipeline status yet)")
        lines.append(self.plan_label)
        if self.current is not None and self.receiver_rim is not None:
            lines.append("pour into the rim at (%.3f, %.3f, %.3f)" % self.receiver_rim)
        if self.scale is not None:
            lines.append("balance: %.2f g%s" % (self.scale[0], "" if self.scale[1] else " (settling)"))
        now = self.node.get_clock().now()
        for name, topic in PERCEIVED.items():
            seen = self.perceived.get(name)
            if seen is None or (now - seen[2]).nanoseconds > 2e9:
                lines.append("%s: not seen" % LABEL[name])
                continue
            plan = self.planned_xy.get(name)
            if plan is None:
                lines.append("%s: seen at (%.3f, %.3f)" % (LABEL[name], seen[0], seen[1]))
            else:
                lines.append("%s: %.1f mm from the plan" % (
                    LABEL[name], 1e3 * math.hypot(seen[0] - plan[0], seen[1] - plan[1])))
        m = Marker()
        m.header.frame_id, m.header.stamp = self.frame, now.to_msg()
        m.ns, m.id, m.type, m.action = "status", 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = 0.35, 0.0, 0.75
        m.pose.orientation.w = 1.0
        m.scale.z = 0.035
        m.color = colour((1.0, 1.0, 1.0))
        m.text = "\n".join(lines)
        self.pub_status.publish(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs-dir", default=os.path.expanduser("~/real_cell_runs"))
    ap.add_argument("--frame", default="base_link")
    a = ap.parse_args()
    rclpy.init()
    node = rclpy.create_node("real_cell_monitor")
    Monitor(node, os.path.abspath(a.runs_dir), a.frame)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
