#!/usr/bin/env python3
"""Record one trial's commanded joint trajectory for replay on the real robot.

Everything the arm does passes through /isaac_arm_commands, the pour included --
the lip-pivot path is walked inside tamp_server and published there like any
other motion -- so recording that topic captures the whole trial without
touching the server.

The gripper is a SERVICE (isaac_gripper_commands), not a topic, so its opens and
closes cannot be recorded directly. /isaac_joint_states carries the gripper
joint, so the events are recovered from the achieved state instead, and
/tamp_current_op marks where each operation begins.

The file is written self-describing on purpose. A trajectory is only valid for
the layout it was planned against: replayed against vessels in different places
it will pour into the bench. The object poses the simulator spawned are stored
alongside the waypoints so the real cell can be set up to match, or the mismatch
noticed before anything moves.
"""
import argparse, json, re, sys, time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class Recorder(Node):
    def __init__(self):
        super().__init__("trajectory_recorder")
        self.commands = []      # commanded arm stream -- what gets replayed
        self.states = []        # achieved states, for the gripper joint
        self.ops = []           # operation markers
        self.t0 = time.time()
        # The state stream is high-rate and handling it in Python competes with
        # planning: recording it from the start took one seed's optimisation
        # loop from 64 s to 245 s and the driver timed the plan out. Only the
        # two low-rate topics are subscribed up front; the state stream is added
        # once the first arm command arrives, which is after planning is done.
        self.create_subscription(JointState, "isaac_arm_commands", self.on_cmd, 50)
        self.create_subscription(String, "tamp_current_op", self.on_op, 10)
        self._state_sub = None

    def _t(self):
        return time.time() - self.t0

    def on_cmd(self, m):
        self.commands.append({"t": round(self._t(), 4),
                              "names": list(m.name),
                              "positions": [float(v) for v in m.position]})
        if self._state_sub is None:
            self._state_sub = self.create_subscription(
                JointState, "isaac_joint_states", self.on_state, 10)
            self.get_logger().info("execution started; recording joint states too")

    def on_state(self, m):
        # subsample: the state stream is much faster than the command stream and
        # only the gripper joint is needed from it
        if self.states and self._t() - self.states[-1]["t"] < 0.02:
            return
        self.states.append({"t": round(self._t(), 4),
                            "names": list(m.name),
                            "positions": [float(v) for v in m.position]})

    def on_op(self, m):
        self.ops.append({"t": round(self._t(), 4), "op": m.data})
        self.get_logger().info("op -> %s" % m.data)


LAYOUT = re.compile(r"\[Task\]\s+(beaker|flask|magnet|box|stirrer)\s+"
                    r"xy=\(([-\d.]+),([-\d.]+)\)\s+yaw=([-\d.]+)deg")
HOME = re.compile(r"\[Task\]\s+home_arm\(rad\) = \[([-\d.,\s]+)\]")


def read_layout(path):
    objs, home = {}, None
    try:
        for ln in open(path, errors="ignore"):
            m = LAYOUT.search(ln)
            if m:
                objs[m.group(1)] = {"xy": [float(m.group(2)), float(m.group(3))],
                                    "yaw_deg": float(m.group(4))}
            m = HOME.search(ln)
            if m and home is None:
                home = [float(v) for v in m.group(1).split(",")]
    except OSError:
        pass
    return objs, home


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--sim-log", default=None,
                    help="the trial's sim log, to store the layout it was planned against")
    ap.add_argument("--seed", default="")
    ap.add_argument("--tool", default="")
    ap.add_argument("--stop-after-idle", type=float, default=25.0,
                    help="stop once no arm command has arrived for this long")
    a = ap.parse_args()

    rclpy.init()
    n = Recorder()
    n.get_logger().info("recording; waiting for arm commands...")
    last_n, last_change = 0, time.time()
    started = False
    while time.time() - n.t0 < a.seconds:
        rclpy.spin_once(n, timeout_sec=0.05)
        if len(n.commands) != last_n:
            last_n = len(n.commands)
            last_change = time.time()
            started = True
        if started and time.time() - last_change > a.stop_after_idle:
            n.get_logger().info("no arm command for %.0f s; stopping" % a.stop_after_idle)
            break

    objs, home = read_layout(a.sim_log) if a.sim_log else ({}, None)
    doc = {
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": a.seed,
        "tool": a.tool,
        "source": "Isaac Sim trial; commanded stream from /isaac_arm_commands",
        "warning": ("valid only for the layout below -- the trajectory is "
                    "position-controlled and does not sense the objects"),
        "layout": objs,
        "home_arm_rad": home,
        "operations": n.ops,
        "commands": n.commands,
        "states_subsampled": n.states,
    }
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print("wrote %s: %d commands, %d states, %d operation markers"
          % (a.out, len(n.commands), len(n.states), len(n.ops)))
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
