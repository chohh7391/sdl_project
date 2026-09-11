#!/usr/bin/env python3
"""Canonical task orchestrator (ORCHESTRATION.md B1/B2/B3).

Runs against an ALREADY-RUNNING, freshly-started Isaac Sim (with SDL_SEED=<seed>
baked into its randomized layout) + a FRESH tamp_server (one set_tamp_cfg per
server lifetime, per the layer-3 constraint). Drives one trial end to end:

    tool_change ag95  (sim: switch gripper so the fixed-joint grasp fires)
    set_tamp_cfg fr5_ag95   (tamp_server: cuTAMP config -- ONCE per server)
    set_tamp_env transfer   (snapshot the randomized GT poses into cuTAMP)
    tamp_plan               (measure plan_success / #satisfying / planning_time)
    plan_execute            (measure execute_success + max transport tilt)

then appends ONE CSV row. Numbers written are the real service results (never
fabricated). While plan_execute runs, the node keeps spinning so it receives
/carried_tilt_deg (sim: attached-vessel tilt from upright, deg) and
/tamp_current_op (tamp_server: op currently executing) and tracks the MAX tilt
seen during the MoveHolding transport phase specifically (R3#2 theta_max).

Env: conda `sdl` (py3.10) + system Humble + colcon overlay, ROS_DOMAIN_ID=100.
"""
import argparse
import csv
import math
import os
import time
import datetime
import pathlib
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

from tamp_interfaces.srv import Plan, Execute, SetTampEnv, SetTampCfg, ToolChange
from simulation_interfaces.srv import GetEntityState
from std_msgs.msg import Float32, Float32MultiArray, String

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "TAMP" / "tamp" / "src"))
from orchestration.registry import get_environment_spec, get_planner_spec

CSV_HEADER = [
    "timestamp", "seed", "task", "robot_cfg",
    "plan_success", "planning_time_s", "num_satisfying",
    "execute_success", "max_transport_tilt_deg", "failure_reason",
    "beaker_xy", "flask_xy",
    # Outcome of the task itself, measured from the settled scene AFTER release.
    # `execute_success` only says the trajectory ran to completion -- a trial that
    # ends with the vessel lying on its side still sets it True -- so it cannot be
    # used as a task-success rate on its own.
    "poured", "final_tilt_deg", "final_xy", "placed_upright", "placed_in_goal",
    "task_success",
    # Which object the outcome was measured on, how far its settled centre ended
    # up from the goal centre, and the task-specific extra condition (Stir's stir
    # bar inside the vessel). Recorded raw so a tighter acceptance threshold can
    # be applied later without re-running the batch.
    "target_obj", "goal_err_mm", "aux_check",
    # Where the pour would actually have landed: the carried vessel's lip at the
    # instant of maximum tilt, as a horizontal distance from the target vessel's
    # axis. RECORDED ONLY -- it is not yet part of task_success, because the
    # threshold should be set from the measured distribution rather than before
    # it (the target vessel's real mouth radius is about 17 mm).
    "pour_peak_tilt_deg", "pour_lip_err_mm",
]

# Goal region for the transfer task, mirroring the xy that
# TAMP/tamp/src/envs/transfer.py gives entities["goal_region"] (0.35, -0.35),
# with ENTITIES["goal_region"].dims = [0.1, 0.1, 0.01] in TAMP/tamp/src/envs/utils.py.
# (Its z is derived there from the table top and the planner z-lift; only the xy
# extent matters for scoring.)
GOAL_REGION_XY = (0.35, -0.35)
GOAL_REGION_HALF_M = 0.05
# A free-standing vessel on a flat surface is not in static equilibrium at a large
# tilt: once released it either settles upright or falls over. This threshold
# separates "standing" from "fell over" in the settled scene; it is NOT the
# transport bound (which is reported separately as max_transport_tilt_deg).
UPRIGHT_AFTER_RELEASE_DEG = 30.0
SETTLE_AFTER_EXECUTE_S = 2.0

# Per-task outcome criteria. Each is read off the environment's goal_state in
# TAMP/tamp/src/envs/<task>.py, and each names the goal by the entity that
# physically defines it, so a randomized layout moves the goal with it (the
# Stir goal rides on the stirrer, which IS randomized; the Move tray is not).
#
# `placed_in_goal` uses the same rule for every task -- the target object's
# settled centre lies inside the goal region's footprint -- while `goal_err_mm`
# records the raw distance, so the manuscript's tighter Move criterion (final
# placement within a stated tolerance of the target) can be applied to the
# recorded numbers without another batch.
TASK_OUTCOMES = {
    "transfer": {
        "target": "beaker",
        # transfer.py: entities["goal_region"] is fixed at (0.35, -0.35) with
        # ENTITIES dims [0.1, 0.1, 0.01]. Same rule B3/B4 were scored with.
        "goal_entity": None,
        "goal_xy": GOAL_REGION_XY,
        "half_xy": (GOAL_REGION_HALF_M, GOAL_REGION_HALF_M),
        "require_pour": True,
        "aux": None,
        # transfer.py pours into movables[1]; pour_region is placed at its xy.
        "pour_target": "flask",
    },
    "move": {
        "target": "box",
        # move.py places the box on box_region, which is the goal tray's own
        # footprint (0.15 x 0.15 m static collider spawned by task.py).
        "goal_entity": "box_goal",
        "half_xy": (0.075, 0.075),
        "require_pour": False,
        "aux": None,
    },
    "stir": {
        "target": "flask",
        # stir.py places the vessel on entities["goal_region"], which is put on
        # top of the stirrer; ENTITIES dims [0.1, 0.1, 0.01].
        "goal_entity": "stirrer",
        "half_xy": (0.05, 0.05),
        "require_pour": False,
        # ... and drops the magnet into that vessel (On + OnBeaker).
        "aux": "magnet_in_vessel",
    },
}
# The stir bar counts as inside the vessel when its centre lies within the
# vessel's CLEAR OPENING and below the vessel's mouth. The flask is a hollow
# collider of 0.07 m outer width with 3 mm walls (isaacsim Task.FLASK_WALL_M /
# envs/constants.VESSEL_WALL_M), so the opening is 0.064 m across.
MAGNET_IN_VESSEL_RADIUS_M = 0.032

# Transport = the vessel is grasped (carried) and NOT actively pouring (R3#2).
# NOTE: cuTAMP folds each MoveHolding's motion into the following operator's
# trajectory (Move_to_Surface / Place), so no executed plan step is literally
# named "MoveHolding" -- the carry motion runs under Move_to_Surface and Place.
# We therefore measure transport tilt as the max carried-vessel tilt over every
# attached instant EXCEPT the dedicated pour sub-phase (op == "pouring") and the
# post-run "idle" marker. This is exactly the "does the carried vessel stay
# upright while being transported?" quantity R3#2 asks about.
NON_TRANSPORT_OPS = {"pouring", "idle"}


class TaskOrchestrator(Node):
    def __init__(self):
        super().__init__("sdl_task_orchestrator")
        self.current_op = None
        self.max_transport_tilt = -1.0   # deg; -1 => never observed in transport
        self.max_any_tilt = -1.0         # deg over the whole attached period (diag)
        self.carried_obj = ""            # which object those samples belong to
        # Transport tilt is the SPILL constraint, so it is only meaningful for the
        # task's target vessel. Stir carries the vessel and then the stir bar, and
        # pooling them reported the bar's 30 deg as a vessel tilt.
        self.tilt_target = None
        self.max_tilt_by_obj = {}
        # Where the carried vessel's pouring lip is, and where it was at the
        # instant of MAXIMUM tilt during the pour -- that is the moment the
        # stream would be running, so it is the only instant at which "is the
        # pour actually over the target vessel?" has an answer.
        self.carried_lip_xy = None
        self.pour_peak_tilt = -1.0
        self.pour_peak_lip_xy = None
        self.saw_pour = False            # did the explicit pour step actually execute?

        latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
        )
        self.create_subscription(String, "tamp_current_op", self._op_cb, latched)
        self.create_subscription(Float32, "carried_tilt_deg", self._tilt_cb, 10)
        self.create_subscription(String, "carried_obj", self._carried_obj_cb, 10)
        self.create_subscription(Float32MultiArray, "carried_lip_xy", self._lip_cb, 10)

        self.tool_change_cli = self.create_client(ToolChange, "tool_change")
        self.cfg_cli = self.create_client(SetTampCfg, "set_tamp_cfg")
        self.env_cli = self.create_client(SetTampEnv, "set_tamp_env")
        self.plan_cli = self.create_client(Plan, "tamp_plan")
        self.exec_cli = self.create_client(Execute, "plan_execute")
        self.ges_cli = self.create_client(GetEntityState, "get_entity_state")

    # --- topic callbacks ------------------------------------------------------
    def _op_cb(self, msg):
        self.current_op = msg.data
        if msg.data == "pouring":
            self.saw_pour = True

    def _carried_obj_cb(self, msg):
        self.carried_obj = msg.data or ""

    def _lip_cb(self, msg):
        d = list(msg.data)
        self.carried_lip_xy = (d[0], d[1]) if len(d) == 2 else None

    def _tilt_cb(self, msg):
        t = float(msg.data)
        if t < 0:
            return  # nothing attached
        self.max_any_tilt = max(self.max_any_tilt, t)
        if self.current_op == "pouring" and t > self.pour_peak_tilt:
            self.pour_peak_tilt = t
            self.pour_peak_lip_xy = self.carried_lip_xy
        # attached (t >= 0) and not in the pour/idle phase => transport carry
        if self.current_op not in NON_TRANSPORT_OPS:
            obj = self.carried_obj
            if obj:
                self.max_tilt_by_obj[obj] = max(self.max_tilt_by_obj.get(obj, -1.0), t)
            # Only the task's target vessel counts toward the reported transport
            # tilt; before the sim has told us what is held, fall back to
            # accepting the sample so a single-object task is unaffected.
            if self.tilt_target is None or not obj or obj == self.tilt_target:
                self.max_transport_tilt = max(self.max_transport_tilt, t)

    # --- helpers --------------------------------------------------------------
    def _call(self, cli, req, timeout=120.0, wait=15.0):
        if not cli.wait_for_service(timeout_sec=wait):
            self.get_logger().error(f"service {cli.srv_name} unavailable")
            return None
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result()

    def _get_xy(self, name):
        pose = self._get_pose(name)
        return "" if pose is None else f"{pose[0]:.4f};{pose[1]:.4f}"

    def _get_pose(self, name):
        """(x, y, z, qw, qx, qy, qz) of an entity, or None."""
        req = GetEntityState.Request()
        req.entity = "/World/" + name
        res = self._call(self.ges_cli, req, timeout=10.0, wait=10.0)
        try:
            p = res.state.pose.position
            o = res.state.pose.orientation
            return (p.x, p.y, p.z, o.w, o.x, o.y, o.z)
        except Exception:
            return None

    @staticmethod
    def _tilt_from_upright_deg(qw, qx, qy, qz):
        """Angle between the body's +z axis and world +z, in degrees."""
        n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if n < 1e-9:
            return float("nan")
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
        # third column of the rotation matrix, z-component
        cos_tilt = max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))
        return math.degrees(math.acos(cos_tilt))

    def _magnet_in_vessel(self, vessel):
        """Is the stir bar inside the vessel? (Stir's terminal condition.)"""
        mag = self._get_pose("magnet")
        ves = self._get_pose(vessel)
        if mag is None or ves is None:
            return None, "pose_unavailable"
        dxy = math.hypot(mag[0] - ves[0], mag[1] - ves[1])
        # ENTITIES: flask dims [0.07, 0.07, 0.12] -> mouth is centre_z + 0.06.
        # A bar that really went in sits on the inner floor, well below this.
        mouth_z = ves[2] + 0.06
        inside = dxy <= MAGNET_IN_VESSEL_RADIUS_M and mag[2] <= mouth_z
        return inside, f"dxy={dxy * 1000:.0f}mm,z={mag[2]:.4f},mouth={mouth_z:.4f}"

    def _score_outcome(self, row, task="transfer"):
        """Measure what actually happened to the target object once it settled."""
        spec = TASK_OUTCOMES.get(task)
        if spec is None:
            row["failure_reason"] = row["failure_reason"] or f"no_outcome_spec:{task}"
            return
        target = spec["target"]
        row["target_obj"] = target

        time.sleep(SETTLE_AFTER_EXECUTE_S)
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.05)

        # The goal may ride on a randomized entity (Stir's stirrer), so read it
        # from the scene rather than assuming the nominal layout.
        goal_xy = spec.get("goal_xy")
        if spec.get("goal_entity") is not None:
            goal_pose = self._get_pose(spec["goal_entity"])
            if goal_pose is None:
                row["failure_reason"] = row["failure_reason"] or "goal_pose_unavailable"
                return
            goal_xy = (goal_pose[0], goal_pose[1])

        pose = self._get_pose(target)
        if pose is None:
            row["failure_reason"] = row["failure_reason"] or "final_pose_unavailable"
            return
        x, y, _z, qw, qx, qy, qz = pose
        tilt = self._tilt_from_upright_deg(qw, qx, qy, qz)
        row["final_tilt_deg"] = f"{tilt:.2f}"
        row["final_xy"] = f"{x:.4f};{y:.4f}"
        row["goal_err_mm"] = f"{math.hypot(x - goal_xy[0], y - goal_xy[1]) * 1000:.1f}"

        upright = tilt <= UPRIGHT_AFTER_RELEASE_DEG
        half_x, half_y = spec["half_xy"]
        in_goal = abs(x - goal_xy[0]) <= half_x and abs(y - goal_xy[1]) <= half_y
        row["placed_upright"] = upright
        row["placed_in_goal"] = in_goal

        # Where the stream would have landed. Measured at the peak-tilt instant of
        # the pour, against the target vessel's axis as it stands at scoring time
        # (the target vessel is never moved in this task).
        if self.pour_peak_tilt >= 0:
            row["pour_peak_tilt_deg"] = f"{self.pour_peak_tilt:.2f}"
        pour_target = spec.get("pour_target")
        if pour_target and self.pour_peak_lip_xy is not None:
            tgt = self._get_pose(pour_target)
            if tgt is not None:
                lx, ly = self.pour_peak_lip_xy
                row["pour_lip_err_mm"] = (
                    f"{math.hypot(lx - tgt[0], ly - tgt[1]) * 1000:.1f}"
                )

        poured_ok = self.saw_pour if spec["require_pour"] else True

        aux_ok = True
        if spec["aux"] == "magnet_in_vessel":
            aux_ok, detail = self._magnet_in_vessel(target)
            row["aux_check"] = f"magnet_in_vessel={aux_ok}({detail})"
            if aux_ok is None:
                aux_ok = False

        row["task_success"] = bool(
            row["plan_success"] and row["execute_success"]
            and poured_ok and upright and in_goal and aux_ok
        )
        if not row["task_success"] and not row["failure_reason"]:
            missing = []
            if spec["require_pour"] and not self.saw_pour:
                missing.append("no_pour_step")
            if not upright:
                missing.append(f"{target}_not_upright({tilt:.1f}deg)")
            if not in_goal:
                missing.append(f"{target}_outside_goal({x:.3f},{y:.3f})")
            if not aux_ok:
                missing.append("magnet_not_in_vessel")
            row["failure_reason"] = "task_incomplete:" + "+".join(missing)
        self.get_logger().info(
            f"outcome[{task}]: task_success={row['task_success']} target={target} "
            f"final_tilt={tilt:.2f}deg xy=({x:.4f},{y:.4f}) "
            f"goal_err={row['goal_err_mm']}mm upright={upright} in_goal={in_goal} "
            f"poured={row['poured']} aux={row['aux_check'] or 'n/a'} "
            f"pour_peak_tilt={row['pour_peak_tilt_deg'] or 'n/a'}deg "
            f"pour_lip_err={row['pour_lip_err_mm'] or 'n/a'}mm"
        )

    def run_trial(self, seed, task, robot_cfg, plan_timeout, exec_timeout):
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "seed": seed, "task": task, "robot_cfg": robot_cfg,
            "plan_success": False, "planning_time_s": "", "num_satisfying": "",
            "execute_success": False, "max_transport_tilt_deg": "",
            "failure_reason": "", "beaker_xy": "", "flask_xy": "",
            "poured": False, "final_tilt_deg": "", "final_xy": "",
            "placed_upright": False, "placed_in_goal": False, "task_success": False,
            "target_obj": "", "goal_err_mm": "", "aux_check": "",
            "pour_peak_tilt_deg": "", "pour_lip_err_mm": "",
        }

        # Canonical sequence: (tool -> cfg) -> env -> plan -> execute.
        planner = get_planner_spec(robot_cfg)
        tc = ToolChange.Request(); tc.desired_tool = planner.tool
        res = self._call(self.tool_change_cli, tc, timeout=120.0)
        if res is None or not getattr(res, "success", False):
            row["failure_reason"] = f"tool_change_{planner.tool}_failed"
            return row
        self.get_logger().info(f"tool_change -> {planner.tool} OK")
        time.sleep(1.0)

        # 2. configure the planner from the canonical registry. The server
        # rejects robot-config switches after its first planning call.
        cfg = SetTampCfg.Request()
        cfg.robot = planner.robot
        cfg.grasp_dof = planner.grasp_dof
        cfg.num_particles = planner.num_particles
        cfg.num_resampling_attempts = planner.num_resampling_attempts
        cfg.num_opt_steps = planner.num_opt_steps
        cfg.num_initial_plans = planner.num_initial_plans
        cfg.approach = planner.approach
        cfg.curobo_plan = True
        cfg.enable_visualizer = False
        cfg.viz_robot_mesh = False
        cfg.enable_experiment_logging = False
        cfg.time_dilation_factor = planner.time_dilation_factor
        res = self._call(self.cfg_cli, cfg, timeout=60.0)
        if res is None or not getattr(res, "success", False):
            row["failure_reason"] = "set_tamp_cfg_failed"
            return row
        self.get_logger().info(f"set_tamp_cfg {robot_cfg} OK")

        outcome_spec = TASK_OUTCOMES.get(task)
        self.tilt_target = outcome_spec["target"] if outcome_spec else None

        # record the ACTUAL (randomized, settled) beaker/flask poses used
        row["beaker_xy"] = self._get_xy("beaker")
        row["flask_xy"] = self._get_xy("flask")

        # 3. set_tamp_env transfer
        spec = get_environment_spec(task)
        env = SetTampEnv.Request()
        env.env_name = spec.name
        env.entities = list(spec.entities)
        env.movables = list(spec.movables)
        env.statics = list(spec.statics)
        env.ex_collision = list(spec.ex_collision)
        env.rearrange_grid = spec.rearrange_grid
        res = self._call(self.env_cli, env, timeout=60.0)
        if res is None or not getattr(res, "success", False):
            row["failure_reason"] = "set_tamp_env_failed"
            return row
        self.get_logger().info(f"set_tamp_env {spec.name} OK")

        # 4. plan
        preq = Plan.Request(); preq.env_name = task
        t0 = time.time()
        res = self._call(self.plan_cli, preq, timeout=plan_timeout)
        dt = time.time() - t0
        row["planning_time_s"] = f"{dt:.2f}"
        if res is None:
            row["failure_reason"] = "plan_no_response"
            return row
        row["plan_success"] = bool(res.plan_success)
        row["num_satisfying"] = int(getattr(res, "total_num_satisfying", 0) or 0)
        self.get_logger().info(
            f"plan: success={row['plan_success']} "
            f"#satisfying={row['num_satisfying']} time={dt:.2f}s"
        )
        if not row["plan_success"]:
            row["failure_reason"] = "plan_no_satisfying"
            return row

        # 5. execute (spin so /carried_tilt_deg + /tamp_current_op keep updating)
        if not self.exec_cli.wait_for_service(timeout_sec=15.0):
            row["failure_reason"] = "execute_service_unavailable"
            return row
        ereq = Execute.Request()
        t0 = time.time()
        fut = self.exec_cli.call_async(ereq)
        last_log = 0.0
        while rclpy.ok() and not fut.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            now = time.time()
            if now - last_log > 5.0:
                self.get_logger().info(
                    f"executing... t={now - t0:4.0f}s op={self.current_op} "
                    f"max_transport_tilt={self.max_transport_tilt:.2f} "
                    f"max_any_tilt={self.max_any_tilt:.2f}"
                )
                last_log = now
            if now - t0 > exec_timeout:
                row["failure_reason"] = "execute_timeout"
                break
        res = fut.result() if fut.done() else None
        if res is not None:
            row["execute_success"] = bool(getattr(res, "execute_success", False))
            if not row["execute_success"] and not row["failure_reason"]:
                row["failure_reason"] = "execute_returned_false"
        row["max_transport_tilt_deg"] = (
            f"{self.max_transport_tilt:.2f}" if self.max_transport_tilt >= 0 else "nan"
        )
        row["poured"] = self.saw_pour
        self.get_logger().info(
            f"execute: success={row['execute_success']} "
            f"max_transport_tilt[{self.tilt_target}]={row['max_transport_tilt_deg']} "
            f"(max_any_tilt={self.max_any_tilt:.2f}, per-object "
            f"{ {k: round(v, 2) for k, v in self.max_tilt_by_obj.items()} }) "
            f"elapsed={time.time() - t0:.1f}s"
        )
        # What actually happened to the target object, from the settled scene.
        self._score_outcome(row, task=task)
        return row


def append_csv(path, row):
    exists = os.path.isfile(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--task", default="transfer")
    ap.add_argument("--robot", default="fr5_ag95")
    ap.add_argument("--plan-timeout", type=float, default=240.0)
    ap.add_argument("--exec-timeout", type=float, default=300.0)
    args = ap.parse_args()

    rclpy.init()
    node = TaskOrchestrator()
    try:
        row = node.run_trial(args.seed, args.task, args.robot,
                             args.plan_timeout, args.exec_timeout)
    except Exception as e:  # never lose a trial: log the failure as a row
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed, "task": args.task, "robot_cfg": args.robot,
            "plan_success": False, "planning_time_s": "", "num_satisfying": "",
            "execute_success": False, "max_transport_tilt_deg": "",
            "failure_reason": f"driver_exception:{type(e).__name__}:{e}",
            "beaker_xy": "", "flask_xy": "",
            "target_obj": "", "goal_err_mm": "", "aux_check": "",
            "pour_peak_tilt_deg": "", "pour_lip_err_mm": "",
        }
        node.get_logger().error(f"trial exception: {e}")
    append_csv(args.csv, row)
    node.get_logger().info(f"CSV row appended -> {args.csv}: {row}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
