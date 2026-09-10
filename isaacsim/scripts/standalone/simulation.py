import rclpy
from rclpy.node import Node
import numpy as np
from isaacsim import SimulationApp

from std_srvs.srv import SetBool
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Wrench
from tamp_interfaces.srv import ToolChange, GetRobotInfo, GetToolInfo
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
import sys, os

ROBOT_STAGE_PATH = "/World/Robot"
ROOT_JOINT_PATH = ROBOT_STAGE_PATH + "/root_joint"
BACKGROUND_STAGE_PATH = "/background"
BACKGROUND_USD_PATH = "/Isaac/Environments/Grid/default_environment.usd"

# Headless is opt-in via the SDL_HEADLESS env var (1/true/yes/on) so batch/CI
# runs need no GUI; the default stays headless=False (GUI) for the normal
# interactive workflow. See scripts/run_isaacsim.sh --headless.
_SDL_HEADLESS = os.environ.get("SDL_HEADLESS", "").strip().lower() in ("1", "true", "yes", "on")
CONFIG = {"renderer": "RaytracedLighting", "headless": _SDL_HEADLESS}


class Simulation(Node):
    
    def __init__(self):
        
        super().__init__("sdl_isaacsim")

        self.simulation_app = SimulationApp(CONFIG)

        from isaacsim.core.api import World
        from isaacsim.core.utils import extensions, prims, viewports
        from isaacsim.core.utils.types import ArticulationAction
        import omni.graph.core as og

        extensions.enable_extension("isaacsim.ros2.sim_control")
        extensions.enable_extension("isaacsim.ros2.bridge")

        # Save Imports
        self.prims = prims
        self.ArticulationAction = ArticulationAction
        self.World = World
        self.og = og

        self._saved_robot_joint_positions = None
        
        self.simulation_app.update()

        self.world = self.World(stage_units_in_meters=1.0)

        # Preparing stage
        viewports.set_camera_view(eye=np.array([1.6, 1.6, 1.2]), target=np.array([0, 0, 0.1]))

        from task import Task

        sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
        from camera import initialize_camera

        sys.path.append(os.path.join(os.path.dirname(__file__), "action_graph"))
        from camera_graph import create_ros_camera_graph
        from robot_control_graph import create_robot_control_graph
        from tf_graph import create_tf_graph
        
        self.create_ros_camera_graph = create_ros_camera_graph
        self.create_robot_control_graph = create_robot_control_graph
        self.create_tf_graph = create_tf_graph
        self.initialize_camera = initialize_camera

        self.task = Task(name="task", robot_prim_path=ROBOT_STAGE_PATH, robot_name="fr5")
        self.world.add_task(self.task)
        self.world.reset()

        self.configure_physics()

        self.simulation_app.update()

        # initialize camera
        for i in range(2):
            self.initialize_camera(self.task.cameras[i])
        
        self.robot = self.world.scene.get_object("fr5")
        if self.robot is None:
            self.get_logger().error("Failed to get robot 'fr5' from scene after initial reset.")
            self.get_logger().error("This is likely due to the USD asset issue. Check startup warnings.")
            self.simulation_app.close()
            return
        
        self.robot.post_reset()

        self.simulation_app.update()

        # need to initialize physics getting any articulation etc.
        self.world.initialize_physics()

        # action graphs
        camera_paths = ["/World/camera_1", "/World/camera_2"]
        camera_names = ["camera_1", "camera_2"]
        self.camera_data_graph = self.create_ros_camera_graph(camera_paths=camera_paths, camera_names=camera_names)
        self.og.Controller.evaluate_sync(self.camera_data_graph)
        self.robot_control_graph = self.create_robot_control_graph(articulation_root_path=ROOT_JOINT_PATH)
        target_prim_paths = [f"/World/camera_{i}" for i in range(1, 3)]
        self.tf_graph = self.create_tf_graph(
            target_prim_paths=target_prim_paths,
            parent_prim_path=ROBOT_STAGE_PATH + "/base_link",
        )

        self.arm_joint_names = ["j1", "j2", "j3", "j4", "j5", "j6"]
        self.arm_joint_ids = []
        self.current_tool = "empty"

        # Fixed-joint grasp state (REFACTOR.md III-1). When a parallel gripper
        # (ag95/dh3) closes on a movable object we weld it to the ee link with a
        # temporary UsdPhysics.FixedJoint and track it here; open removes it.
        self._attached_object_name = None
        self._attached_joint_path = None

        # Difference between the outgoing and incoming tools' home corrections,
        # computed at each tool change and applied when the arm is restored.
        self._home_offset_delta = None

        # MAJOR-1 (REFACTOR.md III): the planner's intended grasp object, pushed
        # by tamp_server on /set_grasp_target just before it calls the gripper
        # CLOSE service. When set (and a valid movable), the close welds THAT
        # object instead of the geometric-nearest one -- required once object
        # positions are randomized (+/-5 cm) and neighbours can be ~0.057 m apart.
        # None => fall back to nearest-object (preserves the old behaviour).
        self._grasp_target_name = None

        # Object a SUCTION tool (vgc10) is carrying. Suction engages through
        # SurfaceGripper rather than a fixed joint, so there is no
        # `_attached_object_name` to monitor and the carried-tilt topic reported
        # -1 for the whole Move task. Tracked separately, and only reported while
        # the object is actually at the end-effector, so a failed suction pick
        # cannot masquerade as a perfectly upright carry.
        self._suction_carry_name = None

        # Post-release diagnostic trace (release/placement failure analysis).
        # Armed by the gripper-open handler; step_cb logs the released object's
        # pose + velocity for SDL_RELEASE_TRACE_S seconds so we can see whether
        # the vessel is dropped, launched by residual finger contact, or settles.
        self._release_trace = None

        if not self.update_joint_ids():
            self.get_logger().error("Failed to initialize joint IDs. Shutting down.")
            self.simulation_app.close()
            return

        self.world.play()

        self.timer_period = 1/60
        self.timer = self.create_timer(self.timer_period, self.step_cb)

        self.gripper_commands_srv = self.create_service(SetBool, "isaac_gripper_commands", self.gripper_commands_cb)
        
        self.get_robot_info_srv = self.create_service(GetRobotInfo, "get_robot_info", self.get_robot_info_cb)
        self.get_tool_info_srv = self.create_service(GetToolInfo, "get_tool_info", self.get_tool_info_cb)
        self.tool_change_srv = self.create_service(ToolChange, "tool_change", self.tool_change_cb)

        self.ft_pub = self.create_publisher(Wrench, "raw_ft_data", 10)
        self.scale_pub = self.create_publisher(Float32, "raw_scale_data", 10)
        self.scale_start_angle = None

        # MAJOR-1 grasp-target intake (latched RELIABLE, matches tamp_server).
        _latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.grasp_target_sub = self.create_subscription(
            String, "set_grasp_target", self.grasp_target_cb, _latched_qos
        )
        # Carried-vessel tilt from upright [deg] of the currently-attached object,
        # published every physics step (-1.0 when nothing is attached). Feeds the
        # trials harness / R3#2 max-transport-tilt measurement.
        self.carried_tilt_pub = self.create_publisher(Float32, "carried_tilt_deg", 10)
        # Name of the object those tilt samples belong to ("" when nothing is
        # held). Stir carries the vessel and then the stir bar, and a stir bar's
        # orientation is not a spill constraint, so the two must not be pooled.
        self.carried_obj_pub = self.create_publisher(String, "carried_obj", 10)

        self.step = 0

        self.get_logger().info("Simulation Start")

    def grasp_target_cb(self, msg):
        """Store the planner's intended grasp-target object name (MAJOR-1)."""
        name = (msg.data or "").strip()
        self._grasp_target_name = name if name else None
        self.get_logger().info(f"grasp target set -> {self._grasp_target_name!r}")

    def _open_gripper(self, max_steps=40, tol=0.02):
        """Open the gripper until it actually REACHES its opened position.

        A parallel gripper here is driven by per-call position deltas, so a
        hardcoded repeat count only works if it exceeds
        (closed - opened) / delta. Measured on Stir: the dh3 closes to 1.16 rad
        in steps of 1.16/5 = 0.232 rad, but the open branch ran 3 steps, which
        leaves the fingers at ~0.46 rad -- still gripping. The "released" vessel
        therefore stayed in the hand and the retracting arm carried it upwards
        (z 0.158 -> 0.207 m at a steady ~25 mm/s) before it fell over 183 mm from
        the stirrer. Driving to the target instead of counting steps removes that
        whole class of bug for every parallel gripper, and a gripper that cannot
        reach its opened position now says so rather than silently half-closing.

        The suction gripper has no joints to converge on, so it is opened once.
        """
        grip = self.robot.gripper
        target = getattr(grip, "joint_opened_positions", None)
        if target is None:                      # SurfaceGripper (vgc10)
            grip.open()
            self.world.step(render=True)
            return True
        target = np.asarray(target, dtype=float)

        def _pos():
            return np.asarray(grip.get_joint_positions(), dtype=float)

        steps = 0
        cur = _pos()
        while steps < max_steps and np.max(np.abs(cur - target)) > tol:
            grip.open()
            self.world.step(render=True)
            cur = _pos()
            steps += 1
        reached = bool(np.max(np.abs(cur - target)) <= tol)
        if reached:
            self.get_logger().info(
                "gripper opened in %d step(s) -> %s" % (steps, np.round(cur, 4).tolist())
            )
        else:
            self.get_logger().warning(
                "gripper did NOT reach its opened position after %d steps: at %s, "
                "target %s -- it is still holding the object"
                % (steps, np.round(cur, 4).tolist(), np.round(target, 4).tolist())
            )
        return reached

    def _obj_state(self, name):
        """(x, y, z, tilt_deg, |v|, |w|) of a movable, or None."""
        cand = self._movable_candidates().get(name)
        if cand is None:
            return None
        try:
            pos, quat = cand.get_world_pose()
            tilt = self._tilt_from_upright_deg(quat)
        except Exception:
            return None
        try:
            v = float(np.linalg.norm(cand.get_linear_velocity()))
            w = float(np.linalg.norm(cand.get_angular_velocity()))
        except Exception:
            v = w = float("nan")
        return (float(pos[0]), float(pos[1]), float(pos[2]), tilt, v, w)

    def _log_obj_state(self, tag, name):
        st = self._obj_state(name)
        if st is None:
            return
        self.get_logger().info(
            "RELEASE_TRACE %s '%s' pos=(%.4f,%.4f,%.4f) tilt=%.2fdeg |v|=%.4f |w|=%.4f"
            % (tag, name, st[0], st[1], st[2], st[3], st[4], st[5])
        )

    def _movable_candidates(self):
        """Name -> rigid-prim handle for the movable grasp candidates."""
        return {
            "beaker": self.task.beaker,
            "flask": self.task.flask,
            "box": self.task.box,
            "magnet": self.task.magnet,
        }

    @staticmethod
    def _tilt_from_upright_deg(quat_wxyz):
        """Angle [deg] between the body's local +Z (mapped to world) and world +Z."""
        w, x, y, z = [float(v) for v in quat_wxyz]
        # world-frame z-component of the body's local +Z axis = R[2,2]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        up_z = max(-1.0, min(1.0, up_z))
        return float(np.degrees(np.arccos(up_z)))

    def configure_physics(self):
        """Explicit PhysX scene configuration (REFACTOR.md II-1).

        Recorded for PLAN.md 3-C / paper Appendix A. Uses the Isaac Sim 6.x
        PhysicsContext API (world.get_physics_context()); these calls are
        verified present in isaacsim.core.api.physics_context.PhysicsContext.
        Values are logged (real, not asserted) so the run is self-documenting.

        Chosen values:
          physics_dt = 1/60 s (60 Hz, matches the step_cb timer period)
          substeps   = 1
          solver     = TGS (temporal Gauss-Seidel)
          gravity    = -9.81 m/s^2 along -Z
          gpu_dynamics = True (RTX 5070 GPU pipeline)
        Per-body solver iteration counts (position=8, velocity=1) are set on the
        movable objects in utils/object.py (PhysxRigidBodyAPI), the correct
        PhysX scope for those.
        """
        ctx = self.world.get_physics_context()
        ctx.set_physics_dt(dt=1.0 / 60.0, substeps=1)
        ctx.set_solver_type("TGS")
        ctx.set_gravity(-9.81)
        ctx.enable_gpu_dynamics(True)
        self.get_logger().info(
            "PhysX config: dt=%.6fs solver=%s gravity=%s gpu_dynamics=on"
            % (ctx.get_physics_dt(), ctx.get_solver_type(), ctx.get_gravity())
        )

    def update_joint_ids(self) -> bool:

        self.get_logger().info("Updating joint IDs...")
        self.arm_joint_ids = []
        if self.robot is None:
            self.get_logger().error("Cannot update joint IDs, self.robot is None.")
            return False
        
        if not self.robot.is_valid():
             self.get_logger().error("Cannot update joint IDs, self.robot is not valid.")
             return False
            
        for joint_name in self.arm_joint_names:
            idx = self.robot.get_dof_index(joint_name)
            if idx == -1:
                self.get_logger().error(f"Failed to find joint '{joint_name}' on robot.")
                return False
            self.arm_joint_ids.append(idx)
        
        self.get_logger().info(f"Updated arm joint IDs: {self.arm_joint_ids}")
        return True

    def step_cb(self):

        if self.simulation_app.is_running():

            # step simulation
            self.world.step(render=True)
            
            if self.robot and self.robot.is_valid():

                # robot control command
                self.og.Controller.set(self.og.Controller.attribute("/ActionGraph/RobotControl/OnImpulseEvent.state:enableImpulse"), True)
                # compute observations
                observations = self.world.get_observations()

                ft_data = observations["ft_data"]
                scale_data = observations["scale_data"]

                # publish
                if ft_data is not None:
                    self.publish_ft(ft_data)
                if scale_data is not None:
                    self.publish_scale(scale_data)

                # carried-vessel tilt (deg from upright) of the attached object,
                # or -1.0 when nothing is welded to the gripper.
                self.publish_carried_tilt()

            self._tick_release_trace()

            self.step += 1

        else:
            self.get_logger().info("Quit ROS2 Node")
            rclpy.try_shutdown()

    def publish_ft(self, ft_data: np.ndarray):
        
        msg = Wrench()
        msg.force.x = float(ft_data[0])
        msg.force.y = float(ft_data[1])
        msg.force.z = float(ft_data[2])
        msg.torque.x = float(ft_data[3])
        msg.torque.y = float(ft_data[4])
        msg.torque.z = float(ft_data[5])
        
        self.ft_pub.publish(msg)

    def publish_scale(self, scale_data: float):

        msg = Float32()
        msg.data = float(scale_data)
        self.scale_pub.publish(msg)

    def publish_carried_tilt(self):
        tilt = -1.0
        carried = self._attached_object_name
        require_gripping = False
        if carried is None and self._suction_carry_name is not None:
            carried = self._suction_carry_name
            require_gripping = True   # suction: only while the cup is engaged
        if carried is not None:
            cand = self._movable_candidates().get(carried)
            if cand is not None:
                try:
                    _, quat = cand.get_world_pose()
                    held = True
                    if require_gripping:
                        # The suction gripper's own status, straight from the
                        # PhysX surface-gripper interface: Closed means it has
                        # actually latched onto something, so a suction pick that
                        # failed cannot report a perfectly upright "carry".
                        held = bool(self.robot.gripper.is_closed())
                    if held:
                        tilt = self._tilt_from_upright_deg(quat)
                except Exception as exc:
                    tilt = -1.0
                    if self.step % 300 == 0:
                        self.get_logger().warning(
                            "carried-tilt for '%s' unavailable: %r" % (carried, exc)
                        )
        msg = Float32()
        msg.data = float(tilt)
        self.carried_tilt_pub.publish(msg)
        name_msg = String()
        name_msg.data = "" if tilt < 0 or carried is None else str(carried)
        self.carried_obj_pub.publish(name_msg)

    def _tick_release_trace(self):
        """Log the released object's state for a short window after release."""
        tr = self._release_trace
        if tr is None:
            return
        if tr["left"] <= 0:
            self._release_trace = None
            return
        tr["left"] -= 1
        if tr["left"] % tr["every"] == 0:
            self._log_obj_state("t+%03d" % (tr["total"] - tr["left"]), tr["name"])

    def arm_commands_cb(self, msg):
        
        if self.robot is None or not self.robot.is_valid():
            self.get_logger().warning("arm_commands_cb: Robot is not valid. Skipping command.")
            return

        action = self.ArticulationAction(joint_positions=msg.position, joint_indices=self.arm_joint_ids)
        self.robot.apply_action(action)


    def gripper_commands_cb(self, request, response):

        if self.robot is None or not self.robot.is_valid() or self.robot.gripper is None:
            self.get_logger().warning("gripper_commands_cb: Robot is not valid. Skipping command.")
            response.success = False
            response.message = "Robot is not valid (currently swapping?)"
            return response

        is_close = request.data
        gripper = self.current_tool
        import grasp as grasp_mod

        if is_close:
            if gripper == "dh3":
                num_repeat = 16
            elif gripper == "ag95":
                num_repeat = 10
            else:
                num_repeat = 1

            for _ in range(num_repeat):
                self.robot.gripper.close()
                self.world.step(render=True)

            # Fixed-joint grasp attach (REFACTOR.md III-1). Only parallel
            # grippers weld the object; vgc10 is a suction tool whose own
            # SurfaceGripper.close() already engaged, so it must NOT get a
            # fixed joint. Guarded by current_tool.
            if gripper in ("ag95", "dh3"):
                attached = self._attach_nearest_object()
                if attached is None:
                    # No graspable object at the ee -> grasp FAIL (fail-closed).
                    response.success = False
                    response.message = (
                        "close gripper: grasp FAIL (no graspable object within "
                        "%.3f m of end-effector; attached nothing)" % grasp_mod.GRASP_THRESHOLD_M
                    )
                    return response
                response.message = "close gripper + grasp attach '%s'" % attached
            else:
                # Suction: nothing is welded, but remember what the planner meant
                # to pick so the carried-tilt monitor has an object to follow.
                # Consume the hint here too, so it cannot leak into a later grasp.
                target = self._grasp_target_name
                self._grasp_target_name = None
                if target is not None and target in self._movable_candidates():
                    self._suction_carry_name = target
                    self.get_logger().info(
                        "suction carry target -> '%s' (no fixed joint; tilt monitored "
                        "while it stays within the grasp threshold of the ee)" % target
                    )
                response.message = "close gripper"

        else:
            held = self._attached_object_name or self._suction_carry_name
            if held is not None:
                self._log_obj_state("pre_release", held)

            # The trajectory replay ends the moment the last waypoint is COMMANDED,
            # so the arm can still be moving when the gripper is told to let go:
            # measured on Stir seed 0, the vessel was released at |v| = 0.41 m/s and
            # |w| = 6.8 rad/s and tumbled to 38 deg. Hold the last commanded pose for
            # a few physics steps first. No new arm command is issued -- this is
            # "let the controller settle", not an extra motion.
            settle = int(os.environ.get("SDL_PLACE_SETTLE_STEPS", "15"))
            for _ in range(settle):
                self.world.step(render=True)
            if settle and held is not None:
                self._log_obj_state("after_settle", held)

            # Release order. The weld stands in for "the fingers are holding the
            # object", so it must be removed only once the fingers have let go.
            # "detach_first" (the original order) removes it while the fingers are
            # still closed, which leaves the vessel a free body in contact with two
            # finger colliders that then sweep outwards through it -- measured on
            # seed 5: the vessel went from 0.30 deg / 0.03 rad/s before the open to
            # 28.85 deg / 1.98 rad/s after it, and toppled (90 deg, 7.5 cm away).
            # "open_first" opens the fingers while the weld still holds the vessel
            # rigidly, so those contacts are gone before it becomes free: the same
            # seed then released at 0.06 deg / 0.015 rad/s and settled upright.
            order = os.environ.get("SDL_RELEASE_ORDER", "open_first")
            released = None

            if order == "open_first":
                self._open_gripper()
                if held is not None:
                    self._log_obj_state("after_open", held)
                if gripper in ("ag95", "dh3"):
                    released = self._detach_object()
            else:
                if gripper in ("ag95", "dh3"):
                    released = self._detach_object()
                if held is not None:
                    self._log_obj_state("after_detach", held)
                self._open_gripper()
                if held is not None:
                    self._log_obj_state("after_open", held)

            if self._suction_carry_name is not None:
                released = released or self._suction_carry_name
                self._suction_carry_name = None

            if released is not None:
                trace_s = float(os.environ.get("SDL_RELEASE_TRACE_S", "3.0"))
                total = max(0, int(trace_s * 60))
                if total:
                    self._release_trace = {
                        "name": released, "left": total, "total": total, "every": 6,
                    }

            if released is not None:
                response.message = "open gripper + grasp release '%s'" % released
            else:
                response.message = "open gripper"

        response.success = True

        return response

    def _weld_parent(self):
        """(prim_path, position, quaternion) of the link a grasped object is welded to.

        Prefer a NON-ACTUATED link (the wrist flange) over the finger-tip link. A
        SIDE grasp offsets the vessel's centre of mass laterally from the grasp
        point, so gravity applies a steady moment about the finger joints and they
        give way: measured on seed 4 as the welded beaker sagging to 28.6 deg while
        the planned carry was 0.39 deg, after which it toppled at release. A top
        grasp does not show this because the vessel hangs directly below the grasp
        point. Welding proximal to the finger joints removes that compliance while
        still moving rigidly with the tool. Falls back to the end-effector link.
        """
        from isaacsim.core.utils.stage import get_current_stage
        ee = self.robot.end_effector
        flange = ee.prim_path.rsplit("/", 1)[0] + "/wrist3_link"
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(flange)
        if not prim or not prim.IsValid():
            pos, quat = ee.get_world_pose()
            return ee.prim_path, pos, quat
        from pxr import Gf, Usd, UsdGeom
        mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = mat.ExtractTranslation()
        q = mat.ExtractRotationQuat().GetNormalized()
        im = q.GetImaginary()
        return (
            flange,
            np.array([t[0], t[1], t[2]], dtype=float),
            np.array([q.GetReal(), im[0], im[1], im[2]], dtype=float),
        )

    def _attach_nearest_object(self):
        """Weld the movable object nearest the end-effector to the gripper link
        with a temporary UsdPhysics.FixedJoint locking the current relative pose
        (REFACTOR.md III-1). Returns the attached object's name, or None on a
        grasp FAIL (nothing within the threshold)."""
        from isaacsim.core.utils.stage import get_current_stage
        import grasp as grasp_mod

        # Clear any stale attachment first (defensive; open normally clears it).
        self._detach_object()

        ee = self.robot.end_effector
        ee_pos, ee_quat = ee.get_world_pose()

        # Movable grasp candidates (task holds these SingleRigidPrim handles).
        candidates = self._movable_candidates()

        # MAJOR-1: if the planner named the target object, weld THAT object rather
        # than the geometric-nearest one. Consume the hint (so it never leaks into
        # the next grasp). The joint locks the CURRENT relative pose, so nothing
        # teleports; we log the ee-to-object distance and warn if it is unexpectedly
        # large (execution drift), but still honour the planner's intent. Fall back
        # to nearest-object only when no target was set.
        target = self._grasp_target_name
        self._grasp_target_name = None
        if target is not None and target in candidates:
            obj_pos, obj_quat = candidates[target].get_world_pose()
            dist = float(np.linalg.norm(np.asarray(obj_pos, dtype=float) - np.asarray(ee_pos, dtype=float)))
            name = target
            if dist > grasp_mod.GRASP_THRESHOLD_M:
                self.get_logger().warning(
                    "grasp target '%s' is %.4f m from the ee (> %.3f m); attaching "
                    "it anyway (planner intent), welding current relative pose"
                    % (name, dist, grasp_mod.GRASP_THRESHOLD_M)
                )
            else:
                self.get_logger().info(
                    "grasp target '%s' selected (dist=%.4f m)" % (name, dist)
                )
        else:
            if target is not None:
                self.get_logger().warning(
                    "grasp target '%s' is not a known movable; falling back to "
                    "nearest-object selection" % target
                )
            poses = {n: prim.get_world_pose() for n, prim in candidates.items()}
            best = grasp_mod.nearest_object(ee_pos, poses)
            if best is None:
                return None
            name, obj_pos, obj_quat, dist = best

            if dist > grasp_mod.GRASP_THRESHOLD_M:
                self.get_logger().warning(
                    "grasp FAIL: nearest movable object '%s' is %.4f m from the ee "
                    "(> %.3f m threshold); attaching nothing"
                    % (name, dist, grasp_mod.GRASP_THRESHOLD_M)
                )
                return None

        # Flush pending transform writes + settle contacts for ONE step BEFORE
        # authoring the joint, then RE-READ the ee + object physics poses. The
        # joint's local frames must be computed from the poses PhysX will cook
        # the joint against. Authoring them from a pose that has not been flushed
        # to PhysX yet (e.g. right after a set_world_pose with no intervening
        # step, or while the object is still settling into the grasp) leaves the
        # two body frames "disjointed" at cook time, so PhysX snaps the bodies
        # together to reconcile them -- the REFACTOR.md III MAJOR-2 symptom
        # ("CreateJoint - found a joint with disjointed body transforms ..."
        # warning + ~cm attach snap). With a flushed, consistent snapshot the
        # authored frames coincide at cook time: no warning, and the object
        # stays at its current world pose (the joint locks the ACTUAL current
        # relative transform). The finger-tip link's USD pose == its PhysX
        # rigid-body pose (verified), so no per-link frame correction is needed;
        # the fix is purely reading a consistent snapshot.
        self.world.step(render=True)
        parent_path, parent_pos, parent_quat = self._weld_parent()
        obj_pos, obj_quat = candidates[name].get_world_pose()

        stage = get_current_stage()
        grasp_mod.create_grasp_fixed_joint(
            stage, grasp_mod.GRASP_JOINT_PATH,
            parent_path, parent_pos, parent_quat,
            candidates[name].prim_path, obj_pos, obj_quat,
        )
        # Step so PhysX parses the newly-authored joint into the running scene.
        for _ in range(2):
            self.world.step(render=True)

        self._attached_object_name = name
        self._attached_joint_path = grasp_mod.GRASP_JOINT_PATH
        self.get_logger().info(
            "grasp ATTACH: welded '%s' to link '%s' (ee dist=%.4f m) via %s"
            % (name, parent_path, dist, grasp_mod.GRASP_JOINT_PATH)
        )
        return name

    def _detach_object(self):
        """Remove the temporary grasp joint (if any) and clear tracking. Returns
        the released object's name, or None if nothing was attached."""
        if self._attached_joint_path is None:
            return None
        from isaacsim.core.utils.stage import get_current_stage
        import grasp as grasp_mod

        name = self._attached_object_name
        grasp_mod.remove_grasp_fixed_joint(get_current_stage(), self._attached_joint_path)
        self._attached_joint_path = None
        self._attached_object_name = None
        self.get_logger().info(
            "grasp RELEASE: removed fixed joint (was holding '%s')" % name
        )
        return name
    
    def tool_change_cb(self, request, response):

        desired_tool = request.desired_tool

        try:
            # Detach any grasped object BEFORE the world rebuild so the grasp
            # joint prim (and its body targets) are gone before world.clear()
            # tears down the stage -- otherwise a stale joint could reference
            # cleared prims and the tracking state would be left dangling
            # (REFACTOR.md III task 5).
            self._detach_object()
            self._suction_carry_name = None

            observations = self.world.get_observations()
            self.task.current_positions = observations["current_positions"]
            self.task.current_orientations = observations["current_orientations"]

            # A tool change restores the arm to where it was (below), which would
            # discard the new tool's home correction. Carry the DIFFERENCE between
            # the two tools' corrections instead, so repeated tool changes neither
            # lose it nor accumulate it.
            self._home_offset_delta = (
                self.task.tool_home_offset(desired_tool)
                - self.task.tool_home_offset(self.current_tool)
            )

            self.task.desired_tool = desired_tool
            self.current_tool = desired_tool

            self._saved_robot_joint_positions = None
            if self.robot and self.robot.is_valid():
                self._saved_robot_joint_positions = self.robot.get_joint_positions(joint_indices=self.arm_joint_ids)
            
            self.timer.cancel()
            self.world.stop()

            self.world.clear()

            self.simulation_app.update()
            
            self.world = self.World(stage_units_in_meters=1.0)
            self.world.add_task(self.task)
            self.world.reset()

            self.configure_physics()

            self.simulation_app.update()

            # initialize camera
            for i in range(2):
                self.initialize_camera(self.task.cameras[i])

            self.robot = self.world.scene.get_object("fr5")
            
            self.robot.post_reset()

            self.simulation_app.update()

            self.world.initialize_physics()

            # action graphs
            camera_paths = ["/World/camera_1", "/World/camera_2"]
            camera_names = ["camera_1", "camera_2"]
            self.camera_data_graph = self.create_ros_camera_graph(camera_paths=camera_paths, camera_names=camera_names)
            self.og.Controller.evaluate_sync(self.camera_data_graph)
            self.robot_control_graph = self.create_robot_control_graph(articulation_root_path=ROOT_JOINT_PATH)
            target_prim_paths = [f"/World/camera_{i}" for i in range(1, 3)]
            self.tf_graph = self.create_tf_graph(
                target_prim_paths=target_prim_paths,
                parent_prim_path=ROBOT_STAGE_PATH + "/base_link",
            )

            restored = np.asarray(self._saved_robot_joint_positions, dtype=float)
            delta = getattr(self, "_home_offset_delta", None)
            if delta is not None and restored.shape == delta.shape and np.any(delta):
                restored = restored + delta
                self.get_logger().info(
                    "tool home correction applied for '%s': arm -> %s"
                    % (self.current_tool, np.round(restored, 4).tolist())
                )
            self.robot.set_joint_positions(
                positions=restored,
                joint_indices=self.arm_joint_ids,
            )
            zero_vels = np.zeros(self.robot.num_dof, dtype="float32")
            self.robot.set_joint_velocities(zero_vels)

            self.update_joint_ids()
            
            self.world.play()
            self.timer = self.create_timer(self.timer_period, self.step_cb)
            self.get_logger().info("Physics simulation resumed. Tool change complete.")

            response.success = True
            response.message = "Robot/Tool change complete and simulation resumed."

        except ValueError as e:

            self.get_logger().info(f"tool_change Service Error: {e}")
            response.success = False
            response.message = "Robot/Tool change Failed"

        return response
    
    def get_robot_info_cb(self, request, response):
        
        response.q_init = self.robot.get_joint_positions()[:6].tolist() # dof = 6
        response.joint_names = self.arm_joint_names
        response.current_tool = self.current_tool

        return response
    
    def get_tool_info_cb(self, request, response):
        
        current_tool = request.current_tool
        desired_tool = request.desired_tool
        assert current_tool in {'empty', 'ag95', 'vgc10', 'dh3'}
        assert desired_tool in {'ag95', 'vgc10', 'dh3'}

        observation = self.world.get_observations()
        gripper_base_position = observation["gripper_base_position"]
        gripper_base_orientation = observation["gripper_base_orientation"]

        # current tool pose
        response.current_tool_position = gripper_base_position[current_tool]
        response.current_tool_orientation = gripper_base_orientation[current_tool]

        # desired tool pose
        response.desired_tool_position = gripper_base_position[desired_tool]
        response.desired_tool_orientation = gripper_base_orientation[desired_tool]

        return response


def main(args=None):

    rclpy.init(args=args)
    sim_node = None
    try:
        sim_node = Simulation()
        if sim_node and rclpy.ok() and sim_node.robot is not None:
            rclpy.spin(sim_node)
        elif sim_node:
             sim_node.get_logger().error("Simulation node initialized but robot not found. Shutting down.")
        else:
            print("Simulation node failed to initialize.")

    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
    except Exception as e:
        print(f"ROS2 Spin Exception: {e}")
    finally:
        if sim_node:
            sim_node.get_logger().info("Shutting down simulation...")
            if sim_node.world:
                sim_node.world.stop()
            sim_node.simulation_app.close()
            sim_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Shutdown complete.")


if __name__ == '__main__':
    main()