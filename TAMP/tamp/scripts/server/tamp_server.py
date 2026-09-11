#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.duration import Duration as RclpyDuration

from tamp_interfaces.msg import PlanStep
from tamp_interfaces.srv import Plan, Execute, SetTampEnv, MoveToTarget, MoveToTargetJs, SetTampCfg
from builtin_interfaces.msg import Duration
from simulation_interfaces.srv import GetEntityState, SetSimulationState
from std_srvs.srv import SetBool

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from typing import Optional, List, Dict
import copy
import time

from cutamp.algorithm import run_cutamp, setup_cutamp
from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.constraint_checker import ConstraintChecker
from cutamp.scripts.utils import (
    default_constraint_to_mult,
    default_constraint_to_tol,
    setup_logging,
    get_tetris_tuned_constraint_to_mult,
)
import logging
from cutamp.cost_reduction import CostReducer

from envs.utils import TAMPEnvManager
from orchestration.planner_api import PlanningResult, with_explicit_pour_steps
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
import os as _os
import numpy as np

from curobo.types.state import JointState as CuroboJointState
from curobo.types.math import Pose as CuroboPose
from curobo.types.base import TensorDeviceType
from std_msgs.msg import Float32, String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
import tf2_ros

import os
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
import csv


class TAMP:

    def __init__(
        self,
        config: TAMPConfiguration,
        use_tetris_tuned_weights: bool = None,
    ):
        validate_tamp_config(config)

        self.config = config

        self.env = None

        self.current_env_name = "unknown"

        setup_logging()

        self.use_tetris_tuned_weights = use_tetris_tuned_weights

        self.constraint_to_mult = (
            get_tetris_tuned_constraint_to_mult() if self.use_tetris_tuned_weights else default_constraint_to_mult.copy()
        )
        self.cost_reducer = CostReducer(self.constraint_to_mult)
        self.constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())
        self._log = logging.getLogger(__name__)

        self.curobo_plan = None
        self.total_num_satisfying = None

        self.max_attempts = 3

        self.cmd_js_names = ["j1", "j2", "j3", "j4", "j5", "j6"]
        self.has_planned = False


    def update_config(self, config: TAMPConfiguration):
        if self.has_planned and config.robot != self.config.robot:
            raise RuntimeError(
                "robot configuration cannot change after planning in this process "
                f"({self.config.robot!r} -> {config.robot!r}); start the planner "
                "process assigned to that configuration"
            )
        self.config = config


    def update_env(
        self,
        name,
        poses: Dict[str, List[float]],
        movables: List[str],
        statics: List[str],
        ex_collision: List[str],
        rearrange_grid: str | None = None,
    ):
        self.current_env_name = name

        # A new manager produces an isolated world snapshot for every request.
        # Reusing the old manager mutated Cuboids in place and leaked state from
        # one task/trial into the next.
        env_manager = TAMPEnvManager()
        env_manager.update_entities(
            poses=poses,
            movables=movables,
            statics=statics,
            ex_collision=ex_collision,
            rearrange_grid=rearrange_grid,
        )

        if name == "transfer" :
            self.env, pour_region_pose = env_manager.load_env(name)
            self._log.info(f"pour_region_pose : {pour_region_pose}")
        else :
            self.env = env_manager.load_env(name)
        
    
    def plan(
        self,
        q_init: Optional[List[float]] = None,
        experiment_id: Optional[str] = None
    ):
        self.has_planned = True
        self.total_num_satisfying = 0
        self.last_plan_error = None

        start_time = time.time()
        attempts_used = 0
        success = False

        if self.env is not None:
            for _ in range(self.max_attempts):

                attempts_used += 1

                env = copy.deepcopy(self.env)

                try:
                    self.curobo_plan, self.total_num_satisfying = run_cutamp(
                        env=env,
                        config=self.config,
                        cost_reducer=self.cost_reducer,
                        constraint_checker=self.constraint_checker,
                        q_init=q_init,
                        experiment_id=experiment_id
                    )
                except Exception as e:
                    # Do NOT silently turn a planner crash into a fake "0 satisfying":
                    # doing so previously reported a genuinely SOLVED plan as a
                    # planning failure (and retried it 3x). Surface the FULL
                    # traceback so downstream crashes (e.g. an empty cuRobo
                    # trajectory in solve_curobo) are diagnosable. Retry behaviour
                    # is preserved by the enclosing loop.
                    self._log.exception(
                        "run_cutamp raised on attempt %d/%d for env '%s'",
                        attempts_used, self.max_attempts, self.current_env_name,
                    )
                    self.last_plan_error = e
                    self.total_num_satisfying = 0

                if self.total_num_satisfying > 0:
                    success = True
                    break
        else:
            raise ValueError("update_env is needed before plan")

        # All attempts crashed (never reached a real 0-satisfying result): make the
        # failure visible rather than masquerading as "planned, found nothing".
        if not success and self.last_plan_error is not None:
            self._log.error(
                "TAMP planning failed on all %d attempt(s) for env '%s' due to an "
                "exception (see traceback above): %r",
                attempts_used, self.current_env_name, self.last_plan_error,
            )
        
        end_time = time.time()
        planning_time = end_time - start_time

        # # CSV 파일에 결과 로깅
        # self._log_planning_result(
        #     task_name=self.current_env_name,
        #     experiment_id=experiment_id,
        #     success=success,
        #     attempts=attempts_used,
        #     planning_time=planning_time
        # )

        failure_reason = ""
        if not success:
            failure_reason = (
                f"planner_exception:{type(self.last_plan_error).__name__}"
                if self.last_plan_error is not None else "no_satisfying_particles"
            )
        return PlanningResult(
            plan=self.curobo_plan,
            success=success,
            total_num_satisfying=int(self.total_num_satisfying or 0),
            attempts=attempts_used,
            planning_time_s=planning_time,
            failure_reason=failure_reason,
        )
    

    def motion_plan(
        self,
        q_init: List,
        ee_translation_goal: np.array,
        ee_orientation_goal: np.array,
    ):
        tensor_args = TensorDeviceType()

        _, _, timer, world = setup_cutamp(self.env, self.config, q_init)
        motion_gen = world.get_motion_gen(collision_activation_distance=self.config.world_activation_distance)
        if self.config.warmup_motion_gen:
            with timer.time("curobo_motion_gen_warmup"):
                motion_gen.warmup()

        plan_config = MotionGenPlanConfig(
            timeout=0.5, time_dilation_factor=self.config.time_dilation_factor
        )

        cu_js = CuroboJointState(
            position=tensor_args.to_device(q_init),
            velocity=tensor_args.to_device(q_init) * 0.0,
            acceleration=tensor_args.to_device(q_init) * 0.0,
            jerk=tensor_args.to_device(q_init) * 0.0,
            joint_names=self.cmd_js_names,
        )
        ik_goal = CuroboPose(
            position=tensor_args.to_device(ee_translation_goal),
            quaternion=tensor_args.to_device(ee_orientation_goal),
        )

        result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
        succ = result.success.item()
        if succ:
            cmd_plan = result.get_interpolated_plan().get_ordered_joint_state(self.cmd_js_names)
        else:
            return None
        
        return cmd_plan


    def motion_plan_js(
        self,
        q_init: List,
        q_des: List,
    ):
        tensor_args = TensorDeviceType()

        _, _, timer, world = setup_cutamp(self.env, self.config, q_init)
        motion_gen = world.get_motion_gen(collision_activation_distance=self.config.world_activation_distance)
        if self.config.warmup_motion_gen:
            with timer.time("curobo_motion_gen_warmup"):
                motion_gen.warmup()

        plan_config = MotionGenPlanConfig(
            max_attempts=1, time_dilation_factor=self.config.time_dilation_factor
        )

        cu_js_init = CuroboJointState(
            position=tensor_args.to_device(q_init),
            velocity=tensor_args.to_device(q_init) * 0.0,
            acceleration=tensor_args.to_device(q_init) * 0.0,
            jerk=tensor_args.to_device(q_init) * 0.0,
            joint_names=self.cmd_js_names,
        )
        cu_js_des = CuroboJointState(
            position=tensor_args.to_device(q_des),
            velocity=tensor_args.to_device(q_des) * 0.0,
            acceleration=tensor_args.to_device(q_des) * 0.0,
            jerk=tensor_args.to_device(q_des) * 0.0,
            joint_names=self.cmd_js_names,
        )

        result = motion_gen.plan_single_js(cu_js_init.unsqueeze(0), cu_js_des.unsqueeze(0), plan_config)
        succ = result.success.item()
        if succ:
            cmd_plan = result.get_interpolated_plan().get_ordered_joint_state(self.cmd_js_names)
        else:
            return None
    
        return cmd_plan


    def _log_planning_result(self, task_name, experiment_id, success, attempts, planning_time):
        """
        Planning 결과를 CSV 파일로 저장합니다.
        """
        env_name = self.current_env_name
        log_file = "/home/home/sdl_ws/src/sdl_project/TAMP/tamp/logs/table_9/tranfer_stir_move/experiments.csv"
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["trial", "env_name", "success", "planning_time_sec", "attempts", "total_num_satisfying"])
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([
                timestamp, 
                env_name,
                success,
                f"{planning_time:.4f}",
                attempts, 
                self.total_num_satisfying
            ])
        
        self._log.info(f"Planning log saved to {log_file} (Task: {task_name}, Time: {planning_time:.4f}s)")

        

class TAMPServer(Node):

    def __init__(self, tamp):
        
        super().__init__("tamp_server")

        self.tamp: TAMP = tamp
        self.get_logger().info(f"Initialize TAMP Module ...")

        self.reentrant_group = ReentrantCallbackGroup()

        # variables
        self.plan_to_execute = None
        self.plan_index = 0
        self.current_plan_step = 0
        self.execute_plan_timer = None
        self.move_to_target_timer = None

        # duration between plan_steps 
        self.is_waiting_after_step = False
        self.wait_start_time = None
        self.wait_duration = RclpyDuration(seconds=0.4)

        # commands
        self.arm_commands_publisher = self.create_publisher(JointState, "isaac_arm_commands", 10)
        self.gripper_commands_cli = self.create_client(SetBool, "isaac_gripper_commands", callback_group=self.reentrant_group)

        # MAJOR-1 (REFACTOR.md III): before a gripper CLOSE the executor tells the
        # sim WHICH object the planner intends to grasp (the Pick op's target), so
        # the sim welds that object instead of the geometric-nearest one. Latched
        # (TRANSIENT_LOCAL + RELIABLE, depth 1) so the sim reliably has the name
        # by the time the close service is called, even across the DDS boundary.
        _latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.grasp_target_pub = self.create_publisher(String, "set_grasp_target", _latched_qos)
        # Publishes the op_name currently being executed so an external harness
        # can attribute measured carried-vessel tilt to the transport (MoveHolding)
        # phase specifically, separate from the pour (R3#2 theta_max question).
        self.current_op_pub = self.create_publisher(String, "tamp_current_op", _latched_qos)
        self.set_simulation_state_cli = self.create_client(SetSimulationState, "set_simulation_state", callback_group=self.reentrant_group)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # subscription
        self.joint_states_subscription = self.create_subscription(JointState, "isaac_joint_states", self.joint_states_cb, 10)

        # services
        self.plan_srv = self.create_service(
            Plan, 'tamp_plan', self.tamp_plan_cb, 
            callback_group=self.reentrant_group
        )
        self.execute_srv = self.create_service(
            Execute, 'plan_execute', self.execute_plan_cb, 
            callback_group=self.reentrant_group
        )
        self.set_tamp_env_srv = self.create_service(
            SetTampEnv, 'set_tamp_env', self.set_tamp_env_cb, 
            callback_group=self.reentrant_group
        )
        self.set_tamp_cfg_srv = self.create_service(
            SetTampCfg, 'set_tamp_cfg', self.set_tamp_cfg_cb,
            callback_group=self.reentrant_group
        )
        self.move_to_target_srv = self.create_service(
            MoveToTarget, 'move_to_target', self.move_to_target_cb,
            callback_group=self.reentrant_group
        )
        self.move_to_target_js_srv = self.create_service(
            MoveToTargetJs, 'move_to_target_js', self.move_to_target_js_cb,
            callback_group=self.reentrant_group
        )
        
        # clients
        self.get_entity_state_cli = self.create_client(
            GetEntityState, 'get_entity_state',
            callback_group=self.reentrant_group
        )

        # robot states
        self.joint_states = JointState()
        self.arm_commands = JointState()

        # operator
        self.last_operator = None
        self.saved_state_after_operator = None 

        self.scale = 0.0
        self.scale_sub = self.create_subscription(
            Float32, "raw_scale_data",
            self.scale_cb, 10
        )


    # # Using Tag Data
    # async def set_tamp_env_cb(self, request, response):

    #     env_name = request.env_name
    #     entities = request.entities
    #     movables = request.movables
    #     statics = request.statics
    #     ex_collision = request.ex_collision

    #     entities_states = {
    #         "poses": {},
    #         "movables": movables,
    #         "statics": statics,
    #         "ex_collision": ex_collision,
    #     }

    #     for entity in entities:
    #         # 1. Perception Node에서 발행하는 TF를 최우선으로 찾습니다. (예: beaker)
    #         target_frame = f"{entity}"
            
    #         try:
    #             # TF Buffer에서 베이스 링크 기준 최신 변환값을 가져옵니다.
    #             trans = self.tf_buffer.lookup_transform(
    #                 'base_link', 
    #                 target_frame, 
    #                 rclpy.time.Time()
    #             )
                
    #             # [꿀팁] 이전 단계에서 발견한 깊이 오차 보정 (Z축 Grounding)
    #             # 물체가 비커나 플라스크일 경우 테이블 높이(0.002)로 Z값을 강제 고정하여 파지 안정성을 높입니다.
    #             z_value = trans.transform.translation.z
    #             if entity in ["beaker", "flask"]:
    #                 z_value = 0.002 
                
    #             entity_pose = [
    #                 trans.transform.translation.x,
    #                 trans.transform.translation.y,
    #                 z_value + 0.01,  # 기존의 여유 충돌 값(+0.01) 유지
    #                 trans.transform.rotation.w,
    #                 trans.transform.rotation.x,
    #                 trans.transform.rotation.y,
    #                 trans.transform.rotation.z
    #             ]
    #             entities_states["poses"][entity] = entity_pose
    #             self.get_logger().info(f"[{entity}] TF Pose 획득 완료 (Frame: {target_frame})")

    #         except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    #             # 2. TF 트리에 없는 정적 물체(table, stirrer 등)는 기존처럼 시뮬레이션 서비스로 획득합니다.
    #             self.get_logger().info(f"[{entity}] TF가 없어 Isaac Sim 서비스로 대체합니다.")
                
    #             if not self.get_entity_state_cli.wait_for_service(timeout_sec=1.0):
    #                 self.get_logger().warn('GetEntityState service not available, skipping...')
    #                 continue

    #             get_entity_state_request = GetEntityState.Request()
    #             get_entity_state_request.entity = "/World/" + entity

    #             get_entity_state_response = await self.get_entity_state_cli.call_async(get_entity_state_request)

    #             if get_entity_state_response.result.result == 1:
    #                 entity_pose = [
    #                     get_entity_state_response.state.pose.position.x,
    #                     get_entity_state_response.state.pose.position.y,
    #                     get_entity_state_response.state.pose.position.z + 0.01,
    #                     get_entity_state_response.state.pose.orientation.w,
    #                     get_entity_state_response.state.pose.orientation.x,
    #                     get_entity_state_response.state.pose.orientation.y,
    #                     get_entity_state_response.state.pose.orientation.z
    #                 ]
    #                 entities_states["poses"][entity] = entity_pose

    #     response.success = True

    #     self.tamp.update_env(
    #         name=env_name,
    #         poses=entities_states["poses"],
    #         movables=entities_states["movables"],
    #         statics=entities_states["statics"],
    #         ex_collision=entities_states["ex_collision"]
    #     )

    #     return response
    
    # World State source. "ground_truth" reads the simulator's own poses;
    # "perception" reads the AprilTag pipeline's fused TF for the tagged
    # vessels and the simulator only for the untagged furniture (table,
    # stirrer, trays), which carries no tag in either the simulated or the
    # physical cell. Set with SDL_STATE_SOURCE.
    PERCEPTION_ENTITIES = ("beaker", "flask")

    # A perception pose older than this is not a current observation. Detection
    # is intermittent for a partly occluded tag, and a TF buffer keeps the last
    # transform indefinitely, so without an age check a planning trial can be
    # built on a pose measured a minute earlier and still be reported as
    # perception-in-the-loop.
    PERCEPTION_MAX_AGE_S = float(os.environ.get("SDL_PERCEPTION_MAX_AGE_S", "2.0"))

    def _perception_pose(self, entity):
        """base_link -> entity from the perception TF, or None.

        Same convention as the ground-truth path: the object's own pose,
        w-first quaternion, no z correction -- the planner lift is applied
        downstream in envs/utils.py, so adding one here would double it.
        """
        try:
            t = self.tf_buffer.lookup_transform(
                "base_link", entity, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(
                "[state] %s has no perception pose: %s" % (entity, exc))
            return None
        stamp = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
        now = self.get_clock().now().nanoseconds * 1e-9
        age = now - stamp
        if age > self.PERCEPTION_MAX_AGE_S:
            self.get_logger().warn(
                "[state] %s perception pose is %.1f s old (limit %.1f s); "
                "treating it as not localized" % (entity, age,
                                                  self.PERCEPTION_MAX_AGE_S))
            return None
        p, q = t.transform.translation, t.transform.rotation
        self.get_logger().info(
            "[state] %s from perception at (%.4f, %.4f, %.4f), age %.2f s"
            % (entity, p.x, p.y, p.z, age))
        return [p.x, p.y, p.z, q.w, q.x, q.y, q.z]

    async def set_tamp_env_cb(self, request, response):

        while not self.get_entity_state_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        state_source = os.environ.get("SDL_STATE_SOURCE", "ground_truth").strip()
        if state_source not in ("ground_truth", "perception"):
            self.get_logger().error(
                "[state] unknown SDL_STATE_SOURCE %r; refusing to guess"
                % state_source)
            response.success = False
            return response

        env_name = request.env_name
        entities = request.entities
        movables = request.movables
        statics = request.statics
        ex_collision = request.ex_collision
        rearrange_grid = request.rearrange_grid

        entities_states = {
            "poses": {},
            "movables": movables,
            "statics": statics,
            "ex_collision": ex_collision,
            "rearrange_grid": rearrange_grid,
        }

        sources = {}
        for entity in entities:
            if (state_source == "perception"
                    and entity in self.PERCEPTION_ENTITIES):
                entity_pose = self._perception_pose(entity)
                if entity_pose is None:
                    # A tagged vessel the pipeline never localized. Falling back
                    # to ground truth here would silently report a
                    # perception-in-the-loop result that was not one.
                    self.get_logger().error(
                        "[state] no perception pose for %s; the trial is a "
                        "perception failure, not a planning one" % entity)
                    response.success = False
                    return response
                entities_states["poses"][entity] = entity_pose
                sources[entity] = "perception"
                continue

            get_entity_state_request = GetEntityState.Request()
            get_entity_state_request.entity = "/World/" + entity

            get_entity_state_response = await self.get_entity_state_cli.call_async(get_entity_state_request)

            if get_entity_state_response.result.result == 1:

                response.success = True
                entity_pose = [
                    get_entity_state_response.state.pose.position.x,
                    get_entity_state_response.state.pose.position.y,
                    get_entity_state_response.state.pose.position.z,
                    get_entity_state_response.state.pose.orientation.w,
                    get_entity_state_response.state.pose.orientation.x,
                    get_entity_state_response.state.pose.orientation.y,
                    get_entity_state_response.state.pose.orientation.z
                ]
                entities_states["poses"][entity] = entity_pose
                sources[entity] = "ground_truth"

        self.get_logger().info(
            "[state] source=%s %s" % (state_source, " ".join(
                "%s:%s" % (k, sources[k]) for k in sorted(sources))))

        self.tamp.update_env(
            name=env_name,
            poses=entities_states["poses"],
            movables=entities_states["movables"],
            statics=entities_states["statics"],
            ex_collision=entities_states["ex_collision"],
            rearrange_grid=entities_states["rearrange_grid"],
        )

        return response
    
    def set_tamp_cfg_cb(self, request, response):

        if request.num_particles == 0:
            request.num_particles = 1024
        if request.robot == "":
            request.robot = "fr5"
        if request.grasp_dof == 0:
            request.grasp_dof = 4
        if request.approach == "":
            request.approach = "optimization"
        if request.num_resampling_attempts == 0:
            request.num_resampling_attempts = 100
        if request.num_opt_steps == 0:
            request.num_opt_steps = 1000
        if request.num_initial_plans == 0:
            request.num_initial_plans = 1
        if request.opt_viz_interval == 0:
            request.opt_viz_interval = 10
        if request.time_dilation_factor == 0.0:
            # 0.0 -> ZeroDivisionError in cuRobo trajectory retime (1.0 /
            # time_dilation_factor). Clients (e.g. tamp_client.set_tamp_cfg) don't
            # set this field, so it arrives as 0.0; default it like every other
            # field above, matching the __main__ default of 0.5.
            request.time_dilation_factor = 0.5

        config = TAMPConfiguration(
            num_particles=request.num_particles,
            robot=request.robot,
            grasp_dof=request.grasp_dof,
            approach=request.approach,
            num_resampling_attempts=request.num_resampling_attempts,
            num_opt_steps=request.num_opt_steps,
            max_loop_dur=None,
            optimize_soft_costs=request.optimize_soft_costs,
            soft_cost=None,
            num_initial_plans=request.num_initial_plans, # Changed from 1 to 30
            cache_subgraphs=None,
            curobo_plan=True,
            enable_visualizer=request.enable_visualizer,
            opt_viz_interval=request.opt_viz_interval,
            viz_robot_mesh=request.viz_robot_mesh,
            enable_experiment_logging=request.enable_experiment_logging,
            time_dilation_factor=request.time_dilation_factor,
            rr_spawn=request.rr_spawn,
        )
        validate_tamp_config(config)

        try:
            self.tamp.update_config(config=config)
            response.success = True
        except RuntimeError as exc:
            self.get_logger().error(str(exc))
            response.success = False

        return response

    def joint_states_cb(self, msg):
        self.joint_states = msg
    
    def scale_cb(self, msg):
        self.scale = msg.data

    def tamp_plan_cb(self, request, response):

        # Planning runs in this process and does not require pausing Isaac Sim.
        # Keeping the timeline playing preserves /isaac_joint_states, so q_init
        # is the actual live state rather than an empty value/q_home fallback.
        q_init = list(self.joint_states.position[:6])  # num_dof = 6
        if len(q_init) < 6:
            # isaac_joint_states not received yet (e.g. the sim is paused, so the
            # Isaac ActionGraph joint-state publisher isn't ticking). An empty
            # q_init makes best_particle["q0"] empty and crashes solve_curobo
            # ("shape '[1, 6]' is invalid for input of size 0"). Fall back to the
            # robot's home configuration so planning still runs.
            from cutamp.robots import get_q_home
            q_init = list(get_q_home(self.tamp.config.robot))
            self.get_logger().warn(
                f"isaac_joint_states unavailable ({len(self.joint_states.position)} dofs); "
                f"using q_home for '{self.tamp.config.robot}' as q_init."
            )

        try:
            result = self.tamp.plan(q_init, None)
            curobo_plan = with_explicit_pour_steps(result.plan)
            total_num_satisfying = result.total_num_satisfying
            response.plan_success = result.success
            response.total_num_satisfying = total_num_satisfying

            if result.success:
                self.plan_to_execute = curobo_plan
                response.curobo_plan = self.process_plan(curobo_plan)
            else:
                # A failed request invalidates any plan left by an earlier call.
                # Do not try to encode None and do not allow stale execution.
                self.plan_to_execute = None
                response.curobo_plan = []

            self.get_logger().info(
                f"TAMP planning finished. Success: {response.plan_success}, "
                f"Satisfying particles: {total_num_satisfying}, "
                f"attempts: {result.attempts}, time: {result.planning_time_s:.3f}s, "
                f"failure_reason: {result.failure_reason or 'none'}"
            )

        except Exception as e:
            self.get_logger().error(f"TAMP planning failed with exception: {e}")
            response.plan_success = False

        return response
    

    def process_plan(self, plan):

        processed_plan = []

        for plan_step in plan:
            value = PlanStep()
            value.op_name = plan_step.get("op_name", "")

            if plan_step["type"] == "trajectory":
                value.type = 0 # TRAJECTORY
                traj = []

                dt = 0.1 # plan_step["dt"]
                duration = Duration()
                dt_nanosec = int(dt * 1e9)

                for i in range(plan_step["plan"].position.shape[0]):
                    traj_point = JointTrajectoryPoint()
                    
                    traj_point.positions = plan_step["plan"].position[i].tolist()
                    traj_point.velocities = plan_step["plan"].velocity[i].tolist()
                    traj_point.accelerations = plan_step["plan"].acceleration[i].tolist()
                    traj_point.time_from_start = duration
                    traj.append(traj_point)

                    duration.nanosec += dt_nanosec
                    
                    if duration.nanosec >= 1e9:
                        duration.sec += 1
                        duration.nanosec = int(duration.nanosec % 1e9)

                value.joint_trajectory.points = traj
                value.joint_trajectory.header.stamp = self.get_clock().now().to_msg()
                value.joint_trajectory.joint_names = plan_step["plan"].joint_names

            elif plan_step["type"] == "gripper":
                value.type = 1 
                value.action = plan_step["action"]
            elif plan_step["type"] == "pour_path":
                # Published as a POUR step: clients only need to know that a pour
                # happens here; the joint path is executed in this process.
                value.type = 2  # POUR
                value.action = "pour_path"

            elif plan_step["type"] == "pour":
                value.type = 2
            else:
                raise ValueError("unsupported plan step type")
            
            processed_plan.append(value)

        return processed_plan
    
    def execute_plan_cb(self, request, response):

        start_req = SetSimulationState.Request()
        start_req.state.state = 1 # play

        if not self.set_simulation_state_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("Simulation Pause Failed")
            return

        self.set_simulation_state_cli.call(start_req)

        if not self.plan_to_execute or len(self.plan_to_execute) == 0:
            self.get_logger().warn("No plan to execute.")
            response.execute_success = False
            return response

        self.get_logger().info("Starting plan execution...")

        for plan_index, plan_part in enumerate(self.plan_to_execute):

            current_op = plan_part.get("op_name", "")
            # Broadcast the op currently executing so an external harness can
            # isolate transport (MoveHolding) tilt from the pour (R3#2).
            self._publish_current_op(current_op)

            # trajectory or gripper execution logic
            plan_type = plan_part.get("type")

            if plan_type == "trajectory":
                plan_trajectory = plan_part["plan"]
                # Replay period per planned waypoint. This deliberately overrides
                # cuRobo's interpolation_dt (~0.02 s): the simulated arm is position
                # controlled and chases each published target, so replaying slower
                # reduces tracking error. That matters much more for a side grasp
                # than a top grasp -- a side grasp maps wrist-roll tracking error
                # onto the held vessel's tilt 1:1 (which is exactly why it can pour
                # at all), whereas under a top grasp the same error becomes yaw.
                # Configurable so the executed-tilt/execution-time trade-off can be
                # measured and reported rather than hidden in a literal.
                dt = float(_os.environ.get("SDL_EXEC_DT", "0.04"))
                num_waypoints = plan_trajectory.position.shape[0]

                for i in range(num_waypoints):
                    self.arm_commands.header.stamp = self.get_clock().now().to_msg()
                    self.arm_commands.name = plan_trajectory.joint_names
                    self.arm_commands.position = plan_trajectory.position[i].tolist()
                    self.arm_commands_publisher.publish(self.arm_commands)
                    time.sleep(dt)

                time.sleep(dt)

            elif plan_type == "gripper":
                self.get_logger().info(f"Executing gripper: {plan_part['action']}")
                self.execute_gripper_action(plan_part)
                time.sleep(1.0)

            elif plan_type == "pour":
                self.saved_state_after_operator = self.joint_states.position[:6]
                self.pouring()
                self.get_logger().info("Explicit pouring step completed.")

            elif plan_type == "pour_path":
                self.saved_state_after_operator = self.joint_states.position[:6]
                self.pouring_along_path(plan_part)
                self.get_logger().info("Lip-pivot pouring step completed.")

            else:
                self.get_logger().error(f"Unknown plan type: {plan_type}")
                response.execute_success = False
                return response

        self.get_logger().info("Plan execution completed.")
        self._publish_current_op("idle")
        response.execute_success = True
        return response

    def _publish_current_op(self, op_name: str):
        msg = String()
        msg.data = str(op_name)
        self.current_op_pub.publish(msg)

    def _pour_shaping_kernel(self, dt, kernel_horizon=2.5,
                             shaping_freq=0.8, shaping_decay=1.2, alpha=0.15):
        """Convolution kernel that smooths the PD output into a wrist command."""
        n_kernel = int(kernel_horizon / dt)
        t_kernel = np.arange(n_kernel) * dt
        raw = np.exp(-shaping_decay * t_kernel) * np.sin(2.0 * np.pi * shaping_freq * t_kernel)
        kernel = raw
        if np.sum(np.abs(raw)) > 1e-6:
            kernel = (1 - alpha) * (raw / np.sum(np.abs(raw))) \
                     + alpha * (raw / (np.max(np.abs(raw)) + 1e-6))
        return kernel, n_kernel

    def pouring_along_path(self, plan_part):
        """Pour by moving ALONG a planned lip-pivot path.

        `pouring()` rotates the wrist joint, whose axis passes through the grasp
        point, so the vessel's lip swings away from the target as it tilts
        (measured 36-81 mm from the flask axis at peak tilt, against a ~17 mm
        mouth radius). This instead walks a joint path that rotates the vessel
        about a horizontal axis through its own lip, so the lip stays where the
        placement put it -- over the mouth -- at every angle.

        The weight controller is unchanged in character: the same PD output and
        convolution shaping, the same rate limit, and the same stopping rule. It
        just advances a position along the path instead of an angle of one joint,
        so the pour is still adaptive. The path's end IS the tilt cap, which also
        removes the runaway the old loop allowed (measured pour angles ranged
        from 30.6 to 84.4 degrees).
        """
        positions = plan_part["positions"]
        path = np.asarray(
            positions.detach().cpu().numpy() if hasattr(positions, "detach") else positions,
            dtype=float,
        )
        n_way = path.shape[0] - 1
        step_rad = float(plan_part.get("step_rad", 0.0))
        if n_way < 1 or step_rad <= 0.0:
            self.get_logger().warn("Empty lip-pivot pour path; falling back to the wrist pour.")
            self.pouring()
            return

        Kp, Kd = 0.008, 0.002
        weight_target = 50.0
        dt = 0.04
        max_steps = 800
        MAX_TILT_RATE = 0.5      # [rad/s] same saturation as the wrist pour

        kernel, n_kernel = self._pour_shaping_kernel(dt)
        self.get_logger().info(
            "Starting lip-pivot pouring: %d waypoints, %.3f rad per waypoint "
            "(max tilt %.1f deg)." % (n_way, step_rad, np.degrees(n_way * step_rad))
        )

        def q_at(u):
            u = float(np.clip(u, 0.0, n_way))
            lo = int(np.floor(u))
            hi = min(lo + 1, n_way)
            frac = u - lo
            return (1.0 - frac) * path[lo] + frac * path[hi]

        def publish(u):
            self.arm_commands.header.stamp = self.get_clock().now().to_msg()
            self.arm_commands.name = self.tamp.cmd_js_names
            self.arm_commands.position = q_at(u).tolist()
            self.arm_commands_publisher.publish(self.arm_commands)

        u = 0.0
        prev_error = 0.0
        pd_history = []
        for step in range(max_steps):
            error = weight_target - self.scale
            d_error = (error - prev_error) / dt
            prev_error = error
            pd_output = Kp * error + Kd * d_error

            pd_history.append(pd_output)
            if len(pd_history) > n_kernel:
                pd_history.pop(0)
            valid = min(len(pd_history), n_kernel)
            v_cmd = float(np.dot(np.array(pd_history[-valid:]), kernel[:valid][::-1]))
            v_cmd = max(-MAX_TILT_RATE, min(MAX_TILT_RATE, v_cmd))

            u = float(np.clip(u + (v_cmd * dt) / step_rad, 0.0, n_way))
            publish(u)

            self.get_logger().info(
                "[Pouring] step=%d, weight=%.2f, err=%.2f, pd=%.3f, v_cmd=%.4f, "
                "tilt=%.1fdeg (u=%.2f/%d)"
                % (step, self.scale, error, pd_output, v_cmd,
                   np.degrees(u * step_rad), u, n_way)
            )

            if abs(error) < 0.5:
                self.get_logger().info("Target weight reached. Stopping pouring.")
                break
            if u >= n_way and v_cmd > 0:
                self.get_logger().warn(
                    "Reached the end of the planned pour path (%.1f deg) without "
                    "reaching the target weight." % np.degrees(n_way * step_rad)
                )
                break
            time.sleep(dt)

        # Un-tilt by reversing along the SAME path, at the same rate limit, so the
        # vessel ends exactly where the placement left it and the remaining plan
        # steps carry it upright.
        back_du = (MAX_TILT_RATE * dt) / step_rad
        while u > 0.0:
            u = max(0.0, u - back_du)
            publish(u)
            time.sleep(dt)
        self.get_logger().info("Lip-pivot pour returned to the upright placement.")

    def pouring(self):

        self.get_logger().info("Starting PD-controlled pouring with convolution shaping...")

        Kp = 0.008
        Kd = 0.002

        weight_target = 50.0
        dt = 0.04
        max_steps = 800

        # --- Tilt-rate limit (IV-2) ---------------------------------------------------
        # The PD output is convolution-shaped, so v_cmd (wrist angular velocity) can spike.
        # Saturate it to a maximum consistent with a *controlled* pour so the wrist cannot
        # slew abruptly (which would slosh/over-pour on real hardware and spike the tilt in sim).
        # 0.5 rad/s (~28.6 deg/s): at the 25 Hz (dt=0.04 s) loop this caps each step to
        # ~0.02 rad (~1.15 deg), keeping the tilt motion smooth.
        MAX_TILT_RATE = 0.5     # [rad/s] wrist angular-velocity saturation
        # --- Absolute pour-tilt safety cap (θ_max for the POUR, not the 5 deg transport bound) -
        # Bounds runaway if the (synthetic) scale never reaches the target: stop after this much
        # cumulative wrist rotation from the pour start. ~2.0 rad (~115 deg) is well past a normal
        # pour (which stops on |error|<0.5), so it never fires on a healthy pour.
        MAX_POUR_TILT = 2.0     # [rad] max cumulative wrist tilt from pour start

        current_q = list(self.joint_states.position[:6])
        last_idx = 5
        theta_start = current_q[last_idx]   # wrist angle at pour start (for the absolute-tilt cap)
        prev_error = 0.0

        kernel_horizon = 2.5   # 필터 길이 1초
        n_kernel = int(kernel_horizon / dt)

        t_kernel = np.arange(n_kernel) * dt

        shaping_freq = 0.8      # Hz (1.25초 주기 근처)
        shaping_decay = 1.2     # exp(-a t) 감쇠

        kernel = np.exp(-shaping_decay * t_kernel) * np.sin(2.0 * np.pi * shaping_freq * t_kernel)

        if np.sum(np.abs(kernel)) > 1e-6:
            kernel_raw = np.exp(-shaping_decay * t_kernel) * np.sin(2*np.pi*shaping_freq * t_kernel)

            kernel_sum_norm = kernel_raw / np.sum(np.abs(kernel_raw))
            kernel_max_norm = kernel_raw / (np.max(np.abs(kernel_raw)) + 1e-6)

            alpha = 0.15   
            kernel = (1 - alpha) * kernel_sum_norm + alpha * kernel_max_norm

        pd_history = []

        self.get_logger().info(f"Pouring kernel length: {n_kernel}, first 5 taps: {kernel[:5]}")

        for step in range(max_steps):

            current_weight = self.scale 
            error = weight_target - current_weight
            d_error = (error - prev_error) / dt
            prev_error = error

            pd_output = Kp * error + Kd * d_error

            pd_history.append(pd_output)
            if len(pd_history) > n_kernel:
                pd_history.pop(0)

            valid_len = min(len(pd_history), n_kernel)
            pd_segment = np.array(pd_history[-valid_len:])           # 최근 valid_len개
            kernel_segment = kernel[:valid_len][::-1]                # 뒤집어서 causal conv

            v_cmd = float(np.dot(pd_segment, kernel_segment))        # wrist 각속도 (rad/s)

            # (5b) tilt-rate limit: saturate wrist angular velocity to +/- MAX_TILT_RATE
            v_cmd = max(-MAX_TILT_RATE, min(MAX_TILT_RATE, v_cmd))

            current_q[last_idx] += v_cmd * dt

            # (5c) absolute pour-tilt safety cap: never rotate the wrist past MAX_POUR_TILT
            tilt_from_start = current_q[last_idx] - theta_start
            if abs(tilt_from_start) >= MAX_POUR_TILT:
                current_q[last_idx] = theta_start + np.sign(tilt_from_start) * MAX_POUR_TILT
                self.get_logger().warn(
                    f"Pour tilt cap reached (|Δθ|>={MAX_POUR_TILT:.2f} rad). Stopping pouring."
                )
                # publish the clamped pose once, then stop
                self.arm_commands.header.stamp = self.get_clock().now().to_msg()
                self.arm_commands.name = self.tamp.cmd_js_names
                self.arm_commands.position = current_q.copy()
                self.arm_commands_publisher.publish(self.arm_commands)
                break

            # (6) 명령 publish
            self.arm_commands.header.stamp = self.get_clock().now().to_msg()
            self.arm_commands.name = self.tamp.cmd_js_names
            self.arm_commands.position = current_q.copy()
            self.arm_commands_publisher.publish(self.arm_commands)

            # (7) 로그
            self.get_logger().info(
                f"[Pouring] step={step}, weight={current_weight:.2f}, "
                f"err={error:.2f}, pd={pd_output:.3f}, v_cmd={v_cmd:.4f}, "
                f"wrist={current_q[last_idx]:.3f}, "
                f"pd_output={pd_output:.2f}"
            )

            # (8) 목표 근처면 정지
            if abs(error) < 0.5:
                self.get_logger().info("Target weight reached. Stopping pouring.")
                break

            time.sleep(dt)

        self.get_logger().info("✔ PD + convolution-shaped pouring finished.")

        # Return the pour joint to its pre-pour attitude before the plan continues.
        # The pour tilts the held vessel by design (with the side grasp the pour
        # joint rotates about the ee approach axis, so vessel tilt tracks it 1:1),
        # but the REMAINING plan steps still carry the vessel to its placement.
        # Carrying it tilted would spill on real hardware and makes the measured
        # transport tilt meaningless: before this, Place executed at 56.7 deg while
        # the planner's own carry segments were 0.13 deg. Rate-bounded with the
        # same MAX_TILT_RATE as the pour, and it ends exactly at theta_start.
        self._untilt_pour_joint(current_q, last_idx, theta_start, MAX_TILT_RATE, dt)

    def _untilt_pour_joint(self, current_q, joint_idx, target, max_rate, dt):
        """Rate-bounded return of the pour joint to `target`, ending exactly there."""
        start = float(current_q[joint_idx])
        delta = float(target) - start
        if abs(delta) <= 1e-9:
            return
        steps = max(1, int(np.ceil(abs(delta) / max(max_rate * dt, 1e-9))))
        for step in range(1, steps + 1):
            current_q[joint_idx] = start + delta * (step / steps)
            self.arm_commands.header.stamp = self.get_clock().now().to_msg()
            self.arm_commands.name = self.tamp.cmd_js_names
            self.arm_commands.position = current_q.copy()
            self.arm_commands_publisher.publish(self.arm_commands)
            time.sleep(dt)
        self.get_logger().info(
            "Pour joint returned to its pre-pour attitude (%.3f rad) over %d rate-bounded steps."
            % (float(target), steps)
        )


    def execute_gripper_action(self, plan_part):
        """
        Executes a gripper action by making a synchronous service call.
        """
        # Wait for the service to be available.
        if not self.gripper_commands_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Gripper command service not available.')
            return

        request = SetBool.Request()

        if plan_part["action"] == "close":
            request.data = True
            # MAJOR-1: publish the planner's intended grasp target (if the plan
            # step carries one) and give the latched topic a moment to be
            # delivered + applied in the sim BEFORE the close service fires, so
            # the sim welds the intended object rather than the nearest one.
            target = plan_part.get("target")
            if target:
                msg = String()
                msg.data = str(target)
                self.grasp_target_pub.publish(msg)
                self.get_logger().info(f"grasp target -> '{target}' (published before close)")
                time.sleep(0.5)
        else:  # open
            request.data = False

        # Use the synchronous service call. This blocks until the call is complete.
        # This is safe due to the MultiThreadedExecutor and the client's ReentrantCallbackGroup.
        response = self.gripper_commands_cli.call(request)

        if not response.success:
            self.get_logger().warn(f"Gripper action failed: {response.message}")


    def move_to_target_cb(self, request, response):

        self.cmd_plan = self.tamp.motion_plan(
            q_init=self.joint_states.position[:6].tolist(),
            ee_translation_goal=np.array(request.target_position),
            ee_orientation_goal=np.array(request.target_orientation),
        )

        self.current_plan_step = 0
        timer_period = 0.016
        
        self.move_to_target_timer = self.create_timer(timer_period, self.move_to_target_timer_cb)
        response.success = True

        return response
    
    def move_to_target_js_cb(self, request, response):

        self.cmd_plan = self.tamp.motion_plan_js(
            q_init=self.joint_states.position[:6].tolist(),
            q_des=request.q_des,
        )

        self.current_plan_step = 0
        timer_period = 0.016
        
        self.move_to_target_timer = self.create_timer(timer_period, self.move_to_target_timer_cb)
        response.success = True

        return response

    def move_to_target_timer_cb(self):

        if not self.cmd_plan or self.current_plan_step >= len(self.cmd_plan):
            if self.move_to_target_timer is not None:
                self.move_to_target_timer.cancel()
                self.move_to_target_timer = None
            return

        self.arm_commands.header.stamp = self.get_clock().now().to_msg()
        self.arm_commands.name = self.tamp.cmd_js_names
        self.arm_commands.position = self.cmd_plan[self.current_plan_step].position.tolist()
        self.arm_commands_publisher.publish(self.arm_commands)

        self.current_plan_step += 1
        

if __name__ == "__main__":

    rclpy.init()

    config = TAMPConfiguration(
        num_particles=1024,
        robot="fr5",
        grasp_dof=6,
        approach="optimization",
        num_resampling_attempts=100,
        num_opt_steps=1000,
        max_loop_dur=None,
        optimize_soft_costs=False,
        soft_cost=None,
        num_initial_plans=1, # Changed from 1 to 30
        cache_subgraphs=None,
        curobo_plan=True,
        enable_visualizer=False,
        # opt_viz_interval=10,
        viz_robot_mesh=False,
        enable_experiment_logging=False,
        time_dilation_factor=0.5,
    )

    tamp = TAMP(
        config=config,
        use_tetris_tuned_weights=None
    )

    tamp_server = TAMPServer(tamp)

    executor = MultiThreadedExecutor(num_threads=4)

    executor.add_node(tamp_server)

    try:
        tamp_server.get_logger().info("Starting TAMP server with MultiThreadedExecutor.")
        executor.spin()
    finally:
        executor.shutdown()
        tamp_server.destroy_node()
        rclpy.shutdown()
