#!/home/home/anaconda3/envs/sdl/bin/python
from typing import Dict
import argparse
import time
import xml.etree.ElementTree as ET
import yaml
import os
import math
import sys

import rclpy
from rclpy.client import Client
from rclpy.node import Node

from tamp_interfaces.srv import (
    Plan, Execute, SetTampEnv, SetTampCfg, ToolChange, MoveToTarget, MoveToTargetJs, GetRobotInfo, GetToolInfo
)
from std_srvs.srv import SetBool
from simulation_interfaces.srv import GetEntityState
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

PROJECT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src", "sdl_project")
sys.path.append(os.path.join(PROJECT_PATH, "LLM"))
# Llama import
from llama.script.action_reasoner.model import ActionReasoner


GRID_PROPERTIES = {
    "G1": "plate zone",
    "G2": "stirrer zone",
    "G7": "workspace edge",
    "G8": "workspace edge",
    "G12": "workspace edge",
}


class TAMPClient(Node):

    def __init__(self):
        rclpy.init(args=None)
        super().__init__(
            "cutamp_client",
            parameter_overrides=[
                rclpy.Parameter(
                    "use_sim_time",
                    rclpy.Parameter.Type.BOOL,
                    True)
            ]
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Service Clients
        self.plan_client = self.create_client(Plan, 'tamp_plan')
        self.execute_client = self.create_client(Execute, 'plan_execute')
        self.set_tamp_env_client = self.create_client(SetTampEnv, 'set_tamp_env')
        self.set_tamp_cfg_client = self.create_client(SetTampCfg, "set_tamp_cfg")
        self.tool_change_client = self.create_client(ToolChange, 'tool_change')
        self.gripper_commands_client = self.create_client(SetBool, "isaac_gripper_commands")
        self.move_to_target_client = self.create_client(MoveToTarget, "move_to_target")
        self.move_to_target_js_client = self.create_client(MoveToTargetJs, "move_to_target_js")
        self.get_robot_info_client = self.create_client(GetRobotInfo, "get_robot_info")
        self.get_tool_info_client = self.create_client(GetToolInfo, "get_tool_info")
        self.get_entity_state_client = self.create_client(GetEntityState, 'get_entity_state')

        while (
            not self.plan_client.wait_for_service(timeout_sec=1.0)
            or not self.execute_client.wait_for_service(timeout_sec=1.0)
            or not self.set_tamp_env_client.wait_for_service(timeout_sec=1.0)
            or not self.set_tamp_cfg_client.wait_for_service(timeout_sec=1.0)
            or not self.tool_change_client.wait_for_service(timeout_sec=1.0)
            or not self.gripper_commands_client.wait_for_service(timeout_sec=1.0)
            or not self.move_to_target_client.wait_for_service(timeout_sec=1.0)
            or not self.get_tool_info_client.wait_for_service(timeout_sec=1.0)
            or not self.get_robot_info_client.wait_for_service(timeout_sec=1.0)
            or not self.get_entity_state_client.wait_for_service(timeout_sec=1.0)
        ):
            self.get_logger().info('service not available, waiting again...')

        self.get_logger().info('All action and service servers are ready.')

    # ---------------------- Utilities ----------------------

    def _call_service_and_wait(self, client: Client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        return response

    # ------------------------ TAMP -------------------------
    def set_tamp_cfg(self, desired_tool: str):

        request = SetTampCfg.Request()

        assert desired_tool in {"empty", "ag95", "vgc10", "dh3"}

        request.curobo_plan = True
        request.enable_visualizer = False
        request.viz_robot_mesh = False
        request.enable_experiment_logging = False
        request.rr_spawn = False
        request.time_dilation_factor = 0.5

        robot_name = "fr5"

        if desired_tool == "empty":
            request.robot = robot_name
        else:
            request.robot = robot_name + "_" + desired_tool
            if desired_tool == "ag95":
                request.grasp_dof = 6
            elif desired_tool == "vgc10":
                request.grasp_dof = 4
            elif desired_tool == "dh3":
                request.grasp_dof = 4

        response = self._call_service_and_wait(self.set_tamp_cfg_client, request)
        if response:
            self.get_logger().info(f"Service call successful, result: {response.success}")
        else:
            self.get_logger().warn("Service call failed")


    def set_tamp_env(self, arg: str, step_attrs: dict = None, rearrange_info: Dict[str, str] = None):

        request = SetTampEnv.Request()
        tag_name = arg.strip().lower()

        all_entities = ["table", "stirrer", "box_goal", "beaker", "flask", "magnet",  "box"]
        request.entities = all_entities

        if tag_name == "transfer":
            from_vessel = step_attrs.get("from_vessel")
            to_vessel = step_attrs.get("to_vessel")
            request.env_name = "transfer"
            request.movables = [from_vessel, to_vessel]
            request.statics = ["table", "goal_region", "stirrer", "magnet", "box"]
            request.ex_collision = ["pour_region", "rearrange_region"]

        elif tag_name == "stir":
            vessel = step_attrs.get("vessel")
            not_vessel = "beaker" if vessel == "flask" else "flask"
            request.env_name = "stir"
            request.movables = [vessel, "magnet"]
            request.statics = ["table", "stirrer", not_vessel, "goal_region", "box"]
            request.ex_collision = ["beaker_region", "rearrange_region"]

        elif tag_name == "default":
            request.env_name = "default"
            request.movables = ["magnet"]
            request.statics = ["table", "stirrer", "beaker", "flask", "box"]
            request.ex_collision = []

        elif tag_name == "move":
            object = step_attrs.get("object")
            request.env_name = "move"
            request.movables = [object]
            base_statics = ["table", "stirrer", "box", "beaker", "flask", "box_goal"]
            request.statics = [obj for obj in base_statics if obj != object]
            request.ex_collision = ["box_region", "rearrange_region"]

        elif tag_name == "rearrange":
            target_object = rearrange_info["target_entity"]
            request.env_name = "rearrange"
            request.movables = [target_object]
            request.rearrange_grid = rearrange_info["target_grid"]
            
            # [수정된 부분] 타겟 물체가 무엇이든 statics 목록에서 자동으로 제외합니다.
            base_statics = ["table", "stirrer", "beaker", "flask", "box_goal"]
            request.statics = [obj for obj in base_statics if obj != target_object]
            
            request.ex_collision = ["pour_region", "beaker_region", "box_region", "rearrange_region"]

        else:
            raise ValueError("arg must be 'transfer' or 'stir' or 'default', 'move'")

        response = self._call_service_and_wait(self.set_tamp_env_client, request)
        if response:
            self.get_logger().info(f"Service call successful, result: {response.success}")
        else:
            self.get_logger().warn("Service call failed")

    def plan(self, arg):
        env_name = arg.strip() if arg else "transfer" # Default to "transfer" if no arg
        if not env_name:
            self.get_logger().info("Usage: plan <env_name>")
            return
        request = Plan.Request()
        request.env_name = env_name
        response = self._call_service_and_wait(self.plan_client, request)

        if response:
            self.get_logger().info(f"Service call successful, result: {response.plan_success}")
        else:
            self.get_logger().warn("Service call failed")
    
    def execute(self, arg):

        request = Execute.Request()

        response = self._call_service_and_wait(self.execute_client, request)

        if response:
            self.get_logger().info(f"Service call successful, result: {response.execute_success}")
        else:
            self.get_logger().warn("Service call failed")

    # --------------------- ToolChange ----------------------
    def home(self):

        home_pos = [0.0, -1.05, -2.18, -1.57, 1.57, 0.0]
        self.move_to_target_js(home_pos)
    
    def move_to_target(self, target_position, target_orientation):

        request = MoveToTarget.Request()
        request.target_position = target_position
        request.target_orientation = target_orientation

        self.set_tamp_env(arg="default") # For Update Env

        response = self._call_service_and_wait(self.move_to_target_client, request)
        time.sleep(3.0)
        if response:
            self.get_logger().info(f"Service call successful, result: {response.success}")
        else:
            self.get_logger().warn("Service call failed")
        

    def move_to_target_js(self, target_js):

        request = MoveToTargetJs.Request()
        request.q_des = target_js

        self.set_tamp_env(arg="default") # For Update Env

        response = self._call_service_and_wait(self.move_to_target_js_client, request)
        time.sleep(3.0)
        if response:
            self.get_logger().info(f"Service call successful, result: {response.success}")
        else:
            self.get_logger().warn("Service call failed")

    def tool_change(self, arg):

        # get robot info
        get_robot_info_request = GetRobotInfo.Request()
        get_robot_info_response = self._call_service_and_wait(self.get_robot_info_client, get_robot_info_request)
        current_tool = get_robot_info_response.current_tool
        desired_tool = arg.strip().lower()

        assert current_tool in ["empty", "ag95", "vgc10", "dh3"], f"Error: Tool '{arg.strip()}' is not supported."
        assert desired_tool in ["ag95", "vgc10", "dh3"], f"Error: Tool '{arg.strip()}' is not supported."
        
        # get tool info
        get_tool_info_request = GetToolInfo.Request()
        get_tool_info_request.current_tool = current_tool
        get_tool_info_request.desired_tool = desired_tool
        get_tool_info_response = self._call_service_and_wait(self.get_tool_info_client, get_tool_info_request)
        current_tool_position = get_tool_info_response.current_tool_position
        current_tool_orientation = get_tool_info_response.current_tool_orientation
        desired_tool_position = get_tool_info_response.desired_tool_position
        desired_tool_orientation = get_tool_info_response.desired_tool_orientation

        if current_tool == "empty":
            # directly move to desired tool pose
            self.move_to_target(desired_tool_position, desired_tool_orientation)

            # grip tool
            tool_change_request = ToolChange.Request()
            tool_change_request.desired_tool = desired_tool

            tool_change_response = self._call_service_and_wait(self.tool_change_client, tool_change_request)
            self.set_tamp_cfg(desired_tool) # Change Robot Cfg

            self.home()
        else:
            # move to home qpos -> move to current tool pose
            # move to home position
            self.home()
            self.set_tamp_cfg("empty") # Change Robot Cfg
            
            # move to current tool pose
            self.move_to_target(current_tool_position, current_tool_orientation)

            # release tool
            tool_change_request = ToolChange.Request()
            tool_change_request.desired_tool = "empty"

            tool_change_response = self._call_service_and_wait(self.tool_change_client, tool_change_request)
            self.set_tamp_cfg("empty") # Change Robot Cfg

            # move to desired_tool pose
            self.move_to_target(desired_tool_position, desired_tool_orientation)

            # grip tool
            tool_change_request = ToolChange.Request()
            tool_change_request.desired_tool = desired_tool

            tool_change_response = self._call_service_and_wait(self.tool_change_client, tool_change_request)
            self.set_tamp_cfg(desired_tool) # Change Robot Cfg

            time.sleep(5.0)

            self.home()


        if tool_change_response:
            self.get_logger().info(f"Service call successful, result: {tool_change_response.success}")
        else:
            self.get_logger().warn("Service call failed")

    # ---------------------- Shutdown -----------------------

    def shutdown(self):
        self.get_logger().info("Shutting down ROS 2 …")
        self.destroy_node()
        rclpy.shutdown()
        

class XDLRunner(TAMPClient):
    def __init__(self):
        super().__init__()

        self.action_reasoner = ActionReasoner()

    def run_xdl(self):
        # XDL 경로
        xdl_path = os.path.join(PROJECT_PATH, "TAMP", "tamp", "content", "configs", "xdl", "xdl.xml")
        ns = {"xdl": "http://www.xdl.org/schema/xdl"}

        # XDL XML 파싱
        tree = ET.parse(xdl_path)
        root = tree.getroot()
        procedure = root.find("xdl:procedure", ns)

        if procedure is None:
            print("<procedure> 태그를 찾을 수 없습니다.")
            return

        print("=== Procedure Steps ===")

        # 현재 툴 정보 확인
        get_robot_info_request = GetRobotInfo.Request()
        get_robot_info_response = self._call_service_and_wait(
            self.get_robot_info_client, get_robot_info_request
        )
        current_tool = get_robot_info_response.current_tool
        self.get_logger().info(f"현재 장착된 툴: {current_tool}")

        # 단계별 실행
        procedure = list(procedure)
        for i in range(len(procedure)):

            current_step = procedure[i]
            tag_name = current_step.tag.split("}")[-1]  # ex) "Transfer", "Stir"
            current_step_attrs = current_step.attrib
            self.get_logger().info(f"[Step {i+1}] Tag: {tag_name} | Attrs: {current_step_attrs}")

            llm_attrs = {
                "current": {},
                "next": {},
            }

            for k, v in current_step_attrs.items():
                llm_attrs["current"][k] = v

            current_attr_str = " ".join([f'{k}="{v}"' for k, v in llm_attrs["current"].items()])
            current_xdl = f'<{tag_name} {current_attr_str} />'
            obstacle_info, candidate_grids = self.get_obstacle_info_and_candidate_grids(llm_attrs["current"])

            if i == len(procedure) - 1:
                next_xdl = "None"
            else:
                next_step = procedure[i + 1]
                next_step_attrs = next_step.attrib
                for k, v in next_step_attrs.items():
                    llm_attrs["next"][k] = v
                next_attr_str = " ".join([f'{k}="{v}"' for k, v in llm_attrs["next"].items()])
                next_xdl = f'<{tag_name} {next_attr_str} />'

            llm_attrs = {
                "current": {},
                "next": {},
            }

            sdl_step = {
                "instruction": {"current_xdl": current_xdl, "next_xdl": next_xdl, "obstacle_info": obstacle_info, "candidate_grids": candidate_grids}
            }
            self.get_logger().info(f"sdl_step: {sdl_step}")

            self.get_logger().info("🧠 LLM에게 다음 행동을 질문합니다...")

            # predict
            main_tool, is_rearrange, rearrange_tool, rearrange_grid = self.action_reasoner.predict(xdl_step=sdl_step)

            self.get_logger().info(f"💡 LLM 판단 결과 => Main Tool: {main_tool} | Need Move: {is_rearrange} | Move Tool: {rearrange_tool}")
            # =======================================================================
            
            # 🚨 [NEW] 1. 공간이 좁아 치우기(rearrange)가 필요한 경우
            if is_rearrange == "True":
                self.get_logger().info("⚠️ 공간이 협소하여 방해물을 먼저 치웁니다 (rearrange 실행)")
                
                # 치우기용 툴 설정 및 교체
                move_tool = rearrange_tool if rearrange_tool in {"ag95", "vgc10", "dh3"} else "empty"
                if move_tool != current_tool:
                    self.get_logger().info(f"치우기용 Tool 변경 필요: {current_tool} → {move_tool}")
                    self.tool_change(move_tool)
                    current_tool = move_tool
                else:
                    self.get_logger().info("치우기용 Tool 변경 없음")
                
                # rearrange 환경 세팅 및 실행
                if obstacle_info and obstacle_info != "None":
                    # "at"이라는 단어를 기준으로 앞부분만 자릅니다.
                    target_entity = obstacle_info.split(" at ")[0].strip()
                else:
                    target_entity = None
                
                rearrange_info = {
                    "target_entity": target_entity,
                    "target_grid": rearrange_grid,
                }
                self.set_tamp_env("rearrange", current_step_attrs, rearrange_info)
                time.sleep(2.0)
                
                self.get_logger().info("Planning for rearrange...")
                self.plan("rearrange")
                
                self.get_logger().info("Executing rearrange...")
                self.execute("rearrange")
                
                self.get_logger().info("✅ 방해물 치우기(rearrange) 완료")
                time.sleep(2.0)

            # 🛠️ 2. 본 작업 실행 (원래 XML 태그)
            main_tool = main_tool if main_tool in {"ag95", "vgc10", "dh3"} else None
            if not main_tool:
                self.get_logger().warn(f"[{tag_name}]에 해당하는 tool이 tool_map.yml에 없습니다. 기본값 'empty' 사용.")
                main_tool = "empty"

            # 본 작업용 툴로 변경 (치우느라 바뀌었다면 여기서 다시 돌아옵니다)
            if main_tool != current_tool:
                self.get_logger().info(f"본 작업용 Tool 변경 필요: {current_tool} → {main_tool}")
                self.tool_change(main_tool)
                current_tool = main_tool
            else:
                self.get_logger().info("본 작업용 Tool 변경 없음")

            # 환경 설정 (본 작업)
            env_name = tag_name.lower()
            self.get_logger().info(f"환경 설정: {env_name}")
            self.set_tamp_env(env_name, current_step_attrs)
            time.sleep(2.0)

            # 계획 생성 및 실행
            self.get_logger().info(f"Planning for {env_name}...")
            self.plan(env_name)

            self.get_logger().info(f"Executing {env_name}...")
            self.execute(env_name)

            self.get_logger().info(f"[Step {i+1}] {tag_name} 완료 ✅")
            print("-" * 40)
            time.sleep(1.0)

        self.get_logger().info("=== 모든 procedure 단계 완료 ✅ ===")
    
    # =======================================================================
    # 물체 위치 조회 및 5cm 이내 충돌/접근 감지 함수 (2차원 평면 기준)
    # =======================================================================
    # GT Based
    def get_obstacle_info_and_candidate_grids(self, step_attrs):
        self.get_logger().info("🔍 [Entity & Grid Obstacle Check]")

        hardcoded_entities = ["beaker", "flask", "magnet", "stirrer", "box", "box_goal"]
        obstacle_reports = []
        
        # 1. 타겟 엔티티 파악
        target_entities = list(set(
            entity_name for key, entity_name in step_attrs.items()
            if any(k in key for k in ["vessel", "object", "place"])
        ))

        # 2. 위치 조회
        all_entities_to_query = list(set(hardcoded_entities + target_entities))
        positions = {}

        for entity_name in all_entities_to_query:
            req = GetEntityState.Request()
            req.entity = "/World/" + entity_name
            res = self._call_service_and_wait(self.get_entity_state_client, req)
            
            if res and res.result.result == 1:
                positions[entity_name] = res.state.pose.position
            else:
                self.get_logger().warn(f"  - {entity_name:10s} : 위치 조회 실패")

        # 3. 거리 검사 및 그리드 기반 문자열 생성
        THRESHOLD = 0.2 # 20cm

        for target in target_entities:
            if target not in positions: continue
            p1 = positions[target]
            
            for other in hardcoded_entities:
                if target == other or other not in positions: continue
                p2 = positions[other]
                
                # 2차원 거리 계산
                dist_2d = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                
                if dist_2d < THRESHOLD:
                    # 좌표를 기반으로 G1, G2 등 구역 이름을 가져옵니다.
                    grid_name = self.get_grid_name(p2.x, p2.y)
                    
                    # 요청하신 포맷: {entity_name} at {candidate_grids} (blocking)
                    report = f"{other} at {grid_name} (blocking)"
                    obstacle_reports.append(report)
                    
                    self.get_logger().warn(f"  ⚠️ [Obstacle] {report}")

        if len(obstacle_reports):
            obstacle_info = obstacle_reports[-1]
        else:
            obstacle_info = "None"

        candidate_grids = self.get_candidate_grids(positions)
        
        return obstacle_info, candidate_grids
    
    def get_candidate_grids(self, all_positions):
        """
        현재 모든 엔티티의 위치(all_positions)를 파악하여 
        비어있는 구역을 속성별로 분류해 string으로 반환합니다.
        """
        all_grids = [f"G{i}" for i in range(1, 13)]
        occupied_grids = set()
        
        # 1. 현재 점유된 구역 파악
        for entity, pos in all_positions.items():
            grid = self.get_grid_name(pos.x, pos.y)
            occupied_grids.add(grid)
        
        # 2. 비어있는 구역 파악
        empty_grids = [g for g in all_grids if g not in occupied_grids]
        
        # 3. 속성별 분류
        categorized = {
            "plate zone": [],
            "stirrer zone": [],
            "workspace edge": [],
            "open area": []
        }
        
        for g in empty_grids:
            prop = GRID_PROPERTIES.get(g, "open area")
            # 이미지 레이아웃에 따라 G3~G6, G9~G11 등을 open area로 간주
            categorized[prop].append(g)
        
        # 4. 데이터셋 포맷으로 문자열 조립
        lines = []
        for category, grids in categorized.items():
            if grids:
                lines.append(f"{category}: [{', '.join(grids)}]")
                
        return "\n".join(lines)
    
    # # QR Based
    # def check_entity_distances(self, step_attrs):
    #     self.get_logger().info("🔍 [Entity Position & 2D Distance Check via TF]")

    #     is_space_constrained = False

    #     hardcoded_entities = ["beaker", "flask", "magnet", "stirrer", "box", "box_goal"]
        
    #     target_entities = []
    #     for key, entity_name in step_attrs.items():
    #         if "vessel" in key or "object" in key or "place" in key:
    #             target_entities.append(entity_name)

    #     all_entities_to_query = list(set(hardcoded_entities + target_entities))
    #     positions = {}

    #     # 4. TF를 통해 각 엔티티의 현재 좌표 조회
    #     for entity_name in all_entities_to_query:
    #         try:
    #             # base_link 프레임을 기준으로 각 엔티티의 위치를 가져옵니다. (최대 0.5초 대기)
    #             t = self.tf_buffer.lookup_transform(
    #                 'base_link', 
    #                 entity_name, 
    #                 rclpy.time.Time(),
    #                 timeout=rclpy.duration.Duration(seconds=0.5)
    #             )
    #             pos = t.transform.translation
    #             positions[entity_name] = pos
    #             self.get_logger().info(
    #                 f"  - {entity_name:10s} : x={pos.x:5.3f}, y={pos.y:5.3f}, z={pos.z:5.3f}"
    #             )
    #         except (LookupException, ConnectivityException, ExtrapolationException) as ex:
    #             self.get_logger().warn(f"  - {entity_name:10s} : TF 위치 조회 실패 ({ex})")

    #     # 5. 타겟 엔티티와 나머지 엔티티 사이의 2차원 거리 계산 (5cm 미만 감지)
    #     for target in target_entities:
    #         if target not in positions:
    #             continue
            
    #         p1 = positions[target]
            
    #         for other in hardcoded_entities:
    #             if target == other or other not in positions:
    #                 continue
                
    #             p2 = positions[other]
                
    #             # 2차원(x, y 평면) 거리 계산
    #             dist_2d = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                
    #             if dist_2d < 0.2:  # 0.2m (20cm)
    #                 self.get_logger().warn(
    #                     f"  ⚠️ [공간 협소 감지] '{target}' 주변 20cm 이내(평면 기준)에 '{other}'(이)가 있습니다! "
    #                     f"(2D 거리: {dist_2d*100:.1f}cm)"
    #                 )
    #                 is_space_constrained = True
        
    #     self.get_logger().info("-" * 40)

    #     return is_space_constrained

    def get_grid_name(self, x, y):
        if -0.3 <= x < 0:
            if -0.75 <= y < -0.45: 
                return "G1"
            if -0.2 <= x < -0.1 and 0.55 <= y < 0.65: # Stirrer Grid (G2) is smaller than others
                return "G2"

        elif 0.0 <= x < 0.3:
            if   -0.75 <= y < -0.45: return "G3"
            elif -0.45 <= y < -0.15: return "G4"
            elif -0.15 <= y <  0.15: return "G5"
            elif  0.15 <= y <  0.45: return "G6"
            elif  0.45 <= y <= 0.75: return "G7"

        elif 0.3 <= x < 0.6:
            if   -0.75 <= y < -0.45: return "G8"
            elif -0.45 <= y < -0.15: return "G9"
            elif -0.15 <= y <  0.15: return "G10"
            elif  0.15 <= y <  0.45: return "G11"
            elif  0.45 <= y <= 0.75: return "G12"
                
        return "Unknown"

    

def main():

    runner = XDLRunner()
    runner.run_xdl()


if __name__ == "__main__":
    main()