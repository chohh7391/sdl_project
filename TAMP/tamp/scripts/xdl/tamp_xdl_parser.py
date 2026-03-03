#!/home/home/miniconda3/envs/sdl/bin/python
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

PROJECT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src", "sdl_project")

sys.path.append(os.path.join(PROJECT_PATH, "LLM"))
# Llama import
from llama.script.tool_selector_action_reasoner.tool_llm import ToolLLM


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


    def set_tamp_env(self, arg: str, step_attrs: dict = None):

        request = SetTampEnv.Request()
        tag_name = arg.strip().lower()

        if tag_name == "transfer":
            from_vessel = step_attrs.get("from_vessel")
            to_vessel = step_attrs.get("to_vessel")
            request.env_name = "transfer"
            request.entities = [from_vessel, to_vessel, "magnet", "stirrer", "box"]
            request.movables = [from_vessel, to_vessel]
            request.statics = ["table", "goal_region", "stirrer", "magnet", "box"]
            request.ex_collision = ["pour_region", "rearrange_region"]

        elif tag_name == "stir":
            vessel = step_attrs.get("vessel")
            not_vessel = "beaker" if vessel == "flask" else "flask"
            
            request.env_name = "stir"
            request.entities = [vessel, not_vessel, "magnet", "stirrer", "box"]
            request.movables = [vessel, "magnet"]
            request.statics = ["table", "stirrer", not_vessel, "goal_region", "box"]
            request.ex_collision = ["beaker_region", "rearrange_region"]

        elif tag_name == "default":
            request.env_name = "default"
            request.entities = ["beaker", "flask", "magnet", "stirrer", "box"]
            request.movables = ["magnet"]
            request.statics = ["table", "stirrer", "beaker", "flask", "box"]
            request.ex_collision = []

        elif tag_name == "move":
            object = step_attrs.get("object")
            request.env_name = "move"
            request.entities = [object, "box_goal"]
            request.movables = [object]
            request.statics = ["table", "stirrer", "beaker", "flask", "box_goal"]
            request.ex_collision = ["box_region", "rearrange_region"]

        elif tag_name == "rearrange":
            target_object = step_attrs.get("object") or step_attrs.get("to_vessel") or step_attrs.get("vessel")
            request.env_name = "rearrange"
            request.entities = [target_object]
            request.movables = [target_object]
            
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

        self.llm = ToolLLM()
    
    # =======================================================================
    # 물체 위치 조회 및 5cm 이내 충돌/접근 감지 함수 (2차원 평면 기준)
    # =======================================================================
    def check_entity_distances(self, step_attrs):
        self.get_logger().info("🔍 [Entity Position & 2D Distance Check]")

        is_space_constrained = False

        # 1. 고정 엔티티 리스트
        hardcoded_entities = ["beaker", "flask", "magnet", "stirrer", "box", "box_goal"]
        
        # 2. 이번 step과 관련된 타겟 엔티티 파악
        target_entities = []
        for key, entity_name in step_attrs.items():
            if "vessel" in key or "object" in key or "place" in key:
                target_entities.append(entity_name)

        # 3. 위치를 조회할 전체 엔티티 (중복 제거)
        all_entities_to_query = list(set(hardcoded_entities + target_entities))
        positions = {}

        # 4. 각 엔티티의 현재 좌표 조회
        for entity_name in all_entities_to_query:
            req = GetEntityState.Request()
            req.entity = "/World/" + entity_name
            res = self._call_service_and_wait(self.get_entity_state_client, req)
            
            if res and res.result.result == 1:
                pos = res.state.pose.position
                positions[entity_name] = pos
                self.get_logger().info(
                    f"  - {entity_name:10s} : x={pos.x:5.3f}, y={pos.y:5.3f}, z={pos.z:5.3f}"
                )
            else:
                self.get_logger().warn(f"  - {entity_name:10s} : 위치 조회 실패")

        # 5. 타겟 엔티티와 나머지 엔티티 사이의 2차원 거리 계산 (5cm 미만 감지)
        for target in target_entities:
            if target not in positions:
                continue
            
            p1 = positions[target]
            
            # 고정 엔티티들을 순회하며 거리 검사
            for other in hardcoded_entities:
                # 자기 자신과의 비교는 제외
                if target == other or other not in positions:
                    continue
                
                p2 = positions[other]
                
                # ★ 3차원이 아닌 2차원(x, y 평면) 거리만 계산 ★
                dist_2d = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                
                # 5cm (0.05m) 미만일 경우 출력
                if dist_2d < 0.2:
                    self.get_logger().warn(
                        f"  ⚠️ [공간 협소 감지] '{target}' 주변 5cm 이내(평면 기준)에 '{other}'(이)가 있습니다! "
                        f"(2D 거리: {dist_2d*100:.1f}cm)"
                    )
                    is_space_constrained = True
        
        self.get_logger().info("-" * 40)

        return is_space_constrained

    def run_xdl(self):
        # XDL 경로
        xdl_path = os.path.join(PROJECT_PATH, "TAMP", "tamp", "content", "configs", "xdl", "xdl.xml")
        tool_map_path = os.path.join(PROJECT_PATH, "TAMP", "tamp", "content", "configs", "xdl", "tool_map.yml")
        ns = {"xdl": "http://www.xdl.org/schema/xdl"}

        # tool_map.yml 불러오기
        if not os.path.exists(tool_map_path):
            raise FileNotFoundError(f"tool_map.yml not found at {tool_map_path}")

        with open(tool_map_path, "r") as f:
            tool_map_data = yaml.safe_load(f)
            tool_map = tool_map_data.get("tool_map", {})

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
            self.get_robot_info_client, get_robot_info_request)
        current_tool = get_robot_info_response.current_tool
        self.get_logger().info(f"현재 장착된 툴: {current_tool}")

        # 단계별 실행
        for i, step in enumerate(list(procedure), start=1):
            tag_name = step.tag.split("}")[-1]  # ex) "Transfer", "Stir"
            step_attrs = step.attrib
            self.get_logger().info(f"[Step {i}] Tag: {tag_name} | Attrs: {step_attrs}")

            # =======================================================================
            llm_attrs = {}
            target_counts = {"beaker": 0, "flask": 0, "box": 0}
            
            for k, v in step_attrs.items():
                if v in target_counts:
                    # 첫 번째 등장은 _A, 두 번째 등장은 _B 부여
                    suffix = "_A" if target_counts[v] == 0 else "_B"
                    llm_attrs[k] = f"{v}{suffix}"
                    target_counts[v] += 1
                else:
                    llm_attrs[k] = v

            # LLM 전용 문자열 조합
            attr_str = " ".join([f'{k}="{v}"' for k, v in llm_attrs.items()])
            xdl_step_str = f'<{tag_name} {attr_str} />'
            is_space_constrained = self.check_entity_distances(step_attrs)

            self.get_logger().info("🧠 LLM에게 다음 행동을 질문합니다...")
            llm_main, llm_need_move, llm_move_tool = self.llm.predict(
                xdl_step=xdl_step_str, 
                is_space_constrained=is_space_constrained
            )
            self.get_logger().info(f"💡 LLM 판단 결과 => Main Tool: {llm_main} | Need Move: {llm_need_move} | Move Tool: {llm_move_tool}")
            # =======================================================================
            
            # 🚨 [NEW] 1. 공간이 좁아 치우기(rearrange)가 필요한 경우
            if llm_need_move == "True":
                self.get_logger().info("⚠️ 공간이 협소하여 방해물을 먼저 치웁니다 (rearrange 실행)")
                
                # 치우기용 툴 설정 및 교체
                move_tool = llm_move_tool if llm_move_tool in {"ag95", "vgc10", "dh3"} else "empty"
                if move_tool != current_tool:
                    self.get_logger().info(f"치우기용 Tool 변경 필요: {current_tool} → {move_tool}")
                    self.tool_change(move_tool)
                    current_tool = move_tool
                else:
                    self.get_logger().info("치우기용 Tool 변경 없음")
                
                # rearrange 환경 세팅 및 실행
                self.set_tamp_env("rearrange", step_attrs)
                time.sleep(2.0)
                
                self.get_logger().info("Planning for rearrange...")
                self.plan("rearrange")
                
                self.get_logger().info("Executing rearrange...")
                self.execute("rearrange")

                is_space_constrained = self.check_entity_distances(step_attrs)
                
                self.get_logger().info("✅ 방해물 치우기(rearrange) 완료")
                time.sleep(2.0)

            # 🛠️ 2. 본 작업 실행 (원래 XML 태그)
            main_tool = llm_main if llm_main in {"ag95", "vgc10", "dh3"} else None
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
            self.set_tamp_env(env_name, step_attrs)
            time.sleep(2.0)

            # 계획 생성 및 실행
            self.get_logger().info(f"Planning for {env_name}...")
            self.plan(env_name)

            self.get_logger().info(f"Executing {env_name}...")
            self.execute(env_name)

            self.get_logger().info(f"[Step {i}] {tag_name} 완료 ✅")
            print("-" * 40)
            time.sleep(1.0)

        self.get_logger().info("=== 모든 procedure 단계 완료 ✅ ===")

def main():

    runner = XDLRunner()
    runner.run_xdl()


if __name__ == "__main__":
    main()