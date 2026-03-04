import rclpy
from rclpy.node import Node
import numpy as np
from isaacsim import SimulationApp

from sensor_msgs.msg import JointState, CameraInfo, Image
from std_srvs.srv import SetBool
from std_msgs.msg import Float32
from tamp_interfaces.srv import ToolChange, GetRobotInfo, GetToolInfo
import time
import sys, os

ROBOT_STAGE_PATH = "/World/Robot"
ROOT_JOINT_PATH = ROBOT_STAGE_PATH + "/root_joint"
BACKGROUND_STAGE_PATH = "/background"
BACKGROUND_USD_PATH = "/Isaac/Environments/Grid/default_environment.usd"

CONFIG = {"renderer": "RaytracedLighting", "headless": False}


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
        viewports.set_camera_view(eye=np.array([1.2, 1.2, 0.8]), target=np.array([0, 0, 0.5]))

        from task_camera import Task

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

        self.scale_pub = self.create_publisher(Float32, "scale", 10)
        self.scale_start_angle = None
        self.scale_value = 0.0
        self.scale_gain = 50.0

        self.step = 0

        self.get_logger().info("Simulation Start")

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

                joint_positions = self.robot.get_joint_positions()
                wrist_angle = joint_positions[self.arm_joint_ids[5]]

                self.og.Controller.set(self.og.Controller.attribute("/ActionGraph/RobotControl/OnImpulseEvent.state:enableImpulse"), True)

                self.compute_and_publish_scale(wrist_angle)

                # obs = self.task.get_observations()
                # info = obs["camera_1"]["info"]

                # # ROS2 CameraInfo 메시지 생성 및 필드 채우기
                # msg = CameraInfo()
                # msg.header.stamp = self.node.get_clock().now().to_msg() # 시뮬레이션 타임 혹은 시스템 타임
                # msg.header.frame_id = "camera_1"
                
                # msg.height = int(info.height)
                # msg.width = int(info.width)
                # msg.distortion_model = info.distortion_model
                
                # # D, K, R, P 행렬 매핑 (numpy array를 list로 변환하여 입력)
                # msg.d = info.d.tolist() if hasattr(info.d, "tolist") else list(info.d)
                # msg.k = info.k.flatten().tolist() if hasattr(info.k, "flatten") else list(info.k)
                # msg.r = info.r.flatten().tolist() if hasattr(info.r, "flatten") else list(info.r)
                # msg.p = info.p.flatten().tolist() if hasattr(info.p, "flatten") else list(info.p)
                
                # # binning 및 ROI 초기화 (기본값)
                # msg.binning_x = 0
                # msg.binning_y = 0
                # msg.roi.x_offset = 0
                # msg.roi.y_offset = 0
                # msg.roi.height = 0
                # msg.roi.width = 0
                # msg.roi.do_rectify = False

                # # 정의하신 퍼블리셔를 통해 메시지 송출
                # self.camera_info_pub.publish(msg)


            self.step += 1

        else:
            self.get_logger().info("Quit ROS2 Node")
            rclpy.try_shutdown()

    def compute_and_publish_scale(self, wrist_angle: float):

        try:
            observations = self.world.get_observations()
            beaker_start_pos = self.task.default_positions["beaker"]
            beaker_pos = observations["current_positions"]["beaker"]
            beaker_moved_distance = np.linalg.norm(np.array(beaker_pos) - np.array(beaker_start_pos))

        except Exception as e:
            self.get_logger().warn(f"Failed to compute scale: {e}")
            return

        if beaker_moved_distance <= 0.022:
            self.scale_value = 0.0
            self.max_pour_angle = None
            return

        if self.max_pour_angle is None:
            self.max_pour_angle = wrist_angle
            return

        if wrist_angle > self.max_pour_angle:
            delta_angle = wrist_angle - self.max_pour_angle
            flow = delta_angle * self.scale_gain
            self.scale_value += max(0.0, flow)
            self.max_pour_angle = wrist_angle
        
        msg = Float32()
        msg.data = float(self.scale_value)
        self.scale_pub.publish(msg)

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
        
        if is_close:
            if gripper == "dh3":
                for _ in range(16):
                    self.robot.gripper.close()
                    self.world.step(render=True)
            else:
                self.robot.gripper.close()
            response.message = "close gripper"
        else:
            if gripper == "dh3":
                for _ in range(3):
                    self.robot.gripper.open()
                    self.world.step(render=True)
            else:
                self.robot.gripper.open()
            response.message = "open gripper"

        response.success = True
        
        return response
    
    def tool_change_cb(self, request, response):

        desired_tool = request.desired_tool
        
        try:
            observations = self.world.get_observations()
            self.task.current_positions = observations["current_positions"]
            self.task.current_orientations = observations["current_orientations"]

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

            self.robot.set_joint_positions(
                positions=self._saved_robot_joint_positions,
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