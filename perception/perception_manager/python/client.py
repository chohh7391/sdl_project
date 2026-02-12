import cmd
import sys
import numpy as np
import threading  # threading 모듈 추가

import rclpy
from rclpy.action import ActionClient
from rclpy.client import Client
from action_msgs.msg import GoalStatus

from perception_interfaces.srv import GetObjectInfo
from perception_interfaces.srv import GetFtData

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from std_msgs.msg import Float32MultiArray, Bool
# --------------------------------------------
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ControlSuiteShell(cmd.Cmd):
    intro = (
        bcolors.OKBLUE
        + "Welcome to the control suite shell.\nType help or ? to list commands.\n"
        + bcolors.ENDC
    )
    prompt = "(csuite) "

    def __init__(self):
        super().__init__()
        rclpy.init(args=None)
        self.node = rclpy.create_node(
            "actions_client_simulation_isaac",
            parameter_overrides=[
            rclpy.Parameter(
                "use_sim_time",
                rclpy.Parameter.Type.BOOL,
                True
            )]
        )

        self.get_pose_client = self.node.create_client(GetObjectInfo, "/perception_manager/get_object_info")

        self.get_ft_data_client = self.node.create_client(GetFtData, "/perception_manager/get_ft_data")


        while not self.get_pose_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('GetObjectInfo service not available, waiting again...')
        while not self.get_ft_data_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('GetFtData service not available, waiting again...')

        self.node.get_logger().info('All action and service servers are ready.')

        # 노드를 백그라운드 스레드에서 스핀하도록 설정
        self.spinner = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spinner.start()
        

    def do_get_object_info(self, arg):
        """Get the pose of a specific object by ID (e.g., get_pose 1)"""
        try:
            object_id = int(arg.strip())
            
            request = GetObjectInfo.Request()
            request.object_id = object_id

            self.node.get_logger().info(f"Requesting pose for object ID: {object_id}")
            response = self._call_service_and_wait(self.get_pose_client, request)

            if response and response.success:
                pose = response.object_pose.pose
                header = response.object_pose.header
                print(f"{bcolors.OKGREEN}Service call successful.{bcolors.ENDC}")
                print(f"  Frame ID: {header.frame_id}")
                print(f"  Position: x={pose.position.x:.4f}, y={pose.position.y:.4f}, z={pose.position.z:.4f}")
                print(f"  Orientation: x={pose.orientation.x:.4f}, y={pose.orientation.y:.4f}, z={pose.orientation.z:.4f}, w={pose.orientation.w:.4f}")
                print(f"  Grasping offset: x={response.grasping_offset.position.x:.4f}, y={response.grasping_offset.position.y:.4f}, z={response.grasping_offset.position.z:.4f}")
                print(f"  Grasping orientation: x={response.grasping_offset.orientation.x:.4f}, y={response.grasping_offset.orientation.y:.4f}, z={response.grasping_offset.orientation.z:.4f}, w={response.grasping_offset.orientation.w:.4f}")
            elif response:
                print(f"{bcolors.WARNING}Service call failed. Object with ID {object_id} not found.{bcolors.ENDC}")
            else:
                print(f"{bcolors.FAIL}Service call failed or timed out.{bcolors.ENDC}")

        except (ValueError, IndexError):
            print(f"{bcolors.FAIL}Invalid argument. Please provide a valid integer object ID.{bcolors.ENDC}")

    def do_get_ft(self, arg):
        """Get the latest Force/Torque sensor data"""
        request = GetFtData.Request()
        
        self.node.get_logger().info("Requesting latest F/T sensor data...")
        response = self._call_service_and_wait(self.get_ft_data_client, request)

        if response and response.success:
            ft_data = response.ft_data
            print(f"{bcolors.OKGREEN}Service call successful.{bcolors.ENDC}")
            print(f"  Force: x={ft_data.force.x:.4f}, y={ft_data.force.y:.4f}, z={ft_data.force.z:.4f}")
            print(f"  Torque: x={ft_data.torque.x:.4f}, y={ft_data.torque.y:.4f}, z={ft_data.torque.z:.4f}")
        else:
            print(f"{bcolors.FAIL}Service call failed or timed out.{bcolors.ENDC}")


    def do_quit(self, arg):
        """Quit shell"""
        print("Shutting down ROS 2 …")
        self.node.destroy_node()
        rclpy.shutdown()
        return True

    # 편의상 EOF (^D) 도 quit 으로 연결
    def do_EOF(self, arg):
        return self.do_quit(arg)
    
    ##################################
    

    # Helper Functions
    def _send_goal_and_wait(self, client: ActionClient, goal_msg) -> bool:
        send_goal_future = client.send_goal_async(goal_msg)
        
        # 콜백이 백그라운드 스레드에서 처리되므로,
        # future가 완료될 때까지 기다리기만 하면 됩니다.
        while rclpy.ok() and not send_goal_future.done():
            # 짧은 시간 동안 슬립하여 CPU 사용을 줄입니다.
            time.sleep(0.1)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            print("Goal rejected")
            return False

        get_result_future = goal_handle.get_result_async()
        
        # 마찬가지로 future가 완료될 때까지 기다립니다.
        while rclpy.ok() and not get_result_future.done():
            time.sleep(0.1)

        result_status = get_result_future.result().status

        return result_status == GoalStatus.STATUS_SUCCEEDED
    
    
    def _call_service_and_wait(self, client: Client, request):
        future = client.call_async(request)

        # rclpy.spin_until_future_complete 대신 간단한 루프를 사용합니다.
        while rclpy.ok() and not future.done():
            time.sleep(0.1) # CPU 사용을 줄이기 위해 잠시 대기합니다.

        if future.done():
            response = future.result()
            if response is not None:
                return response
            else:
                print(f"Exception while calling service: {future.exception()}")
                return None
        
        # rclpy가 종료되어 future가 완료되지 않은 경우를 처리합니다.
        print("Service call was interrupted before completion.")
        return None


if __name__ == "__main__":
    try:
        ControlSuiteShell().cmdloop()
    except KeyboardInterrupt:
        print("\nInterrupt - shutting down …")
        rclpy.shutdown()
        sys.exit(0)