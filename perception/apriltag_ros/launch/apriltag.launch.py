import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

def generate_apriltag_nodes(context: LaunchContext, *args, **kwargs):
    # 1. 전달받은 문자열 인자를 실행 시점에 가져옴 (예: "camera_1,camera_2")
    camera_names_str = LaunchConfiguration('camera_names').perform(context)
    
    # 2. 쉼표(,)를 기준으로 분리하여 리스트로 변환
    cameras = [name.strip() for name in camera_names_str.split(',') if name.strip()]

    apriltag_params = PathJoinSubstitution([
        get_package_share_directory("apriltag_ros"),
        "cfg",
        "tags_36h11.yaml",
    ])

    node_list = []
    # 3. 분리된 카메라 리스트를 돌면서 노드 생성
    for name in cameras:
        node = Node(
            package="apriltag_ros",
            executable="apriltag_node",
            name="apriltag_node",
            namespace=name,
            # 다중 카메라 환경에서 TF 충돌 방지
            parameters=[apriltag_params],
            remappings=[
                ("image_rect", f"/{name}/rgb"),
                ("camera_info", f"/{name}/camera_info"),
                ("detections", f"/{name}/apriltag/detections"),
            ],
            output="log",
        )
        node_list.append(node)
        
    return node_list

def generate_launch_description():
    # 외부에서 입력받을 인자(Argument) 선언
    camera_names_arg = DeclareLaunchArgument(
        'camera_names',
        default_value='camera_1,camera_2,camera_3,camera_4',
        description='Comma-separated list of camera namespaces'
    )

    return LaunchDescription([
        camera_names_arg,
        # OpaqueFunction을 통해 파이썬 로직(generate_apriltag_nodes)을 실행 시점에 평가
        OpaqueFunction(function=generate_apriltag_nodes)
    ])