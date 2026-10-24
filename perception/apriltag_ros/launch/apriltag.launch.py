import os

import yaml
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

    # 각 검출기는 태그 프레임 이름 앞에 자기 카메라 이름을 붙인다
    # (camera_1_beaker_tag, ...). 예전에는 모두 같은 자식 프레임을 /tf에 올려서,
    # 태그의 부모가 마지막에 발행한 카메라로 바뀌었다. 그러면 perception_manager의
    # 카메라별 조회가 다른 카메라의 검출을 거쳐 풀릴 수 있어, 융합하는 관측이
    # 서로 독립이 아니었다.
    with open(apriltag_params.perform(context)) as fh:
        base_frames = yaml.safe_load(fh)["/**"]["ros__parameters"]["tag"]["frames"]

    # 카메라별 태그 크기 덮어쓰기 (예: "camera_3:0.080"). 가까이서 보는 카메라는
    # 먼 고정 카메라에서 잰 크기를 쓰면 거리가 짧게 나온다 (wrist_camera.yaml 참고).
    sizes = {}
    for item in LaunchConfiguration('tag_sizes').perform(context).split(','):
        if ':' in item:
            cam, size = item.split(':', 1)
            sizes[cam.strip()] = float(size)

    node_list = []
    # 3. 분리된 카메라 리스트를 돌면서 노드 생성
    for name in cameras:
        overrides = {"tag.frames": ["%s_%s" % (name, f) for f in base_frames]}
        if name in sizes:
            overrides["size"] = sizes[name]
            overrides["tag.sizes"] = [sizes[name]] * len(base_frames)
        node = Node(
            package="apriltag_ros",
            executable="apriltag_node",
            name="apriltag_node",
            namespace=name,
            # 다중 카메라 환경에서 TF 충돌 방지: 카메라마다 다른 태그 프레임
            parameters=[apriltag_params, overrides],
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

    tag_sizes_arg = DeclareLaunchArgument(
        'tag_sizes',
        default_value='',
        description='Per-camera tag edge overrides, e.g. camera_3:0.080'
    )

    return LaunchDescription([
        camera_names_arg,
        tag_sizes_arg,
        # OpaqueFunction을 통해 파이썬 로직(generate_apriltag_nodes)을 실행 시점에 평가
        OpaqueFunction(function=generate_apriltag_nodes)
    ])