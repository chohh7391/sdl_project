import omni.graph.core as og
import usdrt.Sdf
from typing import List

def create_ros_camera_graph(camera_paths: List[str], camera_names: List[str]):
    """
    여러 대의 카메라 경로와 이름을 입력받아 동적으로 ROS2 퍼블리셔 그래프를 생성합니다.
    """
    assert len(camera_paths) == len(camera_names), "카메라 경로와 이름 리스트의 길이가 같아야 합니다."

    # 고정 해상도 
    width = 640
    height = 480

    # 노드, 연결, 설정값을 담을 리스트 초기화 (기본 OnTick 노드는 항상 하나)
    nodes_list = [("OnTick", "omni.graph.action.OnTick")]
    connect_list = []
    set_values_list = []

    # 각 카메라별로 노드를 생성하고 연결
    for i, (cam_path, cam_name) in enumerate(zip(camera_paths, camera_names)):
        # 노드 이름에 인덱스를 붙여서 고유하게 만듦
        rp_node = f"createRenderProduct_{i}"
        rgb_node = f"cameraHelperRgb_{i}"
        info_node = f"cameraHelperInfo_{i}"
        depth_node = f"cameraHelperDepth_{i}"

        # 1. 노드 생성 (Create Nodes)
        nodes_list.extend([
            (rp_node, "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            (rgb_node, "isaacsim.ros2.bridge.ROS2CameraHelper"),
            (info_node, "isaacsim.ros2.bridge.ROS2CameraInfoHelper"), 
            (depth_node, "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ])

        # 2. 노드 연결 (Connect)
        connect_list.extend([
            ("OnTick.outputs:tick", f"{rp_node}.inputs:execIn"),
            
            # 실행 순서 연결
            (f"{rp_node}.outputs:execOut", f"{rgb_node}.inputs:execIn"),
            (f"{rp_node}.outputs:execOut", f"{info_node}.inputs:execIn"),
            (f"{rp_node}.outputs:execOut", f"{depth_node}.inputs:execIn"),
            
            # 렌더 프로덕트 경로 데이터 연결
            (f"{rp_node}.outputs:renderProductPath", f"{rgb_node}.inputs:renderProductPath"),
            (f"{rp_node}.outputs:renderProductPath", f"{info_node}.inputs:renderProductPath"),
            (f"{rp_node}.outputs:renderProductPath", f"{depth_node}.inputs:renderProductPath"),
        ])

        # 3. 파라미터 설정 (Set Values)
        set_values_list.extend([
            # RenderProduct 설정
            (f"{rp_node}.inputs:cameraPrim", [usdrt.Sdf.Path(cam_path)]),
            (f"{rp_node}.inputs:width", width),
            (f"{rp_node}.inputs:height", height),
            
            # CameraInfo 퍼블리셔 설정
            (f"{info_node}.inputs:frameId", cam_name),
            (f"{info_node}.inputs:topicName", f"/{cam_name}/camera_info"),
            
            # RGB 퍼블리셔 설정
            (f"{rgb_node}.inputs:frameId", cam_name),
            (f"{rgb_node}.inputs:topicName", f"/{cam_name}/rgb"),
            (f"{rgb_node}.inputs:type", "rgb"),
            
            # Depth 퍼블리셔 설정
            (f"{depth_node}.inputs:frameId", cam_name),
            (f"{depth_node}.inputs:topicName", f"/{cam_name}/depth"),
            (f"{depth_node}.inputs:type", "depth"),
        ])

    # 완성된 리스트들을 바탕으로 Action Graph 한 번에 생성
    (ros_camera_graph, _, _, _) = og.Controller.edit(
        {
            "graph_path": "/ActionGraph/MultiCameraData",
            "evaluator_name": "push",
            # 다중 인식을 쌩쌩하게 돌리면서 부하를 줄이려면 ONDEMAND 사용 (렌더 성능 이슈 시 변경)
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            og.Controller.Keys.CREATE_NODES: nodes_list,
            og.Controller.Keys.CONNECT: connect_list,
            og.Controller.Keys.SET_VALUES: set_values_list,
        },
    )

    return ros_camera_graph