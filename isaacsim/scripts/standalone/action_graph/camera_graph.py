import omni.graph.core as og
import usdrt.Sdf
from typing import List


def create_ros_camera_graph(
    camera_path: str, camera_name: str
):
    (ros_camera_graph, _, _, _) = og.Controller.edit(
        {
            "graph_path": "/ActionGraph/CameraData",
            "evaluator_name": "push",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("createViewport", "isaacsim.core.nodes.IsaacCreateViewport"),
                ("getRenderProduct", "isaacsim.core.nodes.IsaacGetViewportRenderProduct"),
                ("setCamera", "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
                ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("cameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"), 
                ("cameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
                ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
                ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
                ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
                ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                ("setCamera.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                ("getRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
                ("getRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
                ("getRenderProduct.outputs:renderProductPath", "cameraHelperDepth.inputs:renderProductPath"),
            ],
            og.Controller.Keys.SET_VALUES: [
                # set camera path
                ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path(camera_path)]),
                # camera_info
                ("cameraHelperInfo.inputs:frameId", camera_name),
                ("cameraHelperInfo.inputs:topicName", camera_name + "/camera_info"),
                ("cameraHelperInfo.inputs:useSystemTime", False),
                # rgb
                ("cameraHelperRgb.inputs:frameId", camera_name),
                ("cameraHelperRgb.inputs:topicName", camera_name + "/rgb"),
                ("cameraHelperRgb.inputs:type", "rgb"),
                # depth
                ("cameraHelperDepth.inputs:frameId", camera_name),
                ("cameraHelperDepth.inputs:topicName", camera_name + "/depth"),
                ("cameraHelperDepth.inputs:type", "depth"),
                # viewport
                ("createViewport.inputs:viewportId", 1),
            ],
        },
    )

    return ros_camera_graph
