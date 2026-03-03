import omni.graph.core as og
import usdrt.Sdf
from typing import List

def create_tf_graph(target_prim_paths: List, parent_prim_path):

    target_paths = [usdrt.Sdf.Path(path) for path in target_prim_paths]
    parent_path = [usdrt.Sdf.Path(parent_prim_path)]

    (tf_graph, _, _, _) = og.Controller.edit(
        {"graph_path": "/ActionGraph/TF_Tree", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishTF.inputs:topicName", "/tf"),
                ("PublishTF.inputs:targetPrims", target_paths),
                ("PublishTF.inputs:parentPrim", parent_path),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
            ],
            
        },
    )
    return tf_graph