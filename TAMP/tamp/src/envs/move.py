from typing import Dict, List, Any
from curobo.geom.types import Obstacle
from curobo.types.base import TensorDeviceType
from cutamp.envs import TAMPEnvironment
from cutamp.tamp_domain import HandEmpty, On
from envs.constants import PLANNER_Z_LIFT


def load_move_env(
    entities: Dict[str, Any],
    movables: List[Obstacle],
    statics: List[Obstacle],
    ex_collision: List[Obstacle],
    tensor_args: TensorDeviceType = TensorDeviceType(),
) -> TAMPEnvironment:
    """Pick-and-place environment with a cylindrical beaker and small MultiSphere near goal."""

    # box_region is the PLANNER-ONLY placement surface for the goal tray
    # (box_goal), which unlike box_region is a real static collider in the
    # simulator. Two things have to line up or the box is not actually set down
    # on the tray (same failure family as transfer's goal_region, REFACTOR.md
    # 2026-09-09):
    #   xy: the region must be the TRAY's footprint. Authored as ENTITIES'
    #       0.2 x 0.2 m it was wider than the 0.15 x 0.15 m tray, so a placement
    #       could satisfy the goal with the box hanging off the tray edge.
    #   z : a placement puts the object's bottom just above the region's top, and
    #       every movable's planner pose is PLANNER_Z_LIFT above its simulator
    #       pose, so the region's top has to sit one lift above the tray's top.
    #       Authored as tray_z + 0.02 it left the box 7 mm in the air at release.
    box_goal = entities["box_goal"]
    box_region = entities["box_region"]
    tray_top = box_goal.pose[2] + box_goal.dims[2] / 2.0
    box_region.dims = [box_goal.dims[0], box_goal.dims[1], box_region.dims[2]]
    box_region.pose = box_goal.pose.copy()
    box_region.pose[2] = tray_top + PLANNER_Z_LIFT - box_region.dims[2] / 2.0

    env = TAMPEnvironment(
        name="move",
        movables=movables,
        statics=statics,
        ex_collision=ex_collision,
        type_to_objects={
            "Movable": movables,
            "Surface": [entities["table"], entities["box_goal"], entities["box_region"]],
            "ExCollision": [entities["box_region"], entities["rearrange_region"]]
        },
        goal_state=frozenset(
            {
                HandEmpty.ground(),
                On.ground(movables[0].name, entities["box_region"].name), 
            }
        )
    )

    return env
