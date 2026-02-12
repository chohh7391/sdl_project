from typing import Optional, Dict, List, Any
import torch
import numpy as np
from curobo.geom.types import Cuboid, Cylinder, Obstacle
from curobo.types.base import TensorDeviceType
from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import unit_quat
from cutamp.tamp_domain import HandEmpty, On
from cutamp.utils.shapes import MultiSphere


def load_Moving_env(
    entities: Dict[str, Any],
    movables: List[Obstacle],
    statics: List[Obstacle],
    ex_collision: List[Obstacle],
    tensor_args: TensorDeviceType = TensorDeviceType(),
) -> TAMPEnvironment:
    """Pick-and-place environment with a cylindrical beaker and small MultiSphere near goal."""
    
    
    entities["box_region"].pose = entities["box_goal"].pose.copy() 
    entities["box_region"].pose[2] += 0.02 

    env = TAMPEnvironment(
        name="move",
        movables=movables,
        statics=statics,
        ex_collision=ex_collision,
        type_to_objects={
            "Movable": movables,
            "Surface": [entities["table"], entities["box_goal"], entities["box_region"]],
            "ExCollision": [entities["box_region"]]
        },
        goal_state=frozenset(
            {
                HandEmpty.ground(),
                On.ground(movables[0].name, entities["box_region"].name), 
            }
        )
    )

    return env