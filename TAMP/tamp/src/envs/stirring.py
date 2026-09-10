from typing import Optional, Dict, List, Any
import torch
from curobo.geom.types import Cuboid, Cylinder, Obstacle
from curobo.types.base import TensorDeviceType
from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import unit_quat
from cutamp.tamp_domain import HandEmpty, On, Poured, OnBeaker
from cutamp.utils.shapes import MultiSphere


def load_Stir_env(
    entities: Dict[str, Any],
    movables: List[Obstacle],
    statics: List[Obstacle],
    ex_collision: List[Obstacle],
    tensor_args: TensorDeviceType = TensorDeviceType(),
) -> TAMPEnvironment:
    """Pick-and-place environment with a cylindrical beaker and small MultiSphere near goal."""
    
    entities["beaker_region"].pose = entities["stirrer"].pose.copy() # [0.3, 0.3, 0.16, *unit_quat]
    entities["beaker_region"].pose[2] += 0.17
    entities["goal_region"].pose = entities["stirrer"].pose.copy()
    entities["goal_region"].pose[2] += 0.07

    env = TAMPEnvironment(
        name="stir",
        movables=movables,
        statics=statics,
        ex_collision=ex_collision,
        type_to_objects={
            "Movable": movables,
            "Surface": [entities["table"], entities["stirrer"], entities["beaker_region"], entities["goal_region"]],
            "ExCollision": [entities["beaker_region"], entities["rearrange_region"]]
        },
        goal_state=frozenset(
            {
                HandEmpty.ground(),
                On.ground(movables[0].name, entities["goal_region"].name), 
                On.ground(entities["magnet"].name, entities["beaker_region"].name),
                OnBeaker.ground(entities["magnet"].name, entities["beaker_region"].name), 
            }
        )
    )

    return env