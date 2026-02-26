from typing import Optional, Dict, List, Any
import torch
from curobo.geom.types import Cuboid, Cylinder, Obstacle
from curobo.types.base import TensorDeviceType
from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import unit_quat
from cutamp.tamp_domain import HandEmpty, On, Poured, OnBeaker
from cutamp.utils.shapes import MultiSphere

def load_Transfer_env(
    entities: Dict[str, Any],
    movables: List[Obstacle],
    statics: List[Obstacle],
    ex_collision: List[Obstacle],
    tensor_args: TensorDeviceType = TensorDeviceType(),
) -> TAMPEnvironment:
    """Pick-and-place environment with a cylindrical beaker and small MultiSphere near goal."""

    entities["pour_region"].pose = movables[1].pose.copy()
    # entities["pour_region"].pose[2] = 0.18
    entities["pour_region"].pose = [entities["pour_region"].pose[0], entities["pour_region"].pose[1], 0.18, *unit_quat]
    # entities["pour_region"].pose = [0.3, 0.2, 0.16, *unit_quat]
    entities["goal_region"].pose = [0.35, -0.35, 0.015, *unit_quat] 
    # entities["goal_region"].pose = [0.51, -0.17, 0.015, *unit_quat]
    # "beaker": np.array([0.51, -0.17, 0.015]),

    # entities["beaker"].pose[2] = 0.06

    # entities["rearrange_region"].pose = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    env = TAMPEnvironment(
        name="transfer",
        movables=movables,
        statics=statics,
        ex_collision=ex_collision,
        type_to_objects={
            "Movable": movables,
            "Surface": [entities["table"], entities["pour_region"], entities["goal_region"]],
            "ExCollision": [entities["pour_region"], entities["rearrange_region"]]
            # "ExCollision": [entities["pour_region"]]
        },
        goal_state=frozenset(
            { 
                On.ground(movables[0].name, entities["goal_region"].name), 
                HandEmpty.ground(),
                Poured.ground(movables[0].name, entities["pour_region"].name),
            }
        )
    )

    return env, entities["pour_region"].pose