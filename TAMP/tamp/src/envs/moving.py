from typing import Dict, List, Any
from curobo.geom.types import Obstacle
from curobo.types.base import TensorDeviceType
from cutamp.envs import TAMPEnvironment
from cutamp.tamp_domain import HandEmpty, On


def load_moving_env(
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
