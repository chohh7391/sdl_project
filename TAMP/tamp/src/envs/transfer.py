from typing import Dict, List, Any
from curobo.geom.types import Obstacle
from curobo.types.base import TensorDeviceType
from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import unit_quat
from cutamp.tamp_domain import HandEmpty, On, Poured


def load_transfer_env(
    entities: Dict[str, Any],
    movables: List[Obstacle],
    statics: List[Obstacle],
    ex_collision: List[Obstacle],
    tensor_args: TensorDeviceType = TensorDeviceType(),
) -> TAMPEnvironment:
    """Pick-and-place environment with a cylindrical beaker and small MultiSphere near goal."""

    # movables = [from_vessel, to_vessel]
    entities["pour_region"].pose = movables[1].pose.copy()
    entities["pour_region"].pose = [entities["pour_region"].pose[0], entities["pour_region"].pose[1], 0.18, *unit_quat]
    entities["goal_region"].pose = [0.35, -0.35, 0.015, *unit_quat]

    env = TAMPEnvironment(
        name="transfer",
        movables=movables,
        statics=statics,
        ex_collision=ex_collision,
        type_to_objects={
            "Movable": movables,
            "Surface": [entities["table"], entities["pour_region"], entities["goal_region"]],
            "ExCollision": [entities["pour_region"], entities["rearrange_region"]]
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