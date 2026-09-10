from typing import Dict, List, Any
from curobo.geom.types import Obstacle
from curobo.types.base import TensorDeviceType
from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import unit_quat
from cutamp.tamp_domain import HandEmpty, On, Poured
from envs.constants import PLANNER_Z_LIFT


def load_transfer_env(
    entities: Dict[str, Any],
    movables: List[Obstacle],
    statics: List[Obstacle],
    ex_collision: List[Obstacle],
    tensor_args: TensorDeviceType = TensorDeviceType(),
) -> TAMPEnvironment:
    """Pick-and-place environment with a cylindrical beaker and small MultiSphere near goal."""

    # movables = [from_vessel, to_vessel]  ->  to_vessel (movables[1]) is the pour target (flask).
    #
    # Pour target = the target vessel's mouth (rim), well-defined from the flask geometry:
    #   xy   : target-vessel center  (spout ends up over the flask mouth)
    #   z    : flask rim (top face) + a small vertical clearance
    # Cuboid.pose is the geometric CENTER, so the rim (top) is center_z + dims_z/2.
    # Previously this was a bare `pose[2] += 0.11` magic constant, which only *coincidentally*
    # lands at rim+clearance for the nominal 0.12 m-tall flask and silently drifts off the rim
    # if the target vessel's height changes (or under position randomization). Derive it instead.
    to_vessel = movables[1]
    POUR_CLEARANCE = 0.05  # [m] gap between beaker bottom and flask rim so the tilted beaker clears the rim; tunable
    rim_z = to_vessel.pose[2] + to_vessel.dims[2] / 2.0  # flask top / mouth height
    entities["pour_region"].pose = to_vessel.pose.copy()  # xy + yaw aligned to the flask mouth
    entities["pour_region"].pose[2] = rim_z + POUR_CLEARANCE
    # The goal region is a PLANNER-ONLY surface: there is no goal-region prim in
    # the simulator, so the physical support at that xy is the table top. A
    # placement puts the vessel's bottom at (surface top + activation distance +
    # 2 mm), and every movable's planner pose is PLANNER_Z_LIFT above its
    # simulator pose, so the vessel's REAL bottom ends up at
    #     surface_top + 2 mm - PLANNER_Z_LIFT
    # above the table -- i.e. the vessel is released in mid-air and dropped.
    # Authored at z = 0.015 (top = 0.020) that is a 12 mm drop; measured on seed 5
    # the vessel was released 15 mm above where it comes to rest, and the fall is
    # what the placement stability depends on. Put the region's top exactly one
    # lift above the real support so the vessel is set down on the table instead.
    goal_region = entities["goal_region"]
    table_top = entities["table"].pose[2] + entities["table"].dims[2] / 2.0
    goal_region.pose = [
        0.35, -0.35, table_top + PLANNER_Z_LIFT - goal_region.dims[2] / 2.0, *unit_quat,
    ]

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