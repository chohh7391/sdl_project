from typing import Dict, List, Any
from curobo.geom.types import Obstacle
from curobo.types.base import TensorDeviceType
from cutamp.envs import TAMPEnvironment
from cutamp.tamp_domain import HandEmpty, On, OnBeaker
from envs.constants import PLANNER_Z_LIFT, VESSEL_WALL_M, region_dims_for  # noqa: F401


# How far off the stirrer's axis each object may be PLANNED. Both errors, plus
# the execution error of each, have to stay inside the vessel's opening or the
# stir bar lands on the rim instead of going in. The opening is
# 70 - 2*3 = 64 mm across and the 10 mm bar needs 14.1 mm of it at worst yaw, so
# the whole budget from the vessel's axis is 24.9 mm: 4 + 4 mm of planned
# allowance leaves ~17 mm for execution error. Sized to the WORST case (a region
# whose yaw is axis-aligned) -- the regions inherit the stirrer's randomized yaw,
# which only ever makes the AABB larger. Authoring these as fixed region sizes is
# what went wrong before: ENTITIES' 0.1 m goal region allowed 10 mm of planned
# vessel offset and measured 24 mm executed, and the bar then missed the vessel
# by 53 mm (seed 2); sizing the region to the vessel exactly made the constraint
# unsatisfiable for the seeds whose stirrer yaw is near 0 (seeds 1 and 3).
VESSEL_CENTRING_ALLOWANCE_M = 0.004
BAR_DROP_ALLOWANCE_M = 0.004


def load_stir_env(
    entities: Dict[str, Any],
    movables: List[Obstacle],
    statics: List[Obstacle],
    ex_collision: List[Obstacle],
    tensor_args: TensorDeviceType = TensorDeviceType(),
) -> TAMPEnvironment:
    """Pick-and-place environment with a cylindrical beaker and small MultiSphere near goal."""
    
    stirrer = entities["stirrer"]
    goal_region = entities["goal_region"]
    beaker_region = entities["beaker_region"]

    # The stir bar is released above the vessel's mouth and dropped in, so the
    # region stays well clear of the rim (the fingers cannot enter the opening).
    # Its xy extent is the vessel's CLEAR OPENING, so that the placement
    # constraint -- which insets the region by the object's own radius -- means
    # "the bar clears the mouth". Authored as ENTITIES' 80 mm square it was wider
    # than the 70 mm vessel, and the bar was planned up to 17.5 mm off the axis
    # (measured 9-34 mm), i.e. onto the rim.
    vessel = movables[0]
    stir_bar = movables[1]
    opening = max(vessel.dims[0] - 2.0 * VESSEL_WALL_M, 2.0 * VESSEL_WALL_M)
    bar_drop = region_dims_for(stir_bar.dims[0], BAR_DROP_ALLOWANCE_M)
    beaker_region.dims = [bar_drop, bar_drop, beaker_region.dims[2]]
    beaker_region.pose = stirrer.pose.copy()
    beaker_region.pose[2] += 0.2

    # goal_region is the PLANNER-ONLY surface that means "the vessel is on the
    # stirrer"; the stirrer itself is a real collider. A placement puts the
    # vessel's bottom just above this region's top, and every movable's planner
    # pose (the stirrer's included) is PLANNER_Z_LIFT above its simulator pose,
    # so the region's top has to coincide with the stirrer's PLANNER top for the
    # vessel to come down on the stirrer in the simulator. Authored as
    # stirrer_z + 0.05 the region sat one thickness higher and the vessel was
    # released 12 mm in the air (measured on stir seed 0: it tumbled and settled
    # at 38 deg). Same failure family as transfer's goal_region and move's
    # box_region (REFACTOR.md 2026-09-09).
    # The vessel has to end up CENTRED on the stirrer plate, not merely somewhere
    # on it: the stir bar is dropped on the plate's axis, so every millimetre the
    # vessel sits off-axis eats into the bar's clearance through the opening.
    # ENTITIES' 0.1 m region let the vessel be planned up to 15 mm off-axis and
    # measured 21.7 mm after execution, which put the bar beside the vessel
    # (seed 2: bar 53 mm from the vessel). Sizing the region to the vessel's own
    # footprint makes "on the stirrer" mean "centred on the stirrer", since the
    # placement constraint insets the region by the object's own radius.
    # Scoped to this env -- transfer's goal region must keep its authored size.
    vessel_place = region_dims_for(vessel.dims[0], VESSEL_CENTRING_ALLOWANCE_M)
    goal_region.dims = [vessel_place, vessel_place, goal_region.dims[2]]
    goal_region.pose = stirrer.pose.copy()
    goal_region.pose[2] = (
        stirrer.pose[2] + stirrer.dims[2] / 2.0 - goal_region.dims[2] / 2.0
    )

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