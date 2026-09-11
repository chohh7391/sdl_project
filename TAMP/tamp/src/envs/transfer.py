from typing import Dict, List, Any

import numpy as np
from curobo.geom.types import Obstacle
from curobo.types.base import TensorDeviceType
from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import unit_quat
from cutamp.tamp_domain import HandEmpty, On, Poured
from envs.constants import PLANNER_Z_LIFT, region_dims_for


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
    from_vessel = movables[0]
    to_vessel = movables[1]
    # Gap between the source vessel's BOTTOM and the target's rim while the
    # source is still upright. It sets how far the stream falls, because the lip
    # is a whole vessel height above the bottom: at 0.05 the lip sat 185 mm over
    # the mouth. Tilting about the lip RAISES the body (the vessel hangs below
    # the pivot), so the upright state is what this has to clear -- and the
    # planner models the vessel PLANNER_Z_LIFT higher than the simulator does, so
    # the real gap is this minus that.
    #
    # Kept at 0.05 rather than tightened: at 0.02 the whole pour placement drops
    # 30 mm and the arm loses reach, and the lip-pivot pour path then failed on
    # most layouts (it fell back to the old wrist pour, which undoes the
    # horizontal fix and is far worse than a longer drop). The fall height is
    # instead reduced by lowering the lip DURING the tilt, once the vessel's body
    # has swung clear -- see POUR_DESCENT_* in cutamp/motion_solver.py.
    POUR_CLEARANCE = 0.05
    rim_z = to_vessel.pose[2] + to_vessel.dims[2] / 2.0  # flask top / mouth height

    # The liquid leaves the source vessel's LIP, not its axis, so centring the
    # source on the target's axis puts the stream one vessel-radius off the mouth
    # before the pour even starts -- and tilting then swings the lip much further
    # (measured: 36-81 mm from the flask axis at peak tilt, against a ~17 mm mouth
    # radius on the flask that was bought). Offset the placement by the source
    # vessel's radius so that the LIP, not the centre, sits over the mouth.
    #
    # The offset direction is also the direction the vessel will be tilted, and
    # the executor has to agree with it, so it is carried in the region's YAW
    # (the region is a square and the vessel is a symmetric cylinder, so its yaw
    # was otherwise unused). Leaning away from the robot base keeps the arm on
    # the near side of the target vessel.
    lean = np.asarray(to_vessel.pose[:2], dtype=float)
    norm = float(np.linalg.norm(lean))
    lean = lean / norm if norm > 1e-6 else np.array([1.0, 0.0])
    lip_radius = from_vessel.dims[0] / 2.0
    yaw = float(np.arctan2(lean[1], lean[0]))

    # How far off the mouth the vessel's lip may be PLANNED. The region was
    # authored 0.08 m wide, which allows 10 mm (0.04 - 0.005 sphere radius -
    # 0.025 vessel half-width) before execution error is added; measured lip
    # error at peak tilt was then 17-27 mm against a ~17 mm mouth radius.
    POUR_LIP_ALLOWANCE_M = 0.004
    pour_span = region_dims_for(from_vessel.dims[0], POUR_LIP_ALLOWANCE_M)
    entities["pour_region"].dims = [pour_span, pour_span, entities["pour_region"].dims[2]]
    entities["pour_region"].pose = [
        float(to_vessel.pose[0] - lip_radius * lean[0]),
        float(to_vessel.pose[1] - lip_radius * lean[1]),
        rim_z + POUR_CLEARANCE,
        float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0)),
    ]
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