"""Constants shared by the environment definitions.

Kept in their own module so both `envs.utils` (which applies them) and the
individual environments (which have to compensate for them) can import them
without a circular import.
"""

# Every movable's pose is raised by this much in the PLANNER's world relative to
# the simulator, so that an object resting exactly on a surface does not start
# out flagged as in contact with it (see `TAMPEnvManager.update_entities`).
#
# The consequence, which every placement surface has to account for, is that the
# planner's world is NOT the simulator's world: a vessel whose planner-side
# bottom sits on a surface is PLANNER_Z_LIFT above that surface in reality.
PLANNER_Z_LIFT = 0.01  # [m]

# Wall thickness of the hollow vessel collider in the simulator
# (isaacsim Task.FLASK_WALL_M). The planner still models the vessel as a solid
# Cuboid of the OUTER dims, so this is only needed where a placement has to
# clear the vessel's opening rather than its outside.
VESSEL_WALL_M = 0.003  # [m]

# Radius of the surface collision spheres cuTAMP fits to every object
# (TAMPWorld coll_sphere_radius). A placement's in-xy constraint compares the
# object's SPHERE CENTRES against the surface AABB inset by each sphere's own
# radius, and those centres sit on the object's surface, so the object's centre
# may be offset from a region's centre by at most
#     region_dims/2 - COLL_SPHERE_RADIUS_M - object_dims/2
# (in the worst case, a region whose yaw is axis-aligned; a yawed region's AABB
# is larger and only adds slack). Sizing a region from a wanted allowance is
# therefore  region_dims = object_dims + 2 * (allowance + COLL_SPHERE_RADIUS_M).
COLL_SPHERE_RADIUS_M = 0.005  # [m]


def region_dims_for(object_dims_xy, allowance):
    """Region xy side length that lets an object's centre sit within `allowance`
    of the region centre (see COLL_SPHERE_RADIUS_M)."""
    return float(object_dims_xy) + 2.0 * (float(allowance) + COLL_SPHERE_RADIUS_M)
