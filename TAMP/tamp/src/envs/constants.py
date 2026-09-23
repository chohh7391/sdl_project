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


# --- vessel geometry -------------------------------------------------------
# The simulated glassware was authored before the real vessels were bought, and
# replaying a recorded trajectory on the robot exposed the mismatch. Both sets
# live here so a run can state which one it used: every measurement in the
# revision so far used "legacy", and "real" is the glassware actually on the
# bench.
#
#   beaker   purchased: 60 mm outer diameter, 72 mm tall, 100 mL
#   flask    purchased: 160 mm tall, 38 mm mouth outer diameter, 300 mL
#
# The flask's BODY diameter was not among the measurements taken, so 87 mm is
# used -- the base diameter of a standard 300 mL Erlenmeyer, consistent with the
# 160 mm height that was measured. Correct FLASK_BODY_DIA_M if the real flask
# differs; nothing else needs to change.
#
# cuTAMP plans every object as a cuboid, so a conical flask is modelled by its
# widest part. That is conservative for collision -- the planner keeps the arm
# clear of the whole body -- but it does NOT describe the opening, which is the
# 38 mm mouth. Anything that has to reach INTO the flask uses FLASK_MOUTH_DIA_M.
import os as _os

FLASK_BODY_DIA_M = 0.087
FLASK_MOUTH_DIA_M = 0.038

_GLASSWARE = {
    "legacy": {
        "beaker": [0.05, 0.05, 0.135],
        "flask": [0.07, 0.07, 0.12],
        "flask_mouth_dia": 0.064,   # the hollow box's clear opening
    },
    "real": {
        "beaker": [0.06, 0.06, 0.072],
        "flask": [FLASK_BODY_DIA_M, FLASK_BODY_DIA_M, 0.160],
        "flask_mouth_dia": FLASK_MOUTH_DIA_M,
    },
}


def glassware_set():
    """Which glassware the run uses: SDL_GLASSWARE=legacy (default) or real."""
    name = _os.environ.get("SDL_GLASSWARE", "legacy").strip().lower()
    if name not in _GLASSWARE:
        raise ValueError(
            "SDL_GLASSWARE=%r is not one of %s" % (name, sorted(_GLASSWARE)))
    return name


def vessel_height_override(name):
    """Extra height added to a vessel, in metres, from SDL_<NAME>_EXTRA_H_M.

    Replaying a planned grasp on the real cell showed the fingers closing at the
    beaker's rim rather than around its body: measured over the eight recorded
    0-10 deg solves, the grasp frame closes 55.9-64.6 mm above the bench on a
    beaker only 72 mm tall, i.e. 7-16 mm below the rim. PLANNER_Z_LIFT raises
    every movable another 10 mm in the planner's world relative to the
    simulator's, so the commanded height sits that much higher again against the
    real object. Any calibration error on top of that closes the gripper in air.

    Raising the modelled vessel moves the sampled grasp band down the body in
    relative terms, so the grasp lands on the wall instead of the lip. It is an
    override rather than a new glassware set because it describes the RIG -- a
    vessel on a riser, or a taller vessel than the one first measured -- not a
    different purchase, and the real cell is where the number comes from.
    """
    key = "SDL_%s_EXTRA_H_M" % name.upper()
    try:
        return float(_os.environ.get(key, "0.0"))
    except ValueError:
        raise ValueError("%s=%r is not a number" % (key, _os.environ.get(key)))


def vessel_dims(name):
    """Outer cuboid dims [x, y, z] of `name` under the selected glassware."""
    dims = list(_GLASSWARE[glassware_set()][name])
    dims[2] = dims[2] + vessel_height_override(name)
    if dims[2] <= 0.0:
        raise ValueError("vessel %r ends up with non-positive height %.4f m"
                         % (name, dims[2]))
    return dims


def flask_mouth_dia():
    return float(_GLASSWARE[glassware_set()]["flask_mouth_dia"])
