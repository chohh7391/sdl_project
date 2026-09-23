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
        # The beaker MEASURES 50 x 50 x 70 mm (author, 2026-09-23). It is
        # modelled 10 mm SHORTER than that on purpose: its upper part tapers, so
        # the parallel fingers need to close below the taper, and the sampler
        # draws the pinch height from the modelled body. Shortening the model
        # moves the whole feasible band down the real vessel rather than biasing
        # within it, which is what SDL_GRASP_H_FRAC does.
        #
        # Earlier values here were 72 mm (from the purchase note) and 90 mm
        # (a mis-measurement); both are superseded.
        "beaker": [0.05, 0.05, 0.060],
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

# --- the real cell's electronic scale ---------------------------------------
# The physical Transfer target stands on a balance, not on the bench, and the
# balance is a large obstacle right where the arm has to pour. Measured on the
# rig (author, 2026-09-23):
#
#   body            250 mm along its long axis x 200 mm across
#   weighing pan    130 x 130 mm, centred across the short axis and 140 mm from
#                   the FAR end of the long axis, i.e. 110 mm from the near end
#   orientation     the pan end faces the robot, so the long axis is radial
#
# The pan TOP height is 90 mm, measured on the rig (author, 2026-09-23). It was
# first inferred as 86 mm from the difference between the two tag readings
# (beaker 0.144 m on the bench, flask 0.230 m on the pan, uniform mount height);
# the measurement supersedes that, and the 4 mm agreement is a useful check that
# the two readings describe what they were taken to describe.
#
# The body is modelled as ONE cuboid spanning the whole footprint up to the pan
# top, rather than a body plus a thinner pan. That is deliberately conservative:
# the planner then keeps the arm clear of the entire balance instead of letting
# it swing through the space beside the pan, which is where the display and the
# draft shield of a real balance are.
SCALE_LONG_M = 0.25
SCALE_SHORT_M = 0.20
SCALE_PAN_XY_M = 0.13
SCALE_PAN_FROM_FAR_M = 0.14
SCALE_TOP_M = float(_os.environ.get("SDL_SCALE_TOP_M", "0.090"))


def scale_dims():
    """Outer cuboid dims of the balance: [long (radial), short, height]."""
    return [SCALE_LONG_M, SCALE_SHORT_M, SCALE_TOP_M]


def scale_pose_from_pan(pan_xy):
    """Balance centre (x, y) and yaw [rad] that put its PAN at `pan_xy`.

    The rig is measured at the pan because that is where the vessel stands, but
    a cuboid is placed by its centre. The pan sits off-centre along the long
    axis, so the body centre is displaced from the pan by half the difference
    between the two end distances, along the radial direction the balance faces.
    """
    import math
    px, py = float(pan_xy[0]), float(pan_xy[1])
    r = math.hypot(px, py)
    if r < 1e-6:
        raise ValueError("the balance pan cannot be at the robot's own axis")
    ux, uy = px / r, py / r                      # robot -> pan, the long axis
    near = SCALE_LONG_M - SCALE_PAN_FROM_FAR_M   # pan centre to the near end
    shift = (SCALE_PAN_FROM_FAR_M - near) / 2.0  # pan centre -> body centre
    return (px + shift * ux, py + shift * uy), math.atan2(uy, ux)

# --- the bench plane --------------------------------------------------------
# How far the table top sits BELOW the robot's base_link plane, beyond what the
# nominal scene assumes. The simulated cell had the table top level with the
# robot base's own reference; the real cell's bench is lower, and replaying a
# trajectory planned against the nominal height closed the gripper above the
# vessel. Measured on the rig as 13 mm (author, 2026-09-23).
#
# Everything that RESTS on the bench moves with it -- the vessels, the balance,
# the stirrer, the box and its tray -- because the offset describes where the
# supporting surface is, not where one object is. The planner's goal region
# derives its height from the table entity, so it follows without help.
TABLE_Z_OFFSET = float(_os.environ.get("SDL_TABLE_Z_M", "0.0"))

# --- riser under the source vessel ------------------------------------------
# The beaker stands on a box, not on the bench. It is there because a LEVEL
# gripper cannot reach low on a vessel that sits on the bench: the wrist's
# collision spheres are 58 mm in radius and, held horizontal, they occupy that
# much below the grasp frame, so the grasp cannot go under 58 mm above the
# bench without driving the wrist through the table. Measured over six solves
# it settled at 62 mm, 4 mm above that floor, on a 70 mm vessel -- i.e. at the
# rim, which is where the beaker tapers.
#
# Raising the vessel moves its body up past that floor instead of fighting it:
# on a 50 mm box the same 62 mm grasp lands 12 mm above the beaker's base.
#
# SDL_BEAKER_RISER_M is the height. The footprint defaults to a square of the
# same size -- "a 5 cm box" -- and SDL_BEAKER_RISER_XY_M overrides it. The
# footprint matters to the planner, not just the height: the riser is an
# obstacle the arm has to come around to reach the vessel on top of it.
BEAKER_RISER_M = float(_os.environ.get("SDL_BEAKER_RISER_M", "0.0"))
BEAKER_RISER_XY_M = float(
    _os.environ.get("SDL_BEAKER_RISER_XY_M", "0.0")) or BEAKER_RISER_M


def beaker_riser_dims():
    """Outer cuboid dims of the riser, or None when the run has no riser."""
    if BEAKER_RISER_M <= 0.0:
        return None
    return [BEAKER_RISER_XY_M, BEAKER_RISER_XY_M, BEAKER_RISER_M]
