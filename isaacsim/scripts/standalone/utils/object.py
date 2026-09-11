import math
import numpy as np
from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from pxr import Usd, UsdGeom, UsdPhysics


# =============================================================================
# Box-collider object builders (REFACTOR.md II-2 / II-3).
#
# cuTAMP (TAMP/tamp/src/envs/utils.py ENTITIES) plans EVERY entity as a curobo
# Cuboid (a BOX). For plan/execution self-consistency the Isaac Sim colliders
# must be the SAME boxes, matching the cuTAMP dims exactly:
#     beaker  [0.05, 0.05, 0.135]   flask [0.07, 0.07, 0.12]
#     box     [0.108, 0.108, 0.08]  stirrer [0.18, 0.18, 0.09]
#     box_goal[0.15, 0.15, 0.01]
# `dims` below are those FULL side lengths (curobo Cuboid convention).
#
# The visual USD is kept as-is (referenced under <prim>/visual) but its own
# collision/rigid-body physics is disabled, so the ONLY collider is the clean
# box we author. This removes the PhysX "triangle mesh ... falling back to
# convexHull" errors (stirrer heat_device meshes, FluidBottle mesh), whose
# colliders sit at deep nested prim paths inside the referenced USDs.
#
# The box collider is centered on the prim origin, so the world pose reported
# by get_world_pose() is the box CENTER -- the same convention curobo uses --
# and the caller places the object at z = dims_z/2 so the box rests on the
# table top (z = 0).
# =============================================================================


def _disable_visual_physics(stage, root_path):
    """Strip every collision / rigid-body physics found ANYWHERE under root_path
    (the referenced visual subtree) so the visual USD contributes NO collider of
    its own -- the clean box we add is the only physics geometry.

    This is what silences the triangle-mesh -> convexHull PhysX fallback: the
    offending colliders live at nested paths (e.g.
    /visual/heat_device/.../plat, /visual/<mesh>/<mesh>). NOTE: merely setting
    physics:collisionEnabled = False is NOT enough -- PhysX still cooks the
    triangle mesh and emits the fallback warning. The collider must be REMOVED
    (RemoveAPI), which composes cleanly because these prims are directly
    referenced (not instance proxies). Instance proxies, if any, are first
    un-instanced so their descendants become authorable."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return

    # Un-instance any instanced subtrees so descendants are directly editable.
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsInstance():
            prim.SetInstanceable(False)

    _apis = [UsdPhysics.CollisionAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI]
    if hasattr(UsdPhysics, "MeshCollisionAPI"):
        _apis.append(UsdPhysics.MeshCollisionAPI)
    for prim in Usd.PrimRange(root):
        for api in _apis:
            if prim.HasAPI(api):
                prim.RemoveAPI(api)


def _add_box_collider(stage, parent_path, dims):
    """Author an invisible box collider (full side lengths == cuTAMP dims),
    centered on the parent origin so the reported pose is the box center."""
    collision_path = parent_path + "/collision"
    cube = UsdGeom.Cube.Define(stage, collision_path)
    half = np.asarray(dims, dtype=float) / 2.0
    # UsdGeom.Cube has base size 2.0 (spans -1..1); scale by half-extents.
    UsdGeom.XformCommonAPI(cube).SetScale((float(half[0]), float(half[1]), float(half[2])))
    UsdGeom.XformCommonAPI(cube).SetTranslate((0.0, 0.0, 0.0))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
    return cube


def _add_hollow_box_collider(stage, parent_path, dims, wall):
    """Author an open-topped box collider: a base slab plus four side walls.

    A vessel built with `_add_box_collider` is a SOLID box, so nothing can ever
    be inside it. Measured on Stir: the stir bar came to rest at exactly
    (flask mouth + its own half-height) on every seed -- it was sitting on the
    lid of a solid block, and the "stir bar inside the vessel" terminal
    condition could not occur in any run.

    The OUTER envelope is unchanged (`dims` full side lengths, centred on the
    prim origin), so the object's external collisions and its resting height are
    exactly as before; only the interior becomes real. The clear opening is
    (dx - 2*wall) x (dy - 2*wall) and the inner floor sits `wall` above the
    outer bottom.
    """
    dx, dy, dz = (float(v) for v in dims)
    t = float(wall)
    if t <= 0.0 or 2.0 * t >= min(dx, dy) or t >= dz:
        raise ValueError(f"wall {t} does not fit inside dims {dims}")

    # (name, half-extents, centre) -- centres are relative to the prim origin,
    # which is the centre of the outer envelope.
    parts = [
        ("base", (dx / 2.0, dy / 2.0, t / 2.0), (0.0, 0.0, -dz / 2.0 + t / 2.0)),
        ("wall_xp", (t / 2.0, dy / 2.0, (dz - t) / 2.0), (dx / 2.0 - t / 2.0, 0.0, t / 2.0)),
        ("wall_xn", (t / 2.0, dy / 2.0, (dz - t) / 2.0), (-dx / 2.0 + t / 2.0, 0.0, t / 2.0)),
        ("wall_yp", ((dx - 2.0 * t) / 2.0, t / 2.0, (dz - t) / 2.0), (0.0, dy / 2.0 - t / 2.0, t / 2.0)),
        ("wall_yn", ((dx - 2.0 * t) / 2.0, t / 2.0, (dz - t) / 2.0), (0.0, -dy / 2.0 + t / 2.0, t / 2.0)),
    ]
    cubes = []
    for part_name, half, centre in parts:
        cube = UsdGeom.Cube.Define(stage, f"{parent_path}/collision_{part_name}")
        UsdGeom.XformCommonAPI(cube).SetScale(tuple(float(h) for h in half))
        UsdGeom.XformCommonAPI(cube).SetTranslate(tuple(float(c) for c in centre))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
        cubes.append(cube)
    return cubes


def _set_solver_iterations(prim, position_iters, velocity_iters):
    """Best-effort per-body PhysX solver iteration counts (contact/rest
    stability). PhysxSchema is only importable once the physx extension is
    loaded, so guard it; UsdPhysics-only fallback simply skips."""
    if position_iters is None:
        return
    try:
        from pxr import PhysxSchema
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateSolverPositionIterationCountAttr(int(position_iters))
        if velocity_iters is not None:
            rb.CreateSolverVelocityIterationCountAttr(int(velocity_iters))
    except Exception:
        pass


def create_box_collider_rigid(
    prim_path, usd_path, name, position, orientation, dims,
    mass=0.1, visual_offset_z=0.0,
    solver_position_iterations=8, solver_velocity_iterations=1,
):
    """Dynamic rigid body: visual USD (physics stripped) + one clean box
    collider matching cuTAMP dims. A positive mass + box collider let PhysX
    auto-compute a valid inertia (no negative-mass/invalid-inertia warning)."""
    stage = get_current_stage()

    parent = UsdGeom.Xform.Define(stage, prim_path)
    UsdPhysics.RigidBodyAPI.Apply(parent.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(parent.GetPrim())
    mass_api.CreateMassAttr(float(mass))

    visual_path = prim_path + "/visual"
    add_reference_to_stage(usd_path=usd_path, prim_path=visual_path)
    if visual_offset_z:
        UsdGeom.XformCommonAPI(stage.GetPrimAtPath(visual_path)).SetTranslate(
            (0.0, 0.0, float(visual_offset_z))
        )
    _disable_visual_physics(stage, visual_path)

    _add_box_collider(stage, prim_path, dims)
    _set_solver_iterations(parent.GetPrim(), solver_position_iterations, solver_velocity_iterations)

    return SingleRigidPrim(
        prim_path=prim_path, name=name, position=position, orientation=orientation,
    )


def create_hollow_box_collider_rigid(
    prim_path, usd_path, name, position, orientation, dims, wall,
    mass=0.1, visual_offset_z=0.0,
    solver_position_iterations=8, solver_velocity_iterations=1,
):
    """Dynamic rigid body whose collider is an OPEN-TOPPED box (see
    `_add_hollow_box_collider`): the visual USD with its physics stripped, plus
    a base slab and four walls of thickness `wall`. Used for a vessel that
    something has to go INTO. The outer envelope still matches the cuTAMP
    Cuboid, so the planner's model of the object is unchanged."""
    stage = get_current_stage()

    parent = UsdGeom.Xform.Define(stage, prim_path)
    UsdPhysics.RigidBodyAPI.Apply(parent.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(parent.GetPrim())
    mass_api.CreateMassAttr(float(mass))

    visual_path = prim_path + "/visual"
    add_reference_to_stage(usd_path=usd_path, prim_path=visual_path)
    if visual_offset_z:
        UsdGeom.XformCommonAPI(stage.GetPrimAtPath(visual_path)).SetTranslate(
            (0.0, 0.0, float(visual_offset_z))
        )
    _disable_visual_physics(stage, visual_path)

    _add_hollow_box_collider(stage, prim_path, dims, wall)
    _set_solver_iterations(parent.GetPrim(), solver_position_iterations, solver_velocity_iterations)

    return SingleRigidPrim(
        prim_path=prim_path, name=name, position=position, orientation=orientation,
    )


def create_box_collider_static(
    prim_path, usd_path, name, position, orientation, dims, visual_offset_z=0.0,
):
    """Static collider: visual USD (physics stripped) + one clean box collider,
    NO RigidBodyAPI on the parent. A collider with no rigid body is a static
    actor in PhysX -- it needs no mass/inertia, so the previous
    'negative mass / invalid inertia' warning cannot occur. Used for box_goal
    (a fixed goal tray that never moves)."""
    stage = get_current_stage()

    UsdGeom.Xform.Define(stage, prim_path)  # plain Xform -> static

    visual_path = prim_path + "/visual"
    add_reference_to_stage(usd_path=usd_path, prim_path=visual_path)
    if visual_offset_z:
        UsdGeom.XformCommonAPI(stage.GetPrimAtPath(visual_path)).SetTranslate(
            (0.0, 0.0, float(visual_offset_z))
        )
    _disable_visual_physics(stage, visual_path)

    _add_box_collider(stage, prim_path, dims)

    return SingleXFormPrim(
        prim_path=prim_path, name=name, position=position, orientation=orientation,
    )


def create_single_rigid_prim_from_usd(
    usd_path: str, prim_path: str, name: str,
    position: np.ndarray, orientation: np.ndarray,
):
    """Legacy raw-USD spawn (kept for backward-compat / reversibility). Spawns
    the USD as a rigid body using whatever collision the USD itself carries --
    this is what produced the triangle-mesh fallback + negative-mass warnings,
    so prefer create_box_collider_rigid / _static above."""
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

    return SingleRigidPrim(
        prim_path=prim_path,
        name=name,
        position=position,
        orientation=orientation,
    )
