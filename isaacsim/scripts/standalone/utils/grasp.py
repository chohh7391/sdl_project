"""Fixed-joint grasp attach/release (REFACTOR.md III-1).

Author decision (2026-09-01): grasping is modelled as a TEMPORARY rigid
attachment, not a friction grasp. On gripper CLOSE, the movable object nearest
the end-effector (within GRASP_THRESHOLD_M) is welded to the gripper link with a
UsdPhysics.FixedJoint whose local frames lock the CURRENT relative pose (so the
object does not snap). On OPEN the joint prim is removed and the object is free
again. This is deterministic and robust across object placements (PLAN.md D3);
the finger position-close in simulation.py stays for visual realism.

Only parallel grippers (ag95 / dh3) use this path. vgc10 is a suction tool that
already has its own SurfaceGripper attach, so it must NOT get a fixed joint.

The relative-pose math uses the clean world poses returned by
SingleRigidPrim.get_world_pose() -> (position, quaternion), quaternion in
scalar-first (w, x, y, z) convention. Because get_world_pose() returns a
scale-free position + unit quaternion, no scale contamination can leak into the
joint frames.
"""

import numpy as np
from pxr import Gf, Sdf, UsdPhysics

# Stage path of the (single) temporary grasp joint. Kept under /World so it is
# swept away by world.clear() on a tool change / rebuild; simulation.py also
# detaches explicitly before a rebuild so the tracking state never goes stale.
GRASP_JOINT_PATH = "/World/GraspFixedJoint"

# Grasp threshold: an object qualifies for attach only if its rigid-body origin
# is within this distance of the end-effector link origin.
#
# Rationale (justified against the real scene geometry, not guessed):
#   * The end-effector link is the FINGER-TIP link (ag95:
#     gripper_finger2_finger_tip_link, dh3: finger3_tip_link), while an object's
#     reported pose is its box-collider CENTER (utils/object.py). During a real
#     grasp the finger tip sits at the object's periphery, so the tip-to-center
#     distance is on the order of the object's own half-extent. The largest
#     movable object is the box (cuTAMP dims 0.108 x 0.108 x 0.08 m -> in-plane
#     half-diagonal ~0.076 m, half-height 0.04 m). 0.12 m covers that periphery
#     offset with margin.
#   * It must stay well below the inter-object spacing so only the truly-grasped
#     object qualifies and a mis-fire cannot grab a neighbour. Objects live on a
#     workspace grid whose cells are >= 0.30 m apart (task.get_grid_xy), so a
#     0.12 m ball around the finger tip can contain at most the intended object.
#   * If nothing is within 0.12 m the grasp fails closed (attach nothing), which
#     is the intended fail-closed behaviour (REFACTOR III-2).
GRASP_THRESHOLD_M = 0.12


def _quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=float)


def _quat_rotate(q, v):
    """Rotate vector v by quaternion q (w, x, y, z)."""
    w, x, y, z = q
    u = np.array([x, y, z], dtype=float)
    v = np.asarray(v, dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def relative_pose(parent_pos, parent_quat, child_pos, child_quat):
    """Pose of the child frame expressed in the parent frame.

    Returns (rel_pos (3,), rel_quat (4,) w-first, unit). Both input quaternions
    are (w, x, y, z).
    """
    parent_pos = np.asarray(parent_pos, dtype=float)
    child_pos = np.asarray(child_pos, dtype=float)
    q_pinv = _quat_conjugate(parent_quat)
    rel_pos = _quat_rotate(q_pinv, child_pos - parent_pos)
    rel_quat = _quat_mul(q_pinv, np.asarray(child_quat, dtype=float))
    n = np.linalg.norm(rel_quat)
    if n > 0.0:
        rel_quat = rel_quat / n
    return rel_pos, rel_quat


def nearest_object(ee_pos, poses):
    """Pick the object whose position is nearest the end-effector.

    poses: dict name -> (position (3,), quaternion (4,)).
    Returns (name, position, quaternion, distance) for the nearest object, or
    None if `poses` is empty.
    """
    ee_pos = np.asarray(ee_pos, dtype=float)
    best = None
    for name, (pos, quat) in poses.items():
        d = float(np.linalg.norm(np.asarray(pos, dtype=float) - ee_pos))
        if best is None or d < best[3]:
            best = (name, pos, quat, d)
    return best


def create_grasp_fixed_joint(
    stage, joint_path,
    body0_path, body0_pos, body0_quat,
    body1_path, body1_pos, body1_quat,
):
    """Author a UsdPhysics.FixedJoint welding body1 (object) to body0 (gripper
    link) at their CURRENT relative pose, so nothing snaps.

    body0 = gripper end-effector link (articulation link).
    body1 = object rigid body.

    The joint frame is chosen to coincide with the object frame: localFrame1 is
    identity, and localFrame0 is the object pose expressed in the ee-link frame.
    With both frames evaluated through the bodies' current world poses they land
    on the same world frame, so PhysX creates the joint at zero error (no snap).
    excludeFromArticulation=True keeps it a plain maximal-coordinate joint
    between the arm articulation and the free object (not folded into the arm's
    articulation, which would be an unsupported runtime topology change).
    """
    rel_pos, rel_quat = relative_pose(body0_pos, body0_quat, body1_pos, body1_quat)

    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(joint_path))
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(rel_pos[0]), float(rel_pos[1]), float(rel_pos[2])))
    joint.CreateLocalRot0Attr().Set(
        Gf.Quatf(float(rel_quat[0]), float(rel_quat[1]), float(rel_quat[2]), float(rel_quat[3]))
    )
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateExcludeFromArticulationAttr().Set(True)
    return joint_path


def remove_grasp_fixed_joint(stage, joint_path):
    """Remove the temporary grasp joint prim if present. Returns True if a prim
    was actually removed."""
    if joint_path is None:
        return False
    prim = stage.GetPrimAtPath(joint_path)
    if prim and prim.IsValid():
        stage.RemovePrim(Sdf.Path(joint_path))
        return True
    return False
