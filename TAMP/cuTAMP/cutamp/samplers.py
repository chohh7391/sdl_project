# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Optional

import roma
import torch
from curobo.geom.types import Obstacle, Cuboid, Mesh
from jaxtyping import Float

from cutamp.utils.common import approximate_goal_aabb, pose_list_to_mat4x4, transform_points
from cutamp.utils.shapes import MultiSphere

# Side-grasp approach inclination band (radians above horizontal). See
# `grasp_side_sampler` for the measurement that picked it.
BETA_MIN = 10.0 * torch.pi / 180.0
BETA_MAX = 35.0 * torch.pi / 180.0

Grasp4DOF = Place4DOF = Float[torch.Tensor, "n 4"]
Grasp6DOF = Place6DOF = Float[torch.Tensor, "n 6"]


def sample_yaw(num_samples: int, num_faces: Optional[int], device: torch.device):
    """Sample yaws. Continuous if num_faces is None, otherwise discrete."""
    if num_faces is None:
        yaw = torch.rand(num_samples, device=device) * 2 * torch.pi  # [0, 2pi)
    else:
        assert num_faces >= 1
        two_pi = 2 * torch.pi
        endpoint = two_pi - (two_pi / num_faces)
        yaw_choices = torch.linspace(0, endpoint, num_faces, device=device)
        yaw_idxs = torch.randint(0, num_faces, (num_samples,), device=device)
        yaw = yaw_choices[yaw_idxs]
    return yaw


def sample_stick_grasps(num_samples: int, stick: MultiSphere) -> Grasp4DOF:
    """Sample 4-DOF grasps for a stick."""
    spheres = stick.spheres
    if not (spheres[:, 1:3] == 0.0).all():
        raise ValueError(f"Expected stick spheres to have y and z positions of 0")

    # Randomly sample x-coordinate of the sphere
    sphere_x = spheres[:, 0]
    x_idxs = torch.randint(0, len(sphere_x), (num_samples,), device=spheres.device)
    sampled_x = sphere_x[x_idxs]

    # Sample yaws, use two faces since we just want original orientation and mirrored 180 degrees
    yaw = sample_yaw(num_samples, num_faces=2, device=spheres.device)

    # Create 4-DOF grasp!
    grasp_4dof = torch.zeros((num_samples, 4), device=spheres.device)
    grasp_4dof[:, 0] = sampled_x
    grasp_4dof[:, 3] = yaw
    return grasp_4dof


def grasp_4dof_sampler(
    num_samples: int, obj: Obstacle, obj_spheres: Float[torch.Tensor, "n 4"], num_faces: Optional[int] = None
) -> Grasp4DOF:
    """
    Sample 4-DOF grasps for the given object in the object's coordinate frame.
    This could be made a lot more sophisticated.
    """
    # Handle the stick as a special case
    if obj.name == "stick":
        obj: MultiSphere
        return sample_stick_grasps(num_samples, obj)

    # Determine point to grasp from the top of the object
    if isinstance(obj, Cuboid):
        obj_half_z = max(0.0, obj.dims[2] / 2 - 0.02)  # grasp 2cm from top of object
    elif isinstance(obj, MultiSphere):
        min_z = obj.spheres[:, 2].min()
        max_z = obj.spheres[:, 2].max()
        obj_half_z = min_z + (max_z - min_z) * (2.0 / 3.0)
    else:
        max_z = obj_spheres[:, 2].max()
        obj_half_z = max_z - 0.02  # 2cm from top of obejct

    # Assume zero translation for now in x and y-axes
    translation = torch.zeros(num_samples, 3, device=obj.tensor_args.device)
    translation[:, 2] = obj_half_z

    # Sample yaw
    yaw = sample_yaw(num_samples, num_faces, obj.tensor_args.device)

    # Form full 4-DOF grasp
    grasp_4dof = torch.cat([translation, yaw.unsqueeze(-1)], dim=1)
    return grasp_4dof


def grasp_side_sampler(
    num_samples: int,
    obj: Obstacle,
    gripper_spheres: Float[torch.Tensor, "n 4"],
) -> Grasp6DOF:
    """Sample natural SIDE grasps on an upright vessel (2-finger gripper).

    The gripper approaches horizontally, the fingers close across the vessel's
    width, and the mouth stays clear -- so pouring is a wrist rotation. This is
    the grasp the paper describes for Transfer ("2-finger gripper ... to enable
    side-grasp pouring").

    Why this sampler exists instead of the two upstream ones (all measured, see
    REFACTOR.md "side grasp"):

    * ``grasp_4dof_sampler`` is a TOP grasp: yaw-only about the vessel axis. For
      a Cuboid vessel ``sample_yaw(num_faces=4)`` yields exactly 4 distinct
      grasps, so 1024 particles explore 4 poses. Over the 30 randomized seeds
      only 1-3 of those 4 are IK-reachable, and for 4 of the 30 seeds none is --
      a structural planning-failure floor that no particle budget can fix.
    * The pour joint (joint 6) rotates about the end-effector approach axis.
      Under a top grasp that axis IS the vessel axis, so the pour spins the
      vessel about itself and tilts it by exactly 0 deg at any joint angle;
      under a side grasp the tilt tracks the joint 1:1.
    * ``grasp_6dof_sampler`` is the upstream bookshelf sampler (its own
      docstring says so): it draws pitch uniformly over 2*pi and samples the
      grasp point *inside* the box, which does not describe a vessel grasp.

    Three continuous parameters:
      * azimuth ``phi ~ U[-pi, pi)`` -- which side to approach from,
      * inclination ``beta`` -- how far ABOVE horizontal the gripper sits, i.e.
        reaching in and down onto the vessel's side the way a hand does,
      * height ``h`` -- where on the body to pinch, bounded so the gripper stays
        on the vessel body (clear of the rim and of the support surface).

    ``beta`` is not cosmetic. cuTAMP constrains the grasp pose but nothing
    constrains the 5 cm pre-grasp pose that cuRobo then has to reach, and a purely
    horizontal approach retreats radially straight out of the workspace. Measured
    over the 30 seeds, grasp+pre-grasp IK both succeed for 46.9% of samples at
    beta=0 but 69.2% at beta=30 deg (81.6-82.1% for the grasp alone), so the band
    below is centred there. A 180 deg wrist flip was measured too and did not
    help, so it is deliberately not sampled.

    Tool frame of a sample: ``+z`` is the approach axis, radially outward and
    tilted up by ``beta`` (so the gripper body ends up outside and above the
    vessel); ``+y`` is tangential and stays horizontal (the finger-opening axis,
    so the fingers close across the vessel); ``+x`` completes the frame, pointing
    mostly down. The origin lies on the vessel axis, matching the 4-DOF
    convention where the tool origin is the grasp centre, not the fingertip.
    """
    if not isinstance(obj, Cuboid):
        raise ValueError(f"Side grasps expect a Cuboid vessel, got {type(obj)} for {obj.name!r}")
    device = obj.tensor_args.device
    half_z = float(obj.dims[2]) / 2.0

    phi = (torch.rand(num_samples, device=device) * 2.0 - 1.0) * torch.pi
    beta = BETA_MIN + (BETA_MAX - BETA_MIN) * torch.rand(num_samples, device=device)
    cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
    cos_beta, sin_beta = torch.cos(beta), torch.sin(beta)

    zeros = torch.zeros_like(phi)
    radial = torch.stack([cos_phi, sin_phi, zeros], dim=1)               # outward, horizontal
    up = torch.zeros_like(radial); up[:, 2] = 1.0
    tool_y = torch.stack([-sin_phi, cos_phi, zeros], dim=1)              # tangential: fingers close along this
    tool_z = cos_beta.unsqueeze(1) * radial + sin_beta.unsqueeze(1) * up  # approach axis, tilted up by beta
    tool_z = tool_z / tool_z.norm(dim=1, keepdim=True)
    tool_x = torch.cross(tool_y, tool_z, dim=1)                          # completes a right-handed frame
    rotmat = torch.stack([tool_x, tool_y, tool_z], dim=2)
    rpy = roma.rotmat_to_euler("XYZ", rotmat)

    # Vertical room the gripper needs around the pinch point, per sample: a
    # gripper sphere at tool coords (a, b, c) with radius r sits at object-frame
    # z = h - a*cos(beta) + c*sin(beta) +- r. Azimuth does not enter, but beta
    # does, so the band is computed per sample rather than once.
    g_a = gripper_spheres[:, 0].unsqueeze(0)
    g_c = gripper_spheres[:, 2].unsqueeze(0)
    g_r = gripper_spheres[:, 3].unsqueeze(0)
    offset = -g_a * cos_beta.unsqueeze(1) + g_c * sin_beta.unsqueeze(1)   # [num_samples, n_spheres]
    above = (offset + g_r).max(dim=1).values
    below = (g_r - offset).max(dim=1).values
    margin = 0.005
    h_lo = -half_z + below + margin
    h_hi = half_z - above - margin
    # Vessel too short to hold the gripper inside its body: pinch at mid-height and
    # let the collision constraints reject it if it is genuinely infeasible, rather
    # than silently sampling a grasp off the object.
    degenerate = h_hi < h_lo
    h_lo = torch.where(degenerate, torch.zeros_like(h_lo), h_lo)
    h_hi = torch.where(degenerate, torch.zeros_like(h_hi), h_hi)
    h = h_lo + (h_hi - h_lo) * torch.rand(num_samples, device=device)

    translation = torch.stack([zeros, zeros, h], dim=1)      # on the vessel axis
    return torch.cat([translation, rpy], dim=1)


def _grasp_6dof_sampler_for_multisphere(num_samples: int, obj: MultiSphere) -> Grasp6DOF:
    """
    Sample 6-DOF side grasps for a MultiSphere object, assuming it's cylindrical like a beaker.
    This sampler generates horizontal grasps around the object.
    """
    tensor_args = obj.tensor_args
    spheres = obj.spheres

    # Find the vertical bounds of the object
    min_z = spheres[:, 2].min()
    max_z = spheres[:, 2].max()

    # Sample a grasp height, avoiding the very top and bottom
    grasp_z = min_z + (max_z - min_z) * 0.75
    # grasp_z = 0.02
    z = torch.full((num_samples,), grasp_z, device=tensor_args.device)

    # Find the maximum radius of the object to determine the grasp distance from the center
    radius = torch.sqrt(spheres[:, 0] ** 2 + spheres[:, 1] ** 2).max()

    # Sample a random angle around the object's z-axis
    angle = torch.rand(num_samples, device=tensor_args.device) * 2 * torch.pi

    # Position the grasp on the surface of the object
    x = radius * torch.cos(angle)
    y = radius * torch.sin(angle)
    translation = torch.stack([x, y, z], dim=1)

    roll = torch.full((num_samples,), torch.pi / 2, device=tensor_args.device)

    pitch = torch.zeros(num_samples, device=tensor_args.device)
    # Yaw is determined by the angle, with an offset to point the z-axis inward.
    # yaw = angle + torch.pi
    yaw = torch.rand(num_samples, device=obj.tensor_args.device) * 2 * torch.pi
    rpy = torch.stack([roll, pitch, yaw], dim=1)

    grasp_6dof = torch.cat([translation, rpy], dim=1)
    return grasp_6dof


def grasp_6dof_sampler(num_samples: int, obj: Obstacle, num_faces: Optional[int] = None) -> Grasp6DOF:
    """
    Sample 6-DOF grasps for the given object in the object's coordinate frame.
    Note: this is a very simple sampler which was written for the bookshelf domain and isn't general enough.
    """

    # if isinstance(obj, MultiSphere):
    #     # This path is for beaker-like objects in environments like 'pouring'
    #     return _grasp_6dof_sampler_for_multisphere(num_samples, obj)
    
    # assert isinstance(obj, Cuboid), "only Cuboid objects supported for 6-dof grasps right now"
    # Sample roll from discrete choices
    roll_choices = torch.tensor(
        # [-torch.pi / 2, -torch.pi / 3, -torch.pi / 4, torch.pi / 4, torch.pi / 3, torch.pi / 2],
        [-torch.pi / 2, -torch.pi / 2.3, torch.pi / 2.3, torch.pi / 2],
        # [-torch.pi / 2, torch.pi / 2],
        device=obj.tensor_args.device,
    )
    roll_idxs = torch.randint(0, len(roll_choices), (num_samples,), device=obj.tensor_args.device)
    roll = roll_choices[roll_idxs]

    # Let pitch be zero for now
    # pitch = torch.zeros(num_samples, device=obj.tensor_args.device)
    pitch = torch.rand(num_samples, device=obj.tensor_args.device) * 2 * torch.pi

    # Sample yaw from discrete choices
    yaw_choices = torch.tensor([-torch.pi / 2, torch.pi / 2], device=obj.tensor_args.device)
    # yaw_choices = torch.tensor([0, torch.pi, -torch.pi/2, torch.pi/2], device=obj.tensor_args.device)
    # yaw_choices = torch.rand(num_samples, device=obj.tensor_args.device) * 2 * torch.pi
    yaw_idxs = torch.randint(0, 2, (num_samples,), device=obj.tensor_args.device)
    yaw = yaw_choices[yaw_idxs]

    # Stack rpy
    rpy = torch.stack([roll, pitch, yaw], dim=1)

    # Compute offsets for gripper translation in object frame
    half_extents = obj.tensor_args.to_device([dim / 2 for dim in obj.dims])
    gripper_offset = 0.01
    upper = (half_extents - gripper_offset).clamp(min=0.0)
    lower = (obj.tensor_args.to_device(3 * [gripper_offset])).clamp(max=upper)
    lower[0] = upper[0] = 0.0  # remove translation in x-axis

    # Sample translation between bounds
    translation = torch.rand(num_samples, 3, device=obj.tensor_args.device)
    translation = lower + (upper - lower) * translation

    # Form 6-DOF grasps
    grasp_6dof = torch.cat([translation, rpy], dim=1)
    return grasp_6dof


def place_4dof_sampler(
    num_samples: int, obj: Obstacle, obj_spheres: Float[torch.Tensor, "n 4"], surface: Obstacle
) -> Place4DOF:
    """Sample 4-DOF placement poses in the world frame. This does not yet fully support surfaces with yaw."""
    if not isinstance(surface, (Cuboid, Mesh)):
        raise NotImplementedError(f"Only Cuboid or Mesh surfaces supported for now, not {type(surface)}")

    # Determine the z-position of the min of the object spheres (in object frame)
    sph_bottom = obj_spheres[:, 2] - obj_spheres[:, 3]
    obj_bottom = sph_bottom.min()  # used as a delta
    obj_z_delta = -obj_bottom

    # Assume the surface is a cuboid, sample xy positions within AABB in local frame
    if isinstance(surface, Cuboid):
        aabb_xy = surface.tensor_args.to_device(
            [[-surface.dims[0] / 2, -surface.dims[1] / 2], [surface.dims[0] / 2, surface.dims[1] / 2]]
        )
        surface_z = surface.dims[2] / 2
    else:
        # Note: this was only used for the Rummy demos in the past, so not tested extensively.
        aabb = approximate_goal_aabb(surface).to(obj.tensor_args.device)
        aabb_xy = aabb[:, :2]
        surface_z = aabb[1, 2]

    xy = torch.rand(num_samples, 2, device=obj.tensor_args.device)
    xy = aabb_xy[0] + xy * (aabb_xy[1] - aabb_xy[0])

    # TODO: consider collision activation distance
    # Sample z-offset and combine with xy
    z_lower, z_upper = 1e-3, 1e-2
    z = torch.rand(num_samples, 1, device=obj.tensor_args.device)
    z = z_lower + (z_upper - z_lower) * z
    z += obj_z_delta + surface_z
    xyz = torch.cat([xy, z], dim=1)

    # Transform to surface coordinate frame
    if isinstance(surface, Cuboid):
        surface_mat4x4 = pose_list_to_mat4x4(surface.pose).to(obj.tensor_args.device)
        xyz_surface = transform_points(xyz, surface_mat4x4)
    else:
        xyz_surface = xyz

    # Sample yaw
    yaw = sample_yaw(num_samples, None, obj.tensor_args.device)
    place_4dof = torch.cat([xyz_surface, yaw.unsqueeze(-1)], dim=1)
    return place_4dof
