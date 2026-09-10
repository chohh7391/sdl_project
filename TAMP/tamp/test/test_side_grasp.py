"""Invariants of the vessel SIDE grasp (cutamp.samplers.grasp_side_sampler).

These encode *why* Transfer uses a 6-DOF side grasp rather than the 4-DOF top
grasp, so the choice cannot be silently reverted:

  * the pour joint (joint 6) rotates the tool about the end-effector approach
    axis, so only a side grasp turns that rotation into a vessel tilt;
  * the grasp must sit on the vessel body with the mouth left clear.

Needs the cuTAMP/cuRobo stack (conda `sdl`) and a GPU, like the planner itself.
"""
import unittest

import numpy as np
import torch
from curobo.geom.types import Cuboid
from curobo.types.base import TensorDeviceType

from cutamp.envs.utils import unit_quat
from cutamp.robots import load_fr5_ag95_container
from cutamp.samplers import grasp_side_sampler
from cutamp.utils.common import action_4dof_to_mat4x4, action_6dof_to_mat4x4

BEAKER_DIMS = [0.05, 0.05, 0.135]   # TAMP/tamp/src/envs/utils.py ENTITIES["beaker"]


def _beaker():
    return Cuboid(name="beaker", pose=[0.0, 0.0, 0.0, *unit_quat], dims=list(BEAKER_DIMS))


def _rot_about_z(theta):
    m = torch.eye(4)
    m[0, 0] = m[1, 1] = float(np.cos(theta))
    m[0, 1] = float(-np.sin(theta))
    m[1, 0] = float(np.sin(theta))
    return m


class SideGraspTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ta = TensorDeviceType()
        cls.container = load_fr5_ag95_container(cls.ta)
        cls.gripper_spheres = cls.container.gripper_spheres
        cls.tool_from_ee = cls.container.tool_from_ee.cpu()
        cls.obj = _beaker()
        cls.grasps = grasp_side_sampler(2048, cls.obj, cls.gripper_spheres)
        cls.mats = action_6dof_to_mat4x4(cls.grasps).cpu()

    def test_grasp_axis_is_on_the_vessel_axis(self):
        """The pinch point is the vessel's own axis, so the fingers close across it."""
        self.assertLess(self.mats[:, :2, 3].abs().max().item(), 1e-6)

    def test_grasp_height_keeps_the_gripper_on_the_body(self):
        """No sample reaches above the rim or below the base."""
        g_x = self.gripper_spheres[:, 0]
        g_r = self.gripper_spheres[:, 3]
        below = float((g_x + g_r).max().item())
        above = float((g_r - g_x).max().item())
        half_z = BEAKER_DIMS[2] / 2
        h = self.mats[:, 2, 3]
        self.assertGreaterEqual(float(h.min()), -half_z + below)
        self.assertLessEqual(float(h.max()), half_z - above)

    def test_frame_reaches_in_from_above_with_a_horizontal_finger_axis(self):
        """+y tangential and horizontal (fingers close across the vessel); +z the
        approach axis, tilted up into the measured inclination band; +x mostly down."""
        from cutamp.samplers import BETA_MAX, BETA_MIN
        tool_x, tool_y, tool_z = self.mats[:, :3, 0], self.mats[:, :3, 1], self.mats[:, :3, 2]
        self.assertLess(tool_y[:, 2].abs().max().item(), 1e-5)          # fingers close horizontally
        beta = torch.asin(tool_z[:, 2].clamp(-1.0, 1.0))                # inclination above horizontal
        self.assertGreaterEqual(float(beta.min()), BETA_MIN - 1e-5)
        self.assertLessEqual(float(beta.max()), BETA_MAX + 1e-5)
        self.assertLess(float(tool_x[:, 2].max()), 0.0)                 # +x points downward
        # right-handed and orthonormal
        self.assertLess((torch.cross(tool_x, tool_y, dim=1) - tool_z).abs().max().item(), 1e-5)

    def test_azimuth_covers_the_whole_circle(self):
        """Reachability comes from being free to approach from any side."""
        phi = torch.atan2(self.mats[:, 1, 2], self.mats[:, 0, 2])
        hist, _ = np.histogram(phi.numpy(), bins=12, range=(-np.pi, np.pi))
        self.assertTrue((hist > 0).all(), f"azimuth gaps in sampled grasps: {hist}")

    def _vessel_tilt_after_pour_joint(self, obj_from_tool, theta):
        """Vessel starts upright; rotate the ee about its own +z (the pour joint)."""
        world_from_ee = obj_from_tool @ self.tool_from_ee            # world_from_obj = I
        world_from_ee = world_from_ee @ _rot_about_z(theta)
        world_from_obj = (world_from_ee
                          @ torch.linalg.inv(self.tool_from_ee)
                          @ torch.linalg.inv(obj_from_tool))
        return float(np.degrees(np.arccos(np.clip(world_from_obj[2, 2].item(), -1.0, 1.0))))

    def test_pour_joint_tilts_the_vessel_for_side_but_not_top_grasp(self):
        """The whole point of the side grasp: the pour joint must actually tilt the
        vessel. Under a top grasp the joint axis IS the vessel axis, so it cannot."""
        top = action_4dof_to_mat4x4(torch.tensor([[0.0, 0.0, BEAKER_DIMS[2] / 2 - 0.02, 0.0]]))[0]
        for deg in (10.0, 30.0, 60.0):
            self.assertAlmostEqual(
                self._vessel_tilt_after_pour_joint(top, float(np.radians(deg))), 0.0, places=3,
                msg="a top grasp cannot pour: the pour joint spins the vessel about its own axis")
        for side in self.mats[:16]:
            tilts = [self._vessel_tilt_after_pour_joint(side, float(np.radians(d)))
                     for d in (0.0, 10.0, 30.0, 60.0)]
            # upright before the pour (0.05 deg absorbs float32 euler round-trip noise)
            self.assertLess(tilts[0], 0.05)
            self.assertEqual(tilts, sorted(tilts), f"tilt must grow with the pour joint: {tilts}")
            # cos(beta) efficiency at worst; 60 deg of joint must give a real pour
            self.assertGreater(tilts[-1], 40.0, f"pour tilt too small: {tilts}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
