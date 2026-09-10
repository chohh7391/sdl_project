import pathlib
import sys
import unittest


SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from orchestration.registry import get_environment_spec, get_planner_spec
from orchestration.planner_api import with_explicit_pour_steps


class RegistryTest(unittest.TestCase):
    def test_transfer_defaults_match_verified_path(self):
        env = get_environment_spec("transfer")
        self.assertEqual(env.entities, ("beaker", "flask", "magnet"))
        self.assertEqual(env.movables, ("beaker", "flask"))
        self.assertEqual(
            env.statics, ("table", "goal_region", "stirrer", "magnet")
        )
        self.assertEqual(env.ex_collision, ("pour_region",))

    def test_xdl_transfer_operands_override_defaults(self):
        env = get_environment_spec(
            "transfer", step_attrs={"from_vessel": "flask", "to_vessel": "beaker"}
        )
        self.assertEqual(env.movables, ("flask", "beaker"))

    def test_rearrange_requires_explicit_llm_output(self):
        with self.assertRaises(ValueError):
            get_environment_spec("rearrange")

    def test_robot_alias_resolves_to_one_config(self):
        self.assertEqual(get_planner_spec("ag95"), get_planner_spec("fr5_ag95"))
        # AG95 = 2-finger side grasp on vessels -> full 6-DOF grasp pose.
        self.assertEqual(get_planner_spec("ag95").grasp_dof, 6)
        # Top-grasp tools stay 4-DOF (yaw about the object axis).
        self.assertEqual(get_planner_spec("vgc10").grasp_dof, 4)
        self.assertEqual(get_planner_spec("dh3").grasp_dof, 4)

    def test_pour_is_an_explicit_step_after_surface_motion(self):
        plan = [
            {"type": "trajectory", "op_name": "Move_to_Surface"},
            {"type": "trajectory", "op_name": "Place"},
        ]
        expanded = with_explicit_pour_steps(plan)
        self.assertEqual([step["type"] for step in expanded], ["trajectory", "pour", "trajectory"])
        self.assertEqual(expanded[1]["op_name"], "pouring")


if __name__ == "__main__":
    unittest.main()
