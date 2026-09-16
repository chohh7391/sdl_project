"""The real-hardware server is a subclass, and these keep it one.

A forked copy of `tamp_server.py` would have let the simulated and physical
paths drift apart -- most damagingly in the pour law, which the paper reports
as one controller. Subclassing instead moves the risk somewhere narrower: an
override whose base method got renamed becomes dead code that is never called
again, and a plan step type added upstream could go unhandled on hardware.
Both are what these check.

Needs the cuTAMP/cuRobo stack (conda `sdl`), like the planner itself, because
importing the server imports the planner.
"""

import ast
import inspect
import os
import sys
import textwrap

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'server'))

import tamp_real_server                                     # noqa: E402
import tamp_server                                          # noqa: E402

BASE = tamp_server.TAMPServer
REAL = tamp_real_server.RealTAMPServer

#: Methods the real server is MEANT to replace. Everything else it defines has
#: to be new, and everything here has to still exist on the base.
INTENDED_OVERRIDES = {
    '_arm_q',
    '_execute_trajectory_step',
    '_finish_execution',
    '_perception_pose',
    '_publish_arm_command',
    '_start_execution',
    'execute_gripper_action',
    'set_tamp_env_cb',
}

#: Methods that exist only on the real server.
NEW_METHODS = {
    '__init__',
    '_check_estop',
    '_estop_cb',
    '_grounded_z',
    '_load_static_poses',
    '_load_table_top',
    '_on_object_pose',
}


def _methods(cls):
    return {name for name, value in vars(cls).items() if inspect.isfunction(value)}


def test_every_override_still_targets_a_base_method():
    # A base rename turns an override into dead code: the subclass keeps its
    # method, the base calls its own, and hardware silently runs the simulated
    # behaviour. Nothing else would catch that.
    missing = sorted(name for name in INTENDED_OVERRIDES if not hasattr(BASE, name))
    assert not missing, f'{missing} no longer exist on TAMPServer'


def test_the_real_server_declares_what_it_replaces():
    defined = _methods(REAL)
    assert defined == INTENDED_OVERRIDES | NEW_METHODS, (
        'RealTAMPServer defines %s; update INTENDED_OVERRIDES / NEW_METHODS and '
        'say which it is' % sorted(defined ^ (INTENDED_OVERRIDES | NEW_METHODS)))


def test_a_new_method_does_not_silently_shadow_the_base():
    shadowed = sorted(name for name in NEW_METHODS
                      if name != '__init__' and hasattr(BASE, name))
    assert not shadowed, f'{shadowed} are declared new but exist on the base'


def test_the_plan_loop_is_shared():
    # The dispatch stays in one place, so a step type added upstream cannot be
    # unhandled on hardware -- it reaches the same hooks.
    assert REAL.execute_plan_cb is BASE.execute_plan_cb


def test_the_pour_laws_are_inherited_verbatim():
    # The claim the paper makes is about ONE adaptive-pour controller. Two
    # implementations of it would be two results.
    for name in ('pouring', 'pouring_along_path', '_untilt_pour_joint',
                 '_pour_shaping_kernel'):
        assert getattr(REAL, name) is getattr(BASE, name), name


def test_the_dispatched_step_types_match_the_declared_ones():
    tree = ast.parse(textwrap.dedent(inspect.getsource(BASE.execute_plan_cb)))
    dispatched = {
        node.comparators[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name) and node.left.id == 'plan_type'
        and isinstance(node.comparators[0], ast.Constant)
    }
    assert dispatched == set(BASE.PLAN_STEP_TYPES)


def _calls_on_self(func):
    """Names of `self.<name>(...)` calls made in *func*'s own body."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == 'self'
    }


def test_both_world_state_paths_localize_through_the_recovery_ladder():
    # THE drift this file exists for, in its sharpest form. The real server
    # overrides BOTH _perception_pose and set_tamp_env_cb, so a recovery ladder
    # written into the base's set_tamp_env_cb would never run on hardware --
    # and nothing would say so, because the base's copy still works in
    # simulation. Both implementations must go through the shared entry point.
    for cls in (BASE, REAL):
        calls = _calls_on_self(cls.set_tamp_env_cb)
        assert '_localize_with_recovery' in calls, (
            '%s.set_tamp_env_cb does not localize through the recovery ladder'
            % cls.__name__)
        assert '_perception_pose' not in calls, (
            '%s.set_tamp_env_cb calls _perception_pose directly, which skips '
            'the recovery ladder' % cls.__name__)


def test_the_recovery_ladder_itself_is_shared():
    # One ladder, reached through the plant hooks -- so the physical cell
    # recovers under the policy the reported numbers were measured under.
    for name in ('_localize_with_recovery', '_poll_perception_pose',
                 '_recovery_retreat', '_recovery_scan', '_log_recovery'):
        assert getattr(REAL, name) is getattr(BASE, name), name


def test_both_plants_track_whether_the_gripper_is_holding():
    # The ladder refuses to move an arm that is holding a vessel. That check is
    # only as good as _holding, which each plant's gripper hook has to set:
    # setting it on the base alone would leave the REAL arm sweeping with a
    # full vessel in the gripper.
    for cls in (BASE, REAL):
        tree = ast.parse(textwrap.dedent(
            inspect.getsource(cls.execute_gripper_action)))
        assigns_holding = any(
            isinstance(node, ast.Attribute) and node.attr == '_holding'
            and isinstance(node.ctx, ast.Store)
            for node in ast.walk(tree))
        assert assigns_holding, (
            '%s.execute_gripper_action never sets self._holding' % cls.__name__)


def test_the_scan_rung_is_off_until_there_is_a_camera_on_the_arm():
    # This cell has two fixed cameras and no wrist camera, so sweeping the arm
    # observes the same two viewpoints it started from. Defaulting the rung on
    # would spend tens of seconds of motion per failed localization and recover
    # nothing.
    assert BASE.RECOVERY_SCAN is False


def test_the_real_server_has_no_planner_config_of_its_own():
    source = inspect.getsource(tamp_real_server)
    assert 'TAMPConfiguration' not in source, (
        'the real server must run default_config(); a second copy of the '
        'particle/step counts is a second set of results')
