"""Placement geometry invariants for the transfer environment.

Two things that a batch's placement stability depends on, and that were both
silently wrong until measured (see REFACTOR.md, 2026-09-09):

1. The planner's world is the simulator's world raised by PLANNER_Z_LIFT for every
   movable, so a placement surface has to sit one lift above the real support or
   the vessel is released in mid-air and dropped onto it.
2. "On a region" has to mean the object's footprint is inside the region, not that
   its collision-sphere centres are -- otherwise a vessel can be placed with half
   of itself overhanging and a few millimetres of tracking error put it outside.
"""

import copy

import torch

from cutamp.costs import dist_from_bounds_inset, dist_from_bounds_jit
from envs.constants import PLANNER_Z_LIFT
from envs.utils import ENTITIES


def _transfer_entities():
    """ENTITIES with the movable poses the transfer env needs, lift applied."""
    from envs.transfer import load_transfer_env

    entities = copy.deepcopy(ENTITIES)
    entities["beaker"].pose = [0.50, 0.15, 0.0675 + PLANNER_Z_LIFT, 1.0, 0.0, 0.0, 0.0]
    entities["flask"].pose = [0.50, 0.35, 0.0600 + PLANNER_Z_LIFT, 1.0, 0.0, 0.0, 0.0]
    movables = [entities["beaker"], entities["flask"]]
    load_transfer_env(entities=entities, movables=movables, statics=[], ex_collision=[])
    return entities


def test_goal_region_top_is_one_lift_above_the_real_support():
    """The vessel's real bottom lands on the table, not PLANNER_Z_LIFT above it."""
    entities = _transfer_entities()
    table = entities["table"]
    goal = entities["goal_region"]
    table_top = table.pose[2] + table.dims[2] / 2.0
    goal_top = goal.pose[2] + goal.dims[2] / 2.0
    assert goal_top == table_top + PLANNER_Z_LIFT, (
        f"goal_region top {goal_top} should be one planner lift above the table top "
        f"{table_top}; a placement on it is PLANNER_Z_LIFT lower in the simulator."
    )


def test_goal_region_xy_is_unchanged():
    """The trials harness scores against this xy (scripts/trials/trial_driver.py)."""
    entities = _transfer_entities()
    assert entities["goal_region"].pose[:2] == [0.35, -0.35]


def test_inset_bounds_require_the_whole_sphere_inside():
    lower = torch.tensor([-0.05, -0.05])
    upper = torch.tensor([0.05, 0.05])
    radii = torch.tensor([[0.025]])
    # Centre on the boundary: inside by the centre-only test, outside once the
    # footprint counts.
    on_boundary = torch.tensor([[0.05, 0.0]])
    assert dist_from_bounds_jit(on_boundary, lower, upper) == 0.0
    assert dist_from_bounds_inset(on_boundary, lower, upper, radii) > 0.0
    # Fully inside: zero under both.
    inside = torch.tensor([[0.02, 0.0]])
    assert dist_from_bounds_inset(inside, lower, upper, radii) == 0.0
    # Exactly tangent to the boundary is the limit case and still counts as inside.
    tangent = torch.tensor([[0.025, 0.0]])
    assert dist_from_bounds_inset(tangent, lower, upper, radii) == 0.0


def test_inset_bounds_collapse_to_the_centre_for_an_oversized_object():
    """An object wider than the surface must degrade to "centred", not to garbage."""
    lower = torch.tensor([-0.01, -0.01])
    upper = torch.tensor([0.01, 0.01])
    radii = torch.tensor([[0.05]])
    at_center = torch.tensor([[0.0, 0.0]])
    assert dist_from_bounds_inset(at_center, lower, upper, radii) == 0.0
    off_center = torch.tensor([[0.02, 0.0]])
    assert dist_from_bounds_inset(off_center, lower, upper, radii) > 0.0
