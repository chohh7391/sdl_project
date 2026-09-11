"""Explicit, mostly-stateless contract around the cuTAMP planning core."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PlanningRequest:
    env_name: str
    poses: Mapping[str, Sequence[float]]
    movables: Tuple[str, ...]
    statics: Tuple[str, ...]
    ex_collision: Tuple[str, ...]
    q_init: Tuple[float, ...]
    rearrange_grid: str = ""
    experiment_id: Optional[str] = None


@dataclass(frozen=True)
class PlanningResult:
    plan: Any
    success: bool
    total_num_satisfying: int
    attempts: int
    planning_time_s: float
    failure_reason: str = ""


def with_explicit_pour_steps(plan: Any) -> Any:
    """Return a plan where pouring is an explicit executor step.

    cuTAMP labels the trajectory that reaches the vessel rim
    ``Move_to_Surface``.  The legacy executor inferred a pour later by watching
    for an operator-name transition.  This boundary adapter makes that control
    flow data: a pour step immediately follows each final trajectory in a
    contiguous Move_to_Surface block.
    """
    if not plan:
        return plan
    result = []
    for index, step in enumerate(plan):
        result.append(step)
        if step.get("op_name") != "Move_to_Surface":
            continue
        nxt = plan[index + 1] if index + 1 < len(plan) else None
        next_op = nxt.get("op_name") if nxt else None
        if next_op == "Move_to_Surface":
            continue
        # The motion solver may already have emitted the pour as an explicit
        # lip-pivot joint path; injecting a bare pour step as well would pour
        # twice. Only synthesise one when there is no path.
        if nxt is not None and nxt.get("type") == "pour_path":
            continue
        result.append({"type": "pour", "op_name": "pouring"})
    return result
