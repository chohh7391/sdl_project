"""Single source of truth for task environments and robot configurations.

The registry contains orchestration data only.  It deliberately does not import
ROS or cuTAMP, which makes it usable by the interactive, batch, and XDL paths.
Dynamic XDL operands are applied through :func:`EnvironmentSpec.resolve`.
"""

from dataclasses import dataclass, replace
from typing import Mapping, Optional, Tuple


ALL_ENTITIES = (
    "table", "stirrer", "box_goal", "beaker", "flask", "magnet", "box"
)


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    entities: Tuple[str, ...]
    movables: Tuple[str, ...]
    statics: Tuple[str, ...]
    ex_collision: Tuple[str, ...]
    rearrange_grid: str = ""

    def resolve(
        self,
        *,
        step_attrs: Optional[Mapping[str, str]] = None,
        rearrange_info: Optional[Mapping[str, str]] = None,
    ) -> "EnvironmentSpec":
        attrs = step_attrs or {}
        rearrange = rearrange_info or {}

        if self.name == "transfer":
            source = attrs.get("from_vessel", "beaker")
            target = attrs.get("to_vessel", "flask")
            return replace(self, movables=(source, target))
        if self.name == "stir":
            vessel = attrs.get("vessel", "flask")
            other = "beaker" if vessel == "flask" else "flask"
            return replace(
                self,
                movables=(vessel, "magnet"),
                statics=("table", "stirrer", other, "goal_region", "box"),
            )
        if self.name == "move":
            target = attrs.get("object", "box")
            base = ("table", "stirrer", "box", "beaker", "flask", "box_goal")
            return replace(
                self,
                movables=(target,),
                statics=tuple(entity for entity in base if entity != target),
            )
        if self.name == "rearrange":
            target = rearrange.get("target_entity")
            grid = rearrange.get("target_grid")
            if not target or not grid:
                raise ValueError("rearrange requires target_entity and target_grid")
            base = ("table", "stirrer", "beaker", "flask", "box_goal")
            return replace(
                self,
                movables=(target,),
                statics=tuple(entity for entity in base if entity != target),
                rearrange_grid=grid,
            )
        return self


@dataclass(frozen=True)
class PlannerSpec:
    tool: str
    robot: str
    grasp_dof: int
    time_dilation_factor: float = 0.5
    num_particles: int = 1024
    num_resampling_attempts: int = 100
    num_opt_steps: int = 1000
    num_initial_plans: int = 1
    approach: str = "optimization"


_ENVIRONMENTS = {
    "transfer": EnvironmentSpec(
        # This is the exact world used by the verified transfer 5/5 path.
        # Do not silently add the XDL parser's legacy box/rearrange_region here:
        # doing so changes the collision problem and invalidates comparisons.
        "transfer", ("beaker", "flask", "magnet"), ("beaker", "flask"),
        ("table", "goal_region", "stirrer", "magnet"), ("pour_region",),
    ),
    "stir": EnvironmentSpec(
        "stir", ALL_ENTITIES, ("flask", "magnet"),
        ("table", "stirrer", "beaker", "goal_region", "box"),
        ("beaker_region", "rearrange_region"),
    ),
    "default": EnvironmentSpec(
        "default", ALL_ENTITIES, ("magnet",),
        ("table", "stirrer", "beaker", "flask", "box"), (),
    ),
    "move": EnvironmentSpec(
        "move", ALL_ENTITIES, ("box",),
        ("table", "stirrer", "beaker", "flask", "box_goal"),
        ("box_region", "rearrange_region"),
    ),
    "rearrange": EnvironmentSpec(
        "rearrange", ALL_ENTITIES, (), (),
        ("pour_region", "beaker_region", "box_region", "rearrange_region"),
    ),
}

_PLANNERS = {
    "empty": PlannerSpec("empty", "fr5", 4),
    # AG95 is the 2-finger gripper and grasps vessels from the SIDE, which needs
    # the full 6-DOF grasp parameterization (`cutamp.samplers.grasp_side_sampler`).
    # A 4-DOF grasp is yaw-only about the vessel axis, i.e. a top grasp: it puts
    # the fingers over the mouth and -- measured -- leaves the vessel tilt at
    # exactly 0 deg for any pour-joint angle, so it cannot pour at all. It is also
    # less robust: the 4-DOF family has only 4 distinct members for a Cuboid
    # vessel, and over the 30 randomized seeds 4 of them admit none that is
    # IK-reachable (vs 30/30 for the side family).
    "ag95": PlannerSpec("ag95", "fr5_ag95", 6),
    "vgc10": PlannerSpec("vgc10", "fr5_vgc10", 4),
    "dh3": PlannerSpec("dh3", "fr5_dh3", 4),
}


def get_environment_spec(
    name: str,
    *,
    step_attrs: Optional[Mapping[str, str]] = None,
    rearrange_info: Optional[Mapping[str, str]] = None,
) -> EnvironmentSpec:
    key = name.strip().lower()
    try:
        spec = _ENVIRONMENTS[key]
    except KeyError as exc:
        raise ValueError(f"unsupported environment: {name!r}") from exc
    return spec.resolve(step_attrs=step_attrs, rearrange_info=rearrange_info)


def get_planner_spec(tool_or_robot: str) -> PlannerSpec:
    key = tool_or_robot.strip().lower()
    if key.startswith("fr5_"):
        key = key[4:]
    elif key == "fr5":
        key = "empty"
    try:
        return _PLANNERS[key]
    except KeyError as exc:
        raise ValueError(f"unsupported tool/robot configuration: {tool_or_robot!r}") from exc
