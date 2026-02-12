# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Sequence
from cutamp.task_planning import Fluent, Parameter, TAMPOperator, State
from cutamp.task_planning.constraints import (
    CollisionFree,
    CollisionFreeGrasp,
    CollisionFreeHolding,
    CollisionFreePlacement,
    KinematicConstraint,
    Motion,
    StablePlacement,
)
from cutamp.task_planning.costs import GraspCost, TrajectoryLength

# -------------------- Type Definitions --------------------
Conf = "conf"
Traj = "traj"
Pose = "pose"
Grasp = "grasp"

Movable = "movable"
Surface = "surface"

# -------------------- Fluents --------------------
At = Fluent("At", [Parameter("q", Conf)])
HandEmpty = Fluent("HandEmpty")
CanMove = Fluent("CanMove")
JustMoved = Fluent("JustMoved")
Holding = Fluent("Holding", [Parameter("obj", Movable)])
HoldingWithGrasp = Fluent("HoldingWithGrasp", [Parameter("obj", Movable), Parameter("grasp", Grasp)])
IsMovable = Fluent("IsMovable", [Parameter("obj", Movable)])
IsSurface = Fluent("IsSurface", [Parameter("surface", Surface)])
HasNotPickedUp = Fluent("HasNotPickedUp", [Parameter("obj", Movable)])
On = Fluent("On", [Parameter("obj", Movable), Parameter("surface", Surface)])
OnBeaker = Fluent("OnBeaker", [Parameter("obj", Movable), Parameter("surface", Surface)])
Poured = Fluent("Poured", [Parameter("obj", Movable), Parameter("surface", Surface)])

all_tamp_fluents = [
    At, 
    HandEmpty, 
    CanMove, 
    JustMoved,
    Holding, 
    HoldingWithGrasp,
    IsMovable, 
    IsSurface, 
    HasNotPickedUp,
    On, 
    OnBeaker, 
    Poured,
]

# -------------------- Parameters --------------------
q = Parameter("q", Conf)
q_start = Parameter("q_start", Conf)
q_end = Parameter("q_end", Conf)
traj = Parameter("traj", Traj)
obj = Parameter("obj", Movable)
surface = Parameter("surface", Surface)
grasp = Parameter("grasp", Grasp)
placement = Parameter("placement", Pose)
beaker = Parameter("beaker", Movable)
goal_surface = Parameter("goal_surface", Surface)
pour_surface = Parameter("pour_surface", Surface)



# -------------------- Base Operators --------------------
MoveFree = TAMPOperator(
    "MoveFree",
    [q_start, traj, q_end],
    preconditions=[At(q_start), HandEmpty(), CanMove()],
    add_effects=[At(q_end), JustMoved()],
    del_effects=[At(q_start), CanMove()],
    constraints=[
        CollisionFree(q_start, traj, q_end),
        Motion(q_start, traj, q_end),
    ],
    costs=[TrajectoryLength(q_start, traj, q_end)],
)

MoveHolding = TAMPOperator(
    "MoveHolding",
    [obj, grasp, q_start, traj, q_end],
    preconditions=[At(q_start), Holding(obj), HoldingWithGrasp(obj, grasp), CanMove()],
    add_effects=[At(q_end), JustMoved()],
    del_effects=[At(q_start), CanMove()],
    constraints=[
        CollisionFreeHolding(obj, grasp, q_start, traj, q_end),
        Motion(q_start, traj, q_end),
    ],
    costs=[TrajectoryLength(q_start, traj, q_end)],
)

Pick = TAMPOperator(
    "Pick",
    [obj, grasp, q],
    preconditions=[
        At(q), HandEmpty(),
        IsMovable(obj),
        JustMoved(),
        HasNotPickedUp(obj),
    ],
    add_effects=[
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        CanMove(),
    ],
    del_effects=[HandEmpty(), JustMoved(), HasNotPickedUp(obj)],
    constraints=[
        KinematicConstraint(q, grasp),
        CollisionFreeGrasp(obj, grasp),
    ],
    costs=[
        # GraspCost(obj, grasp)
        ],
)

Place = TAMPOperator(
    "Place",
    [obj, grasp, placement, surface, q],
    preconditions=[
        At(q),
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        IsSurface(surface),
        JustMoved(),
    ],
    add_effects=[HandEmpty(), CanMove(), On(obj, surface)],
    del_effects=[
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        JustMoved(),
    ],
    constraints=[
        KinematicConstraint(q, placement),
        StablePlacement(obj, grasp, placement, surface),
        CollisionFreePlacement(obj, placement, surface),
    ],
    costs=[],
)

# -------------------- Custom Task Operators --------------------
Place_magnet_to_beaker = TAMPOperator(
    "Place_magnet_to_beaker",
    [obj, grasp, placement, surface, q, beaker, goal_surface],
    preconditions=[
        At(q),
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        IsSurface(surface),
        JustMoved(),
        On(beaker, goal_surface),   
    ],
    add_effects=[
        HandEmpty(),
        CanMove(),
        On(obj, surface),
        OnBeaker(obj, surface),
    ],
    del_effects=[
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        JustMoved(),
    ],
    constraints=[
        KinematicConstraint(q, placement),
        StablePlacement(obj, grasp, placement, surface),
        CollisionFreePlacement(obj, placement, surface),
    ],
    costs=[],
)

Move_to_Surface = TAMPOperator(
    "Move_to_Surface",
    [obj, grasp, placement, pour_surface, q],
    preconditions=[
        At(q),
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        IsSurface(pour_surface),
        JustMoved(),
    ],
    add_effects=[Poured(obj, pour_surface), CanMove()],
    del_effects=[
        JustMoved(),
    ],
    constraints=[
        KinematicConstraint(q, placement),
        StablePlacement(obj, grasp, placement, pour_surface),
        CollisionFreePlacement(obj, placement, pour_surface),
    ],
    costs=[],
)

Place_poured_beaker = TAMPOperator(
    "Place_poured_beaker",
    [obj, grasp, placement, surface, q, pour_surface],
    preconditions=[
        At(q),
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        IsSurface(surface),
        JustMoved(),
        Poured(obj, pour_surface),
    ],
    add_effects=[HandEmpty(), CanMove(), On(obj, surface)],
    del_effects=[
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        JustMoved(),
    ],
    constraints=[
        KinematicConstraint(q, placement),
        StablePlacement(obj, grasp, placement, surface),
        CollisionFreePlacement(obj, placement, surface),
    ],
    costs=[],
)

# -------------------- Operator List --------------------
all_tamp_operators = [
    MoveFree,
    MoveHolding,
    Pick,
    Place,
    Place_magnet_to_beaker,
    Move_to_Surface,
    Place_poured_beaker,
]

# -------------------- Initial State --------------------
def get_initial_state(
    movables: Sequence[str] = (), surfaces: Sequence[str] = (),
) -> State:
    """Ground the initial state of the TAMP domain."""
    initial_state = {At.ground("q0"), HandEmpty.ground(), CanMove.ground()}
    for movable in movables:
        initial_state.add(IsMovable.ground(movable))
        initial_state.add(HasNotPickedUp.ground(movable))
    for surface in surfaces:
        initial_state.add(IsSurface.ground(surface))
    return frozenset(initial_state)
