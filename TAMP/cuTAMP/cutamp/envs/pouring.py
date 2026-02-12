from typing import Optional
import torch
from curobo.geom.types import Cuboid, Cylinder
from curobo.types.base import TensorDeviceType
from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import unit_quat
from cutamp.tamp_domain import HandEmpty, On, Poured
from cutamp.utils.shapes import MultiSphere


def load_pouring_env(
    beaker_pose: Optional[list] = None,
    tensor_args: TensorDeviceType = TensorDeviceType(),
) -> TAMPEnvironment:
    """Pick-and-place environment with a cylindrical beaker and small MultiSphere near goal."""

    if beaker_pose is None:
        beaker_pose = [0.0, 0.4, 0.04, *unit_quat]

    # -------------------- 테이블 --------------------
    table = Cuboid(
        name="table",
        dims=[1.5, 1.5, 0.02],
        pose=[0.0, 0.0, -0.01, *unit_quat],
        color=[235, 196, 145],
    )

    # -------------------- Goal region --------------------
    goal_region = Cuboid(
        name="goal",
        dims=[0.1, 0.1, 0.01],
        pose=[-0.3, 0.0, 0.01, *unit_quat],
        color=[186, 255, 201],
    )

    beaker = Cuboid(
        name="beaker", 
        dims=[0.05, 0.05, 0.1], 
        pose=[0.4, 0.2, 0.075 + 0.01, *unit_quat], 
        color=[0, 0, 255]
    )

    flask = Cuboid(
        name="flask", 
        dims=[0.06, 0.06, 0.08], 
        pose=[0.0, 0.5, 0.05, *unit_quat], 
        color=[0, 255, 0]
    )

    pour_region = Cuboid(
        name="beaker_region",
        dims=[0.06, 0.06, 0.0001],  # practically invisible
        pose=[0.0, 0.5, 0.15, *unit_quat],  # floating above goal
        color=[255, 255, 255],
    )

    obstacle_1 = Cuboid(
        name="obstacle_1", 
        dims=[0.07, 0.07, 0.1], 
        pose=[0.2, 0.2, 0.076, *unit_quat], 
        color=[255, 0, 255]
    )

    obstacle_2 = Cuboid(
        name="obstacle_2", 
        dims=[0.07, 0.07, 0.1], 
        pose=[-0.2, -0.2, 0.076, *unit_quat], 
        color=[255, 0, 255]
    )

    # -------------------- 목표 상태 정의 --------------------
    goal_state = frozenset(
        { 
         On.ground(beaker.name, goal_region.name), 
         HandEmpty.ground(),
         Poured.ground(beaker.name, pour_region.name),
        }
    )

    # -------------------- 환경 구성 --------------------
    env = TAMPEnvironment(
        name="pouring",
        movables=[
            beaker,
            # obstacle_1,
            # obstacle_2, 
            flask,
            ],
        statics=[
            table, 
            goal_region
            ],
        ex_collision=[
            pour_region,
            ],
        type_to_objects={
            "Movable": [
                beaker, 
                # obstacle_1,
                # obstacle_2,
                flask,
                ],
            "Surface": [
                table, 
                goal_region, 
                pour_region,
                ],
            "ExCollision":[
                pour_region,
            ]
        },
        goal_state=goal_state,
    )

    return env
