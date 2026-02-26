from typing import Dict, List
from dataclasses import dataclass, field

import torch
from curobo.geom.types import Cuboid, Cylinder
from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import unit_quat
from cutamp.envs import TAMPEnvironment

from envs.pouring import load_Transfer_env
from envs.stirring import load_Stir_env
from envs.default import load_default_env
from envs.moving import load_Moving_env
from envs.rearranging import load_Rearranging_env


ENTITIES = {
    "table": Cuboid(name="table", pose=[0.0, 0.0, -0.01, *unit_quat], dims=[1.5, 1.5, 0.02], color=[255, 0, 0]),
    "stirrer": Cuboid(name="stirrer", pose=[0.3, 0.3, 0.4, *unit_quat], dims=[0.1, 0.1, 0.075], color=[255, 0, 0]),

    # objects
    # "beaker": Cuboid(name="beaker", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.064, 0.064, 0.12], color=[0, 0, 255]),
    # "flask": Cuboid(name="flask", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.04, 0.04, 0.16], color=[0, 255, 0]),
    "beaker": Cuboid(name="beaker", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.05, 0.05, 0.11], color=[255, 0, 0]),
    "flask": Cuboid(name="flask", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.06, 0.06, 0.08], color=[255, 0, 0]),
    "magnet": Cuboid(name="magnet", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.045, 0.045, 0.03], color=[255, 0, 0]),
    "box" : Cuboid(name="box", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.1, 0.1, 0.08], color=[0, 0, 255]),
    "box_goal" : Cuboid(name="box_goal", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.15, 0.15, 0.01], color=[0, 255, 0]),    

    # Reigion
    "goal_region": Cuboid(name="goal", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.1, 0.1, 0.01], color=[186, 255, 201]),
    "pour_region": Cuboid(name="pour_region", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.08, 0.08, 0.0001], color=[255, 255, 255]),
    "beaker_region": Cuboid(name="beaker_region", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.08, 0.08, 0.0001], color=[255, 255, 255]),
    "box_region": Cuboid(name="box_region", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.2, 0.2, 0.0001], color=[255, 255, 255]),
    "rearrange_region" : Cuboid(name="rearrange_region", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.1, 0.1, 0.0001], color=[255, 255, 255]),

    "obstacle_1": Cuboid(name="obstacle_1", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.07, 0.07, 0.1], color=[255, 0, 255]),
    "obstacle_2": Cuboid(name="obstacle_2", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.07, 0.07, 0.1], color=[255, 0, 255]),
}


class TAMPEnvManager:

    def __init__(self):

        self.entities = ENTITIES

        self.is_update_entities = False
        self.movables = []
        self.statics = []
        self.ex_collision = []


    def update_entities(
        self,
        poses: Dict[str, List[float]],
        movables: List[str] = None,
        statics: List[str] = None,
        ex_collision: List[str] = None,
    ):
        self.movables = []
        self.statics = []
        self.ex_collision = []

        if poses is not None:
            for name, pose in poses.items():
                self.entities[name.lower()].pose = pose
            self.entities["beaker"].pose[2] += 0.06
            self.entities["box"].pose[2] += 0.05
            self.entities["flask"].pose[2] += 0.055

        if movables is not None:
            for name in movables:
                self.movables.append(self.entities[name.lower()])

        if statics is not None:
            for name in statics:
                self.statics.append(self.entities[name.lower()])

        if ex_collision is not None:
            for name in ex_collision:
                self.ex_collision.append(self.entities[name.lower()])

        self.is_update_entities = True


    def load_env(self, name: str) -> TAMPEnvironment:
        
        name = name.lower()

        if self.is_update_entities:

            if name == "transfer":

                env, pour_region_pose = load_Transfer_env(
                    entities=self.entities,
                    movables=self.movables,
                    statics=self.statics,
                    ex_collision=self.ex_collision,
                )

            elif name == "stir":

                env = load_Stir_env(
                    entities=self.entities,
                    movables=self.movables,
                    statics=self.statics,
                    ex_collision=self.ex_collision,
                )
            
            elif name == "default":

                env = load_default_env(
                    entities=self.entities,
                    movables=self.movables,
                    statics=self.statics,
                    ex_collision=self.ex_collision,
                )

            elif name == "move":

                env = load_Moving_env(
                    entities=self.entities,
                    movables=self.movables,
                    statics=self.statics,
                    ex_collision=self.ex_collision,
                )

            elif name == "rearrange":

                env = load_Rearranging_env(
                    entities=self.entities,
                    movables=self.movables,
                    statics=self.statics,
                    ex_collision=self.ex_collision,
                )

            else:
                raise ValueError("name should be only 'transfer' or 'stir' or 'default' or 'move' or 'rearrange'")
            
            self.is_update_entities = False
            
        else:
            raise ValueError("Do 'update_entities' methods before loading env")
        
        if name == "transfer":
            return env, pour_region_pose
        else:
            return env

