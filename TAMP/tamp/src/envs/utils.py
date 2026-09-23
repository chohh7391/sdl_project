from typing import Dict, List
from curobo.geom.types import Cuboid
from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import unit_quat
from cutamp.envs import TAMPEnvironment

from envs.constants import (PLANNER_Z_LIFT, vessel_dims, scale_dims,
                            scale_pose_from_pan, SCALE_TOP_M,
                            TABLE_Z_OFFSET)
from envs.transfer import load_transfer_env
from envs.stir import load_stir_env
from envs.default import load_default_env
from envs.move import load_move_env
from envs.rearrange import load_rearrange_env
import copy


ENTITIES = {
    # The table's own pose is NOT reported by the simulator (it is not among any
    # environment's queried entities), so the bench offset has to be applied to
    # the planner's copy here as well as to the simulator's prim.
    "table": Cuboid(name="table", pose=[0.0, 0.0, -0.01 + TABLE_Z_OFFSET, *unit_quat], dims=[1.5, 1.5, 0.02], color=[255, 0, 0]),
    "stirrer": Cuboid(name="stirrer", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.18, 0.18, 0.09], color=[255, 0, 0]),

    # objects
    # Dims come from envs.constants so the planner, the simulator's colliders
    # and the trial scoring cannot drift apart; SDL_GLASSWARE selects the set.
    "beaker": Cuboid(name="beaker", pose=[0.0, 0.0, 0.0, *unit_quat], dims=vessel_dims("beaker"), color=[255, 0, 0]),
    "flask": Cuboid(name="flask", pose=[0.0, 0.0, 0.0, *unit_quat], dims=vessel_dims("flask"), color=[255, 0, 0]),
    # Magnetic stir bar. Must match isaacsim Task.STIR_BAR_DIMS. It was a 45 mm
    # cube, whose worst-yaw diagonal (63.6 mm) does not clear the flask's 64 mm
    # opening, so the Stir terminal condition ("stir bar inside the vessel") was
    # geometrically impossible. A real PTFE stir bar is ~10 mm across, 30-40 long.
    "magnet": Cuboid(name="magnet", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.010, 0.010, 0.035], color=[255, 0, 0]),
    "box" : Cuboid(name="box", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.108, 0.108, 0.08], color=[0, 0, 255]),
    # The real cell's electronic balance, which the Transfer target stands on.
    # Present only in the "transfer_real" environment; the nominal scenes never
    # reference it, so adding it here cannot change any existing comparison.
    # Its pose is set by the simulator like any other entity (see
    # isaacsim Task.SCALE_*), so this default is a placeholder.
    "scale": Cuboid(name="scale", pose=[0.0, 0.0, SCALE_TOP_M / 2.0, *unit_quat],
                    dims=scale_dims(), color=[128, 128, 128]),
    "box_goal" : Cuboid(name="box_goal", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.15, 0.15, 0.01], color=[0, 255, 0]),

    # Regions
    "goal_region": Cuboid(name="goal_region", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.1, 0.1, 0.01], color=[186, 255, 201]),
    "pour_region": Cuboid(name="pour_region", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.08, 0.08, 0.0001], color=[255, 255, 255]),
    "beaker_region": Cuboid(name="beaker_region", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.08, 0.08, 0.0001], color=[255, 255, 255]),
    "box_region": Cuboid(name="box_region", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.2, 0.2, 0.0001], color=[255, 255, 255]),
    "rearrange_region" : Cuboid(name="rearrange_region", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.3, 0.3, 0.0001], color=[255, 255, 255]),

    "obstacle_1": Cuboid(name="obstacle_1", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.07, 0.07, 0.1], color=[255, 0, 255]),
    "obstacle_2": Cuboid(name="obstacle_2", pose=[0.0, 0.0, 0.0, *unit_quat], dims=[0.07, 0.07, 0.1], color=[255, 0, 255]),
}


class TAMPEnvManager:

    def __init__(self):

        self.entities = copy.deepcopy(ENTITIES)

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
        rearrange_grid: str = None,
    ):
        self.movables = []
        self.statics = []
        self.ex_collision = []
        self.rearrange_grid = None

        if poses is not None:
            for name, pose in poses.items():
                self.entities[name.lower()].pose = pose
            
            # offset for avoiding collision
            offset = PLANNER_Z_LIFT
            self.entities["beaker"].pose[2] += offset
            self.entities["flask"].pose[2] += offset
            self.entities["box"].pose[2] += offset
            self.entities["stirrer"].pose[2] += offset
            self.entities["magnet"].pose[2] += 2 * offset

        if movables is not None:
            for name in movables:
                self.movables.append(self.entities[name.lower()])

        if statics is not None:
            for name in statics:
                self.statics.append(self.entities[name.lower()])

        if ex_collision is not None:
            for name in ex_collision:
                self.ex_collision.append(self.entities[name.lower()])

        if rearrange_grid is not None:
            self.rearrange_grid = rearrange_grid

        self.is_update_entities = True

    def load_env(self, name: str) -> TAMPEnvironment:
        
        name = name.lower()

        if self.is_update_entities:

            if name in ("transfer", "transfer_real"):

                env, pour_region_pose = load_transfer_env(
                    entities=self.entities,
                    movables=self.movables,
                    statics=self.statics,
                    ex_collision=self.ex_collision,
                )

            elif name == "stir":

                env = load_stir_env(
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

                env = load_move_env(
                    entities=self.entities,
                    movables=self.movables,
                    statics=self.statics,
                    ex_collision=self.ex_collision,
                )

            elif name == "rearrange":

                env = load_rearrange_env(
                    entities=self.entities,
                    movables=self.movables,
                    statics=self.statics,
                    ex_collision=self.ex_collision,
                    rearrange_grid=self.rearrange_grid
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

