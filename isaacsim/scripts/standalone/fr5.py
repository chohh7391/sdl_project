# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional

import carb
import numpy as np
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
from isaacsim.robot.manipulators.grippers.surface_gripper import SurfaceGripper

from gripper import SlowParallelGripper


class FR5(Robot):

    def __init__(
        self,
        prim_path: str,
        name: str = "fr5",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        attach_gripper: Optional[bool] = True,
        is_surface_gripper: Optional[bool] = False,
        surface_gripper_path: Optional[str] = None,
        gripper_dof_names: Optional[List[str]] = None,
        use_mimic_joints: bool = False,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        deltas: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name
        self._is_surface_gripper = is_surface_gripper

        if not prim.IsValid():
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            
        self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
            
        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )
        if attach_gripper:
            if is_surface_gripper:
                assert surface_gripper_path is not None
                self._gripper = SurfaceGripper(
                    end_effector_prim_path=self._end_effector_prim_path,
                    surface_gripper_path=surface_gripper_path
                )
            else:
                if gripper_dof_names is not None:
                    if deltas is None:
                        deltas = np.array([-0.13]) / get_stage_units()
                    self._gripper = ParallelGripper(
                        end_effector_prim_path=self._end_effector_prim_path,
                        joint_prim_names=gripper_dof_names,
                        use_mimic_joints=use_mimic_joints,
                        joint_opened_positions=gripper_open_position,
                        joint_closed_positions=gripper_closed_position,
                        action_deltas=deltas,
                    )

        self.joints_default_state = None
        return

    @property
    def end_effector(self) -> SingleRigidPrim:
        return self._end_effector

    @property
    def gripper(self):
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        super().initialize(physics_sim_view)
        self._end_effector = SingleRigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector.initialize(physics_sim_view)
        if isinstance(self._gripper, ParallelGripper):
            self._gripper.initialize(
                physics_sim_view=physics_sim_view,
                articulation_apply_action_func=self.apply_action,
                get_joint_positions_func=self.get_joint_positions,
                set_joint_positions_func=self.set_joint_positions,
                dof_names=self.dof_names,
            )
        if isinstance(self._gripper, SurfaceGripper):
            self._gripper.initialize(physics_sim_view=physics_sim_view, articulation_num_dofs=self.num_dof)
        return

    def post_reset(self) -> None:
        self.set_joints_default_state(
            positions=self.joints_default_state
        )
        super().post_reset()
        
        if self._gripper != None:
            self._gripper.post_reset()
            if not self._is_surface_gripper:
                self._articulation_controller.switch_dof_control_mode(
                dof_index=self.gripper.joint_dof_indicies[0], mode="position"
            )
        
        return
    
    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        if self._gripper != None:
            self._gripper.update()
        return