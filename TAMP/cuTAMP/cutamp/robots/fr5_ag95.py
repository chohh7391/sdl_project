# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from functools import lru_cache
from pathlib import Path

import torch
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_assets_path
from curobo.util_file import join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from jaxtyping import Float
from yourdfpy import URDF

from cutamp.robots.utils import RerunRobot

# or simplified: (0, -1.57, 1.57, -1.57, -1.57, 0)
fr5_home = (0.0, -1.05, -2.18, -1.57, 1.57, 0.0)


@lru_cache(maxsize=1)
def fr5_ag95_curobo_cfg() -> dict:
    assets_dir = Path(__file__).parent / "assets"
    # Note: use ur5e_robotiq_ag95.yml for UR5e with camera mount (on MIT setup)
    cfg = load_yaml(str(assets_dir / "fr5_ag95.yml"))
    # Set some asset paths so cuRobo can load our URDF and meshes
    keys = ["external_asset_path", "external_robot_configs_path"]
    for key in keys:
        if key not in cfg["robot_cfg"]["kinematics"]:
            cfg["robot_cfg"]["kinematics"][key] = str(assets_dir)
    return cfg
    # return load_yaml(join_path(get_robot_configs_path(), "ur5e_robotiq_2f_140.yml"))


def _fr5_ag95_cfg_dict() -> dict:
    return fr5_ag95_curobo_cfg()["robot_cfg"]


def get_fr5_ag95_kinematics_model() -> CudaRobotModel:
    """cuRobo robot kinematics model."""
    robot_cfg = RobotConfig.from_dict(_fr5_ag95_cfg_dict())
    kinematics_model = CudaRobotModel(robot_cfg.kinematics)
    return kinematics_model


def get_fr5_ag95_ik_solver(
    world_cfg: WorldConfig,
    num_seeds: int = 12,
    self_collision_opt: bool = False,
    self_collision_check: bool = True,
    use_particle_opt: bool = False,
) -> IKSolver:
    """
    cuRobo IK solver for UR5e with Robotiq 2F-85 gripper.
    """
    robot_cfg = _fr5_ag95_cfg_dict()
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        num_seeds=num_seeds,
        self_collision_opt=self_collision_opt,
        self_collision_check=self_collision_check,
        use_particle_opt=use_particle_opt,
    )
    ik_solver = IKSolver(ik_config)
    return ik_solver


def get_fr5_ag95_gripper_spheres(tensor_args: TensorDeviceType = TensorDeviceType()) -> Float[torch.Tensor, "num_spheres 4"]:
    """Collision spheres for UR5e with Robotiq 2F-85 gripper. Note: the spheres are in the origin frame with z-up."""
    assets_dir = Path(__file__).parent / "assets"
    spheres_pt = assets_dir / "spheres" / "fr5_ag95.pt"
    if not spheres_pt.exists():
        raise FileNotFoundError(f"File not found: {spheres_pt}")

    spheres = torch.load(spheres_pt, map_location=tensor_args.device)
    assert spheres.ndim == 2 and spheres.shape[1] == 4, f"Invalid shape for FR5 spheres: {spheres.shape}"
    return spheres


def load_fr5_ag95_rerun(load_mesh: bool = True) -> RerunRobot:
    robot_cfg = _fr5_ag95_cfg_dict()
    urdf_rel_path = robot_cfg["kinematics"]["urdf_path"]
    urdf_path = os.path.join(robot_cfg["kinematics"].get("external_asset_path", ""), urdf_rel_path) or join_path(
        get_assets_path(), urdf_rel_path
    )

    def _locate_curobo_asset(fname: str) -> str:
        if fname.startswith("package://"):
            return os.path.join(get_assets_path(), "robot", fname.replace("package://", ""))

        assets_dir = Path(__file__).parent
        return os.path.join(assets_dir, fname)

    urdf = URDF.load(urdf_path, filename_handler=_locate_curobo_asset)
    return RerunRobot("fr5_ag95", urdf, q_neutral=fr5_home, load_mesh=load_mesh)
