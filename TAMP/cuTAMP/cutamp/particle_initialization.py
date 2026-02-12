# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
from typing import Optional

import roma
import torch
from curobo.geom.types import Cuboid
from curobo.types.math import Pose

from cutamp.config import TAMPConfiguration
from cutamp.costs import sphere_to_sphere_overlap
from cutamp.samplers import (
    grasp_4dof_sampler,
    grasp_6dof_sampler,
    place_4dof_sampler,
    sample_yaw,
)
from cutamp.tamp_domain import MoveFree, MoveHolding, Pick, Place, Place_magnet_to_beaker, Move_to_Surface, Place_poured_beaker
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.utils.common import (
    Particles,
    action_4dof_to_mat4x4,
    action_6dof_to_mat4x4,
    pose_list_to_mat4x4,
    sample_between_bounds,
    transform_spheres,
)
from cutamp.utils.shapes import MultiSphere

_log = logging.getLogger(__name__)


class ParticleInitializer:
    def __init__(self, world: TAMPWorld, config: TAMPConfiguration):
        if config.enable_traj:
            raise NotImplementedError("Trajectory initialization not yet supported")
        if config.place_dof != 4:
            raise NotImplementedError(f"Only 4-DOF grasp and placement supported for now, not {config.place_dof}")
        if config.grasp_dof != 4 and config.grasp_dof != 6:
            raise NotImplementedError(f"Only 4-DOF or 6-DOF grasp supported for now, not {config.grasp_dof}")
        self.world = world
        self.config = config
        self.q_init = world.q_init.repeat(config.num_particles, 1)

        # Sampler caching
        self.pick_cache = {}
        self.place_cache = {}

    def __call__(self, plan_skeleton: PlanSkeleton, verbose: bool = True) -> Optional[Particles]:
        config = self.config
        num_particles = self.config.num_particles
        world = self.world
        particles = {"q0": self.q_init.clone()}
        deferred_params = set()
        log_debug = _log.debug if verbose else lambda *args, **kwargs: None

        # Note: we don't consider state after executing earlier samples
        # Iterate through each ground operator in the plan skeleton and initialize and build up particles
        for idx, ground_op in enumerate(plan_skeleton):
            op_name = ground_op.operator.name
            params = ground_op.values
            header = f"{idx + 1}. {ground_op}"

            # MoveFree
            if op_name == MoveFree.name:
                q_start, _traj, q_end = params
                if q_start not in particles:
                    raise ValueError(f"{q_start=} should already be bound")
                deferred_params.add(q_end)
                log_debug(f"{header}. Deferred {q_end}")

            # MoveHolding
            elif op_name == MoveHolding.name:
                obj, grasp, q_start, _traj, q_end = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be bound")
                if q_start not in particles:
                    raise ValueError(f"{q_start=} should already be bound")
                deferred_params.add(q_end)
                log_debug(f"{header}. Deferred {q_end}")

            # Pick
            elif op_name == Pick.name:
                obj, grasp, q = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp in particles:
                    raise ValueError(f"{grasp=} shouldn't already be bound")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                # Note: pick cache currently assumes object is at same pose as when sampled
                if obj in self.pick_cache:
                    # important, we need to clone here
                    particles[grasp] = self.pick_cache[obj]["sampled_grasps"].clone()
                    ik_result = self.pick_cache[obj]["ik_result"]
                    particles[q] = ik_result.solution[:, 0].clone()
                    deferred_params.remove(q)
                    log_debug(
                        f"{header}. Using cached grasp poses for {obj}. {ik_result.success.sum()}/{num_particles} success"
                    )
                    continue

                # Sample grasps
                obj_curobo = world.get_object(obj)
                obj_spheres = world.get_collision_spheres(obj)
                num_faces = 4 if isinstance(obj_curobo, Cuboid) else None

                # Sample 4 times as many grasps as particles
                if config.grasp_dof == 4:
                    sampled_grasps = grasp_4dof_sampler(num_particles * 4, obj_curobo, obj_spheres, num_faces=num_faces)
                    obj_from_grasp = action_4dof_to_mat4x4(sampled_grasps)
                else:
                    sampled_grasps = grasp_6dof_sampler(num_particles * 4, obj_curobo, num_faces=num_faces)
                    obj_from_grasp = action_6dof_to_mat4x4(sampled_grasps)

                # Select the grasps that are not in collision with the object
                grasp_spheres = transform_spheres(world.robot_container.gripper_spheres, obj_from_grasp)
                grasp_coll = sphere_to_sphere_overlap(obj_spheres, grasp_spheres)
                good_idxs = grasp_coll.topk(num_particles, largest=False).indices
                particles[grasp] = sampled_grasps[good_idxs]

                # Transform grasps to hand frame
                if config.random_init:
                    q_sample = sample_between_bounds(num_particles, world.robot_container.joint_limits)
                    particles[q] = q_sample
                else:
                    obj_from_grasp = obj_from_grasp[good_idxs]
                    world_from_obj = pose_list_to_mat4x4(obj_curobo.pose).to(world.tensor_args.device)
                    world_from_grasp = world_from_obj @ obj_from_grasp
                    world_from_ee = world_from_grasp @ world.tool_from_ee

                    # Solve IK with cuRobo
                    world_from_ee = Pose.from_matrix(world_from_ee)
                    ik_result = world.ik_solver.solve_batch(world_from_ee, seed_config=None)  # TODO: seeding
                    log_debug(
                        f"{header}. IK success: {ik_result.success.sum()}/{num_particles}, took {ik_result.solve_time:.2f}s"
                    )
                    particles[q] = ik_result.solution[:, 0]
                deferred_params.remove(q)

                # Store in cache
                if config.cache_subgraphs:
                    self.pick_cache[obj] = {"sampled_grasps": particles[grasp], "ik_result": ik_result}

            # Place
            elif op_name == Place.name or op_name == Place_poured_beaker.name:
                if op_name == Place_poured_beaker.name:
                    obj, grasp, placement, surface, q, _ = params
                else:
                    obj, grasp, placement, surface, q = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be bound")
                if placement in particles:
                    raise ValueError(f"{placement=} shouldn't already be bound")
                if not world.has_object(surface):
                    raise ValueError(f"{surface=} not found in world")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                if (obj, surface) in self.place_cache:
                    # need to make sure the grasps match what is cached
                    actual_grasp = particles[grasp]
                    cached_grasp = self.place_cache[(obj, surface)]["grasp"]
                    if not (actual_grasp == cached_grasp).all():
                        raise RuntimeError(f"Grasps don't match for {obj} on {surface}")

                    # important, we need to clone here
                    sampled_placements = self.place_cache[(obj, surface)]["sampled_placements"].clone()
                    particles[placement] = sampled_placements
                    ik_result = self.place_cache[(obj, surface)]["ik_result"]
                    particles[q] = ik_result.solution[:, 0].clone()
                    deferred_params.remove(q)
                    log_debug(
                        f"{header}. Using cached placement poses for {obj}. {ik_result.success.sum()}/{num_particles} success"
                    )
                    continue

                # Sample placements pose of object (in world frame)
                obj_curobo = world.get_object(obj)
                obj_spheres = world.get_collision_spheres(obj)
                if config.random_init:
                    yaw = sample_yaw(num_particles * 4, None, self.world.tensor_args.device)
                    aabb = world.world_aabb.clone()
                    aabb[0, 2] = 0.0
                    aabb[1, 2] = max(aabb[1, 2], 0.2)
                    xyz = sample_between_bounds(num_particles * 4, aabb)
                    sampled_placements = torch.cat([xyz, yaw.unsqueeze(-1)], dim=1)
                else:
                    surface_curobo = world.get_object(surface)
                    sampled_placements = place_4dof_sampler(num_particles * 4, obj_curobo, obj_spheres, surface_curobo)

                # Select the placements that are not in collision with the object
                world_from_obj = action_4dof_to_mat4x4(sampled_placements)  # desired placement pose
                obj_place_spheres = transform_spheres(obj_spheres, world_from_obj)
                place_coll = world.collision_fn(obj_place_spheres[:, None].contiguous())[:, 0]
                best_idxs = place_coll.topk(num_particles, largest=False).indices
                sampled_placements = sampled_placements[best_idxs]
                world_from_obj = world_from_obj[best_idxs]

                # Set particles and then solve for robot configurations
                particles[placement] = sampled_placements
                if config.random_init:
                    q_sample = sample_between_bounds(num_particles, world.robot_container.joint_limits)
                    particles[q] = q_sample
                else:
                    # Get the hand pose given the placement pose in world frame.
                    # Need to take grasp into account to transform into hand frame.
                    if config.grasp_dof == 4:
                        obj_from_grasp = action_4dof_to_mat4x4(particles[grasp])
                    else:
                        obj_from_grasp = action_6dof_to_mat4x4(particles[grasp])
                    world_from_grasp = world_from_obj @ obj_from_grasp
                    world_from_ee = world_from_grasp @ world.tool_from_ee

                    # Solve IK
                    world_from_ee = Pose.from_matrix(world_from_ee)
                    ik_result = world.ik_solver.solve_batch(world_from_ee, seed_config=None)  # TODO: seeding?
                    log_debug(
                        f"{header}. IK success: {ik_result.success.sum()}/{num_particles}, took {ik_result.solve_time:.2f}s"
                    )
                    particles[q] = ik_result.solution[:, 0]
                deferred_params.remove(q)

                # Store in cache
                if config.cache_subgraphs:
                    self.place_cache[(obj, surface)] = {
                        "sampled_placements": sampled_placements,
                        "ik_result": ik_result,
                        "grasp": particles[grasp],
                    }

            elif op_name == Place_magnet_to_beaker.name:
                obj, grasp, placement, surface, q, _, _ = ground_op.values

                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be bound")
                if placement in particles:
                    raise ValueError(f"{placement=} shouldn't already be bound")
                if not world.has_object(surface):
                    raise ValueError(f"{surface=} not found in world")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                if (obj, surface) in self.place_cache:
                    # need to make sure the grasps match what is cached
                    actual_grasp = particles[grasp]
                    cached_grasp = self.place_cache[(obj, surface)]["grasp"]
                    if not (actual_grasp == cached_grasp).all():
                        raise RuntimeError(f"Grasps don't match for {obj} on {surface}")

                    # important, we need to clone here
                    sampled_placements = self.place_cache[(obj, surface)]["sampled_placements"].clone()
                    particles[placement] = sampled_placements
                    ik_result = self.place_cache[(obj, surface)]["ik_result"]
                    particles[q] = ik_result.solution[:, 0].clone()
                    deferred_params.remove(q)
                    log_debug(
                        f"{header}. Using cached placement poses for {obj}. {ik_result.success.sum()}/{num_particles} success"
                    )
                    continue

                # Sample placements pose of object (in world frame)
                obj_curobo = world.get_object(obj)
                obj_spheres = world.get_collision_spheres(obj)
                if config.random_init:
                    yaw = sample_yaw(num_particles * 4, None, self.world.tensor_args.device)
                    aabb = world.world_aabb.clone()
                    aabb[0, 2] = 0.0
                    aabb[1, 2] = max(aabb[1, 2], 0.2)
                    xyz = sample_between_bounds(num_particles * 4, aabb)
                    sampled_placements = torch.cat([xyz, yaw.unsqueeze(-1)], dim=1)
                else:
                    surface_curobo = world.get_object(surface)
                    sampled_placements = place_4dof_sampler(num_particles * 4, obj_curobo, obj_spheres, surface_curobo)

                # Select the placements that are not in collision with the object
                world_from_obj = action_4dof_to_mat4x4(sampled_placements)  # desired placement pose
                obj_place_spheres = transform_spheres(obj_spheres, world_from_obj)
                place_coll = world.collision_fn(obj_place_spheres[:, None].contiguous())[:, 0]
                best_idxs = place_coll.topk(num_particles, largest=False).indices
                sampled_placements = sampled_placements[best_idxs]
                world_from_obj = world_from_obj[best_idxs]

                # Set particles and then solve for robot configurations
                particles[placement] = sampled_placements
                if config.random_init:
                    q_sample = sample_between_bounds(num_particles, world.robot_container.joint_limits)
                    particles[q] = q_sample
                else:
                    # Get the hand pose given the placement pose in world frame.
                    # Need to take grasp into account to transform into hand frame.
                    if config.grasp_dof == 4:
                        obj_from_grasp = action_4dof_to_mat4x4(particles[grasp])
                    else:
                        obj_from_grasp = action_6dof_to_mat4x4(particles[grasp])
                    world_from_grasp = world_from_obj @ obj_from_grasp
                    world_from_ee = world_from_grasp @ world.tool_from_ee

                    # Solve IK
                    world_from_ee = Pose.from_matrix(world_from_ee)
                    ik_result = world.ik_solver.solve_batch(world_from_ee, seed_config=None)  # TODO: seeding?
                    log_debug(
                        f"{header}. IK success: {ik_result.success.sum()}/{num_particles}, took {ik_result.solve_time:.2f}s"
                    )
                    particles[q] = ik_result.solution[:, 0]
                deferred_params.remove(q)

                # Store in cache
                if config.cache_subgraphs:
                    self.place_cache[(obj, surface)] = {
                        "sampled_placements": sampled_placements,
                        "ik_result": ik_result,
                        "grasp": particles[grasp],
                    }

            # Move_to_Surface
            elif op_name == Move_to_Surface.name:
                obj, grasp, placement, surface, q = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be bound")
                if placement in particles:
                    raise ValueError(f"{placement=} shouldn't already be bound")
                if not world.has_object(surface):
                    raise ValueError(f"{surface=} not found in world")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                if (obj, surface) in self.place_cache:
                    # need to make sure the grasps match what is cached
                    actual_grasp = particles[grasp]
                    cached_grasp = self.place_cache[(obj, surface)]["grasp"]
                    if not (actual_grasp == cached_grasp).all():
                        raise RuntimeError(f"Grasps don't match for {obj} on {surface}")

                    # important, we need to clone here
                    sampled_placements = self.place_cache[(obj, surface)]["sampled_placements"].clone()
                    particles[placement] = sampled_placements
                    ik_result = self.place_cache[(obj, surface)]["ik_result"]
                    particles[q] = ik_result.solution[:, 0].clone()
                    deferred_params.remove(q)
                    log_debug(
                        f"{header}. Using cached placement poses for {obj}. {ik_result.success.sum()}/{num_particles} success"
                    )
                    continue

                # Sample placements pose of object (in world frame)
                obj_curobo = world.get_object(obj)
                obj_spheres = world.get_collision_spheres(obj)
                if config.random_init:
                    yaw = sample_yaw(num_particles * 4, None, self.world.tensor_args.device)
                    aabb = world.world_aabb.clone()
                    aabb[0, 2] = 0.0
                    aabb[1, 2] = max(aabb[1, 2], 0.2)
                    xyz = sample_between_bounds(num_particles * 4, aabb)
                    sampled_placements = torch.cat([xyz, yaw.unsqueeze(-1)], dim=1)
                else:
                    surface_curobo = world.get_object(surface)
                    sampled_placements = place_4dof_sampler(num_particles * 4, obj_curobo, obj_spheres, surface_curobo)

                # Select the placements that are not in collision with the object
                world_from_obj = action_4dof_to_mat4x4(sampled_placements)  # desired placement pose
                obj_place_spheres = transform_spheres(obj_spheres, world_from_obj)
                place_coll = world.collision_fn(obj_place_spheres[:, None].contiguous())[:, 0]
                best_idxs = place_coll.topk(num_particles, largest=False).indices
                sampled_placements = sampled_placements[best_idxs]
                world_from_obj = world_from_obj[best_idxs]

                # Set particles and then solve for robot configurations
                particles[placement] = sampled_placements
                if config.random_init:
                    q_sample = sample_between_bounds(num_particles, world.robot_container.joint_limits)
                    particles[q] = q_sample
                else:
                    # Get the hand pose given the placement pose in world frame.
                    # Need to take grasp into account to transform into hand frame.
                    if config.grasp_dof == 4:
                        obj_from_grasp = action_4dof_to_mat4x4(particles[grasp])
                    else:
                        obj_from_grasp = action_6dof_to_mat4x4(particles[grasp])
                    world_from_grasp = world_from_obj @ obj_from_grasp
                    world_from_ee = world_from_grasp @ world.tool_from_ee

                    # Solve IK
                    world_from_ee = Pose.from_matrix(world_from_ee)
                    ik_result = world.ik_solver.solve_batch(world_from_ee, seed_config=None)  # TODO: seeding?
                    log_debug(
                        f"{header}. IK success: {ik_result.success.sum()}/{num_particles}, took {ik_result.solve_time:.2f}s"
                    )
                    particles[q] = ik_result.solution[:, 0]
                deferred_params.remove(q)

                # Store in cache
                if config.cache_subgraphs:
                    self.place_cache[(obj, surface)] = {
                        "sampled_placements": sampled_placements,
                        "ik_result": ik_result,
                        "grasp": particles[grasp],
                    }
                    
            # Unknown
            else:
                raise NotImplementedError(f"Unsupported operator {op_name}")

        # There should not be any deferred parameters left
        if deferred_params:
            raise RuntimeError(f"Deferred parameters not resolved: {deferred_params}")

        return particles
