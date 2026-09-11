from abc import ABC
from typing import Dict, List, Optional, Tuple
import os, sys
import numpy as np

from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera
from isaacsim.core.api.objects import FixedCuboid, DynamicCylinder, DynamicCuboid, VisualCuboid
from isaacsim.core.api.materials.omni_pbr import OmniPBR
# Isaac Sim 6.x migration: the ROS2 bridge extension was split; read_camera_info
# moved from isaacsim.ros2.bridge to isaacsim.ros2.core (impl.camera_info_utils).
from isaacsim.ros2.core import read_camera_info
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.prims import SingleRigidPrim

from fr5 import FR5

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from object import (create_apriltag_plate, create_box_collider_rigid, create_box_collider_static,
                    create_hollow_box_collider_rigid, create_single_rigid_prim_from_usd)
from camera import CameraInfo, set_world_pose_from_view

ASSET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "TAMP", "tamp", "content", "assets")


class Task(ABC, BaseTask):

    def __init__(self, name: str, robot_prim_path: str, robot_name: str) -> None:
        
        BaseTask.__init__(self, name=name)
        self._asset_root_path = get_assets_root_path()
        if self._asset_root_path is None:
            raise Exception("Could not find Isaac Sim assets folder")
        self._robot_prim_path = robot_prim_path
        self._robot_name = robot_name

        self.camera_info = CameraInfo()
        self.camera_positions = [
            np.array([0.45, 0.0, 1.5]),
            np.array([-0.45, 0.0, 1.5]),
        ]
        self.camera_orientations = [
            rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
            rot_utils.euler_angles_to_quats(np.array([0, 90, 180]), degrees=True),
        ]

        self.current_positions = None
        self.current_orientations = None
        self.desired_tool = None
        self.current_tool = None

        # # G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, G12
        # random_grid = np.random.choice("G1 G3 G4 G5 G6 G7 G8 G9 G10 G11 G12".split())
        # random_x, random_y = self.get_grid_xy(random_grid)
        flask_x, flask_y = self.get_grid_xy("G12")
        box_x, box_y = self.get_grid_xy("G1")
        noise = np.random.uniform(-0.1, 0.1, size=2)


        self.default_positions = {
            "table": np.array([0.0, 0.0, -0.01]),
            # z = cuTAMP box-center height (dims_z / 2) so each box collider
            # rests on the table top (z = 0). x,y layout unchanged. (REFACTOR II-2)
            "stirrer": np.array([-0.15, 0.6, 0.045]),      # 0.09 / 2
            # "beaker": np.array([random_x, random_y, 0.067]),
            "beaker": np.array([0.49, 0.17475, 0.0675]),   # 0.135 / 2
            # "flask": np.array([flask_x + noise[0], flask_y + noise[1], 0.07]),
            "flask": np.array([0.48383, 0.33166, 0.06]),   # 0.12 / 2
            "magnet": np.array([-0.3, 0.416, self.STIR_BAR_DIMS[2] / 2.0]),
            # "box" : np.array([box_x, box_y, 0.06]),
            "box" : np.array([-0.12621, -0.57484, 0.04]),  # 0.08 / 2
            # Goal tray for the Move task, moved from G1 (-0.15, -0.6) to G3.
            # At G1 it sat under the box's own start cell: the box centre was
            # 3.5 cm from the tray centre, i.e. INSIDE the 15 cm tray footprint,
            # so (a) the Move goal was already satisfied at t = 0 and (b) the box
            # spawned with its bottom at z = 0 while the tray top is at z = 0.010,
            # i.e. 10 mm of interpenetration. G3 is the adjacent grid cell on the
            # same table edge, which makes Move a 0.30 m transport that stays clear
            # of the beaker/flask cluster (G11/G12) and of the Transfer goal region.
            # The tray is never randomized and is not in _RANDOMIZE_ORDER, so this
            # cannot change any seeded layout (verified: seeds 0-29 identical).
            "box_goal" : np.array([0.15, -0.6, 0.005]),    # 0.01 / 2
        }
        self.default_orientations = {
            "table": np.array([1.0, 0.0, 0.0, 0.0]),
            "stirrer": np.array([1.0, 0.0, 0.0, 0.0]),
            "beaker": np.array([1.0, 0.0, 0.0, 0.0]),
            "flask": np.array([1.0, 0.0, 0.0, 0.0]),
            "magnet": np.array([1.0, 0.0, 0.0, 0.0]),
            "box": np.array([0.917491, 0, 0, -0.3977565]),
            "box_goal": np.array([1.0, 0.0, 0.0, 0.0]),
        }

        # Nominal ("home") arm configuration, shared by every tool build in
        # set_robot(). Kept as a field so seeded randomization (SDL_SEED, below)
        # can perturb it deterministically without touching the four tool blocks.
        self._nominal_home_arm = np.array([0.0, -1.05, -2.18, -1.57, 1.57, 0.0])
        self._home_arm = self._nominal_home_arm.copy()

        # ------------------------------------------------------------------ #
        # Seed-driven object-position / robot-config randomization (STAGE B1,
        # REFACTOR.md I-7 / PLAN.md D3, R1#8 reproducibility).
        #
        # When the env var SDL_SEED is set to an integer, every MOVABLE object's
        # (x, y) is perturbed within +/-5 cm of its nominal cell, its yaw within
        # +/-180 deg, and the robot's initial arm config within +/-10 deg of
        # home -- DETERMINISTICALLY from the seed (same seed => identical layout).
        # Objects stay on the table, inside the workspace, and non-overlapping
        # (overlaps are rejected and resampled). When SDL_SEED is UNSET the
        # nominal hard-coded layout above is used unchanged (regression-safe).
        # Ranges + RNG are documented in _randomize_layout().
        # ------------------------------------------------------------------ #
        self.seed = None
        _seed_env = os.environ.get("SDL_SEED", "").strip()
        if _seed_env != "":
            try:
                self.seed = int(_seed_env)
            except ValueError:
                print(f"[Task] SDL_SEED={_seed_env!r} is not an int; ignoring (nominal layout).")
        if self.seed is not None:
            self._randomize_layout(self.seed)

        self.scale_data = 0.0
        self.scale_gain = 50.0

        return

    # In-plane bounding radius (circumscribed, = half-diagonal) of each movable's
    # cuTAMP box footprint [dx, dy] -- used for conservative circle-vs-circle
    # non-overlap rejection (holds for ANY yaw). Values match the box colliders
    # spawned in set_object() (which match TAMP/tamp/src/envs/utils.py ENTITIES).
    # AprilTag edge length [m]; must equal the `size`/`sizes` the detector is
    # configured with in perception/apriltag_ros/cfg/tags_36h11.yaml, because the
    # pose it reports scales with it.
    APRILTAG_SIZE_M = 0.08
    # Known table positions for the first perception gate: fixed, so the pose the
    # detector reports can be compared against ground truth.
    APRILTAG_TABLE_POSES = {
        0: (0.35, 0.00),
        1: (0.35, -0.20),
    }

    # Wall thickness of the hollow flask [m]. 3 mm leaves a 64 mm clear opening
    # in the 70 mm outer envelope and is thick enough for stable PhysX contacts.
    FLASK_WALL_M = 0.003
    # Magnetic stir bar [m]. Was a 45 mm cube, whose worst-yaw diagonal (63.6 mm)
    # does not clear the 64 mm opening -- the Stir task was geometrically
    # impossible whatever the vessel. A real PTFE stir bar is about 10 mm across
    # and 30-40 mm long; at 10 mm the worst-yaw need is 14 mm, with wide margin.
    STIR_BAR_DIMS = [0.010, 0.010, 0.035]

    _MOVABLE_FOOTPRINT = {
        "beaker":  (0.05, 0.05),
        "flask":   (0.07, 0.07),
        # Deliberately still the pre-stir-bar 45 mm footprint, NOT STIR_BAR_DIMS.
        # This table only drives the non-overlap rejection during layout
        # sampling, so reserving more space than the object occupies is
        # conservative (it can only add clearance, never miss an overlap), and
        # keeping it fixed means every seeded layout is bit-identical to the ones
        # the earlier batches were measured on.
        "magnet":  (0.045, 0.045),
        "box":     (0.108, 0.108),
        "stirrer": (0.18, 0.18),
    }
    # Objects randomized when SDL_SEED is set (fixed order => deterministic RNG
    # draw sequence). table/box_goal (goal tray) stay fixed like the goal region.
    _RANDOMIZE_ORDER = ["beaker", "flask", "magnet", "box", "stirrer"]

    # Per-tool correction to the shared home arm configuration, applied on top of
    # the (optionally seeded) home in set_robot().
    #
    # The shared home folds the elbow back (j3 = -2.18 rad), which parks the tool
    # close to the shoulder. Measured with cuRobo's own start-state check on the
    # fr5_dh3 model: at the shared home the 3-finger gripper's finger1 links and
    # 'shoulder_link' are inside each other's self-collision buffers, so the home
    # sits ON the self-collision boundary -- the COMMANDED home passes for all 30
    # seeds, but the pose the arm actually settles at does not (seed 0's logged
    # joint state checks INVALID_START_STATE_SELF_COLLISION, constraint 1.3111),
    # and cuRobo then refuses to plan the very first motion segment. That made the
    # Stir task unplannable for every seed. fr5, fr5_ag95 and fr5_vgc10 all have
    # margin here (0/600 infeasible under the same noise), so this is scoped to
    # dh3 rather than changing the shared home for every tool.
    #
    # +0.10 rad on j3 (5.7 deg, unfolding the elbow) is the smallest offset tested
    # that is clean at every noise level: 0/450 infeasible over the 30 seeded homes
    # with +/-10, +/-20 and +/-40 mrad of tracking noise, versus 4-5/450 at zero
    # offset. 0.15/0.20/0.30 rad are equally clean, so the margin is not knife-edge.
    _TOOL_HOME_OFFSET = {
        "dh3": np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0]),
    }

    def tool_home_offset(self, tool):
        """Per-tool correction to the shared home arm configuration [rad, 6]."""
        return np.asarray(
            self._TOOL_HOME_OFFSET.get(tool, np.zeros(6)), dtype=float
        )

    _POS_JITTER_M = 0.05          # +/- position jitter [m] about the nominal cell
    _YAW_RANGE_RAD = np.pi        # +/- yaw range [rad] (== +/-180 deg)
    _ROBOT_JITTER_RAD = np.deg2rad(10.0)   # +/- per-arm-joint jitter [rad]
    _OVERLAP_MARGIN_M = 0.01      # extra clearance between movable footprints [m]
    _BASE_KEEPOUT_M = 0.15        # keep-out radius around the robot base (origin)
    _WORKSPACE_HALF_M = 0.70      # |x|,|y| must stay within this (table is 0.75)
    _MAX_RESAMPLE = 300           # resample tries before falling back to nominal

    @staticmethod
    def _bounding_radius(dims_xy):
        return 0.5 * float(np.hypot(dims_xy[0], dims_xy[1]))

    @staticmethod
    def _yaw_quat(yaw):
        """Scalar-first (w, x, y, z) quaternion for a yaw rotation about +Z."""
        return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])

    def _randomize_layout(self, seed):
        """Deterministically perturb the movable objects + robot home from `seed`.

        RNG: numpy Generator (PCG64) via np.random.default_rng(seed). All draws
        are issued in a FIXED program order (robot arm first, then each movable
        in _RANDOMIZE_ORDER), so a given seed always yields an identical layout
        regardless of how many overlap-resamples occur.

        Ranges (per the paper's randomization spec):
          * object (x, y): nominal +/- _POS_JITTER_M (5 cm), uniform, z unchanged
          * object yaw   : uniform in +/- _YAW_RANGE_RAD (+/-180 deg)
          * robot arm    : each of the 6 joints, home +/- _ROBOT_JITTER_RAD (10 deg)

        Constraints (rejection-sampled per object, up to _MAX_RESAMPLE tries):
          * inside the workspace: |x|,|y| <= _WORKSPACE_HALF_M
          * clear of the robot base: center-to-origin > _BASE_KEEPOUT_M + radius
          * non-overlapping with already-placed movables: center distance
            > r_i + r_j + _OVERLAP_MARGIN_M (circumscribed radii => any-yaw safe)
        If an object cannot be placed within the try budget it falls back to its
        nominal (x, y) (still deterministic). box_goal / table are never moved.
        """
        rng = np.random.default_rng(seed)

        # --- robot initial config: home +/- 10 deg per arm joint ---------------
        arm_jitter = rng.uniform(-self._ROBOT_JITTER_RAD, self._ROBOT_JITTER_RAD, size=6)
        self._home_arm = self._nominal_home_arm + arm_jitter

        # --- movable object (x, y) + yaw --------------------------------------
        placed = {}  # name -> (x, y, radius) accepted so far
        for name in self._RANDOMIZE_ORDER:
            if name not in self.default_positions:
                continue
            nominal = np.asarray(self.default_positions[name], dtype=float)
            radius = self._bounding_radius(self._MOVABLE_FOOTPRINT[name])

            chosen_xy = None
            for _ in range(self._MAX_RESAMPLE):
                dx, dy = rng.uniform(-self._POS_JITTER_M, self._POS_JITTER_M, size=2)
                x, y = nominal[0] + dx, nominal[1] + dy
                # workspace bound (stay on the table)
                if abs(x) > self._WORKSPACE_HALF_M or abs(y) > self._WORKSPACE_HALF_M:
                    continue
                # robot-base keep-out
                if np.hypot(x, y) < self._BASE_KEEPOUT_M + radius:
                    continue
                # pairwise non-overlap with already-placed movables
                ok = True
                for (px, py, pr) in placed.values():
                    if np.hypot(x - px, y - py) < (radius + pr + self._OVERLAP_MARGIN_M):
                        ok = False
                        break
                if ok:
                    chosen_xy = (x, y)
                    break

            if chosen_xy is None:
                # Could not satisfy constraints within the budget: keep nominal
                # xy (deterministic) so the scene is still valid; report it.
                chosen_xy = (float(nominal[0]), float(nominal[1]))
                print(f"[Task] SDL_SEED={seed}: '{name}' fell back to nominal xy "
                      f"(no non-overlapping sample in {self._MAX_RESAMPLE} tries).")

            yaw = float(rng.uniform(-self._YAW_RANGE_RAD, self._YAW_RANGE_RAD))
            self.default_positions[name] = np.array([chosen_xy[0], chosen_xy[1], nominal[2]])
            self.default_orientations[name] = self._yaw_quat(yaw)
            placed[name] = (chosen_xy[0], chosen_xy[1], radius)

        # Log the resolved layout (real, not asserted) so each run self-documents.
        print(f"[Task] SDL_SEED={seed} randomized layout:")
        print(f"[Task]   home_arm(rad) = {np.round(self._home_arm, 4).tolist()}")
        for name in self._RANDOMIZE_ORDER:
            if name in self.default_positions:
                p = self.default_positions[name]; o = self.default_orientations[name]
                yaw_deg = np.rad2deg(2.0 * np.arctan2(o[3], o[0]))
                print(f"[Task]   {name:8s} xy=({p[0]:.4f},{p[1]:.4f}) yaw={yaw_deg:.1f}deg")
    
    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)

        # scene.add_default_ground_plane(z_position=-0.72)
        
        add_reference_to_stage(
            usd_path=os.path.join(ASSET_PATH, "lab", "world.usd"),
            prim_path="/World/background"
        )
        self.backgound = SingleXFormPrim(
            prim_path="/World/background",
            name="background"
        )
        self.backgound.set_world_pose(
            position=[0.0, 0.0, -0.71],
            orientation=[1, 0, 0, 0],
        )

        self.set_object(self.current_positions, self.current_orientations)
        self.set_robot(self.desired_tool)
        self.set_camera()
    

    def set_robot(self, desired_tool = None) -> FR5:

        if desired_tool is None:
            desired_tool = "empty"
        else:
            desired_tool = desired_tool.lower()

        if desired_tool == "empty":

            robot_asset_path = os.path.join(ASSET_PATH, "robot", "dcp_description", "usd", "fr5", "fr5.usd")
            robot_prim_path = find_unique_string_name(
                initial_name=self._robot_prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            robot_name = find_unique_string_name(
                initial_name=self._robot_name, is_unique_fn=lambda x: not self.scene.object_exists(x)
            )

            self._robot = FR5(
                prim_path=robot_prim_path,
                name=robot_name,
                usd_path=robot_asset_path,
                end_effector_prim_name="wrist3_link",
            )
            self._robot.joints_default_state = np.array([
                0.0, -1.05, -2.18, -1.57, 1.57, 0.0, # Arm joint position
            ])

            self.gripper_ag95.set_visibility(True)
            self.gripper_vgc10.set_visibility(True)
            self.gripper_dh3.set_visibility(True)

        elif desired_tool == "ag95":
            robot_asset_path = os.path.join(ASSET_PATH, "robot", "dcp_description", "usd", "fr5_ag95", "fr5_ag95.usd")
            robot_prim_path = find_unique_string_name(
                initial_name=self._robot_prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            robot_name = find_unique_string_name(
                initial_name=self._robot_name, is_unique_fn=lambda x: not self.scene.object_exists(x)
            )

            self._robot = FR5(
                prim_path=robot_prim_path,
                name=robot_name,
                usd_path=robot_asset_path,
                end_effector_prim_name="gripper_finger2_finger_tip_link",
                gripper_dof_names=["gripper_finger1_joint"],
                use_mimic_joints=True,
                gripper_open_position=np.array([0.0]),
                gripper_closed_position=np.array([0.6524]),
                deltas = np.array([-0.6524/5]) / get_stage_units(),
            )
            self._robot.joints_default_state = np.array([
                0.0, -1.05, -2.18, -1.57, 1.57, 0.0, # Arm joint position
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # Gripper joint position
            ])
            self.gripper_ag95.set_visibility(False)
            
        elif desired_tool == "vgc10":
            robot_asset_path = os.path.join(ASSET_PATH, "robot", "dcp_description", "usd", "fr5_vgc10", "fr5_vgc10.usd")
            robot_prim_path = find_unique_string_name(
                initial_name=self._robot_prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            robot_name = find_unique_string_name(
                initial_name=self._robot_name, is_unique_fn=lambda x: not self.scene.object_exists(x)
            )

            self._robot = FR5(
                prim_path=robot_prim_path,
                name=robot_name,
                usd_path=robot_asset_path,
                end_effector_prim_name="suction",
                is_surface_gripper=True,
                surface_gripper_path=robot_prim_path + "/SurfaceGripper",
            )

            self._robot.joints_default_state = np.array([
                0.0, -1.05, -2.18, -1.57, 1.57, 0.0, # Arm joint position
            ])
            self.gripper_vgc10.set_visibility(False)

        elif desired_tool == "dh3":
            robot_asset_path = os.path.join(ASSET_PATH, "robot", "dcp_description", "usd", "fr5_dh3", "fr5_dh3.usd")
            robot_prim_path = find_unique_string_name(
                initial_name=self._robot_prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            robot_name = find_unique_string_name(
                initial_name=self._robot_name, is_unique_fn=lambda x: not self.scene.object_exists(x)
            )

            self._robot = FR5(
                prim_path=robot_prim_path,
                name=robot_name,
                usd_path=robot_asset_path,
                end_effector_prim_name="finger3_tip_link",
                gripper_dof_names=["finger1_joint"],
                use_mimic_joints=True,
                gripper_open_position=np.array([0.0]),
                gripper_closed_position=np.array([1.16]),
                deltas = np.array([-1.16/5]) / get_stage_units()
            )
            self._robot.joints_default_state = np.array([
                0.0, -1.05, -2.18, -1.57, 1.57, 0.0, # Arm joint position
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # Gripper joint position
            ])
            self.gripper_dh3.set_visibility(False)

        else:
            raise ValueError("Available Grippers are only 'empty', 'ag95', 'vgc10', 'dh3'")

        # Apply the (optionally SDL_SEED-randomized) initial arm configuration to
        # the first 6 DOFs; any gripper DOFs keep their default 0.0. When
        # SDL_SEED is unset, self._home_arm == self._nominal_home_arm, so this is
        # a no-op that reproduces the original hard-coded home exactly.
        jd = np.asarray(self._robot.joints_default_state, dtype=float)
        home_arm = self._home_arm + self._TOOL_HOME_OFFSET.get(
            desired_tool, np.zeros(6)
        )
        jd[:6] = home_arm
        self._robot.joints_default_state = jd
        if desired_tool in self._TOOL_HOME_OFFSET:
            print(f"[Task]   home_arm(rad) for '{desired_tool}' (tool offset applied) "
                  f"= {np.round(home_arm, 4).tolist()}")

        self.current_tool = desired_tool

        self.scene.add(self._robot)

        return self._robot
    
    def set_object(self, current_positions = None, current_orientations = None) -> FR5:

        if current_positions is None:
            current_positions = self.default_positions
            current_orientations = self.default_orientations

        # spawn table
        self.table = self.scene.add(
            FixedCuboid(
                prim_path="/World/table",
                name="table",
                position=current_positions["table"],
                orientation=current_orientations["table"],
                scale=np.array([1.5, 1.5, 0.02]),
                size=1.0,
                color=np.array([0.922, 0.769, 0.569])
            )
        )

        # spawn stirrer -- box collider matching cuTAMP dims [0.18,0.18,0.09]
        # (resolves the triangle-mesh -> convexHull fallback on heat_device meshes)
        stirrer_usd_path = os.path.join(ASSET_PATH, "lab", "stirrer.usd")
        self.stirrer = create_box_collider_rigid(
            usd_path=stirrer_usd_path, prim_path="/World/stirrer", name="stirrer",
            position=current_positions["stirrer"],
            orientation=current_orientations["stirrer"],
            dims=[0.18, 0.18, 0.09],
        )
        self.scene.add(self.stirrer)

        # spawn beaker -- box collider matching cuTAMP dims [0.05,0.05,0.135]
        beaker_usd_path = os.path.join(ASSET_PATH, "lab", "beaker.usd")
        self.beaker = create_box_collider_rigid(
            usd_path=beaker_usd_path, prim_path="/World/beaker", name="beaker",
            position=current_positions["beaker"],
            orientation=current_orientations["beaker"],
            dims=[0.05, 0.05, 0.135],
        )
        self.scene.add(self.beaker)

        # spawn flask -- OPEN-TOPPED box collider, outer envelope matching cuTAMP
        # dims [0.07,0.07,0.12]. It has to be hollow because the Stir task drops a
        # stir bar into it: with the previous solid box the bar came to rest at
        # exactly (mouth + its own half-height) on every seed, i.e. on the lid, and
        # the "stir bar inside the vessel" terminal condition could never occur.
        # The outer envelope is unchanged, so the planner's Cuboid model of the
        # flask and every external contact stay exactly as before; only the
        # interior becomes real. FLASK_WALL_M gives a 64 mm clear opening.
        flask_usd_path = os.path.join(ASSET_PATH, "lab", "flask.usd")
        self.flask = create_hollow_box_collider_rigid(
            usd_path=flask_usd_path, prim_path="/World/flask", name="flask",
            position=current_positions["flask"],
            orientation=current_orientations["flask"],
            dims=[0.07, 0.07, 0.12], wall=self.FLASK_WALL_M,
        )
        self.scene.add(self.flask)

        # spawn magnet -- BOX collider matching cuTAMP dims [0.045,0.045,0.03]
        # (TAMP/tamp/src/envs/utils.py ENTITIES["magnet"] is a Cuboid, and the
        # magnet/stir-bar is a grasp candidate for dh3). cuTAMP plans EVERY
        # entity as a box, so the sim collider must be a box too for plan/grasp
        # self-consistency (REFACTOR.md III MINOR / II-2). This procedural stir
        # bar has no visual USD to strip, so -- unlike beaker/flask/box which use
        # object.create_box_collider_rigid on a referenced USD -- a DynamicCuboid
        # (box visual + matching box collider) is the right box primitive here.
        # size=1.0 * scale = full extents (same convention as the table above);
        # z=0.015 (=0.03/2) already rests the box on the table top; mass=0.1
        # matches the other movable box colliders.
        self.magnet = self.scene.add(
            DynamicCuboid(
                prim_path="/World/magnet",
                name="magnet",
                position=current_positions["magnet"],
                orientation=current_orientations["magnet"],
                scale=np.array(self.STIR_BAR_DIMS),
                size=1.0,
                color=np.array([0.0, 0.0, 1.0]),
                mass=0.1,
            )
        )

        # spawn box -- box collider matching cuTAMP dims [0.108,0.108,0.08]
        # (resolves the triangle-mesh -> convexHull fallback on the FluidBottle mesh)
        box_usd_path = os.path.join(ASSET_PATH, "lab", "bottle", "FluidBottle.usd")
        self.box = create_box_collider_rigid(
            usd_path=box_usd_path, prim_path="/World/box", name="box",
            position=current_positions["box"],
            orientation=current_orientations["box"],
            dims=[0.108, 0.108, 0.08],
        )
        self.scene.add(self.box)

        # --- AprilTag markers (perception in the loop, Reviewer 1 #1) ----------
        # Visual-only, so they cannot change any planning or contact result.
        # Enabled with SDL_APRILTAG so existing batches are unaffected; the first
        # gate is simply "does the detector see a tag through the rendered
        # cameras", which is what these fixed table tags answer. Tag ids and edge
        # size must match perception/apriltag_ros/cfg/tags_36h11.yaml.
        if os.environ.get("SDL_APRILTAG", "").strip() not in ("", "0"):
            tag_dir = os.path.join(ASSET_PATH, "apriltag")
            for tag_id, (tx, ty) in self.APRILTAG_TABLE_POSES.items():
                create_apriltag_plate(
                    prim_path="/World/apriltag_%d" % tag_id,
                    png_path=os.path.join(tag_dir, "tag36h11_%02d.png" % tag_id),
                    size=self.APRILTAG_SIZE_M,
                    position=(tx, ty, 0.001),   # lying on the table, just proud of it
                )
            print("[Task]   AprilTag plates spawned: %s (size %.3f m)"
                  % (sorted(self.APRILTAG_TABLE_POSES), self.APRILTAG_SIZE_M))

        # spawn box_goal -- STATIC goal tray. A static collider (no rigid body)
        # needs no mass/inertia, removing the previous PhysX
        # 'negative mass / invalid inertia' warning. Box collider matches
        # cuTAMP dims [0.15,0.15,0.01]. (REFACTOR II-2 fix #1)
        box_goal_usd_path = os.path.join(ASSET_PATH, "lab", "tray.usd")
        self.box_goal = create_box_collider_static(
            usd_path=box_goal_usd_path, prim_path="/World/box_goal", name="box_goal",
            position=current_positions["box_goal"],
            orientation=current_orientations["box_goal"],
            dims=[0.15, 0.15, 0.01],
        )
        self.scene.add(self.box_goal)

        # spawn gripper visual
        gripper_visual_asset_path = os.path.join(ASSET_PATH, "robot", "dcp_description", "usd", "gripper_visual")
        # ag95
        add_reference_to_stage(
            usd_path=os.path.join(gripper_visual_asset_path, "ag95", "ag95.usd"),
            prim_path="/World/gripper_visual/gripper_ag95"
        )
        self.gripper_ag95 = SingleXFormPrim(
            prim_path="/World/gripper_visual/gripper_ag95",
            name="gripper_ag95"
        )
        self.gripper_ag95.set_world_pose(
            position=[-0.6, -0.4, 0.25],
            orientation=[0, 0, 1, 0],
        )
        # vgc10
        add_reference_to_stage(
            usd_path=os.path.join(gripper_visual_asset_path, "vgc10", "vgc10.usd"),
            prim_path="/World/gripper_visual/gripper_vgc10"
        )
        self.gripper_vgc10 = SingleXFormPrim(
            prim_path="/World/gripper_visual/gripper_vgc10",
            name="gripper_vgc10"
        )
        self.gripper_vgc10.set_world_pose(
            position=[-0.6, 0.0, 0.25],
            orientation=[0, 0, 1, 0],
        )
        # dh3
        add_reference_to_stage(
            usd_path=os.path.join(gripper_visual_asset_path, "dh3", "dh3.usd"),
            prim_path="/World/gripper_visual/gripper_dh3"
        )
        self.gripper_dh3 = SingleXFormPrim(
            prim_path="/World/gripper_visual/gripper_dh3",
            name="gripper_dh3"
        )
        self.gripper_dh3.set_world_pose(
            position=[-0.6, 0.4, 0.25],
            orientation=[0, 0, 1, 0],
        )
        # gripper_base_link xform
        self.gripper_base_ag95 = SingleXFormPrim(
            prim_path="/World/gripper_visual/gripper_ag95/gripper_base_link",
            name="gripper_base_ag95"
        )
        self.gripper_base_vgc10 = SingleXFormPrim(
            prim_path="/World/gripper_visual/gripper_vgc10/gripper_base_link",
            name="gripper_base_vgc10"
        )
        self.gripper_base_dh3 = SingleXFormPrim(
            prim_path="/World/gripper_visual/gripper_dh3/gripper_base_link",
            name="gripper_base_dh3"
        )
        self.create_gripper_stand()


    def set_camera(self):
        self.cameras = []
        for i in range(2):
            camera = Camera(
                prim_path=f"/World/camera_{i+1}",
                frequency=30,
                resolution=(self.camera_info.width, self.camera_info.height),
                position=self.camera_positions[i],
                orientation=self.camera_orientations[i],
            )
            
            self.cameras.append(camera)


    def create_gripper_stand(self):
        asset_path = os.path.join(ASSET_PATH, "lab", "texture", "propile.jpg")
        aluminum_material = OmniPBR(
            prim_path="/World/aluminum_material",
            name="aluminum_material",
            color=np.array([1, 0, 0]),
            texture_path=asset_path,
            texture_scale=[1.0, 1.0],
            # texture_translate=[0.5, 0],
        )
        self.ag95_stand_base = self.scene.add(
            VisualCuboid(
                prim_path="/World/gripper_stand/ag95_stand/base",
                name="ag95_stand_base",
                position=np.array([-0.778, -0.4, 0.082]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                scale=np.array([0.05, 0.05, 0.3]),
                visual_material=aluminum_material,
            )
        )
        self.ag95_stand_arm = self.scene.add(
            VisualCuboid(
                prim_path="/World/gripper_stand/ag95_stand/arm",
                name="ag95_stand_arm",
                position=np.array([-0.70, -0.4, 0.224]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                scale=np.array([0.18, 0.05, 0.01]),
                visual_material=aluminum_material,
            )
        )

        self.vgc10_stand_base = self.scene.add(
            VisualCuboid(
                prim_path="/World/gripper_stand/vgc10_stand/base",
                name="vgc10_stand_base",
                position=np.array([-0.778, 0.0, 0.082]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                scale=np.array([0.05, 0.05, 0.3]),
                visual_material=aluminum_material,
            )
        )
        self.vgc10_stand_arm = self.scene.add(
            VisualCuboid(
                prim_path="/World/gripper_stand/vgc10_stand/arm",
                name="vgc10_stand_arm",
                position=np.array([-0.70, 0.0, 0.224]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                scale=np.array([0.18, 0.05, 0.01]),
                visual_material=aluminum_material,
            )
        )

        self.dh3_stand_base = self.scene.add(
            VisualCuboid(
                prim_path="/World/gripper_stand/dh3_stand/base",
                name="dh3_stand_base",
                position=np.array([-0.778, 0.4, 0.082]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                scale=np.array([0.05, 0.05, 0.3]),
                visual_material=aluminum_material,
            )
        )
        self.dh3_stand_arm = self.scene.add(
            VisualCuboid(
                prim_path="/World/gripper_stand/dh3_stand/arm",
                name="dh3_stand_arm",
                position=np.array([-0.70, 0.4, 0.224]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                scale=np.array([0.18, 0.05, 0.01]),
                visual_material=aluminum_material,
            )
        )
        

    def get_observations(self) -> Dict:

        self._ee_joint_idx = self._robot.get_dof_index("j6")
        
        # object pose
        table_pos, table_ori = self.table.get_world_pose()
        stirrer_pos, stirrer_ori = self.stirrer.get_world_pose()
        beaker_pos, beaker_ori = self.beaker.get_world_pose()
        flask_pos, flask_ori = self.flask.get_world_pose()
        magnet_pos, magnet_ori = self.magnet.get_world_pose()
        box_pos, box_ori = self.box.get_world_pose()
        box_goal_pos, box_goal_ori = self.box_goal.get_world_pose()

        # gripper base pose
        gripper_base_ag95_pos, gripper_base_ag95_ori = self.gripper_base_ag95.get_world_pose()
        gripper_base_vgc10_pos, gripper_base_vgc10_ori = self.gripper_base_vgc10.get_world_pose()
        gripper_base_dh3_pos, gripper_base_dh3_ori = self.gripper_base_dh3.get_world_pose()

        ft_data = self._robot.get_measured_joint_forces(self._ee_joint_idx)[0]
        scale_data = self.compute_scale_data(
            wrist_angle=self._robot.get_joint_positions(self._ee_joint_idx)[0],
            default_beaker_position=self.default_positions["beaker"],
            current_beaker_position=beaker_pos,
        )
        
        # observation dict
        observations = {
            "current_positions": {
                "table": table_pos,
                "stirrer": stirrer_pos,
                "beaker": beaker_pos,
                "flask": flask_pos,
                "magnet": magnet_pos,
                "box": box_pos,
                "box_goal": box_goal_pos,
            },
            "current_orientations": {
                "table": table_ori,
                "stirrer": stirrer_ori,
                "beaker": beaker_ori,
                "flask": flask_ori,
                "magnet": magnet_ori,
                "box": box_ori,
                "box_goal": box_goal_ori,
            },
            "gripper_base_position": {
                "empty": [0.0, 0.0, 0.0],
                "ag95": gripper_base_ag95_pos.tolist(),
                "vgc10": gripper_base_vgc10_pos.tolist(),
                "dh3": gripper_base_dh3_pos.tolist(),
            },
            "gripper_base_orientation": {
                "empty": [1.0, 0.0, 0.0, 0.0],
                "ag95": gripper_base_ag95_ori.tolist(),
                "vgc10": gripper_base_vgc10_ori.tolist(),
                "dh3": gripper_base_dh3_ori.tolist(),
            },
            "ft_data": ft_data,
            "scale_data": scale_data if scale_data is not None else 0.0,
        }

        return observations
    
    def compute_scale_data(self, wrist_angle, default_beaker_position, current_beaker_position):
        try:
            beaker_start_pos = np.array(default_beaker_position)
            beaker_pos = np.array(current_beaker_position)
            beaker_moved_distance = np.linalg.norm(beaker_pos - beaker_start_pos)
        except Exception as e:
            print(f"[Warning] compute_scale_data error: {e}")
            return self.scale_data # 에러 시에도 기존 값 반환 보장

        if beaker_moved_distance <= 0.022:
            self.scale_data = 0.0
            self.max_pour_angle = None
            return self.scale_data # None 대신 self.scale_data (0.0) 반환
            
        if self.max_pour_angle is None:
            # wrist_angle이 배열일 경우 스칼라 값으로 안전하게 추출 (로봇 설정에 따라 인덱스가 다를 수 있음)
            if isinstance(wrist_angle, (np.ndarray, list)):
                 self.max_pour_angle = float(wrist_angle[0]) 
            else:
                 self.max_pour_angle = float(wrist_angle)
            return self.scale_data # None 대신 self.scale_data 반환
        
        # 각도 비교 시에도 스칼라 변환 확인
        current_wrist_angle = float(wrist_angle[0]) if isinstance(wrist_angle, (np.ndarray, list)) else float(wrist_angle)

        if current_wrist_angle > self.max_pour_angle:
            delta_angle = current_wrist_angle - self.max_pour_angle
            flow = delta_angle * self.scale_gain
            self.scale_data += max(0.0, flow)
            self.max_pour_angle = current_wrist_angle
        
        return self.scale_data

    def get_grid_xy(self, name: str) -> Tuple[Optional[float], Optional[float]]:
        """
        그리드 이름을 입력받아 해당 영역의 중심 좌표 (x, y)를 반환합니다.
        """
        if name == "G1": return -0.15, -0.6
        elif name == "G2": return -0.15, 0.6
        elif name == "G3": return 0.15, -0.6
        elif name == "G4": return 0.15, -0.3
        elif name == "G5": return 0.15, 0.0
        elif name == "G6": return 0.15, 0.3
        elif name == "G7": return 0.15, 0.6
        elif name == "G8":  return 0.45, -0.6
        elif name == "G9":  return 0.45, -0.3
        elif name == "G10": return 0.45, 0.0
        elif name == "G11": return 0.45, 0.3
        elif name == "G12": return 0.45, 0.6

        # 정의되지 않은 이름일 경우
        return None, None
