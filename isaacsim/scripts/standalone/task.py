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
from isaacsim.core.api.objects import FixedCuboid, DynamicCylinder, VisualCuboid
from isaacsim.core.api.materials.omni_pbr import OmniPBR
from isaacsim.ros2.bridge import read_camera_info
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.prims import SingleRigidPrim

from fr5 import FR5

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from object import create_hybrid_beaker, create_hybrid_box, create_hollow_flask, create_single_rigid_prim_from_usd
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
            "stirrer": np.array([-0.15, 0.6, 0.045]),
            # "beaker": np.array([random_x, random_y, 0.067]),
            "beaker": np.array([0.51, -0.17, 0.07]),
            "flask": np.array([flask_x + noise[0], flask_y + noise[1], 0.07]),
            # "flask": np.array([0.43, -0.086, 0.07]),
            "magnet": np.array([0.3, 0.416, 0.015]),
            # "box" : np.array([box_x, box_y, 0.06]),
            "box" : np.array([0.35, -0.5, 0.06]),
            "box_goal" : np.array([-0.15, -0.6, 0.006]),
        }
        self.default_orientations = {
            "table": np.array([1.0, 0.0, 0.0, 0.0]),
            "stirrer": np.array([1.0, 0.0, 0.0, 0.0]),
            "beaker": np.array([1.0, 0.0, 0.0, 0.0]),
            "flask": np.array([1.0, 0.0, 0.0, 0.0]),
            "magnet": np.array([1.0, 0.0, 0.0, 0.0]),
            "box": np.array([1.0, 0.0, 0.0, 0.0]),
            "box_goal": np.array([1.0, 0.0, 0.0, 0.0]),
        }

        self.scale_data = 0.0
        self.scale_gain = 50.0

        return
    
    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)

        scene.add_default_ground_plane(z_position=-0.72)
        
        # add_reference_to_stage(
        #     usd_path=os.path.join(ASSET_PATH, "lab", "world.usd"),
        #     prim_path="/World/background"
        # )
        # self.backgound = SingleXFormPrim(
        #     prim_path="/World/background",
        #     name="background"
        # )
        # self.backgound.set_world_pose(
        #     position=[0.0, 0.0, -0.71],
        #     orientation=[1, 0, 0, 0],
        # )

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

        # spawn stirrer
        stirrer_usd_path = os.path.join(ASSET_PATH, "lab", "stirrer.usd")
        self.stirrer = create_single_rigid_prim_from_usd(
            usd_path=stirrer_usd_path, prim_path="/World/stirrer", name="stirrer",
            position=current_positions["stirrer"],
            orientation=current_orientations["stirrer"],
        )
        self.scene.add(self.stirrer)

        # spawn beaker
        beaker_usd_path = os.path.join(ASSET_PATH, "lab", "beaker.usd")
        self.beaker = create_single_rigid_prim_from_usd(
            usd_path=beaker_usd_path, prim_path="/World/beaker", name="beaker",
            position=current_positions["beaker"],
            orientation=current_orientations["beaker"],
        )
        self.scene.add(self.beaker)

        # spawn flask
        flask_usd_path = os.path.join(ASSET_PATH, "lab", "flask.usd")
        self.flask = create_single_rigid_prim_from_usd(
            usd_path=flask_usd_path, prim_path="/World/flask", name="flask",
            position=current_positions["flask"],
            orientation=current_orientations["flask"],
        )
        self.scene.add(self.flask)

        # spawn magnet
        self.magnet = self.scene.add(
            DynamicCylinder(
                prim_path="/World/magnet",
                name="magnet",
                position=current_positions["magnet"],
                orientation=current_orientations["magnet"],
                radius=0.012,
                height=0.03,
                color=np.array([0.0, 0.0, 1.0]),
            )
        )

        # spawn box
        box_usd_path = os.path.join(ASSET_PATH, "lab", "bottle", "FluidBottle.usd")
        self.box = create_single_rigid_prim_from_usd(
            usd_path=box_usd_path, prim_path="/World/box", name="box",
            position=current_positions["box"],
            orientation=current_orientations["box"],
        )
        self.scene.add(self.box)

        # spawn box_goal
        box_goal_usd_path = os.path.join(ASSET_PATH, "lab", "tray.usd")
        self.box_goal = create_single_rigid_prim_from_usd(
            usd_path=box_goal_usd_path, prim_path="/World/box_goal", name="box_goal",
            position=current_positions["box_goal"],
            orientation=current_orientations["box_goal"],
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
