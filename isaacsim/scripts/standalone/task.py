from abc import ABC
from typing import Dict
import math

import numpy as np
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import SingleXFormPrim, SingleRigidPrim
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units, get_current_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.storage.native import get_assets_root_path
import numpy as np
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid, DynamicCylinder, DynamicSphere, VisualCuboid
from fr5 import FR5
import os

from pxr import Usd, UsdGeom, Gf, PhysxSchema, UsdPhysics
from isaacsim.core.api.materials.omni_pbr import OmniPBR
from isaacsim.core.utils.prims import create_prim


ASSET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "TAMP", "tamp", "content", "assets"
)


class Task(ABC, BaseTask):

    def __init__(self, name: str, robot_prim_path: str, robot_name: str) -> None:
        
        BaseTask.__init__(self, name=name)
        self._asset_root_path = get_assets_root_path()
        if self._asset_root_path is None:
            raise Exception("Could not find Isaac Sim assets folder")
        self._robot_prim_path = robot_prim_path
        self._robot_name = robot_name

        self.current_positions = None
        self.current_orientations = None
        self.desired_tool = None
        self.current_tool = None

        self.default_positions = {
            "table": np.array([0.0, 0.0, -0.01]),
            "stirrer": np.array([-0.04, 0.45, 0.038]),
            "stirrer_visual": np.array([0.01, 0.45, 0.045]),
            "beaker": np.array([0.51, -0.17, 0.015]),
            "flask": np.array([0.23, 0.2, 0.085]), # 0.23
            "magnet": np.array([0.3, 0.416, 0.015]),
            "box" : np.array([0.35, -0.5, 0.06]),
            "box_goal" : np.array([-0.036, -0.52, 0.006]),
        }
        self.default_orientations = {
            "table": np.array([1.0, 0.0, 0.0, 0.0]),
            "stirrer": np.array([1.0, 0.0, 0.0, 0.0]),
            "stirrer_visual": np.array([0.7071, 0.0, 0.0, 0.7071]),
            "beaker": np.array([1.0, 0.0, 0.0, 0.0]),
            "flask": np.array([1.0, 0.0, 0.0, 0.0]),
            "magnet": np.array([1.0, 0.0, 0.0, 0.0]),
            "box": np.array([1.0, 0.0, 0.0, 0.0]),
            "box_goal": np.array([1.0, 0.0, 0.0, 0.0]),
        }

        return
    
    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)

        # scene.add_default_ground_plane(z_position=-0.72)
        
        add_reference_to_stage(
            usd_path=os.path.join(ASSET_PATH, "lab", "World.usd"),
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
    
    # def _create_hybrid_beaker(self, prim_path, usd_path, position, orientation):
    #     stage = get_current_stage()
        
    #     # 1. 부모 Xform 생성 (여기가 실제 움직이는 RigidBody)
    #     beaker_xform = UsdGeom.Xform.Define(stage, prim_path)
        
    #     # RigidBody API 및 Mass API 적용
    #     UsdPhysics.RigidBodyAPI.Apply(beaker_xform.GetPrim())
    #     mass_api = UsdPhysics.MassAPI.Apply(beaker_xform.GetPrim())
    #     mass_api.CreateMassAttr(0.1) 

    #     # ---------------------------------------------------------
        
    #     # 2. Visual 부분 (USD 파일 로드)
    #     visual_path = prim_path + "/visual"
    #     add_reference_to_stage(usd_path=usd_path, prim_path=visual_path)

    #     visual_prim = stage.GetPrimAtPath(visual_path)
    #     xform_api = UsdGeom.XformCommonAPI(visual_prim)
    #     # Visual 위치 보정 (실린더와 똑같이 올려줍니다)
    #     xform_api.SetTranslate((0, 0, 0.06)) 
        
    #     # 불러온 USD 내부의 물리 속성 강제 삭제
    #     conflict_prim_path = visual_path + "/mesh/mesh" 
    #     conflict_prim = stage.GetPrimAtPath(conflict_prim_path)
        
    #     if conflict_prim.IsValid():
    #         # RigidBody 제거 (부모와 충돌 방지)
    #         if conflict_prim.HasAPI(UsdPhysics.RigidBodyAPI):
    #             conflict_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                
    #         # Collision 제거 (우리가 만든 실린더만 충돌체로 쓰기 위해)
    #         if conflict_prim.HasAPI(UsdPhysics.CollisionAPI):
    #             conflict_prim.RemoveAPI(UsdPhysics.CollisionAPI)
                
    #     # 혹시 모를 상위 그룹(/mesh)에도 있을 수 있으니 체크
    #     parent_mesh_path = visual_path + "/mesh"
    #     parent_mesh_prim = stage.GetPrimAtPath(parent_mesh_path)
    #     if parent_mesh_prim.IsValid() and parent_mesh_prim.HasAPI(UsdPhysics.RigidBodyAPI):
    #         parent_mesh_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

    #     # ---------------------------------------------------------

    #     # 3. Collision 부분 (Cylinder 생성)
    #     collision_path = prim_path + "/collision"
    #     cylinder = UsdGeom.Cylinder.Define(stage, collision_path)
        
    #     radius = 0.038
    #     height = 0.12
    #     cylinder.CreateRadiusAttr(radius)
    #     cylinder.CreateHeightAttr(height)
    #     cylinder.CreateAxisAttr("Z") 
    #     UsdGeom.XformCommonAPI(cylinder).SetTranslate((0, 0, height/2))

    #     UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    #     imageable = UsdGeom.Imageable(cylinder.GetPrim())
    #     imageable.MakeInvisible()

    #     # 4. SingleRigidPrim으로 래핑하여 반환
    #     return SingleRigidPrim(
    #         prim_path=prim_path,
    #         name="beaker_rigid",
    #         position=position,
    #         orientation=orientation
    #     )
    def _create_hybrid_beaker(self, prim_path, usd_path, position, orientation):
        stage = get_current_stage()
        
        # 1. 부모 Xform 생성 (RigidBody)
        beaker_xform = UsdGeom.Xform.Define(stage, prim_path)
        
        # RigidBody API 및 Mass API 적용
        UsdPhysics.RigidBodyAPI.Apply(beaker_xform.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(beaker_xform.GetPrim())
        mass_api.CreateMassAttr(0.1) 

        # ---------------------------------------------------------
        
        # 2. Visual 부분 (USD 파일 로드)
        visual_path = prim_path + "/visual"
        add_reference_to_stage(usd_path=usd_path, prim_path=visual_path)

        visual_prim = stage.GetPrimAtPath(visual_path)
        xform_api = UsdGeom.XformCommonAPI(visual_prim)
        # Visual 위치 보정
        xform_api.SetTranslate((0, 0, 0.06)) 
        
        # 불러온 USD 내부의 물리 속성 강제 삭제
        targets_to_clean = [visual_path, visual_path + "/mesh", visual_path + "/mesh/mesh"]
        for path in targets_to_clean:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)

        # ---------------------------------------------------------

        # 3. Collision 부분 (Cube로 변경 - 사각기둥)
        collision_path = prim_path + "/collision"
        box = UsdGeom.Cube.Define(stage, collision_path)
        
        # 기존 Cylinder 제원: radius=0.038, height=0.12
        # Box 제원 변환: width/depth = 0.038 * 2 = 0.076
        side_length = 0.05
        height = 0.12
        
        # UsdGeom.Cube의 기본 size는 2.0이므로 scale을 (목표크기 / 2.0)으로 설정
        box_scale = np.array([side_length, side_length, height]) / 2.0
        
        UsdGeom.XformCommonAPI(box).SetScale(tuple(box_scale))
        UsdGeom.XformCommonAPI(box).SetTranslate((0, 0, height/2))

        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        imageable = UsdGeom.Imageable(box.GetPrim())
        imageable.MakeInvisible()

        # 4. SingleRigidPrim으로 래핑하여 반환
        return SingleRigidPrim(
            prim_path=prim_path,
            name="beaker_rigid",
            position=position,
            orientation=orientation
        )
    
    def _create_hybrid_box(self, prim_path, usd_path, position, orientation, scale_size):
        """
        Visual은 USD 파일(FluidBottle)을 사용하고, 
        Collision은 단순한 Cube(Box)를 사용하는 하이브리드 박스 생성
        scale_size: (x, y, z) 실제 미터 단위 크기
        """
        stage = get_current_stage()
        
        # 1. 부모 Xform 생성 (RigidBody)
        box_xform = UsdGeom.Xform.Define(stage, prim_path)
        
        # RigidBody API 및 Mass API 적용
        UsdPhysics.RigidBodyAPI.Apply(box_xform.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(box_xform.GetPrim())
        mass_api.CreateMassAttr(0.1) # 질량 설정

        # ---------------------------------------------------------
        
        # 2. Visual 부분 (USD 파일 로드)
        visual_path = prim_path + "/visual"
        add_reference_to_stage(usd_path=usd_path, prim_path=visual_path)

        visual_prim = stage.GetPrimAtPath(visual_path)
        xform_api = UsdGeom.XformCommonAPI(visual_prim)
        # xform_api.SetTranslate((0, 0, 0)) 
        
        # 불러온 USD 내부의 물리 속성 강제 삭제 (충돌 방지)
        # USD 구조에 따라 경로가 다를 수 있으므로 visual 루트와 하위 mesh를 모두 체크
        targets_to_clean = [visual_path, visual_path + "/mesh", visual_path + "/mesh/mesh"]
        
        for path in targets_to_clean:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)

        # ---------------------------------------------------------

        # 3. Collision 부분 (Cube 생성)
        collision_path = prim_path + "/collision"
        cube = UsdGeom.Cube.Define(stage, collision_path)
    
        half_extents = np.array(scale_size) / 2.0
        
        UsdGeom.XformCommonAPI(cube).SetScale((half_extents[0], half_extents[1], half_extents[2]))
        UsdGeom.XformCommonAPI(cube).SetTranslate((0, 0, scale_size[2]/2))

        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        imageable = UsdGeom.Imageable(cube.GetPrim())
        imageable.MakeInvisible()

        return SingleRigidPrim(
            prim_path=prim_path,
            name="box_rigid",
            position=position,
            orientation=orientation
        )
    
    def _create_hollow_flask(self, prim_path, usd_path, position, orientation):
        """
        Visual: USD 파일 (Beaker)
        Collision: 바닥(Cylinder) + 12개의 벽(Cube)으로 구성된 속이 빈 원기둥 근사
        """
        stage = get_current_stage()
        
        # 1. 부모 Xform 생성 (RigidBody)
        beaker_xform = UsdGeom.Xform.Define(stage, prim_path)
        
        UsdPhysics.RigidBodyAPI.Apply(beaker_xform.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(beaker_xform.GetPrim())
        mass_api.CreateMassAttr(0.1) 

        # ---------------------------------------------------------
        
        # 2. Visual 부분 (USD 로드 및 정리)
        visual_path = prim_path + "/visual"
        add_reference_to_stage(usd_path=usd_path, prim_path=visual_path)
        
        visual_prim = stage.GetPrimAtPath(visual_path)
        xform_api = UsdGeom.XformCommonAPI(visual_prim)
        # Visual 위치 보정 (필요 시)
        xform_api.SetTranslate((0, 0, 0.06)) 

        # USD 내부 물리 속성 제거
        targets_to_clean = [visual_path, visual_path + "/mesh", visual_path + "/mesh/mesh"]
        for path in targets_to_clean:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)

        # ---------------------------------------------------------

        # 3. Collision 부분 (다각형 벽 생성)
        collision_root = prim_path + "/collision"
        UsdGeom.Xform.Define(stage, collision_root)

        # 비커 제원 설정
        radius = 0.04       # 반지름
        height = 0.11       # 높이
        wall_thickness = 0.005 # 벽 두께
        bottom_thickness = 0.005 # 바닥 두께
        num_segments = 12   # 벽의 개수 (12각형)

        # (1) 바닥 (Cylinder)
        bottom_path = collision_root + "/bottom"
        bottom = UsdGeom.Cylinder.Define(stage, bottom_path)
        UsdPhysics.CollisionAPI.Apply(bottom.GetPrim())
        bottom.CreateRadiusAttr(radius)
        bottom.CreateHeightAttr(bottom_thickness)
        bottom.CreateAxisAttr("Z")
        # 바닥 위치 설정
        UsdGeom.XformCommonAPI(bottom).SetTranslate((0, 0, bottom_thickness/2))
        UsdGeom.Imageable(bottom.GetPrim()).MakeInvisible()

        # (2) 벽 (12개의 Cube를 원형으로 배치)
        # 한 변의 길이(너비) 계산: 원에 내접/외접하는 다각형의 변 길이
        # 틈이 없어야 하므로 약간 넉넉하게 탄젠트 공식 사용 + 오버랩
        segment_width = 2 * radius * math.tan(math.pi / num_segments) + 0.002
        
        for i in range(num_segments):
            angle_rad = (2 * math.pi / num_segments) * i
            angle_deg = math.degrees(angle_rad)

            wall_path = collision_root + f"/wall_{i}"
            wall = UsdGeom.Cube.Define(stage, wall_path)
            UsdPhysics.CollisionAPI.Apply(wall.GetPrim())

            # 크기 설정 (Cube 기본 2.0)
            # x: 두께, y: 너비, z: 높이
            w_scale = np.array([wall_thickness, segment_width, height]) / 2.0
            UsdGeom.XformCommonAPI(wall).SetScale(tuple(w_scale))

            # 위치 계산 (극좌표 -> 직교좌표)
            # 반지름만큼 떨어진 곳에 배치
            x = radius * math.cos(angle_rad)
            y = radius * math.sin(angle_rad)
            z = height / 2 + bottom_thickness # 바닥 위에 얹기

            # 위치 및 회전 설정
            UsdGeom.XformCommonAPI(wall).SetTranslate((x, y, z))
            # Z축 기준으로 회전 (Isaac Sim/USD는 Euler Angle 순서 주의, 보통 ZYX or XYZ)
            UsdGeom.XformCommonAPI(wall).SetRotate((0, 0, angle_deg))
            
            UsdGeom.Imageable(wall.GetPrim()).MakeInvisible()

        # 4. 반환
        return SingleRigidPrim(
            prim_path=prim_path,
            name="flask_rigid",
            position=position,
            orientation=orientation
        )


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
                deltas = np.array([-0.4]) / get_stage_units(),
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
                deltas = np.array([-0.2]) / get_stage_units() 
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


        # Objects
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
        self.stirrer = self.scene.add(
            FixedCuboid(
                prim_path="/World/stirrer",
                name="stirrer",
                position=current_positions["stirrer"],
                orientation=current_orientations["stirrer"],
                scale=np.array([0.1, 0.1, 0.075]),
                size=1.0,
            )
        )

        stirrer_usd_path = os.path.join(ASSET_PATH, "lab", "heat_device.usd")
        add_reference_to_stage(
            usd_path=stirrer_usd_path,
            prim_path="/World/stirrer_visual"
        )
        self.stirrer_visual = SingleXFormPrim(
            prim_path="/World/stirrer_visual",
            name="stirrer_visual",
        )
        self.stirrer_visual.set_world_pose(
            position=current_positions["stirrer_visual"],
            orientation=current_orientations["stirrer_visual"],
        )

        beaker_usd_path = os.path.join(ASSET_PATH, "lab", "beaker.usd")
        self.beaker = self._create_hybrid_beaker(
            prim_path="/World/beaker",
            usd_path=beaker_usd_path,
            position=current_positions["beaker"],
            orientation=current_orientations["beaker"]
        )
        
        self.scene.add(self.beaker)

        flask_usd_path = os.path.join(ASSET_PATH, "lab", "flask.usd")
        self.flask = self._create_hollow_flask(
            prim_path="/World/flask",
            usd_path=flask_usd_path,
            position=current_positions["flask"],
            orientation=current_orientations["flask"]
        )
        self.scene.add(self.flask)

        self.magnet = self.scene.add(
            DynamicCylinder(
                prim_path="/World/magnet",
                name="magnet",
                position=current_positions["magnet"],
                orientation=current_orientations["magnet"],
                radius=0.012,
                height=0.03,
                color=np.array([0.0, 0.0, 1.0])
            )
        )

        box_usd_path = os.path.join(ASSET_PATH, "lab", "bottle", "FluidBottle.usd")
        box_size = np.array([0.1, 0.1, 0.08]) 
        
        self.box = self._create_hybrid_box(
            prim_path="/World/box",
            usd_path=box_usd_path,
            position=current_positions["box"],
            orientation=current_orientations["box"],
            scale_size=box_size
        )
        self.scene.add(self.box)

        self.box_goal = self.scene.add(
            FixedCuboid(
                prim_path="/World/box_goal",
                name="box_goal",
                position=current_positions["box_goal"],
                orientation=current_orientations["box_goal"],
                scale=np.array([0.2, 0.2, 0.01]),
                size=1.0,
                color=np.array([0.922, 0.769, 0.569]),
                visible=False,
            )
        )

        add_reference_to_stage(
            usd_path=os.path.join(ASSET_PATH, "lab", "Tray.usd"),
            prim_path="/World/box_goal_visual"
        )
        self.box_goal_visual = SingleXFormPrim(
            prim_path="/World/box_goal_visual",
            name="box_goal_visual"
        )
        self.box_goal_visual.set_world_pose(
            position=np.array([-0.036, -0.52, 0.001]),
            orientation=[1, 0, 0, 0],
        )


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
        
        # object pose
        table_pos, table_ori = self.table.get_world_pose()
        stirrer_pos, stirrer_ori = self.stirrer.get_world_pose()
        stirrer_visual_pos, stirrer_visual_ori = self.stirrer_visual.get_world_pose()
        beaker_pos, beaker_ori = self.beaker.get_world_pose()
        flask_pos, flask_ori = self.flask.get_world_pose()
        magnet_pos, magnet_ori = self.magnet.get_world_pose()
        box_pos, box_ori = self.box.get_world_pose()
        box_goal_pos, box_goal_ori = self.box_goal.get_world_pose()

        # gripper base pose
        gripper_base_ag95_pos, gripper_base_ag95_ori = self.gripper_base_ag95.get_world_pose()
        gripper_base_vgc10_pos, gripper_base_vgc10_ori = self.gripper_base_vgc10.get_world_pose()
        gripper_base_dh3_pos, gripper_base_dh3_ori = self.gripper_base_dh3.get_world_pose()
        
        # observation dict
        observations = {
            "current_positions": {
                "table": table_pos,
                "stirrer": stirrer_pos,
                "stirrer_visual": stirrer_visual_pos.tolist(),
                "beaker": beaker_pos,
                "flask": flask_pos,
                "magnet": magnet_pos,
                "box": box_pos,
                "box_goal": box_goal_pos,
            },
            "current_orientations": {
                "table": table_ori,
                "stirrer": stirrer_ori,
                "stirrer_visual": stirrer_visual_ori.tolist(),
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
        }

        return observations
    
