from abc import ABC
from typing import Dict, Literal, List

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
from isaacsim.core.api.materials.omni_pbr import OmniPBR
from fr5 import FR5
import os
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.numpy import rotations as rot_utils
from isaacsim.sensors.camera import Camera, SingleViewDepthSensor
import yaml
from dataclasses import dataclass
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
import omni.usd
import math


ASSET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "TAMP", "tamp", "content", "assets"
)

@dataclass
class CameraInfo:
    """
    RealSense L515 카메라 공통 Base Class
    """
    # 기본값은 비워두거나 더미 데이터로 둡니다.
    yaml_data: str = """
        height: 480
        width: 640
        D: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        K: [601.46000163, 0.0, 334.89998372, 0.0, 601.5933431, 248.15334066, 0.0, 0.0, 1.0]
    """
    
    # [중요] 1.4um * (1920/640) = 4.2um
    pixel_size: float = 4.2
    f_stop: float = 2.0
    focus_distance: float = 0.8
    
    # 계산된 필드
    width: int = 0
    height: int = 0
    K: List[float] = None
    D: List[float] = None
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    horizontal_aperture: float = 0.0
    vertical_aperture: float = 0.0
    focal_length_x: float = 0.0
    focal_length_y: float = 0.0
    focal_length: float = 0.0
    diagonal: float = 0.0
    diagonal_fov: float = 0.0

    def __post_init__(self):
        # YAML이 비어있으면 실행하지 않음 (Safety check)
        if not self.yaml_data:
            return

        data = yaml.safe_load(self.yaml_data)
        
        self.width = data["width"]
        self.height = data["height"]
        self.K = data["K"]
        self.D = data["D"]
        
        self.fx = self.K[0]
        self.cx = self.K[2]
        self.fy = self.K[4]
        self.cy = self.K[5]

        # 물리적 센서 크기 (Aperture) 계산
        self.horizontal_aperture = self.pixel_size * 1e-3 * self.width
        self.vertical_aperture = self.pixel_size * 1e-3 * self.height
        
        # 초점 거리 (Focal Length) 계산
        self.focal_length_x = self.fx * self.pixel_size * 1e-3
        self.focal_length_y = self.fy * self.pixel_size * 1e-3
        self.focal_length = (self.focal_length_x + self.focal_length_y) / 2

        self.diagonal = 2 * math.sqrt(max(self.cx, self.width - self.cx) ** 2 + max(self.cy, self.height - self.cy) ** 2)
        self.diagonal_fov = 2 * math.atan2(self.diagonal, self.fx + self.fy) * 180 / math.pi


def set_world_pose_from_view(
    camera, eye: np.ndarray, target: np.ndarray
):
    # get up axis of current stage
    up_axis = stage_utils.get_stage_up_axis()
    # camera position and rotation in opengl convention
    orientation = rot_utils.rot_matrices_to_quats(
        create_rotation_matrix_from_view(eye, target, up_axis=up_axis)
    )
    camera.set_world_pose(eye, orientation, camera_axes="usd")


def create_rotation_matrix_from_view(
    eye: np.ndarray,
    target: np.ndarray,
    up_axis: Literal["Y", "Z"] = "Z",
):
    if up_axis == "Y":
        up_axis_vec = np.array([0, 1, 0], dtype=np.float32)
    elif up_axis == "Z":
        up_axis_vec = np.array([0, 0, 1], dtype=np.float32)
    else:
        raise ValueError(f"Invalid up axis: {up_axis}. Valid options are 'Y' and 'Z'.")

    # 1. Z-axis (Backward vector in view space, Forward in world space: target - eye)
    # Isaac Sim은 카메라의 view direction을 -Z 축으로 사용 (OpenGL convention)
    forward_dir = target - eye
    # Z-axis는 view direction의 정규화된 음수 벡터
    z_axis = -forward_dir / np.linalg.norm(forward_dir)

    # 2. X-axis (Right vector)
    # Z-axis와 Up-vector의 외적으로 Right vector를 구함
    x_axis = np.cross(up_axis_vec, z_axis)
    # Right vector 정규화
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # 3. Y-axis (Up vector in view space)
    # Z-axis와 X-axis의 외적으로 Up vector를 구함 (Gram-Schmidt orthogonalization)
    y_axis = np.cross(z_axis, x_axis)
    # Y-axis는 이미 정규화되어 있을 가능성이 높지만, 안전을 위해 정규화합니다.
    y_axis = y_axis / np.linalg.norm(y_axis)


    # 특이점 처리 (look direction이 up_axis_vec과 평행할 때)
    # Up-vector와 Z-axis가 평행하면 x_axis의 크기가 0이 됩니다.
    is_close = np.isclose(np.linalg.norm(x_axis), 0.0, atol=5e-3)
    if is_close:
        # 이 경우, forward_dir와 up_axis_vec이 평행합니다.
        # X-axis를 임의의 축으로 설정 (예: Up-vector가 Z일 때 X-axis를 Y나 X로)
        # Z축 업 벡터를 가정했을 때:
        if up_axis == "Z":
            x_axis = np.array([1.0, 0.0, 0.0]) # X축을 새로운 Right vector로 사용
        elif up_axis == "Y":
            x_axis = np.array([1.0, 0.0, 0.0]) # X축을 새로운 Right vector로 사용
        
        # 새 X-axis와 Z-axis를 사용하여 Y-axis 다시 계산
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis) # X-axis를 다시 계산하여 Y, Z축과 직교하도록 함
        x_axis = x_axis / np.linalg.norm(x_axis)

    
    # 회전 행렬 R (각 열이 카메라의 X, Y, Z 축 벡터를 나타냅니다)
    R = np.concatenate((x_axis[:, None], y_axis[:, None], z_axis[:, None]), axis=1)

    return R



class Task(ABC, BaseTask):

    def __init__(self, name: str, robot_prim_path: str, robot_name: str) -> None:
        
        BaseTask.__init__(self, name=name)
        self._asset_root_path = get_assets_root_path()
        if self._asset_root_path is None:
            raise Exception("Could not find Isaac Sim assets folder")
        self._robot_prim_path = robot_prim_path
        self._robot_name = robot_name

        self.camera_info = CameraInfo()

        self.random_eye_range = {
            "r": (0.8, 1.0),   # X축 랜덤 범위
            "theta": (-3.14, 3.14),   # Y축 랜덤 범위
            "height": (0.6, 0.8)    # Z축 높이 범위
        }
        self.random_target_range = {
            "x": (-0.1, 0.1), # 타겟(바라보는 점) 흔들림 범위
            "y": (-0.1, 0.1),
            "z": (0.0, 0.1)
        }

        self.random_object_pos_range = {
            "radius": (0.3, 0.7),
            "theta": (-np.pi, np.pi),
            "z": 0.05
        }

        self.random_light_range = {
            "intensity": (30000, 80000),    
            "exposure": (-1.0, 1.0),
            "radius": (0.2, 0.5),           
            "pos_x": (-2.0, 2.0),           
            "pos_y": (-2.0, 2.0),
            "pos_z": (2.0, 4.0)             
        }
        self.light_prim = None
        return
    
    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)

        # scene.add_default_ground_plane(z_position=-0.1)

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

        stage = omni.usd.get_context().get_stage()
        light_path = "/World/defaultGroundPlane/SphereLight"
        self.light_prim = stage.GetPrimAtPath(light_path)

        if not self.light_prim.IsValid():
            print(f"[Warning] Could not find light at {light_path}")
        else:
            # XformOp가 없는 경우를 대비해 초기화 (보통 GroundPlane에 포함되어 있음)
            xform = UsdGeom.Xformable(self.light_prim)
            if not self.light_prim.GetAttribute("xformOp:translate"):
                 xform.AddTranslateOp()

        self.set_robot()
        self.set_object()
        self.set_camera()
        
    
    def set_object(self) -> FR5:

        # Objects
        self.table = self.scene.add(
            FixedCuboid(
                prim_path="/World/table",
                name="table",
                position=np.array([0.0, 0.0, -0.01]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                scale=np.array([1.0, 1.0, 0.02]),
                size=1.0,
                color=np.array([0.922, 0.769, 0.569])
            )
        )

        self.beaker = self._create_hybrid_beaker(
            prim_path="/World/beaker",
            usd_path=os.path.join(ASSET_PATH, "lab", "beaker.usd"),
            position=np.array([0.51, -0.17, 0.015]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )

        self.flask = self._create_hollow_flask(
            prim_path="/World/flask",
            usd_path=os.path.join(ASSET_PATH, "lab", "flask.usd"),
            position=np.array([0.23, 0.2, 0.085]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
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
        self.create_apriltag()

    def set_robot(self) -> FR5:

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
        
        self.scene.add(self._robot)

        return self._robot


    def set_camera(self):

        # rgb camera
        self.camera = Camera(
            prim_path="/World/camera",
            position=np.array([0.0, 0.0, 25.0]),
            frequency=30,
            resolution=(self.camera_info.width, self.camera_info.height),
        )
        self.camera.set_focal_length(self.camera_info.focal_length / 10.0)
        self.camera.set_focus_distance(self.camera_info.focus_distance)
        self.camera.set_lens_aperture(self.camera_info.f_stop * 100.0)
        self.camera.set_horizontal_aperture(self.camera_info.horizontal_aperture / 10.0)
        self.camera.set_vertical_aperture(self.camera_info.vertical_aperture / 10.0)
        self.camera.set_clipping_range(0.25, 9.0)
        self.camera.set_projection_type("pinhole")
        self.camera.set_rational_polynomial_properties(
            self.camera_info.width, self.camera_info.height,
            self.camera_info.cx, self.camera_info.cy,
            self.camera_info.diagonal_fov, self.camera_info.D
        )
        set_world_pose_from_view(
            camera=self.camera,
            eye=np.array([1.0, 1.0, 1.0]),
            target=np.array([0, 0, 0])
        )

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
    
    def create_gripper_stand(self):
        asset_path = os.path.join(ASSET_PATH, "lab", "texture", "propile.jpg")
        aluminum_material = OmniPBR(
            prim_path="/World/aluminum_material",
            name="aluminum_material",
            color=np.array([0.8, 0.8, 0.8]),
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

    def create_apriltag(self):
        prim_name = "apriltag_0"
        prim_path = f"/World/apriltag/{prim_name}"
        usd_path = os.path.join(ASSET_PATH, "apriltag", f"{prim_name}.usd")
        add_reference_to_stage(
            usd_path=usd_path,
            prim_path=prim_path
        )
        self.backgound = SingleXFormPrim(
            prim_path=prim_path,
            name="prim_name"
        )
        self.backgound.set_world_pose(
            position=[0.4, 0.4, 0.01],
            orientation=[1, 0, 0, 0],
        )

        prim_name = "apriltag_1"
        prim_path = f"/World/apriltag/{prim_name}"
        usd_path = os.path.join(ASSET_PATH, "apriltag", f"{prim_name}.usd")
        add_reference_to_stage(
            usd_path=usd_path,
            prim_path=prim_path
        )
        self.backgound = SingleXFormPrim(
            prim_path=prim_path,
            name="prim_name"
        )
        self.backgound.set_world_pose(
            position=[-0.4, -0.4, 0.01],
            orientation=[1, 0, 0, 0],
        )

        prim_name = "apriltag_2"
        prim_path = f"/World/apriltag/{prim_name}"
        usd_path = os.path.join(ASSET_PATH, "apriltag", f"{prim_name}.usd")
        add_reference_to_stage(
            usd_path=usd_path,
            prim_path=prim_path
        )
        self.backgound = SingleXFormPrim(
            prim_path=prim_path,
            name="prim_name"
        )
        self.backgound.set_world_pose(
            position=[-0.4, 0.4, 0.01],
            orientation=[1, 0, 0, 0],
        )

        prim_name = "apriltag_3"
        prim_path = f"/World/apriltag/{prim_name}"
        usd_path = os.path.join(ASSET_PATH, "apriltag", f"{prim_name}.usd")
        add_reference_to_stage(
            usd_path=usd_path,
            prim_path=prim_path
        )
        self.backgound = SingleXFormPrim(
            prim_path=prim_path,
            name="prim_name"
        )
        self.backgound.set_world_pose(
            position=[0.4, -0.4, 0.01],
            orientation=[1, 0, 0, 0],
        )

    def randomize_camera_pose(self):
        """
        RGB와 Depth 카메라의 위치(eye)와 바라보는 곳(target)을 랜덤하게 변경합니다.
        """
        r = np.random.uniform(*self.random_eye_range["r"])
        theta = np.random.uniform(*self.random_eye_range["theta"])
        height = np.random.uniform(*self.random_eye_range["height"])

        rand_eye = np.array([
            r * np.cos(theta),
            r * np.sin(theta),
            height,
        ])

        rand_target = np.array([
            np.random.uniform(*self.random_target_range["x"]),
            np.random.uniform(*self.random_target_range["y"]),
            np.random.uniform(*self.random_target_range["z"]),
        ])

        set_world_pose_from_view(
            camera=self.camera,
            eye=rand_eye,
            target=rand_target
        )

    def randomize_object_pose(self):

        # beaker
        rand_pos = np.array([
            np.random.uniform(*self.random_object_pos_range["radius"]) \
                * np.cos(np.random.uniform(*self.random_object_pos_range["theta"])),
            np.random.uniform(*self.random_object_pos_range["radius"]) \
                * np.sin(np.random.uniform(*self.random_object_pos_range["theta"])),
            self.random_object_pos_range["z"],
        ])
        self.beaker.set_world_pose(position=rand_pos)

        # flask
        rand_pos = np.array([
            np.random.uniform(*self.random_object_pos_range["radius"]) \
                * np.cos(np.random.uniform(*self.random_object_pos_range["theta"])),
            np.random.uniform(*self.random_object_pos_range["radius"]) \
                * np.sin(np.random.uniform(*self.random_object_pos_range["theta"])),
            self.random_object_pos_range["z"],
        ])
        self.flask.set_world_pose(position=rand_pos)


    def get_observations(self) -> Dict:

        observations = {
            "rgb": self.camera.get_rgb(),
            "depth": self.camera.get_depth(),
            "label": self.camera.get_current_frame().get("instance_id_segmentation")
        }

        return observations