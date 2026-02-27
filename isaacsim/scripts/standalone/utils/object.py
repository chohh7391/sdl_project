import math
import numpy as np
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
import numpy as np
from pxr import UsdGeom, UsdPhysics


def create_hybrid_beaker(prim_path, usd_path, position, orientation):
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
    side_length = 0.055
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

def create_hybrid_box(prim_path, usd_path, position, orientation, scale_size):
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

def create_hollow_flask(prim_path, usd_path, position, orientation):
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