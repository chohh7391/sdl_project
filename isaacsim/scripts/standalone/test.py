# SPDX-License-Identifier: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import argparse
from isaacsim import SimulationApp

# 1. 시뮬레이션 앱 시작 (가장 먼저 실행되어야 함)
simulation_app = SimulationApp({"headless": False})

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Gf
import omni.kit.commands
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid, DynamicCylinder, DynamicSphere
import os

ASSET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "TAMP", "tamp", "content", "assets")

# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

# 2. World 생성
my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# --- Custom Object Setup Function ---
def create_hybrid_beaker(stage, root_path, usd_path):
    # 1. 부모 Xform 생성 (여기가 실제 움직이는 RigidBody)
    beaker_xform = UsdGeom.Xform.Define(stage, root_path)
    
    # RigidBody API 및 Mass API 적용
    UsdPhysics.RigidBodyAPI.Apply(beaker_xform.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(beaker_xform.GetPrim())
    mass_api.CreateMassAttr(0.1) 

    # ---------------------------------------------------------
    
    # 2. Visual 부분 (USD 파일 로드)
    visual_path = root_path + "/visual"
    add_reference_to_stage(usd_path=usd_path, prim_path=visual_path)

    visual_prim = stage.GetPrimAtPath(visual_path)
    xform_api = UsdGeom.XformCommonAPI(visual_prim)
    xform_api.SetTranslate((0, 0, 0.06))  # <--- 이 부분 추가!
    
    # [🔥 중요 수정] 불러온 USD 내부의 물리 속성 강제 삭제
    # 에러 메시지에 나온 경로: /World/beaker/visual/mesh/mesh
    # visual_path 뒤에 내부 구조(/mesh/mesh)를 붙여서 찾습니다.
    # 만약 내부 구조를 정확히 모른다면 Visual 하위를 순회해야 하지만, 
    # 에러 로그에 명확히 나왔으므로 직접 타게팅합니다.
    
    conflict_prim_path = visual_path + "/mesh/mesh" 
    conflict_prim = stage.GetPrimAtPath(conflict_prim_path)
    
    if conflict_prim.IsValid():
        # RigidBody 제거 (부모와 충돌 방지)
        if conflict_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            conflict_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            
        # Collision 제거 (우리가 만든 실린더만 충돌체로 쓰기 위해)
        if conflict_prim.HasAPI(UsdPhysics.CollisionAPI):
            conflict_prim.RemoveAPI(UsdPhysics.CollisionAPI)
            
    # 혹시 모를 상위 그룹(/mesh)에도 있을 수 있으니 체크
    parent_mesh_path = visual_path + "/mesh"
    parent_mesh_prim = stage.GetPrimAtPath(parent_mesh_path)
    if parent_mesh_prim.IsValid() and parent_mesh_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        parent_mesh_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

    # ---------------------------------------------------------

    # 3. Collision 부분 (Cylinder 생성) - 기존과 동일
    collision_path = root_path + "/collision"
    cylinder = UsdGeom.Cylinder.Define(stage, collision_path)
    
    radius = 0.04
    height = 0.12
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.CreateAxisAttr("Z") 
    UsdGeom.XformCommonAPI(cylinder).SetTranslate((0, 0, height/2))

    UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    imageable = UsdGeom.Imageable(cylinder.GetPrim())
    imageable.MakeInvisible()

    return root_path

# --- Main Logic ---

# 현재 스테이지 가져오기
stage = omni.usd.get_context().get_stage()

stirrer = my_world.scene.add(
    FixedCuboid(
        prim_path="/World/stirrer",
        name="stirrer",
        position=np.array([0.0, 0.0, 0.038]),
        # orientation=current_orientations["stirrer"],
        scale=np.array([0.1, 0.1, 0.075]),
        size=1.0,
    )
)

stirrer_usd_path = os.path.join(ASSET_PATH, "lab", "stirrer", "HeatingPlate.usd")

add_reference_to_stage(
    usd_path=stirrer_usd_path,
    prim_path="/World/stirrer_visual"
)
sitrrer_visual = SingleXFormPrim(
    prim_path="/World/stirrer_visual",
    name="stirrer_visual",
)
sitrrer_visual.set_world_pose(
    position=np.array([0.0, 0.0, 0.0005]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
)

# 4. 시뮬레이션 루프
my_world.reset()
# print(f"Beaker created at {beaker_prim_path} with Hybrid Collision.")

step_count = 0
while simulation_app.is_running() and not simulation_app.is_exiting():
    my_world.step(render=True)
    
    step_count += 1
    # if args.test is True and step_count > 100: # Test 모드일 때 짧게 종료
    #     break
    
    # if step_count == 10000 and not args.test:
    #     print("Finished simulating for 10000 steps")
    #     break

simulation_app.close()