import rclpy
from rclpy.node import Node
import numpy as np
from isaacsim import SimulationApp

from PIL import Image
import os

# --- Configuration ---
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "perception", "MODEST", "datasets", "dcp", "set3", "scene1"
)
CONFIG = {"renderer": "RaytracedLighting", "headless": False}
SAVE_INTERVAL = 10

def save_depth_image_png_uint16(
    depth_data: np.ndarray,
    out_dir: str,
    file_name: str,
) -> None:
    """
    32비트 float (미터 단위) 깊이 데이터를 16비트 uint16 (밀리미터 단위) PNG로 저장합니다.

    Args:
        depth_data: 깊이 배열 (float32, 미터 단위).
        out_dir: 출력 디렉토리 경로.
        file_name: 출력 파일 이름 (확장자는 .png로 설정).
    """
    
    # 1. 출력 디렉토리 확인 및 생성
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, file_name)

    # 2. 데이터 형식 및 차원 검증
    if len(depth_data.shape) == 3 and depth_data.shape[2] == 1:
        depth_data = depth_data.squeeze(axis=2)
    elif len(depth_data.shape) != 2:
        raise ValueError(f"예상되는 깊이 데이터 형태는 (H, W) 또는 (H, W, 1) 이었으나, {depth_data.shape}를 받았습니다.")

    valid_mask = np.isfinite(depth_data)
    
    depth_mm = np.zeros_like(depth_data, dtype=np.uint16)
    
    # 1000을 곱하여 밀리미터로 변환하고, 16비트 정수(uint16)의 최대값으로 클리핑합니다.
    # 이를 통해 65.535m를 초과하는 값은 최대값으로 제한됩니다.
    depth_mm[valid_mask] = np.clip(
        depth_data[valid_mask] * 1000.0, 
        0, 
        np.iinfo(np.uint16).max
    ).astype(np.uint16)
    
    # 4. 16비트 그레이스케일 모드('I;16')로 PNG 저장
    try:
        # PIL의 'I;16' 모드는 16비트 정수 이미지를 저장하는 데 사용됩니다.
        img = Image.fromarray(depth_mm, mode="I;16")
        img.save(file_path)
        print(f"✅ Metric depth (uint16 PNG)이 {file_path}에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 16비트 PNG 저장 중 오류 발생: {e}")
        raise

class Simulation(Node):
    
    def __init__(self):
        super().__init__("sdl_isaacsim")

        # 1. 시뮬레이션 앱 초기화 (이게 실행되어야 pxr을 불러올 수 있음)
        self.simulation_app = SimulationApp(CONFIG)

        from isaacsim.core.api import World
        from isaacsim.core.utils import extensions, prims, viewports, stage
        from isaacsim.core.utils.types import ArticulationAction

        extensions.enable_extension("isaacsim.ros2.sim_control")
        extensions.enable_extension("isaacsim.test.utils")
        from isaacsim.test.utils import save_depth_image

        self.prims = prims
        self.stage_utils = stage
        self.ArticulationAction = ArticulationAction
        self.World = World
        self.save_depth_image = save_depth_image

        self.simulation_app.update()

        # 2. 월드 생성
        self.world = self.World(stage_units_in_meters=1.0, physics_dt=1/120)

        viewports.set_camera_view(eye=np.array([1.2, 1.2, 0.8]), target=np.array([0, 0, 0.5]))

        # 3. Task 추가
        from task_camera import Task
        self.task = Task(name="task")
        self.world.add_task(self.task)
        self.world.reset()

        self.task.camera.initialize()
        self.task.camera.set_lens_distortion_model("pinhole")
        self.task.camera.add_normals_to_frame()
        self.task.camera.add_motion_vectors_to_frame()
        self.task.camera.add_occlusion_to_frame()
        self.task.camera.add_distance_to_image_plane_to_frame()
        self.task.camera.add_distance_to_camera_to_frame()
        
        if self.task.camera._annotator_device is not None and self.task.camera._annotator_device != "cuda":
            self.task.camera.add_bounding_box_2d_tight_to_frame()
            self.task.camera.add_bounding_box_2d_loose_to_frame()
            self.task.camera.add_bounding_box_3d_to_frame()
        self.task.camera.add_semantic_segmentation_to_frame()
        self.task.camera.add_instance_id_segmentation_to_frame()
        self.task.camera.add_instance_segmentation_to_frame()
        self.task.camera.add_pointcloud_to_frame()

        self.simulation_app.update()
        self.world.initialize_physics()
        self.world.play()

        # [중요] 불투명 재질 준비
        self.setup_opaque_material()
        
        # 타겟 설정
        self.target_object_paths = [
            self.task.beaker.prim_path,
            self.task.flask.prim_path,
        ]
        self.get_logger().info(f"Target Objects for Material Switch: {self.target_object_paths}")

        self.timer_period = 1/60
        self.timer = self.create_timer(self.timer_period, self.step_cb)
        self.get_logger().info("Simulation Start")
        
        self.step = 0

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

    def setup_opaque_material(self):
        """Depth/Seg 캡처용 불투명 재질(Matte Gray) 생성"""
        
        # [수정됨] pxr 모듈은 함수 안에서 import (SimulationApp 실행 이후이므로 안전함)
        from pxr import UsdShade, Sdf, Gf, Usd 
        
        stage = self.stage_utils.get_current_stage()
        looks_path = "/World/Looks"
        if not stage.GetPrimAtPath(looks_path):
            stage.DefinePrim(looks_path, "Scope")
        
        self.opaque_mat_path = looks_path + "/OpaqueMaterial"
        if not stage.GetPrimAtPath(self.opaque_mat_path):
            material = UsdShade.Material.Define(stage, self.opaque_mat_path)
            shader = UsdShade.Shader.Define(stage, self.opaque_mat_path + "/Shader")
            shader.CreateIdAttr("OmniPBR")
            shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
            shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(1.0)
            shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(1.0)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    def set_objects_material(self, use_opaque=True):
        """타겟 물체 재질 스위칭 (Glass <-> Opaque)"""
        
        # [수정됨] pxr 모듈은 함수 안에서 import
        from pxr import UsdShade, Sdf, Gf, Usd
        
        stage = self.stage_utils.get_current_stage()
        opaque_mat = UsdShade.Material.Get(stage, self.opaque_mat_path)
        
        for path in self.target_object_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid(): continue
            
            binding_api = UsdShade.MaterialBindingAPI(prim)
            if use_opaque:
                binding_api.Bind(opaque_mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            else:
                binding_api.UnbindDirectBinding()

    def step_cb(self):
        if self.simulation_app.is_running():

            # 1. Physics Step (Glass 상태로 물리 연산)
            self.world.step(render=True)

            is_save_step = (self.step % SAVE_INTERVAL == 0)
            
            if is_save_step:
                # 2. 랜덤 포즈 변경
                self.task.randomize_camera_pose()
                # self.task.randomize_object_pose()
                self.task.randomize_light()

                for _ in range(5):
                    self.world.step(render=True)
                    self.world.render() 

                # [PASS 1] RGB 저장 (Glass Material)
                obs_glass = self.task.get_observations()
                rgb_img = Image.fromarray(obs_glass["rgb"])
                depth_raw_data = obs_glass["depth"]
                # label_raw_data = obs_glass["label"]

                # [PASS 2] Depth/Label 저장 (Opaque Material)
                self.set_objects_material(use_opaque=True)
                for _ in range(10):
                    self.world.step(render=True)
                    self.world.render() 
                
                obs_true_opaque = self.task.get_observations()
                depth_true_data = obs_true_opaque["depth"]
                label_true_data = obs_true_opaque["label"]
                
                self.set_objects_material(use_opaque=False)
                for _ in range(10):
                    self.world.step(render=True)
                    self.world.render()

                # Label 처리
                if label_true_data is not None:
                    pixel_map = label_true_data["data"]
                    id_mapping = label_true_data["info"]["idToLabels"] 
                    
                    # 1. 타겟으로 삼을 상위 경로들을 정의합니다.
                    target_prefixes = ["/World/beaker", "/World/flask"]
                    
                    target_ids = []
                    
                    # 2. id_mapping을 순회하며 경로가 target_prefixes 중 하나로 시작하는지 확인합니다.
                    for str_id, prim_path in id_mapping.items():
                        # prim_path가 "/World/beaker" 또는 "/World/flask"로 시작하면 True
                        is_target = any(prim_path.startswith(prefix) for prefix in target_prefixes)
                        
                        if is_target:
                            target_ids.append(int(str_id))

                    # 3. 마스크 생성
                    if target_ids:
                        # target_ids에 포함된 픽셀만 True(1) -> 255(White), 나머지는 0(Black)
                        binary_mask = np.isin(pixel_map, target_ids).astype(np.uint8) * 255
                        
                        # 차원 문제 방지를 위해 squeeze (H, W, 1) -> (H, W)
                        if binary_mask.ndim == 3:
                            binary_mask = binary_mask.squeeze()
                            
                        label_img = Image.fromarray(binary_mask)
                    else:
                        # 타겟이 화면에 없으면 검은색 이미지 저장
                        # depth_data의 shape을 참고하거나 pixel_map shape 사용
                        empty_shape = pixel_map.shape[:2] if pixel_map.ndim >= 2 else (480, 640)
                        label_img = Image.fromarray(np.zeros(empty_shape, dtype=np.uint8))
                else:
                    label_img = None
                
                # 저장
                filename_base = f"{self.step*10:06d}"
                rgb_path = os.path.join(OUTPUT_DIR, f"{filename_base}-color.png")
                label_path = os.path.join(OUTPUT_DIR, f"{filename_base}-label.png")

                rgb_img.save(rgb_path)
                save_depth_image_png_uint16(depth_raw_data, OUTPUT_DIR, f"{filename_base}-depth_raw.png")
                save_depth_image_png_uint16(depth_true_data, OUTPUT_DIR, f"{filename_base}-depth_true.png")
                
                if label_img is not None:
                    label_img.save(label_path)
                
                self.get_logger().info(f"Saved step {self.step}")

            self.step += 1
            
        else:
            self.get_logger().info("Quit ROS2 Node")
            rclpy.try_shutdown()


def main(args=None):
    rclpy.init(args=args)
    sim_node = None
    try:
        sim_node = Simulation()
        rclpy.spin(sim_node)
    except KeyboardInterrupt:
        print("Shutdown...")
    finally:
        if sim_node:
            sim_node.world.stop()
            sim_node.simulation_app.close()
            sim_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()