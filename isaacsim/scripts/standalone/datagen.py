import rclpy
from rclpy.node import Node
import numpy as np
from isaacsim import SimulationApp

import time
import os, sys
import json
from PIL import Image
import carb


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "perception", "MODEST", "datasets", "dcp", "set3", "scene1"
)
ROBOT_STAGE_PATH = "/World/Robot"
CONFIG = {"renderer": "RaytracedLighting", "headless": False}
SAVE_INTERVAL = 10


def save_depth_image_png_uint16(depth_data: np.ndarray, out_dir: str, file_name: str) -> None:
    """32비트 float 깊이 데이터를 16비트 uint16 PNG로 저장"""
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, file_name)

    if len(depth_data.shape) == 3 and depth_data.shape[2] == 1:
        depth_data = depth_data.squeeze(axis=2)
    elif len(depth_data.shape) != 2:
        raise ValueError(f"Unexpected depth shape: {depth_data.shape}")

    valid_mask = np.isfinite(depth_data)
    depth_mm = np.zeros_like(depth_data, dtype=np.uint16)
    
    depth_mm[valid_mask] = np.clip(
        depth_data[valid_mask] * 1000.0, 
        0, 
        np.iinfo(np.uint16).max
    ).astype(np.uint16)
    
    try:
        img = Image.fromarray(depth_mm, mode="I;16")
        img.save(file_path)
    except Exception as e:
        print(f"❌ Depth PNG Save Error: {e}")
        raise

def get_relative_transform(source_pos, source_rot_q, target_pos, target_rot_q):
    """
    Source(예: 카메라) 기준 Target(예: 물체)의 상대 포즈 계산
    rot_q는 (w, x, y, z) 순서여야 함.
    """
    from isaacsim.core.utils.rotations import quat_to_rot_matrix

    # 1. 위치: t_rel = R_src^T * (t_tgt - t_src)
    R_src = quat_to_rot_matrix(source_rot_q) # Source -> World Rotation Matrix
    p_rel = R_src.T @ (target_pos - source_pos)
    
    R_tgt = quat_to_rot_matrix(target_rot_q)
    R_rel = R_src.T @ R_tgt # R_source_to_target

    return p_rel, R_rel


class Simulation(Node):
    
    def __init__(self):
        
        super().__init__("sdl_isaacsim")

        self.simulation_app = SimulationApp(CONFIG)

        # --- [추가된 설정 시작] ---
        # 렌더링 설정 가져오기
        settings = carb.settings.get_settings()
        settings.set_bool("/rtx/post/motionblur/enabled", False)
        settings.set_float("/rtx/post/dlss/execMode", 0) 
        settings.set_bool("/rtx/hydra/subdivision/refinementLevel", 0)

        from isaacsim.core.api import World
        from isaacsim.core.utils import extensions, prims, viewports, stage, rotations
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.core.prims import SingleXFormPrim

        extensions.enable_extension("isaacsim.ros2.sim_control")

        # Save Imports
        self.prims = prims
        self.stage_utils = stage
        self.rot_utils = rotations
        self.ArticulationAction = ArticulationAction
        self.World = World
        
        self.simulation_app.update()

        self.world = self.World(stage_units_in_meters=1.0)

        # Preparing stage
        viewports.set_camera_view(eye=np.array([1.2, 1.2, 0.8]), target=np.array([0, 0, 0.5]))

        from task_camera import Task
        self.task = Task(name="task", robot_prim_path=ROBOT_STAGE_PATH, robot_name="fr5")
        self.world.add_task(self.task)
        self.world.reset()

        self.simulation_app.update()
        self.robot = self.world.scene.get_object("fr5")
        self.robot.post_reset()
        self.simulation_app.update()

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

        self.setup_opaque_material()

        self.target_objects = {
            "beaker": self.task.beaker,
            "flask": self.task.flask
        }
        self.target_object_paths = [obj.prim_path for obj in self.target_objects.values()]

        self.apriltag_prims = {}
        for i in range(4):
            tag_name = f"apriltag_{i}"
            prim_path = f"/World/apriltag/{tag_name}"
            # Task에서 생성된 Prim을 SingleXFormPrim으로 래핑하여 가져옴
            self.apriltag_prims[tag_name] = SingleXFormPrim(prim_path=prim_path, name=tag_name)

        self.timer_period = 1/60
        self.timer = self.create_timer(self.timer_period, self.step_cb)

        self.step = 0

        self.get_logger().info("Simulation Start")


    def step_cb(self):

        if self.simulation_app.is_running():

            # step simulation
            self.world.step(render=True)

            is_save_step = (self.step % SAVE_INTERVAL == 0)

            if is_save_step:
                self.set_objects_material(use_opaque=False)

                # 2. 랜덤 포즈 변경
                # self.task.randomize_camera_pose()
                self.task.randomize_object_pose()

                for _ in range(20):
                    self.world.step(render=True)

                # [PASS 1] RGB 저장 (Glass Material)
                obs_glass = self.task.get_observations()
                rgb_img = Image.fromarray(obs_glass["rgb"])
                depth_raw_data = obs_glass["depth"]
                # label_raw_data = obs_glass["label"]

                cam_world_pos, cam_world_quat = self.task.camera.get_world_pose()
                intrinsics_raw = self.task.camera.get_intrinsics_matrix()

                # 1. 만약 PyTorch Tensor라면 CPU로 내리고 Numpy로 변환
                if hasattr(intrinsics_raw, "cpu"):
                    intrinsics_np = intrinsics_raw.cpu().numpy()
                elif hasattr(intrinsics_raw, "numpy"):
                    intrinsics_np = intrinsics_raw.numpy()
                else:
                    intrinsics_np = intrinsics_raw # 이미 Numpy인 경우

                # 2. 값 추출 시 명시적으로 python float()으로 변환
                # (intrinsics_np[0, 0] 등은 numpy.float32 타입이므로 json 저장 시 에러 발생함)
                fx = float(intrinsics_np[0, 0])
                fy = float(intrinsics_np[1, 1])
                cx = float(intrinsics_np[0, 2])
                cy = float(intrinsics_np[1, 2])

                pose_metadata = {
                    "step": self.step,
                    "intrinsics": {
                        "fx": fx,
                        "fy": fy,
                        "cx": cx,
                        "cy": cy,
                        # NumPy int64 문제 방지를 위해 int() 변환
                        "width": int(self.task.camera.get_resolution()[0]),
                        "height": int(self.task.camera.get_resolution()[1]),
                        # 전체 매트릭스를 리스트로 변환 (NumPy 타입을 Python list[float]로 변환)
                        "matrix_flat": intrinsics_np.flatten().astype(float).tolist()
                    },
                    "camera_pose_world": {
                        "position": cam_world_pos.tolist(),
                        "quaternion_wxyz": cam_world_quat.tolist()
                    },
                    "objects": {},
                    "apriltags": {}
                }

                # 각 물체의 World Pose 및 Camera Frame Pose 계산
                for name, obj_prim in self.target_objects.items():
                    if obj_prim is not None:
                        obj_pos, obj_quat = obj_prim.get_world_pose()
                        
                        p_rel, R_rel = get_relative_transform(cam_world_pos, cam_world_quat, obj_pos, obj_quat)
                        
                        pose_metadata["objects"][name] = {
                            "world_frame": {
                                "position": obj_pos.tolist(),
                                "quaternion_wxyz": obj_quat.tolist()
                            },
                            "camera_frame": {
                                "position": p_rel.tolist(),            # [x, y, z] 상대 위치
                                "rotation_matrix_flat": R_rel.flatten().tolist() # 회전 행렬
                            }
                        }

                for tag_name, tag_prim in self.apriltag_prims.items():
                    if tag_prim.is_valid():
                        tag_pos, tag_quat = tag_prim.get_world_pose()
                        
                        # 카메라 기준 상대 포즈 계산
                        p_rel, R_rel = get_relative_transform(cam_world_pos, cam_world_quat, tag_pos, tag_quat)
                        
                        pose_metadata["apriltags"][tag_name] = {
                            "world_frame": {
                                "position": tag_pos.tolist(),
                                "quaternion_wxyz": tag_quat.tolist()
                            },
                            "camera_frame": {
                                "position": p_rel.tolist(),
                                "rotation_matrix_flat": R_rel.flatten().tolist()
                            }
                        }


                # [PASS 2] Depth/Label 저장 (Opaque Material)
                self.set_objects_material(use_opaque=True)
                for _ in range(20):
                    self.world.step(render=True)
                
                obs_true_opaque = self.task.get_observations()
                depth_true_data = obs_true_opaque["depth"]
                label_true_data = obs_true_opaque["label"]
                
                self.set_objects_material(use_opaque=False)
                for _ in range(20):
                    self.world.step(render=True)

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
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                filename_base = f"{self.step*10:06d}"
                rgb_path = os.path.join(OUTPUT_DIR, f"{filename_base}-color.png")
                label_path = os.path.join(OUTPUT_DIR, f"{filename_base}-label.png")
                json_path = os.path.join(OUTPUT_DIR, f"{filename_base}-meta.json")

                rgb_img.save(rgb_path)
                save_depth_image_png_uint16(depth_raw_data, OUTPUT_DIR, f"{filename_base}-depth_raw.png")
                save_depth_image_png_uint16(depth_true_data, OUTPUT_DIR, f"{filename_base}-depth_true.png")
                
                if label_img is not None:
                    label_img.save(label_path)

                with open(json_path, "w") as f:
                    json.dump(pose_metadata, f, indent=4)
                
                self.get_logger().info(f"Saved step {self.step}")

            self.step += 1

        else:
            self.get_logger().info("Quit ROS2 Node")
            rclpy.try_shutdown()

    def setup_opaque_material(self):
        """Depth/Seg 캡처용 불투명 재질(Matte Dark Grey) 생성"""
        
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
            
            # [수정] 색상을 짙은 회색(거의 검정)으로 변경하여 색상 번짐 방지
            shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.1, 0.1))
            
            # [수정] 거칠기(Roughness)를 1.0으로 높여 빛 반사를 제거 (매트하게)
            shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(1.0)
            shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
            
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


def main(args=None):

    rclpy.init(args=args)
    sim_node = None
    try:
        sim_node = Simulation()
        if sim_node and rclpy.ok():
            rclpy.spin(sim_node)
        else:
            print("Simulation node failed to initialize.")

    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
    except Exception as e:
        print(f"ROS2 Spin Exception: {e}")
    finally:
        if sim_node:
            sim_node.get_logger().info("Shutting down simulation...")
            if sim_node.world:
                sim_node.world.stop()
            sim_node.simulation_app.close()
            sim_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Shutdown complete.")


if __name__ == '__main__':
    main()