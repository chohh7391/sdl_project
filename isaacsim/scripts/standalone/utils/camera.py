from typing import Literal, List
import yaml
import numpy as np
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.numpy import rotations as rot_utils
from dataclasses import dataclass
import math
import os


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


def initialize_camera(camera):
    
    camera.initialize()

    camera_info = CameraInfo()
    camera.set_focal_length(camera_info.focal_length / 10.0)
    camera.set_focus_distance(camera_info.focus_distance)
    camera.set_lens_aperture(camera_info.f_stop * 100.0)
    camera.set_horizontal_aperture(camera_info.horizontal_aperture / 10.0)
    camera.set_vertical_aperture(camera_info.vertical_aperture / 10.0)
    camera.set_clipping_range(0.05, 1.0e5)

    # PINHOLE, deliberately. The focal length and apertures above are set from
    # the published intrinsics with their ratio preserved, so the camera's own
    # pinhole projection matches camera_info exactly (f/A * width = 601.5 px).
    # Overlaying a fisheyePolynomial / rational-polynomial model on top of that
    # did NOT: measured through the rendered image, a tag at a known table pose
    # came out as if the focal length were about 1611 px rather than 601.5, so
    # every pose the detector produced was wrong -- 0.39 m of error for a tag
    # 1.5 m from the camera. The distortion coefficients are all zero anyway, so
    # there is nothing for a distortion model to represent, and
    # set_projection_type is deprecated in Isaac Sim 6 (it warns on every call).
    camera.set_lens_distortion_model("pinhole")

    # Self-check. The pose the detector reports is only as good as the agreement
    # between the intrinsics published on camera_info (which the ROS helper reads
    # off this prim) and the projection the renderer actually uses. Measured
    # through a tag at a known table pose, the render behaved as if the focal
    # length were ~2100 px while camera_info said 601.5 -- a tag 0.2 m away from
    # the first one fell outside the image entirely, which only a much narrower
    # field of view explains. Print what the prim ended up with so the two can be
    # compared instead of assumed.
    try:
        prim = camera.prim
        print("[Camera] %s intended fx=%.1f cx=%.1f (K) | prim focalLength=%s "
              "hAperture=%s vAperture=%s | resolution=%s"
              % (camera.prim_path, camera_info.fx, camera_info.cx,
                 prim.GetAttribute("focalLength").Get(),
                 prim.GetAttribute("horizontalAperture").Get(),
                 prim.GetAttribute("verticalAperture").Get(),
                 camera.get_resolution()), flush=True)
        print("[Camera] %s get_intrinsics_matrix() =\n%s"
              % (camera.prim_path, camera.get_intrinsics_matrix()), flush=True)
        # The remaining unknown: whether the frame the renderer projects from is
        # the frame the TF graph publishes. The intrinsics agree (prim, Isaac and
        # camera_info all give fx = 601.53), so a tag that should land 40 px from
        # the principal point and lands 141 px away has to be a geometry
        # mismatch, not a lens one.
        print("[Camera] %s world pose (world axes) = %s | (usd) = %s | (ros) = %s"
              % (camera.prim_path,
                 camera.get_world_pose(camera_axes="world"),
                 camera.get_world_pose(camera_axes="usd"),
                 camera.get_world_pose(camera_axes="ros")), flush=True)
        # Decisive check: ask the camera itself where a known world point lands
        # in the image, using its own model. If this matches the geometry but the
        # detector disagrees, the problem is downstream (the published image or
        # camera_info); if it matches the detector, the axes were mis-derived.
        import numpy as _np
        probes = _np.array([[0.35, 0.00, 0.001], [0.35, -0.20, 0.001],
                            [0.45, 0.00, 0.001]])
        try:
            print("[Camera] %s projects world points %s -> image %s"
                  % (camera.prim_path, probes.tolist(),
                     camera.get_image_coords_from_world_points(probes).tolist()),
                  flush=True)
        except Exception as exc:
            print("[Camera] projection probe unavailable: %r" % (exc,), flush=True)
        # Dump one frame so the image the detector is given can be inspected
        # directly, rather than inferred from its output.
        dump = os.environ.get("SDL_CAMERA_DUMP", "").strip()
        if dump:
            try:
                import imageio.v2 as _iio
                rgba = camera.get_rgba()
                if rgba is not None and getattr(rgba, "size", 0):
                    out = os.path.join(dump, os.path.basename(camera.prim_path) + ".png")
                    _iio.imwrite(out, (rgba[:, :, :3]).astype("uint8"))
                    print("[Camera] wrote %s shape=%s" % (out, rgba.shape), flush=True)
                else:
                    print("[Camera] get_rgba() empty at init (needs a rendered frame)", flush=True)
            except Exception as exc:
                print("[Camera] frame dump failed: %r" % (exc,), flush=True)
    except Exception as exc:
        print("[Camera] intrinsics self-check unavailable: %r" % (exc,), flush=True)
    # camera.add_normals_to_frame()
    # camera.add_motion_vectors_to_frame()
    # camera.add_distance_to_image_plane_to_frame()


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
