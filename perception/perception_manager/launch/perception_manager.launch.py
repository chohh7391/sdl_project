import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # 패키지 경로 찾기
    pkg_dir = get_package_share_directory('perception_manager')
    config_file_path = os.path.join(pkg_dir, 'config', 'object_configs.yaml')

    # [핵심] 여기에 사용할 카메라들을 쉼표로 구분해서 적어줍니다.
    # 나중에 카메라를 추가하거나 뺄 때 여기서만 문자열을 수정하면 됩니다.
    target_cameras = 'camera_1,camera_2'

    apriltag_launch_file = PathJoinSubstitution([
        FindPackageShare('apriltag_ros'),
        'launch',
        'apriltag.launch.py'
    ])
    
    # IncludeLaunchDescription에 인자 전달
    apriltag_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(apriltag_launch_file),
        launch_arguments={'camera_names': target_cameras}.items()
    )

    perception_manager_node = Node(
        package='perception_manager',
        executable='perception_manager_node',
        name='perception_manager',
        output='screen',
        parameters=[
            config_file_path,
            {"publish_tf": False}
        ]
    )

    ld = LaunchDescription()

    ld.add_action(apriltag_launch) # For Debugging easily
    ld.add_action(perception_manager_node)

    return ld