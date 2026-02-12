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

    apriltag_launch_file = PathJoinSubstitution([
        FindPackageShare('apriltag_ros'),
        'launch',
        'apriltag.launch.py'
    ])
    apriltag_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(apriltag_launch_file),
    )

    perception_manager_node = Node(
        package='perception_manager',
        executable='perception_manager_node',
        name='perception_manager',
        output='screen',
        parameters=[
            config_file_path # YAML 파일을 파라미터로 로드
        ]
    )

    ld = LaunchDescription()

    ld.add_action(apriltag_launch) # later include this if needed, For Debugging easily
    ld.add_action(perception_manager_node)

    return ld
