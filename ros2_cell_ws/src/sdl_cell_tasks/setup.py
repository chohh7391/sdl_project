# Copyright 2026 Hyunho Cho
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'sdl_cell_tasks'
share = os.path.join('share', package_name)

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (share, ['package.xml']),
        # Each task's example task_params file, config/<task>.yaml.
        (os.path.join(share, 'config'), glob('config/*.yaml')),
        # The object tables the pose node and the trees share.
        (os.path.join(share, 'config', 'perception'), glob('config/perception/*.yaml')),
        # The cell layouts a replay of a recording that assumes one is checked against.
        (os.path.join(share, 'config', 'replay'), glob('config/replay/*.yaml')),
        # Recordings made on the real cell, with their meta and the layout they assume.
        (os.path.join(share, 'config', 'replay', 'real'),
         glob('config/replay/real/*.csv') + glob('config/replay/real/*.json')
         + glob('config/replay/real/*.yaml')),
        # Where the wrist camera goes to look at an occluded vessel, and the
        # specs those tables are solved from (scripts/solve_sweep_raster.py).
        (os.path.join(share, 'config', 'sweep'), glob('config/sweep/*.yaml')),
        # The glassware's display meshes, named by config/perception as
        # package://sdl_cell_tasks/meshes/<file>; rviz resolves that through the
        # ament index, so they have to be installed.
        (os.path.join(share, 'meshes'), glob('meshes/*.stl') + ['meshes/README.md', 'meshes/usd_to_stl.py']),
        (os.path.join(share, 'scripts'), glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hyunho Cho',
    maintainer_email='chohh7391@gmail.com',
    description='The FR5 chemistry-bench task trees, registered with cho_task_manager',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        # <task name> = <module>:<{robot_type: tree builder}>. Task names are
        # global across every installed package, hence the sdl_ prefix.
        'cho_task_manager.tasks': [
            'sdl_beaker_transfer = sdl_cell_tasks.tasks.fr5.beaker_transfer:TASK',
            'sdl_grasp_trial = sdl_cell_tasks.tasks.fr5.grasp_trial:TASK',
            'sdl_trajectory_replay = sdl_cell_tasks.tasks.fr5.trajectory_replay:TASK',
            'sdl_occlusion_recovery = sdl_cell_tasks.tasks.fr5.occlusion_recovery:TASK',
            'sdl_occlusion_replay = sdl_cell_tasks.tasks.fr5.occlusion_replay:TASK',
            'sdl_perceived_replay = sdl_cell_tasks.tasks.fr5.perceived_replay:TASK',
            'sdl_vessel_detect = sdl_cell_tasks.tasks.fr5.vessel_detect:TASK',
        ],
        'console_scripts': [
            'fr5_pour_client = sdl_cell_tasks.tools.pour_client:main',
            'scale_relay = sdl_cell_tasks.tools.scale_relay:main',
        ],
    },
)
