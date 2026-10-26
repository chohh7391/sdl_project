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

"""A D435 for the pour cell: the core's d435.launch.py, plus a colour stream to look at the cell.

    CFG=$(ros2 pkg prefix --share cho_bringup_fr5)/config/real
    ros2 launch sdl_cell_bringup d435.launch.py camera_namespace:=side_2 camera_name:=side_2 \\
        serial_no:=_332322072253 camera_config:=$CFG/d435_side.yaml

``stream:=both`` (the default here) is the tag detector's infra1 exactly as
cho_realsense's ``stream:=infra1`` opens it -- same profile, Y8, emitter off,
so the detector and the extrinsics solved on infra1 see what they did -- with
the colour imager on beside it at ``color_profile`` (640x480 at 15 Hz), for an
operator to look at the cell. The two D435s on the bench share one USB 3 hub,
which is why colour gets its own, smaller profile. ``stream:=infra1`` and
``stream:=color`` are the core launch's own, passed straight through.

The core's d435.launch.py (cho_robot_project v0.1.0) takes only infra1 or color,
and is not changed for this. It hands rs_launch.py its stream switches as launch
arguments and ``camera_config`` as rs_launch's config_file, and rs_launch gives
the camera node ``[launch arguments, config file]`` in that order, so a key in
the file wins. With ``both`` this launch includes the core's with
``stream:=infra1`` and a camera_config that is the one given plus
``enable_color: true`` and ``rgb_camera.color_profile: <color_profile>``,
written to a temporary file. Was the laptop core checkout's d435.launch.py
``stream:=both`` (its 3829cd9), never on GitHub.
"""

import os
import re
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import yaml

PACKAGE = 'sdl_cell_bringup'
STREAMS = ('infra1', 'color', 'both')
#: The core launch's own arguments, passed through unchanged.
PASSED = ('profile', 'serial_no', 'camera_name', 'camera_namespace')
PROFILE = re.compile(r'^\d+x\d+x\d+$')


def color_config(camera_config, color_profile):
    """*camera_config* with the colour imager on at *color_profile*; the written file's path."""
    with open(camera_config, encoding='utf-8') as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise RuntimeError('camera_config %s is not a mapping of rs_launch parameters' % camera_config)
    config['enable_color'] = True
    config['rgb_camera.color_profile'] = color_profile
    handle, path = tempfile.mkstemp(prefix='sdl_cell_d435_', suffix='.yaml')
    with os.fdopen(handle, 'w', encoding='utf-8') as stream:
        stream.write('# %s, with the colour stream on (sdl_cell_bringup d435.launch.py)\n'
                     % camera_config)
        yaml.safe_dump(config, stream, sort_keys=False)
    return path


def _launch_setup(context, *args, **kwargs):
    stream = LaunchConfiguration('stream').perform(context)
    if stream not in STREAMS:
        raise RuntimeError("stream must be one of %s, got '%s'" % (list(STREAMS), stream))
    camera_config = LaunchConfiguration('camera_config').perform(context)
    arguments = {name: LaunchConfiguration(name).perform(context) for name in PASSED}
    if stream == 'both':
        color_profile = LaunchConfiguration('color_profile').perform(context)
        if not PROFILE.match(color_profile):
            raise RuntimeError("color_profile must be 'WIDTHxHEIGHTxFPS', got '%s'" % color_profile)
        camera_config = color_config(camera_config, color_profile)
        base = ('/%s/%s' % (arguments['camera_namespace'], arguments['camera_name'])).replace('//', '/')
        print('[%s] publishing %s/color/image_raw at %s beside infra1' % (PACKAGE, base, color_profile))
    core = os.path.join(get_package_share_directory('cho_realsense'), 'launch', 'd435.launch.py')
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(core),
        launch_arguments={
            **arguments,
            'stream': 'infra1' if stream == 'both' else stream,
            'camera_config': camera_config,
        }.items(),
    )]


def generate_launch_description():
    core_share = get_package_share_directory('cho_realsense')
    return LaunchDescription([
        DeclareLaunchArgument(
            'stream', default_value='both',
            description="'both': infra1 exactly as the core's stream:=infra1, plus colour at "
                        "color_profile for viewing; 'infra1' or 'color': the core's own."),
        DeclareLaunchArgument(
            'color_profile', default_value='640x480x15',
            description="The colour stream's 'WIDTHxHEIGHTxFPS' with stream:=both."),
        DeclareLaunchArgument(
            'profile', default_value='848x480x30',
            description="infra1's 'WIDTHxHEIGHTxFPS' (with stream:=color, colour's), as the core "
                        "launch's profile:=."),
        DeclareLaunchArgument('serial_no', default_value=''),
        DeclareLaunchArgument('camera_name', default_value='camera'),
        DeclareLaunchArgument('camera_namespace', default_value='camera'),
        DeclareLaunchArgument(
            'camera_config', default_value=os.path.join(core_share, 'config', 'd435.yaml'),
            description="rs_launch's config_file, as the core launch's camera_config:= (the cell's "
                        'are cho_bringup_fr5 config/real/d435_side.yaml and d435_wrist.yaml).'),
        OpaqueFunction(function=_launch_setup),
    ])
