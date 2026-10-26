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

"""What more than one of the cell's FR5 trees needs, in one place.

So the second tree to need it does not get to pick a different home pose, mode
or vessel topic than the first one did.
"""

from collections import namedtuple

import yaml

from cho_object_pose.objects import parse_objects
# The robot's task home, the registry's poses.task_home -- for the FR5 home 1,
# never the diagnostic-only home 0 (fr5.yaml's schema refuses that).
from cho_task_manager.subtrees import home_joint_state  # noqa: F401

# Every FR5 bringup hard-codes control_mode 'position' (cho_bringup_fr5), and
# that is also the only mode fr5.yaml declares a hold controller for.
CONTROL_MODE = 'position'


#: One tagged vessel on the bench: what it is called, where its detected pose
#: comes out, and the blackboard entry a tree latches it into.
VesselSpec = namedtuple('VesselSpec', 'name topic key')

#: The vessels the cameras track.
#:
#: ``name`` is the joining key and has to be the same word in three places: the
#: object table (``config/perception/vessel_detect.yaml``), which decides the
#: topic; the layout a recording assumes
#: (``layout_the_trajectory_assumes`` in its meta) and the cell layout files in
#: ``config/replay/``, which is what makes a detected pose checkable against a
#: replay. Tests assert the first of those; the recordings are produced
#: elsewhere, so the third is checked at run time by the layout gate saying an
#: object is unverified rather than by anything failing here.
#:
#: Shared rather than repeated because several trees use it --
#: sdl_vessel_detect latches them, the occlusion trees sweep for them,
#: sdl_perceived_replay checks a recording against them -- and a second copy
#: would be a topic string that can go stale in one of them.
VESSELS = (
    VesselSpec('beaker', '/perception/object_pose/beaker', 'beaker_pose'),
    VesselSpec('flask', '/perception/object_pose/flask', 'flask_pose'),
)


def vessels_from_table(path):
    """The objects of an object table (``config/perception/*.yaml``) as vessels.

    Name and topic come from the table -- the same parse cho_object_pose runs,
    so the two cannot read one file differently -- and the blackboard key is
    ``<name>_pose``, as VESSELS has it.
    """
    with open(path, encoding='utf-8') as handle:
        specs = parse_objects(yaml.safe_load(handle))
    return tuple(VesselSpec(spec.name, spec.topic, '%s_pose' % spec.name) for spec in specs)


def tracked_vessels(table_path):
    """The vessels a tree tracks: the objects of the table at *table_path*, else VESSELS.

    *table_path* is a checked path (``utils/params.share_path``) or None.
    """
    if table_path is None:
        return VESSELS
    return vessels_from_table(table_path)
