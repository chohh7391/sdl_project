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

"""sdl_beaker_transfer: the real cell's two-beaker transfer, as its plan lays it out.

Everything the PLAN decides, and nothing that comes before it (author,
2026-09-30: "planning에 의한 것만"). Perception and the occlusion recovery have
measured both beakers, and the planner -- Isaac Sim + cuTAMP, run by
sdl_project's scripts/real_cell/plan_remote.sh -- has laid out a motion for
those positions. This task carries it out on the FR5::

    1_Initialize      to the plan's start pose (home_via; MoveIt by default)
    2_Handover        onto the trajectory controller
    3_Replay          pick beaker1 -> measure the grasp, release its held pose
                      -> carry -> pour by weight into beaker2 (pouring_controller,
                      which stands beaker1 upright first) -> put it back where it
                      was taken from, restore its held pose
    4_Finish          home to the plan's start pose, parked on the hold controller

It is ``sdl_trajectory_replay`` with its inputs taken from the plan. One
parameter names the plan, ``transfer_plan``: the trajectory's path without an
extension, e.g. ``~/real_cell_runs/<run>/deliver/traj_realcell_seed0``. Its
files say the rest:

    <plan>.csv, <plan>.meta.json  the joint trajectory and the layout it assumes
    <plan>_cell.yaml              the cell it was planned for, gated at build time
                                  (``layout:``), and the pour it was planned for
                                  (``pour:``): beaker2's rim and beaker1's bottom
                                  centre as perception gave them to the planner,
                                  and where beaker1 goes back afterwards

What is poured and how much is not the plan's: ``replay_pour_material`` and
``replay_pour_grams`` are required. The trajectory, layout and pour target are
the plan's alone, so passing them as well -- the node's ``replay_trajectory``,
``replay_meta`` or ``replay_layout``, or ``replay_pour_receiver``,
``replay_pour_grasp_marker`` or ``replay_pour_put_back`` in task_params -- is
refused rather than silently preferring one source. The rest of
``sdl_trajectory_replay``'s parameters are taken as it takes them
(:data:`DEFAULTS`)::

    ros2 launch cho_task_manager run_task_manager.launch.py robot_type:=fr5 \\
        task:=sdl_beaker_transfer task_params:=<a file like config/sdl_beaker_transfer.yaml>

On cho_robot_project v0.1.0 the pose node holds no pose and serves no
release/restore (see behaviors/service/release_held_pose.py): those two steps
log that and the replay carries on.
"""

from collections.abc import Mapping
import math
import os

import py_trees
import yaml

from sdl_cell_tasks.subtrees.pour import PUT_BACK
from sdl_cell_tasks.tasks.fr5.trajectory_replay import (
    build_trajectory_replay,
    DEFAULTS as REPLAY_DEFAULTS,
)
from sdl_cell_tasks.utils.params import task_params

TASK_NAME = 'sdl_beaker_transfer'

#: The parameter naming the plan.
TRANSFER_PLAN_KEY = 'transfer_plan'

#: The core node's parameters the plan supplies, and so may not be passed beside it.
PLANNED_NODE_KEYS = ('replay_trajectory', 'replay_meta', 'replay_layout')

#: The pour parameters the plan supplies, and so may not be in task_params beside it.
PLANNED_POUR_KEYS = ('replay_pour_receiver', 'replay_pour_grasp_marker', 'replay_pour_put_back')

#: What task_params may set, and its value when it does not: the plan, then
#: sdl_trajectory_replay's own but for what the plan says. What is poured and how
#: much have no default.
DEFAULTS = {
    TRANSFER_PLAN_KEY: None,
    **{key: value for key, value in REPLAY_DEFAULTS.items() if key not in PLANNED_POUR_KEYS},
    'replay_pour_grams': None,
    'replay_pour_material': None,
}


def plan_files(plan, task=TASK_NAME):
    """``(csv, meta, cell)`` for the plan *plan*, which may also name its CSV."""
    if not isinstance(plan, str) or not plan.strip():
        raise ValueError(
            '%s needs %s: the plan to carry out, <run>/deliver/traj_realcell_seed<N>'
            % (task, TRANSFER_PLAN_KEY))
    base = os.path.expanduser(plan.strip())
    if base.endswith('.csv'):
        base = base[:-len('.csv')]
    files = (base + '.csv', base + '.meta.json', base + '_cell.yaml')
    missing = [path for path in files if not os.path.isfile(path)]
    if missing:
        raise ValueError(
            "%s: %s names no complete plan: missing %s. It is the trajectory's path "
            'without an extension, as plan_remote.sh delivers it '
            '(<run>/deliver/traj_realcell_seed<N>)' % (task, plan, ', '.join(missing)))
    return files


def planned_node_keys(robot_config, task=TASK_NAME):
    """Refuse the core node's recording parameters when a plan names the recording."""
    passed = [key for key in PLANNED_NODE_KEYS if str(robot_config.get(key) or '').strip()]
    if passed:
        raise ValueError(
            '%s: %s come from %s here; passing them as well would leave two answers to '
            'where the arm goes' % (task, ', '.join(passed), TRANSFER_PLAN_KEY))


def _point(value, what, path):
    try:
        point = [float(v) for v in value]
    except (TypeError, ValueError):
        point = []
    if len(point) != 3 or not all(math.isfinite(v) for v in point):
        raise ValueError('%s: pour.%s must be three numbers x, y, z in base_link (m); got %r'
                         % (path, what, value))
    return point


def planned_pour(cell_path):
    """``(receiver_rim, grasp_marker, put_back)`` from the plan's ``pour:`` block."""
    with open(cell_path, encoding='utf-8') as handle:
        loaded = yaml.safe_load(handle) or {}
    pour = loaded.get('pour') if isinstance(loaded, Mapping) else None
    if not isinstance(pour, Mapping):
        raise ValueError(
            "%s has no 'pour:' block, so it does not say what the plan was made to pour "
            'into. It was written before plan_remote.sh recorded that: fetch the run '
            'again (plan_remote.sh --fetch <run>)' % cell_path)
    put_back = str(pour.get('put_back', '')).strip().lower()
    if put_back not in PUT_BACK:
        raise ValueError('%s: pour.put_back must be one of %s; got %r'
                         % (cell_path, list(PUT_BACK), pour.get('put_back')))
    return (_point(pour.get('receiver_rim'), 'receiver_rim', cell_path),
            _point(pour.get('grasp_marker'), 'grasp_marker', cell_path),
            put_back)


def create_sdl_beaker_transfer_tree(robot_config) -> py_trees.behaviour.Behaviour:
    """Pick beaker1, pour into beaker2 and put it back, as the plan lays it out."""
    given = robot_config.get('task_params') or {}
    planned = sorted(key for key in PLANNED_POUR_KEYS
                     if isinstance(given, Mapping) and key in given)
    if planned:
        raise ValueError(
            '%s: %s come from %s here (its _cell.yaml pour: block); passing them as well '
            'would leave two answers to where the pour goes'
            % (TASK_NAME, ', '.join(planned), TRANSFER_PLAN_KEY))
    params = task_params(robot_config, TASK_NAME, DEFAULTS)
    csv_path, meta_path, cell_path = plan_files(params[TRANSFER_PLAN_KEY])
    planned_node_keys(robot_config)
    missing = [key for key in ('replay_pour_material', 'replay_pour_grams')
               if params[key] is None or params[key] == '' or params[key] == 0]
    if missing:
        raise ValueError(
            '%s needs %s: what is poured and how much is not something the plan decides'
            % (TASK_NAME, ' and '.join(missing)))

    rim, marker, put_back = planned_pour(cell_path)
    config = dict(robot_config)
    config.update(replay_trajectory=csv_path, replay_meta=meta_path, replay_layout=cell_path)
    replay_params = {key: value for key, value in params.items() if key != TRANSFER_PLAN_KEY}
    replay_params.update(replay_pour_receiver=rim, replay_pour_grasp_marker=marker,
                         replay_pour_put_back=put_back)
    root = build_trajectory_replay(config, replay_params, task=TASK_NAME,
                                   root_name='SDL_Beaker_Transfer_Root')
    if root.replay_summary['pour'] is None:
        raise ValueError('%s needs replay_pour_grams above 0; got %r'
                         % (TASK_NAME, params['replay_pour_grams']))
    root.transfer_summary = {
        'plan': csv_path[:-len('.csv')],
        'receiver_rim': rim,
        'grasp_marker': marker,
        'put_back': put_back,
    }
    return root


#: The registration setup.py points at: {robot_type: tree builder}.
TASK = {'fr5': create_sdl_beaker_transfer_tree}

__all__ = [
    'DEFAULTS',
    'PLANNED_NODE_KEYS',
    'PLANNED_POUR_KEYS',
    'TASK',
    'TASK_NAME',
    'TRANSFER_PLAN_KEY',
    'create_sdl_beaker_transfer_tree',
    'plan_files',
    'planned_node_keys',
    'planned_pour',
]
