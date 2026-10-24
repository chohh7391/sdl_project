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

"""A task's own parameters: ``robot_config['task_params']``, from ``task_params:=<file.yaml>``.

Each task here declares what it takes and the value when it is not given, and
refuses a key it does not take, as the core template's ``joint_wave_params()``
does: a misspelt key would otherwise run the task on its default. The node
reports the ValueError and exits before anything moves.
"""

from collections.abc import Mapping
import math
import os

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

PACKAGE = 'sdl_cell_tasks'


def task_params(robot_config, task, defaults):
    """*defaults*, overridden by ``robot_config['task_params']``; refuses a key not in *defaults*."""
    given = robot_config.get('task_params') or {}
    if not isinstance(given, Mapping):
        raise ValueError(f'{task}: task_params must be a mapping; got {type(given).__name__}')
    unknown = sorted(set(given) - set(defaults))
    if unknown:
        raise ValueError(f'{task} does not take {unknown}; it takes {sorted(defaults)}')
    return {**defaults, **given}


def number(params, key, task):
    """``params[key]`` as a finite float; ValueError naming the key otherwise."""
    value = params[key]
    if isinstance(value, bool):
        raise ValueError(f'{task}: {key} must be a number; got {value!r}')
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{task}: {key} must be a number; got {value!r}') from None
    if not math.isfinite(result):
        raise ValueError(f'{task}: {key} must be finite; got {value!r}')
    return result


def share_directory():
    """This package's installed share directory."""
    try:
        return get_package_share_directory(PACKAGE)
    except PackageNotFoundError as error:
        raise ValueError(f'{PACKAGE} is not installed: {error}') from error


def share_path(value, key, task):
    """The file *value* names: absolute as given, relative against share/sdl_cell_tasks.

    A task_params file is YAML, which has no ``$(find-pkg-share)``, so a path to
    one of this package's own tables is written relative to its share directory
    (``config/sweep/fr5_bench.yaml``). Raises ValueError for a file that is not there.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{task}: {key} must be a file path; got {value!r}')
    path = os.path.expanduser(value.strip())
    if not os.path.isabs(path):
        path = os.path.join(share_directory(), path)
    if not os.path.isfile(path):
        raise ValueError(f'{task}: {key}: no such file ({path})')
    return path


__all__ = ['PACKAGE', 'number', 'share_directory', 'share_path', 'task_params']
