#!/usr/bin/env bash
# Env wrapper for trial_driver.py: system ROS 2 Humble + colcon overlay +
# conda `sdl` (py3.10, cuTAMP/cuRobo), on ROS_DOMAIN_ID (default 100). Mirrors
# run_tamp.sh's environment so the driver's custom interfaces resolve. All args
# are forwarded to trial_driver.py.
set +u
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
COLCON_WS="$( cd "$PROJ_DIR/../.." && pwd )"

source /opt/ros/humble/setup.bash
[[ -f "$COLCON_WS/install/setup.bash" ]] && source "$COLCON_WS/install/setup.bash"

# locate conda
if [[ -z "${CONDA_BASE:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then CONDA_BASE="$(conda info --base)";
  else for c in "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
    [[ -f "$c/etc/profile.d/conda.sh" ]] && { CONDA_BASE="$c"; break; }; done; fi
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_SDL_ENV:-sdl}"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

exec python "$SCRIPT_DIR/trial_driver.py" "$@"
