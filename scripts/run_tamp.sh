#!/usr/bin/env bash
# =============================================================================
# run_tamp.sh  --  launch the TAMP application node (and optionally the XDL parser)
#
# Runs in the conda `sdl` env (CPython 3.10) against the SYSTEM ROS 2 Humble,
# with cuTAMP / cuRobo (torch 2.7/cu128). It talks to the Isaac Sim node
# (.venv-sdl, see run_isaacsim.sh) purely over DDS -- same ROS_DOMAIN_ID + RMW.
#
# Replaces the old automate_run.sh launch of tamp_server.py, dropping:
#   * the hardcoded /home/home paths and the buggy exclude_install_path,
#   * the NVIDIA `ros2 launch isaacsim ...` vendor launcher, and
#   * the `LD_PRELOAD=/usr/lib/.../libstdc++.so.6` hack (see note below).
#
# Paths derive from this script's location. Override points (env vars):
#   ROS_DOMAIN_ID (default 0)     RMW_IMPLEMENTATION (default rmw_fastrtps_cpp)
#   CONDA_SDL_ENV (default sdl)   CONDA_BASE (autodetected)
#   ROS_HUMBLE_SETUP (default /opt/ros/humble/setup.bash)
#
# Usage:
#   scripts/run_tamp.sh                # launch tamp_server.py
#   scripts/run_tamp.sh --parser       # launch tamp_xdl_parser.py instead
#   scripts/run_tamp.sh <args...>      # forwarded to the launched node
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"                    # .../src/sdl_project
COLCON_WS="$( cd "$PROJ_DIR/../.." && pwd )"                  # .../sdl_ws

TARGET="server"
if [[ "${1:-}" == "--parser" ]]; then TARGET="parser"; shift; fi

# --- system ROS 2 Humble -----------------------------------------------------
ROS_SETUP="${ROS_HUMBLE_SETUP:-/opt/ros/humble/setup.bash}"
[[ -f "$ROS_SETUP" ]] || { echo "ERROR: system ROS 2 Humble not found: $ROS_SETUP" >&2; exit 1; }
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"

# --- colcon overlay: tamp_interfaces / perception_interfaces (+ tamp, once built)
INSTALL_SETUP="$COLCON_WS/install/setup.bash"
if [[ -f "$INSTALL_SETUP" ]]; then
  # shellcheck disable=SC1090
  source "$INSTALL_SETUP"
else
  echo "WARN: colcon overlay not found ($INSTALL_SETUP); custom interfaces may be missing." >&2
fi

# --- conda `sdl` (cuTAMP / cuRobo, py3.10) ------------------------------------
ENVNAME="${CONDA_SDL_ENV:-sdl}"
if [[ -z "${CONDA_BASE:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
  else
    for c in "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
      [[ -f "$c/etc/profile.d/conda.sh" ]] && { CONDA_BASE="$c"; break; }
    done
  fi
fi
[[ -n "${CONDA_BASE:-}" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] || {
  echo "ERROR: cannot locate conda (set CONDA_BASE=/path/to/conda)." >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
# conda's activate hooks can emit a stray non-zero status (deactivate scripts,
# harmless warnings) that would trip `set -e`; relax errexit around activation
# just like nounset is relaxed above.
set +e
conda activate "$ENVNAME"
set -e
set -u

# --- DDS wire settings (must match run_isaacsim.sh) --------------------------
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

# NOTE on LD_PRELOAD: the old launcher force-preloaded the SYSTEM libstdc++.
# The conda `sdl` libstdc++ (>= 6.0.34) is NEWER than the system one (6.0.30),
# so it already provides every GLIBCXX symbol the ROS libs need -- the hack is
# not applied. If a future runtime shows a GLIBCXX/`libstdc++` version error,
# re-enable it explicitly, e.g.:
#   export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

if [[ "$TARGET" == "parser" ]]; then
  SCRIPT="$PROJ_DIR/TAMP/tamp/scripts/xdl/tamp_xdl_parser.py"
  PKG_EXE="tamp_xdl_parser.py"
else
  SCRIPT="$PROJ_DIR/TAMP/tamp/scripts/server/tamp_server.py"
  PKG_EXE="tamp_server.py"
fi

echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID  RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION  conda_env=$ENVNAME"

# Prefer the installed `tamp` ament package. During incremental development,
# SDL_USE_SOURCE=1 runs the checked-out source without copying files into the
# colcon install tree, while still using installed ROS interfaces.
if [[ "${SDL_USE_SOURCE:-0}" != "1" ]] && ros2 pkg prefix tamp >/dev/null 2>&1; then
  exec ros2 run tamp "$PKG_EXE" "$@"
else
  echo "NOTE: running checked-out tamp source (SDL_USE_SOURCE=${SDL_USE_SOURCE:-0})." >&2
  [[ -f "$SCRIPT" ]] || { echo "ERROR: script not found: $SCRIPT" >&2; exit 1; }
  export PYTHONPATH="$PROJ_DIR/TAMP/tamp/src:$( dirname "$SCRIPT" ):${PYTHONPATH:-}"
  exec python "$SCRIPT" "$@"
fi
