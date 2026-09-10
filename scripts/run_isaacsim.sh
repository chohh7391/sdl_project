#!/usr/bin/env bash
# =============================================================================
# run_isaacsim.sh  --  launch the Isaac Sim standalone simulation node
#
# Runs isaacsim/scripts/standalone/simulation.py inside the project's
# pip-based Isaac Sim 6.0.1 environment (.venv-sdl, CPython 3.12), wired to
# Isaac Sim's INTERNAL ROS 2 Humble (rclpy + the project's custom interfaces
# built for py3.12), with the system ROS 2 (/opt/ros) deliberately excluded.
#
# It communicates with the TAMP / perception nodes (conda `sdl`, py3.10, system
# Humble -- see run_tamp.sh) purely over DDS: same ROS_DOMAIN_ID + RMW.
#
# All paths are derived from this script's own location; nothing is hardcoded
# to a home directory. Override points (env vars):
#   ROS_DOMAIN_ID   (default 0)      RMW_IMPLEMENTATION (default rmw_fastrtps_cpp)
#   VENV_SDL        (default <project>/.venv-sdl)
#
# Usage:
#   scripts/run_isaacsim.sh                 # launch simulation.py (GUI)
#   scripts/run_isaacsim.sh --headless      # launch with no GUI (SDL_HEADLESS=1)
#   scripts/run_isaacsim.sh --check         # validate the ROS env only (no GUI)
#   scripts/run_isaacsim.sh <args...>       # forwarded to simulation.py
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

CHECK=0
if [[ "${1:-}" == "--check" ]]; then CHECK=1; shift; fi

# --headless: opt into no-GUI mode by exporting SDL_HEADLESS=1 (read by
# simulation.py). Default is unchanged (GUI). Accept it in any leading position.
if [[ "${1:-}" == "--headless" ]]; then export SDL_HEADLESS=1; shift; fi

# --- locate .venv-sdl (Isaac Sim / py3.12) -----------------------------------
VENV="${VENV_SDL:-$PROJ_DIR/.venv-sdl}"
VENVPY="$VENV/bin/python"
[[ -x "$VENVPY" ]] || { echo "ERROR: .venv-sdl python missing: $VENVPY" >&2; exit 1; }

# --- Isaac Sim internal ROS 2 Humble (bundled, py3.12) -----------------------
HUMBLE="$VENV/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/humble"
[[ -d "$HUMBLE/rclpy" && -d "$HUMBLE/lib" ]] || {
  echo "ERROR: Isaac Sim internal Humble not found under $HUMBLE" >&2; exit 1; }

# --- project-local custom interfaces built for py3.12 ------------------------
IF_WS="$PROJ_DIR/ros2_isaacsim_ws/install"
IF_PY="$IF_WS/lib/python3.12/site-packages"
IF_LIB="$IF_WS/lib"
if [[ ! -d "$IF_PY/tamp_interfaces" ]]; then
  echo "ERROR: py3.12 custom interfaces not built." >&2
  echo "       Build them once with:" >&2
  echo "         bash $PROJ_DIR/ros2_isaacsim_ws/build_interfaces.sh" >&2
  exit 1
fi

# --- scrub any leaked SYSTEM ROS (/opt/ros) from the environment -------------
# Drop ROS bookkeeping vars entirely...
unset AMENT_PREFIX_PATH AMENT_CURRENT_PREFIX CMAKE_PREFIX_PATH COLCON_PREFIX_PATH \
      ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_LOCALHOST_ONLY \
      ROS_AUTOMATIC_DISCOVERY_RANGE RMW_IMPLEMENTATION_WRAPPER PYTHONPATH 2>/dev/null || true

# ...and strip any /opt/ros or ROS-workspace entries from LD_LIBRARY_PATH,
# keeping unrelated (e.g. CUDA) entries.
strip_ros() {  # $1 = path-list; echoes filtered list
  local out="" e
  local IFS=':'
  for e in $1; do
    [[ -z "$e" ]] && continue
    case "$e" in
      /opt/ros/*|*/ros/humble/*|*/install/*/lib|*sdl_ws/install*) continue ;;
    esac
    out="${out:+$out:}$e"
  done
  printf '%s' "$out"
}
LD_KEEP="$(strip_ros "${LD_LIBRARY_PATH:-}")"

# --- set the INTERNAL ROS environment ----------------------------------------
export PYTHONPATH="$HUMBLE/rclpy:$HUMBLE:$IF_PY"
export LD_LIBRARY_PATH="$HUMBLE/lib:$IF_LIB${LD_KEEP:+:$LD_KEEP}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

# --- activate the venv (prefix + PATH) and launch ----------------------------
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

if [[ "$CHECK" == "1" ]]; then
  echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID  RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
  echo "PYTHONPATH=$PYTHONPATH"
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  exec "$VENVPY" - <<'PY'
import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message, deserialize_message
from std_msgs.msg import Float32
from std_srvs.srv import SetBool
from geometry_msgs.msg import Wrench
from tamp_interfaces.msg import PlanStep
from tamp_interfaces.srv import ToolChange, GetRobotInfo, GetToolInfo
from perception_interfaces.srv import GetFtData, GetObjectInfo
rclpy.init()
n = Node("run_isaacsim_check")
n.create_client(GetRobotInfo, "get_robot_info")           # forces typesupport load
_ = ToolChange.Request(); _.desired_tool = "ag95"
# Exercise the C convert_from_py / convert_to_py path with STRING fields: this
# is what SIGABRTs when the generated .so are linked against the wrong libpython
# (the PyUnicode ABI mismatch). A plain import/attr-set does NOT catch that.
step = PlanStep()
step.type = PlanStep.GRIPPER
step.op_name = "ag95"
step.action = "close"
raw = serialize_message(step)                             # C: convert_from_py
back = deserialize_message(raw, PlanStep)                 # C: convert_to_py
assert back.op_name == "ag95" and back.action == "close", back
# Also round-trip a service Request carrying a string field.
req = ToolChange.Request(); req.desired_tool = "ag95"
assert deserialize_message(serialize_message(req), ToolChange.Request).desired_tool == "ag95"
n.destroy_node(); rclpy.shutdown()
print("ENV OK: rclpy + std/geometry msgs + tamp_interfaces + perception_interfaces "
      "import, instantiate, AND round-trip serialize string fields under CPython "
      "3.12 (system ROS excluded).")
PY
fi

SIM="$PROJ_DIR/isaacsim/scripts/standalone/simulation.py"
[[ -f "$SIM" ]] || { echo "ERROR: simulation.py not found: $SIM" >&2; exit 1; }
exec "$VENVPY" "$SIM" "$@"
