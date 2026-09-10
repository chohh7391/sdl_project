#!/usr/bin/env bash
# =============================================================================
# build_interfaces.sh
#
# Build ONLY the project's custom ROS 2 interface packages
# (tamp_interfaces, perception_interfaces) for Isaac Sim's bundled
# CPython 3.12, so that simulation.py (running in .venv-sdl) can
#   `import tamp_interfaces.srv, perception_interfaces`
#
# Strategy (no network required):
#   * Use the SYSTEM ROS 2 Humble install (/opt/ros/humble) purely as the
#     BUILD-TIME SDK (ament_cmake + rosidl generators + std/geometry msgs).
#   * Force the whole rosidl toolchain to emit a CPython 3.12 ABI by pointing
#     Python3_* at the interpreter that backs .venv-sdl (uv-managed cpython-3.12).
#   * The rosidl code generators are pure Python; they are made importable to
#     the 3.12 interpreter through a small, surgical "shim" of pure-python
#     modules (em, lark, catkin_pkg, ...) discovered from the system python.
#     numpy deliberately resolves from .venv-sdl (3.12), never the system copy.
#
# The RESULT (install/) is fully project-local. At runtime it needs the Isaac
# Sim internal Humble libs plus the .venv-sdl CPython 3.12 runtime that backs it
# (the generated extensions link libpython3.12.so.1.0, provided by .venv-sdl --
# see ../scripts/run_isaacsim.sh). It does NOT depend on /opt/ros, and it does
# NOT link the system CPython 3.10 -- the -DPYTHON_INCLUDE_DIR/-DPYTHON_LIBRARY
# args below force the deprecated FindPythonLibs path (used by Humble's
# rosidl_generator_py) onto the 3.12 libpython instead of the system 3.10.
#
# All paths are derived relative to this script; nothing is hardcoded to a home
# directory.  Only requirements: a system ROS 2 Humble install and the project
# .venv-sdl.  Re-runnable and idempotent.
# =============================================================================
set -euo pipefail

WS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_DIR="$( cd "$WS_DIR/.." && pwd )"

# --- locate the venv-sdl (Isaac Sim / py3.12) interpreter --------------------
VENV="${VENV_SDL:-$PROJ_DIR/.venv-sdl}"
VENVPY="$VENV/bin/python"
if [[ ! -x "$VENVPY" ]]; then
  echo "ERROR: .venv-sdl python not found at $VENVPY" >&2
  echo "       set VENV_SDL=/path/to/.venv-sdl and re-run." >&2
  exit 1
fi

# uv-managed base prefix that actually ships the headers + libpython
UVBASE="$("$VENVPY" -c 'import sys; print(sys.base_prefix)')"
PYVER="$("$VENVPY" -c 'import sys; print("%d.%d"%sys.version_info[:2])')"   # 3.12
PYINC="$UVBASE/include/python$PYVER"
PYLIB="$UVBASE/lib/libpython$PYVER.so"
for p in "$PYINC/Python.h" "$PYLIB"; do
  [[ -e "$p" ]] || { echo "ERROR: expected Python 3.12 build asset missing: $p" >&2; exit 1; }
done

# --- system ROS 2 Humble = build-time SDK ------------------------------------
ROS_SETUP="${ROS_HUMBLE_SETUP:-/opt/ros/humble/setup.bash}"
if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ERROR: system ROS 2 Humble not found ($ROS_SETUP)." >&2
  echo "       This build step needs ROS 2 Humble installed as the build SDK." >&2
  exit 1
fi
# ROS setup scripts reference unbound vars; relax nounset only for the source.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u

# --- build a surgical shim of PURE-python generator deps for the 3.12 interp --
# (system python3 has them for 3.10; they are pure python so import fine on 3.12)
SHIM="$WS_DIR/.gen_shim_py312"
rm -rf "$SHIM"; mkdir -p "$SHIM"
SYS_PY="$(command -v python3)"
for mod in em lark catkin_pkg pyparsing docutils dateutil; do
  loc="$("$SYS_PY" - "$mod" <<'PY' 2>/dev/null || true
import importlib, os, sys
m = importlib.import_module(sys.argv[1])
f = m.__file__
# single-file module (em.py) vs package dir
print(f if os.path.basename(f) != "__init__.py" else os.path.dirname(f))
PY
)"
  if [[ -n "$loc" && -e "$loc" ]]; then
    ln -sf "$loc" "$SHIM/$(basename "$loc")"
  else
    echo "WARN: could not locate pure-python generator dep '$mod' (build may still succeed)" >&2
  fi
done
# Prepend shim so the 3.12 generator interpreter finds em/lark/catkin_pkg,
# while numpy still resolves from .venv-sdl's own site-packages.
export PYTHONPATH="$SHIM:${PYTHONPATH:-}"

echo "=============================================================="
echo " Building custom interfaces for CPython $PYVER"
echo "   workspace : $WS_DIR"
echo "   interp    : $VENVPY"
echo "   base      : $UVBASE"
echo "   ROS SDK   : $(dirname "$(dirname "$ROS_SETUP")")"
echo "=============================================================="

cd "$WS_DIR"
python3 -m colcon build \
  --merge-install \
  --cmake-args \
    -DBUILD_TESTING=OFF \
    -DPython3_ROOT_DIR="$UVBASE" \
    -DPython3_EXECUTABLE="$VENVPY" \
    -DPython3_INCLUDE_DIR="$PYINC" \
    -DPython3_LIBRARY="$PYLIB" \
    -DPython3_FIND_STRATEGY=LOCATION \
    -DPYTHON_EXECUTABLE="$VENVPY" \
    -DPYTHON_INCLUDE_DIR="$PYINC" \
    -DPYTHON_LIBRARY="$PYLIB"

echo
echo "Build finished. Install prefix: $WS_DIR/install"
echo "Sanity: generated CPython extension ABI tags ->"
find "$WS_DIR/install" -name '*.cpython-*.so' -printf '  %f\n' 2>/dev/null | sort -u | head
