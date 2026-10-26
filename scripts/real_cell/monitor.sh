#!/usr/bin/env bash
# Watch the real two-beaker cell in rviz: robot, perceived and planned beakers,
# the planned grasp path, the pipeline's status and the three camera images.
#
#   scripts/real_cell/monitor.sh            # leave it open for the whole run
#
# It follows plan_remote.sh on its own: the plan shown is RUNS/CURRENT, which
# plan_remote.sh sets once a fetched trajectory has passed its checks, and the
# status line is RUNS/STATUS. Start it before plan_remote.sh, next to the
# bringup, the camera drivers and the pose node (pour_beakers.yaml).
#
# Env: CELL_WS (default: this checkout's ros2_cell_ws, whose setup.bash brings the core's),
#      REAL_CELL_RUNS (default real_cell_runs, under $HOME)
# NOT set -u: the ROS setup scripts read unbound variables.
set -o pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS="${REAL_CELL_RUNS:-real_cell_runs}"
case "$RUNS" in /*) ;; *) RUNS="$HOME/$RUNS" ;; esac
mkdir -p "$RUNS"
source /opt/ros/humble/setup.bash
source "${CELL_WS:-$DIR/../../ros2_cell_ws}/install/setup.bash"
python3 "$DIR/monitor_node.py" --runs-dir "$RUNS" &
MON=$!
trap 'kill "$MON" 2>/dev/null; wait "$MON" 2>/dev/null' EXIT
rviz2 -d "$DIR/rviz/real_cell.rviz"
