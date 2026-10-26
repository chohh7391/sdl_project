#!/usr/bin/env bash
# Run plan_remote.sh and record the real cell's raw data around it, for the
# perception and pouring videos rendered afterwards (Rerun).
#
#   scripts/real_cell/bag_run.sh LABEL [plan_remote.sh arguments ...]
#
# Two bags under ~/real_cell_runs/bags/<stamp>_<LABEL>/, and nothing recorded
# while rci-pc plans (10 minutes in which nothing moves):
#
#   1_perceive   from the start until the planning is sent off: the standing
#                cameras' view and the wrist's recovery sweep
#   2_replay     from the trajectory being ready to the end: pick, carry,
#                levelling, pour, put back
#
# To start the replay only when the cameras are rolling, plan without
# --replay, then replay the fetched run into the same directory:
#
#   scripts/real_cell/bag_run.sh LABEL --perceive --beakers ... (no --replay)
#   BAG_DIR=<that directory> scripts/real_cell/bag_run.sh LABEL --fetch RUN ... --replay
#
# A --fetch run records one bag, 2_replay, from its start; a --measure-only run
# (a grasp trial's measurement after the release) one bag, 3_measure.
#
# What goes in is what a render needs and nothing it can recompute: every
# detector's own image (so a tag outline lands exactly on its tag), the
# colour streams for looking at, all detections, TF, joints, the pose node's
# poses and markers, the scale, the pour's feedback and the log. About
# 12 MB/s, so a run is roughly 4 GB.
set -u
[[ $# -ge 1 ]] || { sed -n '2,20p' "$0"; exit 2; }
LABEL=$1; shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=${BAG_DIR:-~/real_cell_runs/bags/${STAMP}_${LABEL}}
FIRST=1_perceive
REPLAYS=""
for arg in "$@"; do
  [[ $arg == --fetch ]] && FIRST=2_replay
  [[ $arg == --measure-only ]] && FIRST=3_measure
  [[ $arg == --replay ]] && REPLAYS=1
done
LOG=$OUT/plan_remote_${FIRST%%_*}.log
mkdir -p "$OUT"

TOPICS=(
  /side_1/left/image_raw/compressed /side_1/left/camera_info
  /side_2/side_2/infra1/image_rect_raw/compressed /side_2/side_2/infra1/camera_info
  /side_2/side_2/color/image_raw/compressed /side_2/side_2/color/camera_info
  /wrist/wrist/infra1/image_rect_raw/compressed /wrist/wrist/infra1/camera_info
  /wrist/wrist/color/image_raw/compressed /wrist/wrist/color/camera_info
  /side_1/detections /side_2/detections /wrist/detections
  /tf /tf_static /joint_states /robot_description
  /perception/object_pose/held_beaker /perception/object_pose/beaker2
  /perception/object_pose/held_beaker/in_hand /perception/object_pose/beaker2/in_hand
  /perception/object_markers /perception/object_visibility
  /scale/reading /real_cell/fixed_scene /real_cell/planned_path /real_cell/status
  /joint_trajectory_controller/controller_state /pouring_controller/controller_state
  /controller_action_server/pouring_controller/_action/feedback
  /controller_action_server/pouring_controller/_action/status
  /rosout
)

set +u
source /opt/ros/humble/setup.bash
source "${CELL_WS:-$ROOT/ros2_cell_ws}/install/setup.bash"   # the cell overlay; brings the core's
set -u

BAG_PID=""
start_bag() {  # start_bag NAME
  ros2 bag record --include-hidden-topics -o "$OUT/$1" "${TOPICS[@]}" \
    > "$OUT/$1.record.log" 2>&1 &
  BAG_PID=$!
  echo "bag_run: recording $OUT/$1"
}
stop_bag() {
  [[ -n "$BAG_PID" ]] || return 0
  kill -INT "$BAG_PID" 2>/dev/null
  wait "$BAG_PID" 2>/dev/null
  echo "bag_run: stopped (pid $BAG_PID)"
  BAG_PID=""
}
trap 'stop_bag' EXIT

start_bag "$FIRST"
"$ROOT/scripts/real_cell/plan_remote.sh" "$@" > "$LOG" 2>&1 &
RUN_PID=$!
phase=perceive; [[ $FIRST == 1_perceive ]] || phase=replay
while kill -0 "$RUN_PID" 2>/dev/null; do
  if [[ $phase == perceive ]] && grep -a -q "^=== planning on" "$LOG"; then
    stop_bag; phase=planning
  elif [[ $phase == planning && -n $REPLAYS ]] && grep -a -q "^=== ready:" "$LOG"; then
    start_bag 2_replay; phase=replay
  fi
  sleep 1
done
wait "$RUN_PID"; rc=$?
stop_bag
cp -f "$LOG" ~/real_cell_runs/live_logs/"bag_${STAMP}_${LABEL}.log" 2>/dev/null
echo "bag_run: plan_remote exited $rc; bags in $OUT"
du -sh "$OUT"/* 2>/dev/null
exit $rc
