#!/usr/bin/env bash
# OPTIONAL. Perception accuracy in the rendered scene: for each of the 30 seeded
# layouts, bring up Isaac Sim and the AprilTag pipeline and compare every fused
# vessel pose with the simulator's own spawn pose (measure_perception_seed.py).
#
# Nothing here is timed, so the GPU does not change the answer; it is here so a
# re-run can reproduce the whole perception table on the same commit. Run it
# when the GPU is free -- it brings the simulator up 30 times.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
rr_scene_guard
[[ -f "$RR_OUT/COMMIT" ]] || rr_die "run 00_preflight.sh first"
[[ "$(git -C "$RR_ROOT" rev-parse HEAD)" == "$(cat "$RR_OUT/COMMIT")" ]] \
  || rr_die "HEAD differs from this tag's commit $(cat "$RR_OUT/COMMIT")"

OUTV="$RR_OUT/perception"; LOGS="$RR_OUT/logs/perception_accuracy"
mkdir -p "$OUTV/frames" "$LOGS"
CSV="$OUTV/perception_seeds.csv"
rr_monitor_start

kill_group() {  # TERM a whole process group, then KILL what is left
  local g="$1"; [[ -z "$g" ]] && return 0
  kill -TERM -"$g" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 -"$g" 2>/dev/null || return 0; sleep 1; done
  kill -KILL -"$g" 2>/dev/null
}
rows_for() {  # how many rows (one per vessel) a seed already has
  python3 -c "import csv,os,sys
p,s=sys.argv[1],sys.argv[2]
print(sum(1 for r in csv.DictReader(open(p)) if r.get('seed')==s) if os.path.exists(p) and os.path.getsize(p) else 0)" "$CSV" "$1"
}

rr_batch_begin perception_accuracy
for seed in $(seq 0 29); do
  [[ "$(rows_for "$seed")" -ge 2 ]] && continue
  sim_log="$LOGS/sim_seed${seed}.log"
  SDL_SEED="$seed" ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}" \
    setsid bash "$RR_ROOT/scripts/run_isaacsim.sh" --headless > "$sim_log" 2>&1 < /dev/null &
  SIM_G=$!
  for _ in $(seq 1 45); do grep -q "Simulation Start" "$sim_log" 2>/dev/null && break; sleep 10; done
  if ! grep -q "Simulation Start" "$sim_log"; then
    rr_log "perception seed $seed: simulator never started"; kill_group "$SIM_G"; continue
  fi
  ( source /opt/ros/humble/setup.bash && source "$RR_WS/install/setup.bash" &&
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}" &&
    exec setsid ros2 launch perception_manager perception_manager.launch.py ) \
    > "$LOGS/pm_seed${seed}.log" 2>&1 < /dev/null &
  PM_G=$!
  sleep 20
  ( source /opt/ros/humble/setup.bash && source "$RR_WS/install/setup.bash" &&
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}" &&
    python3 "$RR_ANALYSIS/measure_perception_seed.py" "$seed" "$sim_log" "$CSV" "$OUTV/frames" ) \
    > "$LOGS/measure_seed${seed}.log" 2>&1
  kill_group "$PM_G"; kill_group "$SIM_G"
  rr_log "perception seed $seed: $(tail -1 "$LOGS/measure_seed${seed}.log" | cut -c1-120)"
done
status=done
rr_check_scene_logs "$LOGS" || status=scene_contaminated
rr_batch_end perception_accuracy "$status"
