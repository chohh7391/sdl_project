#!/usr/bin/env bash
# =============================================================================
# run_trials.sh -- randomized-trials harness (STAGE B1, REFACTOR.md I-7 / D3)
#
# For each SEED: bring up a FRESH Isaac Sim (with SDL_SEED baked into a seeded,
# randomized object layout) + a FRESH tamp_server (one set_tamp_cfg per server
# lifetime -- works around the unfixed layer-3 config-switch corruption), drive
# ONE transfer trial (tool_change ag95 -> set_tamp_cfg fr5_ag95 -> set_tamp_env
# transfer -> plan -> execute), append a CSV row, then tear both down cleanly.
#
# Each sim/server is launched in its OWN process group (setsid) and killed by
# group id, so nothing else on the machine (e.g. an unrelated GPU job) is
# touched -- no broad pkill.
#
# Usage:
#   scripts/run_trials.sh                 # seeds 0 1 2 3 4 (default)
#   scripts/run_trials.sh 0               # a single seed (smoke test)
#   scripts/run_trials.sh 0 1 2 3 4 5 6   # explicit seed list
#
# Env overrides:
#   ROS_DOMAIN_ID (default 100)   SIM_READY_TIMEOUT (default 360)
#   CSV (default <proj>/_2026__IEEE_Access/revision/analysis/data/transfer_trials.csv)
#   TASK  (default transfer)  -- transfer | move | stir, per orchestration/registry
#   ROBOT (default: the tool TASK is evaluated with, per content/configs/xdl/tool_map.yml:
#          transfer -> fr5_ag95 (2-finger, side grasp), move -> fr5_vgc10 (suction),
#          stir -> fr5_dh3 (3-finger))
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"                    # .../src/sdl_project
SIM_SH="$SCRIPT_DIR/run_isaacsim.sh"
TAMP_SH="$SCRIPT_DIR/run_tamp.sh"
ORCHESTRATOR="$SCRIPT_DIR/trials/run_orchestrator.sh"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CSV="${CSV:-$PROJ_DIR/_2026__IEEE_Access/revision/analysis/data/transfer_trials.csv}"
TASK="${TASK:-transfer}"
case "$TASK" in
  transfer) DEFAULT_ROBOT="fr5_ag95" ;;
  move)     DEFAULT_ROBOT="fr5_vgc10" ;;
  stir)     DEFAULT_ROBOT="fr5_dh3" ;;
  *)        DEFAULT_ROBOT="fr5_ag95" ;;
esac
ROBOT="${ROBOT:-$DEFAULT_ROBOT}"
LOGDIR="${LOGDIR:-$PROJ_DIR/scripts/trials/logs}"
mkdir -p "$LOGDIR" "$(dirname "$CSV")"

SIM_READY_TIMEOUT="${SIM_READY_TIMEOUT:-360}"
TAMP_READY_TIMEOUT="${TAMP_READY_TIMEOUT:-120}"

SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then SEEDS=(0 1 2 3 4); fi

echo "=== run_trials.sh: task=$TASK robot=$ROBOT seeds=[${SEEDS[*]}] domain=$ROS_DOMAIN_ID csv=$CSV ==="

# Wait until $1 (a log file) contains regex $2, up to $3 seconds. Returns 0/1.
wait_for() {
  local log="$1" pat="$2" timeout="$3" waited=0
  while (( waited < timeout )); do
    if grep -qE "$pat" "$log" 2>/dev/null; then return 0; fi
    sleep 2; waited=$((waited + 2))
  done
  return 1
}

# Kill a whole process group (setsid leader pid == pgid): TERM, then KILL.
kill_group() {
  local pgid="$1"
  [[ -z "$pgid" ]] && return 0
  kill -TERM -"$pgid" 2>/dev/null
  local w=0
  while (( w < 20 )); do
    kill -0 -"$pgid" 2>/dev/null || return 0
    sleep 1; w=$((w + 1))
  done
  kill -KILL -"$pgid" 2>/dev/null
  sleep 1
}

# Kill any Isaac Sim / tamp_server left over from an earlier run. A survivor
# answers on the same ROS domain, so the driver can talk to a simulator holding a
# DIFFERENT seed's layout: it then plans against the wrong world and the CSV
# silently records poses that belong to another trial (observed 2026-09-10 --
# three seeds in a row read the same beaker/flask coordinates, which were a
# fourth seed's). Fail loudly rather than produce a plausible-looking batch.
reap_strays() {
  local pids
  pids="$(pgrep -f 'standalone/simulation\.py|tamp_server\.py' || true)"
  [[ -z "$pids" ]] && return 0
  echo "[pre-flight] STRAY sim/server process(es) still alive: $pids -- killing."
  kill -TERM $pids 2>/dev/null
  sleep 5
  pids="$(pgrep -f 'standalone/simulation\.py|tamp_server\.py' || true)"
  [[ -n "$pids" ]] && { kill -KILL $pids 2>/dev/null; sleep 2; }
  return 0
}

for seed in "${SEEDS[@]}"; do
  reap_strays
  ts="$(date +%Y%m%d_%H%M%S)"
  SIM_LOG="$LOGDIR/sim_seed${seed}_${ts}.log"
  TAMP_LOG="$LOGDIR/tamp_seed${seed}_${ts}.log"
  DRV_LOG="$LOGDIR/driver_seed${seed}_${ts}.log"
  echo "----------------------------------------------------------------------"
  echo "[seed $seed] sim_log=$SIM_LOG"

  # --- 1. fresh Isaac Sim with the seeded layout ---------------------------
  # Isaac Sim occasionally dies during startup without writing a single line to
  # its log (observed twice in ~45 launches). Retry a bounded number of times,
  # and if it still will not boot, RECORD the seed as an infrastructure failure
  # instead of skipping it: a missing CSV row is invisible in the aggregate (the
  # first 30-seed batch silently produced 29 rows), whereas a row with
  # failure_reason=sim_boot_failed is auditable and clearly not a planning failure.
  SIM_PGID=""
  sim_ready=0
  for sim_try in 1 2 3; do
    SDL_SEED="$seed" ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
      setsid bash "$SIM_SH" --headless >"$SIM_LOG" 2>&1 &
    SIM_PGID=$!
    echo "[seed $seed] sim pgid=$SIM_PGID (try $sim_try/3); waiting for 'Simulation Start' (<= ${SIM_READY_TIMEOUT}s)"
    if wait_for "$SIM_LOG" "Simulation Start" "$SIM_READY_TIMEOUT"; then
      sim_ready=1
      break
    fi
    echo "[seed $seed] sim did not reach 'Simulation Start' on try $sim_try (log $(wc -c <"$SIM_LOG") bytes); tearing down."
    kill_group "$SIM_PGID"
    SIM_PGID=""
    sleep 10
  done
  if (( sim_ready == 0 )); then
    echo "[seed $seed] SIM WILL NOT BOOT after 3 tries -- recording sim_boot_failed and moving on."
    if [[ ! -s "$CSV" ]]; then
      echo "timestamp,seed,task,robot_cfg,plan_success,planning_time_s,num_satisfying,execute_success,max_transport_tilt_deg,failure_reason,beaker_xy,flask_xy,poured,final_tilt_deg,final_xy,placed_upright,placed_in_goal,task_success,target_obj,goal_err_mm,aux_check,pour_peak_tilt_deg,pour_lip_err_mm,pour_start_lip_err_mm" >"$CSV"
    fi
    echo "$(date -Is),$seed,$TASK,$ROBOT,False,,,False,,sim_boot_failed,,,False,,,False,False,False,,,,,," >>"$CSV"
    continue
  fi
  echo "[seed $seed] sim ready. randomized layout:"
  grep -E "SDL_SEED=$seed|randomized layout|home_arm|beaker |flask |magnet |box |stirrer " "$SIM_LOG" | tail -n 10 | sed "s/^/[seed $seed]   /"
  sleep 3

  # --- 2. fresh tamp_server ------------------------------------------------
  ROS_DOMAIN_ID="$ROS_DOMAIN_ID" SDL_USE_SOURCE=1 \
    setsid bash "$TAMP_SH" >"$TAMP_LOG" 2>&1 &
  TAMP_PGID=$!
  echo "[seed $seed] tamp pgid=$TAMP_PGID; waiting for server ready (<= ${TAMP_READY_TIMEOUT}s)"
  if ! wait_for "$TAMP_LOG" "Starting TAMP server|Initialize TAMP Module" "$TAMP_READY_TIMEOUT"; then
    echo "[seed $seed] TAMP SERVER NOT READY -- see $TAMP_LOG; tearing down."
    kill_group "$TAMP_PGID"; kill_group "$SIM_PGID"
    continue
  fi
  sleep 5   # let services + subscriptions settle

  # --- 3. drive the trial (appends the CSV row itself) ---------------------
  echo "[seed $seed] running canonical task orchestrator..."
  ROS_DOMAIN_ID="$ROS_DOMAIN_ID" bash "$ORCHESTRATOR" \
      --seed "$seed" --csv "$CSV" --task "$TASK" --robot "$ROBOT" >"$DRV_LOG" 2>&1
  echo "[seed $seed] driver done. CSV tail:"
  tail -n 2 "$CSV" | sed "s/^/[seed $seed]   /"

  # Integrity check: the pose the driver read back must be the pose THIS sim
  # spawned. A mismatch means the driver talked to a different simulator (see
  # reap_strays above), which invalidates the row.
  spawn_xy="$(grep -oE 'flask +xy=\(-?[0-9.]+,-?[0-9.]+\)' "$SIM_LOG" | head -1 |
              grep -oE '\(-?[0-9.]+,-?[0-9.]+\)' | tr -d '()')"
  read_xy="$(tail -n 1 "$CSV" | awk -F'","|^"|","|"$' '{print}' |
             python3 -c 'import csv,sys; r=list(csv.reader(sys.stdin)); print(r[0][11] if r and len(r[0])>11 else "")')"
  if [[ -n "$spawn_xy" && -n "$read_xy" ]]; then
    if ! python3 -c "
import sys
sx, sy = [float(v) for v in '$spawn_xy'.split(',')]
rx, ry = [float(v) for v in '$read_xy'.split(';')]
sys.exit(0 if abs(sx - rx) <= 0.02 and abs(sy - ry) <= 0.02 else 1)
" 2>/dev/null; then
      echo "[seed $seed] !! POSE MISMATCH: sim spawned flask at ($spawn_xy) but the driver read ($read_xy)."
      echo "[seed $seed] !! This row is INVALID -- the driver was talking to another simulator."
    fi
  fi

  # --- 4. teardown ---------------------------------------------------------
  echo "[seed $seed] tearing down (tamp then sim)."
  kill_group "$TAMP_PGID"
  kill_group "$SIM_PGID"
  sleep 5
done

echo "======================================================================"
echo "run_trials.sh complete. CSV: $CSV"
echo "logs under: $LOGDIR"
