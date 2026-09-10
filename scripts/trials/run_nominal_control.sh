#!/usr/bin/env bash
# One nominal-layout control trial (SDL_SEED UNSET => hard-coded default layout),
# using the SAME code path as the seeded batch. Anchors the batch: confirms the
# planning pipeline still succeeds on the baseline layout with the STAGE-B1
# changes in place, so any per-seed plan failures are layout-induced (not a
# regression). Writes ONE row to the control CSV (seed=-1 sentinel = nominal).
set -uo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOGDIR="$PROJ_DIR/scripts/trials/logs"; mkdir -p "$LOGDIR"
CSV="${CSV:-$PROJ_DIR/_2026__IEEE_Access/revision/analysis/data/transfer_control_nominal.csv}"
ts="$(date +%Y%m%d_%H%M%S)"
SIM_LOG="$LOGDIR/nom_sim_$ts.log"; TAMP_LOG="$LOGDIR/nom_tamp_$ts.log"; DRV_LOG="$LOGDIR/nom_driver_$ts.log"

kill_group(){ local g="$1"; [[ -z "$g" ]]&&return 0; kill -TERM -"$g" 2>/dev/null; local w=0; while ((w<20)); do kill -0 -"$g" 2>/dev/null||return 0; sleep 1; w=$((w+1)); done; kill -KILL -"$g" 2>/dev/null; }
wait_for(){ local l="$1" p="$2" t="$3" w=0; while ((w<t)); do grep -qE "$p" "$l" 2>/dev/null&&return 0; sleep 2; w=$((w+2)); done; return 1; }

echo "=== nominal control (SDL_SEED unset) csv=$CSV ==="
env -u SDL_SEED ROS_DOMAIN_ID="$ROS_DOMAIN_ID" setsid bash "$SCRIPT_DIR/../run_isaacsim.sh" --headless >"$SIM_LOG" 2>&1 &
SIM_PGID=$!
if ! wait_for "$SIM_LOG" "Simulation Start" 360; then echo "SIM NOT READY"; kill_group "$SIM_PGID"; exit 1; fi
echo "sim ready (nominal)."; sleep 3
ROS_DOMAIN_ID="$ROS_DOMAIN_ID" setsid bash "$SCRIPT_DIR/../run_tamp.sh" >"$TAMP_LOG" 2>&1 &
TAMP_PGID=$!
if ! wait_for "$TAMP_LOG" "Starting TAMP server|Initialize TAMP Module" 120; then echo "TAMP NOT READY"; kill_group "$TAMP_PGID"; kill_group "$SIM_PGID"; exit 1; fi
sleep 5
ROS_DOMAIN_ID="$ROS_DOMAIN_ID" bash "$SCRIPT_DIR/run_trial_driver.sh" --seed -1 --csv "$CSV" >"$DRV_LOG" 2>&1
echo "driver done. CSV tail:"; tail -n 2 "$CSV"
kill_group "$TAMP_PGID"; kill_group "$SIM_PGID"; sleep 3
echo "nominal control complete."
