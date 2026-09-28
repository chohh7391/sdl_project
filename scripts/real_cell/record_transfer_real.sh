#!/usr/bin/env bash
# REAL-CELL scene only: plan and record Transfer trajectories for replay on the
# physical FR5. NEVER a source of paper numbers -- the paper's simulation is the
# unchanged scene run by scripts/rerun/. This writes OUTSIDE the paper's data
# directory and refuses a path inside it.
#
#   scripts/real_cell/record_transfer_real.sh ~/real_cell_runs/$(date +%Y%m%d) 0 1 2 3 4 5 6 7
#   scripts/real_cell/score_real_trajectories.py ~/real_cell_runs/<date>
#
# The scene, as measured on the rig (2026-09-23); override any value by
# exporting it before the call:
#   bench          13 mm below the robot base's reference plane
#   beaker         measures 50 x 50 x 70 mm, modelled 60 mm tall (tapered top),
#                  standing on a 50 mm box at base_link (0.584, -0.298)
#   flask          87 x 160 mm, on the balance pan at (0.520, 0.320)
#   balance        250 x 200 mm, pan top 90 mm, pan end facing the robot
#   grasp          level: 0-10 deg above horizontal
#   after pouring  the beaker goes back on its box
#
# Why the rig differs from the paper's scene: the level side grasp puts the
# wrist body ~58 mm below the grasp point, so a vessel standing on the bench
# could only be grasped at its tapered rim; the box raises it past that.
# NOT set -u: /opt/ros/humble/setup.bash reads unbound variables.
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:?usage: $0 OUTDIR seed...}"; shift
[[ $# -gt 0 ]] || { echo "give at least one seed" >&2; exit 2; }
OUT="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"
case "$OUT/" in "$ROOT/_2026__IEEE_Access/"*)
  echo "refusing to write real-cell runs under the paper's directory: $OUT" >&2; exit 3 ;; esac
mkdir -p "$OUT/logs"

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}" RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SDL_GLASSWARE="${SDL_GLASSWARE:-real}"
export SDL_TABLE_Z_M="${SDL_TABLE_Z_M:--0.013}"
export SDL_BEAKER_RISER_M="${SDL_BEAKER_RISER_M:-0.05}"
export SDL_GRASP_BETA_MIN_DEG="${SDL_GRASP_BETA_MIN_DEG:-0}" SDL_GRASP_BETA_MAX_DEG="${SDL_GRASP_BETA_MAX_DEG:-10}"
export SDL_GRASP_H_FRAC="${SDL_GRASP_H_FRAC:-0.0}"
export SDL_SCALE_PAN_XY="${SDL_SCALE_PAN_XY:-0.520,0.320}"
export SDL_LAYOUT_XY="${SDL_LAYOUT_XY:-beaker:0.584,-0.298}"
export TASK=transfer_real SDL_STATE_SOURCE=ground_truth PLAN_TIMEOUT="${PLAN_TIMEOUT:-300}"
export LOGDIR="$OUT/logs" CSV="$OUT/transfer_real.csv"
env | grep -E '^SDL_' | sort > "$OUT/SCENE_ENV.txt"
git -C "$ROOT" rev-parse HEAD > "$OUT/COMMIT"

for seed in "$@"; do
  [[ -s "$OUT/raw_seed${seed}.json" ]] && { echo "seed $seed: already recorded"; continue; }
  echo "=== seed $seed  $(date +%H:%M:%S)"
  python3 "$ROOT/scripts/trials/record_trajectory.py" \
    --out "$OUT/raw_seed${seed}.json" --seed "$seed" --tool fr5_ag95 \
    --seconds 900 --stop-after-idle 25 --start-timeout 600 > "$OUT/logs/recorder_${seed}.log" 2>&1 &
  REC=$!
  sleep 2
  bash "$ROOT/scripts/run_trials.sh" "$seed" > "$OUT/logs/run_seed${seed}.log" 2>&1
  # The trial is over: let the recorder flush, then stop it rather than wait it out.
  for _ in $(seq 1 40); do kill -0 "$REC" 2>/dev/null || break; sleep 1; done
  kill -TERM "$REC" 2>/dev/null; wait "$REC" 2>/dev/null
  tail -1 "$CSV" 2>/dev/null | cut -d, -f2,5,10,18
done
echo "=== done; score with: $ROOT/scripts/real_cell/score_real_trajectories.py $OUT"
