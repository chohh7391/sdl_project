#!/usr/bin/env bash
# cuTAMP in Isaac Sim, the paper's scene: Transfer / Move / Stir with the
# simulator's ground-truth state, then Transfer with the rendered-perception
# state, RERUN_REPS repetitions of the same 30 layouts each.
#
# cuTAMP is not reproducible from a seed, so each layout's outcome is estimated
# from several draws; repetition r uses planner seed offset r*1000, the same
# scheme as the 2026-09-12/13 campaign. Batches are ordered by repetition, so an
# interrupted campaign still holds complete repetitions of every task.
#
# Resumable: re-running skips complete batches and fills in the missing seeds of
# a partial one. PLAN_TIMEOUT is generous on purpose -- the full time to
# solution is recorded and budgets are applied as censoring in the analysis.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
rr_scene_guard
[[ -f "$RR_OUT/COMMIT" ]] || rr_die "run 00_preflight.sh first"
[[ "$(git -C "$RR_ROOT" rev-parse HEAD)" == "$(cat "$RR_OUT/COMMIT")" ]] \
  || rr_die "HEAD differs from this tag's commit $(cat "$RR_OUT/COMMIT")"

OUTC="$RR_OUT/cutamp"; mkdir -p "$OUTC"
REPS="${RERUN_REPS:-3}"
rr_monitor_start

run_batch() {   # task robot state_source rep
  local task="$1" robot="$2" src="$3" rep="$4"
  local name="${task}_${src}_rep${rep}"
  local csv="$OUTC/${name}.csv" logs="$RR_OUT/logs/$name"
  local missing; missing="$(rr_missing_seeds "$csv" 30)"
  if [[ -z "$missing" ]]; then rr_log "skip $name (complete)"; return 0; fi
  local foreign; foreign="$(rr_gpu_foreign)"
  [[ -z "$foreign" ]] || rr_die "another process is on the GPU, not starting $name: $foreign"
  rr_batch_begin "$name"
  SDL_STATE_SOURCE="$src" PLANNER_SEED_OFFSET="$((rep * 1000))" CSV="$csv" \
    TASK="$task" ROBOT="$robot" LOGDIR="$logs" PLAN_TIMEOUT="${RERUN_PLAN_TIMEOUT:-600}" \
    bash "$RR_ROOT/scripts/run_trials.sh" $missing > "$RR_OUT/logs/${name}.out" 2>&1
  local status=done
  rr_check_complete "$csv" 30 || status=incomplete
  rr_check_scene_logs "$logs" || status=scene_contaminated
  rr_batch_end "$name" "$status"
}

for ((rep = 0; rep < REPS; rep++)); do
  run_batch transfer fr5_ag95  ground_truth "$rep"
  run_batch move     fr5_vgc10 ground_truth "$rep"
  run_batch stir     fr5_dh3   ground_truth "$rep"
done
for ((rep = 0; rep < REPS; rep++)); do
  run_batch transfer fr5_ag95 perception "$rep"
done
rr_log "cuTAMP campaign finished"
