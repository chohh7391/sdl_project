#!/usr/bin/env bash
# cuTAMP in Isaac Sim, the paper's scene: Transfer / Move / Stir with the
# simulator's ground-truth state, then the same three with the rendered-
# perception state, RERUN_REPS repetitions of the same 30 layouts each.
#
# cuTAMP is not reproducible from a seed, so each layout's outcome is estimated
# from several draws; repetition r uses planner seed offset r*1000, the same
# scheme as the 2026-09-12/13 campaign. Batches are ordered by repetition, so an
# interrupted campaign still holds complete repetitions of every task.
#
# Resumable: re-running skips complete batches and fills in the missing seeds of
# a partial one. The planner restarts until the largest declared budget; the
# attempt running when it ends is allowed to finish, so PLAN_TIMEOUT is
# generous on purpose -- the full time to solution is recorded and budgets are
# applied as censoring in the analysis.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
rr_scene_guard
[[ -f "$RR_OUT/COMMIT" ]] || rr_die "run 00_preflight.sh first"
# The perception batches alone may be measured again at a later commit, into
# the same tag, when the ground-truth batches are complete (they are then only
# skipped, never extended at the new commit) and nothing outside the perception
# path, the baseline, the campaign scripts and the analysis changed since the
# tag's commit: RERUN_ALLOW_COMMIT_CHANGE=1. Move the old perception CSVs and
# logs aside first. The commit goes to PERCEPTION_COMMIT and RUN_INFO.md.
HEAD_NOW="$(git -C "$RR_ROOT" rev-parse HEAD)"; BASE="$(cat "$RR_OUT/COMMIT")"
COMMIT_CHANGED=0
if [[ "$HEAD_NOW" != "$BASE" ]]; then
  [[ -n "${RERUN_ALLOW_COMMIT_CHANGE:-}" ]] || rr_die "HEAD differs from this tag's commit $BASE"
  OTHER="$(git -C "$RR_ROOT" diff --name-only "$BASE" "$HEAD_NOW" \
    | grep -vE '^(perception/|TAMP/tamp/scripts/server/tamp_server\.py$|TAMP/tamp/test/|experiments/pddlstream/|scripts/rerun/|_2026__IEEE_Access/revision/analysis/[^/]+\.py$|_2026__IEEE_Access/revision/analysis/data/rerun/|RERUN\.md$|\.gitignore$)' || true)"
  [[ -z "$OTHER" ]] || rr_die "HEAD changes more than the perception path since $BASE: $(echo $OTHER)"
  # The campaign records under data/rerun/ change during a run (rerun.log, the
  # CSVs, RESULTS.md); they are outputs, not the code the commit describes.
  [[ -z "$(git -C "$RR_ROOT" status --porcelain --untracked-files=no -- . ':!_2026__IEEE_Access/revision/analysis/data/rerun')" ]] \
    || rr_die "uncommitted changes; the batches' commit would not describe their code"
  COMMIT_CHANGED=1
  if [[ "$(cat "$RR_OUT/PERCEPTION_COMMIT" 2>/dev/null)" != "$HEAD_NOW" ]]; then
    echo "$HEAD_NOW" > "$RR_OUT/PERCEPTION_COMMIT"
    {
      echo
      echo "## Perception batches measured again at a later commit ($(date '+%F %T'))"
      echo
      echo "The ground-truth batches keep \`$BASE\`."
      echo
      echo "| | |"
      echo "|---|---|"
      echo "| commit | \`$HEAD_NOW\` |"
      echo "| changed since \`$BASE\` | $(git -C "$RR_ROOT" diff --name-only "$BASE" "$HEAD_NOW" | sed 's/.*/`&`/' | paste -sd, - | sed 's/,/, /g') |"
    } >> "$RR_OUT/RUN_INFO.md"
  fi
  rr_log "cuTAMP perception batches at $HEAD_NOW (tag commit $BASE)"
fi

OUTC="$RR_OUT/cutamp"; mkdir -p "$OUTC"
REPS="${RERUN_REPS:-3}"
# cuTAMP plans until the LARGEST declared budget: it restarts until it finds a
# plan or runs out, and the smaller budgets are censoring of the same run.
MAXB="$(tr ',' '\n' < "$RR_OUT/BUDGETS" | sort -n | tail -1)"
rr_monitor_start

run_batch() {   # task robot state_source rep
  local task="$1" robot="$2" src="$3" rep="$4"
  local name="${task}_${src}_rep${rep}"
  local csv="$OUTC/${name}.csv" logs="$RR_OUT/logs/$name"
  local missing; missing="$(rr_missing_seeds "$csv" 30)"
  if [[ -z "$missing" ]]; then rr_log "skip $name (complete)"; return 0; fi
  [[ "$COMMIT_CHANGED" == 0 || "$src" == perception ]] \
    || rr_die "$name is incomplete, and at $HEAD_NOW only perception batches may run"
  local foreign; foreign="$(rr_gpu_foreign)"
  [[ -z "$foreign" ]] || rr_die "another process is on the GPU, not starting $name: $foreign"
  rr_batch_begin "$name"
  SDL_STATE_SOURCE="$src" PLANNER_SEED_OFFSET="$((rep * 1000))" CSV="$csv" \
    TASK="$task" ROBOT="$robot" LOGDIR="$logs" PLAN_TIMEOUT="${RERUN_PLAN_TIMEOUT:-600}" PLAN_BUDGET_S="$MAXB" \
    bash "$RR_ROOT/scripts/run_trials.sh" $missing >> "$RR_OUT/logs/${name}.out" 2>&1
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
# The same tasks with the rendered-perception World State: the tagged vessels
# from the AprilTag cameras (fixed pair, wrist-camera recovery scan, the check
# before execution), everything else from the simulator.
for ((rep = 0; rep < REPS; rep++)); do
  run_batch transfer fr5_ag95  perception "$rep"
  run_batch move     fr5_vgc10 perception "$rep"
  run_batch stir     fr5_dh3   perception "$rep"
done
rr_log "cuTAMP campaign finished"
