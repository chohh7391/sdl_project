#!/usr/bin/env bash
# PDDLStream baseline on the SAME 30 layouts (analysis/data/seed_layouts.json,
# exported bit-identically from Isaac Sim), RERUN_PDDL_STREAMS planner seeds per
# task, stopped at RERUN_PDDL_MAX_TIME. CPU only, single thread. A solve() that
# gives up early is restarted with fresh samples until that time is spent, as
# cuTAMP's rounds are; RERUN_PDDL_RESTART=0 runs one solve() per layout, as the
# 20261001c/e campaigns did.
#
# RERUN_PDDL_STATE=perception plans instead from the vessel poses cuTAMP's
# perception trial of each layout planned from (cutamp/<task>_perception_rep0
# and its trials' perception reports, exported by
# analysis/export_perceived_layouts.py into pddlstream_perception/). It needs
# the cuTAMP perception batches first; the rows go to pddlstream_perception/.
#
# The baseline alone may be measured again at a later commit, into the same
# tag, when nothing but the baseline and the analysis changed since the commit
# its inputs came from -- the tag's commit, or for the perception state the
# perception batches' (PERCEPTION_COMMIT): RERUN_ALLOW_COMMIT_CHANGE=1. The
# commit it ran at goes to PDDL_COMMIT (PDDL_PERCEPTION_COMMIT) and
# RUN_INFO.md. Move an old output directory aside first; the runner refuses to
# append rows with a different column set.
#
# Run it on the same machine as 10_cutamp.sh -- the paper states the two
# planners share a workstation -- but NOT at the same time: they would contend
# for the CPU, and the baseline's times would stop meaning anything.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
rr_scene_guard
[[ -f "$RR_OUT/COMMIT" ]] || rr_die "run 00_preflight.sh first"

STATE="${RERUN_PDDL_STATE:-ground_truth}"
case "$STATE" in
  ground_truth) OUTP="$RR_OUT/pddlstream"; PREFIX=pddlstream; COMMIT_FILE=PDDL_COMMIT
                BASE="$(cat "$RR_OUT/COMMIT")" ;;
  perception)   OUTP="$RR_OUT/pddlstream_perception"; PREFIX=pddlstream_perception
                COMMIT_FILE=PDDL_PERCEPTION_COMMIT
                BASE="$(cat "$RR_OUT/PERCEPTION_COMMIT" 2>/dev/null || cat "$RR_OUT/COMMIT")" ;;
  *) rr_die "RERUN_PDDL_STATE must be ground_truth or perception, not $STATE" ;;
esac

HEAD_NOW="$(git -C "$RR_ROOT" rev-parse HEAD)"
if [[ "$HEAD_NOW" != "$BASE" ]]; then
  [[ -n "${RERUN_ALLOW_COMMIT_CHANGE:-}" ]] || rr_die "HEAD differs from this tag's commit $BASE"
  # Everything else the tag holds was measured at $BASE, so only the baseline,
  # the campaign scripts and the analysis may have changed since.
  OTHER="$(git -C "$RR_ROOT" diff --name-only "$BASE" "$HEAD_NOW" \
    | grep -vE '^(experiments/pddlstream/|scripts/rerun/|_2026__IEEE_Access/revision/analysis/[^/]+\.py$|RERUN\.md$)' || true)"
  [[ -z "$OTHER" ]] || rr_die "HEAD changes more than the baseline since $BASE: $(echo $OTHER)"
  [[ -z "$(git -C "$RR_ROOT" status --porcelain --untracked-files=no -- .)" ]] \
    || rr_die "uncommitted changes; the baseline's commit would not describe its code"
  if [[ "$(cat "$RR_OUT/$COMMIT_FILE" 2>/dev/null)" != "$HEAD_NOW" ]]; then
    echo "$HEAD_NOW" > "$RR_OUT/$COMMIT_FILE"
    {
      echo
      echo "## PDDLStream ($STATE state) measured at a later commit ($(date '+%F %T'))"
      echo
      echo "| | |"
      echo "|---|---|"
      echo "| commit | \`$HEAD_NOW\` |"
      echo "| changed since \`$BASE\` | $(git -C "$RR_ROOT" diff --name-only "$BASE" "$HEAD_NOW" | sed 's/.*/`&`/' | paste -sd, - | sed 's/,/, /g') |"
      echo "| PDDLStream | ${RERUN_PDDL_STREAMS:-5} planner seeds x 30 layouts, max ${RERUN_PDDL_MAX_TIME:-180} s; $([[ "${RERUN_PDDL_RESTART:-1}" == 1 ]] && echo 'adaptive algorithm, restarted with fresh samples until the limit is spent' || echo 'adaptive algorithm, one solve() per trial, no restart') |"
      if [[ "$STATE" == perception ]]; then
        echo "| state | the vessel poses (x, y, yaw) cuTAMP's perception trial of each layout planned from; heights from the table, as in the ground-truth state |"
      fi
    } >> "$RR_OUT/RUN_INFO.md"
  fi
  rr_log "PDDLStream ($STATE) at $HEAD_NOW (inputs from $BASE; only the baseline and the analysis differ)"
fi
pgrep -f "$RR_ROOT/scripts/run_trials.sh" >/dev/null && rr_die "a cuTAMP batch is running; run the baseline after it"

mkdir -p "$OUTP"
STREAMS="${RERUN_PDDL_STREAMS:-5}"
MAXT="${RERUN_PDDL_MAX_TIME:-180}"
RESTART=(); [[ "${RERUN_PDDL_RESTART:-1}" == 1 ]] || RESTART=(--no-restart)
PY="$RR_ROOT/.venv-pddl/bin/python"
rr_monitor_start

for task in transfer move stir; do
  LAYOUTS=()
  if [[ "$STATE" == perception ]]; then
    lay="$OUTP/perceived_layouts_${task}.json"
    if [[ ! -s "$lay" ]]; then
      python3 "$RR_ANALYSIS/export_perceived_layouts.py" "$RR_OUT" --task "$task" --out "$lay" \
        || rr_die "could not export the perceived layouts of $task"
    fi
    LAYOUTS=(--layouts "$lay")
  fi
  # The file name keeps the "5streams" suffix planner_comparison.py reads,
  # whatever RERUN_PDDL_STREAMS is; the planner_seed column says how many.
  csv="$OUTP/pddlstream_${task}_5streams.csv"
  for ((ps = 0; ps < STREAMS; ps++)); do
    name="${PREFIX}_${task}_ps${ps}"
    missing="$(rr_missing_pddl_seeds "$csv" "$ps" 30)"
    if [[ -z "$missing" ]]; then rr_log "skip $name (complete)"; continue; fi
    rr_batch_begin "$name"
    # >>: a batch measured again keeps the earlier attempt's output above its own
    ( cd "$RR_ROOT/experiments/pddlstream" &&
      PYTHONPATH=. "$PY" examples/pybullet/fr5_paired/run_paired_trials.py \
        --task "$task" --max-time "$MAXT" --planner-seed "$ps" "${RESTART[@]}" \
        "${LAYOUTS[@]}" --seeds "$missing" --csv "$csv" ) >> "$RR_OUT/logs/${name}.out" 2>&1
    status=done
    rr_check_complete "$csv" 30 "$ps" || status=incomplete
    rr_batch_end "$name" "$status"
  done
done
rr_log "PDDLStream ($STATE) campaign finished"
