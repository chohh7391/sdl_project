#!/usr/bin/env bash
# PDDLStream baseline on the SAME 30 layouts (analysis/data/seed_layouts.json,
# exported bit-identically from Isaac Sim), RERUN_PDDL_STREAMS planner seeds per
# task, stopped at RERUN_PDDL_MAX_TIME. CPU only, single thread.
#
# Run it on the same machine as 10_cutamp.sh -- the paper states the two
# planners share a workstation -- but NOT at the same time: they would contend
# for the CPU, and the baseline's times would stop meaning anything.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
rr_scene_guard
[[ -f "$RR_OUT/COMMIT" ]] || rr_die "run 00_preflight.sh first"
[[ "$(git -C "$RR_ROOT" rev-parse HEAD)" == "$(cat "$RR_OUT/COMMIT")" ]] \
  || rr_die "HEAD differs from this tag's commit $(cat "$RR_OUT/COMMIT")"
pgrep -f "$RR_ROOT/scripts/run_trials.sh" >/dev/null && rr_die "a cuTAMP batch is running; run the baseline after it"

OUTP="$RR_OUT/pddlstream"; mkdir -p "$OUTP"
STREAMS="${RERUN_PDDL_STREAMS:-5}"
MAXT="${RERUN_PDDL_MAX_TIME:-180}"
PY="$RR_ROOT/.venv-pddl/bin/python"
rr_monitor_start

for task in transfer move stir; do
  # The file name keeps the "5streams" suffix planner_comparison.py reads,
  # whatever RERUN_PDDL_STREAMS is; the planner_seed column says how many.
  csv="$OUTP/pddlstream_${task}_5streams.csv"
  for ((ps = 0; ps < STREAMS; ps++)); do
    name="pddlstream_${task}_ps${ps}"
    missing="$(rr_missing_pddl_seeds "$csv" "$ps" 30)"
    if [[ -z "$missing" ]]; then rr_log "skip $name (complete)"; continue; fi
    rr_batch_begin "$name"
    ( cd "$RR_ROOT/experiments/pddlstream" &&
      PYTHONPATH=. "$PY" examples/pybullet/fr5_paired/run_paired_trials.py \
        --task "$task" --max-time "$MAXT" --planner-seed "$ps" \
        --seeds "$missing" --csv "$csv" ) > "$RR_OUT/logs/${name}.out" 2>&1
    status=done
    rr_check_complete "$csv" 30 "$ps" || status=incomplete
    rr_batch_end "$name" "$status"
  done
done
rr_log "PDDLStream campaign finished"
