#!/usr/bin/env bash
# The whole campaign, in order. Run it inside tmux: it takes most of a day.
#
#   export RERUN_TAG=rtx5080_$(date +%Y%m%d)
#   tmux new -s rerun -e RERUN_TAG="$RERUN_TAG" \
#     "scripts/rerun/run_all.sh 2>&1 | tee -a ~/rerun_${RERUN_TAG}.log"
#
# The console log goes OUTSIDE the repository: a file written inside it would
# read as an uncommitted change and stop the pre-flight when the run resumes.
#
# Resumable: after an interruption, run it again with the SAME RERUN_TAG and it
# continues where it stopped. RERUN_PERCEPTION=1 adds the optional perception-
# accuracy stage.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
rr_scene_guard
D="$(dirname "${BASH_SOURCE[0]}")"
bash "$D/00_preflight.sh"   || exit 1
bash "$D/10_cutamp.sh"      || rr_log "10_cutamp.sh stopped early; its completed batches are kept"
bash "$D/20_pddlstream.sh"  || rr_log "20_pddlstream.sh stopped early; its completed batches are kept"
if [[ "${RERUN_PERCEPTION:-0}" == "1" ]]; then
  bash "$D/30_perception.sh" || rr_log "30_perception.sh stopped early"
fi
bash "$D/40_llm.sh"         || rr_log "40_llm.sh stopped early"
bash "$D/90_analyse.sh"
rr_monitor_stop
rr_log "campaign $RERUN_TAG: run_all.sh finished -- read $RR_OUT/RESULTS.md, audit first"
