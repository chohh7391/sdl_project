#!/usr/bin/env bash
# The language-model experiments, on the checkpoints the paper evaluates:
#   XDL Generator   generation + field-level scoring + inference time  (R1#5)
#   XDL Validator   the injected-error benchmark                       (R1#6)
#   Action Reasoner held-out 600 (model_split2400), the unseen levels
#                   (model_unseen), inference time, and the computed
#                   tool rule against the same labels                  (R1#7)
#
# Accuracy should reproduce; inference TIME is what this machine changes, and
# the paper states both. Nothing is retrained.
#
# Every model runs with its working directory under logs/, because unsloth
# writes a compile cache into the working directory and the copy beside the
# scripts is tracked in git -- running there would dirty the tree.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
rr_scene_guard
[[ -f "$RR_OUT/COMMIT" ]] || rr_die "run 00_preflight.sh first"
[[ "$(git -C "$RR_ROOT" rev-parse HEAD)" == "$(cat "$RR_OUT/COMMIT")" ]] \
  || rr_die "HEAD differs from this tag's commit $(cat "$RR_OUT/COMMIT")"
pgrep -f "$RR_ROOT/scripts/run_trials.sh" >/dev/null && rr_die "a simulator batch is running; the timings would be shared"

OUTL="$RR_OUT/llm"; CWD="$RR_OUT/logs/llm_cwd"; mkdir -p "$OUTL" "$CWD"
X="$RR_ROOT/LLM/llama/script/experiments/XDL_generator"
AR="$RR_ROOT/LLM/llama/script/experiments/ActionReasoner"
TRAIN_XDL="$RR_ROOT/LLM/llama/data/dataset/pre_data/procedure_instruction_v1.jsonl"
rr_monitor_start
rr_conda_sdl

# --- XDL Generator -------------------------------------------------------------
if [[ ! -s "$OUTL/xdl_generations.json" ]]; then
  [[ -z "$(rr_gpu_foreign)" ]] || rr_die "another process is on the GPU"
  rr_batch_begin llm_xdl_generator
  ( cd "$CWD" && XDL_GENERATIONS_OUT="$OUTL/xdl_generations.json" python "$X/exper.py" ) \
    > "$OUTL/xdl_generator.log" 2>&1
  rr_batch_end llm_xdl_generator "$([[ -s "$OUTL/xdl_generations.json" ]] && echo done || echo failed)"
fi
python "$X/score_xdl.py" --preds "$OUTL/xdl_generations.json" --train "$TRAIN_XDL" \
  > "$OUTL/xdl_score.txt" 2>&1

# --- XDL Validator (CPU, deterministic) ---------------------------------------------
python "$X/validator_benchmark.py" --out "$OUTL/validator_benchmark.json" \
  > "$OUTL/validator.txt" 2>&1

# --- Action Reasoner -----------------------------------------------------------------
ar_eval() {  # name model_dir test_jsonl
  local name="$1" model="$2" test="$3"
  [[ -s "$OUTL/ar_${name}.jsonl" ]] && { rr_log "skip ar_$name (done)"; return 0; }
  [[ -z "$(rr_gpu_foreign)" ]] || rr_die "another process is on the GPU"
  rr_batch_begin "llm_ar_$name"
  ( cd "$CWD" && AR_MODEL_DIR="$model" AR_TEST_PATH="$test" AR_RESULT_PATH="$OUTL/ar_${name}.jsonl" \
      python "$AR/script/eval_action_reasoner.py" ) > "$OUTL/ar_${name}.txt" 2>&1
  rr_batch_end "llm_ar_$name" "$([[ -s "$OUTL/ar_${name}.jsonl" ]] && echo done || echo failed)"
}
ar_eval heldout_split2400 "$AR/model_split2400/checkpoint" "$AR/dataset/action_reasoner_test.jsonl"
for S in control_in_distribution L1_unseen_class L2_unseen_layout L3_unseen_pair; do
  ar_eval "unseen_$S" "$AR/model_unseen/checkpoint" "$AR/dataset/unseen/$S.jsonl"
done

# The computed tool rule, against the same labels (no model, CPU).
{
  python "$AR/script/tool_rule.py" "$AR/dataset/action_reasoner_test.jsonl"
  for S in control_in_distribution L1_unseen_class L2_unseen_layout L3_unseen_pair; do
    python "$AR/script/tool_rule.py" "$AR/dataset/unseen/$S.jsonl"
  done
} > "$OUTL/tool_rule.txt" 2>&1
conda deactivate 2>/dev/null
rr_log "LLM experiments finished"
