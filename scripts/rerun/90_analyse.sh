#!/usr/bin/env bash
# Turn a campaign directory into results/ and RESULTS.md. Read-only on the data;
# safe to re-run at any point, including halfway through a campaign.
#
# The audit comes first and its verdict heads RESULTS.md: a number from a batch
# that shared the GPU, did not finish, or ran the wrong scene is not a result.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
A="$RR_ANALYSIS"; RES="$RR_OUT/results"; mkdir -p "$RES"
BUDGETS="$(cat "$RR_OUT/BUDGETS" 2>/dev/null || echo 60,120,180)"
MAXB="$(echo "$BUDGETS" | tr ',' '\n' | sort -n | tail -1)"

python3 "$A/rerun_audit.py" "$RR_OUT" > "$RES/audit.md" 2>&1; AUDIT_RC=$?

python3 "$A/planner_comparison.py" --cutamp "$RR_OUT/cutamp" --pddlstream "$RR_OUT/pddlstream" \
  --budgets "$BUDGETS" --out "$RES/planner.md" > /dev/null 2> "$RES/planner.err" \
  || echo "planner comparison failed: $(cat "$RES/planner.err")" > "$RES/planner.md"

for csv in "$RR_OUT"/cutamp/*.csv; do
  [[ -f "$csv" ]] || continue
  python3 "$A/summarize_trials.py" "$csv" --budget "$MAXB" > "$RES/trials_$(basename "$csv" .csv).txt" 2>&1
done

python3 "$A/paired_outcome.py" --field task_success \
  --a "$RR_OUT/cutamp/transfer_ground_truth_rep*.csv" --label-a ground_truth \
  --b "$RR_OUT/cutamp/transfer_perception_rep*.csv"   --label-b perception \
  > "$RES/state_source.md" 2>&1

for csv in "$RR_OUT"/cutamp/transfer_ground_truth_rep*.csv; do
  [[ -f "$csv" ]] || continue
  python3 "$A/analyse_pour_vector.py" "$csv" > "$RES/pour_$(basename "$csv" .csv).txt" 2>&1
done

[[ -s "$RR_OUT/perception/perception_seeds.csv" ]] && \
  python3 "$A/analyse_perception.py" "$RR_OUT/perception/perception_seeds.csv" > "$RES/perception.txt" 2>&1

L="$RR_OUT/llm"
{
  echo "## XDL Generator"
  grep -h "Avg Inference Time" "$L/xdl_generator.log" 2>/dev/null
  echo '```'; cat "$L/xdl_score.txt" 2>/dev/null; echo '```'
  echo "## XDL Validator"
  echo '```'; grep -E "benchmark:|TP |FP |accuracy|^  [a-z_]+ +[0-9]+/" "$L/validator.txt" 2>/dev/null; echo '```'
  for f in "$L"/ar_*.txt; do
    [[ -f "$f" ]] || continue
    echo "## Action Reasoner: $(basename "$f" .txt | sed 's/^ar_//')"
    echo '```'
    grep -E "Total:|Exact Match|Main Tool|Need Rearrange|Aux Tool|Target Grid|Avg Inference Time|^  (no_rearrange|rearrange_)" "$f"
    echo '```'
  done
  echo "## Tool rule (computed, against the same labels)"
  echo '```'; cat "$L/tool_rule.txt" 2>/dev/null; echo '```'
} > "$RES/llm.md"

{
  echo "# 재실험 결과 — \`$RERUN_TAG\`"
  echo
  echo "\`scripts/rerun/90_analyse.sh\`가 $(date '+%F %T')에 생성했습니다. 손으로 고치지 말고 다시 생성하세요."
  echo "환경과 코드는 [RUN_INFO.md](RUN_INFO.md), 절차와 원고 반영 위치는 저장소 루트의 \`RERUN.md\`에 있습니다."
  echo
  if [[ $AUDIT_RC -eq 0 ]]; then
    echo "> **감사: CLEAN.** 모든 배치가 끝났고, 실행 중 GPU를 공유한 프로세스가 없었고, 논문 scene에서 돌았습니다."
  else
    echo "> **감사: NOT CLEAN.** 아래 감사 표에서 굵게 표시되거나 \`done\`이 아닌 배치의 수치는 결과로 쓰지 마세요."
  fi
  echo
  cat "$RES/audit.md"
  echo; echo "---"; echo
  sed 's/^# /## /' "$RES/planner.md"
  echo; echo "---"; echo
  echo "## End-to-end Transfer: 정답 상태 vs 인식 상태"
  sed 's/^# /### /' "$RES/state_source.md"
  echo; echo "---"; echo
  echo "## 시행 요약 (예산 ${MAXB} s)"
  for f in "$RES"/trials_*.txt; do
    [[ -f "$f" ]] || continue
    echo "### $(basename "$f" .txt | sed 's/^trials_//')"
    echo '```'; grep -E "task success|planning success|time \[s\]|time <=|placement error" "$f"; echo '```'
  done
  echo; echo "---"; echo
  echo "## 붓기 립 오차 (정답 상태 Transfer)"
  for f in "$RES"/pour_*.txt; do
    [[ -f "$f" ]] || continue
    echo "### $(basename "$f" .txt | sed 's/^pour_//')"; echo '```'; cat "$f"; echo '```'
  done
  if [[ -f "$RES/perception.txt" ]]; then
    echo; echo "---"; echo; echo "## 인식 정확도 (렌더 카메라)"; echo '```'; cat "$RES/perception.txt"; echo '```'
  fi
  echo; echo "---"; echo
  echo "# 언어 모델"
  cat "$RES/llm.md"
} > "$RR_OUT/RESULTS.md"
echo "wrote $RR_OUT/RESULTS.md  (audit: $([[ $AUDIT_RC -eq 0 ]] && echo CLEAN || echo NOT CLEAN))"
