import os
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
import json
import torch
from collections import defaultdict
from unsloth import FastLanguageModel

# =========================
# Config
# =========================
# LLAMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "llama")
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ar_prompt import build_prompt, geometry_enabled
print("[eval] geometry grounding: %s" % ("ON" if geometry_enabled() else "off"))

_AR = "/home/home/sdl_ws/src/sdl_project/LLM/llama/script/experiments/ActionReasoner"
# Overridable so the same evaluation can be pointed at a checkpoint trained on
# the 2400-sample split, which the shipped one was not (see train.py).
MODEL_DIR = os.environ.get("AR_MODEL_DIR", os.path.join(_AR, "model/checkpoint"))
TEST_PATH = os.environ.get("AR_TEST_PATH",
                           os.path.join(_AR, "dataset/action_reasoner_test.jsonl"))
RESULT_PATH = os.environ.get("AR_RESULT_PATH",
                             os.path.join(_AR, "results/eval_results.jsonl"))
print("[eval] model=%s" % MODEL_DIR)
print("[eval] test =%s" % TEST_PATH)

MAX_SEQ_LEN = 512


# =========================
# Grid category matching
# =========================
def parse_candidate_categories(candidate_grids_str):
    """candidate_grids 문자열을 파싱하여 {그리드: 카테고리} 매핑 반환"""
    grid_to_cat = {}
    for line in candidate_grids_str.strip().split("\n"):
        if ":" not in line:
            continue
        cat, grids_part = line.split(":", 1)
        cat = cat.strip()
        grids_part = grids_part.strip().strip("[]")
        for g in grids_part.split(","):
            g = g.strip()
            if g:
                grid_to_cat[g] = cat
    return grid_to_cat


def is_same_grid_category(gt_grid, pred_grid, candidate_grids_str):
    """GT와 Pred 그리드가 동일 카테고리에 속하는지 확인"""
    if gt_grid == pred_grid:
        return True
    if gt_grid == "None" or pred_grid == "None":
        return gt_grid == pred_grid
    grid_to_cat = parse_candidate_categories(candidate_grids_str)
    gt_cat = grid_to_cat.get(gt_grid)
    pred_cat = grid_to_cat.get(pred_grid)
    if gt_cat and pred_cat and gt_cat == pred_cat:
        return True
    return False

# =========================
# Load model
# =========================
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_DIR,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(model)
print("Model loaded.\n")

# =========================
# Load test data
# =========================
print("Loading test data...")
test_data = []
with open(TEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        test_data.append(json.loads(line))
print(f"Test samples: {len(test_data)}\n")

# =========================
# Inference
# =========================
def predict(example):
    inst = example["instruction"]
    prompt = build_prompt(inst)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip().split("\n")[0].strip()

# =========================
# Evaluate
# =========================
print("Running evaluation...")

results = {
    "total": 0,
    "format_errors": 0,
    "exact_match": 0,
    "main_tool_correct": 0,
    "need_rearrange_correct": 0,
    "aux_tool_correct": 0,
    "target_grid_correct": 0,
}

category_results = defaultdict(lambda: {"total": 0, "exact_match": 0})
eval_entries = []

for i, example in enumerate(test_data):
    gt = example["output"]
    pred = predict(example)

    gt_tokens = gt.split(",")
    pred_tokens = pred.split(",")

    results["total"] += 1

    # 카테고리 판별
    is_rearrange = gt_tokens[1] == "True"
    obs_info = example["instruction"]["obstacle_info"]
    next_xdl = example["instruction"]["next_xdl"]

    if not is_rearrange:
        category = "no_rearrange"
    else:
        obs_name = obs_info.split(" at ")[0] if obs_info != "None" else ""
        if obs_name and obs_name in next_xdl:
            category = "rearrange_context"
        else:
            category = "rearrange_irrelevant"

    category_results[category]["total"] += 1

    # 결과 엔트리
    candidate_grids_str = example["instruction"]["candidate_grids"]
    grid_cat_match = is_same_grid_category(
        gt_tokens[3].strip() if len(gt_tokens) == 4 else "",
        pred_tokens[3].strip() if len(pred_tokens) == 4 else "",
        candidate_grids_str
    )

    entry = {
        "index": i,
        "category": category,
        "gt": gt,
        "pred": pred,
        "instruction": example["instruction"],
    }

    # 토큰 수 불일치
    if len(pred_tokens) != 4:
        results["format_errors"] += 1
        entry["correct"] = False
        entry["error"] = "format_error"
        eval_entries.append(entry)
        continue

    # 개별 토큰 비교
    gt_main, gt_rearrange, gt_aux, gt_grid = [t.strip() for t in gt_tokens]
    pred_main, pred_rearrange, pred_aux, pred_grid = [t.strip() for t in pred_tokens]

    main_ok = pred_main == gt_main
    rearrange_ok = pred_rearrange == gt_rearrange
    aux_ok = pred_aux == gt_aux
    grid_ok = is_same_grid_category(gt_grid, pred_grid, candidate_grids_str)

    if main_ok:
        results["main_tool_correct"] += 1
    if rearrange_ok:
        results["need_rearrange_correct"] += 1
    if aux_ok:
        results["aux_tool_correct"] += 1
    if grid_ok:
        results["target_grid_correct"] += 1

    # 전체 일치: main_tool + need_rearrange + aux_tool 정확 + target_grid 카테고리 일치
    all_correct = main_ok and rearrange_ok and aux_ok and grid_ok
    entry["correct"] = all_correct
    entry["grid_exact"] = (pred_grid == gt_grid)
    entry["grid_category_match"] = grid_ok

    if all_correct:
        results["exact_match"] += 1
        category_results[category]["exact_match"] += 1

    eval_entries.append(entry)

    # 진행률
    if (i + 1) % 50 == 0:
        acc = results["exact_match"] / (i + 1) * 100
        print(f"  [{i+1}/{len(test_data)}] running accuracy: {acc:.1f}%")

# =========================
# Print results
# =========================
total = results["total"]
valid = total - results["format_errors"]

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(f"\nTotal: {total}, Format errors: {results['format_errors']}")

print(f"\n--- Overall Accuracy ---")
print(f"  Exact Match:      {results['exact_match']}/{total} ({100*results['exact_match']/total:.1f}%)")
if valid > 0:
    print(f"  Main Tool:        {results['main_tool_correct']}/{valid} ({100*results['main_tool_correct']/valid:.1f}%)")
    print(f"  Need Rearrange:   {results['need_rearrange_correct']}/{valid} ({100*results['need_rearrange_correct']/valid:.1f}%)")
    print(f"  Aux Tool:         {results['aux_tool_correct']}/{valid} ({100*results['aux_tool_correct']/valid:.1f}%)")
    print(f"  Target Grid (category match): {results['target_grid_correct']}/{valid} ({100*results['target_grid_correct']/valid:.1f}%)")

    grid_exact = sum(1 for e in eval_entries if e.get("grid_exact", False))
    print(f"  Target Grid (exact match):    {grid_exact}/{valid} ({100*grid_exact/valid:.1f}%)")

print(f"\n--- Accuracy by Category ---")
for cat in ["no_rearrange", "rearrange_context", "rearrange_irrelevant"]:
    if cat in category_results:
        cr = category_results[cat]
        acc = 100 * cr["exact_match"] / cr["total"] if cr["total"] > 0 else 0
        print(f"  {cat}: {cr['exact_match']}/{cr['total']} ({acc:.1f}%)")

print(f"\n--- Error Samples (first 10) ---")
error_entries = [e for e in eval_entries if not e["correct"]]
for e in error_entries[:10]:
    inst = e["instruction"]
    print(f"\n  [{e['index']}] Category: {e['category']}")
    print(f"    Current XDL:     {inst['current_xdl']}")
    print(f"    Next XDL:        {inst['next_xdl']}")
    print(f"    Obstacle:        {inst['obstacle_info']}")
    print(f"    Candidate Grids: {inst['candidate_grids']}")
    print(f"    GT:   {e['gt']}")
    print(f"    Pred: {e['pred']}")

# =========================
# Save results
# =========================
with open(RESULT_PATH, "w", encoding="utf-8") as f:
    for entry in eval_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"\nDetailed results saved to: {RESULT_PATH}")
