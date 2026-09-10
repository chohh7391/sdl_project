import os
import json
import time
import torch
import re
from unsloth import FastLanguageModel

# =========================
# Config & Paths
# =========================
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
MODEL_PATH = "/home/home/sdl_ws/src/sdl_project/LLM/llama/model/checkpoint/tool_move/checkpoint"
DATA_PATH = "/home/home/sdl_ws/src/sdl_project/LLM/llama/script/experiments/tool_select_data.json"

# =========================
# Load Model & Tokenizer
# =========================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=512,
    load_in_4bit=True,
)
model.eval()
FastLanguageModel.for_inference(model)

# =========================
# Prompt Template
# =========================
PROMPT_TEMPLATE = """You are a tool selection system for laboratory automation.

Tool Selection Rules based on Task and Object Size:
- dh3 (3-finger): MUST be used for ALL <Stir> operations on cylindrical vessels (e.g., tube, beaker, flask). Also used to <Move> these vessels.
- ag95 (2-finger): MUST be used for ALL <Transfer> operations.
- vgc10 (suction): MUST be used whenever moving or transferring objects that are larger than the gripper's span (e.g., box, plate, tray, lid).

Given a single XDL step and a space constraint flag, output:
1) the main tool for the XDL step
2) whether a Move is needed (True or False)
3) the tool required for the Move (or None)

Rules:
- Output EXACTLY three tokens separated by commas.
- The order MUST be: main_tool, need_move, move_tool
- Allowed tool tokens: dh3, ag95, vgc10, None
- Allowed boolean tokens: True, False
- Do NOT add explanations or extra text.

XDL Step:
{xdl}

is_space_constrained:
{is_space_constrained}

Output:
"""

# =========================
# Evaluation Stats
# =========================
results = {
    "Total": {"correct": 0, "total": 0},
    "dh3": {"correct": 0, "total": 0},
    "ag95": {"correct": 0, "total": 0},
    "vgc10": {"correct": 0, "total": 0}
}
inference_times = []

# =========================
# Run Experiment
# =========================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"🚀 Starting Experiment on {len(test_data)} samples...\n")

for i, sample in enumerate(test_data):
    xdl = sample["instruction"]["xdl"]
    is_space = sample["instruction"]["is_space_constrained"]
    expected_main_tool = sample["expected_output"][0]

    prompt = PROMPT_TEMPLATE.format(xdl=xdl, is_space_constrained=is_space)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        start_time = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        end_time = time.perf_counter()

    # Time recording (skip first few for warm-up effect)
    if i > 5:
        inference_times.append(end_time - start_time)

    # Output Parsing
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = decoded[len(prompt):].strip().splitlines()[0].strip()
    parts = [p.strip() for p in generated.split(",")]

    predicted_main_tool = parts[0] if len(parts) > 0 else "INVALID"

    # Stat Update
    results["Total"]["total"] += 1
    results[expected_main_tool]["total"] += 1

    if predicted_main_tool == expected_main_tool:
        results["Total"]["correct"] += 1
        results[expected_main_tool]["correct"] += 1
    else:
        print(f"❌ Mismatch [Sample {i}]: XDL: {xdl} | Expected: {expected_main_tool} | Predicted: {predicted_main_tool}")

# =========================
# Final Report (Table 6)
# =========================
print("\n" + "="*50)
print("       📊 TOOL SELECTION EVALUATION RESULT")
print("="*50)
print(f"{'Tool':<10} | {'Correct':<10} | {'Total':<10} | {'Selected Rate (%)'}")
print("-" * 50)

for tool in ["dh3", "ag95", "vgc10"]:
    correct = results[tool]["correct"]
    total = results[tool]["total"]
    rate = (correct / total * 100) if total > 0 else 0
    print(f"{tool:<10} | {correct:<10} | {total:<10} | {rate:.2f}%")

total_rate = (results["Total"]["correct"] / results["Total"]["total"] * 100)
avg_time = sum(inference_times) / len(inference_times) if inference_times else 0

print("-" * 50)
print(f"{'OVERALL':<10} | {results['Total']['correct']:<10} | {results['Total']['total']:<10} | {total_rate:.2f}%")
print(f"\n⏱️ Average Inference Time: {avg_time:.6f} seconds")
print("="*50)