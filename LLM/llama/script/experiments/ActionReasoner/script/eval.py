import os
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
import torch
from unsloth import FastLanguageModel

# =========================
# Config
# =========================
MODEL_DIR = "/home/home/sdl_ws/src/sdl_project/LLM/llama/script/experiments/ActionReasoner/model/checkpoint"
MAX_SEQ_LEN = 512

PROMPT_TEMPLATE = """You are a tool and rearrangement planner for laboratory automation.

Tool Rules:
- dh3 (3-finger): Move, Stir for cylindrical vessels (beaker, flask, tube)
- ag95 (2-finger): All Transfer operations
- vgc10 (suction): Objects exceeding gripper span (box, plate, bottle)

Given the current and next XDL steps, the obstacle status, and candidate grids,
output exactly four tokens separated by commas:
main_tool, need_rearrange, aux_tool, target_grid

Allowed values:
- main_tool: dh3, ag95, vgc10
- aux_tool: dh3, ag95, vgc10, None
- need_rearrange: True, False
- target_grid: grid ID (e.g., G5) or None

Current XDL:
{current_xdl}

Next XDL:
{next_xdl}

Obstacle:
{obstacle_info}

Candidate Grids:
{candidate_grids}

Output:
"""

# =============================================================
# 여기에 테스트할 입력을 직접 수정하세요
# =============================================================
TEST_INPUT = {
    "current_xdl": '<Stir vessel="beaker" time="15 s" />',
    "next_xdl": "None",
    "obstacle_info": "flask at G10 (blocking)",
    "candidate_grids": "workspace edge: [G7, G8, G12]\nopen area: [G3, G5, G6, G11]",
}

# 정답이 있으면 비교용으로 입력 (없으면 None)
EXPECTED_OUTPUT = "dh3,True,dh3,G7"


# =========================
# Load model
# =========================
print("=" * 50)
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
# Inference
# =========================
prompt = PROMPT_TEMPLATE.format(
    current_xdl=TEST_INPUT["current_xdl"],
    next_xdl=TEST_INPUT["next_xdl"],
    obstacle_info=TEST_INPUT["obstacle_info"],
    candidate_grids=TEST_INPUT["candidate_grids"],
)

print("=" * 50)
print("[PROMPT]")
print(prompt)
print("=" * 50)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)
pred = generated.strip().split("\n")[0].strip()

# =========================
# Result
# =========================
print("[PREDICTION]")
print(f"  Raw output : {generated.strip()}")
print(f"  Parsed     : {pred}")
print()

if EXPECTED_OUTPUT:
    gt_tokens = EXPECTED_OUTPUT.split(",")
    pred_tokens = pred.split(",")

    print(f"[EXPECTED]")
    print(f"  {EXPECTED_OUTPUT}")
    print()

    if len(pred_tokens) == 4:
        labels = ["main_tool", "need_rearrange", "aux_tool", "target_grid"]
        print("[TOKEN COMPARISON]")
        for label, gt_t, pred_t in zip(labels, gt_tokens, pred_tokens):
            gt_t, pred_t = gt_t.strip(), pred_t.strip()
            match = "OK" if gt_t == pred_t else "MISMATCH"
            print(f"  {label:20s}: GT={gt_t:6s}  Pred={pred_t:6s}  [{match}]")
    else:
        print(f"  FORMAT ERROR: expected 4 tokens, got {len(pred_tokens)}")

print("=" * 50)