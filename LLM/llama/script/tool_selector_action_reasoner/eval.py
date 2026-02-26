import os
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

import unsloth
from unsloth import FastLanguageModel
import torch
import time
import re

# =========================
# Paths
# =========================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "model", "checkpoint", "tool_move", "checkpoint")

# =========================
# Load model
# =========================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=256,
    load_in_4bit=True,   # eval에서는 굳이 4bit 안 써도 됨
)

model.eval()
FastLanguageModel.for_inference(model)

# =========================
# Prompt (training과 동일해야 함)
# =========================

PROMPT_TEMPLATE = """You are a tool selection system for laboratory automation.

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
# Test XDL input
# =========================

# xdl_step = '<Stir vessel="flask_A" time="1 min" />'
# xdl_step = '<Move object="box_A" place="plate_B" />'
xdl_step = '<Transfer from_vessel="beaker_A" to_vessel="flask_B" volume="25 mL" />'
# xdl_step = '<Transfer from_vessel="beaker" to_vessel="flask" volume="25 mL" />'

is_space_constrained = False   # 테스트용 플래그

prompt = PROMPT_TEMPLATE.format(
    xdl=xdl_step,
    is_space_constrained=is_space_constrained
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# =========================
# Inference (with warm-up)
# =========================

# warm-up
for _ in range(2):
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            temperature=0.0,
        )

torch.cuda.synchronize()
start_time = time.perf_counter()

with torch.no_grad():
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

print(f"\n⏱️ Pure inference time: {end_time - start_time:.6f} seconds")

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# Output Extraction
# =========================

# 프롬프트 뒤에 생성된 부분만 잘라내기
generated = decoded[len(prompt):].strip()

# 혹시 줄바꿈/공백 섞여 나오면 첫 줄만 사용
generated = generated.splitlines()[0].strip()

# 정확히 3토큰 파싱
parts = [p.strip() for p in generated.split(",")]

VALID_TOOLS = {"dh3", "ag95", "vgc10", "None"}
VALID_BOOLEANS = {"True", "False"}

if len(parts) != 3:
    print("⚠️ Unexpected output format:", repr(generated))
    print("===== RAW MODEL OUTPUT =====")
    print(decoded)
    main_tool = "INVALID"
    need_move = "INVALID"
    move_tool = "INVALID"
else:
    main_tool, need_move, move_tool = parts

    if main_tool not in VALID_TOOLS:
        print("⚠️ Invalid main_tool:", repr(main_tool))
    if need_move not in VALID_BOOLEANS:
        print("⚠️ Invalid need_move:", repr(need_move))
    if move_tool not in VALID_TOOLS:
        print("⚠️ Invalid move_tool:", repr(move_tool))

# =========================
# Print Result
# =========================

print("\n===== Tool LLM Output =====")
print("XDL Step:", xdl_step)
print("is_space_constrained:", is_space_constrained)
print("Predicted main_tool :", main_tool)
print("Predicted need_move :", need_move)
print("Predicted move_tool :", move_tool)
