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

LLAMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "llama")
MODEL_PATH = os.path.join(LLAMA_PATH, "model", "checkpoint", "tool_llm", "checkpoint")

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

Given a single XDL step, output the name of the tool required to execute it.

Rules:
- Output ONLY one of the following tokens: dh3, ag95, vgc10, None
- Do NOT add explanations or extra text.
- If the XDL step is invalid or unsupported, output: None

XDL Step:
{instruction}

Tool:
"""

# =========================
# Test XDL input
# =========================

xdl_step = '<Stir vessel="flask_A" time="1 min" />'
# xdl_step = '<Move object="box_A" place="plate_B" />'
# xdl_step = '<Transfer from_vessel="box_B" to_vessel="beaker_B" volume="10 mL" />'
# xdl_step = '<Transfer from_vessel="beaker_A" to_vessel="flask_B" volume="25 mL" />'
# xdl_step = '<Transfer from_vessel="box_A" to_vessel="flask_B" volume="10 mL" />'

prompt = PROMPT_TEMPLATE.format(instruction=xdl_step)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# =========================
# Inference (with warm-up)
# =========================

# warm-up
for _ in range(2):
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=0.0,
        )

torch.cuda.synchronize()
start_time = time.perf_counter()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
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
# Tool Extraction
# =========================

# 프롬프트 뒤에 생성된 부분만 잘라내기
generated = decoded[len(prompt):].strip()

# 첫 토큰만 사용
tool = re.split(r"\s+", generated)[0]

# 허용 토큰만 필터링
VALID_TOOLS = {"dh3", "ag95", "vgc10", "None"}

if tool not in VALID_TOOLS:
    print("⚠️ Unexpected output token:", repr(tool))
    print("===== RAW MODEL OUTPUT =====")
    print(decoded)
    tool = "INVALID"

# =========================
# Print Result
# =========================

print("\n===== Tool LLM Output =====")
print("XDL Step:", xdl_step)
print("Predicted Tool:", tool)
