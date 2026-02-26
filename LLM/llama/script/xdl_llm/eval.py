import os
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

import unsloth
from unsloth import FastLanguageModel
import torch
from datetime import datetime
import re
import time

# 🔹 Validator import
from validator import ProcedureValidator, ProcedureValidationError

# =========================
# Paths
# =========================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "model", "checkpoint", "xdl_llm", "checkpoint")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "llama", "result", "xdl_llm")
os.makedirs(RESULT_DIR, exist_ok=True)

# =========================
# Load model
# =========================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=1024,
    load_in_4bit=False,
)

model.eval()
FastLanguageModel.for_inference(model)

print(tokenizer.tokenize("</procedure>"))

# for name, _ in model.named_modules():
#     if "lora" in name.lower():
#         print(name)

# model.print_trainable_parameters()

# =========================
# Prompt
# =========================

prompt = """You are a laboratory automation system.
Convert the following instruction into a procedure description.
All XML tags must start with a capital letter (e.g., Add, Stir, Transfer, CleanVessel).
Output ONLY valid XML.
Do NOT add explanations or extra text.

Instruction:
Pour 10 mL of water into beaker_B. Mix the contents of beaker_B for 2 min. Pour 50 mL from beaker_B into flask_A. Clean flask_A.

Procedure:
<procedure>
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# =========================
# Inference
# =========================
# warm-up
for _ in range(2):
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.0,
        )


torch.cuda.synchronize()
start_time = time.perf_counter()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.convert_tokens_to_ids("</procedure>"),
    )
torch.cuda.synchronize()
end_time = time.perf_counter()

print(f"\n⏱️ Pure inference time: {end_time - start_time:.4f} seconds")

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# XML Extraction
# =========================

match = re.search(r"<procedure>.*?</procedure>", decoded, re.DOTALL)
if not match:
    print("===== RAW MODEL OUTPUT =====")
    print(decoded)
    raise ValueError("⚠️ Valid <procedure> XML block not found.")

xml_content = match.group(0)

# =========================
# Validation
# =========================

validator = ProcedureValidator()

try:
    validator.validate(xml_content)
    print("✅ Procedure validation passed.")
except ProcedureValidationError as e:
    print("❌ Validation failed:", e)
    print("===== INVALID XML =====")
    print(xml_content)
    raise

# =========================
# Save XML
# =========================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_path = os.path.join(RESULT_DIR, f"procedure_{timestamp}.xml")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(xml_content)

print("\n===== Generated XML =====")
print(xml_content)
print(f"\n✅ XML saved to: {file_path}")