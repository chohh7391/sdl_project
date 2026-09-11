import os
import sys
import re
import time
import json
from datetime import datetime
from collections import defaultdict

os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

import unsloth
from unsloth import FastLanguageModel
import torch

# 🔹 같은 폴더에 있는 Validator import
from validator import ProcedureValidator, ProcedureValidationError

# ==========================================
# 0. 설정 및 모델 로드
# ==========================================
MODEL_PATH = "/home/home/sdl_ws/src/sdl_project/LLM/llama/model/checkpoint/xdl_generator/checkpoint"
# test_data.json lives beside this script; test_data_gen.py's SAVE_PATH points
# one directory up, where no such file exists, so this path was broken.
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data.json")
# Every generation is also written out, so field-level scoring (score_xdl.py)
# can run against the recovered labels without re-running inference.
DUMP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "xdl_generations.json")

print("⏳ Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=1024,
    load_in_4bit=False,
)
model.eval()
FastLanguageModel.for_inference(model)
print("✅ Model loaded.")

# --- GPU 웜업 ---
print("🔥 GPU Warming up...")
dummy_inputs = tokenizer("Warm up prompt", return_tensors="pt").to("cuda")
for _ in range(2):
    with torch.no_grad():
        _ = model.generate(**dummy_inputs, max_new_tokens=10)
print("✅ Warm-up complete.")

# ==========================================
# 1. 프롬프트 및 LLM 추론 함수 (Slicing 로직 적용)
# ==========================================
BASE_PROMPT = """You are a laboratory automation system.
Convert the following instruction into a procedure description.
Output ONLY valid XML inside <procedure> tags.

Strict Rules:
- Tags are CASE-SENSITIVE: MUST use <Add>, <Stir>, <HeatChill>, <Transfer>, <CleanVessel>, <Move>.
- DO NOT use lowercase tags like <add> or <clean_vessel>.
- <Transfer> MUST use attributes: from_vessel, to_vessel, volume.

Example:
Instruction: Add 50 mL of water to beaker_A. Transfer 10 mL from beaker_A to flask_A. Clean flask_A.
Procedure:
<procedure>
  <Add vessel="beaker_A" reagent="water" volume="50 mL" />
  <Transfer from_vessel="beaker_A" to_vessel="flask_A" volume="10 mL" />
  <CleanVessel vessel="flask_A" />
</procedure>

Instruction:
{instruction}

Procedure:
<procedure>

"""

def generate_xdl(prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    input_length = inputs["input_ids"].shape[1]
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200, 
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.convert_tokens_to_ids("</procedure>"),
        )
        
    torch.cuda.synchronize()
    inf_time = time.perf_counter() - start_time

    # 1. 프롬프트(예시 포함)를 제외한 순수 생성 토큰만 디코딩
    generated_tokens = outputs[0][input_length:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # 2. </procedure> 이후의 잡담(Junk) 제거 및 XML 추출
    # 프롬프트 끝에서 미리 열어준 <procedure>와 합칩니다.
    full_text = "<procedure>\n" + decoded
    
    match = re.search(r"(<procedure>.*?</procedure>)", full_text, re.DOTALL)
    if match:
        xml_content = match.group(1)
    else:
        # 태그가 안 닫혔을 경우 강제 보정
        xml_content = full_text if full_text.endswith("</procedure>") else full_text + "\n</procedure>"

    return xml_content, inf_time

# ==========================================
# 2. 에러 분류기 및 실험 루프
# ==========================================
def categorize_error(error_msg):
    error_msg = str(error_msg)
    if "empty vessel" in error_msg or "Cannot" in error_msg:
        return "Physically infeasible sequence"
    if any(k in error_msg for k in ["Invalid vessel", "Invalid object", "Invalid reagent", "Invalid place"]):
        return "Invalid operator reference"
    if any(k in error_msg for k in ["must have", "Invalid tag", "must differ", "Invalid volume"]):
        return "Missing/Invalid required field"
    if any(k in error_msg for k in ["Invalid XML format", "Root must be", "empty"]):
        return "XML Syntax Error"
    return "Other Errors"

def run_generation_experiment(test_instructions):
    print("\n" + "="*60)
    print("▶️ [실험 1&2] Language Command to XDL 시작")
    print("="*60)
    
    valid_count = 0
    invalid_count = 0
    total_inf_time = 0.0
    error_stats = defaultdict(lambda: {"count": 0, "example": ""})
    generations = {}
    inf_times = []

    for i, instruction in enumerate(test_instructions, 1):
        prompt = BASE_PROMPT.format(instruction=instruction)
        generated_xml, inf_time = generate_xdl(prompt)
        total_inf_time += inf_time
        generations[str(i - 1)] = generated_xml
        inf_times.append(inf_time)
        
        validator = ProcedureValidator()
        try:
            validator.validate(generated_xml)
            valid_count += 1
            print(f"[{i}/{len(test_instructions)}] ✅ Valid")
            print(f"📌 [Instruction]: {instruction}")
            print(f"🤖 [Generated XML]:\n{generated_xml}")
            print('-------------------------------------------------------')
        except ProcedureValidationError as e:
            invalid_count += 1
            category = categorize_error(e)
            error_stats[category]["count"] += 1
            if not error_stats[category]["example"]: error_stats[category]["example"] = str(e)
            print(f"[{i}/{len(test_instructions)}] ❌ Invalid ({category})")
            print("-" * 40)
            print(f"📌 [Instruction]: {instruction}")
            print(f"🤖 [Generated XML]:\n{generated_xml}")
            print(f"⚠️ [Error Message]: {e}")
            print('-------------------------------------------------------')

    avg_time = total_inf_time / len(test_instructions) if test_instructions else 0
    with open(DUMP_PATH, "w") as fh:
        json.dump({"model": MODEL_PATH, "data": DATA_PATH,
                   "generations": generations, "inference_times_s": inf_times},
                  fh, indent=1)
    print("wrote generations to %s" % DUMP_PATH)
    return len(test_instructions), valid_count, invalid_count, error_stats, avg_time

def run_validator_accuracy_experiment(labeled_xdl_data):
    print("\n" + "="*60)
    print("▶️ [실험 3] Validator Accuracy 시작")
    print("="*60)
    TP, FP, TN, FN = 0, 0, 0, 0
    for data in labeled_xdl_data:
        validator = ProcedureValidator()
        try:
            validator.validate(data["xml"])
            v_pass = True
        except ProcedureValidationError:
            v_pass = False
            
        if data["is_actually_valid"] and v_pass: TP += 1
        elif data["is_actually_valid"] and not v_pass: FN += 1
        elif not data["is_actually_valid"] and not v_pass: TN += 1
        else: FP += 1
    return TP, FP, TN, FN

# ==========================================
# 3. 결과 출력
# ==========================================
def print_paper_tables(N, v_c, iv_c, err_s, avg_t, TP, FP, TN, FN):
    print("\n\n" + "#"*80)
    print("### 논문 결과 출력 ###")
    print("#"*80)
    
    print(f"\n[Table 4.2.1]\nAttempts: {N} | Success: {v_c} | Fail: {iv_c} | Rate: {(v_c/N)*100:.1f}%")
    print(f"Avg Inference Time: {avg_t:.4f}s")
    
    print("\n[Table 4.2.2-1] Error Analysis")
    for err, data in err_s.items():
        print(f"- {err:<30}: {data['count']} cases (e.g., {data['example'][:50]})")
        
    print("\n[Table 4.2.2-2] Confusion Matrix")
    print(f"TP: {TP} | FN: {FN}\nFP: {FP} | TN: {TN}")
    acc = ((TP+TN)/(TP+TN+FP+FN))*100 if (TP+TN+FP+FN)>0 else 0
    print(f"Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    N, v_c, iv_c, err_s, avg_t = run_generation_experiment(data["test_instructions"])
    TP, FP, TN, FN = run_validator_accuracy_experiment(data["labeled_xdl_testset"])
    print_paper_tables(N, v_c, iv_c, err_s, avg_t, TP, FP, TN, FN)