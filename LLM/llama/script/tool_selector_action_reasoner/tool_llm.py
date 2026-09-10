import os
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
import torch
import time
from unsloth import FastLanguageModel


class ToolLLM:
    def __init__(self):
        print("⏳ [ToolLLM] 모델을 로딩 중입니다. 잠시만 기다려주세요... (약 10~20초 소요)")
        self.llama_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "llama")
        self.model_path = os.path.join(self.llama_path, "model", "checkpoint", "tool_move", "checkpoint")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=512,
            load_in_4bit=True,
        )
        self.model.eval()
        FastLanguageModel.for_inference(self.model)

        self.prompt_template = """You are a tool selection system for laboratory automation.
Tool Selection Rules based on Task and Object Size:
- dh3 (3-finger): MUST be used for ALL <Stir> operations on cylindrical vessels (e.g., tube, beaker, flask). Also used to <Move> these vessels.
- ag95 (2-finger): MUST be used for ALL <Transfer> operations.
- vgc10 (suction): MUST be used whenever moving objects that are larger than the gripper's span (e.g., box, plate, tray, lid).

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

        self.valid_tools = {"dh3", "ag95", "vgc10", "None"}
        self.valid_booleans = {"True", "False"}

        # 초기 웜업 (실제 추론 시 속도 향상을 위해 Dummy 데이터로 2회 실행)
        print("🔥 [ToolLLM] 모델 웜업 중...")
        dummy_prompt = self.prompt_template.format(xdl='<Stir vessel="flask" />', is_space_constrained=False)
        inputs = self.tokenizer(dummy_prompt, return_tensors="pt").to("cuda")
        for _ in range(2):
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=15, do_sample=False, temperature=0.0)
        torch.cuda.synchronize()
        print("✅ [ToolLLM] 모델 로딩 및 웜업 완료!")

    def predict(self, xdl_step: str, is_space_constrained: bool):
        """
        XDL 스텝과 공간 제약 여부를 입력받아 필요한 툴과 작업 정보를 반환합니다.
        """

        print('xdl_step:', xdl_step)
        prompt = self.prompt_template.format(
            xdl=xdl_step,
            is_space_constrained=is_space_constrained
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.05,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # 디코딩 및 프롬프트 제거
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = decoded[len(prompt):].strip()
        generated = generated.splitlines()[0].strip()  # 첫 줄만 사용

        # 쉼표 기준으로 파싱
        parts = [p.strip() for p in generated.split(",")]

        # 유효성 검사
        if len(parts) != 3:
            print(f"⚠️ [ToolLLM] Unexpected output format: {repr(generated)}")
            return "INVALID", "INVALID", "INVALID"
            
        main_tool, need_move, move_tool = parts

        if main_tool not in self.valid_tools: main_tool = "INVALID"
        if need_move not in self.valid_booleans: need_move = "INVALID"
        if move_tool not in self.valid_tools: move_tool = "INVALID"

        print(f"🧠 [ToolLLM] Inference Time: {end_time - start_time:.4f}s | Result: {main_tool}, {need_move}, {move_tool}")
        return main_tool, need_move, move_tool