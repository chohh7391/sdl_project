import os
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
import torch
import time
from unsloth import FastLanguageModel
import copy

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

class ActionReasoner:
    def __init__(self):
        print("⏳ [ToolLLM] 모델을 로딩 중입니다. 잠시만 기다려주세요... (약 10~20초 소요)")
        self.llama_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "llama")
        self.model_path = os.path.join(self.llama_path, "model", "checkpoint", "action_reasoner")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=512,
            load_in_4bit=True,
            dtype=None,
        )
        self.model.eval()
        FastLanguageModel.for_inference(self.model)

        self.prompt_template = copy.deepcopy(PROMPT_TEMPLATE)
        self.valid_tools = {"dh3", "ag95", "vgc10", "None"}
        self.valid_booleans = {"True", "False"}
        self.valid_grids = {"G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12", "None"}

        # 초기 웜업 (실제 추론 시 속도 향상을 위해 Dummy 데이터로 2회 실행)
        print("🔥 [ToolLLM] 모델 웜업 중...")
        dummy_prompt = self.prompt_template.format(
            current_xdl='<Add vessel=\"beaker_B\" reagent=\"water\" volume=\"20 mL\" />',
            next_xdl='None',
            obstacle_info="flask_B at G12 (blocking)",
            candidate_grids="plate zone: [G1]\nworkspace edge: [G8, G12]\nopen area: [G2, G3, G4, G5, G6, G9, G10, G11]",
        )
        inputs = self.tokenizer(dummy_prompt, return_tensors="pt").to("cuda")
        for _ in range(2):
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        torch.cuda.synchronize()
        print("✅ [ToolLLM] 모델 로딩 및 웜업 완료!")

    def predict(self, xdl_step: dict):
        """
        XDL 스텝과 공간 제약 여부를 입력받아 필요한 툴과 작업 정보를 반환합니다.
        """

        inst = xdl_step["instruction"]
        prompt = self.prompt_template.format(
            current_xdl=inst["current_xdl"],
            next_xdl=inst["next_xdl"],
            obstacle_info=inst["obstacle_info"],
            candidate_grids=inst["candidate_grids"],
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                temperature=0.0,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        raw_output = generated.strip().split("\n")[0].strip()
        tokens = [t.strip() for t in raw_output.split(",")]

        # 유효성 검사
        if len(tokens) != 4:
            print(f"⚠️ [ToolLLM] Unexpected output format: {repr(raw_output)}")
            return "INVALID", "INVALID", "INVALID", "INVALID"
            
        main_tool, is_rearrange, rearrange_tool, rearrange_grid = tokens

        if main_tool not in self.valid_tools: main_tool = "INVALID"
        if is_rearrange not in self.valid_booleans: need_move = "INVALID"
        if rearrange_tool not in self.valid_tools: move_tool = "INVALID"
        if rearrange_grid not in self.valid_grids: rearrange_grid = "INVALID"

        print(f"🧠 [ToolLLM] Inference Time: {end_time - start_time:.4f}s | Result | main tool: {main_tool}, is_rearrange: {is_rearrange}, rearrange_tool: {rearrange_tool}, rearrange_grid: {rearrange_grid}")
        return main_tool, is_rearrange, rearrange_tool, rearrange_grid
    
    def parse_candidate_categories(self, candidate_grids_str):
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
    
    def is_same_grid_category(self, gt_grid, pred_grid, candidate_grids_str):
        """GT와 Pred 그리드가 동일 카테고리에 속하는지 확인"""
        if gt_grid == pred_grid:
            return True
        if gt_grid == "None" or pred_grid == "None":
            return gt_grid == pred_grid
        grid_to_cat = self.parse_candidate_categories(candidate_grids_str)
        gt_cat = grid_to_cat.get(gt_grid)
        pred_cat = grid_to_cat.get(pred_grid)
        if gt_cat and pred_cat and gt_cat == pred_cat:
            return True
        return False