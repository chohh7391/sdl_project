import os
import json
import random

# ==========================================
# 0. 저장 경로 설정 (사용자 지정 경로)
# ==========================================
SAVE_PATH = "/home/home/sdl_ws/src/sdl_project/LLM/llama/script/experiments/test_data.json"

# ==========================================
# 1. 학습 데이터와 100% 동일한 사전 정의
# ==========================================
VESSELS = ["beaker_A", "beaker_B", "flask_A", "flask_B"]
OBJECTS = ["box_A", "bottle_A"]
PLATES = ["plate_A", "plate_B"]
REAGENTS = ["water", "ethanol"]
VOLUMES = ["10 mL", "25 mL", "50 mL"]
STIR_TIMES = ["30 s", "1 min", "2 min"]
TEMPS = ["0 C", "25 C", "60 C"]

def nl_add(vessel, reagent, volume): return random.choice([f"Add {volume} of {reagent} to {vessel}.", f"Pour {volume} of {reagent} into {vessel}.", f"Introduce {volume} of {reagent} into the {vessel}.", f"Place {volume} of {reagent} into {vessel}."])
def nl_stir(vessel, time): return random.choice([f"Stir {vessel} for {time}.", f"Mix the contents of {vessel} for {time}.", f"Agitate {vessel} for {time}.", f"Stir the solution in {vessel} for {time}."])
def nl_heatchill(vessel, temp): return random.choice([f"Heat {vessel} to {temp}.", f"Set the temperature of {vessel} to {temp}.", f"Adjust {vessel} to {temp}.", f"Bring {vessel} to {temp}."])
def nl_transfer(from_v, to_v, volume): return random.choice([f"Transfer {volume} from {from_v} to {to_v}.", f"Pour {volume} from {from_v} into {to_v}.", f"Send {volume} from {from_v} to {to_v}."])
def nl_clean(vessel): return random.choice([f"Clean {vessel}.", f"Wash {vessel}.", f"Rinse and clean {vessel}.", f"Perform cleaning on {vessel}."])
def nl_move(obj, place): return random.choice([f"Move {obj} to {place}.", f"Place {obj} on {place}.", f"Relocate {obj} onto {place}.", f"Put {obj} on top of {place}."])

# ==========================================
# 2. 데이터 생성 함수들
# ==========================================
def generate_paper_test_instructions(n_total=100):
    instructions = []
    for _ in range(n_total):
        steps = []
        vessel = random.choice(VESSELS)
        num_steps = random.choices([1, 2, 3], weights=[30, 40, 30])[0]
        
        reagent = random.choice(REAGENTS)
        vol = random.choice(VOLUMES)
        steps.append(nl_add(vessel, reagent, vol))
        
        for _ in range(num_steps - 1):
            action = random.choice(["stir", "heatchill", "transfer", "clean", "move"])
            if action == "stir":
                steps.append(nl_stir(vessel, random.choice(STIR_TIMES)))
            elif action == "heatchill":
                steps.append(nl_heatchill(vessel, random.choice(TEMPS)))
            elif action == "transfer":
                target = random.choice([v for v in VESSELS if v != vessel])
                steps.append(nl_transfer(vessel, target, random.choice(VOLUMES)))
                vessel = target
            elif action == "clean":
                steps.append(nl_clean(vessel))
            elif action == "move":
                steps.append(nl_move(random.choice(OBJECTS), random.choice(PLATES)))
                
        instructions.append(" ".join(steps))
    return instructions

def generate_validator_testset():
    dataset = []
    # [Valid Cases]
    dataset.append({"is_actually_valid": True, "xml": "<procedure><Add vessel='beaker_A' reagent='water' volume='10 mL' /></procedure>"})
    dataset.append({"is_actually_valid": True, "xml": "<procedure><Add vessel='flask_A' reagent='ethanol' volume='50 mL' /><Stir vessel='flask_A' time='1 min' /></procedure>"})
    dataset.append({"is_actually_valid": True, "xml": "<procedure><Add vessel='beaker_B' reagent='water' volume='25 mL' /><Transfer from_vessel='beaker_B' to_vessel='flask_B' volume='10 mL' /></procedure>"})
    dataset.append({"is_actually_valid": True, "xml": "<procedure><Move object='box_A' place='plate_A' /></procedure>"})

    # [Invalid Cases]
    dataset.append({"is_actually_valid": False, "xml": "<procedure><Stir vessel='beaker_A' time='30 s' /></procedure>"}) # 빈 그릇 젓기
    dataset.append({"is_actually_valid": False, "xml": "<procedure><Add vessel='beaker_A' reagent='water' volume='10 mL' /><Transfer from_vessel='beaker_A' to_vessel='beaker_A' volume='10 mL' /></procedure>"}) # 같은 그릇 이송
    dataset.append({"is_actually_valid": False, "xml": "<procedure><Add vessel='magic_beaker' reagent='water' volume='10 mL' /></procedure>"}) # 없는 이름
    dataset.append({"is_actually_valid": False, "xml": "<procedure><Add vessel='beaker_A' reagent='acid' volume='10 mL' /></procedure>"}) # 없는 시약
    dataset.append({"is_actually_valid": False, "xml": "<procedure><Add vessel='beaker_A' volume='10 mL' /></procedure>"}) # 필수 속성 누락
    dataset.append({"is_actually_valid": False, "xml": "<procedure><Add vessel='beaker_A' reagent='water' volume='10 mL' /><HeatChill vessel='beaker_A' temp='60 C' /></procedure>"}) # active 누락
    dataset.append({"is_actually_valid": False, "xml": "<procedure><Add vessel='beaker_A' reagent='water' volume='10 mL' > </procedure>"}) # 닫히지 않은 태그
    dataset.append({"is_actually_valid": False, "xml": "<procedure><Add vessel='beaker_A' reagent='water' volume='10 mL' /></wrong_root>"}) # 잘못된 루트
    return dataset

# ==========================================
# 3. 데이터 생성 및 JSON 파일 저장
# ==========================================
if __name__ == "__main__":
    print("🔄 데이터를 생성하는 중...")
    
    test_instructions = generate_paper_test_instructions(n_total=100)
    labeled_xdl_testset = generate_validator_testset()
    
    # 딕셔너리 형태로 묶기
    output_data = {
        "test_instructions": test_instructions,
        "labeled_xdl_testset": labeled_xdl_testset
    }
    
    # 저장할 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    # JSON 파일로 저장 (한글 깨짐 방지 및 보기 좋게 들여쓰기)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 테스트 데이터가 성공적으로 저장되었습니다!")
    print(f"📁 저장 위치: {SAVE_PATH}")
    print(f"📊 구성: 자연어 명령어 {len(test_instructions)}개, Validator 테스트용 XDL {len(labeled_xdl_testset)}개")