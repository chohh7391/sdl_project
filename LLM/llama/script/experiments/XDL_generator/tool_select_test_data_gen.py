import json
import random
import os

# =========================
# 설정 및 경로 정의
# =========================
# 사용자가 지정한 절대 경로
OUTPUT_FILE = "/home/home/sdl_ws/src/sdl_project/LLM/llama/script/experiments/tool_select_data.json"
NUM_SAMPLES = 100

# 객체 및 장소 분류
SMALL_VESSELS = ["beaker", "flask", "tube"]
LARGE_OBJECTS = ["bottle", "box"]
PLACES = ["tray", "plate"]
TARGET_VESSELS = ["beaker", "tube", "flask"]

def generate_test_data():
    # 저장 경로의 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    samples = []
    
    # 시나리오별 할당량 (총 100개) [cite: 341]
    # 복합 명령 구성을 위해 25개씩 균등 배분
    scenarios = [
        ("Move", SMALL_VESSELS, 25),
        ("Move", LARGE_OBJECTS, 25),
        ("Transfer", SMALL_VESSELS, 25),
        ("Stir", SMALL_VESSELS, 25)
    ]

    for task, object_list, quota in scenarios:
        for _ in range(quota):
            obj_type = random.choice(object_list)
            is_space_constrained = random.choice([True, False])
            
            # 1. Move 작업 생성
            if task == "Move":
                obj_id = f"{obj_type}_A"
                place = random.choice(PLACES)
                xdl = f'<Move object="{obj_id}" place="{place}" />'
                main_tool = "dh3" if obj_type in SMALL_VESSELS else "vgc10"
                
            # 2. Transfer 작업 생성 (규칙: ag95 사용)
            elif task == "Transfer":
                from_vessel_type = obj_type
                to_vessel_type = random.choice(TARGET_VESSELS)
                
                # 명칭 중복 방지 로직 (사용자 요청 반영)
                if from_vessel_type == to_vessel_type:
                    from_id = f"{from_vessel_type}_A"
                    to_id = f"{to_vessel_type}_B"
                else:
                    from_id = f"{from_vessel_type}_A"
                    to_id = f"{to_vessel_type}_A"
                
                xdl = f'<Transfer from_vessel="{from_id}" to_vessel="{to_id}" volume="10mL" />'
                main_tool = "ag95"
                
            # 3. Stir 작업 생성 (규칙: dh3 사용)
            elif task == "Stir":
                obj_id = f"{obj_type}_A"
                xdl = f'<Stir vessel="{obj_id}" time="30s" />'
                main_tool = "dh3"

            # 4. 정답(Ground Truth) 및 제약 조건 설정
            need_move = "True" if is_space_constrained else "False"
            if is_space_constrained:
                # 장애물 제거 도구는 메인 객체 크기에 따름
                move_tool = "dh3" if obj_type in SMALL_VESSELS else "vgc10"
            else:
                move_tool = "None"

            sample = {
                "instruction": {
                    "xdl": xdl,
                    "is_space_constrained": str(is_space_constrained)
                },
                "expected_output": [main_tool, need_move, move_tool]
            }
            samples.append(sample)

    # 무작위 섞기
    random.shuffle(samples)
    
    # JSON 파일로 저장 (표준 리스트 형태)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)

    print(f"✅ 테스트 데이터셋 생성 완료: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_test_data()