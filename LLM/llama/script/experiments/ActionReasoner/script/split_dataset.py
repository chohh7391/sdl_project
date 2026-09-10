"""
학습 데이터에서 테스트셋을 분리하여 별도 jsonl로 저장
"""
import json
import random

random.seed(42)

# =========================
# Config
# =========================
DATA_PATH = "/home/home/sdl_ws/src/sdl_project/LLM/llama/script/experiments/ActionReasoner/dataset/action_reasoner_dataset.jsonl"  # 원본 데이터 경로
TRAIN_PATH = "/home/home/sdl_ws/src/sdl_project/LLM/llama/script/experiments/ActionReasoner/dataset/action_reasoner_train.jsonl"
TEST_PATH = "/home/home/sdl_ws/src/sdl_project/LLM/llama/script/experiments/ActionReasoner/dataset/action_reasoner_test.jsonl"
TEST_RATIO = 0.2

# =========================
# Load
# =========================
all_data = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        all_data.append(json.loads(line))

random.shuffle(all_data)

# =========================
# Split
# =========================
split_idx = int(len(all_data) * (1 - TEST_RATIO))
train_data = all_data[:split_idx]
test_data = all_data[split_idx:]

# =========================
# Save
# =========================
with open(TRAIN_PATH, "w", encoding="utf-8") as f:
    for entry in train_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

with open(TEST_PATH, "w", encoding="utf-8") as f:
    for entry in test_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Total: {len(all_data)}")
print(f"Train: {len(train_data)} -> {TRAIN_PATH}")
print(f"Test:  {len(test_data)} -> {TEST_PATH}")
