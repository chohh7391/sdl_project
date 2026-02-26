import os
import json
import random
from collections import defaultdict

# -----------------------------
# Output path
# -----------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "tool_jsonl")
OUTPUT_FILE = "xdl_2_tool_move.jsonl"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Entity definitions
# -----------------------------

BEAKERS = ["beaker_A", "beaker_B"]
FLASKS = ["flask_A", "flask_B"]
BOXES = ["box_A", "box_B"]

PLATES = ["plate_A", "plate_B"]

VOLUMES = ["10 mL", "25 mL", "50 mL"]
STIR_TIMES = ["30 s", "1 min", "2 min"]

ALL_OBJECTS = BEAKERS + FLASKS + BOXES

# -----------------------------
# Tool decision table
# -----------------------------

TOOL_TABLE = {
    "Move": {
        "beaker": "dh3",
        "flask": "dh3",
        "box": "vgc10",
    },
    "Stir": {
        "beaker": "dh3",
        "flask": "dh3",
    },
    "Transfer": {
        "beaker": "ag95",
        "flask": "ag95",
    },
}

FAILURE_TOKEN = "None"

# -----------------------------
# Utility
# -----------------------------

def object_type_from_name(name: str) -> str:
    if name.startswith("beaker"):
        return "beaker"
    if name.startswith("flask"):
        return "flask"
    if name.startswith("box"):
        return "box"
    raise ValueError(f"Unknown object type: {name}")

def extract_main_object_from_xdl(xdl: str) -> str:
    if xdl.startswith("<Stir"):
        key = 'vessel="'
    elif xdl.startswith("<Transfer"):
        key = 'from_vessel="'
    elif xdl.startswith("<Move"):
        key = 'object="'
    else:
        raise ValueError(f"Unknown XDL: {xdl}")

    start = xdl.index(key) + len(key)
    end = xdl.index('"', start)
    return xdl[start:end]

# -----------------------------
# Generate all unique XDL steps
# -----------------------------

all_samples = []

# 1) Move
for obj in ALL_OBJECTS:
    obj_type = object_type_from_name(obj)
    main_tool = TOOL_TABLE["Move"].get(obj_type, FAILURE_TOKEN)

    for place in PLATES:
        xdl = f'<Move object="{obj}" place="{place}" />'

        # 공간 제약 여부 (랜덤)
        is_space_constrained = random.choice([True, False])

        # Move 자체는 추가 Move 불필요
        need_move = False
        move_tool = "None"

        all_samples.append({
            "instruction": {
                "xdl": xdl,
                "is_space_constrained": is_space_constrained,
            },
            "output": [
                main_tool,
                "False",
                "None",
            ],
        })

# 2) Stir
for vessel in ALL_OBJECTS:
    obj_type = object_type_from_name(vessel)
    main_tool = TOOL_TABLE["Stir"].get(obj_type, FAILURE_TOKEN)

    for time in STIR_TIMES:
        xdl = f'<Stir vessel="{vessel}" time="{time}" />'

        is_space_constrained = random.choice([True, False])

        # 공간 제약이면 사전 Move 필요하다고 가정
        need_move = is_space_constrained and main_tool != FAILURE_TOKEN
        move_tool = (
            TOOL_TABLE["Move"].get(obj_type, FAILURE_TOKEN)
            if need_move else "None"
        )

        all_samples.append({
            "instruction": {
                "xdl": xdl,
                "is_space_constrained": is_space_constrained,
            },
            "output": [
                main_tool,
                str(need_move),
                move_tool,
            ],
        })

# 3) Transfer
for from_vessel in ALL_OBJECTS:
    from_type = object_type_from_name(from_vessel)

    for to_vessel in ALL_OBJECTS:
        for volume in VOLUMES:
            if to_vessel == from_vessel:
                main_tool = FAILURE_TOKEN
            else:
                main_tool = TOOL_TABLE["Transfer"].get(from_type, FAILURE_TOKEN)

            xdl = (
                f'<Transfer from_vessel="{from_vessel}" '
                f'to_vessel="{to_vessel}" volume="{volume}" />'
            )

            is_space_constrained = random.choice([True, False])

            need_move = is_space_constrained and main_tool != FAILURE_TOKEN
            move_tool = (
                TOOL_TABLE["Move"].get(from_type, FAILURE_TOKEN)
                if need_move else "None"
            )

            all_samples.append({
                "instruction": {
                    "xdl": xdl,
                    "is_space_constrained": is_space_constrained,
                },
                "output": [
                    main_tool,
                    str(need_move),
                    move_tool,
                ],
            })

print("Total unique XDL steps:", len(all_samples))

# -----------------------------
# Group by main_tool label
# -----------------------------

by_tool = defaultdict(list)
for s in all_samples:
    main_tool = s["output"][0]
    by_tool[main_tool].append(s)

for k, v in by_tool.items():
    print(f"{k:6s} : {len(v)}")

# -----------------------------
# Balance classes (by main_tool)
# -----------------------------

TARGET_PER_CLASS = 300   # 클래스당 300개

balanced_samples = []

for tool, samples in by_tool.items():
    if len(samples) >= TARGET_PER_CLASS:
        chosen = random.sample(samples, TARGET_PER_CLASS)
    else:
        repeats = TARGET_PER_CLASS // len(samples)
        remainder = TARGET_PER_CLASS % len(samples)

        chosen = samples * repeats + random.sample(samples, remainder)

    balanced_samples.extend(chosen)

random.shuffle(balanced_samples)

# -----------------------------
# Save JSONL
# -----------------------------

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for s in balanced_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print("\nSaved balanced JSONL dataset to:")
print(OUTPUT_PATH)
print(f"Total balanced samples: {len(balanced_samples)}")

# -----------------------------
# Final sanity check
# -----------------------------

final_count = defaultdict(int)
for s in balanced_samples:
    final_count[s["output"][0]] += 1

print("\nFinal class distribution (by main_tool):")
for k, v in final_count.items():
    print(f"{k:6s} : {v}")
