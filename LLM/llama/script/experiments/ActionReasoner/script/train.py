import os
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# =========================
# Config
# =========================
MODEL_NAME = "unsloth/Llama-3.2-1B-bnb-4bit"
DATA_PATH = "/home/home/sdl_ws/src/sdl_project/LLM/llama/script/experiments/ActionReasoner/dataset/action_reasoner_dataset.jsonl"
OUTPUT_DIR = "/home/home/sdl_ws/src/sdl_project/LLM/llama/script/experiments/ActionReasoner/model/"
SAVE_DIR = os.path.join(OUTPUT_DIR, "checkpoint")

MAX_SEQ_LEN = 512
LR = 1e-4
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8

# =========================
# Load model (4bit)
# =========================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)

# =========================
# Apply LoRA
# =========================
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "up_proj", "down_proj", "gate_proj"
    ],
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
)
model.print_trainable_parameters()

# =========================
# Load dataset
# =========================
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

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

EOS = tokenizer.eos_token

def format_example(example):
    inst = example["instruction"]
    prompt = PROMPT_TEMPLATE.format(
        current_xdl=inst["current_xdl"],
        next_xdl=inst["next_xdl"],
        obstacle_info=inst["obstacle_info"],
        candidate_grids=inst["candidate_grids"],
    )
    target = example["output"]
    example["text"] = prompt + target + EOS
    return example

dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# =========================
# Trainer
# =========================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=10,
        save_steps=100,
        optim="adamw_8bit",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="none",
    ),
)

trainer.train()

os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("✅ Training finished.")
print(f"Model saved to: {SAVE_DIR}")