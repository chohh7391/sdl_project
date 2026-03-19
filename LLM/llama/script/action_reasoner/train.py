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

LLAMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "llama")
DATA_PATH = os.path.join(LLAMA_PATH, "data", "dataset", "tool_jsonl", "xdl_2_tool_move.jsonl")
OUTPUT_DIR = os.path.join(LLAMA_PATH, "model", "checkpoint", "tool_move")
SAVE_DIR = os.path.join(OUTPUT_DIR, "checkpoint")

MAX_SEQ_LEN = 256
LR = 1e-5
EPOCHS = 6
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

PROMPT_TEMPLATE = """You are a tool selection system for laboratory automation.

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

EOS = tokenizer.eos_token

def format_example(example):
    xdl = example["instruction"]["xdl"]
    is_space_constrained = example["instruction"]["is_space_constrained"]

    main_tool, need_move, move_tool = example["output"]

    prompt = PROMPT_TEMPLATE.format(
        xdl=xdl,
        is_space_constrained=is_space_constrained
    )

    # 출력은 정확히: tool,True/False,tool 형태
    target = f"{main_tool},{need_move},{move_tool}"

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
        save_steps=50,
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
