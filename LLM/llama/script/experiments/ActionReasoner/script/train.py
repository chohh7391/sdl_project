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
# NOTE: the default DATA_PATH is the FULL 3000-sample dataset, not the 2400
# train split that split_dataset.py writes. The shipped checkpoint was trained
# with it -- max_steps 564 == ceil(3000/16) * 3 epochs, where the 2400 split
# would give 450 -- so the 600-item "test" set was inside the training data and
# the accuracies measured on it are in-sample. AR_DATA_PATH / AR_OUTPUT_DIR
# override both so a split-respecting run can be done without disturbing the
# original checkpoint.
DATA_PATH = os.environ.get(
    "AR_DATA_PATH",
    "/home/home/sdl_ws/src/sdl_project/LLM/llama/script/experiments/ActionReasoner/dataset/action_reasoner_dataset.jsonl")
OUTPUT_DIR = os.environ.get(
    "AR_OUTPUT_DIR",
    "/home/home/sdl_ws/src/sdl_project/LLM/llama/script/experiments/ActionReasoner/model/")
SAVE_DIR = os.path.join(OUTPUT_DIR, "checkpoint")
print("[train] data=%s" % DATA_PATH)
print("[train] output=%s" % OUTPUT_DIR)

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

# The prompt lives in ar_prompt so training and evaluation cannot drift apart,
# and so AR_GEOMETRY=1 adds the object widths to both at once.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from ar_prompt import build_prompt, geometry_enabled
print("[train] geometry grounding: %s" % ("ON" if geometry_enabled() else "off"))

EOS = tokenizer.eos_token

def format_example(example):
    inst = example["instruction"]
    prompt = build_prompt(inst)
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