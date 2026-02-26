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
LLAMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "llama")

MODEL_NAME = "unsloth/Llama-3.2-1B-bnb-4bit"
DATA_PATH = os.path.join(LLAMA_PATH, "data", "dataset", "pre_xdl_data", "procedure_instruction_v1.jsonl")
OUTPUT_DIR = os.path.join(LLAMA_PATH, "model", "checkpoint")

os.path.join(os.path.dirname(__file__), "..", "..", "..", "llama", "result", "xdl_llm")

MAX_SEQ_LEN = 1024
LR = 1e-5
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

PROMPT_TEMPLATE = """You are a laboratory automation system.
Convert the following instruction into a procedure description.
Output ONLY valid XML inside <procedure> tags.
Do NOT add explanations or extra text.

Instruction:
{instruction}

Procedure:
"""

EOS = tokenizer.eos_token

def format_example(example):
    prompt = PROMPT_TEMPLATE.format(instruction=example["instruction"])
    example["text"] = prompt + example["output"] + EOS
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
    packing=True,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=10,
        save_steps=200,
        optim="adamw_8bit",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="none",
    ),
)

trainer.train()

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training finished.")
print(f"Model saved to: {OUTPUT_DIR}")
