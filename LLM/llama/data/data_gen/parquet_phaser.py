import os
import json
import pyarrow as pa
import pyarrow.parquet as pq

# -----------------------------
# Paths
# -----------------------------

INPUT_JSONL = os.path.join(os.path.dirname(__file__), "..", "dataset", "pre_xdl_data", "procedure_instruction_v1.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "parquet_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Config
# -----------------------------

SHARD_SIZE = 1000   # samples per parquet
BASENAME = "train"

# -----------------------------
# Load JSONL
# -----------------------------

samples = []

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

total = len(samples)
print(f"Loaded {total} samples")

# -----------------------------
# Write Parquet shards
# -----------------------------

for shard_id in range((total + SHARD_SIZE - 1) // SHARD_SIZE):
    start = shard_id * SHARD_SIZE
    end = min(start + SHARD_SIZE, total)

    shard_samples = samples[start:end]

    table = pa.Table.from_pylist(
        shard_samples,
        schema=pa.schema([
            ("instruction", pa.string()),
            ("output", pa.string()),
        ])
    )

    out_path = os.path.join(
        OUTPUT_DIR,
        f"{BASENAME}_{shard_id:03d}.parquet"
    )

    pq.write_table(
        table,
        out_path,
        compression="zstd"  # 권장
    )

    print(f"Saved {out_path} ({len(shard_samples)} samples)")

print("JSONL → Parquet conversion complete.")
