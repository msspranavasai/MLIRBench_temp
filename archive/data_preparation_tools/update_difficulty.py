#!/usr/bin/env python3

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

RAW = Path("data/raw/MLIRBench")
PROCESSED = Path("data/processed/MLIRBench")

def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

provenance = read_jsonl(
    RAW / "mutation_provenance/mutation_provenance.jsonl"
)
evaluation = read_jsonl(
    PROCESSED / "evaluation/evaluation.jsonl"
)

assert len(provenance) == 1768
assert len({r["program_id"] for r in provenance}) == 1768

scores = defaultdict(list)
models = defaultdict(set)

for row in evaluation:
    pid = row["program_id"]
    model = row["evaluated_model"]

    assert model not in models[pid], (pid, model)
    models[pid].add(model)

    score = row["total_score"]
    assert score is not None, (pid, model)
    scores[pid].append(float(score))

assert set(scores) == {r["program_id"] for r in provenance}

def bucket(level):
    level = int(level)

    if level in (1, 2):
        return "easy"
    if level == 3:
        return "medium"
    if level == 4:
        return "hard"
    if level == 5:
        return "expert"

    raise ValueError(f"Unexpected level: {level}")

rows = []

for source in provenance:
    pid = source["program_id"]

    rows.append({
        "program_id": pid,
        "dialect": source["dialect"],
        "level": source["level"],
        "validation_attempts": source["validation_attempts"],
        "avg_model_score": mean(scores[pid]),
        "difficulty_bucket": bucket(source["level"]),
    })

assert len(rows) == 1768
assert len({r["program_id"] for r in rows}) == 1768

distribution = Counter(len(models[r["program_id"]]) for r in rows)
assert distribution == {9: 1763, 8: 5}, distribution

output = PROCESSED / "difficulty"
output.mkdir(parents=True, exist_ok=True)

jsonl_path = output / "difficulty.jsonl"
csv_path = output / "difficulty.csv"

with jsonl_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

metadata_path = output / "dataset_info.json"

if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["num_examples"] = len(rows)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

print("Difficulty reconstruction complete.")
print("Rows:", len(rows))
print("Unique programs:", len({r["program_id"] for r in rows}))
print("Model counts:", dict(distribution))
print("Difficulty buckets:", dict(Counter(
    r["difficulty_bucket"] for r in rows
)))
print("Dialect counts:", dict(Counter(
    r["dialect"] for r in rows
)))
print("Updated:", jsonl_path)
print("Updated:", csv_path)
