#!/usr/bin/env python3

import csv
import json
import re
from collections import Counter
from pathlib import Path

RAW = Path("data/raw/MLIRBench")
OUT = Path("data/processed/MLIRBench")

def load(view, root=RAW):
    path = root / view / f"{view}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

provenance = {r["program_id"]: r for r in load("mutation_provenance")}
reasoning = {}

for row in load("reasoning"):
    reasoning.setdefault(row["program_id"], row)

expected_ids = set(provenance)

assert len(expected_ids) == 1768
assert set(reasoning) == expected_ids

def deduplicate(rows):
    result = {}

    for row in rows:
        pid = row["program_id"]

        if pid in result:
            assert result[pid] == row, f"Conflicting duplicate: {pid}"
        else:
            result[pid] = row

    return result

def seed_from_id(pid):
    match = re.search(r"(?:^|_)seed(\d+)_", pid)
    assert match, pid
    return int(match.group(1))

def recover(pid, columns):
    p = provenance[pid]
    r = reasoning[pid]

    source = {
        **r,
        **p,
        "seed": seed_from_id(pid),
        "valid_status": (
            bool(p["validation_history"][-1]["validation"]["valid"])
            if p.get("validation_history")
            else None
        ),
    }

    # Mutation validation does not establish execution success.
    # Preserve unknown execution and timing fields as null.
    return {field: source.get(field) for field in columns}

for view in ("core", "master", "performance"):
    original = load(view)
    columns = list(original[0])

    existing = deduplicate(original)
    missing = sorted(expected_ids - set(existing))
    unexpected = sorted(set(existing) - expected_ids)

    assert not unexpected, (view, unexpected)
    assert len(existing) == 1765, (view, len(existing))
    assert len(missing) == 3, (view, missing)

    repaired = list(existing.values())

    for pid in missing:
        repaired.append(recover(pid, columns))

    assert len(repaired) == 1768
    assert {r["program_id"] for r in repaired} == expected_ids

    destination = OUT / view
    destination.mkdir(parents=True, exist_ok=True)

    jsonl_path = destination / f"{view}.jsonl"
    csv_path = destination / f"{view}.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in repaired:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in repaired:
            writer.writerow({
                k: (
                    json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (dict, list))
                    else v
                )
                for k, v in row.items()
            })

    metadata_path = destination / "dataset_info.json"

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["num_examples"] = len(repaired)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"\n=== {view.upper()} ===")
    print("Original rows:", len(original))
    print("Original unique IDs:", len(existing))
    print("Removed identical duplicates:", len(original) - len(existing))
    print("Recovered programs:", missing)
    print("Final rows:", len(repaired))
    print("Final unique IDs:", len({r["program_id"] for r in repaired}))
    print("Recovered fields with null values:")

    for pid in missing:
        row = next(r for r in repaired if r["program_id"] == pid)
        null_fields = [k for k, v in row.items() if v is None]
        print(f"  {pid}: {null_fields}")

print("\nBenchmark-view repair complete.")
