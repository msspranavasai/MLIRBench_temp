#!/usr/bin/env python3
"""
Merge Gemini 3.1 Pro reasoning results into MLIRBench.

Inputs:
  data/raw/MLIRBench/reasoning/reasoning.jsonl
  data/raw/gemini/gemini_lin_alg/runs.jsonl
  data/raw/gemini/gemini_torch/runs.jsonl

Outputs:
  data/processed/MLIRBench/reasoning/reasoning.jsonl
  data/processed/MLIRBench/reasoning/reasoning.csv
  data/processed/MLIRBench/reasoning/gemini_provenance.jsonl

The original Hugging Face dataset remains unchanged.
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data/raw"
OUTPUT = ROOT / "data/processed/MLIRBench/reasoning"

ORIGINAL = RAW / "MLIRBench/reasoning/reasoning.jsonl"

GEMINI_FILES = [
    RAW / "gemini/gemini_lin_alg/runs.jsonl",
    RAW / "gemini/gemini_torch/runs.jsonl",
]

MODEL_KEY = "gemini-3.1-pro-preview"

CANONICAL_FIELDS = [
    "program_id",
    "dialect",
    "level",
    "requirement_id",
    "input_mlir",
    "mutated_mlir",
    "judge__outputs_diverge",
    "judge__semantic_explanation",
]


def read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    records = []

    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc

    return records


def get_canonical_id(record):
    seed_match = re.fullmatch(
        r"seed_(\d+)", record["seed_id"]
    )
    program_match = re.fullmatch(
        r"program_(\d+)", record["program_id"]
    )

    if not seed_match or not program_match:
        raise ValueError(
            f"Invalid Gemini identifiers: {record['pair_id']}"
        )

    seed = int(seed_match.group(1))
    program = int(program_match.group(1))
    mutation = record["mutation_id"]
    dialect = record["dialect"]

    if dialect == "linalg":
        return f"seed{seed}_linalg_prog{program}_{mutation}"

    if dialect == "torch":
        return f"torch_seed{seed}_prog{program}_{mutation}"

    raise ValueError(f"Unknown dialect: {dialect}")


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main():
    print("=" * 60)
    print("MLIRBENCH GEMINI MERGE")
    print("=" * 60)

    original = read_jsonl(ORIGINAL)

    gemini = []
    for path in GEMINI_FILES:
        gemini.extend(read_jsonl(path))

    print(f"Original records: {len(original):,}")
    print(f"Gemini records:   {len(gemini):,}")

    assert len(original) == 14144
    assert len(gemini) == 1768

    fields = list(original[0].keys())

    assert all(set(record) == set(fields) for record in original)

    grouped = defaultdict(list)

    for record in original:
        key = (record["dialect"], record["program_id"])
        grouped[key].append(record)

    assert len(grouped) == 1768

    for key, records in grouped.items():
        assert len(records) == 8, (
            f"Expected eight model records for {key}"
        )

        reference = records[0]

        assert len(set(r["model_key"] for r in records)) == 8

        for record in records[1:]:
            for field in CANONICAL_FIELDS:
                assert record[field] == reference[field], (
                    f"Canonical field mismatch: {key}, {field}"
                )

    print("Canonical benchmark consistency: PASS")

    converted = []
    provenance = []
    seen = set()

    for record in gemini:
        canonical_id = get_canonical_id(record)
        dialect = record["dialect"]

        key = (dialect, canonical_id)

        assert key in grouped, (
            f"Missing benchmark pair: {key}"
        )

        assert key not in seen, (
            f"Duplicate Gemini pair: {key}"
        )

        seen.add(key)

        reference = grouped[key][0]

        assert (
            reference["requirement_id"]
            == record["mutation_id"]
        )

        assert record["model_key"] == MODEL_KEY

        assert record["status"] in {
            "SUCCESS",
            "MODEL_FAILURE",
        }

        if record["status"] == "MODEL_FAILURE":
            assert record.get("predicted_label") is None

        converted_record = {
            field: reference[field]
            for field in CANONICAL_FIELDS
        }

        converted_record.update({
            "model_key": MODEL_KEY,
            "model_family": "proprietary",
            "provider": record["provider"],
            "predicted_label": record.get("predicted_label"),
            "confidence": record.get("confidence"),
            "predicted_output_a": record.get(
                "predicted_output_A"
            ),
            "predicted_output_b": record.get(
                "predicted_output_B"
            ),
            "explanation": record.get("explanation"),
            "witness": record.get("witness"),
            "key_differences": record.get(
                "key_differences"
            ),
            "latency_ms": record.get("latency_ms"),
        })

        assert set(converted_record) == set(fields)

        converted.append({
            field: converted_record[field]
            for field in fields
        })

        provenance.append({
            "canonical_program_id": canonical_id,
            "dialect": dialect,
            **record,
        })

    assert len(seen) == 1768

    print("Gemini identifier matching: PASS")

    merged = original + converted

    assert len(merged) == 15912

    model_counts = Counter(
        record["model_key"]
        for record in merged
    )

    assert len(model_counts) == 9
    assert all(
        count == 1768
        for count in model_counts.values()
    )

    dialect_counts = Counter(
        record["dialect"]
        for record in merged
    )

    assert dialect_counts == {
        "linalg": 7992,
        "torch": 7920,
    }

    # All validation occurs before writing.
    OUTPUT.mkdir(parents=True, exist_ok=True)

    jsonl_path = OUTPUT / "reasoning.jsonl"
    csv_path = OUTPUT / "reasoning.csv"
    provenance_path = OUTPUT / "gemini_provenance.jsonl"

    write_jsonl(jsonl_path, merged)
    write_jsonl(provenance_path, provenance)

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in merged:
            csv_record = {
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                )
                for key, value in record.items()
            }

            writer.writerow(csv_record)

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)

    print(f"Original: {len(original):,}")
    print(f"Gemini:   {len(converted):,}")
    print(f"Total:    {len(merged):,}")

    print("\nModels:")
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count:,}")

    print("\nDialects:")
    for dialect, count in sorted(dialect_counts.items()):
        print(f"  {dialect}: {count:,}")

    print("\nGemini statuses:")
    for status, count in sorted(
        Counter(r["status"] for r in gemini).items()
    ):
        print(f"  {status}: {count:,}")

    print("\nOutput files:")
    print(jsonl_path)
    print(csv_path)
    print(provenance_path)


if __name__ == "__main__":
    main()
