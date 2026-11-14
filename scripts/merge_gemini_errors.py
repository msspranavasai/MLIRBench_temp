#!/usr/bin/env python3
"""Add Gemini 3.1 Pro to the MLIRBench errors view."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/MLIRBench"
PROCESSED = ROOT / "data/processed/MLIRBench"
MODEL = "gemini-3.1-pro-preview"


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    original = read_jsonl(RAW / "errors/errors.jsonl")
    reasoning = read_jsonl(PROCESSED / "reasoning/reasoning.jsonl")

    assert len(original) == 14144

    gemini = {
        row["program_id"]: row
        for row in reasoning
        if row["model_key"] == MODEL
    }

    assert len(gemini) == 1768

    columns = list(original[0])

    benchmark_fields = (
        "program_id",
        "dialect",
        "level",
        "input_mlir",
        "mutated_mlir",
        "failure_category",
        "failure_category_short",
        "pairs__error_message",
        "judge__semantic_explanation",
    )

    canonical = {}

    for row in original:
        pid = row["program_id"]
        if pid not in canonical:
            canonical[pid] = row
        else:
            assert all(
                row[field] == canonical[pid][field]
                for field in benchmark_fields
            ), pid

    assert len(canonical) == 1768
    assert set(canonical) == set(gemini)

    additions = []

    for pid in sorted(gemini):
        reference = canonical[pid]
        prediction = gemini[pid]

        # Ensure Gemini's benchmark fields match the canonical pair.
        for field in (
            "program_id",
            "dialect",
            "level",
            "input_mlir",
            "mutated_mlir",
            "judge__semantic_explanation",
        ):
            assert prediction[field] == reference[field], (
                pid, field
            )

        record = {
            field: reference[field]
            for field in benchmark_fields
        }

        record.update({
            "model_key": MODEL,
            "predicted_label": prediction["predicted_label"],
            "explanation": prediction["explanation"],
        })

        assert set(record) == set(columns)
        additions.append({
            column: record[column]
            for column in columns
        })

    merged = original + additions

    assert len(merged) == 15912

    directory = PROCESSED / "errors"
    metadata_path = directory / "dataset_info.json"

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert metadata["columns"] == columns

    # All validations complete before writing.
    with (directory / "errors.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for record in merged:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    with (directory / "errors.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for record in merged:
            writer.writerow(record)

    metadata["rows"] = len(merged)

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Original records preserved:", len(original))
    print("Gemini records added:", len(additions))
    print("Total errors records:", len(merged))
    print("Metadata rows:", metadata["rows"])
    print("PASS: Gemini errors merge complete")


if __name__ == "__main__":
    main()
