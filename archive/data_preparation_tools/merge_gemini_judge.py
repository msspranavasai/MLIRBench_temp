#!/usr/bin/env python3
"""Import verified Gemini judge scores into MLIRBench evaluation views."""

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/MLIRBench"
PROCESSED = ROOT / "data/processed/MLIRBench"
JUDGE = ROOT / "data/raw/gemini/judge/gemini_full_2026_09_23.json"
MODEL = "gemini-3.1-pro-preview"

VIEWS = ("evaluation", "model_arena")


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_view(view, records, columns):
    directory = PROCESSED / view
    assert all(set(record) == set(columns) for record in records)

    jsonl_path = directory / f"{view}.jsonl"
    csv_path = directory / f"{view}.csv"
    metadata_path = directory / "dataset_info.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({
                k: json.dumps(v, ensure_ascii=False)
                if isinstance(v, (dict, list)) else v
                for k, v in record.items()
            })

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["columns"] == columns
    metadata["rows"] = len(records)

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"{view}: {len(records):,} records")


def main():
    judge = json.loads(JUDGE.read_text(encoding="utf-8"))
    scores = judge["scores"]

    assert judge["schema"] == "openai_simple_scores_v1"
    assert len(scores) == 1763
    assert judge["n_scored"] == 1763

    reasoning = read_jsonl(
        PROCESSED / "reasoning/reasoning.jsonl"
    )

    gemini = {
        row["program_id"]: row
        for row in reasoning
        if row["model_key"] == MODEL
    }

    assert len(gemini) == 1768

    originals = {}
    columns = {}

    for view in VIEWS:
        originals[view] = read_jsonl(
            RAW / view / f"{view}.jsonl"
        )
        assert len(originals[view]) == 14144

        columns[view] = list(originals[view][0])

        assert all(
            list(row) == columns[view]
            for row in originals[view]
        )

    new_evaluation = []
    new_arena = []
    seen = set()

    for score in scores:
        pid = score["program_id"]

        assert pid in gemini, f"Unknown program: {pid}"
        assert pid not in seen, f"Duplicate program: {pid}"
        assert score["evaluated_model"] == MODEL

        reasoning_row = gemini[pid]

        assert (
            score["predicted_label"]
            == reasoning_row["predicted_label"]
        ), pid

        seen.add(pid)

        sub = score["subscores"]
        weighted = score["weighted_scores"]

        evaluation = {
            "program_id": pid,
            "evaluated_model": MODEL,
            "ground_truth_label": score["ground_truth_label"],
            "predicted_label": score["predicted_label"],
            "verdict": score["verdict"],
            "total_score": score["total_score"],
            "explanation_rating": sub["explanation_rating"],
            "hard_label_accuracy": sub["hard_label_accuracy"],
            "result_accuracy": sub["result_accuracy"],
            "weighted_explanation_score": weighted["explanation_rating"],
            "weighted_hard_label_score": weighted["hard_label"],
            "weighted_result_score": weighted["result_accuracy"],
        }

        arena = {
            "program_id": pid,
            "model_key": MODEL,
            "model_family": reasoning_row["model_family"],
            "provider": reasoning_row["provider"],
            "predicted_label": reasoning_row["predicted_label"],
            "confidence": reasoning_row["confidence"],
            "latency_ms": reasoning_row["latency_ms"],
            "total_score": evaluation["total_score"],
            "explanation_rating": evaluation["explanation_rating"],
            "hard_label_accuracy": evaluation["hard_label_accuracy"],
            "result_accuracy": evaluation["result_accuracy"],
        }

        assert set(evaluation) == set(columns["evaluation"])
        assert set(arena) == set(columns["model_arena"])

        new_evaluation.append(evaluation)
        new_arena.append(arena)

    missing = sorted(set(gemini) - seen)

    assert len(missing) == 5

    provenance = read_jsonl(
        PROCESSED / "reasoning/gemini_provenance.jsonl"
    )

    statuses = {
        row["canonical_program_id"]: row["status"]
        for row in provenance
    }

    assert len(statuses) == 1768
    assert all(statuses[pid] == "MODEL_FAILURE" for pid in missing)
    assert all(statuses[pid] == "SUCCESS" for pid in seen)

    # Confirm judge ground truth agrees with existing evaluation.
    ground_truth = {}

    for row in originals["evaluation"]:
        pid = row["program_id"]
        label = row["ground_truth_label"]

        if pid in ground_truth:
            assert ground_truth[pid] == label, pid
        else:
            ground_truth[pid] = label

    for row in new_evaluation:
        assert (
            row["ground_truth_label"]
            == ground_truth[row["program_id"]]
        ), f"Ground truth mismatch: {row['program_id']}"

    assert len(new_evaluation) == 1763
    assert len(new_arena) == 1763

    print("Judge/prediction matching: PASS")
    print("Ground-truth consistency: PASS")
    print("Five unscored failures: VERIFIED")

    print("\nJudge verdicts:")
    for verdict, count in sorted(
        Counter(row["verdict"] for row in new_evaluation).items()
    ):
        print(f"  {verdict}: {count:,}")

    print("\nWriting views:")

    for view, additions in (
        ("evaluation", new_evaluation),
        ("model_arena", new_arena),
    ):
        merged = originals[view] + additions
        assert len(merged) == 15907

        save_view(view, merged, columns[view])

    print("\nPASS: Gemini judge import complete")
    print("Scored records per view: 15,907")
    print("Unscored Gemini failures: 5")


if __name__ == "__main__":
    main()
