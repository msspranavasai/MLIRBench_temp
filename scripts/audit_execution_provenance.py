"""Extract a provenance sidecar from an original execution CSV, without editing it."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csv', type=Path)
    ap.add_argument('--dataset', type=Path, default=Path('data/processed/MLIRBench/master/master.jsonl'))
    ap.add_argument('--output', type=Path, default=Path('data/replay/historical_execution.jsonl'))
    args = ap.parse_args()
    csv.field_size_limit(10000000)
    rows = list(csv.DictReader(args.csv.open()))
    dataset = {r['program_id']: r for r in map(json.loads, args.dataset.open())}
    grouped = collections.defaultdict(list)
    for row in rows: grouped[row['program_id']].append(row)
    source_hash = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    extracted = []
    for pid in sorted(dataset):
        matches = grouped[pid]
        if len(matches) != 1: raise ValueError(f'{pid}: expected one original CSV record, found {len(matches)}')
        row = matches[0]
        evidence = {k: v for k, v in row.items() if any(s in k for s in ('source_file', 'output', 'embedded_inputs', 'input_spec', 'backend', 'run_config', 'compilation_status', 'compile_status', 'pairs__status'))}
        extracted.append({'program_id': pid, 'provenance': {'source_filename': args.csv.name, 'source_sha256': source_hash, 'row_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()},
                          'source_sha256': [hashlib.sha256(dataset[pid][k].encode()).hexdigest() for k in ('input_mlir', 'mutated_mlir')], 'historical_fields': evidence})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        for r in extracted: out.write(json.dumps(r, sort_keys=True) + '\n')
    print(json.dumps({'source_rows': len(rows), 'published_matches': len(extracted), 'source_sha256': source_hash, 'output': str(args.output)}, indent=2))


if __name__ == '__main__': main()
