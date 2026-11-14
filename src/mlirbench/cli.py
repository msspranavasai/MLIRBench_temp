"""Offline artifact commands. Imports no provider clients and never reads .env."""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / 'data/processed/MLIRBench/master/master.jsonl'
HISTORICAL = ROOT / 'data/historical/historical_execution.jsonl'
KNOWN = ('seed1_linalg_prog7_L1_R2', 'seed1_linalg_prog10_L3_R4',
         'seed1_linalg_prog10_L4_R4', 'seed2_linalg_prog16_L4_R4',
         'seed2_linalg_prog13_L4_R4', 'torch_seed4_prog40_T1_R1',
         'torch_seed1_prog4_T2_R3', 'torch_seed2_prog16_T2_R5')


def load(path):
    with path.open() as f: return [json.loads(line) for line in f if line.strip()]


def select_sample(rows):
    groups = {}
    for r in sorted(rows, key=lambda r: r['program_id']):
        groups.setdefault((r['dialect'], r['level'], str(r['judge__outputs_diverge'])), r)
    chosen = {r['program_id']: r for r in groups.values()}
    chosen.update((r['program_id'], r) for r in rows if r['program_id'] in KNOWN)
    return [chosen[k] for k in sorted(chosen)]


def published_comparison(record, report):
    from mlirbench.execution.pair_validation import published_label
    label, verdict = published_label(record), report['verdict']
    if label == 'DIVERGE':
        return 'REPRODUCED' if verdict == 'DIVERGE' else 'NOT_REPRODUCED' if verdict == 'NO_DIFFERENCE_OBSERVED' else 'UNVERIFIED'
    if label == 'EQUIV':
        return 'COUNTEREXAMPLE_FOUND' if verdict == 'DIVERGE' else 'NO_COUNTEREXAMPLE_OBSERVED' if verdict == 'NO_DIFFERENCE_OBSERVED' else 'UNVERIFIED'
    if record.get('pairs__status') == 'expected_invalid' and verdict == 'COMPILE_ERROR':
        c = report['compilation']
        if c[0]['status'] == 'success' and c[1]['status'] == 'error': return 'EXPECTED_MUTATION_COMPILE_ERROR'
    return 'UNVERIFIED'


def write_reports(path, reports):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse accidental report overwrite as well as any dataset destination.
    with path.open('x') as f:
        for r in reports: f.write(json.dumps(r, sort_keys=True, allow_nan=False) + '\n')
    with path.with_suffix('.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['program_id', 'published_label', 'verdict', 'published_comparison'])
        w.writeheader()
        w.writerows({k: r.get(k) for k in w.fieldnames} for r in reports)
    counts = collections.Counter(r['verdict'] for r in reports)
    text = '# Execution audit\n\nFinite equal-output tests do not prove equivalence. Published records are unchanged.\n\n'
    text += '\n'.join(f'- {k}: {v}' for k, v in sorted(counts.items())) + '\n\n'
    text += '| ID | Published | Observed | Published comparison |\n|---|---|---|---|\n'
    text += ''.join(f"| {r['program_id']} | {r['published_label']} | {r['verdict']} | {r['published_comparison']} |\n" for r in reports)
    with path.with_suffix('.md').open('x') as f: f.write(text)
    print(json.dumps({'records': len(reports), 'verdicts': dict(counts), 'output': str(path)}, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    for name, help_text in [('quickstart', 'Run the offline reviewer workflow and write one readiness report'),
                            ('reproduce', 'Regenerate paper-facing metrics from archived observations'),
                            ('reproduce-downstream', 'Regenerate downstream metrics from saved predictions')]:
        parser = sub.add_parser(name, help=help_text)
        parser.add_argument('--output', type=Path, required=True)
    parser = sub.add_parser('readiness', help='Report unfinished AE requirements explicitly')
    parser.add_argument('--results', type=Path, required=True)
    sub.add_parser('prepare-data', help='Unpack and verify immutable data bundles')
    sub.add_parser('check', help='Check local toolchain and execute tiny programs')
    info = sub.add_parser('dataset-info'); info.add_argument('--dataset', type=Path, default=DATASET)
    v = sub.add_parser('validate', help='Canonical deterministic suite or explicit recorded input replay')
    v.add_argument('--dataset', type=Path, default=DATASET)
    group = v.add_mutually_exclusive_group(required=True)
    group.add_argument('--id', action='append'); group.add_argument('--sample', action='store_true'); group.add_argument('--all', action='store_true')
    v.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 42, 123, 2026])
    v.add_argument('--inputs', type=Path, help='JSON positional typed input list; requires one ID')
    v.add_argument('--stored-metadata', action='store_true', help='Use original embedded-input metadata as shared inputs; runtime provenance unproven')
    v.add_argument('--timeout', type=int, default=30); v.add_argument('--repeats', type=int, default=2)
    v.add_argument('--output', type=Path, required=True)
    h = sub.add_parser('historical-candidate', help='Test independent Linalg embedding hypothesis; never a semantic verdict')
    h.add_argument('--id', required=True); h.add_argument('--output', type=Path, required=True)
    sub.add_parser('test', help='Run offline unit and compiler integration tests')
    sub.add_parser('score-recorded', help='Summarize recorded labels/scores and expose missing predictions')
    args = ap.parse_args(argv)
    if args.command == 'prepare-data':
        from .ae import prepare_data
        print(json.dumps(prepare_data(ROOT), indent=2)); return 0
    if args.command == 'quickstart':
        from .ae import quickstart
        return quickstart(ROOT, args.output)
    if args.command == 'reproduce':
        from .evaluation.reproduce import reproduce
        print(json.dumps(reproduce(ROOT, args.output), indent=2)); return 0
    if args.command == 'reproduce-downstream':
        from .evaluation.downstream import reproduce_downstream
        print(json.dumps(reproduce_downstream(ROOT, args.output), indent=2)); return 0
    if args.command == 'readiness':
        from .ae import readiness
        report = readiness(ROOT, args.results)
        print(json.dumps(report, indent=2)); return 0 if report['status'] == 'READY_FOR_SUBMISSION_REVIEW' else 2
    if args.command == 'dataset-info':
        rows = load(args.dataset)
        print(json.dumps({'records': len(rows), 'sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                          'labels': dict(collections.Counter(str(r['judge__outputs_diverge']) for r in rows)),
                          'dialects': dict(collections.Counter(r['dialect'] for r in rows)),
                          'statuses': dict(collections.Counter(str(r['pairs__status']) for r in rows))}, indent=2)); return 0
    if args.command == 'score-recorded':
        from .evaluation.recorded import summarize
        path = ROOT / 'data/processed/MLIRBench/evaluation/evaluation.jsonl'
        report = summarize(load(DATASET), load(path))
        report['master_sha256'] = hashlib.sha256(DATASET.read_bytes()).hexdigest()
        report['evaluation_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(json.dumps(report, indent=2, allow_nan=False)); return 0
    if args.command == 'test':
        from .ae import run_tests
        report = run_tests(ROOT)
        print(json.dumps(report, indent=2)); return 0 if report['passed'] else 1
    from mlirbench.execution.pair_validation import validate_pair, environment, published_label
    if args.command == 'check':
        from .ae import check_environment
        report = check_environment()
        print(json.dumps(report, indent=2)); return 0 if report['passed'] else 1
    if args.command == 'historical-candidate':
        from mlirbench.execution.historical_replay import replay_candidate
        row = next((r for r in load(DATASET) if r['program_id'] == args.id), None)
        if row is None: ap.error('Unknown program ID')
        evidence = next(r for r in load(HISTORICAL) if r['program_id'] == args.id)
        report = replay_candidate(row, evidence)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f: json.dump(report, f, indent=2, allow_nan=False)
        print(json.dumps({k: report[k] for k in ('program_id', 'status')})); return 0
    if args.output.suffix != '.jsonl': ap.error('--output must end in .jsonl')
    rows = load(args.dataset)
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if args.id:
        missing = set(args.id) - {r['program_id'] for r in rows}
        if missing: ap.error(f'Unknown IDs: {sorted(missing)}')
        selected = [r for r in rows if r['program_id'] in args.id]
    else: selected = select_sample(rows) if args.sample else rows
    if args.inputs and (len(selected) != 1 or args.stored_metadata): ap.error('--inputs requires one ID and cannot combine with --stored-metadata')
    # Preflight all destinations before spending compilation time.
    for p in (args.output, args.output.with_suffix('.csv'), args.output.with_suffix('.md')):
        if p.exists(): ap.error(f'Output exists: {p}; choose a new report path')
    evidence = {r['program_id']: r for r in load(HISTORICAL)} if args.stored_metadata else {}
    reports = []
    for i, r in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {r['program_id']}", file=sys.stderr, flush=True)
        recorded = json.loads(args.inputs.read_text()) if args.inputs else None
        if args.stored_metadata:
            from mlirbench.execution.pair_validation import argument_names
            from mlirbench.execution.execution import parse_input_spec
            values = json.loads(evidence[r['program_id']]['historical_fields']['judge__embedded_inputs_json'])
            specs = parse_input_spec(r['input_mlir'])
            recorded = [{'position': i, **s, 'data': values[n]} for i, (n, s) in enumerate(zip(argument_names(r['input_mlir']), specs))]
        report = validate_pair(r['input_mlir'], r['mutated_mlir'], r['dialect'], seeds=args.seeds,
                               recorded_inputs=recorded, repeats=args.repeats, timeout=args.timeout)
        report.update(program_id=r['program_id'], published_label=published_label(r),
                      dataset_sha256=dataset_hash,
                      stored_metadata_runtime_provenance='UNPROVEN' if args.stored_metadata else None)
        report['published_comparison'] = published_comparison(r, report)
        reports.append(report)
        print(f"  {report['verdict']}", file=sys.stderr, flush=True)
    write_reports(args.output, reports)
    return 0  # Scientific discrepancies are report contents, not infrastructure exit failures.
