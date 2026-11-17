"""Immutable source inventory plus explicit, operation-aware label auditing."""
import csv
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import re

from .execution import parse_input_spec
from .historical_replay import replay_candidate
from .input_policy import InputPolicyError, check_inputs, POLICY_VERSION
from .pair_validation import (argument_names, deserialize_inputs, digest,
                              published_label, validate_pair, DEFAULT_SEEDS)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def verify_sources(root, master, historical):
    folder = root / 'data/historical'
    manifest = json.loads((folder / 'source_manifest.json').read_text())
    packed = (folder / manifest['archive_filename']).read_bytes()
    if sha(packed) != manifest['archive_sha256']:
        raise ValueError('Original historical CSV hash mismatch')
    raw = gzip.decompress(packed)
    if sha(raw) != manifest['source_sha256']:
        raise ValueError('Original historical CSV hash mismatch')
    if sha((folder / manifest['sidecar_filename']).read_bytes()) != manifest['sidecar_sha256']:
        raise ValueError('Historical sidecar hash mismatch')
    master_path = root / 'data/processed/MLIRBench/master/master.jsonl'
    if sha(master_path.read_bytes()) != manifest['published_master_sha256']:
        raise ValueError('Published master differs from the historical-source inventory')
    csv.field_size_limit(10_000_000)
    raw_rows = list(csv.DictReader(io.StringIO(raw.decode())))
    by_id = defaultdict(list)
    for row in raw_rows:
        by_id[row['program_id']].append(row)
    if len(raw_rows) != manifest['source_rows'] or len(master) != manifest['published_rows'] or set(master) != set(historical):
        raise ValueError('Historical/source ID coverage differs')
    for pid, record in master.items():
        candidates = by_id[pid]
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous historical row: {pid}')
        row, evidence = candidates[0], historical[pid]
        fields = {k: v for k, v in row.items() if any(s in k for s in (
            'source_file', 'output', 'embedded_inputs', 'input_spec', 'backend',
            'run_config', 'compilation_status', 'compile_status', 'pairs__status'))}
        if (sha(json.dumps(row, sort_keys=True).encode()) != evidence['provenance']['row_sha256']
                or evidence['provenance']['source_sha256'] != manifest['source_sha256']
                or fields != evidence['historical_fields']
                or [sha(record[k].encode()) for k in ('input_mlir', 'mutated_mlir')] != evidence['source_sha256']):
            raise ValueError(f'Historical evidence mismatch: {pid}')
    return {'passed': True, 'original_csv_rows': len(raw_rows), 'published_rows_verified': len(master),
            'source_sha256': manifest['source_sha256'], 'sidecar_sha256': manifest['sidecar_sha256'],
            'master_sha256': manifest['published_master_sha256'],
            'source_manifest_sha256': sha((folder / 'source_manifest.json').read_bytes()),
            'runtime_inputs_proven': False,
            'runtime_input_reason': manifest['runtime_input_provenance']}


def load_verified_sources(root):
    """Return the frozen observations only after checking their original sources.

    Callers consume these checked in-memory records, not a second unverified
    read of the sidecar. No compilation, input generation or relabeling occurs.
    """
    root = Path(root)
    folder = root / 'data/historical'
    manifest = json.loads((folder / 'source_manifest.json').read_text())
    master_rows = read_rows(root / 'data/processed/MLIRBench/master/master.jsonl')
    historical_rows = read_rows(folder / manifest['sidecar_filename'])
    master = {r['program_id']: r for r in master_rows}
    historical = {r['program_id']: r for r in historical_rows}
    if len(master) != len(master_rows) or len(historical) != len(historical_rows):
        raise ValueError('Duplicate program IDs in historical sources')
    verification = verify_sources(root, master, historical)
    verification['sidecar_filename'] = manifest['sidecar_filename']
    return master, historical, verification


TOKEN = re.compile(r'(?P<string>"(?:\\.|[^"\\])*")|(?P<comment>//[^\n]*)|(?P<ssa>%[\w.$-]+)|(?P<space>\s+)|(?P<other>.)', re.S)


def flat_ssa_form(source):
    """Candidate equality only for a single flat function; not a general IR prover.

    Strings, symbol names, types, operations, literals and operand order remain
    significant. Nested regions/control-flow and unfamiliar name syntax are refused.
    Compiler verification and well-defined execution are separate requirements.
    """
    tokens = list(TOKEN.finditer(source))
    masked = ''.join(' ' * len(m[0]) if m.lastgroup in ('string', 'comment') else m[0] for m in tokens)
    if (len(re.findall(r'\bfunc.func\b', masked)) != 1 or not re.search(r'\bfunc.func\s+@main\(', masked)
            or masked.count('{') != 2 or masked.count('}') != 2 or '^' in masked or '%"' in masked):
        return None
    names, result = {}, []
    for match in tokens:
        kind, token = match.lastgroup, match[0]
        if kind in ('comment', 'space'):
            continue
        if kind == 'ssa':
            token = names.setdefault(token, f'%v{len(names)}')
        result.append((kind, token))
    return result


def source_relation(a, b):
    left, right = flat_ssa_form(a), flat_ssa_form(b)
    if left is None or right is None:
        return 'UNSUPPORTED_STRUCTURAL_ANALYSIS'
    return 'IDENTICAL_UP_TO_SSA_NAMES' if left == right else 'DIFFERENT_TOKEN_STRUCTURE'


def stored_input_check(record, evidence):
    """Validate metadata safely; never execute it as unproven historical input."""
    try:
        sources = [record[k] for k in ('input_mlir', 'mutated_mlir')]
        specs = [parse_input_spec(s) for s in sources]
        names = [argument_names(s) for s in sources]
        if specs[0] != specs[1]:
            return {'status': 'INCOMPARABLE_INPUT_SIGNATURES'}
        values = json.loads(evidence['historical_fields'].get('judge__embedded_inputs_json') or 'null')
        if not isinstance(values, dict):
            return {'status': 'METADATA_UNAVAILABLE'}
        items = [{'position': i, **spec, 'data': values[name]}
                 for i, (name, spec) in enumerate(zip(names[0], specs[0]))]
        arrays = deserialize_inputs(items, specs[0])
        for source, argument_ids in zip(sources, names):
            check_inputs(source, argument_ids, arrays, record['dialect'])
        return {'status': 'VALID_AS_SHARED_INPUT', 'actual_historical_runtime_use_proven': False}
    except InputPolicyError as exc:
        return {'status': exc.status, 'detail': str(exc)}
    except (ValueError, KeyError, TypeError) as exc:
        return {'status': 'METADATA_UNAVAILABLE_OR_UNSUPPORTED', 'detail': str(exc)}


def inventory_record(record, evidence):
    fields = evidence['historical_fields']
    outputs = []
    for name in ('judge__programA_output_json', 'judge__programB_output_json'):
        try: value = json.loads(fields.get(name) or 'null')
        except (ValueError, TypeError): value = None
        outputs.append(value is not None)
    return {'program_id': record['program_id'], 'dialect': record['dialect'], 'level': record['level'],
            'published_label': published_label(record), 'source_sha256': evidence['source_sha256'],
            'historical_outputs_present': outputs, 'historical_runtime_inputs_proven': False,
            'metadata_input_check': stored_input_check(record, evidence),
            'source_relation': source_relation(record['input_mlir'], record['mutated_mlir'])}


def audit_case(task):
    record, evidence, seeds, repeats, timeout = task
    report = validate_pair(record['input_mlir'], record['mutated_mlir'], record['dialect'],
                           seeds=seeds, repeats=repeats, timeout=timeout)
    report.update(program_id=record['program_id'], published_label=published_label(record),
                  source_relation=source_relation(record['input_mlir'], record['mutated_mlir']))
    verdict, label = report['verdict'], report['published_label']
    if label == 'EQUIV' and verdict == 'DIVERGE': assessment = 'LABEL_COUNTEREXAMPLE'
    elif label == 'DIVERGE' and verdict == 'NO_DIFFERENCE_OBSERVED':
        assessment = ('STRUCTURAL_LABEL_CONFLICT' if report['source_relation'] == 'IDENTICAL_UP_TO_SSA_NAMES'
                      else 'PUBLISHED_WITNESS_NOT_REPRODUCED')
    elif label == 'DIVERGE' and verdict == 'DIVERGE': assessment = 'DIVERGENCE_WITNESS_REPRODUCED'
    elif label == 'EQUIV' and verdict == 'NO_DIFFERENCE_OBSERVED': assessment = 'NO_COUNTEREXAMPLE_OBSERVED'
    elif verdict in ('COMPILE_ERROR', 'RUNTIME_ERROR'): assessment = 'CURRENT_ERROR_OBSERVED'
    else: assessment = 'UNVERIFIED'
    report['label_assessment'] = assessment
    return report


def audit_provenance(root, output, *, all_records=False, workers=2, seeds=DEFAULT_SEEDS, repeats=2, timeout=30):
    if not 1 <= workers <= 8:
        raise ValueError('Use between one and eight workers')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    master_rows = read_rows(root / 'data/processed/MLIRBench/master/master.jsonl')
    historical_rows = read_rows(root / 'data/historical/historical_execution.jsonl')
    master = {r['program_id']: r for r in master_rows}
    historical = {r['program_id']: r for r in historical_rows}
    if len(master) != len(master_rows) or len(historical) != len(historical_rows):
        raise ValueError('Duplicate program IDs in provenance inputs')
    sources = verify_sources(root, master, historical)
    inventory = [inventory_record(r, historical[r['program_id']]) for r in sorted(master_rows, key=lambda r:r['program_id'])]
    from ..cli import select_sample
    selected = sorted(master_rows, key=lambda r:r['program_id']) if all_records else select_sample(master_rows)
    output.mkdir(parents=True)
    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    save('source_verification.json', sources)
    with (output/'inventory.jsonl').open('x') as stream:
        for record in inventory: stream.write(json.dumps(record, allow_nan=False)+'\n')
    tasks = [(r, historical[r['program_id']], tuple(seeds), repeats, timeout) for r in selected]
    reports = []
    def consume(iterator):
        with (output/'canonical_audit.jsonl').open('x') as stream:
            for report in iterator:
                reports.append(report)
                stream.write(json.dumps(report, allow_nan=False)+'\n'); stream.flush()
                print(f"Provenance audit {len(reports)}/{len(tasks)}: {report['program_id']} {report['label_assessment']}", flush=True)
    if workers == 1:
        consume(map(audit_case, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            consume(pool.map(audit_case, tasks))
    pid = 'seed1_linalg_prog7_L1_R2'
    candidate = replay_candidate(master[pid], historical[pid])
    save('linalg_historical_candidate.json', candidate)
    assessments = dict(Counter(r['label_assessment'] for r in reports))
    from ..ae import source_fingerprint
    summary = {'source_fingerprint': source_fingerprint(root), 'schema': 'mlirbench-provenance-audit-v1', 'policy': POLICY_VERSION,
               'validation_system_passed': sources['passed'] and len(reports) == len(selected),
               'source_verification': sources, 'inventory_records': len(inventory),
               'historical_metadata_domains': dict(Counter(r['metadata_input_check']['status'] for r in inventory)),
               'historical_outputs_missing': sum(not all(r['historical_outputs_present']) for r in inventory),
               'structural_candidates': dict(Counter(r['source_relation'] for r in inventory)),
               'sample_mode': 'all_published_pairs' if all_records else 'stratified_diagnostic_sample',
               'canonical_records': len(reports), 'seeds': list(seeds), 'repeats': repeats,
               'assessment_counts': assessments,
               'historical_candidate_status': candidate['status'],
               'published_label_provenance_status': 'REVIEW_REQUIRED',
               'limitations': [
                   'Actual per-program historical runtime inputs and original toolchain identity are not recorded in the available export.',
                   'Metadata input validity does not prove those values were used at runtime.',
                   'Independent-embedding replay is a historical-output hypothesis, not a semantic counterexample.',
                   'Finite equal trials do not prove equivalence; flat SSA-name identity is a structural observation requiring valid IR and well-defined execution.',
                   'Diagnostic sampling is not an estimate of the dataset error rate.' if not all_records else 'Full coverage is limited to the supported operation domains and tested inputs.',
               ], 'published_data_modified': False}
    save('summary.json', summary)
    lines = ['# Execution provenance audit', '',
             '**Validation system: ' + ('PASS' if summary['validation_system_passed'] else 'FAIL') + '**', '',
             '**Published label provenance: REVIEW_REQUIRED**', '',
             f"Verified {len(master)} published records against the original {sources['original_csv_rows']}-row Supabase CSV without modifying either.", '',
             'Actual historical runtime inputs are not recorded. Embedded-input metadata is checked independently and is never treated as proof of runtime use.', '',
             f"Canonical coverage: {len(reports)} pairs, {len(seeds)} seeds, {repeats} fresh-process repeats per program/input. Mode: {summary['sample_mode']}.", '',
             '| Program | Published label | Current observation | Assessment |', '|---|---|---|---|']
    lines += [f"| {r['program_id']} | {r['published_label']} | {r['verdict']} | {r['label_assessment']} |" for r in reports]
    lines += ['', '## Historical evidence and limitations', '',
              f"Linalg independent-embedding replay: {candidate['status']}. It does not establish the original runtime-input path.", '',
              'Metadata domain counts: ' + json.dumps(summary['historical_metadata_domains'], sort_keys=True), '',
              'Published labels and paper results are preserved. Conflicts and unsupported cases are explicit annotations, not automatic relabeling.', '']
    lines += ['- ' + text for text in summary['limitations']]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary
