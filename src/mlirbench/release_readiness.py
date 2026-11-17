"""Read-only, evidence-linked release assessment; never changes scientific records."""
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

from .compiler_experiments import BACKEND_VERSION, PASSES, instructions, scan_rows, sha, torch_ops, truth
from .evaluation.numeric_agreement import paper_agreement_passes
from .evaluation.reproduce import LABELS

SCHEMA = 'mlirbench-release-evidence-v1'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare_release_data(root):
    """Verify frozen inputs and unpack local bundles using only the standard library."""
    import subprocess
    import sys
    root = Path(root).resolve()
    baseline = json.loads((root / 'experiments/release_invariants.json').read_text())
    for name, spec in baseline['frozen_inputs'].items():
        expected = spec if isinstance(spec, str) else spec['sha256']
        candidate = root / name
        require(candidate.resolve().is_relative_to(root) and file_hash(candidate) == expected,
                'Frozen reporter input differs: ' + name)
    for destination, script in [('data/processed', 'tools/data/unpack_artifact_data.py'),
                                ('data/downstream/records', 'tools/data/unpack_downstream.py')]:
        if not (root / destination).exists():
            subprocess.run([sys.executable, str(root / script)], cwd=root, check=True)
    from .ae import verify_data
    verify_data(root)
    downstream = root / 'data/downstream/records'
    manifest = json.loads((root / 'data/downstream/manifest.json').read_text())
    for name, spec in manifest['files'].items():
        candidate = downstream / name
        require(candidate.resolve().is_relative_to(downstream.resolve()) and file_hash(candidate) == spec['sha256'],
                'Downstream input differs: ' + name)


def index_evidence(archive, output, ci=None, run=None):
    """Pin supplied files, not assertions of success. The assessor rechecks contents."""
    archive, output = Path(archive).resolve(), Path(output).resolve()
    require(not output.exists(), 'Evidence index already exists')
    record = {'schema': SCHEMA, 'archive': {'path': os.path.relpath(archive, output.parent),
              'sha256': file_hash(archive)}, 'run': run, 'ci': None}
    if ci is not None:
        ci = Path(ci).resolve()
        files = {str(p.relative_to(ci)): file_hash(p) for p in sorted(ci.rglob('*.json'))}
        require(bool(files), 'CI directory has no evidence JSON files')
        record['ci'] = {'path': os.path.relpath(ci, output.parent), 'files': files}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    return record


class Archive:
    def __init__(self, path):
        self.zip = zipfile.ZipFile(path)
        names = self.zip.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP members')
        self.names = set(names)
        self.used = {}

    def read(self, name):
        raw = self.zip.read(name)
        self.used[name] = {'sha256': sha(raw), 'bytes': len(raw)}
        return raw

    def json(self, name):
        return json.loads(self.read(name))

    def rows(self, name):
        return [json.loads(line) for line in self.read(name).splitlines() if line.strip()]


def source_identity(archive, prefix):
    """Same definition as ae.source_fingerprint, over the tested source snapshot."""
    folders = ('src/', 'tests/', 'tools/', 'experiments/', 'optional/live_pipeline/')
    extras = ('Dockerfile', 'ae.sh', 'requirements/core.txt', 'requirements/live.txt',
              'requirements/backend.txt', 'requirements/runtime-lock.txt')
    names = []
    for member in archive.names:
        if not member.startswith(prefix):
            continue
        name = member[len(prefix):]
        if name in extras or (name.startswith(folders) and Path(name).suffix in ('.py', '.json')
                              and '__pycache__' not in Path(name).parts):
            names.append(name)
    require(all(name in names for name in extras), 'Incomplete tested source snapshot')
    return sha(json.dumps({n: sha(archive.read(prefix + n)) for n in sorted(names)}, sort_keys=True).encode())


def validate_tests(report, identity):
    require(report.get('passed') is True and type(report.get('tests_run')) is int
            and report['tests_run'] > 0 and report.get('source_fingerprint') == identity
            and all(type(report.get(k)) is int and report[k] == 0 for k in ('failures', 'errors', 'skipped')),
            'Tests are incomplete, failed, skipped, or for another source')


def validate_cpu(environment, arch):
    cpu = environment.get('cpu_execution', {})
    require(arch in ('amd64', 'arm64') and environment.get('passed') is True and cpu.get('emulated') is False
            and cpu.get('container_arch') == arch and cpu.get('docker_host_arch') == arch
            and cpu.get('requested_platform') == 'linux/' + arch,
            'Native execution on the claimed CPU is not established')


def validate_command(record):
    code = record.get('returncode')
    require(type(code) is int and code >= 0 and record.get('timeout') is False
            and record.get('success') is (code == 0), 'Inconsistent or timed-out compiler command')
    return code == 0


def compiler_evidence(root, archive, folder, dialect, riscv=False):
    summary = archive.json(folder + 'summary.json')
    rows, scan = scan_rows(root, dialect)
    selected = [r for r in rows if truth(r['fusion_success'])] if riscv else rows
    expected = {r['dt1_id']: r for r in selected}
    record_names = [n for n in archive.names if n.startswith(folder + 'records/') and n.endswith('.json')]
    require(all('/' not in n[len(folder + 'records/'):] for n in record_names), 'Unexpected nested compiler records')
    paths = {Path(n).stem: n for n in record_names}
    require(set(paths) == set(expected), 'Compiler records do not cover the exact complete corpus')
    require(summary.get('mode') == 'fresh_compiler_execution' and summary.get('dialect') == dialect
            and summary.get('full_corpus') is True and summary.get('programs') == len(expected)
            and summary.get('expected_programs') == len(rows) and summary.get('riscv_requested') is riscv
            and summary.get('driver_sha256') == file_hash(Path(root) / 'src/mlirbench/compiler_experiments.py')
            and summary.get('source_scan_sha256') == file_hash(scan)
            and BACKEND_VERSION in summary.get('toolchain', {}).get('iree-compile', {}).get('version', '')
            and summary.get('mismatches') == [] and summary.get('transformation_agreement') is True,
            'Wrong compiler scope, driver, source scan, toolchain, or reported mismatch')
    historical = {}
    if riscv:
        from .compiler_experiments import read_csv
        historical = {r['dt1_id']: r for r in read_csv(Path(root) /
            'data/downstream/records/DT1_ELEMENTWISE/outputs/riscv_backend_extension/riscv_backend_results.csv')}
        require(set(historical) == set(expected) and summary.get('riscv_mismatches') == [],
                'RISC-V coverage or summary mismatch')
    outcomes = collections.Counter()
    both_count = 0
    for pid, row in expected.items():
        actual = archive.json(paths[pid])
        require(actual.get('dt1_id') == pid and actual.get('mlir_hash') == row['mlir_hash']
                and actual.get('dialect') == dialect, 'Compiler record identity mismatch')
        program = folder + 'programs/' + pid + '/'
        require(sha(archive.read(program + 'source.mlir').decode().strip().encode()) == row['mlir_hash'],
                'Compiler input program hash mismatch')
        baseline = actual['baseline']
        accepted = validate_command(baseline)
        require(accepted == truth(row['baseline_success']), 'Baseline compilation differs from archive')
        experiments = {'fusion': ['--linalg-fuse-elementwise-ops']} if dialect == 'linalg' else PASSES
        passes = actual.get('passes', {})
        require(set(passes) == (set(experiments) if accepted else set()), 'Missing or unexpected compiler passes')
        for name in experiments:
            transformed = passes.get(name, {})
            success = validate_command(transformed) if accepted else False
            require(success == truth(row.get(name + '_success') or 'False'), 'Transformation status differs')
            if success:
                before, after = baseline['stdout'], transformed['stdout']
                normalized = lambda s: re.sub(r'\s+', ' ', s).strip()
                changed = before != after if dialect == 'linalg' else normalized(before) != normalized(after)
                counter = (lambda s: len(re.findall(r'\blinalg\.[A-Za-z0-9_]+', s))) if dialect == 'linalg' else torch_ops
                nb, na = counter(before), counter(after)
                require(transformed.get('changed_ir') is changed and transformed.get('ops_before') == nb
                        and transformed.get('ops_after') == na and transformed.get('ops_removed') == nb - na,
                        'Stored transformation counts disagree with actual IR')
                require(changed == truth(row[name + '_changed_ir']), 'Transformed IR comparison differs')
                if dialect == 'linalg':
                    require(nb == int(float(row['linalg_ops_before'])) and na == int(float(row['linalg_ops_after'])),
                            'Linalg operation counts differ')
                else:
                    require(nb - na == int(float(row[name + '_ops_removed'])), 'Torch operation counts differ')
        if riscv:
            pair = actual.get('riscv', {})
            require(set(pair) == {'baseline', 'fused'}, 'Missing RISC-V code-generation records')
            counts, successes = {}, {}
            for label in ('baseline', 'fused'):
                sub = program + label + '/'
                codegen = pair[label]
                success = validate_command(codegen)
                command = codegen.get('command', [])
                require('--iree-llvmcpu-target-triple=riscv64' in command, 'Not RISC-V code generation')
                ir = baseline['stdout'] if label == 'baseline' else passes['fusion']['stdout']
                require(archive.read(sub + 'input.mlir').decode() == ir, 'RISC-V input IR differs')
                assembly = sorted(n for n in archive.names if n.startswith(sub) and n.endswith('.s'))
                stream = [line for n in assembly for line in instructions(archive.read(n).decode(errors='replace'))]
                counts[label] = len(stream)
                successes[label] = success and len(assembly) > 0
                require(codegen.get('assembly_files') == len(assembly)
                        and codegen.get('instruction_count') == len(stream)
                        and codegen.get('instruction_stream_sha256') == (sha('\n'.join(stream).encode()) if stream else None),
                        'RISC-V assembly differs from recorded counts or fingerprints')
            both = all(successes.values())
            old = historical[pid]
            require(both == truth(old['both_compile']), 'RISC-V compilation outcome differs')
            if both:
                both_count += 1
                for label in counts:
                    require(counts[label] == int(float(old[label + '_instruction_count'])), 'RISC-V instruction count differs')
                outcome = 'IMPROVED' if counts['fused'] < counts['baseline'] else 'WORSENED' if counts['fused'] > counts['baseline'] else 'TIED'
                require(outcome == old['instruction_outcome'], 'RISC-V improvement outcome differs')
                outcomes[outcome] += 1
    return {'programs': len(expected), 'both_compile': both_count if riscv else None,
            'instruction_outcomes': dict(outcomes), 'scope': 'static_codegen' if riscv else 'full_scan',
            'driver_sha256': summary['driver_sha256']}


def preserved(root, archive, source, run):
    baseline_path = Path(root) / 'experiments/release_invariants.json'
    baseline = json.loads(baseline_path.read_text())
    report = archive.json(run + 'preservation.json')
    require(report.get('passed') is True and report.get('baseline_sha256') == file_hash(baseline_path),
            'Preservation baseline or outcome differs')
    recorded = {(r['kind'], r['path']): r for r in report.get('checks', [])}
    count = 0
    for kind, prefix, entries in [('frozen_input', source, baseline['frozen_inputs']),
                                   ('recomputed_output', run, baseline['recomputed_outputs'])]:
        for name, spec in entries.items():
            expected, mode = (spec, 'bytes') if isinstance(spec, str) else (spec['sha256'], spec['mode'])
            raw = archive.read(prefix + name)
            if mode == 'canonical_json':
                raw = json.dumps(json.loads(raw), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
            require(sha(raw) == expected, 'Frozen input or recalculated result differs: ' + name)
            item = recorded.get((kind, name), {})
            require(item.get('passed') is True and item.get('expected_sha256') == expected
                    and item.get('computed_sha256') == expected, 'Invalid preservation evidence: ' + name)
            count += 1
    require(len(recorded) == len(report['checks']) == count, 'Partial or duplicate preservation checks')
    return count


def normalize_downstream(report):
    result = dict(report)
    normalized = []
    for row in result['input_provenance']:
        marker = 'data/downstream/records/'
        require(marker in row['path'], 'Unknown downstream input provenance path')
        normalized.append({**row, 'path': row['path'].split(marker, 1)[1]})
    result['input_provenance'] = normalized
    return result


def validate_ci_preservation(root, report):
    baseline_path = Path(root) / 'experiments/release_invariants.json'
    baseline = json.loads(baseline_path.read_text())
    expected = {(kind, name): spec if isinstance(spec, str) else spec['sha256']
                for kind, entries in [('frozen_input', baseline['frozen_inputs']),
                                      ('recomputed_output', baseline['recomputed_outputs'])]
                for name, spec in entries.items()}
    checks = report.get('checks', [])
    actual = {(c['kind'], c['path']): c for c in checks}
    require(report.get('passed') is True and report.get('baseline_sha256') == file_hash(baseline_path)
            and len(actual) == len(checks) == len(expected) and set(actual) == set(expected),
            'CI preservation evidence is incomplete or uses another baseline')
    require(all(actual[k].get('passed') is True and actual[k].get('expected_sha256') == digest
                and actual[k].get('computed_sha256') == digest for k, digest in expected.items()),
            'CI preservation hashes disagree with the immutable baseline')


def assess_release(root, evidence_path):
    root, evidence_path = Path(root), Path(evidence_path).resolve()
    index = json.loads(evidence_path.read_text())
    require(index.get('schema') == SCHEMA, 'Unknown release evidence index')
    archive_path = (evidence_path.parent / index['archive']['path']).resolve()
    require(file_hash(archive_path) == index['archive']['sha256'], 'Evidence archive checksum mismatch')
    archive = Archive(archive_path)
    gates = []
    def gate(name, section, action):
        try:
            detail = action()
            gates.append({'id': name, 'section': section, 'status': 'PASS', 'details': detail})
        except (ValueError, KeyError, FileNotFoundError, OSError, TypeError, zipfile.BadZipFile) as exc:
            gates.append({'id': name, 'section': section, 'status': 'REVIEW_REQUIRED', 'reason': str(exc)})
    try:
        candidates = sorted(n[:-len('STATUS.tsv')] for n in archive.names
                            if '/MLIRBench-AE-runs/independent-' in n and n.endswith('/STATUS.tsv') and not n.startswith('__MACOSX/'))
        if index.get('run'):
            candidates = [n for n in candidates if Path(n.rstrip('/')).name == index['run']]
        require(len(candidates) == 1, 'Select exactly one independent run with --run')
        base = candidates[0]
        sources = [n[:-len('ae.sh')] for n in archive.names if n.endswith('/ae.sh')
                   and n[:-len('ae.sh')] + 'src/mlirbench/ae.py' in archive.names and not n.startswith('__MACOSX/')]
        require(len(sources) == 1, 'Missing or ambiguous tested source checkout')
        source = sources[0]
        identity = source_identity(archive, source)
        def quickstart(name):
            folder = base + name + '/'
            tests = archive.json(folder + 'tests.json')
            validate_tests(tests, identity)
            env = archive.json(folder + 'environment.json')
            validate_cpu(env, env['cpu_execution']['container_arch'])
            count = preserved(root, archive, source, folder)
            comparison = archive.json(folder + 'pipeline/metrics/paper_comparison.json')
            require(paper_agreement_passes(root, comparison), 'Paper metric agreement failed')
            return {'tested_source_fingerprint': identity, 'tests': tests['tests_run'],
                    'preservation_checks': count, 'native_cpu': env['cpu_execution']['container_arch'],
                    'paper_checks_accepted': comparison['numeric_agreement']['accepted']}
        gate('downloaded_image_workflow', 'software', lambda: quickstart('quickstart'))
        gate('source_build_workflow', 'software', lambda: quickstart('source-build-quickstart'))
        def lineage():
            checked = 0
            for folder in ('replay-mutation', 'replay-execution', 'replay-reasoning', 'replay-judge', 'replay-metrics'):
                prefix = base + folder + '/'
                manifest = archive.json(prefix + 'manifest.json')
                require(manifest.get('status') == 'COMPLETE' and manifest.get('source_fingerprint') == identity
                        and manifest.get('paid_api_calls_requested') is False, 'Incomplete or mismatched replay')
                for stage, spec in manifest['artifacts'].items():
                    raw = archive.read(prefix + stage + '.jsonl')
                    require(sha(raw) == spec['sha256'] and len(raw.splitlines()) == spec['records'], 'Stage hash or count differs')
                    require(raw == archive.read(base + 'quickstart/pipeline/' + stage + '.jsonl'), 'Independent replay differs from preservation-verified archived outputs')
                    checked += 1
            return {'checked_stage_files': checked}
        gate('independent_stage_replay', 'software', lineage)
        def paper_metrics():
            comparison = archive.json(base + 'replay-metrics/metrics/paper_comparison.json')
            require(paper_agreement_passes(root, comparison), 'Independent paper metrics failed the existing policy')
            return comparison['numeric_agreement']
        gate('paper_metric_agreement', 'metrics', paper_metrics)
        def repeatability():
            for dialect in ('linalg', 'torch'):
                left = archive.read(base + 'fresh-' + dialect + '-1/execution.jsonl')
                right = archive.read(base + 'fresh-' + dialect + '-2/execution.jsonl')
                require(left == right and bool(left.strip()), 'Fresh execution repetitions differ')
                row = json.loads(left)
                require(row['mode'] == 'fresh' and row['data']['ground_truth_label'] == 'EQUIV', 'Fresh demonstration did not agree')
            left = archive.rows(base + 'validation-sample-1.jsonl')
            right = archive.rows(base + 'validation-sample-2.jsonl')
            require(left == right and bool(left), 'Validation sample repetitions differ')
            return {'sample_pairs': len(left), 'scope': 'two dialect demonstrations and repeated diagnostic sample, not all pairs'}
        gate('deterministic_samples', 'software', repeatability)
        for name, dialect, rv in [('compiler-linalg-full', 'linalg', False), ('compiler-torch-full', 'torch', False), ('compiler-riscv-full', 'linalg', True)]:
            gate(name, 'compiler', lambda name=name, dialect=dialect, rv=rv: compiler_evidence(root, archive, base + name + '/', dialect, rv))
        compiler_gates = [g for g in gates if g['section'] == 'compiler']
        gates.append({'id': 'compiler_backend_experiments', 'section': 'compiler',
                      'status': 'PASS' if all(g['status'] == 'PASS' for g in compiler_gates) else 'REVIEW_REQUIRED',
                      'reason': 'Requires verified full Linalg, Torch and RISC-V records; static code generation is not a hardware timing experiment.'})
        def downstream():
            from .evaluation.downstream import reproduce_downstream
            with tempfile.TemporaryDirectory() as temp:
                reproduce_downstream(root, Path(temp) / 'downstream')
                computed = json.loads((Path(temp) / 'downstream/summary.json').read_text())
            recorded = archive.json(base + 'downstream-replay/summary.json')
            require(normalize_downstream(computed) == normalize_downstream(recorded) and computed['comparisons'] > 0 and computed['mismatches'] == 0
                    and computed['split_leakage_checks_failed'] == 0 and computed['excluded_prediction_rows'] == 0,
                    'Saved-prediction metrics differ from independent recalculation')
            return {'comparisons': computed['comparisons'], 'scope': 'saved predictions; no GPU inference or training'}
        gate('archived_downstream_replay', 'metrics', downstream)
        def labels():
            folder = base + 'full-provenance-audit/'
            summary = archive.json(folder + 'summary.json')
            rows = archive.rows(folder + 'canonical_audit.jsonl')
            master = {r['program_id']: r for r in (json.loads(l) for l in
                (root / 'data/processed/MLIRBench/master/master.jsonl').read_text().splitlines())}
            require(len(rows) == len(master) and {r['program_id'] for r in rows} == set(master)
                    and summary.get('canonical_records') == len(rows) and summary.get('source_fingerprint') == identity,
                    'Audit does not cover this complete dataset and tested source')
            for row in rows:
                record = master[row['program_id']]
                require(row['published_label'].replace('ERROR_OR_UNAVAILABLE', 'ERROR') == LABELS[str(record['judge__outputs_diverge'])]
                        and row['source_sha256'] == [sha(record[k].encode()) for k in ('input_mlir', 'mutated_mlir')],
                        'Audit program or published label differs')
            counts = dict(collections.Counter(r['label_assessment'] for r in rows))
            require(counts == summary['assessment_counts'], 'Audit counts disagree with records')
            return {'records': len(rows), 'assessment_counts': counts,
                    'nondeterministic_cases': [r['program_id'] for r in rows if any(t['status'] == 'NONDETERMINISTIC' for t in r.get('trials', []))],
                    'runtime_inputs_proven': summary['source_verification']['runtime_inputs_proven']}
        gate('full_label_audit_completed', 'software', labels)
        def disposition_documentation():
            from .label_dispositions import load_dossier
            dossier, _, matches = load_dossier(root)
            return {'reviewed_cases': len(dossier['cases']), 'status': dossier['status'],
                    'execution_code_matches_review': matches, 'document': 'docs/label-dispositions.md',
                    'historical_runtime_inputs_proven': False}
        gate('label_conflict_dispositions', 'documentation', disposition_documentation)
        audit = next(g for g in gates if g['id'] == 'full_label_audit_completed')
        gates.append({'id': 'execution_label_provenance', 'section': 'scientific', 'status': 'REVIEW_REQUIRED',
                      'details': audit.get('details', {}),
                      'reason': 'A completed audit and documented disposition do not certify historical labels. See docs/label-dispositions.md for the 30-case review. Original runtime inputs remain unrecorded; unsupported domains and remaining provenance gaps stay explicit.'})
        ci = index.get('ci')
        ci_files_verified = {}
        def ci_read(relative):
            require(ci is not None, 'No linked CI evidence; native coverage is not inferred')
            require(relative in ci['files'], 'Missing pinned CI report: ' + relative)
            directory = (evidence_path.parent / ci['path']).resolve()
            path = directory / relative
            require(file_hash(path) == ci['files'][relative], 'CI report checksum mismatch: ' + relative)
            ci_files_verified[relative] = ci['files'][relative]
            return json.loads(path.read_text())
        for arch in ('amd64', 'arm64'):
            def native(arch=arch):
                require(ci is not None, 'No linked CI evidence; native coverage is not inferred')
                publication = ci_read('published-image/published-image.json')
                digest = ci_read('image-evidence-' + arch + '/' + arch + '/digest.json')
                require(publication.get('source_fingerprint') == identity and digest.get('arch') == arch
                        and set(publication.get('platforms', {})) == {'amd64', 'arm64'}
                        and publication['platforms'].get(arch) == digest.get('digest')
                        and re.fullmatch(r'sha256:[a-f0-9]{64}', publication.get('index_digest', ''))
                        and re.fullmatch(r'sha256:[a-f0-9]{64}', digest.get('tested_config_digest', '')),
                        'Published image and tested platform identities disagree')
                for phase in ('image-evidence-', 'pulled-image-evidence-'):
                    prefix = phase + arch + '/' + arch + '/'
                    def read(name):
                        return ci_read(prefix + name)
                    tests, env, data, preservation = read('tests.json'), read('environment.json'), read('data_integrity.json'), read('preservation.json')
                    validate_tests(tests, identity)
                    require(tests['tests_run'] == archive.json(base + 'quickstart/tests.json')['tests_run'], 'CI test suite count differs for the same tested source')
                    validate_cpu(env, arch)
                    require(data.get('passed') is True and data.get('master_sha256') ==
                            file_hash(root / 'data/processed/MLIRBench/master/master.jsonl'), 'CI dataset differs')
                    validate_ci_preservation(root, preservation)
                    require(paper_agreement_passes(root, read('paper_comparison.json')), 'CI paper agreement failed')
                return {'tested_source_fingerprint': identity, 'arch': arch, 'phases': ['native build/test', 'fresh registry pull/test']}
            gate('native_' + arch, 'platform', native)
        for name, reason in [('gpu_checkpoint_evaluation', 'Saved predictions do not establish fresh GPU inference or training.'),
                             ('dependency_freeze', 'A successful source build does not freeze future apt repository resolution.')]:
            gates.append({'id': name, 'section': 'scientific' if name.startswith('gpu') else 'build_provenance', 'status': 'OPEN', 'reason': reason})
        gates.append({'id': 'judge_reliability', 'section': 'scientific', 'status': 'EXCLUDED_FROM_REQUIRED_SCOPE',
                      'required': False, 'reason': 'Author-excluded study remains unreproduced; no PASS is inferred.'})
        sections = {}
        for section in sorted({g['section'] for g in gates}):
            selected = [g for g in gates if g['section'] == section and g.get('required', True)]
            sections[section] = 'PASS' if selected and all(g['status'] == 'PASS' for g in selected) else 'REVIEW_REQUIRED'
        from .ae import source_fingerprint
        return {'schema': 'mlirbench-release-readiness-v1', 'status': 'READY_FOR_SUBMISSION_REVIEW' if all(g['status'] == 'PASS' for g in gates if g.get('required', True)) else 'NOT_SUBMISSION_READY',
                'sections': sections, 'gates': gates, 'tested_source_fingerprint': identity,
                'reporter_source_fingerprint': source_fingerprint(root),
                'evidence_index_sha256': file_hash(evidence_path), 'archive_sha256': index['archive']['sha256'],
                'selected_run': base, 'verified_archive_members': archive.used, 'verified_ci_files': ci_files_verified,
                'evidence_inputs': {'index': str(evidence_path), 'archive': str(archive_path),
                                    'ci_directory': str((evidence_path.parent / ci['path']).resolve()) if ci else None},
                'note': 'Read-only aggregation of pinned evidence. Checksums establish file consistency, not independent authenticity. Historical reports, labels, scoring, tolerances and conclusions are unchanged. Software evidence is not scientific certification.'}
    finally:
        archive.zip.close()


def write_report(report, output):
    output = Path(output)
    require(not output.exists(), 'Choose a new report directory')
    output.mkdir(parents=True)
    (output / 'readiness.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# Release evidence assessment', '', '**' + report['status'] + '**', '',
             'This consolidated report incorporates independent sibling runs. Original per-run readiness reports are preserved.', '',
             '| Area | Status |', '|---|---|']
    lines += ['| ' + name + ' | ' + status + ' |' for name, status in report['sections'].items()]
    lines += ['', '## Verified gates and remaining issues', '']
    for gate in report['gates']:
        detail = gate.get('details', {})
        description = gate.get('reason', '')
        if 'tests' in detail:
            description = f"{detail['tests']} tests; {detail['preservation_checks']} preservation checks; {detail['paper_checks_accepted']} accepted paper checks; native {detail['native_cpu']}."
        elif 'accepted' in detail:
            description = f"{detail['accepted']}/{detail['checks']} checks accepted: {detail['exact']} exact and {detail['within_tolerance']} within the existing declared policy."
        elif 'programs' in detail:
            description = f"{detail['programs']} complete-corpus records verified."
            if detail.get('both_compile') is not None:
                description += f" Both configurations compile for {detail['both_compile']}; static instruction outcomes: "
                description += ', '.join(f"{k.lower()} {v}" for k, v in sorted(detail['instruction_outcomes'].items())) + '.'
        elif 'sample_pairs' in detail:
            description = f"Two dialect demonstrations and a {detail['sample_pairs']}-pair sample agree across repetitions; full-corpus repeatability is not inferred."
        elif 'checked_stage_files' in detail:
            description = f"{detail['checked_stage_files']} stage files match their manifests and preservation-verified archived outputs."
        elif 'comparisons' in detail:
            description = f"{detail['comparisons']} downstream comparisons independently recalculated from saved predictions; no GPU inference or training."
        elif 'arch' in detail:
            description = 'Native ' + detail['arch'] + ': verified build/test and fresh registry-pull/test evidence.'
        elif 'reviewed_cases' in detail:
            description = f"{detail['reviewed_cases']} case dispositions verified against the dossier and immutable master; historical provenance remains unproven. See docs/label-dispositions.md."
        elif 'records' in detail and gate['id'] == 'full_label_audit_completed':
            description = f"All {detail['records']} published pairs audited; completion does not establish label validity."
        lines += ['- **' + gate['id'] + ': ' + gate['status'] + '**. ' + description]
        if gate['id'] == 'execution_label_provenance' and 'assessment_counts' in detail:
            lines += ['  Assessment counts: ' + '; '.join(f"{name}: {count}" for name, count in sorted(detail['assessment_counts'].items())) + '.']
            if detail.get('nondeterministic_cases'):
                lines += ['  Nondeterministic cases: ' + ', '.join('`' + n + '`' for n in detail['nondeterministic_cases']) + '.']
    lines += ['', '## Evidence links', '']
    for label, path in report.get('evidence_inputs', {}).items():
        if path:
            lines.append('- [' + label + '](' + os.path.relpath(path, output) + ')')
    lines += ['', 'Tested source: `' + report['tested_source_fingerprint'] + '`.',
              'Reporter source: `' + report['reporter_source_fingerprint'] + '`.', '',
              'These are separate identities: archived tests are not silently attributed to changed reporting code.', '', report['note']]
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
