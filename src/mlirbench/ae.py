"""Reviewer workflows and explicit readiness reporting."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
import unittest


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def cpu_execution():
    """Report architecture compatibility using the Docker daemon, not the client CPU."""
    aliases = {'aarch64': 'arm64', 'arm64': 'arm64', 'x86_64': 'amd64', 'amd64': 'amd64'}
    machine = platform.machine()
    host = os.environ.get('MLIRBENCH_DOCKER_HOST_ARCH')
    host_arch = aliases.get(host)
    container_arch = aliases.get(machine)
    return {'machine': machine, 'container_arch': container_arch,
            'docker_host_arch': host_arch,
            'image_arch': os.environ.get('MLIRBENCH_IMAGE_ARCH'),
            'requested_platform': os.environ.get('MLIRBENCH_DOCKER_PLATFORM'),
            'emulated': host_arch != container_arch if host_arch and container_arch else None,
            'logical_cpus': os.cpu_count(),
            'affinity_cpus': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None}


def check_environment():
    from .execution.pair_validation import environment, validate_pair
    report = environment()
    report['cpu_execution'] = cpu_execution()
    report['executables'] = {n: shutil.which(n) for n in ('mlir-opt', 'torch-mlir-opt', 'iree-compile', 'iree-benchmark-module')}
    programs = {
        'linalg': 'module { func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> { return %arg0 : tensor<2xf32> } }',
        'torch': 'module { func.func @main(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> { return %arg0 : !torch.vtensor<[2],f32> } }',
    }
    smoke = {d: validate_pair(s, s, d, seeds=[42]) for d, s in programs.items()}
    report['smoke'] = {d: r['verdict'] for d, r in smoke.items()}
    report['smoke_failures'] = {d: r for d, r in smoke.items() if r['verdict'] != 'NO_DIFFERENCE_OBSERVED'}
    report['passed'] = all(report['executables'].values()) and not report['smoke_failures']
    return report


def source_fingerprint(root):
    from .evaluation.reproduce import sha256
    files = sorted(p for folder in ('src', 'tests', 'tools', 'experiments') for p in (root / folder).rglob('*')
                   if p.is_file() and p.suffix in ('.py', '.json') and '__pycache__' not in p.parts)
    files += [root / name for name in ('Dockerfile', 'ae.sh', 'requirements/core.txt') if (root / name).is_file()]
    return hashlib.sha256(json.dumps({str(p.relative_to(root)): sha256(p) for p in files}, sort_keys=True).encode()).hexdigest()


def run_tests(root):
    import sys
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'src'))
    start = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(root / 'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return {'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
            'failures': len(result.failures), 'errors': len(result.errors),
            'skipped': len(result.skipped), 'seconds': round(time.monotonic() - start, 3), 'source_fingerprint': source_fingerprint(root)}


def verify_data(root):
    from .evaluation.reproduce import sha256
    manifest = json.loads((root / 'data/bundle/manifest.json').read_text())
    for name, spec in manifest.items():
        p = root / 'data/processed' / name
        if not p.is_file() or sha256(p) != spec['sha256']:
            raise ValueError(f'Missing or modified published data: {p}; run prepare-data in a fresh checkout')
    return {'passed': True, 'verified_files': len(manifest),
            'master_sha256': sha256(root / 'data/processed/MLIRBench/master/master.jsonl'),
            'bundle_manifest_sha256': sha256(root / 'data/bundle/manifest.json')}


def prepare_data(root):
    import subprocess
    import sys
    destination = root / 'data/processed'
    if not destination.exists():
        subprocess.run([sys.executable, str(root / 'tools/data/unpack_artifact_data.py')], cwd=root, check=True)
    report = verify_data(root)
    downstream = root / 'data/downstream/records'
    if not downstream.exists() and (root / 'data/downstream/snapshot.tar.gz').exists():
        subprocess.run([sys.executable, str(root / 'tools/data/unpack_downstream.py')], cwd=root, check=True)
    if (root / 'data/downstream/manifest.json').exists():
        from .evaluation.reproduce import sha256
        manifest = json.loads((root / 'data/downstream/manifest.json').read_text())
        for name, spec in manifest['files'].items():
            path = downstream / name
            if not path.is_file() or sha256(path) != spec['sha256']:
                raise ValueError(f'Missing or modified downstream input: {name}')
        report['verified_downstream_files'] = len(manifest['files'])
    return report


def readiness(root, results):
    results = Path(results)
    claims = json.loads((root / 'experiments/claims.json').read_text())
    gates = []
    for name, filename in [('published_data', 'data_integrity.json'), ('compiler_execution', 'environment.json'), ('automated_tests', 'tests.json')]:
        path = results / filename
        if path.exists():
            evidence = json.loads(path.read_text())
            passed = evidence.get('passed') is True
            if name == 'automated_tests':
                passed = passed and evidence.get('tests_run', 0) > 0 and evidence.get('skipped', 0) == 0
                passed = passed and evidence.get('source_fingerprint') == source_fingerprint(root)
            gates.append({'id': name, 'status': 'PASS' if passed else 'FAIL', 'evidence': str(path)})
        else:
            gates.append({'id': name, 'status': 'NOT_RUN', 'evidence': str(path)})
    environment_path = results / 'environment.json'
    if environment_path.exists():
        environment = json.loads(environment_path.read_text())
        cpu = environment.get('cpu_execution', {})
        native = cpu.get('emulated') is False and environment.get('passed') is True
        gates.append({'id': 'native_cpu_execution',
                      'status': 'PASS' if native else 'REVIEW_REQUIRED',
                      'cpu_execution': cpu,
                      'reason': 'Native execution verified.' if native else
                      'Use ./ae.sh on the Docker host architecture; direct Docker runs without host metadata cannot establish native execution.'})
    else:
        gates.append({'id': 'native_cpu_execution', 'status': 'NOT_RUN'})
    comparison = results / 'paper_metrics/paper_comparison.json'
    if comparison.exists():
        record = json.loads(comparison.read_text())
        passed = record['mismatched'] == 0 and record['reference_status'] == 'FINAL_ACCEPTED_MANUSCRIPT_CONFIRMED'
        gates.append({'id': 'paper_metric_agreement', 'status': 'PASS' if passed else 'REVIEW_REQUIRED',
                      'matched': record['matched'], 'mismatched': record['mismatched'],
                      'reference_status': record['reference_status']})
    else:
        gates.append({'id': 'paper_metric_agreement', 'status': 'NOT_RUN'})
    downstream = results / 'downstream_metrics/summary.json'
    if downstream.exists():
        record = json.loads(downstream.read_text())
        gates.append({'id': 'archived_downstream_replay', 'status': 'PASS' if record['mismatches'] == 0 and record.get('split_leakage_checks_failed', 0) == 0 and record.get('excluded_prediction_rows', 0) == 0 else 'REVIEW_REQUIRED',
                      'comparisons': record['comparisons'], 'mismatches': record['mismatches']})
    else:
        gates.append({'id': 'archived_downstream_replay', 'status': 'NOT_RUN'})
    for claim in claims['claims']:
        if claim['status'] != 'verified':
            gates.append({'id': claim['id'], 'status': 'OPEN', 'reason': claim['remaining']})
    return {'status': 'READY_FOR_SUBMISSION_REVIEW' if all(g['status'] == 'PASS' for g in gates) else 'NOT_SUBMISSION_READY',
            'gates': gates, 'note': 'Passing container tests does not establish paper-result or label validity.'}


def quickstart(root, output):
    from .evaluation.reproduce import reproduce
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    output.mkdir(parents=True)
    start = time.monotonic()
    print('[1/5] Verifying published data', flush=True)
    save(output / 'data_integrity.json', prepare_data(root))
    print('[2/5] Checking real Linalg and Torch execution', flush=True)
    environment = check_environment()
    save(output / 'environment.json', environment)
    print('[3/5] Running core tests', flush=True)
    tests = run_tests(root)
    save(output / 'tests.json', tests)
    print('[4/5] Recomputing archived paper metrics', flush=True)
    reproduce(root, output / 'paper_metrics')
    print('[5/5] Recomputing downstream prediction metrics', flush=True)
    if (root / 'data/downstream/records').exists():
        from .evaluation.downstream import reproduce_downstream
        reproduce_downstream(root, output / 'downstream_metrics')
    report = readiness(root, output)
    report['seconds'] = round(time.monotonic() - start, 3)
    save(output / 'readiness.json', report)
    lines = ['# Artifact readiness', '', f"**{report['status']}**", '',
             '| Check | Status |', '|---|---|']
    lines += [f"| {g['id']} | {g['status']} |" for g in report['gates']]
    lines += ['', '## Remaining actions', '']
    lines += [f"- {g['id']}: {g.get('reason', 'Inspect the associated comparison or evidence file.')}" for g in report['gates'] if g['status'] != 'PASS']
    (output / 'START_HERE.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'basic_checks_passed': environment['passed'] and tests['passed'],
                      'submission_readiness': report['status'], 'read_first': str(output / 'START_HERE.md')}, indent=2))
    return 0 if environment['passed'] and tests['passed'] else 1
