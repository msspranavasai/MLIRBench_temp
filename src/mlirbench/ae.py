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
    files = sorted(p for folder in ('src', 'tests', 'tools', 'experiments', 'optional/live_pipeline') for p in (root / folder).rglob('*')
                   if p.is_file() and p.suffix in ('.py', '.json') and '__pycache__' not in p.parts)
    files += [root / name for name in ('Dockerfile', 'ae.sh', 'requirements/core.txt', 'requirements/live.txt', 'requirements/backend.txt', 'requirements/runtime-lock.txt') if (root / name).is_file()]
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
    from .execution.provenance import load_verified_sources
    _, _, report['historical_source_verification'] = load_verified_sources(root)
    return report


def readiness(root, results):
    results = Path(results)
    claims = json.loads((root / 'experiments/claims.json').read_text())
    gates = []
    for name, filename in [('published_data', 'data_integrity.json'), ('compiler_execution', 'environment.json'), ('automated_tests', 'tests.json'), ('paper_preservation', 'preservation.json'), ('archived_compiler_metrics','compiler_metrics.json')]:
        path = results / filename
        if path.exists():
            evidence = json.loads(path.read_text())
            passed = evidence.get('passed') is True
            if name == 'paper_preservation':
                from .preservation import verify_preservation
                current = verify_preservation(root,results)
                passed = passed and current['passed'] and evidence.get('baseline_sha256') == current['baseline_sha256']
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
    paper_status = 'NOT_RUN'
    comparison = results / 'paper_metrics/paper_comparison.json'
    if not comparison.exists(): comparison = results / 'pipeline/metrics/paper_comparison.json'
    if comparison.exists():
        record = json.loads(comparison.read_text())
        from .evaluation.numeric_agreement import paper_agreement_passes
        passed = paper_agreement_passes(root, record)
        paper_status = 'PASS' if passed else 'REVIEW_REQUIRED'
        gates.append({'id': 'paper_metric_agreement', 'status': 'PASS' if passed else 'REVIEW_REQUIRED',
                      'matched': record['matched'], 'mismatched': record['mismatched'],
                      'reference_status': record['reference_status'],
                      'numeric_agreement': record.get('numeric_agreement'),
                      'reason': 'Paper-score agreement under the declared policy; exact history and label validity are separate.'})
    else:
        gates.append({'id': 'paper_metric_agreement', 'status': 'NOT_RUN'})
    downstream = results / 'downstream_metrics/summary.json'
    if downstream.exists():
        record = json.loads(downstream.read_text())
        gates.append({'id': 'archived_downstream_replay', 'status': 'PASS' if record['mismatches'] == 0 and record.get('split_leakage_checks_failed', 0) == 0 and record.get('excluded_prediction_rows', 0) == 0 else 'REVIEW_REQUIRED',
                      'comparisons': record['comparisons'], 'mismatches': record['mismatches']})
    else:
        gates.append({'id': 'archived_downstream_replay', 'status': 'NOT_RUN'})
    pipeline_manifest = results/'pipeline/manifest.json'
    if pipeline_manifest.exists():
        from .stage_artifacts import load_run
        try:
            stages, pipeline_record = load_run(results/'pipeline')
            from .pipeline import verify_archived_executions
            verify_archived_executions(root, stages.get('execution', []),
                                       {r['program_id']:r['data'] for r in stages.get('mutation', [])})
            pipeline_passed = (pipeline_record.get('status')=='COMPLETE'
                               and pipeline_record.get('source_fingerprint')==source_fingerprint(root)
                               and pipeline_record.get('metrics',{}).get('scope')=='full_published_archive'
                               and paper_status=='PASS')
        except (ValueError,KeyError,FileNotFoundError):
            pipeline_passed = False
        gates.append({'id':'pipeline_replay','status':'PASS' if pipeline_passed else 'REVIEW_REQUIRED',
                      'reason':'Verified archived stage chain and independently recalculated score totals and paper metrics.'})
    provenance = results / 'execution_provenance/summary.json'
    if provenance.exists():
        record = json.loads(provenance.read_text())
        from .evaluation.reproduce import sha256
        from .execution.input_policy import POLICY_VERSION
        manifest_path = root / 'data/historical/source_manifest.json'
        passed = (record.get('validation_system_passed') is True
                  and record.get('canonical_records', 0) > 0
                  and record.get('source_fingerprint') == source_fingerprint(root)
                  and record.get('policy') == POLICY_VERSION
                  and record.get('repeats', 0) >= 2
                  and manifest_path.exists()
                  and record['source_verification'].get('source_manifest_sha256') == sha256(manifest_path))
        gates.append({'id': 'execution_validation_system', 'status': 'PASS' if passed else 'REVIEW_REQUIRED',
                      'canonical_records': record.get('canonical_records'),
                      'reason': 'Operation-aware shared-input auditing and original-source verification.'})
        label_passed = (passed and record['source_verification'].get('runtime_inputs_proven') is True
                        and record.get('sample_mode') == 'all_published_pairs'
                        and record.get('published_label_provenance_status') == 'PASS')
        gates.append({'id': 'execution_label_provenance', 'status': 'PASS' if label_passed else 'REVIEW_REQUIRED',
                      'assessment_counts': record.get('assessment_counts'),
                      'reason': 'See execution_provenance/REPORT.md. Actual historical runtime inputs are unrecorded; conflicts and unsupported cases remain explicit. Functional validation does not certify published labels.'})
    else:
        cpu_passed = True
        from .stage_artifacts import load_run
        for dialect in ('linalg','torch'):
            try:
                stages, cpu_record = load_run(results/'cpu_execution'/dialect)
                cpu_passed = cpu_passed and cpu_record.get('source_fingerprint')==source_fingerprint(root)
                cpu_passed = cpu_passed and len(stages['execution'])==1 and all(r['mode']=='fresh' and r['data'].get('ground_truth_label')=='EQUIV' for r in stages['execution'])
            except (ValueError,KeyError,FileNotFoundError): cpu_passed = False
        gates += [{'id':'execution_validation_system','status':'PASS' if cpu_passed else 'NOT_RUN',
                   'reason':'Real repeated Linalg and Torch execution demonstrations; full label audit remains separate.'},
                  {'id':'execution_label_provenance','status':'REVIEW_REQUIRED',
                   'reason':'Actual original runtime inputs/toolchain remain unavailable. This historical scientific question is separate from current pipeline functionality; see docs/execution-provenance.md.'}]
    for claim in claims['claims']:
        if claim['id'] == 'archived_model_metrics':
            gates.append({'id': claim['id'], 'status': paper_status,
                          'reason': 'Archived numeric metrics use the same declared agreement policy as paper_metric_agreement.'})
        elif claim['id'] == 'execution_label_provenance':
            continue
        elif claim.get('required',True) is False:
            gates.append({'id':claim['id'],'status':'EXCLUDED_FROM_REQUIRED_SCOPE',
                          'required':False,'reason':claim['remaining']})
        elif claim['status'] != 'verified':
            gates.append({'id': claim['id'], 'status': 'OPEN', 'reason': claim['remaining']})
    return {'status': 'READY_FOR_SUBMISSION_REVIEW' if all(g['status'] == 'PASS' for g in gates if g.get('required',True)) else 'NOT_SUBMISSION_READY',
            'scope': 'single_run', 'gates': gates, 'note': 'This report describes one run; use release-readiness to verify linked independent compiler and CI evidence. Passing container tests does not establish paper-result or label validity.'}


def quickstart(root, output):
    from .pipeline import run_pipeline
    output = Path(output)
    if output.exists(): raise FileExistsError(f'Use a new output directory: {output}')
    output.mkdir(parents=True)
    start = time.monotonic()
    print('[1/6] Verifying frozen published inputs', flush=True)
    save(output/'data_integrity.json',prepare_data(root))
    print('[2/6] Checking actual compiler/runtime execution', flush=True)
    environment = check_environment(); save(output/'environment.json',environment)
    print('[3/6] Testing stage integrity and compiler integration', flush=True)
    tests = run_tests(root); save(output/'tests.json',tests)
    print('[4/6] Replaying mutation -> execution -> reasoning -> judge -> computed paper metrics (no APIs)', flush=True)
    pipeline = run_pipeline(root,output/'pipeline')
    print('[5/6] Recomputing downstream metrics from saved predictions', flush=True)
    if (root/'data/downstream/records').exists():
        from .evaluation.downstream import reproduce_downstream
        reproduce_downstream(root,output/'downstream_metrics')
    from .compiler_experiments import archived_compiler_metrics
    compiler = archived_compiler_metrics(root)
    save(output/'compiler_metrics.json',compiler)
    print('[6/6] Rerunning the execution agent on one Linalg and one Torch pair', flush=True)
    cpu = {}
    for dialect,pid in [('linalg','seed1_linalg_prog1_L1_R1'),('torch','torch_seed1_prog10_T1_R1')]:
        cpu[dialect] = run_pipeline(root,output/'cpu_execution'/dialect,stage='execution',mode='fresh',ids=[pid])
    save(output/'pipeline_summary.json',{'replay':pipeline,'fresh_cpu':cpu})
    from .preservation import verify_preservation
    preservation = verify_preservation(root,output)
    save(output/'preservation.json',preservation)
    report = readiness(root,output)
    report['seconds'] = round(time.monotonic()-start,3)
    save(output/'readiness.json',report)
    passed = environment['passed'] and tests['passed'] and preservation['passed'] and compiler['passed'] and all(r['execution_labels']=={'EQUIV':1} for r in cpu.values())
    lines = ['# MLIRBench reviewer reproduction','',
             '**'+('Reproduction workflow completed' if passed else 'Functional checks failed')+'**','',
             'No paid model calls were made. Published program pairs, execution observations, model responses and subjective judge ratings were replayed. Objective label correctness, weighted score totals and aggregate paper metrics were recomputed from those records.','',
             '[Stage-by-stage replay and source identities](pipeline/REPORT.md)','',
             '[Computed paper tables and agreement](pipeline/metrics/REPORT.md)','',
             '[Preservation check against the pre-hardening baseline](preservation.json)','',
             '[Fresh Linalg execution](cpu_execution/linalg/REPORT.md) and [fresh Torch execution](cpu_execution/torch/REPORT.md) use the real compiler/runtime. They are independent demonstrations and do not replace the archived execution observations used for paper metrics.','',
             f"Tests: {tests['tests_run']}; failures: {tests['failures']}; errors: {tests['errors']}; skips: {tests['skipped']}.",'',
             'Artifacts retain actual inputs, parent/source hashes, score components and computed values. Readiness and unresolved scientific requirements are recorded separately in readiness.json. Historical runtime inputs remain unproven.','',
             'Use pipeline --stage mutation|execution|reasoning|judge|metrics for individual stages. Replay is the default. Fresh execution costs no API fees; fresh model stages require explicit opt-in and configured credentials.','']
    (output/'START_HERE.md').write_text('\n'.join(lines))
    print(json.dumps({'workflow_completed':passed,'paper_metrics':pipeline['metrics'],
                      'submission_readiness':report['status'],'read_first':str(output/'START_HERE.md')},indent=2))
    return 0 if passed else 1
