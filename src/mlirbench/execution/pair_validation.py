"""Canonical shared-input validation and recorded-input replay; no dataset writes."""
import base64
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import numpy as np
from .execution import compile_mlir, run_compiled, parse_input_spec, _compile_extra_args
from .input_policy import (POLICY_VERSION, InputPolicyError, generate_input_case, check_inputs)

COMPARISON = 'typed-exact-value-v1'  # signed zeros equal; NaNs/nonfinite unverified
DEFAULT_SEEDS = (0, 1, 42, 123, 2026)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def argument_names(source):
    match = re.search(r'func.func\s+@main\s*\(([^)]*)\)', source)
    if not match:
        raise ValueError('Missing @main')
    return re.findall(r'(%[\w.$-]+)\s*:', match[1])


def environment():
    packages = {}
    for p in ('numpy', 'iree-compiler', 'iree-runtime', 'torch-mlir'):
        try: packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: packages[p] = None
    root = Path(__file__).parent
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'machine': platform.machine(), 'packages': packages,
            'compiler_flags': _compile_extra_args('llvm-cpu'),
            'code_sha256': {n: hashlib.sha256((root / n).read_bytes()).hexdigest()
                            for n in ('execution.py', 'input_policy.py', 'pair_validation.py')}}


def serialize_inputs(values):
    return [{'position': i, 'dtype': v.dtype.str, 'shape': list(v.shape),
             'data': v.tolist(), 'bytes_b64': base64.b64encode(v.tobytes()).decode('ascii')}
            for i, v in enumerate(values)]


def deserialize_inputs(items, specs):
    if not isinstance(items, list) or len(items) != len(specs):
        raise InputPolicyError('INVALID_INPUT', 'Expected one typed input per argument position')
    values = []
    for i, (item, spec) in enumerate(zip(items, specs)):
        if not isinstance(item, dict) or not {'dtype', 'shape'} <= item.keys():
            raise InputPolicyError('INVALID_INPUT', f'Missing typed input fields at {i}')
        try:
            dtype = np.dtype(item['dtype'])
        except (ValueError, TypeError) as exc:
            raise InputPolicyError('INVALID_INPUT', str(exc)) from exc
        if item.get('position', i) != i or dtype != np.dtype(spec['dtype']) or item['shape'] != spec['shape']:
            raise InputPolicyError('INVALID_INPUT', f'Input type/shape mismatch at {i}')
        try:
            if 'bytes_b64' in item:
                v = np.frombuffer(base64.b64decode(item['bytes_b64'], validate=True), dtype=item['dtype']).reshape(item['shape']).copy()
                if 'data' in item and not np.array_equal(v, np.asarray(item['data'], dtype=item['dtype']).reshape(item['shape'])):
                    raise ValueError('data disagrees with bytes')
            else:
                raw = np.asarray(item['data'])
                v = np.asarray(item['data'], dtype=item['dtype']).reshape(item['shape'])
                if dtype.kind in 'iub' and not np.array_equal(raw.reshape(item['shape']), v):
                    raise ValueError('Integer/bool data cannot be silently truncated')
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise InputPolicyError('INVALID_INPUT', str(exc)) from exc
        values.append(v)
    return values


def compare_typed(a, b):
    """Returns EQUAL, DIFFERENT, or NONFINITE; no dtype coercion/tolerance."""
    def finite(x):
        if x['kind'] == 'sequence': return all(finite(v) for v in x['items'])
        return bool(np.all(np.isfinite(array(x))))
    def array(x):
        return np.frombuffer(base64.b64decode(x['bytes_b64']), dtype=x['dtype']).reshape(x['shape'])
    def equal(x, y):
        if x['kind'] != y['kind']: return False
        if x['kind'] == 'sequence':
            return len(x['items']) == len(y['items']) and all(equal(i, j) for i, j in zip(x['items'], y['items']))
        return x['dtype'] == y['dtype'] and x['shape'] == y['shape'] and bool(np.array_equal(array(x), array(y)))
    if not finite(a) or not finite(b): return 'NONFINITE'
    return 'EQUAL' if equal(a, b) else 'DIFFERENT'


def published_label(record):
    return {'true': 'DIVERGE', 'false': 'EQUIV', 'n/a': 'ERROR_OR_UNAVAILABLE'}.get(str(record.get('judge__outputs_diverge')).lower(), 'UNKNOWN')


def validate_pair(a, b, dialect, seeds=DEFAULT_SEEDS, recorded_inputs=None,
                  repeats=2, timeout=30):
    """Compile once per program; preflight and repeat each shared input in fresh processes.

    recorded_inputs is a positional typed list (single trial) or a list of such lists.
    Equal finite outputs mean NO_DIFFERENCE_OBSERVED, never semantic equivalence.
    """
    if repeats < 2: raise ValueError('At least two repeats required to check determinism')
    if timeout <= 0: raise ValueError('timeout must be positive')
    seeds = tuple(seeds)
    if any(not isinstance(s, int) or s < 0 for s in seeds): raise ValueError('Seeds must be nonnegative integers')
    report = {'schema': 'mlirbench-pair-validation-v1', 'policy': POLICY_VERSION,
              'comparison': COMPARISON, 'mode': 'recorded_input_replay' if recorded_inputs is not None else 'canonical_validation',
              'source_sha256': [hashlib.sha256(s.encode()).hexdigest() for s in (a, b)],
              'environment': environment(), 'trials': [], 'verdict': 'UNVERIFIED'}
    try:
        specs, other = parse_input_spec(a), parse_input_spec(b)
        names = [argument_names(s) for s in (a, b)]
        report['input_specs'] = [specs, other]
        report['argument_mapping'] = names
        if specs != other:
            report['verdict'] = 'INCOMPARABLE_INPUT_SIGNATURES'; return report
        if any(len(n) != len(specs) for n in names): raise ValueError('Incomplete argument mapping')
    except ValueError as exc:
        report.update(verdict='UNSUPPORTED_INPUT_SIGNATURE', error=str(exc)); return report
    if recorded_inputs is not None:
        suites = recorded_inputs if recorded_inputs and isinstance(recorded_inputs[0], list) else [recorded_inputs]
        cases = [(None, items) for items in suites]
    else:
        cases = [(seed, None) for seed in seeds]
    if not cases: raise ValueError('Input suite must not be empty')
    # Compile even when domain analysis is unavailable, so invalid mutations can be audited.
    compiled = [compile_mlir(s, dialect=dialect) for s in (a, b)]
    report['compilation'] = [{k: v for k, v in c.items() if k != 'compiled_module'} for c in compiled]
    if any(c['status'] != 'success' for c in compiled):
        report['verdict'] = 'COMPILE_TIMEOUT' if any(c['status'] == 'timeout' for c in compiled) else 'COMPILE_ERROR'
        return report
    report['compiled_sha256'] = [hashlib.sha256(c['compiled_module']).hexdigest() for c in compiled]
    for seed, items in cases:
        trial = {'seed': seed, 'status': 'UNVERIFIED'}
        report['trials'].append(trial)
        try:
            if items is not None:
                values = deserialize_inputs(items, specs)
            else:
                values, trial['generation'] = generate_input_case(a, b, specs, *names, seed, dialect=dialect)
            trial['inputs'] = serialize_inputs(values)
            trial['input_sha256'] = digest(trial['inputs'])
            for source, ns in zip((a, b), names): check_inputs(source, ns, values, dialect)
        except InputPolicyError as exc:
            trial.update(status=exc.status, error=str(exc)); continue
        results = []
        for source, ns, c in zip((a, b), names, compiled):
            # Independent fresh processes; do not reuse runtime state across repetitions.
            runs = [run_compiled(c['compiled_module'], mlir_source=source, input_values=dict(zip(ns, values)), timeout_seconds=timeout) for _ in range(repeats)]
            # Typed output carries values losslessly; omit nonstandard JSON NaN/Inf in legacy output.
            results.append([{k: v for k, v in r.items() if k != 'output'} for r in runs])
        trial['runs'] = results
        flat = [r for side in results for r in side]
        if any(r['status'] != 'success' for r in flat):
            trial['status'] = 'RUNTIME_TIMEOUT' if any(r['status'] == 'timeout' for r in flat) else 'RUNTIME_ERROR'; continue
        trial['output_sha256'] = [[digest(r['output_typed']) for r in side] for side in results]
        # Byte fingerprints detect instability including sign of zero, independently of value comparison.
        if any(len(set(hashes)) > 1 for hashes in trial['output_sha256']):
            trial['status'] = 'NONDETERMINISTIC'; continue
        comparison = compare_typed(results[0][0]['output_typed'], results[1][0]['output_typed'])
        trial['status'] = {'EQUAL': 'NO_DIFFERENCE_OBSERVED', 'DIFFERENT': 'DIVERGE', 'NONFINITE': 'NONFINITE_OUTPUT'}[comparison]
    statuses = [t['status'] for t in report['trials']]
    if 'DIVERGE' in statuses: report['verdict'] = 'DIVERGE'
    elif all(s == 'NO_DIFFERENCE_OBSERVED' for s in statuses): report['verdict'] = 'NO_DIFFERENCE_OBSERVED'
    elif len(set(statuses)) == 1: report['verdict'] = statuses[0]
    else: report['verdict'] = 'UNVERIFIED'
    return report
