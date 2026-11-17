"""Compatibility adapter for the pipeline; canonical suite lives in pair_validation."""
from .pair_validation import validate_pair, deserialize_inputs
from .execution import _normalize_result
import base64
import numpy as np


def _argument_names(source):
    from .pair_validation import argument_names
    return argument_names(source)


def _decode(output):
    if output['kind'] == 'sequence': return [_decode(o) for o in output['items']]
    return _normalize_result(np.frombuffer(base64.b64decode(output['bytes_b64']), dtype=output['dtype']).reshape(output['shape']))


def execute_pair_shared_inputs(program_a, program_b, dialect, seed=None):
    kwargs = {'seeds': [seed]} if seed is not None else {}
    report = validate_pair(program_a, program_b, dialect, **kwargs)
    trial = None
    if report['verdict'] in {'DIVERGE', 'NO_DIFFERENCE_OBSERVED'}:
        trial = next((t for t in report['trials'] if t['status'] == report['verdict']), None)
    metadata = {'strategy': report['mode'], 'validation_report': report,
                'program_a': {}, 'program_b': {}}
    if trial:
        values = deserialize_inputs(trial['inputs'], report['input_specs'][0])
        for key, names in zip(('program_a', 'program_b'), report['argument_mapping']):
            metadata[key] = {n: v.tolist() for n, v in zip(names, values)}
        results = []
        for side in trial['runs']:
            result = side[0]
            results.append({'compilation_status': 'success', 'execution_status': 'success',
                            'output': _decode(result['output_typed']), 'error_message': None})
        return *results, metadata
    results = []
    for i in range(2):
        c = report.get('compilation', [{}, {}])[i]
        results.append({'compilation_status': 'success' if c.get('status') == 'success' else c.get('status', 'skipped'),
                        'execution_status': report['verdict'], 'output': None,
                        'error_message': c.get('error_message', report.get('error', report['verdict']))})
    return *results, metadata
