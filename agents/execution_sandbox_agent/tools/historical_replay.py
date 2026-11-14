"""Historical hypothesis replay, deliberately separate from semantic validation."""
import hashlib
import json
from .execution import compile_and_run_mlir, compare_outputs, parse_input_spec
from .mlir_embed import embed_inputs
from .pair_validation import argument_names, environment, serialize_inputs, compare_typed
from .input_policy import check_inputs, InputPolicyError
import numpy as np


def replay_candidate(record, evidence):
    report = {'program_id': record['program_id'], 'mode': 'independent_embedding_hypothesis',
              'semantic_verdict': 'NOT_APPLICABLE', 'environment': environment(),
              'evidence': evidence, 'status': 'UNVERIFIED', 'programs': []}
    sources = [record[k] for k in ('input_mlir', 'mutated_mlir')]
    if [hashlib.sha256(s.encode()).hexdigest() for s in sources] != evidence['source_sha256']:
        raise ValueError('Evidence source mismatch')
    if record['dialect'] != 'linalg':
        report.update(status='UNSUPPORTED_HISTORICAL_PATH', detail='Torch historical random inputs may be invalid; unsafe execution is not replayed')
        return report
    matches = []
    for source, key in zip(sources, ('judge__programA_output_json', 'judge__programB_output_json')):
        names, specs = argument_names(source), parse_input_spec(source)
        _, embedded = embed_inputs(source)
        values = [np.asarray(embedded[n], dtype=s['dtype']) for n, s in zip(names, specs)]
        trial = {'inputs': serialize_inputs(values)}; report['programs'].append(trial)
        try: check_inputs(source, names, values, 'linalg')
        except InputPolicyError as exc:
            report.update(status=exc.status, detail=str(exc)); return report
        runs = [compile_and_run_mlir(source, dialect='linalg', input_values=dict(zip(names, values))) for _ in range(2)]
        trial['runs'] = [{k: v for k, v in r.items() if k != 'compiled_module'} for r in runs]
        if any(r['execution_status'] != 'success' for r in runs):
            report['status'] = 'EXECUTION_ERROR'; return report
        if any(compare_typed(r['output_typed'], r['output_typed']) == 'NONFINITE' for r in runs):
            # Typed bytes retain these outputs without nonstandard JSON NaN/Infinity.
            for r in trial['runs']: r.pop('output', None)
            report['status'] = 'NONFINITE_OUTPUT'; return report
        expected = json.loads(evidence['historical_fields'][key])
        matches.append(all(not compare_outputs(r['output'], expected)['diverges'] for r in runs))
    report['matches_original_outputs'] = matches
    report['inputs_equal'] = report['programs'][0]['inputs'] == report['programs'][1]['inputs']
    report['status'] = 'HISTORICAL_OUTPUTS_REPRODUCED' if all(matches) else 'HISTORICAL_OUTPUTS_NOT_REPRODUCED'
    return report
