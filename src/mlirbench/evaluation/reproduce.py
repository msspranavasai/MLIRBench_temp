"""Recompute paper summaries from immutable archived records, without APIs."""
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP, localcontext
import csv
import hashlib
import json
import math
from pathlib import Path
import re

LABELS = {'True': 'DIVERGE', 'False': 'EQUIV', 'N/A': 'ERROR'}


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rounded(value, digits=2):
    return str(Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def average(records, field):
    values = [float(r[field]) for r in records]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError(f'Missing/nonfinite {field}')
    return math.fsum(values) / len(values)


def rounded_dialect_mean_score(dialect_groups):
    """Diagnostic: round each dialect mean first; preserve the decimal midpoint.

    Return a string so JSON/CSV serialization does not reintroduce a binary
    floating-point approximation before the final printed-value rounding.
    This is a hypothesis about table formatting, not a confirmed historical rule.
    """
    if set(dialect_groups) != {'linalg', 'torch'}:
        raise ValueError('Expected Linalg and Torch groups')
    if not all(dialect_groups.values()):
        return None
    with localcontext() as context:
        context.prec = 50
        means = []
        for records in dialect_groups.values():
            values = [Decimal(str(r['total_score'])) for r in records]
            if not all(v.is_finite() for v in values):
                raise ValueError('Nonfinite total_score')
            mean = sum(values, Decimal(0)) / len(values)
            means.append(mean.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        return str(sum(means, Decimal(0)) / 2)


def summarize(master_rows, evaluation_rows):
    master, models = {}, defaultdict(dict)
    for r in master_rows:
        pid = r['program_id']
        if pid in master or str(r['judge__outputs_diverge']) not in LABELS:
            raise ValueError(f'Duplicate ID or unknown published label: {pid}')
        master[pid] = r
    for r in evaluation_rows:
        pid, model = r['program_id'], r['evaluated_model']
        if pid not in master or pid in models[model]:
            raise ValueError(f'Unknown/duplicate evaluation: {model}/{pid}')
        if r['ground_truth_label'] != LABELS[str(master[pid]['judge__outputs_diverge'])]:
            raise ValueError(f'Ground truth disagrees with master: {model}/{pid}')
        models[model][pid] = r
    if not master or not models:
        raise ValueError('Master and evaluation records must not be empty')
    totals, levels, dialects, confusion = [], [], [], []
    for model, mapping in sorted(models.items()):
        available = list(mapping.values())
        correct = sum(r['predicted_label'] == r['ground_truth_label'] for r in available)
        totals.append({'model': model, 'expected_pairs': len(master), 'available_scores': len(available),
                       'missing_predictions': len(master) - len(available), 'hard_label_correct': correct,
                       'hard_label_accuracy_percent': 100 * correct / len(master),
                       'aggregate_score_available': average(available, 'total_score'),
                       'aggregate_score_missing_as_zero': math.fsum(float(r['total_score']) for r in available) / len(master),
                       'result_behavior_score': average(available, 'result_accuracy'),
                       'explanation_quality_score': average(available, 'explanation_rating'),
                       'missing_program_ids': sorted(set(master) - mapping.keys())})
        for field, target in (('level', levels), ('dialect', dialects)):
            for value in sorted({r[field] for r in master.values()}):
                group = [r for r in available if master[r['program_id']][field] == value]
                expected_count = sum(r[field] == value for r in master.values())
                if group:
                    entry = {
                        'model': model,
                        field: value,
                        'expected_pairs': expected_count,
                        'available_scores': len(group),
                        'aggregate_score': average(group, 'total_score'),
                        'hard_label_accuracy_percent': (
                            100 * sum(
                                r['predicted_label'] == r['ground_truth_label']
                                for r in group
                            ) / expected_count
                        ),
                        'result_behavior_score': average(group, 'result_accuracy'),
                        'explanation_quality_score': average(group, 'explanation_rating'),
                    }

                    if field == 'level':
                        dialect_groups = {
                            dialect: [
                                r for r in group
                                if master[r['program_id']]['dialect'] == dialect
                            ]
                            for dialect in ('linalg', 'torch')
                        }

                        # Each dialect contributes 50%, regardless of record count.
                        # Missing an entire dialect makes this average unavailable.
                        entry['equal_dialect_score'] = (
                            math.fsum(
                                average(records, 'total_score')
                                for records in dialect_groups.values()
                            ) / 2
                            if all(dialect_groups.values())
                            else None
                        )
                        entry['equal_dialect_score_rounded_means'] = rounded_dialect_mean_score(dialect_groups)

                    target.append(entry)
        for label in ('EQUIV', 'DIVERGE', 'ERROR'):
            for predicted in ('EQUIV', 'DIVERGE', 'ERROR'):
                confusion.append({'model': model, 'ground_truth': label, 'prediction': predicted,
                                  'count': sum(r['ground_truth_label'] == label and r['predicted_label'] == predicted for r in available)})
    sources = {(r['seed'], int(re.search(r'_prog(\d+)', r['program_id'])[1])) for r in master.values()}
    dataset = {'pairs': len(master), 'source_programs': len(sources),
               'dialects': dict(Counter(r['dialect'] for r in master.values())),
               'published_labels': dict(Counter(str(r['judge__outputs_diverge']) for r in master.values())),
               'statuses': dict(Counter(str(r['pairs__status']) for r in master.values())),
               'mutation_rules': {d: len({r['requirement_id'] for r in master.values() if r['dialect'] == d}) for d in ('linalg', 'torch')}}
    summary = {'model_count': len(totals), 'evaluation_rows': sum(len(m) for m in models.values()),
               'reasoning_attempts_expected': len(master) * len(totals)}
    for source, output in [('aggregate_score_available', 'mean_aggregate_score_across_models'),
                           ('hard_label_accuracy_percent', 'mean_hard_label_accuracy_percent'),
                           ('result_behavior_score', 'mean_result_behavior_score'),
                           ('explanation_quality_score', 'mean_explanation_quality_score')]:
        summary[output] = math.fsum(r[source] for r in totals) / len(totals)
    all_scores = [r for mapping in models.values() for r in mapping.values()]
    summary['mean_aggregate_score_available_rows'] = average(all_scores, 'total_score')
    return {'dataset': dataset, 'summary': summary, 'models': totals, 'by_level': levels,
            'by_dialect': dialects, 'confusion': confusion}


def compare_reference(result, reference, *, use_submitted_gemini_weighting=False,
                      use_intermediate_rounding=False):
    checks = []
    for spec in reference['checks']:
        path = spec['path']
        # Table 4's submitted Gemini row pooled scored LinAlg and Torch pairs.
        # Use that documented rule only in the separate submitted-weighting check.
        if (
            use_submitted_gemini_weighting
            and len(path) == 3
            and path[0] == 'by_level'
            and isinstance(path[1], dict)
            and path[1].get('model') == 'gemini-3.1-pro-preview'
            and path[2] == 'equal_dialect_score'
        ):
            path = [*path[:-1], 'aggregate_score']
        if (use_intermediate_rounding and len(path) == 3
                and path[0] == 'by_level' and path[2] == 'equal_dialect_score'):
            path = [*path[:-1], 'equal_dialect_score_rounded_means']
        value = result
        for part in path:
            if isinstance(part, dict):
                matches = [r for r in value if all(r.get(k) == v for k, v in part.items())]
                if len(matches) != 1:
                    raise ValueError(f'Ambiguous reference selector: {part}')
                value = matches[0]
            else:
                value = value[part]
        actual = rounded(value, spec['digits']) if 'digits' in spec else value
        checks.append({**spec, 'evaluated_path': path, 'actual': actual,
                       'status': 'MATCH' if actual == spec['expected'] else 'MISMATCH'})
    return {'reference_status': reference['status'], 'source': reference['source'],
            'matched': sum(c['status'] == 'MATCH' for c in checks),
            'mismatched': sum(c['status'] == 'MISMATCH' for c in checks), 'checks': checks}


def write_csv(path, records):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def reproduce(root, output, *, input_paths=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    data = Path(root) / 'data/processed/MLIRBench'
    inputs = {'master': data / 'master/master.jsonl', 'evaluations': data / 'evaluation/evaluation.jsonl',
              'reasoning': data / 'reasoning/reasoning.jsonl'}
    if input_paths is not None:
        inputs = {key: Path(input_paths[key]) for key in ('master','evaluations','reasoning')}
    result = summarize(rows(inputs['master']), rows(inputs['evaluations']))
    seen, attempts = set(), Counter()
    for r in rows(inputs['reasoning']):
        key = (r['program_id'], r['model_key'])
        if key in seen:
            raise ValueError(f'Duplicate reasoning attempt: {key}')
        seen.add(key)
        attempts[r['model_key']] += 1
    result['reasoning_records'] = {'total': len(seen), 'per_model': dict(sorted(attempts.items()))}
    result['input_sha256'] = {name: sha256(path) for name, path in inputs.items()}
    result['score_contract'] = {'hard_labels': 'all published pairs; missing attempts incorrect',
                                'aggregate_behavior_explanation': 'available recorded judge scores; denominators emitted',
                                'judge_calls': 'none; subjective judge observations are archived inputs',
                                'labels': 'published labels, not a newly validated semantic dataset'}
    reference = json.loads((Path(root) / 'experiments/paper_reference.json').read_text())
    comparison = compare_reference(result, reference)
    submitted_weighting = compare_reference(
        result, reference, use_submitted_gemini_weighting=True
    )
    rounding_hypothesis = compare_reference(
        result, reference, use_submitted_gemini_weighting=True,
        use_intermediate_rounding=True
    )
    rounding_hypothesis['hypothesis'] = (
        'Gemini uses pooled archived scores; other level cells equally average '
        'the two dialect means rounded first to two decimals using decimal '
        'ROUND_HALF_UP. Historical table-generation procedure remains unconfirmed.'
    )
    from .numeric_agreement import assess_agreement, report_lines
    policy_path = Path(root) / 'experiments/numeric_agreement.json'
    policy = json.loads(policy_path.read_text())
    policy_digest = sha256(policy_path)
    comparison = assess_agreement(comparison, policy, policy_digest)
    submitted_weighting = assess_agreement(submitted_weighting, policy, policy_digest)
    rounding_hypothesis = assess_agreement(rounding_hypothesis, policy, policy_digest)
    output.mkdir(parents=True)
    (output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (output / 'paper_comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')
    (output / 'paper_comparison_submitted_weighting.json').write_text(
        json.dumps(submitted_weighting, indent=2, allow_nan=False) + '\n'
    )
    (output / 'paper_comparison_rounding_hypothesis.json').write_text(
        json.dumps(rounding_hypothesis, indent=2, allow_nan=False) + '\n'
    )
    for name in ('models', 'by_level', 'by_dialect', 'confusion'):
        write_csv(output / f'{name}.csv', result[name])
    note_path = Path(root) / 'experiments/level2_dataset_change.json'
    snapshot_note = json.loads(note_path.read_text()) if note_path.exists() else None
    lines = report_lines(result, [
        ('Equal dialect weighting', comparison),
        ('Submitted Gemini weighting', submitted_weighting),
        ('Intermediate rounding hypothesis', rounding_hypothesis),
    ], snapshot_note)
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    agreement = comparison['numeric_agreement']
    return {'output': str(output), 'reference_status': comparison['reference_status'],
            'agreement_status': 'PASS' if agreement['passed'] else 'REVIEW_REQUIRED',
            'accepted': agreement['accepted'], 'checks': agreement['checks'],
            'exact': agreement['exact'], 'within_tolerance': agreement['within_tolerance'],
            'requires_review': agreement['requires_review']}
