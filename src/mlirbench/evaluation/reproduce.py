"""Recompute paper summaries from immutable archived records, without APIs."""
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
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
                    target.append({'model': model, field: value, 'expected_pairs': expected_count,
                                   'available_scores': len(group), 'aggregate_score': average(group, 'total_score'),
                                   'hard_label_accuracy_percent': 100 * sum(r['predicted_label'] == r['ground_truth_label'] for r in group) / expected_count,
                                   'result_behavior_score': average(group, 'result_accuracy'),
                                   'explanation_quality_score': average(group, 'explanation_rating')})
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


def compare_reference(result, reference):
    checks = []
    for spec in reference['checks']:
        value = result
        for part in spec['path']:
            if isinstance(part, dict):
                matches = [r for r in value if all(r.get(k) == v for k, v in part.items())]
                if len(matches) != 1:
                    raise ValueError(f'Ambiguous reference selector: {part}')
                value = matches[0]
            else:
                value = value[part]
        actual = rounded(value, spec['digits']) if 'digits' in spec else value
        checks.append({**spec, 'actual': actual, 'status': 'MATCH' if actual == spec['expected'] else 'MISMATCH'})
    return {'reference_status': reference['status'], 'source': reference['source'],
            'matched': sum(c['status'] == 'MATCH' for c in checks),
            'mismatched': sum(c['status'] == 'MISMATCH' for c in checks), 'checks': checks}


def write_csv(path, records):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def reproduce(root, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    data = Path(root) / 'data/processed/MLIRBench'
    inputs = {'master': data / 'master/master.jsonl', 'evaluations': data / 'evaluation/evaluation.jsonl',
              'reasoning': data / 'reasoning/reasoning.jsonl'}
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
    output.mkdir(parents=True)
    (output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (output / 'paper_comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')
    for name in ('models', 'by_level', 'by_dialect', 'confusion'):
        write_csv(output / f'{name}.csv', result[name])
    lines = ['# Archived-results reproduction', '', f"Paper reference: **{comparison['reference_status']}**.",
             f"Checks: **{comparison['matched']} match / {comparison['mismatched']} differ**.", '',
             'Metric aggregation uses archived responses and judge observations; it does not validate historical semantic labels.', '',
             '| Model | Available / expected | Label accuracy (%) | Recorded aggregate | Result /10 | Explanation /4 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in result['models']:
        lines.append(f"| {r['model']} | {r['available_scores']} / {r['expected_pairs']} | {r['hard_label_accuracy_percent']:.2f} | {r['aggregate_score_available']:.2f} | {r['result_behavior_score']:.2f} | {r['explanation_quality_score']:.2f} |")
    lines += ['', '## Reference differences', '']
    lines += [f"- {r['claim']}: expected {r['expected']}; computed {r['actual']}." for r in comparison['checks'] if r['status'] != 'MATCH'] or ['None at the stated rounding precision.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return {'output': str(output), 'reference_status': comparison['reference_status'],
            'matched': comparison['matched'], 'mismatched': comparison['mismatched']}
