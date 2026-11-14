"""CPU recomputation of DT1/DT2 figures from original saved predictions."""
import csv
import json
import math
from pathlib import Path
from .reproduce import sha256, write_csv

BENEFIT = ('LOW', 'MEDIUM', 'HIGH')
SEMANTIC = ('EQUIV', 'DIVERGE', 'ERROR')


def normalize(value, labels):
    text = str(value).strip().upper()
    aliases = {'A': 'LOW', 'B': 'MEDIUM', 'C': 'HIGH'} if labels == BENEFIT else {'EQUIVALENT': 'EQUIV', 'DIVERGENT': 'DIVERGE'}
    return aliases.get(text, text)


def classification_metrics(records, labels):
    if not records:
        raise ValueError('No valid prediction records')
    true = [r['_gold'] for r in records]
    pred = [r['_pred'] for r in records]
    n = len(records)
    scores, recalls = [], []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(true, pred))
        fp = sum(g != label and p == label for g, p in zip(true, pred))
        fn = sum(g == label and p != label for g, p in zip(true, pred))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
        if tp + fn:
            recalls.append(tp / (tp + fn))
    result = {'n': n, 'accuracy': sum(g == p for g, p in zip(true, pred)) / n,
              'balanced_accuracy': math.fsum(recalls) / len(recalls),
              'macro_f1': math.fsum(scores) / len(scores)}
    result.update({f'f1_{label.lower()}': score for label, score in zip(labels, scores)})
    if labels == SEMANTIC:
        unsafe = sum(g in ('DIVERGE', 'ERROR') for g in true)
        result['false_equiv_rate'] = sum(g in ('DIVERGE', 'ERROR') and p == 'EQUIV' for g, p in zip(true, pred)) / unsafe if unsafe else None
    return result


def load_predictions(path, labels, provenance):
    csv.field_size_limit(20_000_000)
    with path.open() as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        gold_options = ('fusion_benefit', 'simplification_benefit', 'gold', 'gold_label', 'target', 'label', 'true_label', 'semantic_label', 'expected_label')
        pred_options = ('prediction', 'predicted_label', 'pred_label', 'predicted', 'prediction_label', 'model_prediction')
        gold = next((c for c in gold_options if c in columns), None)
        pred = next((c for c in pred_options if c in columns), None)
        if gold is None or pred is None:
            raise ValueError(f'Missing explicit gold/prediction columns: {path}')
        result, rejected = [], 0
        for row in reader:
            g, p = normalize(row[gold], labels), normalize(row[pred], labels)
            if g not in labels or p not in labels:
                rejected += 1
                continue  # Match archived figure script; report every exclusion explicitly.
            result.append({**row, '_gold': g, '_pred': p})
    provenance.append({'path': str(path), 'sha256': sha256(path), 'valid_rows': len(result), 'excluded_rows': rejected})
    return result


def pooled(root, method, provenance):
    result, ids = [], set()
    for fold in range(5):
        directory = root / 'outputs/final_cv' / method
        choices = (directory / f'fold_{fold}_val_predictions.csv', directory / f'fold_{fold}/val_predictions.csv')
        path = next((p for p in choices if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f'Missing {method} fold {fold}')
        heldout = json.loads((root / 'data/folds' / f'fold_{fold}/val.json').read_text())
        gold = {r.get('dt1_id') or r.get('program_id'): normalize(r.get('fusion_benefit', r.get('simplification_benefit')), BENEFIT) for r in heldout}
        predicted = load_predictions(path, BENEFIT, provenance)
        identifiers = [r.get('dt1_id') or r.get('program_id') for r in predicted]
        if len(gold) != len(heldout) or set(identifiers) != set(gold):
            raise ValueError(f'DT1 {method} fold {fold} predictions differ from the original held-out split')
        if any(r['_gold'] != gold[r.get('dt1_id') or r.get('program_id')] for r in predicted):
            raise ValueError(f'DT1 {method} fold {fold} gold labels disagree with the original split')
        for row in predicted:
            identifier = row.get('dt1_id') or row.get('program_id')
            if not identifier or identifier in ids:
                raise ValueError(f'Missing/duplicate held-out prediction ID: {path}/{identifier}')
            ids.add(identifier)
            result.append({**row, '_fold': fold})
    return result


def mean_metrics(records):
    keys = set.intersection(*(set(r) for r in records)) - {'n'}
    return {k: math.fsum(r[k] for r in records) / len(records) for k in keys if all(r[k] is not None for r in records)}


def split_audit(root):
    checks = []
    for task in ('DT1_ELEMENTWISE', 'DT1_TORCH_OPT'):
        for fold in range(5):
            base = root / task / 'data/folds' / f'fold_{fold}'
            train = json.loads((base / 'train.json').read_text())
            val = json.loads((base / 'val.json').read_text())
            train_families = {r['base_family'] for r in train}
            val_families = {r['base_family'] for r in val}
            checks.append({'task': task, 'fold': fold, 'train_rows': len(train), 'validation_rows': len(val),
                           'overlapping_families': sorted(train_families & val_families),
                           'status': 'PASS' if not train_families & val_families else 'LEAKAGE_FOUND'})
    split_files = {name: json.loads((root / 'DT2_SPLITS' / f'{name}.json').read_text()) for name in ('train', 'val', 'test')}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        a = {r['family_id'] for r in split_files[left]}
        b = {r['family_id'] for r in split_files[right]}
        checks.append({'task': 'DT2', 'partitions': [left, right],
                       'overlapping_families': sorted(a & b), 'status': 'PASS' if not a & b else 'LEAKAGE_FOUND'})
    return checks


def compare_csv(expected_path, actual, keys):
    with expected_path.open() as stream:
        expected = list(csv.DictReader(stream))
    comparisons = []
    for row in expected:
        matches = [r for r in actual if all(str(r[k]) == row[k] for k in keys)]
        if len(matches) != 1:
            raise ValueError(f'Cannot identify reference row {row}')
        result = matches[0]
        for field, value in row.items():
            if field in keys or value == '':
                continue
            got, want = float(result[field]), float(value)
            # Representation agreement at 12 decimal places, far finer than the paper's one-decimal percentages.
            matched = round(got, 12) == round(want, 12)
            comparisons.append({'row': {k: row[k] for k in keys}, 'metric': field,
                                'expected': want, 'actual': got, 'status': 'MATCH' if matched else 'MISMATCH'})
    return comparisons


def reproduce_downstream(root, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    records = Path(root) / 'data/downstream/records'
    provenance, dt1, dt2, dialects, classes = [], [], [], [], []
    for task, directory, seeds, sft_name in [
        ('Linalg fusion', 'DT1_ELEMENTWISE', (42, 123), 'SFT'),
        ('Torch simplification', 'DT1_TORCH_OPT', (42, 123, 2026), 'Balanced SFT')]:
        base = records / directory
        zero = classification_metrics(pooled(base, 'zero', provenance), BENEFIT)
        few = mean_metrics([classification_metrics(pooled(base, f'three_shot_seed_{seed}', provenance), BENEFIT) for seed in seeds])
        sft = classification_metrics(pooled(base, 'sft', provenance), BENEFIT)
        dt1.extend([{'task': task, 'method': 'Zero-shot', **zero},
                    {'task': task, 'method': 'Few-shot mean', 'n': None, **few},
                    {'task': task, 'method': sft_name, **sft}])
    base = records / 'outputs/dt2_all-dataset/predictions'
    files = {'Zero-shot': ['all-dataset_zero_shot_predictions.csv'],
             '6-shot mean': ['all-dataset_6shot_seed_42_predictions.csv', 'all-dataset_6shot_seed_123_predictions.csv'],
             'SFT': ['all-dataset_sft_test_predictions.csv']}
    original_test = {r['program_id']: r['label'] for r in json.loads((records / 'DT2_SPLITS/test.json').read_text())}
    for method, names in files.items():
        attempts = [load_predictions(base / name, SEMANTIC, provenance) for name in names]
        for attempt in attempts:
            ids = [r['program_id'] for r in attempt]
            if len(set(ids)) != len(ids) or set(ids) != set(original_test):
                raise ValueError(f'DT2 {method} predictions differ from the original held-out test set')
            if any(r['_gold'] != original_test[r['program_id']] for r in attempt):
                raise ValueError(f'DT2 {method} gold labels disagree with the original test split')
        metrics = [classification_metrics(r, SEMANTIC) for r in attempts]
        combined = metrics[0] if len(metrics) == 1 else {'n': int(round(sum(m['n'] for m in metrics) / len(metrics))), **mean_metrics(metrics)}
        dt2.append({'method': method, **combined})
        for dialect in ('linalg', 'torch'):
            scores = []
            for attempt in attempts:
                subset = [r for r in attempt if str(r.get('dialect', r.get('mlir_dialect', ''))).lower().strip() == dialect]
                scores.append(classification_metrics(subset, SEMANTIC)['macro_f1'])
            dialects.append({'dialect': dialect, 'method': method, 'macro_f1': math.fsum(scores) / len(scores)})
        for label in SEMANTIC:
            classes.append({'method': method, 'class': label, 'f1': combined[f'f1_{label.lower()}']})
    comparisons = []
    for rel, actual, keys in [('DT1/dt1_metrics_summary.csv', dt1, ('task', 'method')),
                              ('DT2/dt2_metrics_summary.csv', dt2, ('method',)),
                              ('DT2/dt2_dialect_metrics.csv', dialects, ('dialect', 'method')),
                              ('DT2/dt2_class_metrics.csv', classes, ('method', 'class'))]:
        comparisons.extend(compare_csv(records / 'FINAL_FIGURES' / rel, actual, keys))
    splits = split_audit(records)
    summary = {'mode': 'saved_prediction_metric_replay', 'comparisons': len(comparisons),
               'mismatches': sum(c['status'] != 'MATCH' for c in comparisons),
               'split_checks': splits, 'split_leakage_checks_failed': sum(s['status'] != 'PASS' for s in splits),
               'excluded_prediction_rows': sum(p['excluded_rows'] for p in provenance),
               'gpu_inference_rerun': False, 'fine_tuning_rerun': False,
               'dt1': dt1, 'dt2': dt2, 'by_dialect': dialects, 'by_class': classes,
               'comparison_details': comparisons, 'input_provenance': provenance}
    output.mkdir(parents=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    for name, data in [('dt1_metrics', dt1), ('dt2_metrics', dt2), ('dt2_dialect', dialects), ('dt2_class', classes)]:
        write_csv(output / f'{name}.csv', data)
    lines = ['# Downstream archived-prediction reproduction', '',
             f"Comparisons with archived final-figure CSVs: {len(comparisons) - summary['mismatches']} match / {summary['mismatches']} differ.",
             f"Family-split checks (DT1 + DT2): {len(splits) - summary['split_leakage_checks_failed']} pass / {summary['split_leakage_checks_failed']} overlap.",
             f"Invalid-label rows excluded under the original figure protocol: {summary['excluded_prediction_rows']}.",
             '', 'Metrics were recomputed from per-example predictions. Model inference and fine-tuning were not rerun.', '',
             '| Task / method | n | Accuracy (%) | Macro-F1 (%) |', '|---|---:|---:|---:|']
    for r in dt1 + [{'task': 'DT2', **r} for r in dt2]:
        lines.append(f"| {r['task']} / {r['method']} | {r['n'] or 'seed mean'} | {100*r['accuracy']:.2f} | {100*r['macro_f1']:.2f} |")
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return {'output': str(output), 'comparisons': len(comparisons), 'mismatches': summary['mismatches'],
            'split_leakage_checks_failed': summary['split_leakage_checks_failed']}
