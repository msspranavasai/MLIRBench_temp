"""Deterministic summaries of recorded evaluations; never reassigns ground truth."""
from collections import defaultdict
import math


def summarize(master, evaluations):
    labels = {'True': 'DIVERGE', 'False': 'EQUIV', 'N/A': 'ERROR'}
    published = {r['program_id']: labels[str(r['judge__outputs_diverge'])] for r in master}
    grouped = defaultdict(dict)
    for r in evaluations:
        pid, model = r['program_id'], r['evaluated_model']
        if pid not in published: raise ValueError(f'Evaluation ID absent from master: {pid}')
        if pid in grouped[model]: raise ValueError(f'Duplicate evaluation: {model}/{pid}')
        grouped[model][pid] = r
    reports = []
    for model, rows in sorted(grouped.items()):
        recorded_matches, published_matches, total_score = 0, 0, 0.
        disagreements = []
        for pid, r in rows.items():
            predicted, ground = r.get('predicted_label'), r.get('ground_truth_label')
            recorded_matches += predicted == ground and predicted in {'EQUIV', 'DIVERGE', 'ERROR'}
            published_matches += predicted == published[pid]
            if ground != published[pid]: disagreements.append(pid)
            score = float(r['total_score'])
            if not math.isfinite(score): raise ValueError(f'Nonfinite recorded score: {model}/{pid}')
            total_score += score
        n, expected = len(rows), len(published)
        reports.append({'model': model, 'expected_pairs': expected, 'recorded_rows': n,
                        'missing_program_ids': sorted(set(published) - rows.keys()),
                        'ground_truth_disagrees_with_master': sorted(disagreements),
                        'label_matches_recorded_ground_truth': recorded_matches,
                        'label_matches_published_master': published_matches,
                        'hard_label_accuracy_percent_all_pairs': 100 * published_matches / expected,
                        'hard_label_accuracy_percent_available_rows': 100 * published_matches / n,
                        'mean_recorded_judge_score_available_rows': total_score / n,
                        'mean_recorded_judge_score_missing_as_zero': total_score / expected})
    return {'mode': 'recorded_evaluation_summary', 'denominator_policy': 'all published pairs; missing predictions incorrect/zero',
            'judge_scores': 'summarized from stored values, not recomputed model judgments', 'models': reports}
