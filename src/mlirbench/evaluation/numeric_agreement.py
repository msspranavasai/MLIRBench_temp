"""Explicit paper-score agreement; independent of program-output comparison."""
from decimal import Decimal, InvalidOperation
from itertools import combinations


def number(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def table4_orderings(checks, policy):
    cells = {}
    for check in checks:
        path = check.get('evaluated_path', check['path'])
        if not path or path[0] != 'by_level':
            continue
        if len(path) != 3 or not isinstance(path[1], dict) or path[2] not in policy['table4_metrics']:
            return {'passed': False, 'reason': 'Unexpected Table 4 metric or selector.'}
        key = (path[1].get('model'), path[1].get('level'))
        values = (number(check['expected']), number(check['actual']))
        if key in cells or None in key or None in values:
            return {'passed': False, 'reason': 'Duplicate, missing or nonfinite Table 4 cell.'}
        cells[key] = values
    models = {key[0] for key in cells}
    levels = set(policy['expected_levels'])
    if (len(models) != policy['expected_model_count']
            or set(cells) != {(model, level) for model in models for level in levels}):
        return {'passed': False, 'reason': 'The complete expected model/level table is required.'}
    changes, ranking_checks, level_checks = [], 0, 0
    for (a, av), (b, bv) in combinations(sorted(cells.items()), 2):
        if a[0] != b[0] and a[1] != b[1]:
            continue
        kind = 'model_ranking' if a[1] == b[1] else 'level_ordering'
        ranking_checks += kind == 'model_ranking'
        level_checks += kind == 'level_ordering'
        old, new = av[0] - bv[0], av[1] - bv[1]
        if (old > 0) - (old < 0) != (new > 0) - (new < 0):
            changes.append({'kind': kind, 'a': list(a), 'b': list(b),
                            'expected_difference': str(old), 'actual_difference': str(new)})
    return {'passed': not changes, 'model_ranking_checks': ranking_checks,
            'level_ordering_checks': level_checks, 'changed_orderings': changes,
            'scope': 'Table 4 pairwise orderings only; not validation of every paper claim.'}


def assess_agreement(comparison, policy, policy_sha256):
    tolerance = number(policy['absolute_score_tolerance'])
    minimum, maximum = number(policy['score_min']), number(policy['score_max'])
    if tolerance is None or minimum is None or maximum is None or not 0 <= tolerance <= maximum - minimum:
        raise ValueError('Invalid absolute score tolerance or score range')
    checks, deltas = [], []
    for original in comparison['checks']:
        check = dict(original)
        path = check.get('evaluated_path', check['path'])
        scoped = (len(path) == 3 and path[0] == 'by_level'
                  and isinstance(path[1], dict) and path[2] in policy['table4_metrics'])
        expected, actual = number(check['expected']), number(check['actual'])
        delta = abs(actual - expected) if expected is not None and actual is not None else None
        in_range = (expected is not None and actual is not None
                    and minimum <= expected <= maximum and minimum <= actual <= maximum)
        exact = check['actual'] == check['expected']
        accepted = in_range and delta <= tolerance if scoped else exact
        if scoped and delta is not None:
            deltas.append(delta)
        check.update(absolute_difference=str(delta) if delta is not None else None,
                     allowed_absolute_difference=str(tolerance) if scoped else '0',
                     agreement_status=('EXACT' if exact and accepted else 'WITHIN_TOLERANCE'
                                       if accepted else 'REVIEW_REQUIRED'))
        checks.append(check)
    stability = table4_orderings(checks, policy)
    accepted = sum(c['agreement_status'] != 'REVIEW_REQUIRED' for c in checks)
    result = dict(comparison)
    result['checks'] = checks
    result['numeric_agreement'] = {
        'schema': policy['schema'], 'policy_sha256': policy_sha256,
        'passed': accepted == len(checks) and (stability['passed'] or not policy['requires_unchanged_table4_orderings']),
        'checks': len(checks), 'accepted': accepted,
        'exact': sum(c['agreement_status'] == 'EXACT' for c in checks),
        'within_tolerance': sum(c['agreement_status'] == 'WITHIN_TOLERANCE' for c in checks),
        'requires_review': len(checks) - accepted,
        'absolute_table4_tolerance': str(tolerance),
        'maximum_table4_difference': str(max(deltas)) if deltas else None,
        'table4_orderings': stability,
        'comparison_basis': policy['comparison_basis'], 'authority': policy['authority'],
        'strict_checks': policy['strict_checks'],
    }
    return result


def paper_agreement_passes(root, record):
    """Recheck policy and evidence rather than trusting a saved PASS flag."""
    import json
    from .reproduce import sha256
    path = root / 'experiments/numeric_agreement.json'
    if not path.exists() or record['reference_status'] != 'FINAL_ACCEPTED_MANUSCRIPT_CONFIRMED':
        return False
    digest = sha256(path)
    if record.get('numeric_agreement', {}).get('policy_sha256') != digest:
        return False
    return assess_agreement(record, json.loads(path.read_text()), digest)['numeric_agreement']['passed']


def report_lines(result, comparisons, snapshot_note=None):
    primary = comparisons[0][1]
    agreement = primary['numeric_agreement']
    status = 'PASS' if agreement['passed'] else 'REVIEW_REQUIRED'
    lines = ['# Archived-results reproduction', '',
             f"Paper reference: **{primary['reference_status']}**.", '',
             f'Numerical agreement: **{status} under the author-defined policy**.', '',
             f"Checks accepted: **{agreement['accepted']} / {agreement['checks']}** "
             f"({agreement['exact']} exact; {agreement['within_tolerance']} within tolerance).", '',
             '**Numerical-agreement disclaimer:** Table 4 scores are compared with an absolute '
             f"tolerance of **{agreement['absolute_table4_tolerance']} points on a 0–100 scale**, "
             'at the paper’s printed precision. Small differences can arise from dialect weighting, '
             'rounding and historical dataset snapshots. Their exact historical reconstruction '
             'remains incomplete. This author-defined tolerance is not a CGO-approved threshold '
             'or a badge decision. Detailed exact differences remain in the comparison JSON files.', '',
             'All other paper checks remain exact. Program-output equality, labels, IDs, input '
             'validity and dataset integrity are unaffected. This report replays archived judge '
             'observations and does not independently validate historical semantic labels.', '',
             '## Computed model summaries', '',
             '| Model | Available / expected | Label accuracy (%) | Recorded aggregate | Result /10 | Explanation /4 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in result['models']:
        lines.append(f"| {r['model']} | {r['available_scores']} / {r['expected_pairs']} | {r['hard_label_accuracy_percent']:.2f} | {r['aggregate_score_available']:.2f} | {r['result_behavior_score']:.2f} | {r['explanation_quality_score']:.2f} |")
    lines += ['', '## Agreement across reporting conventions', '',
              '| Convention | Accepted | Exact | Within tolerance | Requires review | Maximum Table 4 difference | Orderings |',
              '|---|---:|---:|---:|---:|---:|---|']
    for name, comparison in comparisons:
        a = comparison['numeric_agreement']
        lines.append(f"| {name} | {a['accepted']}/{a['checks']} | {a['exact']} | {a['within_tolerance']} | {a['requires_review']} | {a['maximum_table4_difference']} | {'unchanged' if a['table4_orderings']['passed'] else 'review required'} |")
    stability = agreement['table4_orderings']
    lines += ['', f"Primary comparison checked {stability.get('model_ranking_checks', 0)} model-ranking "
              f"and {stability.get('level_ordering_checks', 0)} mutation-level ordering relationships. "
              'These checks cover Table 4 only. The primary calculation retains equal dialect '
              'weighting with rounding at the end; Gemini pooling and intermediate rounding are '
              'alternative diagnostics, not changes to the source scores.', '', '## Results requiring review', '']
    failures = [f"- {c['claim']}: expected {c['expected']}; computed {c['actual']}."
                for c in primary['checks'] if c['agreement_status'] == 'REVIEW_REQUIRED']
    lines += failures or ['None under the declared numerical policy. Individual within-tolerance differences are retained in JSON.']
    if not stability['passed']:
        lines += ['', 'Table 4 orderings require review; see numeric_agreement.table4_orderings in paper_comparison.json.']
    if snapshot_note is not None:
        lines += ['', '## Historical snapshot note', '',
                  'The author reports 149 historical Linalg Level 2 pairs versus 148 in the '
                  f"artifact after exclusion of `{snapshot_note['excluded_program_id']}`; Torch has "
                  '160 Level 2 pairs. The excluded pair’s eight original judge scores have not '
                  'been recovered, so the exact numerical effect is unconfirmed. This explanation '
                  'is preserved in experiments/level2_dataset_change.json. Numeric agreement '
                  'does not imply that the missing historical evidence has been recovered.']
    return lines
