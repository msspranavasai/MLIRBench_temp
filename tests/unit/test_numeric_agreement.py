import copy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mlirbench.ae import readiness
from mlirbench.evaluation.numeric_agreement import assess_agreement, report_lines


class NumericAgreement(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((Path(__file__).resolve().parents[2] / 'experiments/numeric_agreement.json').read_text())

    def comparison(self):
        checks = []
        for model in range(9):
            for level in range(1, 6):
                value = str(Decimal(20) + 5 * model + Decimal(level) / 2)
                checks.append({'claim': f'm{model} L{level}',
                               'path': ['by_level', {'model': f'm{model}', 'level': level}, 'equal_dialect_score'],
                               'expected': value, 'actual': value, 'status': 'MATCH'})
        checks.append({'claim': 'pairs', 'path': ['dataset', 'pairs'],
                       'expected': 1768, 'actual': 1768, 'status': 'MATCH'})
        return {'reference_status': 'FINAL_ACCEPTED_MANUSCRIPT_CONFIRMED', 'source': {},
                'matched': 46, 'mismatched': 0, 'checks': checks}

    def changed(self, actual):
        record = self.comparison()
        record['checks'][0].update(actual=actual, status='MISMATCH')
        record.update(matched=45, mismatched=1)
        return record

    def test_inclusive_boundary_and_exact_history_are_preserved(self):
        record = self.changed('20.65')  # Expected 20.5; difference exactly 0.15.
        before = copy.deepcopy(record)
        result = assess_agreement(record, self.policy, 'digest')
        self.assertTrue(result['numeric_agreement']['passed'])
        self.assertEqual(result['numeric_agreement']['within_tolerance'], 1)
        self.assertEqual(result['numeric_agreement']['table4_orderings']['model_ranking_checks'], 180)
        self.assertEqual(result['numeric_agreement']['table4_orderings']['level_ordering_checks'], 90)
        self.assertEqual(result['checks'][0]['status'], 'MISMATCH')
        self.assertEqual(result['checks'][0]['agreement_status'], 'WITHIN_TOLERANCE')
        self.assertEqual(result['mismatched'], 1)
        self.assertEqual(record, before)

    def test_score_above_boundary_requires_review(self):
        result = assess_agreement(self.changed('20.66'), self.policy, 'digest')
        self.assertFalse(result['numeric_agreement']['passed'])
        self.assertEqual(result['numeric_agreement']['requires_review'], 1)

    def test_small_difference_cannot_pass_a_changed_ranking(self):
        record = self.changed('20.64')
        # Model m1 L1 was just above m0 L1 in the paper; the new value reverses it.
        record['checks'][5].update(expected='20.60', actual='20.60')
        result = assess_agreement(record, self.policy, 'digest')['numeric_agreement']
        self.assertEqual(result['requires_review'], 0)
        self.assertFalse(result['table4_orderings']['passed'])
        self.assertFalse(result['passed'])

    def test_missing_table_cells_cannot_pass(self):
        record = self.comparison()
        record['checks'].pop(0)
        self.assertFalse(assess_agreement(record, self.policy, 'digest')['numeric_agreement']['passed'])

    def test_counts_and_other_metrics_remain_exact(self):
        for check in [
            {'claim': 'pairs', 'path': ['dataset', 'pairs'], 'expected': 1768, 'actual': 1767, 'status': 'MISMATCH'},
            {'claim': 'behavior', 'path': ['summary', 'mean_behavior'], 'expected': '3.63', 'actual': '3.64', 'status': 'MISMATCH'},
        ]:
            with self.subTest(claim=check['claim']):
                policy = {**self.policy, 'absolute_score_tolerance': '1'}
                record = self.comparison(); record['checks'][-1] = check
                self.assertFalse(assess_agreement(record, policy, 'digest')['numeric_agreement']['passed'])

    def test_nonfinite_and_out_of_range_scores_do_not_pass(self):
        for value in ['NaN', 'Infinity', '-1', '101']:
            with self.subTest(value=value):
                self.assertFalse(assess_agreement(self.changed(value), self.policy, 'digest')['numeric_agreement']['passed'])
        invalid = {**self.policy, 'absolute_score_tolerance': 'NaN'}
        with self.assertRaises(ValueError):
            assess_agreement(self.comparison(), invalid, 'digest')

    def test_human_report_summarizes_small_differences_and_json_keeps_them(self):
        result = assess_agreement(self.changed('20.65'), self.policy, 'digest')
        text = '\n'.join(report_lines({'models': []}, [('primary', result)]))
        self.assertIn('PASS under the author-defined policy', text)
        self.assertIn('not a CGO-approved threshold', text)
        self.assertNotIn('expected 20.5; computed', text)
        self.assertIn('20.65', json.dumps(result))

    def test_readiness_uses_current_policy_and_keeps_unrelated_gates_open(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'experiments').mkdir()
            policy_path = root / 'experiments/numeric_agreement.json'
            policy_path.write_text(json.dumps(self.policy))
            (root / 'experiments/claims.json').write_text(json.dumps({'claims': [
                {'id': 'archived_model_metrics', 'status': 'runtime_checked', 'remaining': 'Numeric agreement.'},
                {'id': 'historical_labels', 'status': 'open', 'remaining': 'Historical provenance.'},
            ]}))
            output = root / 'results'; (output / 'paper_metrics').mkdir(parents=True)
            digest = hashlib.sha256(policy_path.read_bytes()).hexdigest()
            record = assess_agreement(self.changed('20.65'), self.policy, digest)
            (output / 'paper_metrics/paper_comparison.json').write_text(json.dumps(record))
            report = readiness(root, output)
            gates = {g['id']: g for g in report['gates']}
            self.assertEqual(gates['paper_metric_agreement']['status'], 'PASS')
            self.assertEqual(gates['archived_model_metrics']['status'], 'PASS')
            self.assertEqual(gates['historical_labels']['status'], 'OPEN')
            self.assertEqual(report['status'], 'NOT_SUBMISSION_READY')
            policy_path.write_text(json.dumps({**self.policy, 'absolute_score_tolerance': '0.14'}))
            gates = {g['id']: g for g in readiness(root, output)['gates']}
            self.assertEqual(gates['paper_metric_agreement']['status'], 'REVIEW_REQUIRED')
            self.assertEqual(gates['archived_model_metrics']['status'], 'REVIEW_REQUIRED')
