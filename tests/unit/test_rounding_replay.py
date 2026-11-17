import copy
import unittest

from mlirbench.evaluation.reproduce import (
    compare_reference, rounded, rounded_dialect_mean_score,
)


class IntermediateRounding(unittest.TestCase):
    def score(self, linalg, torch):
        return rounded_dialect_mean_score({
            'linalg': [{'total_score': value} for value in linalg],
            'torch': [{'total_score': value} for value in torch],
        })

    def test_each_dialect_has_half_weight_despite_unequal_counts(self):
        self.assertEqual(self.score([10, 10], [90]), '50.00')
        self.assertEqual(rounded(self.score([62.7687027], [67.47685])), '65.13')

    def test_decimal_midpoints_do_not_become_float_artifacts(self):
        self.assertEqual(self.score([19.949135], [22.395375]), '21.175')
        self.assertEqual(rounded(self.score([19.949135], [22.395375])), '21.18')
        # Do not force this cell down to the paper's 62.47.
        self.assertEqual(rounded(self.score([60.900378], [64.0523])), '62.48')

    def test_missing_and_nonfinite_dialects_are_not_silently_averaged(self):
        self.assertIsNone(self.score([], [90]))
        with self.assertRaises(ValueError):
            self.score([float('nan')], [90])

    def test_diagnostic_preserves_default_and_gemini_comparisons(self):
        result = {'by_level': [
            {'model': 'm', 'level': 4, 'equal_dialect_score': 65.122776,
             'equal_dialect_score_rounded_means': '65.125'},
            {'model': 'gemini-3.1-pro-preview', 'level': 1,
             'equal_dialect_score': 68.25, 'aggregate_score': 68.16},
        ]}
        reference = {'status': 'test', 'source': {}, 'checks': [
            {'claim': model, 'path': ['by_level', {'model': model, 'level': level}, 'equal_dialect_score'],
             'digits': 2, 'expected': expected}
            for model, level, expected in [('m', 4, '65.13'), ('gemini-3.1-pro-preview', 1, '68.16')]
        ]}
        before = copy.deepcopy(reference)
        self.assertEqual(compare_reference(result, reference)['matched'], 0)
        self.assertEqual(compare_reference(result, reference, use_submitted_gemini_weighting=True)['matched'], 1)
        diagnostic = compare_reference(result, reference, use_submitted_gemini_weighting=True,
                                       use_intermediate_rounding=True)
        self.assertEqual(diagnostic['matched'], 2)
        self.assertEqual(diagnostic['checks'][0]['evaluated_path'][-1], 'equal_dialect_score_rounded_means')
        self.assertEqual(diagnostic['checks'][1]['evaluated_path'][-1], 'aggregate_score')
        self.assertEqual(reference, before)
