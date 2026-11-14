import unittest
from mlirbench.evaluation.recorded import summarize


class RecordedScores(unittest.TestCase):
    def test_missing_predictions_count_against_full_denominator(self):
        master = [{'program_id': 'a', 'judge__outputs_diverge': 'False'}, {'program_id': 'b', 'judge__outputs_diverge': 'True'}]
        evaluations = [{'program_id': 'a', 'evaluated_model': 'm', 'ground_truth_label': 'EQUIV', 'predicted_label': 'EQUIV', 'total_score': 80}]
        r = summarize(master, evaluations)['models'][0]
        self.assertEqual(r['hard_label_accuracy_percent_all_pairs'], 50)
        self.assertEqual(r['hard_label_accuracy_percent_available_rows'], 100)
        self.assertEqual(r['mean_recorded_judge_score_missing_as_zero'], 40)
        self.assertEqual(r['missing_program_ids'], ['b'])
        with self.assertRaises(ValueError): summarize(master, evaluations * 2)

    def test_recorded_master_disagreement_is_exposed(self):
        master = [{'program_id': 'a', 'judge__outputs_diverge': 'False'}]
        evaluations = [{'program_id': 'a', 'evaluated_model': 'm', 'ground_truth_label': 'DIVERGE', 'predicted_label': 'DIVERGE', 'total_score': 80}]
        r = summarize(master, evaluations)['models'][0]
        self.assertEqual(r['label_matches_recorded_ground_truth'], 1)
        self.assertEqual(r['label_matches_published_master'], 0)
        self.assertEqual(r['ground_truth_disagrees_with_master'], ['a'])
