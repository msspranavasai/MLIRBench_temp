import json
from pathlib import Path
import tempfile
import unittest
from mlirbench.evaluation.reproduce import summarize, compare_reference
from mlirbench.evaluation.downstream import classification_metrics, load_predictions, pooled, SEMANTIC
from mlirbench.ae import readiness


def pair(number, label='False'):
    return {'program_id': f'seed1_linalg_prog{number}_L1_R1', 'judge__outputs_diverge': label,
            'seed': 1, 'level': 1, 'dialect': 'linalg', 'requirement_id': 'L1_R1', 'pairs__status': 'success'}


def evaluation(pid, model='m', label='EQUIV', predicted='EQUIV', score=80):
    return {'program_id': pid, 'evaluated_model': model, 'ground_truth_label': label,
            'predicted_label': predicted, 'total_score': score, 'result_accuracy': 4, 'explanation_rating': 3}


class PaperReplay(unittest.TestCase):
    def test_missing_predictions_and_available_denominators(self):
        master = [pair(1), pair(2, 'True')]
        result = summarize(master, [evaluation(master[0]['program_id'])])
        model = result['models'][0]
        self.assertEqual(model['hard_label_accuracy_percent'], 50)
        self.assertEqual(model['aggregate_score_available'], 80)
        self.assertEqual(model['aggregate_score_missing_as_zero'], 40)
        self.assertEqual(model['missing_program_ids'], [master[1]['program_id']])

    def test_duplicates_and_ground_truth_changes_are_rejected(self):
        master = [pair(1)]
        row = evaluation(master[0]['program_id'])
        with self.assertRaises(ValueError):
            summarize(master, [row, row])
        with self.assertRaises(ValueError):
            summarize(master, [{**row, 'ground_truth_label': 'DIVERGE'}])
        with self.assertRaises(ValueError):
            summarize([], [])

    def test_rounding_differences_are_reported_without_extra_tolerance(self):
        ref = {'status': 'reference', 'source': {}, 'checks': [{'claim': 'x', 'path': ['x'], 'expected': '1.00', 'digits': 2}]}
        self.assertEqual(compare_reference({'x': 1.004}, ref)['mismatched'], 0)
        self.assertEqual(compare_reference({'x': 1.005}, ref)['mismatched'], 1)

    def test_pooled_and_model_mean_are_distinct(self):
        master = [pair(1), pair(2)]
        result = summarize(master, [evaluation(master[0]['program_id'], 'a', score=100),
                                    evaluation(master[1]['program_id'], 'a', score=100),
                                    evaluation(master[0]['program_id'], 'b', score=0)])
        self.assertEqual(result['summary']['mean_aggregate_score_across_models'], 50)
        self.assertAlmostEqual(result['summary']['mean_aggregate_score_available_rows'], 200/3)


class DownstreamReplay(unittest.TestCase):
    def test_independent_confusion_math(self):
        records = [{'_gold': a, '_pred': b} for a,b in zip(
            ['EQUIV','EQUIV','DIVERGE','ERROR'], ['EQUIV','DIVERGE','DIVERGE','EQUIV'])]
        result = classification_metrics(records, SEMANTIC)
        self.assertEqual(result['accuracy'], .5)
        self.assertAlmostEqual(result['macro_f1'], 7/18)
        self.assertEqual(result['false_equiv_rate'], .5)
        self.assertEqual(result['f1_error'], 0)

    def test_invalid_predictions_are_counted_in_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'predictions.csv'
            p.write_text('label,prediction\nEQUIV,EQUIV\nDIVERGE,INVALID\n')
            provenance = []
            records = load_predictions(p, SEMANTIC, provenance)
            self.assertEqual(len(records), 1)
            self.assertEqual(provenance[0]['excluded_rows'], 1)


    def test_prediction_file_must_match_original_heldout_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            predictions = root/'outputs/final_cv/zero'; predictions.mkdir(parents=True)
            (predictions/'fold_0_val_predictions.csv').write_text('dt1_id,fusion_benefit,prediction\nwrong,LOW,LOW\n')
            split = root/'data/folds/fold_0'; split.mkdir(parents=True)
            (split/'val.json').write_text(json.dumps([{'dt1_id':'correct', 'fusion_benefit':'LOW'}]))
            with self.assertRaisesRegex(ValueError, 'held-out split'):
                pooled(root, 'zero', [])


class Readiness(unittest.TestCase):
    def test_passing_tests_alone_cannot_mean_submission_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'experiments').mkdir()
            (root/'experiments/claims.json').write_text(json.dumps({'claims': [
                {'id':'historical_labels','status':'open','remaining':'Resolve evidence.'}]}))
            results = root/'results'; results.mkdir()
            (results/'tests.json').write_text(json.dumps({'passed':True,'tests_run':100,'skipped':0}))
            report = readiness(root, results)
            self.assertEqual(report['status'], 'NOT_SUBMISSION_READY')
            self.assertTrue(any(g['id']=='historical_labels' and g['status']=='OPEN' for g in report['gates']))
            self.assertTrue(any(g['id']=='automated_tests' and g['status']=='FAIL' for g in report['gates']))
