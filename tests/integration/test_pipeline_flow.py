"""Real execution between mocked model endpoints; no model/API access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlirbench.pipeline import run_pipeline, score_components
from mlirbench.stage_artifacts import load_run
from tests.unit.test_pipeline import fixture, SOURCE


class CompletePipeline(unittest.TestCase):
    def setUp(self):
        preflight = patch('mlirbench.live.preflight', return_value={'test_fixture': True})
        preflight.start()
        self.addCleanup(preflight.stop)

    def test_failed_reasoning_preserves_completed_mutation_and_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); pair,_=fixture(root)
            with patch('mlirbench.live.mutate',return_value=pair),patch('mlirbench.live.reason',side_effect=RuntimeError('test endpoint unavailable')):
                with self.assertRaises(RuntimeError):
                    run_pipeline(root,root/'out',mode='fresh',models=['fixture-model'],allow_paid=True)
            artifacts,manifest=load_run(root/'out')
            self.assertEqual(manifest['status'],'FAILED')
            self.assertEqual(manifest['active_stage'],'reasoning')
            self.assertEqual(set(artifacts),{'mutation','execution'})

    def test_fresh_chain_uses_new_mutation_real_execution_and_selected_response(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); pair,_=fixture(root)
            changed={**pair,'mutated_mlir':SOURCE+'\n// fresh mutation fixture\n'}
            def reason(project,actual_pair,execution,model):
                self.assertEqual(actual_pair['mutated_mlir'],changed['mutated_mlir'])
                self.assertEqual(execution['ground_truth_label'],'EQUIV')
                self.assertEqual(execution['label_scope'],'declared_valid_input_suite')
                return {'model_key':model,'predicted_label':'EQUIV','explanation':'new response fixture'}
            def judge(project,actual_pair,execution,prediction):
                self.assertEqual(prediction['explanation'],'new response fixture')
                return {'program_id':actual_pair['program_id'],'evaluated_model':'fixture-model',
                        **score_components(prediction['predicted_label'],execution['ground_truth_label'],10,4)}
            with patch('mlirbench.live.mutate',return_value=changed),patch('mlirbench.live.reason',side_effect=reason),patch('mlirbench.live.judge',side_effect=judge):
                result=run_pipeline(root,root/'out',mode='fresh',models=['fixture-model'],allow_paid=True)
            artifacts,_=load_run(root/'out')
            self.assertTrue(artifacts['mutation'][0]['program_id'].startswith('fresh_'))
            self.assertTrue(all(r['mode']=='fresh' for values in artifacts.values() for r in values))
            self.assertEqual(artifacts['judge'][0]['data']['total_score'],100.)
            self.assertEqual(result['metrics']['scored_records'],1)
            self.assertNotEqual(result['metrics']['scope'],'full_published_archive')
