"""Frozen execution observations must be verified before replay or paid work."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlirbench import cli
from mlirbench.ae import prepare_data
from mlirbench.dataset import ensure_data
from mlirbench.execution.provenance import load_verified_sources
from mlirbench.pipeline import run_pipeline
from mlirbench.stage_artifacts import digest, load_run
from tests.unit.test_pipeline import fixture


def change_observation(root):
    path = root/'data/historical/historical_execution.jsonl'
    value = json.loads(path.read_text())
    value['historical_fields']['judge__programA_output_json'] = '[999999]'
    path.write_text(json.dumps(value)+'\n')


def rewrite_checkpoint(folder, change):
    """Make the checkpoint internally consistent, isolating source verification."""
    path = folder/'execution.jsonl'
    value = json.loads(path.read_text())
    change(value)
    value.pop('record_sha256')
    value['record_sha256'] = digest(value)
    path.write_text(json.dumps(value)+'\n')
    import hashlib
    manifest = json.loads((folder/'manifest.json').read_text())
    manifest['artifacts']['execution']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder/'manifest.json').write_text(json.dumps(manifest))
    load_run(folder)  # File, record and parent hashes alone accept this fixture.


class HistoricalIntegrity(unittest.TestCase):
    def setUp(self):
        preflight = patch('mlirbench.live.preflight', return_value={'test_fixture': True})
        preflight.start()
        self.addCleanup(preflight.stop)

    def test_valid_source_check_records_integrity_without_certifying_runtime_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); pair,_ = fixture(root); ensure_data(root)
            master, historical, verification = load_verified_sources(root)
            self.assertEqual(master[pair['program_id']], pair)
            self.assertEqual(historical[pair['program_id']]['historical_fields']['judge__programA_output_json'], '[1.0]')
            self.assertTrue(verification['passed'])
            self.assertFalse(verification['runtime_inputs_proven'])

    def test_every_archived_pipeline_entry_rejects_changed_sidecar(self):
        for stage in ('execution','reasoning','judge','metrics','all'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); fixture(root); change_observation(root)
                with self.assertRaisesRegex(ValueError, 'Historical sidecar hash mismatch'):
                    run_pipeline(root,root/'out',stage=stage)

    def test_data_preparation_rejects_changed_sidecar(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root); ensure_data(root); change_observation(root)
            with self.assertRaisesRegex(ValueError, 'Historical sidecar hash mismatch'):
                prepare_data(root)

    def test_corrupt_original_archive_is_rejected_before_decompression(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root); ensure_data(root)
            (root/'data/historical/original.csv.gz').write_bytes(b'not a gzip archive')
            with self.assertRaisesRegex(ValueError, 'Original historical CSV hash mismatch'):
                load_verified_sources(root)

    def test_missing_original_archive_is_not_replaced_or_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root); ensure_data(root)
            (root/'data/historical/original.csv.gz').unlink()
            with self.assertRaises(FileNotFoundError):
                run_pipeline(root,root/'out',stage='execution')

    def test_duplicate_sidecar_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root); ensure_data(root)
            path = root/'data/historical/historical_execution.jsonl'
            path.write_text(path.read_text()*2)
            with self.assertRaisesRegex(ValueError, 'Duplicate program IDs'):
                load_verified_sources(root)

    def test_changed_history_stops_fresh_judging_before_any_model_call(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root); change_observation(root)
            with patch('mlirbench.live.judge') as call:
                with self.assertRaisesRegex(ValueError, 'Historical sidecar hash mismatch'):
                    run_pipeline(root,root/'out',stage='judge',mode='fresh',models=['fixture-model'],allow_paid=True)
            call.assert_not_called()

    def test_resume_rechecks_original_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root)
            run_pipeline(root,root/'prior',stage='execution')
            change_observation(root)
            with patch('mlirbench.live.reason') as call:
                with self.assertRaisesRegex(ValueError, 'Historical sidecar hash mismatch'):
                    run_pipeline(root,root/'out',stage='reasoning',mode='fresh',models=['fixture-model'],allow_paid=True,input_run=root/'prior')
            call.assert_not_called()

    def test_self_consistent_checkpoint_cannot_change_frozen_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root)
            prior = root/'prior'; run_pipeline(root,prior,stage='execution')
            rewrite_checkpoint(prior,lambda r:r['data'].update(programA_output=[999999]))
            with patch('mlirbench.live.reason') as call:
                with self.assertRaisesRegex(ValueError, 'observation differs from frozen source'):
                    run_pipeline(root,root/'out',stage='reasoning',mode='fresh',models=['fixture-model'],allow_paid=True,input_run=prior)
            call.assert_not_called()

    def test_self_consistent_checkpoint_cannot_claim_another_source_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root)
            prior = root/'prior'; run_pipeline(root,prior,stage='execution')
            rewrite_checkpoint(prior,lambda r:r['source'].update(file_sha256='0'*64))
            with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
                run_pipeline(root,root/'out',stage='reasoning',input_run=prior)

    def test_fresh_cpu_execution_does_not_require_historical_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root)
            shutil.rmtree(root/'data/historical')
            with patch('mlirbench.execution.pair_validation.validate_pair',return_value={'verdict':'UNSUPPORTED_INPUT_PRECONDITIONS'}):
                run_pipeline(root,root/'out',stage='execution',mode='fresh')
            stages, manifest = load_run(root/'out')
            self.assertEqual(stages['execution'][0]['mode'], 'fresh')
            self.assertNotIn('historical_source_verification', manifest)

    def test_historical_candidate_cli_checks_sources_before_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); pair,_ = fixture(root); ensure_data(root); change_observation(root)
            with patch.object(cli,'ROOT',root),patch('mlirbench.execution.historical_replay.replay_candidate') as call:
                with self.assertRaisesRegex(ValueError, 'Historical sidecar hash mismatch'):
                    cli.main(['historical-candidate','--id',pair['program_id'],'--output',str(root/'out.json')])
            call.assert_not_called()

    def test_stored_metadata_cli_checks_sources_before_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); pair,_ = fixture(root); ensure_data(root); change_observation(root)
            with patch.object(cli,'ROOT',root),patch('mlirbench.execution.pair_validation.validate_pair') as call:
                with self.assertRaisesRegex(ValueError, 'Historical sidecar hash mismatch'):
                    cli.main(['validate','--dataset',str(root/'data/processed/MLIRBench/master/master.jsonl'),
                              '--id',pair['program_id'],'--stored-metadata','--output',str(root/'out.jsonl')])
            call.assert_not_called()
