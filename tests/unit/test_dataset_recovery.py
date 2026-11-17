"""Distribution repair is byte-exact, local-first and consent-gated."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock,patch

from mlirbench.dataset import ensure_data,prepare_distribution
from mlirbench.pipeline import run_pipeline
from mlirbench.stage_artifacts import load_run
from tests.unit.test_pipeline import fixture,SOURCE
from tests.unit.test_missing_judge_observations import fresh_checkpoint


class DistributionRecovery(unittest.TestCase):
    def test_verified_raw_repairs_missing_gzip_without_prompt_or_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,manifest=fixture(root);ensure_data(root)
            originals={n:(root/'data/bundle'/(n+'.gz')).read_bytes() for n in manifest}
            for n in manifest:(root/'data/bundle'/(n+'.gz')).unlink()
            fetch=Mock(side_effect=AssertionError('network'));prompt=Mock(side_effect=AssertionError('consent unnecessary'))
            result=ensure_data(root,prompt=prompt,fetch=fetch)
            self.assertEqual(set(result['repaired_bundles']),set(manifest))
            for n,data in originals.items():self.assertEqual((root/'data/bundle'/(n+'.gz')).read_bytes(),data)
            fetch.assert_not_called();prompt.assert_not_called()

    def test_intact_build_bundles_do_not_unpack_on_host(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,manifest=fixture(root)
            result=prepare_distribution(root)
            self.assertEqual(result['verified_build_bundles'],len(manifest))
            self.assertFalse((root/'data/processed').exists())

    def test_all_manifest_entries_are_checked_even_outside_stage_views(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,manifest=fixture(root)
            name='MLIRBench/nonstage.json';raw=b'{"fixture":true}\n'
            from mlirbench.dataset import packed_bytes
            # Use the same frozen header recipe for this disposable fixture.
            import io
            buffer=io.BytesIO()
            with gzip.GzipFile(filename='',fileobj=buffer,mode='wb',mtime=0,compresslevel=9) as stream:stream.write(raw)
            packed=buffer.getvalue()
            manifest[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'gzip_sha256':hashlib.sha256(packed).hexdigest()}
            (root/'data/bundle/manifest.json').write_text(json.dumps(manifest))
            path=root/'data/processed'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            result=prepare_distribution(root,prompt=Mock(side_effect=AssertionError('prompt')),fetch=Mock(side_effect=AssertionError('network')))
            self.assertEqual(result['verified_build_bundles'],4)
            self.assertEqual((root/'data/bundle'/(name+'.gz')).read_bytes(),packed)
            self.assertEqual(result['repaired_bundles'],[name])

    def test_corrupt_gzip_is_rejected_even_with_valid_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,manifest=fixture(root);ensure_data(root);name=next(iter(manifest))
            path=root/'data/bundle'/(name+'.gz');path.write_bytes(b'corrupt')
            fetch=Mock()
            with self.assertRaisesRegex(ValueError,'Modified data bundle'):ensure_data(root,fetch=fetch)
            self.assertEqual(path.read_bytes(),b'corrupt');fetch.assert_not_called()

    def test_modified_raw_is_not_overwritten_from_valid_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,manifest=fixture(root);ensure_data(root);name=next(iter(manifest))
            path=root/'data/processed'/name;path.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError,'refusing to overwrite'):ensure_data(root)
            self.assertEqual(path.read_bytes(),b'modified')

    def test_recompression_mismatch_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,manifest=fixture(root);ensure_data(root);name=next(iter(manifest))
            path=root/'data/bundle'/(name+'.gz');path.unlink()
            original=(root/'data/processed'/name).read_bytes()
            with patch('mlirbench.dataset.packed_bytes',side_effect=ValueError('frozen gzip mismatch')):
                with self.assertRaisesRegex(ValueError,'mismatch'):prepare_distribution(root)
            self.assertFalse(path.exists());self.assertEqual((root/'data/processed'/name).read_bytes(),original)

    def test_build_recovery_retains_yes_no_consent_and_pinned_hash_validation(self):
        for consent in (False,True):
            with self.subTest(consent=consent),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);_,manifest=fixture(root);name=next(iter(manifest))
                path=root/'data/bundle'/(name+'.gz');raw=gzip.decompress(path.read_bytes());path.unlink()
                fetch=Mock(return_value=raw)
                if consent:
                    result=prepare_distribution(root,prompt=lambda _:'yes',fetch=fetch)
                    self.assertEqual(result['verified_build_bundles'],len(manifest))
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),manifest[name]['gzip_sha256'])
                    self.assertIn('1af6c3c7a29946a7cd0e2aafd9386686d7fec9d1',fetch.call_args.args[0])
                else:
                    with self.assertRaises(PermissionError):prepare_distribution(root,prompt=lambda _:'no',fetch=fetch)
                    fetch.assert_not_called();self.assertFalse(path.exists())

    def test_wrong_remote_bytes_never_create_a_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,manifest=fixture(root);name=next(iter(manifest));path=root/'data/bundle'/(name+'.gz');path.unlink()
            with self.assertRaisesRegex(ValueError,'differs from frozen'):
                prepare_distribution(root,consent=True,fetch=lambda _:b'wrong')
            self.assertFalse(path.exists());self.assertFalse((root/'data/processed'/name).exists())


class ScopedPipelineData(unittest.TestCase):
    def setUp(self):
        preflight=patch('mlirbench.live.preflight',return_value={'test_fixture':True})
        preflight.start();self.addCleanup(preflight.stop)

    def test_self_contained_fresh_checkpoint_metrics_needs_no_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prior,_=fresh_checkpoint(root);shutil.rmtree(root/'data')
            with patch('mlirbench.pipeline.ensure_data',side_effect=AssertionError('Unexpected published dependency')):
                result=run_pipeline(root,root/'metrics',stage='metrics',input_run=prior)
            self.assertEqual(result['metrics']['scored_records'],2)
            self.assertEqual(len(load_run(root/'metrics')[1]['missing_judge_observations']),2)

    def test_fresh_execution_reasoning_and_judge_can_use_checkpoint_without_dataset(self):
        for stage in ('execution','reasoning','judge'):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);prior,_=fresh_checkpoint(root);shutil.rmtree(root/'data')
                with patch('mlirbench.pipeline.ensure_data',side_effect=AssertionError('Unexpected published dependency')), \
                     patch('mlirbench.execution.pair_validation.validate_pair',return_value={'verdict':'UNSUPPORTED_INPUT_PRECONDITIONS'}), \
                     patch('mlirbench.live.reason',return_value={'model_key':'alpha','predicted_label':'EQUIV'}), \
                     patch('mlirbench.live.judge',return_value={'total_score':None,'status':'unscored_execution_unverified'}):
                    run_pipeline(root,root/'out',stage=stage,mode='fresh',models=['alpha'],all_records=True,allow_paid=True,input_run=prior)
                self.assertEqual(load_run(root/'out')[1]['status'],'COMPLETE')

    def test_custom_fresh_run_can_continue_without_any_published_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'new.mlir';source.write_text(SOURCE)
            pair={'input_mlir':SOURCE,'mutated_mlir':SOURCE,'dialect':'linalg','level':1,'seed':0,'requirement_id':'L1_R1'}
            options=dict(mode='fresh',models=['alpha'],allow_paid=True)
            with patch('mlirbench.pipeline.ensure_data',side_effect=AssertionError('Unexpected published dependency')), \
                 patch('mlirbench.live.mutate',return_value=pair), \
                 patch('mlirbench.execution.pair_validation.validate_pair',return_value={'verdict':'UNSUPPORTED_INPUT_PRECONDITIONS'}), \
                 patch('mlirbench.live.reason',side_effect=RuntimeError('Synthetic failure')):
                with self.assertRaises(RuntimeError):
                    run_pipeline(root,root/'failed',source=source,dialect='linalg',level=1,**options)
            with patch('mlirbench.pipeline.ensure_data',side_effect=AssertionError('Unexpected published dependency')), \
                 patch('mlirbench.live.mutate',side_effect=AssertionError('Repeated mutation')), \
                 patch('mlirbench.live.reason',return_value={'model_key':'alpha','predicted_label':'EQUIV'}), \
                 patch('mlirbench.live.judge',return_value={'total_score':None,'status':'unscored_execution_unverified'}):
                result=run_pipeline(root,root/'resumed',input_run=root/'failed',resume_incomplete=True,**options)
            self.assertEqual(result['metrics']['scored_records'],0)
            self.assertFalse((root/'data').exists())

    def test_fresh_execution_from_published_pair_only_resolves_master(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root)
            with patch('mlirbench.pipeline.ensure_data',wraps=ensure_data) as resolve, \
                 patch('mlirbench.execution.pair_validation.validate_pair',return_value={'verdict':'UNSUPPORTED_INPUT_PRECONDITIONS'}):
                run_pipeline(root,root/'out',stage='execution',mode='fresh')
            self.assertEqual(resolve.call_args.args[1],['MLIRBench/master/master.jsonl'])
            self.assertFalse((root/'data/processed/MLIRBench/reasoning/reasoning.jsonl').exists())

    def test_replay_from_fresh_parent_fails_before_any_data_resolution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prior,_=fresh_checkpoint(root);shutil.rmtree(root/'data')
            with patch('mlirbench.pipeline.ensure_data',side_effect=AssertionError('Unexpected published dependency')):
                with self.assertRaisesRegex(ValueError,'archived downstream'):
                    run_pipeline(root,root/'out',stage='judge',input_run=prior)
