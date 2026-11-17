"""Scientific stage boundaries: replay never becomes fresh evidence or a copied score."""
import gzip
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from mlirbench.dataset import ensure_data, HF_REVISION
from mlirbench.pipeline import run_pipeline, score_components, execution_data
from mlirbench.stage_artifacts import load_run, digest

SOURCE = 'module { func.func @main(%x: tensor<1xf32>) -> tensor<1xf32> { return %x : tensor<1xf32> } }'


def fixture(root):
    pair = {'program_id':'seed1_linalg_prog1_L1_R1','input_mlir':SOURCE,'mutated_mlir':SOURCE,
            'dialect':'linalg','level':1,'seed':1,'requirement_id':'L1_R1',
            'judge__outputs_diverge':'False','pairs__status':'success'}
    reasoning = {**pair,'model_key':'fixture-model','predicted_label':'EQUIV','explanation':'Same program'}
    evaluation = {'program_id':pair['program_id'],'evaluated_model':'fixture-model',
                  'ground_truth_label':'EQUIV','predicted_label':'EQUIV',
                  'result_accuracy':5.,'explanation_rating':2.,'hard_label_accuracy':1.,
                  'total_score':99.,'verdict':'PARTIALLY_CORRECT'}
    manifest = {}
    for name, row in [('master/master.jsonl',pair),('reasoning/reasoning.jsonl',reasoning),('evaluation/evaluation.jsonl',evaluation)]:
        name = 'MLIRBench/'+name
        data = (json.dumps(row)+'\n').encode()
        buf = io.BytesIO()
        with gzip.GzipFile(filename='',fileobj=buf,mode='wb',mtime=0) as stream: stream.write(data)
        packed = buf.getvalue()
        path = root/'data/bundle'/(name+'.gz'); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(packed)
        manifest[name] = {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'gzip_sha256':hashlib.sha256(packed).hexdigest()}
    (root/'data/bundle/manifest.json').write_text(json.dumps(manifest))
    folder = root/'data/historical'; folder.mkdir()
    fields = {'judge__compilation_status_A':'success','judge__compilation_status_B':'success',
              'judge__embedded_inputs_json':'{"%x":[1.0]}','judge__programA_output_json':'[1.0]',
              'judge__programB_output_json':'[1.0]'}
    row = {'program_id':pair['program_id'], **fields}
    stream = io.StringIO(); writer = csv.DictWriter(stream,fieldnames=list(row))
    writer.writeheader(); writer.writerow(row)
    raw = stream.getvalue().encode(); packed = gzip.compress(raw,mtime=0)
    evidence = {'program_id':pair['program_id'],'historical_fields':fields,
                'source_sha256':[hashlib.sha256(SOURCE.encode()).hexdigest()]*2,
                'provenance':{'source_sha256':hashlib.sha256(raw).hexdigest(),
                              'row_sha256':hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()}}
    sidecar = (json.dumps(evidence)+'\n').encode()
    (folder/'historical_execution.jsonl').write_bytes(sidecar)
    (folder/'original.csv.gz').write_bytes(packed)
    (folder/'source_manifest.json').write_text(json.dumps({
        'archive_filename':'original.csv.gz','archive_sha256':hashlib.sha256(packed).hexdigest(),
        'source_sha256':hashlib.sha256(raw).hexdigest(),'source_rows':1,
        'sidecar_filename':'historical_execution.jsonl','sidecar_sha256':hashlib.sha256(sidecar).hexdigest(),
        'published_master_sha256':manifest['MLIRBench/master/master.jsonl']['sha256'],
        'published_rows':1,'runtime_input_provenance':'unrecorded fixture metadata'}))
    return pair,manifest


class PipelineBoundaries(unittest.TestCase):
    def setUp(self):
        preflight = patch('mlirbench.live.preflight', return_value={'test_fixture': True})
        preflight.start()
        self.addCleanup(preflight.stop)

    def test_judge_replay_recomputes_score_instead_of_copying_stored_total(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root)
            with patch('mlirbench.live.judge',side_effect=AssertionError('Paid call attempted')):
                result = run_pipeline(root,root/'out',stage='judge')
            stages,manifest = load_run(root/'out')
            score = stages['judge'][0]['data']
            self.assertEqual(score['total_score'],66.67)
            self.assertEqual(score['archived_total_score'],99.)
            self.assertFalse(result['paid_api_calls_requested'])
            self.assertFalse(manifest['historical_end_to_end_runtime_proven'])

    def test_modified_file_is_rejected_before_resuming(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root); run_pipeline(root,root/'out',stage='execution')
            path = root/'out/mutation.jsonl'; path.write_text(path.read_text().replace('return %x','return %other'))
            with self.assertRaisesRegex(ValueError,'Modified stage'): load_run(root/'out')

    def test_changed_parent_is_rejected_even_after_file_hash_is_updated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root); run_pipeline(root,root/'out',stage='execution')
            folder = root/'out'; p = folder/'execution.jsonl'; row = json.loads(p.read_text())
            row['parents'] = ['missing-parent']; row.pop('record_sha256'); row['record_sha256']=digest(row)
            p.write_text(json.dumps(row)+'\n')
            manifest = json.loads((folder/'manifest.json').read_text())
            manifest['artifacts']['execution']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            (folder/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'parents'): load_run(folder)

    def test_archived_reasoning_cannot_be_attached_to_fresh_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root); run_pipeline(root,root/'out',stage='execution')
            folder = root/'out'; p = folder/'execution.jsonl'; row = json.loads(p.read_text())
            row['mode']='fresh'; row.pop('record_sha256'); row['record_sha256']=digest(row); p.write_text(json.dumps(row)+'\n')
            manifest = json.loads((folder/'manifest.json').read_text()); manifest['artifacts']['execution']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            (folder/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'archived downstream'):
                run_pipeline(root,root/'reason',stage='reasoning',input_run=folder)

    def test_resume_reuses_verified_prerequisites_and_writes_requested_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root); run_pipeline(root,root/'execution',stage='execution')
            run_pipeline(root,root/'reasoning',stage='reasoning',input_run=root/'execution')
            prior,_ = load_run(root/'execution'); current,manifest = load_run(root/'reasoning')
            self.assertEqual(manifest['reused_stages'],['mutation','execution'])
            self.assertEqual(prior['execution'],current['execution'])
            self.assertEqual(len(current['reasoning']),1)
            self.assertNotIn('judge',current)

    def test_live_calls_are_disabled_before_data_or_provider_access(self):
        for stage in ('mutation','reasoning','judge','all'):
            with self.assertRaises(PermissionError): run_pipeline(Path('/not-a-project'),Path('/not-an-output'),stage=stage,mode='fresh')

    def test_custom_source_does_not_require_or_download_published_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'program.mlir'; source.write_text(SOURCE)
            pair={'program_id':'fixture','input_mlir':SOURCE,'mutated_mlir':SOURCE,'dialect':'linalg','level':1,'seed':0,'requirement_id':'L1_R1'}
            with patch('mlirbench.pipeline.ensure_data',side_effect=AssertionError('Unexpected archive/download dependency')),patch('mlirbench.live.mutate',return_value=pair):
                run_pipeline(root,root/'out',stage='mutation',mode='fresh',allow_paid=True,source=source,dialect='linalg',level=1)
            artifacts,_=load_run(root/'out')
            self.assertTrue(artifacts['mutation'][0]['program_id'].startswith('fresh_'))

    def test_invalid_judge_scores_cannot_be_aggregated(self):
        for result,explanation in [(11,2),(5,float('nan')),(5,-1)]:
            with self.assertRaises(ValueError): score_components('EQUIV','EQUIV',result,explanation)

    def test_suite_agreement_is_scoped_equiv_and_unsupported_is_unscored(self):
        typed = {'kind':'array','dtype':'<f4','shape':[1],'bytes_b64':'AACAPw=='}
        trial = {'status':'NO_DIFFERENCE_OBSERVED','inputs':[{'data':[1.]}],
                 'runs':[[{'output_typed':typed}],[{'output_typed':typed}]]}
        result = execution_data({'verdict':'NO_DIFFERENCE_OBSERVED','argument_mapping':[['%x'],['%y']],'trials':[trial]})
        self.assertEqual(result['ground_truth_label'],'EQUIV')
        self.assertEqual(result['label_scope'],'declared_valid_input_suite')
        self.assertFalse(result['outputs_diverge'])
        self.assertIsNone(execution_data({'verdict':'UNSUPPORTED_INPUT_PRECONDITIONS'})['ground_truth_label'])


class DownloadConsent(unittest.TestCase):
    def test_local_bundle_works_without_network_or_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); fixture(root); fetch=Mock(side_effect=AssertionError('network')); prompt=Mock(side_effect=AssertionError('prompt'))
            ensure_data(root,prompt=prompt,fetch=fetch); fetch.assert_not_called(); prompt.assert_not_called()

    def test_decline_makes_no_network_request(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); _,manifest=fixture(root)
            name=next(iter(manifest)); (root/'data/bundle'/(name+'.gz')).unlink(); fetch=Mock()
            with self.assertRaises(PermissionError): ensure_data(root,[name],prompt=lambda _: 'no',fetch=fetch)
            fetch.assert_not_called()

    def test_yes_downloads_only_pinned_verified_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); _,manifest=fixture(root); name=next(iter(manifest))
            packed=root/'data/bundle'/(name+'.gz'); raw=gzip.decompress(packed.read_bytes()); packed.unlink()
            fetch=Mock(return_value=raw)
            result=ensure_data(root,[name],prompt=lambda _: 'yes',fetch=fetch)
            self.assertIn(HF_REVISION,fetch.call_args.args[0])
            self.assertEqual((root/'data/processed'/name).read_bytes(),raw)
            self.assertEqual(result['files'][name],'consented_pinned_huggingface_download')

    def test_wrong_remote_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); _,manifest=fixture(root); name=next(iter(manifest)); (root/'data/bundle'/(name+'.gz')).unlink()
            with self.assertRaisesRegex(ValueError,'differs from frozen'):
                ensure_data(root,[name],consent=True,fetch=lambda _: b'wrong')
            self.assertFalse((root/'data/processed'/name).exists())


class OriginalJudgeIntegration(unittest.TestCase):
    def test_matching_suite_reaches_judge_and_keeps_evidence_scope(self):
        import sys
        root=Path(__file__).resolve().parents[2]
        sys.path.insert(0,str(root/'optional/live_pipeline'))
        from agents.judge_agent.node import judge_agent
        payload={'program_id':'fixture','programA':SOURCE,'programB':SOURCE,'dialect':'linalg',
                 'compilation_status_A':'success','compilation_status_B':'success',
                 'programA_output':[1.],'programB_output':[1.],'outputs_diverge':False,
                 'embedded_inputs':{'%x':[1.]},'validation_verdict':'NO_DIFFERENCE_OBSERVED',
                 'label_scope':'declared_valid_input_suite',
                 'prediction':{'model_key':'fixture','predicted_label':'EQUIV','predicted_output_A':[1.],
                               'predicted_output_B':[1.],'explanation':'Same return operation'}}
        with patch('agents.judge_agent.node._invoke_llm_judge',return_value={'scores':{'result_accuracy':10,'explanation_rating':4},'evidence':{},'reasoning':'test fixture'}) as call:
            result=judge_agent({'judge_input':payload})
        call.assert_called_once()
        self.assertEqual(result['judge_details']['ground_truth_label'],'EQUIV')
        self.assertEqual(result['judge_details']['label_scope'],'declared_valid_input_suite')
        self.assertEqual(result['judge_details']['final_score'],100.)
