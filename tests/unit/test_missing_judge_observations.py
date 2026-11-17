"""Resume metadata follows selected records, never stale manifest counts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlirbench.pipeline import missing_judge_observations, run_pipeline, score_components
from mlirbench.stage_artifacts import load_run, record, save_records
from tests.unit.test_pipeline import fixture


def fresh_checkpoint(root, *, metadata=None, unscored=False):
    """Two models on two pairs, with one absent judgment for each model.

    Synthetic fresh stage records isolate resume behavior from published data.
    Model calls and compiler execution are not claimed by these test fixtures.
    """
    pair,_ = fixture(root)
    pairs = [pair, {**pair, 'program_id':'fixture_second_pair'}]
    stages = {name: [] for name in ('mutation','execution','reasoning','judge')}
    for index, pair in enumerate(pairs):
        mutation = record('mutation', pair, pair, mode='fresh', source={'kind':'unit_test_fixture'})
        execution = record('execution', pair, {'ground_truth_label':'EQUIV'}, mode='fresh',
                           parents=[mutation['record_sha256']], source={'kind':'unit_test_fixture'})
        stages['mutation'].append(mutation); stages['execution'].append(execution)
        for model in ('alpha','beta'):
            reasoning = record('reasoning', pair, {'model_key':model, 'predicted_label':'EQUIV'},
                               mode='fresh', parents=[execution['record_sha256']],
                               model=model, source={'kind':'unit_test_fixture'})
            stages['reasoning'].append(reasoning)
            # First pair is missing beta; second pair is missing alpha.
            if model == ('beta' if index == 0 else 'alpha'):
                continue
            data = ({'total_score':None, 'status':'unscored_execution_unverified'} if unscored
                    else score_components('EQUIV','EQUIV',5,2))
            stages['judge'].append(record('judge',pair,data,mode='fresh',model=model,
                                          parents=[execution['record_sha256'],reasoning['record_sha256']],
                                          source={'kind':'unit_test_fixture'}))
    folder = root/'prior'; folder.mkdir()
    manifest = {'artifacts':{name:save_records(folder/(name+'.jsonl'),values) for name,values in stages.items()}}
    if metadata is not None:
        manifest['missing_judge_observations'] = metadata
    (folder/'manifest.json').write_text(json.dumps(manifest)+'\n')
    load_run(folder)
    return folder, pairs


def keys(manifest):
    return {(r['program_id'],r['model']) for r in manifest['missing_judge_observations']}


class MissingJudgeMetadata(unittest.TestCase):
    def resume(self, root, prior, output='resumed', **selection):
        with patch('mlirbench.live.preflight', side_effect=AssertionError('Unexpected live configuration')), \
             patch('mlirbench.live.mutate', side_effect=AssertionError('Unexpected mutation API')), \
             patch('mlirbench.live.reason', side_effect=AssertionError('Unexpected reasoning API')), \
             patch('mlirbench.live.judge', side_effect=AssertionError('Unexpected judge API')):
            run_pipeline(root,root/output,stage='metrics',input_run=prior,**selection)
        return load_run(root/output)

    def test_full_resume_recovers_missing_metadata_even_from_older_broken_runs(self):
        for old_metadata in (None, [], [{'program_id':'unrelated','model':'other','reason':'stale metadata'}]):
            with self.subTest(old_metadata=old_metadata), tempfile.TemporaryDirectory() as temp:
                root=Path(temp); prior,pairs=fresh_checkpoint(root,metadata=old_metadata)
                stages,manifest=self.resume(root,prior)
                self.assertEqual(keys(manifest),{(pairs[0]['program_id'],'beta'),(pairs[1]['program_id'],'alpha')})
                self.assertEqual(manifest['metrics']['scored_records'],2)
                original,_=load_run(prior)
                self.assertEqual(stages,original)
                self.assertIn('Missing judge observations: 2;', (root/'resumed/REPORT.md').read_text())

    def test_id_model_and_combined_subsets_filter_missing_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); prior,pairs=fresh_checkpoint(root)
            first,second=[p['program_id'] for p in pairs]
            cases=[({'ids':[first]}, {(first,'beta')}),
                   ({'models':['alpha']}, {(second,'alpha')}),
                   ({'ids':[first],'models':['alpha']}, set()),
                   ({'ids':[first],'models':['beta']}, {(first,'beta')})]
            for index,(selection,expected) in enumerate(cases):
                with self.subTest(selection=selection):
                    _,manifest=self.resume(root,prior,output=f'subset{index}',**selection)
                    self.assertEqual(keys(manifest),expected)

    def test_resume_of_a_subset_does_not_restore_excluded_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); prior,pairs=fresh_checkpoint(root)
            selection={'ids':[pairs[0]['program_id']],'models':['beta']}
            _,first=self.resume(root,prior,output='subset',**selection)
            _,second=self.resume(root,root/'subset',output='subset_again')
            self.assertEqual(first['missing_judge_observations'],second['missing_judge_observations'])
            self.assertEqual(len(second['missing_judge_observations']),1)
            self.assertEqual(second['metrics']['scored_records'],0)

    def test_explicit_unscored_observations_are_not_counted_as_absent(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); prior,pairs=fresh_checkpoint(root,unscored=True)
            _,manifest=self.resume(root,prior)
            self.assertEqual(len(manifest['missing_judge_observations']),2)
            self.assertEqual(manifest['metrics']['unscored_records'],2)
            self.assertEqual(manifest['metrics']['scored_records'],0)

    def test_judge_not_requested_is_not_reported_as_missing(self):
        self.assertEqual(missing_judge_observations({'reasoning':[{'program_id':'p','model':'m'}]}),[])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); fixture(root)
            run_pipeline(root,root/'reasoning',stage='reasoning')
            _,manifest=load_run(root/'reasoning')
            self.assertEqual(manifest['missing_judge_observations'],[])

    def test_judge_replay_and_metrics_resume_use_the_same_missing_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); pair,_=fixture(root)
            # Add an archived reasoning case without a judge observation.
            # Existing observation/source data stay intact in this fixture.
            import gzip,hashlib
            name='MLIRBench/reasoning/reasoning.jsonl'
            path=root/'data/bundle'/(name+'.gz')
            original=gzip.decompress(path.read_bytes())
            absent={**json.loads(original), 'model_key':'missing-model'}
            raw=original+(json.dumps(absent)+'\n').encode()
            packed=gzip.compress(raw,mtime=0);path.write_bytes(packed)
            path=root/'data/bundle/manifest.json';bundles=json.loads(path.read_text())
            bundles[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                           'gzip_sha256':hashlib.sha256(packed).hexdigest()}
            path.write_text(json.dumps(bundles))
            run_pipeline(root,root/'judge',stage='judge')
            _,original=load_run(root/'judge')
            self.assertEqual(keys(original),{(pair['program_id'],'missing-model')})
            _,resumed=self.resume(root,root/'judge',models=['missing-model'])
            self.assertEqual(resumed['missing_judge_observations'],original['missing_judge_observations'])
            self.assertEqual(resumed['metrics']['scored_records'],0)
