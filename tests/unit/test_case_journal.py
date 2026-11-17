"""Durable successes survive later failures, checkpoint errors and process exit."""
import json
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mlirbench.case_journal import load_journal
from mlirbench.pipeline import run_pipeline, score_components
from mlirbench.stage_artifacts import atomic_json, load_run, save_records
from tests.unit.test_missing_judge_observations import fresh_checkpoint
from tests.unit.test_pipeline import fixture


def completed_cases(folder):
    stages, manifest = load_run(folder)
    hashes = {r['record_sha256']:r for values in stages.values() for r in values}
    return load_journal(folder, manifest, by_hash=hashes)


def prediction(root, pair, execution, model):
    return {'model_key':model,'predicted_label':'EQUIV','explanation':'Synthetic test response'}


def judgment(root, pair, execution, answer):
    return score_components(answer['predicted_label'],execution['ground_truth_label'],5,2)


class CaseResumption(unittest.TestCase):
    def setUp(self):
        mocked = patch('mlirbench.live.preflight',return_value={'test_fixture':True})
        mocked.start();self.addCleanup(mocked.stop)

    def run_stage(self, root, output, prior, stage='reasoning', resume=False, **kw):
        return run_pipeline(root,root/output,stage=stage,mode='fresh',models=['alpha'] if stage!='mutation' else None,
                            input_run=prior,allow_paid=True,all_records=True,resume_incomplete=resume,**kw)

    def test_reasoning_failure_saves_success_and_retries_only_failed_case(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prior,_=fresh_checkpoint(root)
            calls=[]
            def fail_second(*args):
                calls.append(args[1]['program_id'])
                if len(calls)==2:raise RuntimeError('Synthetic endpoint failure')
                return prediction(*args)
            with patch('mlirbench.live.reason',side_effect=fail_second):
                with self.assertRaises(RuntimeError):self.run_stage(root,'failed',prior)
            entries=completed_cases(root/'failed')
            good=next(e['attempts'][-1]['record'] for e in entries.values() if e['attempts'][-1]['status']=='COMPLETE')
            self.assertEqual(len(entries),2)
            self.assertNotIn('reasoning',load_run(root/'failed')[0])
            with patch('mlirbench.live.reason',side_effect=prediction) as retry:
                self.run_stage(root,'resumed',root/'failed',resume=True)
            retry.assert_called_once()
            self.assertEqual(retry.call_args.args[1]['program_id'],calls[1])
            stages,manifest=load_run(root/'resumed')
            self.assertIn(good,stages['reasoning'])
            self.assertEqual(len(stages['reasoning']),2)
            self.assertEqual(manifest['reused_case_observations']['reasoning'],1)
            retried=next(e for e in completed_cases(root/'resumed').values() if len(e['attempts'])==2)
            self.assertEqual([a['status'] for a in retried['attempts']],['FAILED','COMPLETE'])
            self.assertEqual(retried['attempts'][0]['failure_type'],'RuntimeError')
            self.assertEqual(load_run(root/'failed')[1]['status'],'FAILED')

    def test_judge_failure_reuses_successful_rating_without_fabrication(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prior,_=fresh_checkpoint(root);count=[]
            def fail_second(*args):
                count.append(args[1]['program_id'])
                if len(count)==2:raise RuntimeError('Synthetic judge failure')
                return judgment(*args)
            with patch('mlirbench.live.judge',side_effect=fail_second):
                with self.assertRaises(RuntimeError):self.run_stage(root,'failed',prior,stage='judge')
            with patch('mlirbench.live.judge',side_effect=judgment) as retry:
                self.run_stage(root,'resumed',root/'failed',stage='judge',resume=True)
            retry.assert_called_once()
            stages,manifest=load_run(root/'resumed')
            self.assertEqual(len(stages['judge']),2)
            self.assertEqual(manifest['reused_case_observations']['judge'],1)
            self.assertEqual(manifest['missing_judge_observations'],[])

    def test_partial_mutation_recovers_original_pending_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prior,_=fresh_checkpoint(root);calls=[]
            def mutate(project,pair):
                return {**pair,'mutated_mlir':pair['mutated_mlir']+'\n// mutation '+pair['program_id']+'\n'}
            def fail_second(*args):
                calls.append(args[1]['program_id'])
                if len(calls)==2:raise RuntimeError('Synthetic mutation failure')
                return mutate(*args)
            with patch('mlirbench.live.mutate',side_effect=fail_second):
                with self.assertRaises(RuntimeError):self.run_stage(root,'failed',prior,stage='mutation')
            self.assertEqual(load_run(root/'failed')[0],{})
            with patch('mlirbench.live.mutate',side_effect=mutate) as retry:
                self.run_stage(root,'resumed',root/'failed',stage='mutation',resume=True)
            retry.assert_called_once()
            self.assertEqual(retry.call_args.args[1]['program_id'],calls[1])
            stages,_=load_run(root/'resumed')
            self.assertEqual(len(stages['mutation']),2)
            self.assertEqual(len({r['program_id'] for r in stages['mutation']}),2)

    def test_full_pipeline_resume_reuses_all_completed_upstream_stages(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pair,_=fixture(root)
            options=dict(mode='fresh',models=['alpha'],allow_paid=True)
            report={'verdict':'COMPILE_ERROR','compilation':[{'status':'error'},{'status':'error'}]}
            answer={'model_key':'alpha','predicted_label':'ERROR'}
            with patch('mlirbench.live.mutate',return_value=pair), \
                 patch('mlirbench.execution.pair_validation.validate_pair',return_value=report), \
                 patch('mlirbench.live.reason',return_value=answer), \
                 patch('mlirbench.live.judge',side_effect=RuntimeError('Synthetic endpoint failure')):
                with self.assertRaises(RuntimeError):run_pipeline(root,root/'failed',**options)
            with patch('mlirbench.live.mutate',side_effect=AssertionError('Mutation repeated')), \
                 patch('mlirbench.execution.pair_validation.validate_pair',side_effect=AssertionError('Execution repeated')), \
                 patch('mlirbench.live.reason',side_effect=AssertionError('Reasoning repeated')), \
                 patch('mlirbench.live.judge',return_value=score_components('ERROR','ERROR',5,2)) as judge:
                result=run_pipeline(root,root/'resumed',input_run=root/'failed',resume_incomplete=True,**options)
            judge.assert_called_once()
            self.assertEqual(result['metrics']['scored_records'],1)
            self.assertEqual(load_run(root/'resumed')[1]['reused_stages'],['mutation','execution','reasoning'])

    def test_committed_cases_survive_failure_before_stage_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prior,_=fresh_checkpoint(root)
            def fail_stage(path,values):
                if path.name=='reasoning.jsonl':raise OSError('Synthetic checkpoint failure')
                return save_records(path,values)
            with patch('mlirbench.live.reason',side_effect=prediction),patch('mlirbench.pipeline.save_records',side_effect=fail_stage):
                with self.assertRaises(OSError):self.run_stage(root,'failed',prior)
            self.assertEqual(len(completed_cases(root/'failed')),2)
            with patch('mlirbench.live.reason',side_effect=AssertionError('Paid call repeated')):
                self.run_stage(root,'resumed',root/'failed',resume=True)
            self.assertEqual(len(load_run(root/'resumed')[0]['reasoning']),2)

    def test_abrupt_process_exit_preserves_first_response_and_marks_second_in_flight(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prior,_=fresh_checkpoint(root)
            script='''
import os,sys
from pathlib import Path
from unittest.mock import patch
from mlirbench.pipeline import run_pipeline
root=Path(sys.argv[1]);calls=[]
def reason(*args):
    calls.append(1)
    if len(calls)==2:os._exit(42)
    return {'model_key':'alpha','predicted_label':'EQUIV','explanation':'durable fixture'}
with patch('mlirbench.live.preflight',return_value={'test_fixture':True}),patch('mlirbench.live.reason',side_effect=reason):
    run_pipeline(root,root/'killed',stage='reasoning',mode='fresh',models=['alpha'],allow_paid=True,all_records=True,input_run=root/'prior')
'''
            process=subprocess.run([sys.executable,'-c',script,str(root)],capture_output=True,text=True)
            self.assertEqual(process.returncode,42,process.stderr)
            entries=completed_cases(root/'killed')
            self.assertEqual(sorted(e['attempts'][-1]['status'] for e in entries.values()),['COMPLETE','IN_FLIGHT'])
            with patch('mlirbench.live.reason',side_effect=prediction) as retry:
                self.run_stage(root,'resumed',root/'killed',resume=True)
            retry.assert_called_once()
            self.assertEqual(load_run(root/'resumed')[1]['status'],'COMPLETE')

    def test_keyboard_interrupt_records_interruption_and_preserves_completed_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prior,_=fresh_checkpoint(root)
            with patch('mlirbench.live.reason',side_effect=[prediction(None,None,None,'alpha'),KeyboardInterrupt()]):
                with self.assertRaises(KeyboardInterrupt):self.run_stage(root,'interrupted',prior)
            self.assertEqual(load_run(root/'interrupted')[1]['status'],'INTERRUPTED')
            self.assertEqual(sorted(e['attempts'][-1]['status'] for e in completed_cases(root/'interrupted').values()),['COMPLETE','INTERRUPTED'])

    def test_ordinary_input_still_reruns_the_requested_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prior,_=fresh_checkpoint(root)
            with patch('mlirbench.live.reason',side_effect=prediction) as first:
                self.run_stage(root,'first',prior)
            with patch('mlirbench.live.reason',side_effect=prediction) as second:
                self.run_stage(root,'second',root/'first')
            self.assertEqual(first.call_count,2);self.assertEqual(second.call_count,2)

    def test_changed_configuration_or_code_is_rejected_before_any_call(self):
        for change in ('configuration','code','model','selection'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root=Path(temp);prior,pairs=fresh_checkpoint(root)
                with patch('mlirbench.live.reason',side_effect=RuntimeError('fixture failure')):
                    with self.assertRaises(RuntimeError):self.run_stage(root,'failed',prior)
                if change=='code':
                    (root/'src').mkdir();(root/'src/changed.py').write_text('# new code\n')
                with patch('mlirbench.live.reason') as call:
                    with self.assertRaisesRegex(ValueError,'unchanged'):
                        with patch('mlirbench.live.preflight',return_value={'test_fixture':change} if change=='configuration' else {'test_fixture':True}):
                            options=dict(stage='reasoning',mode='fresh',allow_paid=True,all_records=True,
                                         models=['other'] if change=='model' else ['alpha'],
                                         ids=[pairs[0]['program_id']] if change=='selection' else None)
                            run_pipeline(root,root/'resumed',input_run=root/'failed',resume_incomplete=True,**options)
                call.assert_not_called()

    def test_corrupt_case_or_plan_is_rejected_before_resume(self):
        for part in ('case','plan'):
            with self.subTest(part=part), tempfile.TemporaryDirectory() as temp:
                root=Path(temp);prior,_=fresh_checkpoint(root)
                with patch('mlirbench.live.reason',side_effect=[prediction(None,None,None,'alpha'),RuntimeError('fixture failure')]):
                    with self.assertRaises(RuntimeError):self.run_stage(root,'failed',prior)
                path=(root/'failed/case-plan.json') if part=='plan' else next((root/'failed/cases/reasoning').glob('*.json'))
                value=json.loads(path.read_text());value['changed']=True;path.write_text(json.dumps(value))
                with patch('mlirbench.live.reason') as call:
                    with self.assertRaisesRegex(ValueError,'Modified'):
                        self.run_stage(root,'resumed',root/'failed',resume=True)
                call.assert_not_called()

    def test_cli_exposes_explicit_continuation_without_changing_default(self):
        from mlirbench import cli
        with patch('mlirbench.pipeline.run_pipeline',return_value={'fixture':True}) as run,contextlib.redirect_stdout(io.StringIO()):
            cli.main(['pipeline','--stage','reasoning','--mode','fresh','--model','alpha',
                      '--input','/results/stopped','--output','/results/new',
                      '--resume-incomplete','--all','--allow-paid-api'])
        self.assertTrue(run.call_args.kwargs['resume_incomplete'])
        self.assertTrue(run.call_args.kwargs['all_records'])

    def test_completed_run_cannot_accidentally_be_continued_as_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prior,_=fresh_checkpoint(root)
            with patch('mlirbench.live.reason',side_effect=prediction):self.run_stage(root,'complete',prior)
            with patch('mlirbench.live.reason') as call:
                with self.assertRaisesRegex(ValueError,'stopped incomplete'):
                    self.run_stage(root,'resumed',root/'complete',resume=True)
            call.assert_not_called()

    def test_metrics_do_not_consume_partial_judgments_as_a_completed_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prior,_=fresh_checkpoint(root)
            with patch('mlirbench.live.judge',side_effect=[score_components('EQUIV','EQUIV',5,2),RuntimeError('fixture failure')]):
                with self.assertRaises(RuntimeError):self.run_stage(root,'failed',prior,stage='judge')
            with self.assertRaisesRegex(ValueError,'missing judge'):
                run_pipeline(root,root/'metrics',stage='metrics',input_run=root/'failed')


class AtomicArtifacts(unittest.TestCase):
    def test_failed_rename_leaves_original_manifest_readable(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'manifest.json';atomic_json(path,{'state':'old'})
            with patch('mlirbench.stage_artifacts.os.replace',side_effect=OSError('Synthetic rename failure')):
                with self.assertRaises(OSError):atomic_json(path,{'state':'new'})
            self.assertEqual(json.loads(path.read_text()),{'state':'old'})
            self.assertEqual([p.name for p in path.parent.iterdir()],['manifest.json'])

    def test_stage_bytes_stay_compatible_and_existing_output_is_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'stage.jsonl';values=[{'fixture':1},{'fixture':2}]
            save_records(path,values)
            self.assertEqual(path.read_text(),''.join(json.dumps(v)+'\n' for v in values))
            self.assertEqual(path.stat().st_mode & 0o777,0o644)
            with self.assertRaises(FileExistsError):save_records(path,[])
