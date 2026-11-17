"""Optional live judging rejects malformed observations and preserves the rubric."""
import contextlib
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from mlirbench import live
from mlirbench.pipeline import run_pipeline,score_components
from mlirbench.stage_artifacts import load_run
from tests.unit.test_pipeline import fixture,SOURCE

ROOT=Path(__file__).resolve().parents[2]
ENABLE=live.enable
ENABLE(ROOT)
from agents.judge_agent import node
from agents.mutation_agent.selector import get_requirement_by_id

VALID={'scores':{'result_accuracy':5.5,'explanation_rating':3},'evidence':{},'reasoning':'Synthetic assessment only'}
EXECUTION={'ground_truth_label':'EQUIV','compilation_status_A':'success','compilation_status_B':'success',
           'programA_output':[1.],'programB_output':[1.],'outputs_diverge':False,
           'embedded_inputs':{'%x':[1.]},'label_scope':'declared_valid_input_suite',
           'validation_verdict':'NO_DIFFERENCE_OBSERVED'}
PREDICTION={'model_key':'fixture-model','predicted_label':'EQUIV','predicted_output_A':[1.],
            'predicted_output_B':[1.],'explanation':'Synthetic explanation only'}
PAIR={'program_id':'fixture','input_mlir':SOURCE,'mutated_mlir':SOURCE,
      'dialect':'linalg','level':1,'requirement_id':'L1_R1'}


def response(raw,finish='stop'):
    def send(client,request,*args,**kwargs):
        body=json.loads(request.content)
        return httpx.Response(200,request=request,json={
            'id':'chat_fixture','object':'chat.completion','created':0,'model':body['model'],
            'choices':[{'index':0,'finish_reason':finish,'message':{'role':'assistant','content':raw}}]})
    return send


class StrictJudgeResponses(unittest.TestCase):
    def setUp(self):
        env=patch.dict(os.environ,{'OPENAI_API_KEY':'fixture-not-a-real-key'})
        env.start();self.addCleanup(env.stop)

    def judge(self,raw,finish='stop',execution=None):
        with patch.object(httpx.Client,'send',response(raw,finish)),contextlib.redirect_stdout(io.StringIO()):
            return live.judge(ROOT,PAIR,execution or EXECUTION,PREDICTION)

    def test_actual_sdk_rejects_missing_wrong_type_nonfinite_and_out_of_range_scores(self):
        cases=[{},None,[],{'scores':{}},
               {**VALID,'scores':{'result_accuracy':5}},
               {**VALID,'scores':{'explanation_rating':2}},
               {**VALID,'scores':{'result_accuracy':'5','explanation_rating':2}},
               {**VALID,'scores':{'result_accuracy':True,'explanation_rating':2}},
               {**VALID,'scores':{'result_accuracy':5,'explanation_rating':False}},
               {**VALID,'scores':{'result_accuracy':-1,'explanation_rating':2}},
               {**VALID,'scores':{'result_accuracy':11,'explanation_rating':2}},
               {**VALID,'scores':{'result_accuracy':5,'explanation_rating':999}},
               {**VALID,'scores':{'result_accuracy':5,'explanation_rating':-1}},
               {**VALID,'scores':{'result_accuracy':5,'explanation_rating':3.5}},
               {**VALID,'scores':{'result_accuracy':float('nan'),'explanation_rating':2}},
               {**VALID,'scores':{'result_accuracy':float('inf'),'explanation_rating':2}},
               {**VALID,'scores':{'result_accuracy':5,'explanation_rating':float('-inf')}},
               {**VALID,'reasoning':None},{**VALID,'reasoning':''},
               {**VALID,'evidence':[]}, {**VALID,'scores':None}]
        for value in cases:
            raw=json.dumps(value)
            with self.subTest(raw=raw):
                with self.assertRaises(node.JudgeResponseError) as caught:self.judge(raw)
                self.assertEqual(caught.exception.observation['raw_response'],raw)
                self.assertEqual(caught.exception.observation['raw_response_sha256'],hashlib.sha256(raw.encode()).hexdigest())
                self.assertNotIn('total_score',caught.exception.observation)

    def test_truncated_duplicate_key_and_overflowing_json_are_rejected(self):
        for raw,finish in [('not JSON','stop'), ('{"scores":','stop'),
                           ('{"scores":{},"scores":{}}','stop'),
                           ('{"scores":{"result_accuracy":1e999,"explanation_rating":2}}','stop'),
                           (json.dumps(VALID),'length')]:
            with self.subTest(raw=raw,finish=finish):
                with self.assertRaises(node.JudgeResponseError) as caught:self.judge(raw,finish)
                self.assertEqual(caught.exception.observation['raw_response'],raw)

    def test_valid_zero_and_fractional_result_scores_keep_original_arithmetic(self):
        for result,explanation in [(0,0),(5.5,3),(10,4.0)]:
            body={**VALID,'scores':{'result_accuracy':result,'explanation_rating':explanation}}
            raw=json.dumps(body);scored=self.judge(raw)
            self.assertEqual(scored['total_score'],score_components('EQUIV','EQUIV',result,explanation)['total_score'])
            self.assertEqual(scored['judge_details']['judge_response_validation'],'PASS')
            self.assertEqual(scored['judge_details']['judge_response_observation']['raw_response'],raw)

    def test_invalid_scores_cannot_be_hidden_by_the_error_label_override(self):
        execution={**EXECUTION,'ground_truth_label':'ERROR','compilation_status_A':'error',
                   'programA_output':None,'outputs_diverge':None,'validation_verdict':'COMPILE_ERROR'}
        with self.assertRaises(node.JudgeResponseError):self.judge('{}',execution=execution)

    def test_supplied_component_limits_must_match_the_mutation_rubric(self):
        for components in ({'shape_structure_points':2}, {'shape_structure_max':2},
                           {'exactness_points':7}, {'exactness_max':5},
                           {'value_precision_points':4}, {'error_prediction_points':11}):
            with self.subTest(components=components):
                body={**VALID,'evidence':{'result_accuracy_components':components}}
                with self.assertRaises(node.JudgeResponseError):self.judge(json.dumps(body))
        body={**VALID,'evidence':{'result_accuracy_components':{'shape_structure_points':1,
             'shape_structure_max':1,'exactness_points':1.5,'exactness_max':6,'value_precision_points':3}}}
        self.assertEqual(self.judge(json.dumps(body))['result_accuracy'],5.5)

    def test_transport_error_is_not_converted_to_a_numeric_judgment(self):
        with patch.object(httpx.Client,'send',side_effect=httpx.ConnectError('Synthetic transport failure')):
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(Exception):
                live.judge(ROOT,PAIR,EXECUTION,PREDICTION)

    def test_invalid_response_is_durably_preserved_as_a_failed_unscored_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root);raw='{}'
            with patch('mlirbench.live.enable',side_effect=lambda _:ENABLE(ROOT)), \
                 patch('mlirbench.live.preflight',return_value={'test_fixture':True}), \
                 patch.object(httpx.Client,'send',response(raw)),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(node.JudgeResponseError):
                    run_pipeline(root,root/'failed',stage='judge',mode='fresh',models=['fixture-model'],allow_paid=True)
            stages,manifest=load_run(root/'failed')
            self.assertEqual(manifest['status'],'FAILED');self.assertNotIn('judge',stages)
            entries=list((root/'failed/cases/judge').glob('*.json'));self.assertEqual(len(entries),1)
            attempt=json.loads(entries[0].read_text())['attempts'][-1]
            self.assertEqual(attempt['status'],'FAILED');self.assertNotIn('record',attempt)
            self.assertEqual(attempt['failure_observation']['raw_response'],raw)
            self.assertNotIn('fixture-not-a-real-key',entries[0].read_text())
            with patch('mlirbench.live.enable',side_effect=lambda _:ENABLE(ROOT)), \
                 patch('mlirbench.live.preflight',return_value={'test_fixture':True}), \
                 patch.object(httpx.Client,'send',response(json.dumps(VALID))),contextlib.redirect_stdout(io.StringIO()):
                run_pipeline(root,root/'resumed',stage='judge',mode='fresh',models=['fixture-model'],allow_paid=True,
                             input_run=root/'failed',resume_incomplete=True)
            stages,manifest=load_run(root/'resumed');self.assertEqual(len(stages['judge']),1)
            attempts=json.loads(next((root/'resumed/cases/judge').glob('*.json')).read_text())['attempts']
            self.assertEqual([a['status'] for a in attempts],['FAILED','COMPLETE'])
            self.assertEqual(attempts[0]['failure_observation']['raw_response'],raw)


class MutationRubricContext(unittest.TestCase):
    def capture(self,pair):
        contexts=[]
        def invoke(*args):contexts.append(copy.deepcopy(args[3]));return copy.deepcopy(VALID)
        with patch.object(node,'_invoke_llm_judge',side_effect=invoke),contextlib.redirect_stdout(io.StringIO()):
            scored=live.judge(ROOT,pair,EXECUTION,PREDICTION)
        return contexts[0],scored

    def test_adapter_and_preserved_graph_use_identical_rule_specific_rubrics(self):
        from graph.pipeline import judge_node
        for dialect,level,rid in [('linalg',1,'L1_R1'),('linalg',2,'L2_R1'),('torch',1,'T1_R1')]:
            with self.subTest(rule=rid):
                rule=get_requirement_by_id(dialect,level,rid);metadata={'variables':['%x'],'fixture':rid}
                pair={**PAIR,'dialect':dialect,'level':level,'requirement_id':rid,
                      'mutation_observation':{'requirement':rule,'metadata':metadata}}
                actual,scored=self.capture(pair);contexts=[]
                def invoke(*args):contexts.append(copy.deepcopy(args[3]));return copy.deepcopy(VALID)
                state={'toExecutionAgent':{'input_mlir':SOURCE,'mutated_mlir':SOURCE,'dialect':dialect},
                       'Agents':{'Mutation_Agent':{'context_results':{'requirement':rule,'metadata':metadata}},
                                 'Execution_Agent':{'execution_results':EXECUTION},
                                 'LLM_Reasoning_Agent':{'results':[PREDICTION]}}}
                with patch.object(node,'_invoke_llm_judge',side_effect=invoke),contextlib.redirect_stdout(io.StringIO()):
                    judge_node(state)
                self.assertEqual(actual['rule'],contexts[0]['rule'])
                self.assertEqual(actual['metadata'],contexts[0]['metadata'])
                self.assertEqual(actual['structure_rubric'],contexts[0]['structure_rubric'])
                if rid=='L1_R1':
                    self.assertEqual(actual['structure_rubric']['structure_points_max'],1)
                    self.assertEqual(actual['structure_rubric']['exactness_points_max'],6)
                self.assertEqual(scored['judge_details']['inspection_metadata'],metadata)

    def test_archived_pair_resolves_catalog_and_retains_recorded_inspection(self):
        metadata={'variables':['%x'],'shapes':{'%x':[1]}}
        context,scored=self.capture({**PAIR,'pipeline_metadata':metadata})
        self.assertEqual(context['rule'],get_requirement_by_id('linalg',1,'L1_R1'))
        self.assertEqual(context['metadata'],metadata)
        source=scored['judge_details']['mutation_context_provenance']
        self.assertIn('packaged_rule_catalog',source['rule_source'])
        self.assertEqual(source['metadata_source'],'archived_pipeline_metadata')
        self.assertEqual(metadata,{'variables':['%x'],'shapes':{'%x':[1]}})

    def test_recorded_rule_is_preserved_instead_of_replaced_by_catalog_text(self):
        rule=copy.deepcopy(get_requirement_by_id('linalg',1,'L1_R1'));rule['description']='Recorded custom inspection rule'
        context,_=self.capture({**PAIR,'mutation_observation':{'requirement':rule,'metadata':{}}})
        self.assertEqual(context['rule'],rule)

    def test_unknown_inconsistent_and_incomplete_rule_contexts_stop_before_api(self):
        rule=get_requirement_by_id('linalg',1,'L1_R1')
        pairs=[{**PAIR,'requirement_id':'missing'}, {**PAIR,'requirement_id':None},
               {**PAIR,'level':2}, {**PAIR,'dialect':'torch'},
               {**PAIR,'mutation_observation':{'requirement':{**rule,'id':'wrong'}}},
               {**PAIR,'mutation_observation':{'requirement':{'id':'L1_R1'}}},
               {**PAIR,'pipeline_metadata':[]},
               {**PAIR,'mutation_observation':{'requirement':rule},'mutation_rule':{**rule,'description':'conflict'}}]
        for pair in pairs:
            with self.subTest(pair=pair):
                with patch.object(node,'_invoke_llm_judge') as call:
                    with self.assertRaises(ValueError):live.judge(ROOT,pair,EXECUTION,PREDICTION)
                call.assert_not_called()

    def test_direct_node_refuses_known_rule_id_without_full_rule(self):
        payload={'program_id':'fixture','programA':SOURCE,'programB':SOURCE,'dialect':'linalg',
                 'mutation_id':'L1_R1','prediction':PREDICTION,**EXECUTION}
        with patch.object(node,'_invoke_llm_judge') as call,contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'complete matching rule'):node.judge_agent({'judge_input':payload})
        call.assert_not_called()

    def test_rule_context_cannot_be_overridden_by_conflicting_nested_rubric(self):
        pair={**PAIR,'metadata':{'dynamic_structure_rubric':{'structure_points_max':2,'exactness_points_max':5}}}
        with patch.object(node,'_invoke_llm_judge') as call,contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'conflicts'):live.judge(ROOT,pair,EXECUTION,PREDICTION)
        call.assert_not_called()

    def test_unverified_execution_remains_unscored_without_a_judge_call(self):
        with patch.object(node,'_invoke_llm_judge') as call:
            result=live.judge(ROOT,PAIR,{'ground_truth_label':None},PREDICTION)
        self.assertIsNone(result['total_score']);call.assert_not_called()

    def test_every_published_pair_has_a_resolvable_rule_and_recorded_metadata(self):
        rule_ids=set();count=0
        with gzip.open(ROOT/'data/bundle/MLIRBench/master/master.jsonl.gz','rt') as stream:
            for line in stream:
                pair=json.loads(line);context=live.mutation_context(ROOT,pair)
                self.assertEqual(context['mutation_id'],pair['requirement_id'])
                self.assertEqual(context['metadata'],pair['pipeline_metadata'])
                rule_ids.add((pair['dialect'],pair['level'],pair['requirement_id']));count+=1
        self.assertEqual(count,1768);self.assertEqual(len(rule_ids),46)
