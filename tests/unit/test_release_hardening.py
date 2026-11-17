"""Failure paths use real SDK serialization and compiler validation, without networking."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import httpx
from mlirbench import live
from mlirbench.case_journal import CaseJournal
from mlirbench.observations import ObservationError, redact
from mlirbench.preservation import verify_preservation
from tests.unit.test_pipeline import SOURCE

ROOT=Path(__file__).resolve().parents[2]
live.enable(ROOT)
from agents.mutation_agent import node as mutation
from agents.llm_reasoning_agent import node as reasoning
from agents.llm_reasoning_agent.model_registry import get_model_client, get_spec

KEY='fixture-release-private-credential-9347'
BAD='module { func.func @main(%x: tensor<1xf32>) -> tensor<2xf32> { return %x : tensor<1xf32> } }'


def completion(request,text):
    model=json.loads(request.content)['model']
    if request.url.path.endswith('/responses'):
        return httpx.Response(200,request=request,json={'id':'resp_fixture','object':'response','created_at':0,
            'model':model,'status':'completed','output':[{'type':'message','id':'msg_fixture','role':'assistant',
            'status':'completed','content':[{'type':'output_text','annotations':[],'text':text}]}]})
    return httpx.Response(200,request=request,json={'id':'chat_fixture','object':'chat.completion','created':0,
        'model':model,'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':text}}]})


class Generation(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{'OPENAI_API_KEY':KEY,'MLIRBENCH_MUTATION_MAX_ATTEMPTS':'2'})
        self.env.start();self.addCleanup(self.env.stop)
        self.payload={'program_id':'new-source','input_mlir':SOURCE,'dialect':'linalg','level':1,'requirement_id':'L1_R1'}

    def generate(self,candidate,payload=None):
        with patch.object(mutation,'generate_mutation',return_value=candidate):
            return live.mutate(ROOT,payload or self.payload)

    def test_invalid_nonempty_generation_stops_instead_of_falling_back(self):
        with self.assertRaises(ObservationError) as ctx:self.generate('not MLIR')
        self.assertEqual(ctx.exception.observation['generation_status'],'FAILED')
        self.assertEqual(len(ctx.exception.observation['mlir-opt-verify-diagnostics']['history']),2)

    def test_valid_mutation_is_accepted(self):
        self.assertEqual(self.generate(SOURCE)['mutation_observation']['generation_status'],'VALIDATED')

    def test_level5_accepts_declared_invalid_mutation_but_not_garbage_or_valid_code(self):
        payload={**self.payload,'level':5,'requirement_id':'L5_R1'}
        result=self.generate(BAD,payload)
        self.assertEqual(result['mutation_observation']['generation_status'],'EXPECTED_INVALID')
        self.assertFalse(result['mutation_observation']['mlir-opt-verify-diagnostics']['final_validation']['valid'])
        for bad in ('not MLIR',BAD[:-2],SOURCE,'module { func.func nonsense }',
                    'module { func.func @main() { totally.random } }'):
            with self.subTest(candidate=bad),self.assertRaises(ObservationError):self.generate(bad,payload)

    def test_dependency_breaking_level5_accepts_only_recognized_ssa_error(self):
        candidate=SOURCE.replace('return %x','return %undefined')
        payload={**self.payload,'level':5,'requirement_id':'L5_R4'}
        self.assertEqual(self.generate(candidate,payload)['mutation_observation']['generation_status'],'EXPECTED_INVALID')
        with self.assertRaises(ObservationError):self.generate(candidate,{**payload,'requirement_id':'L5_R1'})

    def test_invalid_level5_original_stops_before_generation(self):
        with patch.object(mutation,'generate_mutation') as call,self.assertRaisesRegex(ValueError,'valid original'):
            live.mutate(ROOT,{**self.payload,'input_mlir':BAD,'level':5,'requirement_id':'L5_R1'})
        call.assert_not_called()

    def test_expected_error_comments_cannot_bypass_ordinary_verification(self):
        candidate=BAD.replace('return %x','\n// expected-error@+1 {{type of return operand}}\nreturn %x')
        with self.assertRaises(ObservationError):self.generate(candidate)

    def test_unavailable_validator_stops_without_more_generation(self):
        from services.mlir.validator_client import ValidatorInfrastructureError
        with patch.object(mutation,'validate_mutation',side_effect=ValidatorInfrastructureError('missing')), \
             patch.object(mutation,'generate_mutation',return_value=SOURCE) as call, \
             self.assertRaises(ValidatorInfrastructureError):live.mutate(ROOT,self.payload)
        self.assertEqual(call.call_count,1)

    def test_actual_sdk_retry_history_redacts_echoed_credential(self):
        requests=[]
        def send(client,request,*args,**kwargs):
            requests.append(request)
            if len(requests)==1:return httpx.Response(401,request=request,json={'error':{'message':'Rejected API key '+KEY}})
            return completion(request,SOURCE)
        with patch.object(httpx.Client,'send',send):result=live.mutate(ROOT,self.payload)
        self.assertEqual(len(requests),2)
        history=result['mutation_observation']['mlir-opt-verify-diagnostics']['history']
        self.assertIn('[REDACTED]',json.dumps(history));self.assertNotIn(KEY,json.dumps(result))
        self.assertTrue(history[-1]['validation']['valid'])


class FreshDiagnostics(unittest.TestCase):
    def test_failed_reasoning_keeps_full_response_and_journal_evidence(self):
        raw='not json '+('x'*600)+' retained-tail '+KEY
        def send(client,request,*args,**kwargs):return completion(request,raw)
        pair={'program_id':'example','input_mlir':SOURCE,'mutated_mlir':SOURCE,'dialect':'linalg'}
        with patch.dict(os.environ,{'OPENAI_API_KEY':KEY}),patch.object(httpx.Client,'send',send),tempfile.TemporaryDirectory() as temp:
            journal=CaseJournal(temp,{'schema':'mlirbench-case-plan-v1'})
            with self.assertRaises(ObservationError):
                journal.call('reasoning',pair,lambda:live.reason(ROOT,pair,{},'gpt-4.1-mini'))
            saved=next((Path(temp)/'cases/reasoning').glob('*.json')).read_text()
            self.assertNotIn(KEY,saved);self.assertIn('retained-tail',saved);self.assertIn('PARSE_ERROR',saved)
            attempt=json.loads(saved)['attempts'][-1]
            self.assertEqual(attempt['status'],'FAILED');self.assertNotIn('record',attempt)
            self.assertEqual(attempt['failure_observation']['raw_response_sha256'],hashlib.sha256(raw.encode()).hexdigest())

    def test_successful_records_with_credential_echo_are_not_committed(self):
        pair={'program_id':'example','input_mlir':SOURCE,'mutated_mlir':SOURCE,'dialect':'linalg'}
        with patch.dict(os.environ,{'OPENAI_API_KEY':KEY}),tempfile.TemporaryDirectory() as temp:
            journal=CaseJournal(temp,{'schema':'mlirbench-case-plan-v1'})
            with self.assertRaises(ObservationError):journal.call('reasoning',pair,lambda:{'response':KEY})
            saved=next((Path(temp)/'cases/reasoning').glob('*.json')).read_text()
            self.assertNotIn(KEY,saved);self.assertNotIn('COMPLETE',saved)

    def test_nonfinite_confidence_is_an_explicit_schema_failure(self):
        raw='{"label":"EQUIV","confidence":NaN,"explanation":"same","predicted_output_A":"same","predicted_output_B":"same"}'
        def send(client,request,*args,**kwargs):return completion(request,raw)
        with patch.dict(os.environ,{'OPENAI_API_KEY':KEY}),patch.object(httpx.Client,'send',send):
            result=reasoning.reason_about_pair(SOURCE,SOURCE,'linalg','gpt-4.1-mini',max_retries=0)
        self.assertEqual(result['status'],'SCHEMA_ERROR')

    def test_header_and_text_redaction(self):
        with patch.dict(os.environ,{'OPENAI_API_KEY':KEY}):
            value=redact({'headers':{'Authorization':'Bearer '+KEY},'error':'api_key="unknown-token" '+KEY})
        self.assertNotIn(KEY,json.dumps(value));self.assertNotIn('unknown-token',json.dumps(value))

    def test_openrouter_uses_chat_completions_with_preserved_reasoning_body(self):
        captured=[]
        def send(client,request,*args,**kwargs):
            captured.append(request);return completion(request,'fixture')
        with patch.dict(os.environ,{'OPENROUTER_API_KEY':KEY}),patch.object(httpx.Client,'send',send):
            for model in ('glm-5.1','kimi-k2.6','gpt-oss-120b'):
                get_model_client(model).invoke('fixture')
                request=captured[-1];body=json.loads(request.content)
                self.assertEqual(request.url.path,'/api/v1/chat/completions')
                self.assertEqual(body['reasoning'],get_spec(model).extra_body['reasoning'])
                self.assertEqual(body['max_completion_tokens'],4096)


class Preservation(unittest.TestCase):
    def test_recomputed_results_and_frozen_inputs_must_both_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'experiments').mkdir();results=root/'out';results.mkdir()
            (root/'input').write_bytes(b'frozen');(results/'score').write_bytes(b'calculated')
            spec={'schema':'mlirbench-release-invariants-v1','frozen_inputs':{'input':hashlib.sha256(b'frozen').hexdigest()},
                  'recomputed_outputs':{'score':{'mode':'bytes','sha256':hashlib.sha256(b'calculated').hexdigest()}}}
            (root/'experiments/release_invariants.json').write_text(json.dumps(spec))
            self.assertTrue(verify_preservation(root,results)['passed'])
            (results/'score').write_bytes(b'changed');self.assertFalse(verify_preservation(root,results)['passed'])
            (results/'score').write_bytes(b'calculated');(root/'input').unlink()
            self.assertFalse(verify_preservation(root,results)['passed'])

    def test_environment_file_variants_are_ignored_at_any_depth(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shutil.copy2(ROOT/'.gitignore',root/'.gitignore')
            subprocess.run(['git','init','-q',str(root)],check=True)
            names=['.env','.env.local','.env.live','.env.production','nested/.env.local']
            result=subprocess.run(['git','check-ignore','--stdin'],input='\n'.join(names)+'\n',cwd=root,capture_output=True,text=True,check=True)
            self.assertEqual(result.stdout.splitlines(),names)
