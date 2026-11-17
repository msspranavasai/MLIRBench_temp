"""Live configuration errors stop before paid calls; SDK requests have budgets."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from mlirbench import live
from mlirbench.pipeline import run_pipeline
from mlirbench.stage_artifacts import load_run
from tests.unit.test_pipeline import fixture

ROOT = Path(__file__).resolve().parents[2]
live.enable(ROOT)
from agents.api_config import request_limits
from agents.llm_reasoning_agent.model_registry import MODELS, get_spec, get_model_client
from agents.mutation_agent import llm as mutation_llm
from agents.judge_agent import node as judge_node

KEYS = {'OPENAI_API_KEY':'fixture-openai-not-a-real-key',
        'OPENROUTER_API_KEY':'fixture-router-not-a-real-key',
        'ANTHROPIC_API_KEY':'fixture-anthropic-not-a-real-key',
        'GOOGLE_API_KEY':'fixture-google-not-a-real-key'}


class LivePreflight(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, KEYS, clear=True)
        env.start(); self.addCleanup(env.stop)
        # SDK construction is allowed; actual HTTP is always forbidden here.
        transport = patch.object(httpx.Client, 'send', side_effect=AssertionError('Unexpected HTTP'))
        self.http = transport.start(); self.addCleanup(transport.stop)

    def test_exact_paper_gemini_is_registered_without_changing_legacy_alias(self):
        spec = get_spec('gemini-3.1-pro-preview')
        self.assertEqual(spec.model_id, 'gemini-3.1-pro-preview')
        self.assertEqual(spec.api_key_env, 'GOOGLE_API_KEY')
        self.assertEqual(get_spec('gemini').key, 'gemini-3-flash')
        paper = json.loads((ROOT/'experiments/paper_reference.json').read_text())
        keys = {check['path'][1]['model'] for check in paper['checks'] if check['path'][0] == 'by_level'}
        self.assertEqual(len(keys), 9)
        for key in keys:
            self.assertEqual(get_spec(key).key, key)

    def test_every_registry_sdk_constructs_without_http(self):
        for name in MODELS:
            with self.subTest(model=name):
                config = live.preflight(ROOT, ('reasoning',), [name])
                self.assertEqual(config['stages']['reasoning']['model_id'], get_spec(name).model_id)
                self.assertFalse(config['remote_access_verified'])
        self.http.assert_not_called()

    def pipeline_failure(self, message, *, model='gemini-3.1-pro-preview', env=None, sdk_failure=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(os.environ, env or {}), \
                 patch.object(live, 'enable'), \
                 patch('mlirbench.pipeline.ensure_data') as data, \
                 patch('mlirbench.live.mutate') as mutation, \
                 patch('mlirbench.live.reason') as reason, \
                 patch('mlirbench.live.judge') as judge:
                with contextlib.ExitStack() as stack:
                    if sdk_failure:
                        stack.enter_context(patch('agents.judge_agent.node._get_judge_llm', side_effect=ImportError('test SDK missing')))
                    with self.assertRaisesRegex((ValueError, RuntimeError, KeyError, ImportError), message):
                        run_pipeline(root, root/'out', mode='fresh', models=[model], allow_paid=True)
                for call in (data, mutation, reason, judge): call.assert_not_called()
                self.assertFalse((root/'out').exists())
        self.http.assert_not_called()

    def test_unknown_model_stops_before_mutation_or_data_access(self):
        self.pipeline_failure('Unknown model', model='missing-paper-model')

    def test_missing_later_reasoning_credentials_stops_before_mutation(self):
        self.pipeline_failure('GOOGLE_API_KEY', env={'GOOGLE_API_KEY':''})

    def test_all_missing_credentials_are_reported_without_values(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'GOOGLE_API_KEY, OPENAI_API_KEY'):
                live.preflight(ROOT, ('mutation','reasoning','judge'), ['gemini-3.1-pro-preview'])

    def test_later_judge_sdk_error_stops_before_mutation(self):
        self.pipeline_failure('test SDK missing', sdk_failure=True)

    def test_invalid_later_stage_limits_stop_before_mutation(self):
        for env in ({'MLIRBENCH_JUDGE_MAX_OUTPUT_TOKENS':'0'},
                    {'MLIRBENCH_API_TIMEOUT_SECONDS':'nan'},
                    {'MLIRBENCH_API_MAX_RETRIES':'-1'},
                    {'MLIRBENCH_MUTATION_MAX_ATTEMPTS':'1.5'},
                    {'OPENAI_MODEL':''}, {'MLIRBENCH_MUTATION_MODEL':'bad model'}):
            with self.subTest(env=env): self.pipeline_failure('must', env=env)

    def test_independent_stages_require_only_their_credentials(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY':KEYS['OPENAI_API_KEY']}, clear=True):
            self.assertEqual(set(live.preflight(ROOT, ('mutation',))['stages']), {'mutation'})
            self.assertEqual(set(live.preflight(ROOT, ('judge',), ['gemini-3.1-pro-preview'])['stages']), {'judge'})
        with patch.dict(os.environ, {'GOOGLE_API_KEY':KEYS['GOOGLE_API_KEY']}, clear=True):
            self.assertEqual(set(live.preflight(ROOT, ('reasoning',), ['gemini-3.1-pro-preview'])['stages']), {'reasoning'})

    def test_fresh_run_records_resolved_configuration_without_credentials(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); pair,_ = fixture(root)
            with patch.object(live, 'enable'), patch('mlirbench.live.mutate', return_value=pair):
                run_pipeline(root, root/'out', stage='mutation', mode='fresh', allow_paid=True)
            _, manifest = load_run(root/'out')
            config = manifest['live_configuration']
            self.assertEqual(config['stages']['mutation']['model_id'], 'gpt-5-mini')
            self.assertEqual(config['stages']['mutation']['agent_attempts'], 3)
            for secret in KEYS.values(): self.assertNotIn(secret, json.dumps(manifest))

    def test_replay_and_fresh_cpu_execution_do_not_require_live_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root)
            with patch.dict(os.environ, {}, clear=True), patch.object(live, 'preflight', side_effect=AssertionError('Unexpected live preflight')):
                run_pipeline(root, root/'replay', stage='judge')
                with patch('mlirbench.execution.pair_validation.validate_pair', return_value={'verdict':'UNSUPPORTED_INPUT_PRECONDITIONS'}):
                    run_pipeline(root, root/'cpu', stage='execution', mode='fresh')


class RequestBudgets(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, KEYS, clear=True)
        env.start(); self.addCleanup(env.stop)

    def test_actual_mutation_and_judge_requests_carry_limits(self):
        captured = []
        def send(client, request, *args, **kwargs):
            body = json.loads(request.content)
            captured.append((request, body))
            if request.url.path.endswith('/responses'):
                content = {'id':'resp_fixture','object':'response','created_at':0,'model':body['model'],
                           'status':'completed','output':[{'type':'message','id':'msg_fixture','role':'assistant',
                           'status':'completed','content':[{'type':'output_text','annotations':[],'text':'module {}'}]}]}
            else:
                content = {'id':'chat_fixture','object':'chat.completion','created':0,'model':body['model'],
                           'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'{}'}}]}
            return httpx.Response(200, request=request, json=content)
        with patch.dict(os.environ, {'MLIRBENCH_API_TIMEOUT_SECONDS':'17',
                                    'MLIRBENCH_MUTATION_MAX_OUTPUT_TOKENS':'1234',
                                    'MLIRBENCH_JUDGE_MAX_OUTPUT_TOKENS':'567',
                                    'MLIRBENCH_MUTATION_MODEL':'fixture-mutation', 'OPENAI_MODEL':'fixture-judge'}), \
             patch.object(httpx.Client, 'send', send):
            self.assertEqual(mutation_llm.generate_mutation('fixture'), 'module {}')
            judge = judge_node._get_judge_llm()
            judge.invoke('fixture')
            self.assertEqual(judge.max_retries, 0)
        self.assertEqual(captured[0][1]['model'], 'fixture-mutation')
        self.assertEqual(captured[0][1]['max_output_tokens'], 1234)
        self.assertEqual(captured[1][1]['model'], 'fixture-judge')
        self.assertEqual(captured[1][1]['max_completion_tokens'], 567)
        for request,_ in captured:
            self.assertEqual(request.extensions['timeout']['read'], 17)

    def test_gemini_request_uses_exact_paper_model_id(self):
        requests = []
        def send(client, request, *args, **kwargs):
            requests.append(request)
            return httpx.Response(200, request=request, json={
                'candidates':[{'content':{'role':'model','parts':[{'text':'fixture response'}]},
                               'finishReason':'STOP','index':0}],
                'usageMetadata':{'promptTokenCount':1,'candidatesTokenCount':1,'totalTokenCount':2}})
        with patch.object(httpx.Client, 'send', send):
            response = get_model_client('gemini-3.1-pro-preview').invoke('fixture')
        self.assertTrue(requests)
        self.assertTrue(all('gemini-3.1-pro-preview' in str(r.url) for r in requests))
        self.assertNotIn('gemini-3-flash', str([r.url for r in requests]))

    def test_mutation_sdk_does_not_silently_retry_failed_request(self):
        requests = []
        def send(client, request, *args, **kwargs):
            requests.append(request)
            return httpx.Response(503, request=request, json={'error':{'message':'fixture failure'}})
        with patch.object(httpx.Client, 'send', send):
            with self.assertRaises(Exception): mutation_llm.generate_mutation('fixture')
        self.assertEqual(len(requests), 1)

    def test_reasoning_clients_obey_token_timeout_and_retry_budgets(self):
        with patch.dict(os.environ, {'MLIRBENCH_REASONING_MAX_OUTPUT_TOKENS':'789',
                                    'MLIRBENCH_API_TIMEOUT_SECONDS':'19', 'MLIRBENCH_API_MAX_RETRIES':'1'}):
            for name in ('gpt-4.1-mini','qwen3-coder-next','claude-sonnet-4-5','gemini-3.1-pro-preview'):
                with self.subTest(model=name):
                    client = get_model_client(name)
                    self.assertEqual(client.max_retries, 1)
                    tokens = client.max_output_tokens if get_spec(name).via == 'google' else client.max_tokens
                    self.assertEqual(tokens, 789)
                    timeout = getattr(client, 'request_timeout', getattr(client, 'default_request_timeout', getattr(client, 'timeout', None)))
                    self.assertEqual(timeout, 19)

    def test_mutation_attempt_budget_controls_original_agent_loop(self):
        from agents.mutation_agent import node
        from tests.unit.test_pipeline import SOURCE
        payload = {'input_mlir':SOURCE,'dialect':'linalg','level':1,'requirement_id':'L1_R1'}
        with patch.dict(os.environ, {'MLIRBENCH_MUTATION_MAX_ATTEMPTS':'2'}), \
             patch.object(node, 'generate_mutation', side_effect=RuntimeError('fixture API failure')) as call, \
             contextlib.redirect_stdout(io.StringIO()):
            node.mutation_agent(payload)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(request_limits('mutation', 8192)['agent_attempts'], 3)
