"""Installed SDKs -> ordinary compiler -> real CPU runtime -> SDK judge.

All HTTP is intercepted with synthetic responses. These scores are test fixtures,
not empirical model results or a claim of current remote model access.
"""
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
from tests.unit.test_release_hardening import completion,KEY

ROOT=Path(__file__).resolve().parents[2]
live.enable(ROOT)
from agents.llm_reasoning_agent.model_registry import MODELS

LINALG='module { func.func @main(%x: tensor<1xf32>) -> tensor<1xf32> { return %x : tensor<1xf32> } }'
TORCH='module { func.func @main(%x: !torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32> { return %x : !torch.vtensor<[1],f32> } }'
ANSWER={'label':'EQUIV','confidence':0.9,'predicted_output_A':'input tensor',
        'predicted_output_B':'input tensor','explanation':'Synthetic fixture: both return input.',
        'key_differences':[],'witness':None}


class SdkPipeline(unittest.TestCase):
    def setUp(self):
        env=patch.dict(os.environ,{key:KEY for key in ('OPENAI_API_KEY','OPENROUTER_API_KEY','ANTHROPIC_API_KEY','GOOGLE_API_KEY')})
        env.start();self.addCleanup(env.stop)
        self.requests=[];self.mutation=LINALG
        transport=patch.object(httpx.Client,'send',self.send)
        transport.start();self.addCleanup(transport.stop)

    def send(self,request,*args,**kwargs):
        self.requests.append(request);body=json.loads(request.content)
        if request.url.path.endswith('/messages'):
            return httpx.Response(200,request=request,json={'id':'msg_fixture','type':'message','role':'assistant','model':body['model'],
                'content':[{'type':'text','text':json.dumps(ANSWER)}],'stop_reason':'end_turn','stop_sequence':None,
                'usage':{'input_tokens':1,'output_tokens':1}})
        if 'generateContent' in request.url.path:
            return httpx.Response(200,request=request,json={'candidates':[{'content':{'role':'model','parts':[{'text':json.dumps(ANSWER)}]},
                'finishReason':'STOP','index':0}],'usageMetadata':{'promptTokenCount':1,'candidatesTokenCount':1,'totalTokenCount':2}})
        content=(self.mutation if body['model']=='gpt-5-mini' else
                 json.dumps({'scores':{'result_accuracy':10,'explanation_rating':4},'evidence':{},'reasoning':'Synthetic judge fixture only'})
                 if body['model']=='gpt-5.4-mini' else json.dumps(ANSWER))
        return completion(request,content)

    def test_both_dialects_support_independent_fresh_stages_with_real_execution(self):
        for dialect,source,rule in [('linalg',LINALG,'L1_R1'),('torch',TORCH,'T1_R1')]:
            with self.subTest(dialect=dialect),tempfile.TemporaryDirectory() as temp:
                folder=Path(temp);file=folder/'source.mlir';file.write_text(source)
                self.mutation=source.replace('%x','%renamed')
                run_pipeline(ROOT,folder/'mutation',stage='mutation',mode='fresh',source=file,dialect=dialect,level=1,requirement=rule,allow_paid=True)
                prior=folder/'mutation'
                for stage in ('execution','reasoning','judge'):
                    out=folder/stage
                    run_pipeline(ROOT,out,stage=stage,mode='fresh',input_run=prior,
                                 models=['gpt-4.1-mini'] if stage!='execution' else None,allow_paid=stage!='execution')
                    prior=out
                summary=run_pipeline(ROOT,folder/'metrics',stage='metrics',input_run=prior)
                stages,manifest=load_run(folder/'metrics')
                self.assertEqual(manifest['status'],'COMPLETE')
                self.assertTrue(all(r['mode']=='fresh' for rows in stages.values() for r in rows))
                self.assertEqual(stages['execution'][0]['data']['ground_truth_label'],'EQUIV')
                self.assertEqual(summary['metrics']['scored_records'],1)
                self.assertNotEqual(summary['metrics']['scope'],'full_published_archive')
                self.assertFalse(any(KEY in p.read_text() for p in folder.rglob('*.json*')))

    def test_every_registered_reasoning_sdk_parses_a_synthetic_success(self):
        from agents.llm_reasoning_agent.node import reason_about_pair
        for model in MODELS:
            with self.subTest(model=model):
                record=reason_about_pair(LINALG,LINALG,'linalg',model,embedded_inputs={'%x':[1.]},max_retries=0)
                self.assertEqual(record['status'],'SUCCESS',record.get('error'))
