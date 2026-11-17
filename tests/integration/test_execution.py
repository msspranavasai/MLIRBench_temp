import importlib.util
import unittest
from unittest.mock import patch
from mlirbench.execution.pair_validation import validate_pair, deserialize_inputs
from tests.unit.test_pair_validation import IDENTITY, GATHER


@unittest.skipUnless(importlib.util.find_spec('iree'), 'IREE required; use Docker artifact')
class IreeExecution(unittest.TestCase):
    def test_signed_inputs_expose_a_difference_hidden_by_positive_only_inputs(self):
        absolute = '''module {
 func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {
 %e = tensor.empty() : tensor<2xf32>
 %r = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]} ins(%x : tensor<2xf32>) outs(%e : tensor<2xf32>) {
 ^bb0(%v: f32, %out: f32):
 %a = math.absf %v : f32
 linalg.yield %a : f32
 } -> tensor<2xf32>
 return %r : tensor<2xf32>
 } }'''
        report = validate_pair(IDENTITY, absolute, 'linalg', seeds=[42])
        self.assertEqual(report['verdict'], 'DIVERGE', report)
        self.assertEqual(report['trials'][0]['generation']['pattern'], 'negative_uniform')

    def test_identical_and_ssa_renamed(self):
        r = validate_pair(IDENTITY, IDENTITY.replace('%x', '%renamed'), 'linalg', seeds=[0,42])
        self.assertEqual(r['verdict'], 'NO_DIFFERENCE_OBSERVED', r)
        self.assertEqual(len(r['trials']), 2)
        self.assertNotEqual(r['trials'][0]['input_sha256'], r['trials'][1]['input_sha256'])
        replay = validate_pair(IDENTITY, IDENTITY, 'linalg', recorded_inputs=r['trials'][0]['inputs'])
        self.assertEqual(replay['trials'][0]['output_sha256'], r['trials'][0]['output_sha256'])

    def test_changed_op(self):
        b = IDENTITY.replace('return %x', '%one = arith.constant dense<1.0> : tensor<2xf32>\n %y = arith.addf %x, %one : tensor<2xf32>\n return %y')
        self.assertEqual(validate_pair(IDENTITY, b, 'linalg', seeds=[42])['verdict'], 'DIVERGE')

    def test_valid_gather_and_replay(self):
        r = validate_pair(GATHER, GATHER.replace('%data', '%a').replace('%idx', '%b'), 'torch', seeds=[42])
        self.assertEqual(r['verdict'], 'NO_DIFFERENCE_OBSERVED', r)
        values = r['trials'][0]['inputs']; values[1].pop('bytes_b64'); values[1]['data'] = [-1,0]
        r = validate_pair(GATHER, GATHER, 'torch', recorded_inputs=values)
        self.assertEqual(r['verdict'], 'INVALID_INPUT')
        self.assertNotIn('runs', r['trials'][0])


    def test_torch_identity_requires_explicit_lowering(self):
        s = 'module { func.func @main(%x: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> { return %x : !torch.vtensor<[2],f32> } }'
        r = validate_pair(s, s, 'torch', seeds=[42])
        self.assertEqual(r['verdict'], 'NO_DIFFERENCE_OBSERVED', r)
        self.assertEqual([c['input_type'] for c in r['compilation']], ['torch', 'torch'])
