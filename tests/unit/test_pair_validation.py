import unittest
from unittest.mock import patch
import numpy as np
from mlirbench.execution.execution import parse_input_spec, typed_output, compare_outputs
from mlirbench.execution.pair_validation import (validate_pair, serialize_inputs, deserialize_inputs, compare_typed, digest, published_label)
from mlirbench.execution.input_policy import (InputPolicyError, check_inputs, generate_inputs)

IDENTITY = 'module { func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> { return %x : tensor<2xf32> } }'
GATHER = '''module {
func.func @main(%data: !torch.vtensor<[2,3],f32>, %idx: !torch.vtensor<[2],si64>) -> !torch.vtensor<[2,1],f32> {
 %dim = torch.constant.int 1
 %false = torch.constant.bool false
 %i = torch.aten.unsqueeze %idx, %dim : !torch.vtensor<[2],si64>, !torch.int -> !torch.vtensor<[2,1],si64>
 %r = torch.aten.gather %data, %dim, %i, %false : !torch.vtensor<[2,3],f32>, !torch.int, !torch.vtensor<[2,1],si64>, !torch.bool -> !torch.vtensor<[2,1],f32>
 return %r : !torch.vtensor<[2,1],f32>
} }'''


class Contracts(unittest.TestCase):
    def test_unknown_dynamic_and_missing_signatures(self):
        for source in ('no main', IDENTITY.replace('2xf32', '?xf32'), IDENTITY.replace('2xf32', '2xbf16'), IDENTITY.replace('tensor<2xf32>', '!unknown')):
            with self.subTest(source=source), self.assertRaises(ValueError): parse_input_spec(source)
        with self.assertRaises(ValueError): parse_input_spec(GATHER.replace('[2,3]', '[?,3]'))

    def test_incompatible_dtype_never_compiles(self):
        with patch('mlirbench.execution.pair_validation.compile_mlir') as compile:
            r = validate_pair(IDENTITY, IDENTITY.replace('f32', 'f64'), 'linalg')
        self.assertEqual(r['verdict'], 'INCOMPARABLE_INPUT_SIGNATURES'); compile.assert_not_called()

    def test_typed_roundtrip_scalars_empty_and_signed_zero(self):
        values = [np.array(-0., dtype=np.float32), np.empty((0, 3), np.int64)]
        specs = [{'shape': list(v.shape), 'dtype': str(v.dtype)} for v in values]
        items = serialize_inputs(values); result = deserialize_inputs(items, specs)
        self.assertEqual(digest(items), digest(serialize_inputs(result)))
        self.assertTrue(np.signbit(result[0]))
        items[0]['dtype'] = 'float64'
        with self.assertRaises(InputPolicyError): deserialize_inputs(items, specs)

    def test_exact_value_contract(self):
        def t(v, dtype='float32'): return typed_output(np.asarray(v, dtype=dtype))
        self.assertEqual(compare_typed(t(0.), t(-0.)), 'EQUAL')
        self.assertNotEqual(digest(t(0.)), digest(t(-0.)))
        self.assertEqual(compare_typed(t(1), t(1, 'float64')), 'DIFFERENT')
        self.assertEqual(compare_typed(t([1]), t([[1]])), 'DIFFERENT')
        self.assertEqual(compare_typed(t(1), t(np.nextafter(np.float32(1), np.float32(2)))), 'DIFFERENT')
        for v in (np.nan, np.inf, -np.inf): self.assertEqual(compare_typed(t(v), t(v)), 'NONFINITE')
        self.assertFalse(compare_outputs(0., -0.)['diverges'])
        self.assertTrue(compare_outputs(np.nan, np.nan)['diverges'])

    def test_string_labels(self):
        for value, label in [('False', 'EQUIV'), ('True', 'DIVERGE'), ('N/A', 'ERROR_OR_UNAVAILABLE')]:
            self.assertEqual(published_label({'judge__outputs_diverge': value}), label)

    def test_gather_generation_and_invalid_indices(self):
        specs = parse_input_spec(GATHER); names = ['%data', '%idx']
        a = generate_inputs(GATHER, GATHER, specs, names, names, 42)
        b = generate_inputs(GATHER, GATHER, specs, names, names, 42)
        self.assertEqual(digest(serialize_inputs(a)), digest(serialize_inputs(b)))
        self.assertTrue(np.all((a[1] >= 0) & (a[1] < 3)))
        check_inputs(GATHER, names, a, 'torch')
        for index in (-1, 3):
            with self.assertRaises(InputPolicyError) as e: check_inputs(GATHER, names, [a[0], np.array([index,0], dtype=np.int64)], 'torch')
            self.assertEqual(e.exception.status, 'INVALID_INPUT')

    def test_unsupported_operation_and_nonfinite_input(self):
        with self.assertRaises(InputPolicyError) as e: check_inputs(GATHER.replace('torch.aten.gather', 'torch.aten.scatter_add'), ['%data', '%idx'], [np.ones((2,3)), np.zeros(2, dtype=np.int64)], 'torch')
        self.assertEqual(e.exception.status, 'UNSUPPORTED_INPUT_PRECONDITIONS')
        with self.assertRaises(InputPolicyError): check_inputs(IDENTITY, ['%x'], [np.array([np.inf,1])], 'linalg')

    def test_input_suite_and_repeat_contract(self):
        with self.assertRaises(ValueError): validate_pair(IDENTITY, IDENTITY, 'linalg', repeats=1)
        with self.assertRaises(ValueError): validate_pair(IDENTITY, IDENTITY, 'linalg', seeds=[])

    def test_classification_with_controlled_runtime(self):
        compile_result = {'status': 'success', 'compiled_module': b'module'}
        def output(v): return {'status': 'success', 'output_typed': typed_output(np.float32(v))}
        for runs, verdict in [([output(1)]*4, 'NO_DIFFERENCE_OBSERVED'),
                              ([output(1)]*2 + [output(2)]*2, 'DIVERGE'),
                              ([output(1),output(2)]*2, 'NONDETERMINISTIC'),
                              ([output(np.nan)]*4, 'NONFINITE_OUTPUT'),
                              ([{'status': 'timeout'}]*4, 'RUNTIME_TIMEOUT')]:
            with self.subTest(verdict=verdict), patch('mlirbench.execution.pair_validation.compile_mlir', return_value=compile_result), patch('mlirbench.execution.pair_validation.run_compiled', side_effect=runs):
                self.assertEqual(validate_pair(IDENTITY, IDENTITY, 'linalg', seeds=[42])['verdict'], verdict)

    def test_malformed_recorded_inputs(self):
        specs = [{'shape': [1], 'dtype': 'int64'}]
        for item in ({}, {'dtype': 'bogus', 'shape': [1]}, {'dtype': 'int64', 'shape': [1]},
                     {'dtype': 'int64', 'shape': [1], 'data': [1.5]}):
            with self.subTest(item=item), self.assertRaises(InputPolicyError): deserialize_inputs([item], specs)

    def test_literal_division_log_and_sqrt_domains(self):
        source = GATHER.replace(' %dim =', ' %literal = torch.vtensor.literal(dense<9.0> : tensor<f32>) : !torch.vtensor<[],f32>\n %dim =')
        check_inputs(source, ['%data', '%idx'], [np.ones((2,3), dtype=np.float32), np.zeros(2, dtype=np.int64)], 'torch')
        for op, values in [('log', [0., 1.]), ('sqrt', [-1., 1.])]:
            source = f'module {{\n func.func @main(%x: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> {{\n %r = torch.aten.{op} %x : !torch.vtensor<[2],f32> -> !torch.vtensor<[2],f32>\n return %r : !torch.vtensor<[2],f32>\n }} }}'
            with self.assertRaises(InputPolicyError) as exc: check_inputs(source, ['%x'], [np.array(values, dtype=np.float32)], 'torch')
            self.assertEqual(exc.exception.status, 'INVALID_INPUT')


    def test_linalg_denominator_uses_declared_precision(self):
        source = """module {
 func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {
 %c = arith.constant dense<1.0> : tensor<f32>
 %e = tensor.empty() : tensor<2xf32>
 %r = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->()>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]} ins(%x, %c : tensor<2xf32>, tensor<f32>) outs(%e : tensor<2xf32>) {
 ^bb0(%v: f32, %den: f32, %out: f32):
 %d = arith.divf %v, %den : f32
 linalg.yield %d : f32
 } -> tensor<2xf32>
 return %r : tensor<2xf32>
 } }"""
        check_inputs(source, ['%x'], [np.ones(2, dtype=np.float32)], 'linalg')
        with self.assertRaises(InputPolicyError): check_inputs(source.replace('dense<1.0>', 'dense<1e-50>'), ['%x'], [np.ones(2, dtype=np.float32)], 'linalg')
        # Trusting the mutable output region argument would be unsound.
        unsafe = """module {
 func.func @main(%x: tensor<2xf32>) -> tensor<f32> {
 %c = arith.constant dense<1.0> : tensor<f32>
 %r = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->()>], iterator_types = ["reduction"]} ins(%x : tensor<2xf32>) outs(%c : tensor<f32>) {
 ^bb0(%v: f32, %out: f32):
 %d = arith.divf %v, %out : f32
 linalg.yield %d : f32
 } -> tensor<f32>
 return %r : tensor<f32>
 } }"""
        with self.assertRaises(InputPolicyError): check_inputs(unsafe, ['%x'], [np.ones(2, dtype=np.float32)], 'linalg')
