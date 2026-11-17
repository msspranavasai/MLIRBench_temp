import csv
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mlirbench.execution.input_policy import generate_input_case, InputPolicyError, check_inputs, torch_float_literal
from mlirbench.execution.pair_ground_truth import execute_pair_shared_inputs
from mlirbench.execution.execution import parse_input_spec
from mlirbench.execution.provenance import (source_relation, stored_input_check, verify_sources, sha, audit_case)
from tests.unit.test_pair_validation import IDENTITY, GATHER

LOG = '''module {
func.func @main(%x: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> {
 %r = torch.aten.log %x : !torch.vtensor<[2],f32> -> !torch.vtensor<[2],f32>
 return %r : !torch.vtensor<[2],f32>
} }'''


class Generation(unittest.TestCase):
    def test_hexadecimal_float_literals_are_parsed_as_ieee_values(self):
        self.assertEqual(torch_float_literal('0x3FF0000000000000'), 1.0)
        self.assertEqual(torch_float_literal('0xFFF0000000000000'), float('-inf'))
        with self.assertRaises(InputPolicyError) as exc:
            torch_float_literal('unrecognized-literal')
        self.assertEqual(exc.exception.status, 'UNSUPPORTED_INPUT_PRECONDITIONS')

    def test_valid_hex_constant_does_not_turn_into_an_input_failure(self):
        source = LOG.replace(' %r =', ' %sentinel = torch.constant.float 0xFFF0000000000000\n %r =')
        values, trace = generate_input_case(source, source, parse_input_spec(source), ['%x'], ['%x'], 42)
        self.assertTrue(np.all(values[0] > 0))

    def test_sign_and_zero_coverage_is_not_globally_positive(self):
        specs = parse_input_spec(IDENTITY)
        for seed, relation in [(1, lambda a: np.all(a > 0)),
                               (42, lambda a: np.all(a < 0)),
                               (123, lambda a: np.all(a == 0))]:
            values, trace = generate_input_case(IDENTITY, IDENTITY, specs, ['%x'], ['%x'], seed)
            self.assertTrue(relation(values[0]))
            self.assertIn('pattern', trace)

    def test_domain_rejection_produces_a_valid_replayable_case(self):
        specs = parse_input_spec(LOG)
        a, trace = generate_input_case(LOG, LOG, specs, ['%x'], ['%x'], 42)
        b, other = generate_input_case(LOG, LOG, specs, ['%x'], ['%x'], 42)
        self.assertTrue(np.all(a[0] > 0))
        self.assertGreater(len(trace['rejected_candidates']), 0)
        np.testing.assert_array_equal(a[0], b[0]); self.assertEqual(trace, other)

    def test_incompatible_domains_never_produce_an_unsafe_case(self):
        negative_log = LOG.replace(' %r =', ' %n = torch.aten.neg %x : !torch.vtensor<[2],f32> -> !torch.vtensor<[2],f32>\n %r =').replace('torch.aten.log %x', 'torch.aten.log %n')
        with self.assertRaises(InputPolicyError) as exc:
            generate_input_case(LOG, negative_log, parse_input_spec(LOG), ['%x'], ['%x'], 42)
        self.assertEqual(exc.exception.status, 'INPUT_GENERATION_FAILED')

    def test_unparsed_torch_operation_is_refused(self):
        source = LOG.replace('\n', ' ')
        with self.assertRaises(InputPolicyError) as exc:
            check_inputs(source, ['%x'], [np.ones(2, np.float32)], 'torch')
        self.assertEqual(exc.exception.status, 'UNSUPPORTED_INPUT_PRECONDITIONS')


class Adapter(unittest.TestCase):
    def test_partial_success_cannot_override_an_unverified_suite(self):
        report = {'verdict': 'UNVERIFIED', 'mode': 'canonical_validation',
                  'trials': [{'status': 'NO_DIFFERENCE_OBSERVED'}],
                  'compilation': [{'status': 'success'}, {'status': 'success'}]}
        with patch('mlirbench.execution.pair_ground_truth.validate_pair', return_value=report):
            a, b, metadata = execute_pair_shared_inputs(IDENTITY, IDENTITY, 'linalg')
        self.assertEqual(a['execution_status'], 'UNVERIFIED')
        self.assertEqual(b['execution_status'], 'UNVERIFIED')
        self.assertIsNone(a['output'])
        self.assertEqual(metadata['validation_report']['verdict'], 'UNVERIFIED')


class StructuralAudit(unittest.TestCase):
    def test_ssa_names_and_comments_are_ignored_in_flat_functions(self):
        other = '// comment\n' + IDENTITY.replace('%x', '%renamed')
        self.assertEqual(source_relation(IDENTITY, other), 'IDENTICAL_UP_TO_SSA_NAMES')

    def test_literals_strings_and_operand_relationships_remain_significant(self):
        a = 'module { func.func @main(%x: i32, %y: i32) -> i32 { %r = arith.subi %x, %y : i32 return %r : i32 } }'
        b = a.replace('arith.subi %x, %y', 'arith.subi %y, %x')
        self.assertEqual(source_relation(a,b), 'DIFFERENT_TOKEN_STRUCTURE')
        a = IDENTITY.replace('return %x', '%s = torch.constant.str "%x"\n return %x')
        self.assertEqual(source_relation(a,a.replace('%x','%z')), 'DIFFERENT_TOKEN_STRUCTURE')
        a = IDENTITY.replace('return %x', '%c = arith.constant 1 : i32\n return %x')
        self.assertEqual(source_relation(a,a.replace('constant 1','constant 2')), 'DIFFERENT_TOKEN_STRUCTURE')

    def test_nested_scopes_are_not_claimed_equivalent(self):
        a = IDENTITY.replace('return %x', 'scf.execute_region { scf.yield }\n return %x')
        self.assertEqual(source_relation(a,a), 'UNSUPPORTED_STRUCTURAL_ANALYSIS')

    def test_invalid_gather_metadata_is_never_executed(self):
        record = {'input_mlir':GATHER, 'mutated_mlir':GATHER, 'dialect':'torch'}
        evidence = {'historical_fields': {'judge__embedded_inputs_json': json.dumps({
            '%data': [[1,2,3],[4,5,6]], '%idx': [-1,3]})}}
        report = stored_input_check(record,evidence)
        self.assertEqual(report['status'], 'INVALID_INPUT')

    def test_structural_conflict_is_an_annotation_not_relabeling(self):
        record = {'program_id':'p', 'judge__outputs_diverge':'True',
                  'input_mlir':IDENTITY,'mutated_mlir':IDENTITY.replace('%x','%renamed'),'dialect':'linalg'}
        with patch('mlirbench.execution.provenance.validate_pair', return_value={'verdict':'NO_DIFFERENCE_OBSERVED'}):
            report = audit_case((record,{},[42],2,30))
        self.assertEqual(report['label_assessment'], 'STRUCTURAL_LABEL_CONFLICT')
        self.assertEqual(record['judge__outputs_diverge'], 'True')
        self.assertEqual(report['published_label'], 'DIVERGE')


class HistoricalSources(unittest.TestCase):
    def fixture(self, root):
        folder = root/'data/historical';folder.mkdir(parents=True)
        master_path = root/'data/processed/MLIRBench/master/master.jsonl';master_path.parent.mkdir(parents=True)
        record = {'program_id':'p','input_mlir':IDENTITY,'mutated_mlir':IDENTITY}
        master_path.write_text(json.dumps(record)+'\n')
        row = {'program_id':'p','judge__outputs_diverge':'False','judge__embedded_inputs_json':'{"%x":[1,2]}'}
        stream = io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
        raw=stream.getvalue().encode();packed=gzip.compress(raw,mtime=0)
        evidence={'program_id':'p','historical_fields':{k:v for k,v in row.items() if k!='program_id'},
                  'source_sha256':[sha(IDENTITY.encode())]*2,
                  'provenance':{'source_sha256':sha(raw),'row_sha256':sha(json.dumps(row,sort_keys=True).encode())}}
        sidecar=(json.dumps(evidence)+'\n').encode()
        (folder/'raw.csv.gz').write_bytes(packed);(folder/'sidecar.jsonl').write_bytes(sidecar)
        manifest={'archive_filename':'raw.csv.gz','archive_sha256':sha(packed),'source_sha256':sha(raw),
                  'source_rows':1,'sidecar_filename':'sidecar.jsonl','sidecar_sha256':sha(sidecar),
                  'published_rows':1,'published_master_sha256':sha(master_path.read_bytes()),'runtime_input_provenance':'unrecorded'}
        (folder/'source_manifest.json').write_text(json.dumps(manifest))
        return {'p':record},{'p':evidence}

    def test_original_rows_and_programs_are_verified_without_runtime_claims(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);master,historical=self.fixture(root)
            report=verify_sources(root,master,historical)
            self.assertTrue(report['passed']);self.assertFalse(report['runtime_inputs_proven'])

    def test_changed_extracted_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);master,historical=self.fixture(root)
            historical['p']['historical_fields']['judge__outputs_diverge']='True'
            with self.assertRaises(ValueError):verify_sources(root,master,historical)
