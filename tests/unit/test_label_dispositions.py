"""The disposition overlay must preserve data and distinguish comparison contracts."""
import base64
import json
from pathlib import Path
import tempfile
import unittest

from mlirbench.label_dispositions import digest, iterator_control, load_dossier, numeric_equal, review_labels, sha


class LabelDispositions(unittest.TestCase):
    def fixture(self, root):
        source=root/'src/mlirbench/execution';source.mkdir(parents=True)
        code={}
        for name in ('execution.py','input_policy.py','pair_validation.py'):
            (source/name).write_text(name);code[name]=sha(name.encode())
        master=root/'data/processed/MLIRBench/master/master.jsonl';master.parent.mkdir(parents=True)
        row={'program_id':'case','input_mlir':'original','mutated_mlir':'mutated','judge__outputs_diverge':'False'}
        master.write_text(json.dumps(row)+'\n')
        inputs=[{'position':0,'dtype':'<f4','shape':[1],'data':[0.0],'bytes_b64':'AAAAAA=='}]
        case={'program_id':'case','published_label':'EQUIV','original_assessment':'LABEL_COUNTEREXAMPLE',
              'source_sha256':[sha(b'original'),sha(b'mutated')],'published_record_modified':False,
              'historical_runtime_inputs_proven':False,'witness_inputs':inputs,'witness_input_sha256':digest(inputs),
              'documented_disposition':'DOCUMENTED','diagnosis':'Input scopes differ.','action':'Preserve the record.'}
        ledger={'schema':'mlirbench-label-dispositions-v1','scope':{'cases':1,'published_master_sha256':sha(master.read_bytes())},
                'disposition_counts':{'DOCUMENTED':1},'scientific_limits':['Historical inputs are unproven.'],'cases':[case]}
        dossier=root/'docs/dispositions.json';dossier.parent.mkdir();dossier.write_text(json.dumps(ledger))
        manifest={'schema':'mlirbench-label-review-manifest-v1','dossier':'docs/dispositions.json',
                  'dossier_sha256':sha(dossier.read_bytes()),'published_master_sha256':sha(master.read_bytes()),
                  'case_ids':['case'],'reviewed_execution_code_sha256':code}
        folder=root/'experiments';folder.mkdir();(folder/'label_review_manifest.json').write_text(json.dumps(manifest))
        return ledger,dossier,manifest,master

    def repin(self, root, ledger, dossier, manifest):
        dossier.write_text(json.dumps(ledger));manifest['dossier_sha256']=sha(dossier.read_bytes())
        (root/'experiments/label_review_manifest.json').write_text(json.dumps(manifest))

    def test_verified_overlay_never_changes_dataset_or_claims_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ledger,dossier,manifest,master=self.fixture(root);before=master.read_bytes()
            report=review_labels(root,root/'review')
            self.assertEqual(report['status'],'DOCUMENTED_WITH_LIMITATIONS')
            self.assertFalse(report['historical_runtime_inputs_proven'])
            self.assertFalse(report['published_dataset_modified'])
            self.assertFalse(report['paid_api_calls_requested'])
            self.assertEqual(master.read_bytes(),before)
            with self.assertRaises(FileExistsError):review_labels(root,root/'review')
            with self.assertRaises(ValueError):review_labels(root,root/'data/new-diagnostics')

    def test_changed_dossier_or_master_is_rejected(self):
        for changed in ('dossier','master'):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);ledger,dossier,manifest,master=self.fixture(root)
                (dossier if changed=='dossier' else master).write_text('changed')
                with self.assertRaises(ValueError):load_dossier(root)

    def test_repinning_cannot_change_published_labels_or_programs(self):
        for field,value in [('published_label','DIVERGE'),('source_sha256',['x','y']),
                            ('published_record_modified',True),('historical_runtime_inputs_proven',True)]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);ledger,dossier,manifest,master=self.fixture(root)
                ledger['cases'][0][field]=value;self.repin(root,ledger,dossier,manifest)
                with self.assertRaises(ValueError):load_dossier(root)

    def test_missing_duplicate_cases_and_changed_inputs_are_rejected(self):
        for kind in ('missing','duplicate','input'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);ledger,dossier,manifest,master=self.fixture(root)
                if kind=='missing':ledger['cases']=[]
                elif kind=='duplicate':ledger['cases']*=2
                else:ledger['cases'][0]['witness_inputs'][0]['data']=[1.0]
                self.repin(root,ledger,dossier,manifest)
                with self.assertRaises(ValueError):load_dossier(root)

    def test_changed_execution_code_is_not_attributed_to_recorded_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            (root/'src/mlirbench/execution/input_policy.py').write_text('new policy')
            self.assertFalse(load_dossier(root)[2])
            report=review_labels(root,root/'read-only-review')
            self.assertFalse(report['execution_code_matches_recorded_review'])
            with self.assertRaises(ValueError):review_labels(root,root/'fresh',fresh=True,all_cases=True)

    def test_control_modifies_only_a_separate_string_and_refuses_ambiguity(self):
        old='linalg.generic {indexing_maps = [#map, #map3], iterator_types = ["parallel", "reduction"]}'
        new=iterator_control(old)
        self.assertIn('["reduction", "parallel"]',new)
        self.assertIn('["parallel", "reduction"]',old)
        for text in ('unrelated',old+old):
            with self.assertRaises(ValueError):iterator_control(text)

    def test_value_and_typed_contracts_are_distinct_without_epsilon(self):
        import numpy as np
        from mlirbench.execution.pair_validation import compare_typed
        def typed(value,dtype):
            a=np.asarray(value,dtype=dtype)
            return {'kind':'array','dtype':a.dtype.str,'shape':list(a.shape),'bytes_b64':base64.b64encode(a.tobytes()).decode()}
        a,b=typed(0,'float32'),typed(0,'float64')
        self.assertTrue(numeric_equal(a,b))
        self.assertEqual(compare_typed(a,b),'DIFFERENT')
        self.assertFalse(numeric_equal(typed(1/3,'float32'),typed(1/3,'float64')))
        self.assertFalse(numeric_equal(typed([0],'float32'),typed([[0]],'float32')))

    def test_fresh_options_cannot_silently_change_selection_or_input_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            for options in ({'ids':['unknown']},{'repeats':1},{'numeric_witness':True},
                            {'fresh':True,'ids':['case'],'numeric_witness':True},
                            {'fresh':True,'ids':['case'],'control':True}):
                with self.subTest(options=options),self.assertRaises(ValueError):review_labels(root,root/'out',**options)
