"""Missing, inconsistent and edited release evidence must never pass."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from mlirbench.release_readiness import (Archive, assess_release, compiler_evidence, file_hash,
    index_evidence, normalize_downstream, preserved, sha, validate_ci_preservation,
    validate_command, validate_cpu, validate_tests)


class MemoryArchive:
    def __init__(self, files):
        self.files = {n: v.encode() if isinstance(v, str) else v for n, v in files.items()}
        self.names = set(files)

    def read(self, name):
        return self.files[name]

    def json(self, name):
        return json.loads(self.read(name))


class ReleaseEvidence(unittest.TestCase):
    def test_index_pins_archive_and_ci_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); archive = base/'results.zip'; archive.write_bytes(b'archived evidence')
            ci = base/'ci'; ci.mkdir(); (ci/'report.json').write_text('{}')
            index = base/'index.json'
            record = index_evidence(archive, index, ci=ci)
            self.assertEqual(record['archive']['sha256'], file_hash(archive))
            self.assertEqual(record['ci']['files']['report.json'], file_hash(ci/'report.json'))
            with self.assertRaises(ValueError):index_evidence(archive, index)
            archive.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):assess_release(base, index)

    def test_duplicate_zip_members_are_rejected_without_extracting(self):
        import warnings
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'duplicate.zip'
            with warnings.catch_warnings(), zipfile.ZipFile(p,'w') as z:
                warnings.simplefilter('ignore');z.writestr('same','one');z.writestr('same','two')
            with self.assertRaisesRegex(ValueError,'Duplicate ZIP'):Archive(p)

    def test_failed_skipped_empty_or_wrong_source_tests_do_not_pass(self):
        good={'passed':True,'tests_run':195,'failures':0,'errors':0,'skipped':0,'source_fingerprint':'source'}
        validate_tests(good,'source')
        for key,value in [('passed',False),('tests_run',0),('failures',1),('errors',1),('skipped',1),('source_fingerprint','other'),('failures',False)]:
            with self.subTest(key=key),self.assertRaises(ValueError):validate_tests({**good,key:value},'source')

    def test_native_coverage_rejects_emulation_and_wrong_cpu(self):
        good={'passed':True,'cpu_execution':{'emulated':False,'container_arch':'arm64','docker_host_arch':'arm64','requested_platform':'linux/arm64'}}
        validate_cpu(good,'arm64')
        for key,value in [('emulated',True),('container_arch','amd64'),('docker_host_arch','amd64'),('requested_platform','linux/amd64')]:
            changed=copy.deepcopy(good);changed['cpu_execution'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_cpu(changed,'arm64')

    def test_timeout_or_fabricated_compiler_success_is_rejected(self):
        validate_command({'returncode':0,'success':True,'timeout':False})
        self.assertFalse(validate_command({'returncode':1,'success':False,'timeout':False}))
        for record in [{'returncode':None,'success':False,'timeout':True},
                       {'returncode':1,'success':True,'timeout':False},
                       {'returncode':0,'success':True,'timeout':True}]:
            with self.subTest(record=record),self.assertRaises(ValueError):validate_command(record)

    def test_ci_preservation_requires_exact_baseline_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'experiments').mkdir()
            baseline=root/'experiments/release_invariants.json'
            baseline.write_text(json.dumps({'frozen_inputs':{'data':'a'},'recomputed_outputs':{'result':{'sha256':'b','mode':'bytes'}}}))
            good={'passed':True,'baseline_sha256':file_hash(baseline),'checks':[
                {'kind':'frozen_input','path':'data','expected_sha256':'a','computed_sha256':'a','passed':True},
                {'kind':'recomputed_output','path':'result','expected_sha256':'b','computed_sha256':'b','passed':True}]}
            validate_ci_preservation(root,good)
            for checks in [[],good['checks'][:1],good['checks']+[good['checks'][0]],
                           [good['checks'][0],{**good['checks'][1],'computed_sha256':'changed'}]]:
                with self.subTest(checks=checks),self.assertRaises(ValueError):validate_ci_preservation(root,{**good,'checks':checks})

    def test_preservation_rehashes_bytes_instead_of_trusting_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'experiments').mkdir()
            baseline=root/'experiments/release_invariants.json'
            baseline.write_text(json.dumps({'frozen_inputs':{'data':sha(b'original')},'recomputed_outputs':{}}))
            report={'passed':True,'baseline_sha256':file_hash(baseline),'checks':[
                {'kind':'frozen_input','path':'data','passed':True,'expected_sha256':sha(b'original'),'computed_sha256':sha(b'original')}]}
            archive=MemoryArchive({'source/data':'changed','run/preservation.json':json.dumps(report)})
            with self.assertRaisesRegex(ValueError,'differs: data'):preserved(root,archive,'source/','run/')

    def test_downstream_normalizes_only_machine_path_not_numbers(self):
        a={'accuracy':0.5,'input_provenance':[{'path':'/app/data/downstream/records/test.csv','sha256':'x'}]}
        b=copy.deepcopy(a);b['input_provenance'][0]['path']='/other/data/downstream/records/test.csv'
        self.assertEqual(normalize_downstream(a),normalize_downstream(b))
        b['accuracy']=0.500000001
        self.assertNotEqual(normalize_downstream(a),normalize_downstream(b))

    def test_scientific_open_questions_stay_open_after_functional_success(self):
        import contextlib
        import io
        from mlirbench.cli import main
        report={'status':'NOT_SUBMISSION_READY','sections':{'software':'PASS','scientific':'REVIEW_REQUIRED'},
                'gates':[{'section':'software','status':'PASS'},
                         {'section':'scientific','status':'REVIEW_REQUIRED'},
                         {'section':'build_provenance','status':'OPEN'}]}
        with patch('mlirbench.release_readiness.prepare_release_data'), \
             patch('mlirbench.release_readiness.assess_release',return_value=report), \
             patch('mlirbench.release_readiness.write_report') as write, \
             contextlib.redirect_stdout(io.StringIO()) as text:
            result=main(['release-readiness','--evidence','index.json','--output','assessment'])
            self.assertEqual(result,0)
            self.assertEqual(json.loads(text.getvalue())['status'],'NOT_SUBMISSION_READY')
            self.assertIs(write.call_args.args[0],report)
            report['gates'][0]['status']='REVIEW_REQUIRED'
            self.assertEqual(main(['release-readiness','--evidence','index.json','--output','other']),1)

    def compiler_fixture(self, root):
        (root/'src/mlirbench').mkdir(parents=True)
        driver=root/'src/mlirbench/compiler_experiments.py';driver.write_text('fixture driver')
        scan=root/'scan.csv';scan.write_text('fixture scan')
        row={'dt1_id':'p','mlir_hash':sha(b'module {}'),'baseline_success':'True','fusion_success':'True',
             'fusion_changed_ir':'True','linalg_ops_before':'1','linalg_ops_after':'2'}
        command=lambda stdout:{'returncode':0,'success':True,'timeout':False,'stdout':stdout}
        actual={'dt1_id':'p','mlir_hash':row['mlir_hash'],'dialect':'linalg',
                'baseline':command('linalg.add'),'passes':{'fusion':{**command('linalg.add linalg.mul'),
                'changed_ir':True,'ops_before':1,'ops_after':2,'ops_removed':-1}}}
        summary={'mode':'fresh_compiler_execution','dialect':'linalg','full_corpus':True,'programs':1,
                 'expected_programs':1,'riscv_requested':False,'driver_sha256':file_hash(driver),
                 'source_scan_sha256':file_hash(scan),'mismatches':[],'transformation_agreement':True,
                 'toolchain':{'iree-compile':{'version':'3.12.0rc20260916'}}}
        files={'run/summary.json':json.dumps(summary),'run/records/p.json':json.dumps(actual),
               'run/programs/p/source.mlir':'module {}\n'}
        return row,scan,actual,summary,files

    def test_full_compiler_checks_records_and_ir_instead_of_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);row,scan,actual,summary,files=self.compiler_fixture(root)
            with patch('mlirbench.release_readiness.scan_rows',return_value=([row],scan)):
                self.assertEqual(compiler_evidence(root,MemoryArchive(files),'run/','linalg')['programs'],1)
                for failure in ('missing_record','wrong_source','wrong_count','wrong_driver','subset','duplicate_record','changed_ir'):
                    changed=dict(files);record=copy.deepcopy(actual);header=copy.deepcopy(summary)
                    if failure=='missing_record':del changed['run/records/p.json']
                    elif failure=='wrong_source':changed['run/programs/p/source.mlir']='modified module'
                    elif failure=='wrong_count':record['passes']['fusion']['ops_after']=99
                    elif failure=='wrong_driver':header['driver_sha256']='other'
                    elif failure=='subset':header['full_corpus']=False
                    elif failure=='duplicate_record':changed['run/records/nested/p.json']=json.dumps(record)
                    elif failure=='changed_ir':record['passes']['fusion']['stdout']='linalg.add'
                    changed['run/summary.json']=json.dumps(header)
                    if failure!='missing_record':changed['run/records/p.json']=json.dumps(record)
                    with self.subTest(failure=failure),self.assertRaises(ValueError):compiler_evidence(root,MemoryArchive(changed),'run/','linalg')

    def test_riscv_recounts_assembly_and_rejects_changed_or_missing_files(self):
        import csv
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);row,scan,actual,summary,files=self.compiler_fixture(root)
            historical=root/'data/downstream/records/DT1_ELEMENTWISE/outputs/riscv_backend_extension/riscv_backend_results.csv'
            historical.parent.mkdir(parents=True)
            with historical.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['dt1_id','both_compile','baseline_instruction_count','fused_instruction_count','instruction_outcome'])
                w.writeheader();w.writerow({'dt1_id':'p','both_compile':'True','baseline_instruction_count':1,'fused_instruction_count':1,'instruction_outcome':'TIED'})
            summary.update(riscv_requested=True,riscv_mismatches=[])
            actual['riscv']={}
            for label,text in [('baseline','linalg.add'),('fused','linalg.add linalg.mul')]:
                actual['riscv'][label]={'returncode':0,'success':True,'timeout':False,
                    'command':['iree-compile','--iree-llvmcpu-target-triple=riscv64'],
                    'assembly_files':1,'instruction_count':1,'instruction_stream_sha256':sha(b'add a0, a0, a1')}
                files['run/programs/p/'+label+'/input.mlir']=text
                files['run/programs/p/'+label+'/code.s']='.text\nadd a0, a0, a1\n'
            files['run/summary.json']=json.dumps(summary);files['run/records/p.json']=json.dumps(actual)
            with patch('mlirbench.release_readiness.scan_rows',return_value=([row],scan)):
                report=compiler_evidence(root,MemoryArchive(files),'run/','linalg',True)
                self.assertEqual(report['both_compile'],1)
                self.assertEqual(report['instruction_outcomes'],{'TIED':1})
                for missing in (False,True):
                    changed=dict(files);path='run/programs/p/fused/code.s'
                    if missing:del changed[path]
                    else:changed[path]+='addi a0, a0, 1\n'
                    with self.subTest(missing=missing),self.assertRaises(ValueError):compiler_evidence(root,MemoryArchive(changed),'run/','linalg',True)
