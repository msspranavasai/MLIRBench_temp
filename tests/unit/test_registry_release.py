"""Registry publication requires two tested native configurations and fixed data."""
import copy
import unittest
from tools.release.collect_evidence import validate_native
from tools.release.publish_index import validate_platforms


class RegistryEvidence(unittest.TestCase):
    def reports(self):
        return [
            {'passed':True,'cpu_execution':{'emulated':False,'container_arch':'amd64','docker_host_arch':'amd64','requested_platform':'linux/amd64'}},
            {'passed':True,'tests_run':190,'failures':0,'errors':0,'skipped':0,'source_fingerprint':'source'},
            {'passed':True,'checks':[{'passed':True}]},
            {'passed':True,'master_sha256':'master'},
            {'passed':True,'checks':[{'passed':True}]},
        ]

    def validate(self,reports):
        validate_native(*reports,arch='amd64',identity='source',master_sha='master')

    def test_complete_native_evidence_is_accepted(self):self.validate(self.reports())

    def test_emulation_or_mismatched_architecture_is_rejected(self):
        for field,value in [('emulated',True),('container_arch','arm64'),('docker_host_arch','arm64'),('requested_platform','linux/arm64')]:
            with self.subTest(field=field):
                records=self.reports();records[0]['cpu_execution'][field]=value
                with self.assertRaises(ValueError):self.validate(records)

    def test_partial_tests_changed_source_or_changed_data_are_rejected(self):
        for index,field,value in [(1,'tests_run',0),(1,'skipped',1),(1,'failures',1),(1,'source_fingerprint','stale'),(3,'master_sha256','changed')]:
            with self.subTest(field=field):
                records=self.reports();records[index][field]=value
                with self.assertRaises(ValueError):self.validate(records)

    def test_failed_preservation_cannot_be_published(self):
        records=self.reports();records[2]['checks'][0]['passed']=False
        with self.assertRaises(ValueError):self.validate(records)

    def test_index_requires_exactly_two_verified_platforms(self):
        expected={'amd64':'a','arm64':'b'}
        index={'manifests':[{'platform':{'os':'linux','architecture':a},'digest':d} for a,d in expected.items()]}
        validate_platforms(index,expected)
        for manifests in [index['manifests'][:1],index['manifests']+[index['manifests'][0]],
                          [index['manifests'][0],{'platform':{'os':'linux','architecture':'arm64'},'digest':'wrong'}]]:
            with self.subTest(manifests=manifests),self.assertRaises(ValueError):validate_platforms({'manifests':manifests},expected)
