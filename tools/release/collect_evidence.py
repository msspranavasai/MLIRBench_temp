"""Require a complete native reviewer run before a platform may be published."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from mlirbench.ae import source_fingerprint
from mlirbench.evaluation.numeric_agreement import paper_agreement_passes


def validate_native(environment,tests,preservation,data,compiler,*,arch,identity,master_sha):
    cpu=environment.get('cpu_execution',{})
    if not environment.get('passed') or cpu.get('emulated') is not False:
        raise ValueError('Native compiler execution was not established')
    if cpu.get('container_arch')!=arch or cpu.get('docker_host_arch')!=arch or cpu.get('requested_platform')!='linux/'+arch:
        raise ValueError('Runner, container and requested platform differ')
    if (not tests.get('passed') or tests.get('tests_run',0)<1 or any(tests.get(k)!=0 for k in ('failures','errors','skipped'))
            or tests.get('source_fingerprint')!=identity):
        raise ValueError('Complete tests for this exact source are required')
    if not preservation.get('passed') or not preservation.get('checks') or any(c.get('passed') is not True for c in preservation['checks']):
        raise ValueError('Frozen records and recalculated numbers must match')
    if data.get('passed') is not True or data.get('master_sha256')!=master_sha:
        raise ValueError('Published data identity differs')
    if compiler.get('passed') is not True or not compiler.get('checks'):
        raise ValueError('Archived compiler arithmetic must pass')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--arch',choices=['amd64','arm64'],required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();folder=args.results
    read=lambda name:json.loads((folder/name).read_text())
    identity=source_fingerprint(ROOT)
    manifest=json.loads((ROOT/'data/bundle/manifest.json').read_text())
    values={name:read(name) for name in ('environment.json','tests.json','preservation.json','data_integrity.json','compiler_metrics.json')}
    validate_native(values['environment.json'],values['tests.json'],values['preservation.json'],values['data_integrity.json'],
        values['compiler_metrics.json'],arch=args.arch,identity=identity,
        master_sha=manifest['MLIRBench/master/master.jsonl']['sha256'])
    comparison=read('pipeline/metrics/paper_comparison.json')
    if not paper_agreement_passes(ROOT,comparison):raise ValueError('Paper comparison did not pass the existing policy')
    args.output.mkdir(parents=True,exist_ok=True)
    for name in values:shutil.copyfile(folder/name,args.output/name)
    shutil.copyfile(folder/'pipeline/metrics/paper_comparison.json',args.output/'paper_comparison.json')
    report={'arch':args.arch,'source_fingerprint':identity,'native':True,'tests_run':values['tests.json']['tests_run'],
            'preservation_checks':len(values['preservation.json']['checks']),'passed':True}
    (args.output/'native-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
