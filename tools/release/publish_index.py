"""Combine two tested native manifests under one immutable versioned image tag."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def validate_platforms(index,expected):
    manifests=index.get('manifests',[])
    actual={}
    for item in manifests:
        platform=item.get('platform',{});key=(platform.get('os'),platform.get('architecture'))
        if key in actual:raise ValueError('Duplicate image architecture')
        actual[key]=item.get('digest')
    wanted={('linux',arch):digest for arch,digest in expected.items()}
    if actual!=wanted:raise ValueError('Published image does not contain exactly the two tested native manifests')


def inspect(reference):
    p=subprocess.run(['docker','buildx','imagetools','inspect','--raw',reference],capture_output=True,text=True)
    if p.returncode:
        if re.search(r'not found|manifest unknown|404',p.stderr,re.I):return None
        raise RuntimeError('Registry inspection failed; refusing to replace a version tag')
    return json.loads(p.stdout)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--image',required=True);ap.add_argument('--version',required=True)
    ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();expected={};identity=None
    for arch in ('amd64','arm64'):
        base=args.evidence/arch
        verified=json.loads((base/'native-verification.json').read_text())
        digest=json.loads((base/'digest.json').read_text())
        if not verified.get('passed') or verified.get('arch')!=arch or digest.get('arch')!=arch:
            raise ValueError('Unverified or mismatched native image evidence')
        if identity is not None and identity!=verified['source_fingerprint']:raise ValueError('Native builds use different source')
        identity=verified['source_fingerprint'];expected[arch]=digest['digest']
        if not re.fullmatch(r'sha256:[a-f0-9]{64}',digest['digest']):raise ValueError('Invalid image digest')
        raw=inspect(args.image+'@'+digest['digest'])
        if raw is None or raw.get('config',{}).get('digest')!=digest['tested_config_digest']:
            raise ValueError('Pushed platform differs from the tested image configuration')
    reference=args.image+':'+args.version
    existing=inspect(reference)
    if existing is not None:
        validate_platforms(existing,expected)
    else:
        subprocess.run(['docker','buildx','imagetools','create','--tag',reference,
                        *[args.image+'@'+expected[a] for a in ('amd64','arm64')]],check=True)
    published=inspect(reference);validate_platforms(published,expected)
    digest=json.loads(subprocess.check_output(['docker','buildx','imagetools','inspect',reference,'--format','{{json .Manifest}}'],text=True))['digest']
    report={'image':reference,'index_digest':digest,'platforms':expected,'source_fingerprint':identity,
            'native_tests_required':True,'one_image_reference':True}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
