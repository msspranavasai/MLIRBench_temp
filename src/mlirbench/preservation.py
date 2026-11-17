"""Detect drift in frozen scientific inputs and independently recomputed results."""
import hashlib
import json
from pathlib import Path


def checksum(path, mode):
    if mode == 'bytes': raw=path.read_bytes()
    elif mode == 'canonical_json':
        raw=json.dumps(json.loads(path.read_text()),sort_keys=True,separators=(',',':'),
                       ensure_ascii=False,allow_nan=False).encode()
    else: raise ValueError('Unknown preservation checksum mode: '+mode)
    return hashlib.sha256(raw).hexdigest()


def verify_preservation(root, results):
    root,results=Path(root),Path(results)
    path=root/'experiments/release_invariants.json'
    baseline=json.loads(path.read_text())
    if baseline.get('schema') != 'mlirbench-release-invariants-v1':
        raise ValueError('Unknown preservation baseline')
    checks=[]
    for kind,folder,entries in (('frozen_input',root,baseline['frozen_inputs']),
                                ('recomputed_output',results,baseline['recomputed_outputs'])):
        for name,spec in sorted(entries.items()):
            expected,mode=(spec,'bytes') if isinstance(spec,str) else (spec['sha256'],spec['mode'])
            candidate=folder/name
            if Path(name).is_absolute() or not candidate.resolve().is_relative_to(folder.resolve()):
                raise ValueError('Unsafe preservation baseline path')
            actual,error=None,None
            try: actual=checksum(candidate,mode)
            except (OSError,ValueError) as exc: error=type(exc).__name__
            checks.append({'kind':kind,'path':name,'passed':actual==expected,
                           'expected_sha256':expected,'computed_sha256':actual,'error':error})
    return {'passed':bool(checks) and all(c['passed'] for c in checks),
            'baseline_sha256':checksum(path,'bytes'),'checks':checks,
            'note':'Checks recalculated archived results and frozen inputs; does not certify label validity or force fresh experiments to match the paper.'}
