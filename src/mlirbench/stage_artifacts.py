"""Lossless stage records with content identities and explicit replay/fresh provenance."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

STAGES = ('mutation','execution','reasoning','judge')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pair_digest(pair):
    return digest({k:pair[k] for k in ('input_mlir','mutated_mlir','dialect')})


def record(stage, pair, data, *, mode, parents=(), source=None, model=None):
    value = {'schema':'mlirbench-stage-v1', 'stage':stage, 'mode':mode,
             'program_id':pair['program_id'], 'pair_sha256':pair_digest(pair),
             'model':model, 'parents':list(parents), 'source':source or {}, 'data':data}
    value['record_sha256'] = digest(value)
    return value


def atomic_write(path, text):
    """Commit a complete file by rename, with file and directory fsync."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            os.fchmod(stream.fileno(), 0o644)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, allow_nan=False)+'\n')


def save_records(path, records):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    atomic_write(path, ''.join(json.dumps(value,allow_nan=False)+'\n' for value in records))
    return {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'records':len(records)}


def validate_record(value, stage, by_hash):
    if value['schema'] != 'mlirbench-stage-v1' or value['stage'] != stage:
        raise ValueError('Unexpected stage schema')
    copy = dict(value); actual = copy.pop('record_sha256')
    if digest(copy) != actual: raise ValueError('Stage record digest differs')
    if stage == 'mutation':
        if pair_digest(value['data']) != value['pair_sha256']: raise ValueError('Mutation source differs')
    else:
        required = {'execution':{'mutation'},'reasoning':{'execution'},'judge':{'execution','reasoning'}}[stage]
        parents = [by_hash.get(h) for h in value['parents']]
        if any(p is None for p in parents) or {p['stage'] for p in parents} != required:
            raise ValueError('Missing or incompatible stage parents')
        if any(p['pair_sha256'] != value['pair_sha256'] or p['program_id'] != value['program_id'] for p in parents):
            raise ValueError('Program changed across stage boundary')
        if stage == 'judge' and any(p['model'] != value['model'] for p in parents if p['stage']=='reasoning'):
            raise ValueError('Judge response/model differs from reasoning parent')
    by_hash[actual] = value


def load_run(folder):
    """Check all saved bytes and parent identities before permitting a resume."""
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    result, by_hash = {}, {}
    for stage in STAGES:
        if stage not in manifest['artifacts']: continue
        path = folder/(stage+'.jsonl')
        spec = manifest['artifacts'][stage]
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError(f'Modified stage artifact: {stage}')
        with path.open() as stream:
            values = [json.loads(line) for line in stream if line.strip()]
        if len(values) != spec['records']: raise ValueError('Stage record count differs')
        seen = set()
        for value in values:
            key = value['program_id'],value['model']
            if key in seen: raise ValueError('Duplicate stage record')
            seen.add(key)
            validate_record(value, stage, by_hash)
        result[stage] = values
    if 'case_plan' in manifest:
        from .case_journal import load_journal
        load_journal(folder, manifest, by_hash=by_hash)
    return result, manifest
