"""Local-first published data resolution; downloads require affirmative consent."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import urllib.request

HF_REPOSITORY = 'mlirbench/MLIRBench'
HF_REVISION = '1af6c3c7a29946a7cd0e2aafd9386686d7fec9d1'


def manifest_entries(root, names=None):
    manifest = json.loads((Path(root)/'data/bundle/manifest.json').read_text())
    names = list(manifest) if names is None else list(names)
    # Validate the whole requested set before any write or download.
    for name in names:
        rel = PurePosixPath(name)
        if (name not in manifest or rel.is_absolute() or '..' in rel.parts
                or '\\' in name or str(rel) != name):
            raise ValueError(f'Unknown or unsafe dataset file: {name}')
    return {name:manifest[name] for name in names}


def write_missing(path, data):
    """Publish verified bytes atomically; never replace an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            os.fchmod(stream.fileno(), 0o644)
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic exclusive creation, including races.
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def packed_bytes(raw, spec):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename='', fileobj=buffer, mode='wb', mtime=0, compresslevel=9) as stream:
        stream.write(raw)
    packed = buffer.getvalue()
    if hashlib.sha256(packed).hexdigest() != spec['gzip_sha256']:
        raise ValueError('Verified raw data retained, but this Python/zlib cannot reproduce the frozen gzip bytes. No bundle was written. Use the artifact Python/Docker environment to rebuild; the frozen manifest must not be changed.')
    return packed


def ensure_data(root, names=None, *, consent=False, prompt=None, fetch=None):
    root = Path(root)
    entries = manifest_entries(root, names)
    missing, resolved, repaired = [], {}, []
    for name, spec in entries.items():
        dest = root/'data/processed'/name
        packed = root/'data/bundle'/(name+'.gz')
        def valid(raw):
            return len(raw) == spec['bytes'] and hashlib.sha256(raw).hexdigest() == spec['sha256']
        raw = None
        if dest.exists():
            raw = dest.read_bytes()
            if not valid(raw):
                raise ValueError(f'Modified local published data; refusing to overwrite: {name}')
        compressed = None
        if packed.exists():
            compressed = packed.read_bytes()
            if hashlib.sha256(compressed).hexdigest() != spec['gzip_sha256']:
                raise ValueError(f'Modified data bundle: {name}')
        if raw is not None:
            if compressed is None:
                write_missing(packed, packed_bytes(raw, spec))
                repaired.append(name)
            resolved[name] = 'verified_local'
        elif compressed is not None:
            raw = gzip.decompress(compressed)
            if not valid(raw): raise ValueError(f'Published hash mismatch: {name}')
            write_missing(dest, raw)
            resolved[name] = 'verified_bundled_archive'
        else:
            missing.append(name)
    if missing:
        if not consent:
            if prompt is None and not sys.stdin.isatty():
                raise FileNotFoundError('Missing published data. No download attempted. Run ./ae.sh fetch-data and answer yes.')
            answer = (prompt or input)(f'Download {len(missing)} missing files from Hugging Face {HF_REPOSITORY}@{HF_REVISION}? [yes/NO] ')
            consent = answer.strip().lower() in ('yes', 'y')
        if not consent:
            raise PermissionError('Download declined; no network request made.')
        for name in missing:
            spec = entries[name]
            remote_name = name.removeprefix('MLIRBench/')
            url = f'https://huggingface.co/datasets/{HF_REPOSITORY}/resolve/{HF_REVISION}/{remote_name}'
            if fetch is None:
                with urllib.request.urlopen(url, timeout=120) as response: raw = response.read()
            else: raw = fetch(url)
            if len(raw) != spec['bytes'] or hashlib.sha256(raw).hexdigest() != spec['sha256']:
                raise ValueError(f'Download differs from frozen published file: {name}')
            # Keep verified raw bytes even if this zlib cannot recreate gzip.
            write_missing(root/'data/processed'/name, raw)
            write_missing(root/'data/bundle'/(name+'.gz'), packed_bytes(raw, spec))
            resolved[name] = 'consented_pinned_huggingface_download'
    return {'repository':HF_REPOSITORY, 'revision':HF_REVISION,
            'files':resolved, 'repaired_bundles':repaired}


def prepare_distribution(root, *, consent=False, prompt=None, fetch=None):
    """Check every Docker build input; recover only absent gzip bundles.

    Intact bundles need only compressed hash verification on the host. Docker
    still verifies their raw hashes while unpacking. This avoids unpacking the
    full published distribution on every host-side launch.
    """
    root = Path(root)
    entries = manifest_entries(root)
    missing, resolved = [], {}
    for name, spec in entries.items():
        path = root/'data/bundle'/(name+'.gz')
        if not path.exists():
            missing.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != spec['gzip_sha256']:
            raise ValueError(f'Modified data bundle: {name}')
        else:
            resolved[name] = 'verified_build_bundle'
    repaired = []
    if missing:
        recovered = ensure_data(root, missing, consent=consent, prompt=prompt, fetch=fetch)
        resolved.update(recovered['files']); repaired = recovered['repaired_bundles']
        for name in missing:
            path = root/'data/bundle'/(name+'.gz')
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entries[name]['gzip_sha256']:
                raise ValueError(f'Docker build input was not restored: {name}')
    return {'repository':HF_REPOSITORY, 'revision':HF_REVISION, 'files':resolved,
            'verified_build_bundles':len(entries), 'repaired_bundles':repaired}
