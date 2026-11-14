"""Package a verified read-only HPC import as a deterministic distribution."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    root = Path.cwd()
    records = root / 'data/downstream/records'
    verified = json.loads((root / 'data/provenance/hpc_verified_source_hashes.json').read_text())
    manifest = verified['files']
    for name, spec in manifest.items():
        data = (records / name).read_bytes()
        if len(data) != spec['bytes'] or hashlib.sha256(data).hexdigest() != spec['sha256']:
            raise ValueError(f'HPC import differs from remote original: {name}')
    target = root / 'data/downstream/snapshot.tar.gz'
    with target.open('xb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as archive:
                for name, spec in sorted(manifest.items()):
                    data = (records / name).read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mode = 0o644
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(data))
    result = {'schema': 'mlirbench-downstream-snapshot-v1', 'archive_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
              'scope': 'archived source, final-figure predictions and original splits; model weights excluded',
              'files': manifest}
    (root / 'data/downstream/manifest.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'files': len(manifest), 'archive_bytes': target.stat().st_size,
                      'uncompressed_bytes': sum(r['bytes'] for r in manifest.values())}, indent=2))


if __name__ == '__main__':
    main()
