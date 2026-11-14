"""Verify the downstream archive and restore original bytes without overwriting."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def unpack(bundle, destination):
    spec = json.loads((bundle / 'manifest.json').read_text())
    source = bundle / 'snapshot.tar.gz'
    if hashlib.sha256(source.read_bytes()).hexdigest() != spec['archive_sha256']:
        raise ValueError('Downstream archive checksum mismatch')
    if destination.exists():
        raise FileExistsError(destination)
    with tarfile.open(source, 'r:gz') as archive:
        members = archive.getmembers()
        if {m.name for m in members} != set(spec['files']) or len(members) != len(spec['files']):
            raise ValueError('Archive and manifest disagree')
        for member in members:
            p = PurePosixPath(member.name)
            if not member.isfile() or p.is_absolute() or '..' in p.parts or '\\' in member.name:
                raise ValueError(f'Unsafe archive member: {member.name}')
        for member in members:
            data = archive.extractfile(member).read()
            expected = spec['files'][member.name]
            if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                raise ValueError(f'Downstream file checksum mismatch: {member.name}')
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return len(members)


if __name__ == '__main__':
    root = Path.cwd()
    print(f"Verified and unpacked {unpack(root / 'data/downstream', root / 'data/downstream/records')} downstream files")
