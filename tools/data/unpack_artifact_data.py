"""Verify and unpack the artifact bundle to a new processed-data directory."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


def unpack(bundle, destination):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if destination.exists(): raise FileExistsError(destination)
    # Validate all paths before creating any files.
    for name in manifest:
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError(f'Unsafe bundle path {name}')
    for name, record in manifest.items():
        compressed = (bundle / (name + '.gz')).read_bytes()
        if hashlib.sha256(compressed).hexdigest() != record['gzip_sha256']: raise ValueError(f'Compressed hash mismatch {name}')
        data = gzip.decompress(compressed)
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']: raise ValueError(f'Published hash mismatch {name}')
        output = destination / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
    return len(manifest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', type=Path, default=Path('data/bundle'))
    ap.add_argument('--destination', type=Path, default=Path('data/processed'))
    args = ap.parse_args()
    print(f'Verified and unpacked {unpack(args.bundle, args.destination)} published files')


if __name__ == '__main__': main()
