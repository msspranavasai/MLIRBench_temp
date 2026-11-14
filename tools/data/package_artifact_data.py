"""Create a deterministic compressed distribution without changing published data."""
import gzip
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def package(source, target):
    if target.exists(): raise FileExistsError(target)
    target.mkdir(parents=True)
    manifest = {}
    for p in sorted(source.rglob('*')):
        if not p.is_file() or p.suffix not in ('.jsonl', '.json', '.md') or any(part.startswith('.') for part in p.relative_to(source).parts): continue
        rel = p.relative_to(source)
        out = target / (str(rel) + '.gz')
        out.parent.mkdir(parents=True, exist_ok=True)
        with p.open('rb') as src, out.open('xb') as dst:
            with gzip.GzipFile(filename='', fileobj=dst, mode='wb', mtime=0) as gz:
                shutil.copyfileobj(src, gz)
        manifest[str(rel)] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size,
                              'gzip_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}
    (target / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return {'files': len(manifest), 'uncompressed_bytes': sum(r['bytes'] for r in manifest.values()),
                      'compressed_bytes': sum(p.stat().st_size for p in target.rglob('*.gz'))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path('data/processed'))
    ap.add_argument('--destination', type=Path, default=Path('data/bundle'))
    args = ap.parse_args()
    print(json.dumps(package(args.source, args.destination), indent=2))


if __name__ == '__main__': main()
