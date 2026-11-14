"""Create a deterministic compressed distribution without changing published data."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path


def main():
    source, target = Path('data/processed'), Path('data/artifact_bundle')
    if target.exists(): raise FileExistsError(target)
    target.mkdir(parents=True)
    manifest = {}
    for p in sorted(source.rglob('*')):
        if not p.is_file() or p.suffix not in ('.jsonl', '.json', '.md'): continue
        rel = p.relative_to(source)
        out = target / (str(rel) + '.gz')
        out.parent.mkdir(parents=True, exist_ok=True)
        with p.open('rb') as src, out.open('xb') as dst:
            with gzip.GzipFile(filename='', fileobj=dst, mode='wb', mtime=0) as gz:
                shutil.copyfileobj(src, gz)
        manifest[str(rel)] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size,
                              'gzip_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}
    (target / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'files': len(manifest), 'uncompressed_bytes': sum(r['bytes'] for r in manifest.values()),
                      'compressed_bytes': sum(p.stat().st_size for p in target.rglob('*.gz'))}, indent=2))


if __name__ == '__main__': main()
