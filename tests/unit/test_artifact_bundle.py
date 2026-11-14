import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from tools.data.unpack_artifact_data import unpack


class ArtifactBundle(unittest.TestCase):
    def fixture(self, root, name='MLIRBench/master/master.jsonl'):
        bundle = root / 'bundle'; bundle.mkdir()
        data = b'{"published": "False"}\n'; compressed = gzip.compress(data, mtime=0)
        entry = bundle / (name + '.gz'); entry.parent.mkdir(parents=True, exist_ok=True); entry.write_bytes(compressed)
        record = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'gzip_sha256': hashlib.sha256(compressed).hexdigest()}
        (bundle / 'manifest.json').write_text(json.dumps({name: record}))
        return bundle, data, entry

    def test_byte_exact_unpack_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); bundle, data, entry = self.fixture(root)
            out = root / 'processed'
            self.assertEqual(unpack(bundle, out), 1)
            self.assertEqual((out / 'MLIRBench/master/master.jsonl').read_bytes(), data)
            with self.assertRaises(FileExistsError): unpack(bundle, out)

    def test_corrupt_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); bundle, _, entry = self.fixture(root); entry.write_bytes(b'corrupt')
            with self.assertRaises(ValueError): unpack(bundle, root / 'processed')

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); bundle = root / 'bundle'; bundle.mkdir()
            (bundle / 'manifest.json').write_text(json.dumps({'../outside': {}}))
            with self.assertRaises(ValueError): unpack(bundle, root / 'processed')

    def test_pack_excludes_hidden_cache_and_is_deterministic(self):
        from tools.data.package_artifact_data import package
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source'; (source / '.cache').mkdir(parents=True)
            (source / '.cache/state.json').write_text('{}')
            (source / 'master.jsonl').write_text('{"label":"False"}\n')
            package(source, root / 'one'); package(source, root / 'two')
            self.assertEqual(json.loads((root / 'one/manifest.json').read_text()).keys(), {'master.jsonl'})
            self.assertEqual((root / 'one/master.jsonl.gz').read_bytes(), (root / 'two/master.jsonl.gz').read_bytes())
