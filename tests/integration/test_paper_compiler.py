"""Real isolated historical compiler; core runtime package remains unchanged."""
import importlib.metadata
from pathlib import Path
import subprocess
import tempfile
import unittest
from mlirbench.compiler_experiments import toolchain,fresh_compiler_experiments
ROOT=Path(__file__).resolve().parents[2]


class PaperCompiler(unittest.TestCase):
    def test_original_backend_is_isolated_from_core_execution(self):
        info=toolchain()
        self.assertIn('3.12.0rc20260916',info['iree-compile']['version'])
        self.assertEqual(importlib.metadata.version('iree-compiler'),'20241104.1068')
        p=subprocess.run(['iree-compile','--version'],capture_output=True,text=True,check=True)
        self.assertIn('20241104.1068',p.stdout+p.stderr)

    def test_both_dialects_reproduce_first_record_transformation(self):
        with tempfile.TemporaryDirectory() as temp:
            for dialect in ('linalg','torch'):
                with self.subTest(dialect=dialect):
                    report=fresh_compiler_experiments(ROOT,Path(temp)/dialect,dialect=dialect,limit=1,workers=1)
                    self.assertTrue(report['transformation_agreement'],report['mismatches'])
                    self.assertFalse(report['full_corpus'])
