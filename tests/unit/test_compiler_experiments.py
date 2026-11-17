"""Compiler metric replay is arithmetic over observations, never copied summaries."""
import json
from pathlib import Path
import tempfile
import unittest
from mlirbench.compiler_experiments import instructions,torch_ops,truth,source_lookup,toolchain,archived_compiler_metrics
ROOT=Path(__file__).resolve().parents[2]


class CompilerReplay(unittest.TestCase):
    def test_assembly_definition_ignores_labels_directives_and_comments(self):
        text='.text\nmain:\n  add a0, a1, a2 # note\n  vsetvli a0, zero, e32 // note\n# ignored\n ret\n'
        self.assertEqual(instructions(text),['add a0, a1, a2','vsetvli a0, zero, e32','ret'])

    def test_torch_counter_matches_archived_definition(self):
        self.assertEqual(torch_ops('module {\nfunc.func @main() {\n%x = "torch.prim.ConstantInt"()\n"func.return"()\n}\n}'),2)

    def test_string_false_is_not_truthy_and_missing_boolean_is_rejected(self):
        self.assertFalse(truth('False'));self.assertTrue(truth('True'))
        with self.assertRaises(ValueError):truth('')

    def test_source_resolution_uses_exact_trimmed_program_hash(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=root/'data/processed/MLIRBench/master/master.jsonl';p.parent.mkdir(parents=True)
            p.write_text(json.dumps({'input_mlir':'  module {}\n','mutated_mlir':'module { func.func @main() { return } }'})+'\n')
            lookup=source_lookup(root)
            self.assertEqual(lookup[hashlib.sha256(b'module {}').hexdigest()],'module {}')
            self.assertEqual(len(lookup),2)

    def test_missing_pinned_toolchain_fails_before_an_experiment(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):toolchain(temp)

    def test_full_archived_compiler_arithmetic_matches_reference(self):
        report=archived_compiler_metrics(ROOT)
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['checks']),1369)
        self.assertEqual(report['riscv_improvement']['MEDIUM']['improved'],50)
        self.assertFalse(report['fresh_compilation'])
