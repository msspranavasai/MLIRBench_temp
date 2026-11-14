"""Architecture selection must follow the Docker daemon, including remote contexts."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mlirbench.ae import cpu_execution


class CpuEvidence(unittest.TestCase):
    def test_native_emulated_and_unknown_host_are_distinguished(self):
        for machine, host, emulated in [('aarch64', 'arm64', False),
                                         ('x86_64', 'aarch64', True),
                                         ('x86_64', 'amd64', False),
                                         ('aarch64', None, None)]:
            with self.subTest(machine=machine, host=host):
                env = {'MLIRBENCH_DOCKER_HOST_ARCH': host} if host else {}
                with patch.dict(os.environ, env, clear=True), patch('platform.machine', return_value=machine):
                    self.assertIs(cpu_execution()['emulated'], emulated)


class Launcher(unittest.TestCase):
    def launch(self, architecture, requested=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = Path(__file__).resolve().parents[2] / 'ae.sh'
            (root / 'ae.sh').write_bytes(source.read_bytes())
            docker = root / 'docker'
            docker.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['AE_TEST_LOG'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1:2] == ['info'] and '--format' in sys.argv:
    print(os.environ['AE_TEST_ARCH'])
''')
            docker.chmod(0o755)
            log = root / 'calls.jsonl'
            env = {k: v for k, v in os.environ.items() if k not in ('AE_PLATFORM', 'AE_IMAGE')}
            env.update(PATH=str(root) + os.pathsep + env['PATH'], AE_TEST_LOG=str(log), AE_TEST_ARCH=architecture)
            if requested:
                env['AE_PLATFORM'] = requested
            run = subprocess.run(['bash', str(root / 'ae.sh'), 'dataset-info'], env=env,
                                 capture_output=True, text=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            return run, calls

    def test_defaults_to_docker_daemon_architecture(self):
        for daemon, arch in [('aarch64', 'arm64'), ('x86_64', 'amd64')]:
            with self.subTest(daemon=daemon):
                run, calls = self.launch(daemon)
                self.assertEqual(run.returncode, 0, run.stderr)
                build = next(c for c in calls if c[0] == 'build')
                container = next(c for c in calls if c[0] == 'run')
                for args in (build, container):
                    self.assertEqual(args[args.index('--platform') + 1], 'linux/' + arch)
                    self.assertIn('mlirbench-ae:dev-' + arch, args)
                self.assertIn('MLIRBENCH_DOCKER_HOST_ARCH=' + arch, container)
                self.assertEqual(container[container.index('--network') + 1], 'none')

    def test_cross_architecture_is_explicit(self):
        run, calls = self.launch('aarch64', 'linux/amd64')
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('emulation', run.stdout)
        container = next(c for c in calls if c[0] == 'run')
        self.assertIn('MLIRBENCH_DOCKER_HOST_ARCH=arm64', container)
        self.assertEqual(container[container.index('--platform') + 1], 'linux/amd64')

    def test_unsupported_host_or_requested_platform_stops_before_build(self):
        for daemon, requested in [('s390x', None), ('aarch64', 'linux/riscv64')]:
            with self.subTest(daemon=daemon, requested=requested):
                run, calls = self.launch(daemon, requested)
                self.assertNotEqual(run.returncode, 0)
                self.assertFalse(any(c[0] in ('build', 'run') for c in calls))
