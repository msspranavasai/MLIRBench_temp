"""Architecture selection must follow the Docker daemon, including remote contexts."""
import json
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mlirbench.ae import cpu_execution
from tests.unit.test_pipeline import fixture
from mlirbench.dataset import ensure_data


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
    def launch(self, architecture, requested=None, arguments=None, environment=None, damage=None, shell="bash"):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = Path(__file__).resolve().parents[2] / 'ae.sh'
            (root / 'ae.sh').write_bytes(source.read_bytes())
            project=source.parent
            shutil.copy2(project/'run.py',root/'run.py')
            shutil.copytree(project/'src',root/'src')
            shutil.copytree(project/'tools/release',root/'tools/release')
            fixture(root)
            # A fourth build input, outside the old three-file sentinels.
            import gzip,hashlib
            extra='MLIRBench/README.md';raw=b'Frozen fixture readme\n'
            import io
            buffer=io.BytesIO()
            with gzip.GzipFile(filename='',fileobj=buffer,mode='wb',mtime=0,compresslevel=9) as stream:stream.write(raw)
            packed=buffer.getvalue();(root/'data/bundle'/(extra+'.gz')).write_bytes(packed)
            manifest_path=root/'data/bundle/manifest.json';manifest=json.loads(manifest_path.read_text())
            manifest[extra]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'gzip_sha256':hashlib.sha256(packed).hexdigest()}
            manifest_path.write_text(json.dumps(manifest))
            if damage=='missing_extra_with_raw':
                ensure_data(root);(root/'data/bundle'/(extra+'.gz')).unlink()
            elif damage=='corrupt_extra':
                (root/'data/bundle'/(extra+'.gz')).write_bytes(b'corrupt fixture')
            elif damage=='missing_extra':
                (root/'data/bundle'/(extra+'.gz')).unlink()
            elif damage=='no_data':
                shutil.rmtree(root/'data');(root/'run.py').unlink()
            docker = root / 'docker'
            docker.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['AE_TEST_LOG'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1:2] == ['info'] and '--format' in sys.argv:
    print(os.environ['AE_TEST_ARCH'])
if sys.argv[1:3] == ['image','inspect'] and '--help' in sys.argv:
    print('' if os.environ.get('AE_TEST_OLD_INSPECT') else '--platform')
if sys.argv[1:2] == ['pull'] and os.environ.get('AE_TEST_PULL_FAIL'):
    raise SystemExit(1)
if sys.argv[1:3] == ['image','inspect'] and '--format' in sys.argv:
    if 'source-fingerprint' in sys.argv[sys.argv.index('--format')+1]:
        import subprocess
        print(os.environ.get('AE_TEST_IMAGE_SOURCE_ID') or subprocess.check_output([sys.executable,'tools/release/source_identity.py'],text=True).strip())
    else:
        print(os.environ.get('AE_TEST_IMAGE_PLATFORM') or os.environ.get('AE_PLATFORM') or ('linux/arm64' if os.environ['AE_TEST_ARCH'] in ('aarch64','arm64') else 'linux/amd64'))
if sys.argv[1:2] == ['build']:
    from pathlib import Path
    import hashlib
    root=Path.cwd()
    for name,spec in json.loads((root/'data/bundle/manifest.json').read_text()).items():
        assert hashlib.sha256((root/'data/bundle'/(name+'.gz')).read_bytes()).hexdigest()==spec['gzip_sha256']

''')
            docker.chmod(0o755)
            log = root / 'calls.jsonl'
            env = {k: v for k, v in os.environ.items() if k not in ('AE_PLATFORM', 'AE_IMAGE', 'AE_SKIP_BUILD', 'AE_IMAGE_SOURCE')}
            env.update(PATH=str(root) + os.pathsep + env['PATH'], AE_TEST_LOG=str(log), AE_TEST_ARCH=architecture, AE_OUTPUT_ROOT=str(root/'outputs'))
            env['AE_IMAGE_SOURCE']='build'
            if environment and 'AE_SKIP_BUILD' in environment and 'AE_IMAGE_SOURCE' not in environment:
                env.pop('AE_IMAGE_SOURCE')
            env.update(environment or {})
            if requested:
                env['AE_PLATFORM'] = requested
            run = subprocess.run([shell, str(root / 'ae.sh'), *(arguments or ['dataset-info'])], env=env,
                                 capture_output=True, text=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            return run, calls

    def test_system_bash_handles_empty_and_populated_optional_arrays(self):
        for shell in dict.fromkeys(('bash', '/bin/bash')):
            for populated in (False, True):
                with self.subTest(shell=shell, populated=populated):
                    env={'AE_IMAGE_SOURCE':'pull','AE_TEST_OLD_INSPECT':'1'}
                    args=['dataset-info']
                    if populated:
                        env['OPENAI_API_KEY']='dummy-test-key'
                        args.append('--allow-paid-api')
                    run,calls=self.launch('arm64',environment=env,arguments=args,shell=shell)
                    self.assertEqual(run.returncode,0,run.stderr)
                    containers=[c for c in calls if c[0]=='run']
                    self.assertEqual(len(containers),1)
                    self.assertEqual('--env' in containers[0],populated)

    def test_defaults_to_docker_daemon_architecture(self):
        for daemon, arch in [('aarch64', 'arm64'), ('x86_64', 'amd64')]:
            with self.subTest(daemon=daemon):
                run, calls = self.launch(daemon)
                self.assertEqual(run.returncode, 0, run.stderr)
                build = next(c for c in calls if c[0] == 'build')
                container = next(c for c in calls if c[0] == 'run')
                for args in (build, container):
                    self.assertEqual(args[args.index('--platform') + 1], 'linux/' + arch)
                    self.assertIn('mlirbench-ae:reviewer', args)
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

    def test_live_settings_are_forwarded_only_with_explicit_paid_opt_in(self):
        settings = {'MLIRBENCH_API_TIMEOUT_SECONDS':'17', 'MLIRBENCH_API_MAX_RETRIES':'0',
                    'MLIRBENCH_MUTATION_MAX_OUTPUT_TOKENS':'1234',
                    'MLIRBENCH_REASONING_MAX_OUTPUT_TOKENS':'789',
                    'MLIRBENCH_JUDGE_MAX_OUTPUT_TOKENS':'567',
                    'MLIRBENCH_MUTATION_MAX_ATTEMPTS':'1',
                    'OPENAI_API_KEY':'fixture-private-value'}
        for paid in (False, True):
            with self.subTest(paid=paid):
                args = ['pipeline', '--allow-paid-api'] if paid else ['pipeline']
                run, calls = self.launch('aarch64', arguments=args, environment=settings)
                self.assertEqual(run.returncode, 0, run.stderr)
                container = next(c for c in calls if c[0] == 'run')
                forwarded = {container[i+1] for i, arg in enumerate(container[:-1]) if arg == '--env'}
                self.assertEqual(forwarded.intersection(settings), set(settings) if paid else set())
                self.assertNotIn(settings['OPENAI_API_KEY'], str(calls))

    def test_missing_non_sentinel_bundle_is_repaired_before_build(self):
        run,calls=self.launch('aarch64',damage='missing_extra_with_raw')
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertIn('MLIRBench/README.md',run.stdout)
        self.assertTrue(any(c[0]=='build' for c in calls))

    def test_missing_or_corrupt_non_sentinel_stops_before_build(self):
        for damage in ('missing_extra','corrupt_extra'):
            with self.subTest(damage=damage):
                run,calls=self.launch('aarch64',damage=damage)
                self.assertNotEqual(run.returncode,0)
                self.assertFalse(any(c[0] in ('build','run') for c in calls))

    def test_reuse_existing_image_needs_no_host_data_or_build(self):
        run,calls=self.launch('aarch64',environment={'AE_SKIP_BUILD':'1'},damage='no_data')
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertTrue(any(c[:2]==['image','inspect'] for c in calls))
        self.assertFalse(any(c[0]=='build' for c in calls))
        self.assertTrue(any(c[0]=='run' for c in calls))

    def test_wrong_architecture_local_image_stops_before_container(self):
        run,calls=self.launch('aarch64', environment={'AE_SKIP_BUILD':'1','AE_TEST_IMAGE_PLATFORM':'linux/amd64'})
        self.assertNotEqual(run.returncode,0)
        self.assertIn('Rebuild',run.stderr)
        self.assertFalse(any(c[0]=='run' for c in calls))

    def test_default_pulls_one_native_image_without_host_data_or_build(self):
        for arch in ('aarch64','x86_64'):
            with self.subTest(arch=arch):
                run,calls=self.launch(arch,environment={'AE_IMAGE_SOURCE':''},damage='no_data')
                self.assertEqual(run.returncode,0,run.stderr)
                pulls=[c for c in calls if c[0]=='pull'];containers=[c for c in calls if c[0]=='run']
                self.assertEqual(len(pulls),1);self.assertEqual(len(containers),1)
                self.assertIn('ghcr.io/msspranavasai/mlirbench-ae:0.3.0rc5',pulls[0])
                self.assertFalse(any(c[0]=='build' for c in calls))
                self.assertEqual(containers[0][containers[0].index('--network')+1],'none')

    def test_pull_or_source_identity_failure_never_starts_container(self):
        for extra in ({'AE_TEST_PULL_FAIL':'1'},{'AE_TEST_IMAGE_SOURCE_ID':'stale'}):
            with self.subTest(extra=extra):
                run,calls=self.launch('aarch64',environment={'AE_IMAGE_SOURCE':'pull',**extra})
                self.assertNotEqual(run.returncode,0)
                self.assertFalse(any(c[0] in ('build','run') for c in calls))

    def test_pull_mode_inspects_the_selected_platform(self):
        run,calls=self.launch('aarch64','linux/amd64',environment={'AE_IMAGE_SOURCE':'pull'})
        self.assertEqual(run.returncode,0,run.stderr)
        inspect=next(c for c in calls if c[:2]==['image','inspect'] and '--format' in c)
        self.assertEqual(inspect[inspect.index('--platform')+1],'linux/amd64')

    def test_invalid_image_source_stops_before_pull_build_or_run(self):
        run,calls=self.launch('aarch64',environment={'AE_IMAGE_SOURCE':'unknown'})
        self.assertNotEqual(run.returncode,0)
        self.assertFalse(any(c[0] in ('pull','build','run') for c in calls))
