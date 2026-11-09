"""Transport and push admission, using an isolated fake adb only."""
import hashlib
import json
import os
import shutil
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from android import Android, AndroidApp, Invocation, Unavailable, sync_assets
import android_push

HERE = Path(__file__).resolve().parent


class AndroidTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / 'adb.jsonl'
        self.remote = self.root / 'device'
        self.remote.mkdir()
        self.env = dict(os.environ, MOE_ADB=str(HERE / 'fake_adb.py'), MOE_ANDROID_ROOT=str(self.remote),
                        FAKE_ADB_LOG=str(self.log), ANDROID_SERIAL='fake-only')
        self.patch = patch.dict(os.environ, self.env)
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.device = Android()

    def commands(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def asset(self):
        local = self.root / 'assets'; local.mkdir()
        (local / 'weight.bin').write_bytes(b'weight')
        digest = hashlib.sha256(b'weight').hexdigest()
        (local / 'assets.tsv').write_text(f'role\tbytes\tsha256\tfilename\nbase\t6\t{digest}\tweight.bin\n')
        (local / 'model.json').write_text('{}')
        return local

    def test_push_check_and_mismatch(self):
        local = self.asset(); remote = str(self.remote / '35b')
        self.device.reachable()
        self.assertTrue(sync_assets(self.device, local, remote))
        self.assertFalse(sync_assets(self.device, local, remote, check=True))
        self.log.write_text('')
        self.assertFalse(sync_assets(self.device, local, remote))
        self.assertFalse(any(c[0] == 'push' for c in self.commands()))
        (Path(remote) / 'weight.bin').write_text('bad')
        self.log.write_text('')
        self.assertEqual(sync_assets(self.device, local, remote, check=True), ['weight.bin'])
        self.assertFalse(any(c[0] == 'push' for c in self.commands()))
        self.assertEqual(sync_assets(self.device, local, remote), ['weight.bin'])
        self.assertEqual((Path(remote) / 'weight.bin').read_bytes(), b'weight')

    def test_manifest_mapping_and_temporary_cleanup(self):
        local = self.asset(); remote = str(self.remote / '35b')
        sync_assets(self.device, local, remote)
        self.device.env['MOE_ASSETS_35B'] = str(local)
        with Invocation(self.device) as invocation:
            self.assertEqual(invocation.path(str(local)), remote)
            temporary = self.root / 'input file'; temporary.write_text('hello')
            mapped = invocation.path(str(temporary))
            self.assertEqual(Path(mapped).read_text(), 'hello')
            argv = invocation.arguments(['run', '--model', str(local), '--user-file', str(temporary), '--user', str(temporary)])
            self.assertEqual(argv[2], remote)
            self.assertEqual(argv[4], mapped)
            self.assertEqual(argv[6], str(temporary))
            work = Path(invocation.remote)
        self.assertFalse(work.exists())
        (Path(remote) / 'assets.tsv').write_text('different')
        with Invocation(self.device) as invocation:
            with self.assertRaisesRegex(RuntimeError, 'manifest differs'):
                invocation.path(str(local))

    def test_push_cli_check_is_read_only(self):
        source = self.asset()
        tiny = self.root / 'tiny'
        for name in ('tiny-a', 'tiny-b'):
            shutil.copytree(source, tiny/name)
        binary = self.root / 'binary'; binary.write_bytes(b'fake executable')
        env = dict(self.env, MOE_TINY=str(tiny), MOE_ASSETS_35B=str(source))
        command = [sys.executable, str(HERE/'android_push.py'), '--binary', str(binary)]
        missing = subprocess.run(command + ['--check'], env=env, capture_output=True, text=True)
        self.assertEqual(missing.returncode, 1, missing.stderr)
        self.assertFalse(any(c[0] == 'push' or (c[0] == 'shell' and 'mkdir' in c[-1]) for c in self.commands()))
        installed = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(installed.returncode, 0, installed.stderr)
        self.log.write_text('')
        checked = subprocess.run(command + ['--check'], env=env, capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertFalse(any(c[0] == 'push' for c in self.commands()))

    def test_push_selection(self):
        source = self.asset()
        tiny = self.root / 'tiny'
        for name in ('tiny-a', 'tiny-b'):
            shutil.copytree(source, tiny / name)
        binary = self.root / 'binary'; binary.write_bytes(b'fake')
        env = dict(self.env, MOE_ASSETS_35B=str(source))
        env.pop('MOE_TINY', None)
        command = [sys.executable, str(HERE / 'android_push.py'), '--binary', str(binary)]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.remote / '35b/assets.tsv').exists())
        self.assertFalse((self.remote / 'tiny-a').exists())
        result = subprocess.run(command + ['--tiny'], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('--tiny requires MOE_TINY', result.stderr)
        env['MOE_TINY'] = str(tiny)
        result = subprocess.run(command + ['--tiny'], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.remote / 'tiny-a/assets.tsv').exists())
        self.assertTrue((self.remote / 'tiny-b/assets.tsv').exists())

    def test_app_mode_target_and_check(self):
        source = self.asset()
        private = self.root / 'app-private'; private.mkdir()
        observed = []
        def execute(argv, **kwargs):
            args = list(argv)
            if 'shell' in args:
                words = shlex.split(args[-1])
                observed.append(words)
                if words[:2] == ['run-as', 'example.synthetic']:
                    words = [str(private / w) if w.startswith('files/') else w for w in words[2:]]
                    args[-1] = shlex.join(words)
            return subprocess.run(args, **kwargs)
        def device_factory(package, environ):
            return AndroidApp(package, execute=execute, environ=environ)
        with patch.dict(os.environ, dict(self.env, MOE_ASSETS_35B=str(source)), clear=True), \
             patch.object(android_push, 'AndroidApp', side_effect=device_factory):
            with patch.object(sys, 'argv', ['android_push.py', '--app', 'example.synthetic', '--check']):
                self.assertEqual(android_push.main(), 1)
            self.assertFalse(any(c[0] == 'push' or (c[0] == 'shell' and 'mkdir' in c[-1]) for c in self.commands()))
            with patch.object(sys, 'argv', ['android_push.py', '--app', 'example.synthetic']):
                self.assertEqual(android_push.main(), 0)
            self.assertEqual((private / 'files/assets/35b/weight.bin').read_bytes(), b'weight')
            self.assertFalse(list(self.remote.glob('app-transfer-*')))
            self.log.write_text('')
            with patch.object(sys, 'argv', ['android_push.py', '--app', 'example.synthetic', '--check']):
                self.assertEqual(android_push.main(), 0)
            self.assertFalse(any(c[0] == 'push' or (c[0] == 'shell' and 'mkdir' in c[-1]) for c in self.commands()))
        self.assertTrue(any(c[:3] == ['run-as', 'example.synthetic', 'cp'] for c in observed))
        self.assertFalse(any('edge0-infer' in str(c) for c in observed))

    def test_offline_exit(self):
        env = dict(self.env, FAKE_ADB_OFFLINE='1')
        for script in ('android-cli', 'android_push.py'):
            result = subprocess.run([sys.executable, str(HERE / script), *(['info'] if script == 'android-cli' else [])],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 77, result.stderr)


if __name__ == '__main__':
    if os.name == 'nt':
        print('NOT EXECUTED on Windows: the Android device-push suite models POSIX device '
              'paths (PurePosixPath) and uses a host directory as a fake device root, which a '
              'Windows path is not; Android is not a Windows target (decision 6)', flush=True)
        raise SystemExit(77)
    unittest.main()
