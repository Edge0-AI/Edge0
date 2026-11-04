"""iOS transport and manifest synchronization without a real device."""
import os
import json
import shutil
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ios import IOS, Unavailable, sync_assets


class TransportTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.env=dict(os.environ,MOE_IOS_DEVICE='synthetic-device',MOE_IOS_TEAM='synthetic-team',
                      MOE_DEVICECTL=str(Path(__file__).with_name('fake_devicectl.py')),FAKE_IOS_ROOT=str(self.root))
        self.device=IOS(environ=self.env)

    def test_connection(self):
        self.device.reachable()
        for key,value in [('FAKE_IOS_UNREACHABLE','1'),('FAKE_IOS_TRANSPORT','wifi'),('FAKE_IOS_LOCKED','1')]:
            with self.assertRaises(Unavailable):IOS(environ=dict(self.env,**{key:value})).reachable()

    def test_sync(self):
        local=self.root/'local';local.mkdir();(local/'base').write_bytes(b'abc')
        (local/'assets.tsv').write_text('filename\tbytes\nbase\t3\n')
        self.assertTrue(sync_assets(self.device,local,'assets/test',check=True))
        self.assertFalse((self.root/'Documents').exists())
        self.assertTrue(sync_assets(self.device,local,'assets/test'))
        self.assertFalse(sync_assets(self.device,local,'assets/test'))
        (self.root/'Documents/assets/test/base').write_bytes(b'x')
        self.assertEqual(sync_assets(self.device,local,'assets/test',check=True),['base'])
        self.assertEqual(sync_assets(self.device,local,'assets/test'),['base'])
        self.assertEqual((self.root/'Documents/assets/test/base').read_bytes(),b'abc')

    def test_push_selection_and_read_only_check(self):
        local = self.root / 'local'; local.mkdir()
        (local / 'base').write_bytes(b'abc')
        (local / 'assets.tsv').write_text('filename\tbytes\nbase\t3\n')
        tiny = self.root / 'tiny'
        for name in ('tiny-a', 'tiny-b'):
            shutil.copytree(local, tiny / name)
        env = dict(self.env, MOE_ASSETS_35B=str(local))
        env.pop('MOE_TINY', None)
        command = [sys.executable, str(Path(__file__).with_name('ios_push.py')), '--app', str(self.root / 'example.app')]
        result = subprocess.run(command + ['--check'], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse((self.root / 'Documents').exists())
        calls = [json.loads(line)['args'] for line in (self.root / 'calls.jsonl').read_text().splitlines()]
        self.assertFalse(any(c[:3] in (['device', 'install', 'app'], ['device', 'copy', 'to']) for c in calls))
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / 'Documents/assets/35b/assets.tsv').exists())
        self.assertFalse((self.root / 'Documents/assets/tiny-a').exists())
        result = subprocess.run(command + ['--tiny'], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('--tiny requires MOE_TINY', result.stderr)
        env['MOE_TINY'] = str(tiny)
        result = subprocess.run(command + ['--tiny'], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ('tiny-a', 'tiny-b'):
            self.assertTrue((self.root / 'Documents/assets' / name / 'assets.tsv').exists())
        result = subprocess.run(command + ['--tiny', '--check'], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_reject_paths(self):
        for path in ('../assets','/assets','Documents','inv/../assets'):
            with self.assertRaises(ValueError):self.device.safe_path(path)


if __name__=='__main__':unittest.main()
