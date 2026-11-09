"""File transport, output preservation, and signal forwarding through fake devicectl."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import suite
from ios_test import TransportTest
from ios import Invocation
from ios_cli import console_result, execute
from android_cli import ChatInput


class CLITest(TransportTest):
    def test_console(self):
        output,code=console_result(b'12:00:00  Acquired tunnel connection to device.\nLaunched application x\nWaiting for the application to terminate\ntext\nApplication exited with exit code 7\n')
        self.assertEqual((output,code),(b'text\n',7))
        self.assertEqual(console_result(b'Application exited with signal 9\n')[1],137)
        self.assertIsNone(console_result(b'text\n')[1])
        self.assertEqual(console_result(b'answer without newlineThe app terminated with the exit code 0.\n'), (b'answer without newline', 0))
        self.assertEqual(console_result(b'12:00:00 a generated answer\r\n')[0], b'12:00:00 a generated answer\n')
        self.assertEqual(console_result(b'first\r\noriginal CRLF\r\r\nThe app terminated with the exit code 2.\n'),
                         (b'first\noriginal CRLF\r\n', 2))

    def test_real_signal_trailer(self):
        for prefix in (b'', b'partial output', b'partial output\r\n'):
            for number in (9, 15):
                data = prefix + f'App terminated due to signal {number}.\n'.encode()
                self.assertEqual(console_result(data), (prefix.replace(b'\r\n', b'\n'), 128 + number))

    def test_signal_preserves_failure_and_stops_suite(self):
        marker = self.root / 'terminated'
        failure = self.root / 'failures'
        self.device.env.update(FAKE_IOS_SIGNAL='9', MOE_IOS_TERMINATION_REPORT=str(marker),
                               MOE_IOS_FAILURE_DIR=str(failure))
        output = self.root / 'failed-output'
        code, stdout, _ = execute(self.device, ['run', '--out', str(output)], b'input')
        self.assertEqual((code, stdout), (137, b'example output'))
        self.assertTrue(marker.exists())
        probe = next(failure.glob('*/probe.json'))
        remote = self.root / 'Documents/inv' / probe.parent.name
        self.assertEqual(probe.read_bytes(), (remote / 'probe.json').read_bytes())
        self.assertTrue((output / 'result.txt').exists())
        calls = (self.root / 'calls.jsonl').read_bytes()
        self.assertEqual(execute(self.device, ['info'], b'')[0], 137)
        with self.assertRaises(RuntimeError):
            self.device.clean()
        self.assertEqual(calls, (self.root / 'calls.jsonl').read_bytes())
        self.assertTrue(remote.exists())
        def stopped(command, **kwargs):
            Path(os.environ['MOE_IOS_TERMINATION_REPORT']).write_text(marker.read_text())
            return subprocess.CompletedProcess(command, 1)
        with patch.object(sys, 'argv', ['suite', '--platform', 'ios', '--only', '35b-numerical,35b-describe']), \
             patch.dict(os.environ), patch.object(suite.subprocess, 'run', side_effect=stopped) as run, \
             patch('ios.IOS.clean') as clean:
            self.assertEqual(suite.main(), 1)
            self.assertEqual(run.call_count, 1)
            clean.assert_not_called()

    def test_signal_status_survives_pull_failure(self):
        marker = self.root / 'terminated'
        self.device.env.update(FAKE_IOS_SIGNAL='9', MOE_IOS_TERMINATION_REPORT=str(marker),
                               MOE_IOS_FAILURE_DIR=str(self.root / 'failures'))
        with patch.object(self.device, 'pull', side_effect=RuntimeError('copy failed')):
            code, _, errors = execute(self.device, ['info'], b'')
        self.assertEqual(code, 137)
        self.assertTrue(marker.exists())
        self.assertIn(b'probe could not be retrieved', errors)

    def test_observe_skip_only_ios(self):
        for platform, config in suite.PLATFORMS.items():
            command = suite.scenarios(**config)['35b-numerical']
            self.assertEqual('--skip-observe' in command, platform == 'ios')
            if platform == 'ios':
                self.assertEqual(command[command.index('--skip-observe') + 1], 'long-mixed')

    def test_mapping(self):
        model=self.root/'model';model.mkdir();(model/'assets.tsv').write_text('manifest')
        self.device.env['MOE_ASSETS_35B']=str(model)
        remote=self.root/'Documents/assets/35b';remote.mkdir(parents=True);(remote/'assets.tsv').write_text('wrong')
        with self.assertRaises(RuntimeError):Invocation(self.device).path(str(model))
        (remote/'assets.tsv').write_text('manifest')
        self.assertEqual(Invocation(self.device).path(str(model)),'assets/35b')
        file=self.root/'text';file.write_text('input')
        inv=Invocation(self.device);chat=ChatInput(inv)
        self.assertIn(b'inv/',chat.line(('/file '+str(file)+'\n').encode()))
        chat.line(b'"""\n');line=('/file '+str(file)+'\n').encode()
        self.assertEqual(chat.line(line),line)
        chat.line(b'"""\n');self.assertEqual(chat.line(b'/file missing\n'),b'/file missing\n')

    def test_outputs_on_success_and_failure(self):
        for code in (0,7):
            self.device.env['FAKE_IOS_EXIT']=str(code)
            output=self.root/f'out-{code}'
            got,stdout,stderr=execute(self.device,['run','--out',str(output)],b'hello\n')
            self.assertEqual(got,code);self.assertEqual(stdout,b'example output\n');self.assertEqual(stderr,b'example stderr\n')
            self.assertEqual((output/'result.txt').read_text(),'result')
            self.assertEqual((output/'nested/bytes.bin').read_bytes(),bytes(range(256)))
            calls=[json.loads(line) for line in (self.root/'calls.jsonl').read_text().splitlines()]
            launch=next(c for c in reversed(calls) if c['args'][:3]==['device','process','launch'])
            env=launch['env'];self.assertIn('DEVICECTL_CHILD_MOE_INVOCATION_DIR',env)
            self.assertEqual((self.root/'Documents'/env['DEVICECTL_CHILD_MOE_STDIN_FILE']).read_bytes(),b'hello\n')
        self.device.env['FAKE_IOS_EXIT']='0';self.device.clean()
        self.assertFalse((self.root/'Documents/inv').exists())

    def _ios_cli(self):
        # Windows cannot execute a shebang script as a process image.
        return [sys.executable, str(Path(__file__).with_name('ios-cli'))]

    def test_skip(self):
        for args,extra in [(['serve'],{}),(['info'],{'FAKE_IOS_UNREACHABLE':'1'}),(['info'],{'FAKE_IOS_TRANSPORT':'wifi'}),(['info'],{'FAKE_IOS_LOCKED':'1'})]:
            self.assertEqual(subprocess.run([*self._ios_cli(),*args],env=dict(self.env,**extra),input=b'',capture_output=True).returncode,77)

    def test_signals(self):
        if os.name == 'nt':
            self.skipTest('POSIX signal exit codes are not available on Windows')
        for sig in (signal.SIGINT,signal.SIGTERM):
            marker=self.root/'launched';marker.unlink(missing_ok=True)
            p=subprocess.Popen([*self._ios_cli(),'info'],env=dict(self.env,FAKE_IOS_WAIT='1', MOE_IOS_FAILURE_DIR=str(self.root/'failures')),stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                while not marker.exists():
                    self.assertIsNone(p.poll(), 'Transport exited before its signal handshake')
                    time.sleep(.01)
                p.send_signal(sig)
                p.communicate();self.assertEqual(p.returncode,128+sig)
            finally:
                if p.poll() is None:p.kill();p.wait()

    def test_forbidden_option(self):
        forbidden='--remove-'+'existing-content'
        for name in ('ios.py','ios_cli.py','ios_push.py','ios-cli'):
            self.assertNotIn(forbidden,Path(__file__).with_name(name).read_text())


if __name__=='__main__':unittest.main()
