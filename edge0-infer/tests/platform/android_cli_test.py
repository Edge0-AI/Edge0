"""End-to-end wrapper contracts with real pipes and a local fake transport."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from android_test import AndroidTest, HERE

FAKE_CLI = '''#!/usr/bin/env python3
import json, os, signal, sys, time
from pathlib import Path
args = sys.argv[1:]
mode = args[0]
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
signal.signal(signal.SIGINT, lambda *_: sys.exit(130))
out = Path(args[args.index('--out')+1]) if '--out' in args else None
if out:
    out.mkdir()
    (out/'argv.json').write_text(json.dumps(args))
if mode == 'serve':
    print('Listening: http://127.0.0.1:12345 model=test', file=sys.stderr, flush=True)
    folder=out/'1'; folder.mkdir()
    (folder/'tokens.json').write_text('{}')
    (folder/'result.json').write_text('{}')
    print('{"stop_reason":"length"}', file=sys.stderr, flush=True)
    while True: time.sleep(.01)
elif mode == 'wait' or (mode == 'chat' and '--wait' in args):
    print('ready', flush=True)
    while True: time.sleep(.01)
else:
    data = sys.stdin.buffer.read()
    sys.stdout.buffer.write(data); sys.stdout.buffer.flush()
    if out:
        (out/'stdin').write_bytes(data)
        (out/'uploaded.json').write_text(json.dumps({str(p): p.read_text() for p in out.parent.glob('input-*') if p.is_file()}))
    sys.exit(9 if mode == 'fail' else 0)
'''


class WrapperTest(AndroidTest):
    def setUp(self):
        super().setUp()
        cli = self.remote / 'edge0-infer'
        cli.write_text(FAKE_CLI); cli.chmod(0o755)

    def test_stdin_exit_and_failed_outputs(self):
        for mode, code in (('run', 0), ('fail', 9)):
            output = self.root / mode
            data = b'raw\x00input\n/file unchanged\n'
            result = subprocess.run([str(HERE/'android-cli'), mode, '--out', str(output)], input=data,
                                    capture_output=True, env=self.env, timeout=20)
            self.assertEqual(result.returncode, code, result.stderr)
            self.assertEqual(result.stdout, data)
            self.assertEqual((output/'stdin').read_bytes(), data)
            self.assertEqual(list((self.remote/'calls').iterdir()), [])

    def test_auxiliary_commands_do_not_consume_stdin(self):
        self.env['FAKE_ADB_READ_CONTROL_STDIN'] = '1'
        prompt = self.root / 'prompt.txt'
        prompt.write_text('Input asset')
        output = self.root / 'isolated-stdin'
        data = b'Hello\nOne more turn\n/quit\n'
        result = subprocess.run([str(HERE/'android-cli'), 'chat', '--user-file', str(prompt),
                                 '--out', str(output)], input=data, capture_output=True,
                                env=self.env, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, data)
        self.assertEqual((output/'stdin').read_bytes(), data)
        self.assertEqual(list((self.remote/'calls').iterdir()), [])

    def test_chat_file_mapping_preserves_other_bytes(self):
        prompt = self.root / 'prompt with spaces.txt'
        prompt.write_text('File contents before forwarding')
        path = os.fsencode(prompt)
        missing = os.fsencode(self.root / 'missing.txt')
        data = (b'raw\x00\xff text\r\n  /file\t' + path + b' \t\r\n'
                b' \t"""\r\n/file ' + path + b'\n""" \n'
                b'/file ' + missing + b'\r\n/filex ' + path + b'\n/file\n'
                b'/file ' + os.fsencode(self.root) + b'\n/file ' + path)
        output = self.root / 'chat-out'
        result = subprocess.run([str(HERE/'android-cli'), 'chat', '--out', str(output)],
                                input=data, capture_output=True, env=self.env, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        uploads = json.loads((output/'uploaded.json').read_text())
        self.assertEqual(len(uploads), 1)
        mapped, contents = next(iter(uploads.items()))
        self.assertEqual(contents, prompt.read_text())
        expected = data.replace(b'  /file\t' + path, b'  /file\t' + os.fsencode(mapped), 1)
        expected = expected[:-len(path)] + os.fsencode(mapped)
        self.assertEqual(result.stdout, expected)
        self.assertEqual((output/'stdin').read_bytes(), expected)
        pushes = [c for c in self.commands() if c[0] == 'push']
        self.assertEqual(len(pushes), 1)
        self.assertEqual(list((self.remote/'calls').iterdir()), [])

    def test_non_chat_file_text_is_not_mapped(self):
        prompt = self.root / 'prompt.txt'
        prompt.write_text('Local input')
        data = b'/file ' + os.fsencode(prompt) + b'\n'
        result = subprocess.run([str(HERE/'android-cli'), 'run'], input=data,
                                capture_output=True, env=self.env, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, data)
        self.assertFalse(any(c[0] == 'push' for c in self.commands()))

    def test_signal_forwarding(self):
        for sig, code in ((signal.SIGINT, 130), (signal.SIGTERM, 0)):
            process = subprocess.Popen([str(HERE/'android-cli'), 'wait'], env=self.env,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                self.assertEqual(process.stdout.readline(), b'ready\n')
                process.send_signal(sig)
                stdout, stderr = process.communicate(timeout=20)
                self.assertEqual(process.returncode, code, stderr)
            finally:
                if process.poll() is None:
                    process.kill(); process.communicate()
        kills = [c for c in self.commands() if c[0] == 'shell' and 'kill -' in c[-1]]
        self.assertTrue(any('kill -INT' in c[-1] for c in kills))
        self.assertTrue(any('kill -TERM' in c[-1] for c in kills))

    def test_chat_signal_while_stdin_stays_open(self):
        process = subprocess.Popen([str(HERE/'android-cli'), 'chat', '--wait'], env=self.env,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.assertEqual(process.stdout.readline(), b'ready\n')
            process.stdin.write(b'partial line'); process.stdin.flush()
            process.send_signal(signal.SIGINT)
            code = process.wait(timeout=20)
            _, stderr = process.communicate(timeout=20)
            self.assertEqual(code, 130, stderr)
            self.assertEqual(list((self.remote/'calls').iterdir()), [])
        finally:
            if process.poll() is None:
                process.kill(); process.communicate()

    def test_live_evidence_before_summary_and_forward_cleanup(self):
        output = self.root / 'serve-out'
        process = subprocess.Popen([str(HERE/'android-cli'), 'serve', '--port', '0', '--out', str(output)], env=self.env,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.assertIn(b':43210 ', process.stderr.readline())
            self.assertIn(b'stop_reason', process.stderr.readline())
            self.assertEqual(json.loads((output/'1/result.json').read_text()), {})
            self.assertIsNone(process.poll())
            process.terminate()
            _, stderr = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertIn(['forward', '--remove', 'tcp:43210'], self.commands())
            self.assertEqual(list((self.remote/'calls').iterdir()), [])
        finally:
            if process.poll() is None:
                process.kill(); process.communicate()


if __name__ == '__main__':
    if os.name == 'nt':
        print('NOT EXECUTED on Windows: the Android wrapper suite uses POSIX device paths and '
              'SIGTERM-based fakes; Android is not a Windows target (decision 6)', flush=True)
        raise SystemExit(77)
    unittest.main()
