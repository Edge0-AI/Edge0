"""Isolated ephemeral-port server fixtures for Portal tests."""
import http.client
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import urllib.request


class Server:
    def __init__(self, cli, model, root, backend='cpu', extra=(), alias='test'):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.out = self.root / 'out'
        self.lines = []
        self.changed = threading.Condition()
        self.ready = threading.Event()
        # On Windows a graceful stop is a console Ctrl+Break, which can only target a
        # dedicated process group; create one so stop() can signal just this server.
        creation = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        self.process = subprocess.Popen([str(cli), 'serve', '--model', str(model), '--device', backend,
            *(['--alias', alias] if alias else []), '--port', '0', '--temperature', '0', '--presence-penalty', '0',
            '--max-tokens', '4', *(['--context', '4096'] if Path(model).name == 'tiny-a' else []), '--out', str(self.out), *extra], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=creation)
        def read():
            for line in self.process.stderr:
                with self.changed:
                    self.lines.append(line)
                    self.changed.notify_all()
                match = re.search(r'Listening: http://[^:]+:(\d+)', line)
                if match:
                    self.port = int(match[1])
                    self.ready.set()
            self.ready.set()
            with self.changed:
                self.changed.notify_all()
        self.reader = threading.Thread(target=read)
        self.reader.start()
        self.ready.wait()
        assert hasattr(self, 'port'), ''.join(self.lines)
        self.base = f'http://127.0.0.1:{self.port}'

    def request(self, endpoint, body=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port)
        conn.request('GET' if body is None else 'POST', endpoint,
                     None if body is None else json.dumps(body), {'Content-Type': 'application/json'})
        response = conn.getresponse()
        status, text = response.status, response.read().decode()
        conn.close()
        return status, text

    def chat(self, messages=None, **kwargs):
        return {'model': 'test', 'messages': messages or [{'role': 'user', 'content': 'hello'}], **kwargs}

    def summaries(self):
        return [json.loads(line) for line in list(self.lines) if line.startswith('{') and 'stop_reason' in line]

    def evidence(self, expected_requests=None, timeout=30):
        if expected_requests is not None:
            with self.changed:
                available = self.changed.wait_for(
                    lambda: len(self.summaries()) >= expected_requests or self.process.poll() is not None,
                    timeout=timeout)
                assert available and len(self.summaries()) >= expected_requests, ''.join(self.lines)

        return [(json.loads((p / 'tokens.json').read_text(encoding='utf-8')), json.loads((p / 'result.json').read_text(encoding='utf-8')))
                for p in sorted(self.out.glob('*'), key=lambda p: int(p.name))]

    def stop(self):
        if self.process.poll() is None:
            # Windows cannot catch terminate() (TerminateProcess), so deliver a
            # graceful Ctrl+Break the server handles (SIGBREAK) instead.
            self.process.send_signal(signal.CTRL_BREAK_EVENT if os.name == 'nt' else signal.SIGTERM)
        code = self.process.wait()
        self.reader.join()
        assert code == 0, (code, ''.join(self.lines))
        return ''.join(self.lines)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()


def events(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']


def tiny():
    root = Path(os.environ.get('MOE_TINY', '/nonexistent')) / 'tiny-a'
    if not (root / 'assets.tsv').is_file():
        print('SKIP: missing tiny-a')
        raise SystemExit(77)
    return root
