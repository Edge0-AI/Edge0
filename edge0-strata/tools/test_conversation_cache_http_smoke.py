"""Offline tests of HTTP smoke-test orchestration and safety defaults."""
from collections import OrderedDict
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # runnable as tools/test_x.py from the repo root too

from tools.conversation_cache_http_smoke import run


class FakeClient:
    def __init__(self):
        self.history = []
        self.cache = OrderedDict()

    def metrics(self):
        return {'live': {'state': 'idle', 'queued': 0},
                'engine': {'model': 'fake', 'conversation_cache_slots': 4, 'conversation_cache_mib': 8192},
                'requests': self.history[::-1]}

    def request(self, model, content, stream=False, count=8):
        reused = 100 if content in self.cache else 0
        self.cache.pop(content, None)
        self.cache[content] = True
        while len(self.cache) > 5:  # active plus four parked entries
            self.cache.popitem(last=False)
        self.history.append({'time': len(self.history) + 1, 'reused': reused, 'output_tokens': 5,
                             'finish': 'disconnect' if stream else 'stop'})
        data = {'choices': [{'delta': {'content': '1'}}]} if stream else {
            'choices': [{'message': {'role': 'assistant', 'content': 'blue'}}]}
        return io.BytesIO((('data: ' if stream else '') + json.dumps(data) + '\n').encode())


class HttpSmoke(unittest.TestCase):
    def test_orchestration_offline(self):
        result = run(FakeClient())
        self.assertEqual(len(result['records']), 12)
        self.assertEqual(result['records'][-1]['name'], 'cancelled-A-restored')
        self.assertGreater(result['records'][-1]['reused'], 0)

    def test_busy_server_receives_no_requests(self):
        client = Mock()
        client.metrics.return_value = {'live': {'state': 'generating', 'queued': 0}}
        with self.assertRaisesRegex(RuntimeError, 'busy'):
            run(client)
        client.request.assert_not_called()

    def test_disabled_cache_receives_no_requests(self):
        client = Mock()
        client.metrics.return_value = {'live': {'state': 'idle', 'queued': 0}, 'engine': {}}
        with self.assertRaisesRegex(RuntimeError, 'enabled cache'):
            run(client)
        client.request.assert_not_called()

    def test_dry_run_does_not_connect_or_create_output(self):
        with tempfile.TemporaryDirectory(prefix='strata-http-dry-') as directory:
            output = Path(directory) / 'not-created.json'
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('conversation_cache_http_smoke.py')),
                                     '--url', 'invalid://not-a-server', '--output', str(output)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('No network traffic', result.stdout)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
