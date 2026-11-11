"""Elastic encoder failures reach clients, can be retried, and do not prevent recovery."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
import urllib.request

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, Vision, serve
from serve.test_vision_elastic import ElasticVision
from serve.test_vram import VramEngine


class EncoderFailureHTTP(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tok = ByteTokenizer()
        self.engine = VramEngine(tok)
        self.engine.loaded = True
        self.engine.elastic = False  # as on a layer split: the cache cannot make room
        self.vision = ElasticVision(tmp.name)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / 'chat_template.jinja'),
                           vision=self.vision)
        httpd = serve(self.svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.addCleanup(self.drop_embeddings)
        self.base = f'http://127.0.0.1:{httpd.server_address[1]}'
        self.attempts = 0
        self.original_restart = self.vision.restart

    def drop_embeddings(self):
        path = getattr(getattr(self.svc, 'embeddings', None), 'path', None)
        if path is not None:
            path.unlink(missing_ok=True)

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        return urllib.request.urlopen(req, timeout=10)

    @staticmethod
    def chat_picture():
        return {'model': 'm', 'max_tokens': 8, 'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,eA=='}}]}]}

    def fail_with(self, error_type):
        def fail():
            self.attempts += 1
            raise error_type('synthetic encoder startup failure')
        self.vision.restart = fail

    def assert_503(self, path, body):
        with self.assertRaises(urllib.error.HTTPError) as got:
            self.post(path, body)
        with got.exception as response:
            self.assertEqual(response.code, 503)
            payload = json.load(response)
            self.assertIn('synthetic encoder startup failure', payload['error']['message'])

    def test_start_failure_and_retry_on_all_three_apis(self):
        cases = [
            ('/v1/chat/completions', self.chat_picture()),
            ('/v1/messages', {'model': 'm', 'max_tokens': 8, 'messages': [{'role': 'user', 'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'eA=='}}]}]}),
            ('/v1/responses', {'model': 'm', 'max_output_tokens': 8, 'input': [{'role': 'user', 'content': [
                {'type': 'input_image', 'image_url': 'data:image/png;base64,eA=='}]}]}),
        ]
        for error_type in (RuntimeError, OSError):
            self.fail_with(error_type)
            for path, body in cases:
                for stream in (False, True):
                    for retry in range(2):
                        with self.subTest(error=error_type.__name__, path=path, stream=stream, retry=retry):
                            before = self.attempts
                            self.assert_503(path, {**body, 'stream': stream})
                            self.assertEqual(self.attempts, before + 1)

    def test_text_requests_and_image_recovery_after_a_failed_start(self):
        self.fail_with(RuntimeError)
        text = {'model': 'm', 'max_tokens': 8, 'messages': [{'role': 'user', 'content': 'hi'}]}
        with self.post('/v1/chat/completions', text) as response:
            self.assertEqual(response.status, 200)
            response.read()
        self.assertEqual(self.attempts, 0)
        self.assert_503('/v1/chat/completions', self.chat_picture())
        self.assertEqual(self.attempts, 1)
        self.vision.restart = self.original_restart
        with self.post('/v1/chat/completions', self.chat_picture()) as response:
            self.assertEqual(response.status, 200)
            response.read()
        self.assertEqual(self.vision.starts, 1)
        with self.post('/v1/chat/completions', text) as response:
            self.assertEqual(response.status, 200)
            response.read()
        self.assertEqual(self.vision.starts, 1)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'child reaping is checked through /proc')
    def test_real_encoder_failure_retries_and_reaps_each_child(self):
        with tempfile.TemporaryDirectory() as directory:
            started = Path(directory) / 'started-pids'
            executable = Path(directory) / 'encoder-exits'
            executable.write_text(
                '#!' + sys.executable + '\nimport os\n'
                'with open(' + repr(str(started)) + ', "a") as output:\n'
                '    output.write(str(os.getpid()) + "\\n")\n'
                'raise SystemExit(71)\n')
            executable.chmod(0o700)
            vision = Vision({'exe': str(executable), 'mmproj': 'unused', 'model': 'unused',
                             'elastic': True}, lazy=True)
            self.addCleanup(vision.unload)
            self.svc.vision = vision
            cases = [
                ('/v1/chat/completions', self.chat_picture()),
                ('/v1/messages', {'model': 'm', 'max_tokens': 8, 'messages': [{'role': 'user', 'content': [
                    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'eA=='}}]}]}),
                ('/v1/responses', {'model': 'm', 'max_output_tokens': 8, 'input': [{'role': 'user', 'content': [
                    {'type': 'input_image', 'image_url': 'data:image/png;base64,eA=='}]}]}),
            ]
            for path, body in cases:
                for stream in (False, True):
                    for retry in range(2):
                        with self.subTest(path=path, stream=stream, retry=retry):
                            previous = started.read_text().splitlines() if started.exists() else []
                            with self.assertRaises(urllib.error.HTTPError) as got:
                                self.post(path, {**body, 'stream': stream})
                            with got.exception as response:
                                self.assertEqual(response.code, 503)
                                self.assertIn('the image encoder failed to start', json.load(response)['error']['message'])
                            children = started.read_text().splitlines()
                            self.assertEqual(len(children), len(previous) + 1)
                            self.assertFalse(vision.alive())
                            self.assertFalse(Path('/proc', children[-1]).exists())


if __name__ == '__main__':
    unittest.main()
