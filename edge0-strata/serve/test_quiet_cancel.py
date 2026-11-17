"""A disconnect reaches the real engine pipe during a quiet prompt read, before its first PP or token.

The native stand-in deliberately says nothing until STOP. This covers the boundary that a MockEngine watching
the cancel Event itself cannot test: HTTP disconnect -> Event -> STOP -> DONE/BADM -> the next request.
No GPU or model is used.
"""
import json
import io
from pathlib import Path
import queue
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.request

from serve import server


FAKE = r'''import json, queue, sys, threading, time
from pathlib import Path
log = Path(sys.argv[sys.argv.index('--events') + 1])
batch = int(sys.argv[sys.argv.index('--batch') + 1])
commands, stop = queue.Queue(), threading.Event()
def note(event):
    with log.open('a') as f:
        f.write(json.dumps({'event': event, 'at': time.monotonic()}) + '\n')
def reader():
    for line in sys.stdin:
        line = line.strip()
        if line == 'STOP':
            note('stop')
            stop.set()
        else:
            commands.put(line)
    commands.put('QUIT')
threading.Thread(target=reader, daemon=True).start()
if batch:
    print(f'INFO batch_slots={batch}', flush=True)
print('READY 4096 stop', flush=True)
count = 0
while True:
    line = commands.get()
    if line == 'QUIT':
        break
    if not line.startswith(('GEN ', 'BGEN ')):
        continue
    count += 1
    stop.clear()
    note('request_' + str(count))
    fields = line.split()
    slot = int(fields[1]) if fields[0] == 'BGEN' else None
    if count == 1:
        # The test's cleanup also sends STOP, so even a failed assertion leaves no long-lived child.
        stop.wait(15)
        time.sleep(.15)  # STOP was received, but the current chunk has not finished yet: it must drain.
        print('DONE 0 128 1 0 cancel 0 0 0', flush=True)
    else:
        print('T 90\nDONE 1 20 1 1 stop', flush=True)
    if slot is not None:
        print(f'BADM {slot} 0', flush=True)
'''


class QuietPromptCancel(unittest.TestCase):
    def check_hangup(self, api, stream, batch, max_new=1):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, events = root / 'engine.py', root / 'events.jsonl'
            script.write_text(FAKE, encoding='utf-8')
            real_popen = server.subprocess.Popen
            with mock.patch.object(server.subprocess, 'Popen', lambda cmd, **kw:
                                   real_popen([sys.executable, str(script), *cmd[1:]], **kw)):
                engine = server.StrataEngine('strata', ['--batch', str(batch), '--events', str(events)])
            service = server.Service(engine, server.ByteTokenizer(),
                                     server.ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
            http = server.serve(service, port=0)
            port = http.server_address[1]
            sock = None
            reply, errors = [], []
            follower = None
            try:
                body = {'model': 'm', 'stream': stream, 'max_tokens': max_new,
                        'messages': [{'role': 'user', 'content': 'a long prompt'}],
                        'reasoning_effort': 'none'}
                if api == 'responses':
                    body = {'model': 'm', 'stream': stream, 'max_output_tokens': max_new,
                            'input': 'a long prompt', 'reasoning': {'effort': 'none'}}
                data = json.dumps(body).encode()
                path = {'openai': '/v1/chat/completions', 'anthropic': '/v1/messages',
                        'responses': '/v1/responses'}[api]
                sock = socket.create_connection(('127.0.0.1', port), timeout=5)
                sock.sendall(f'POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n'
                             f'Content-Length: {len(data)}\r\n\r\n'.encode() + data)

                def has(event):
                    return events.exists() and any(json.loads(row)['event'] == event
                                                   for row in events.read_text().splitlines())

                deadline = time.monotonic() + 3
                while not has('request_1') and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(has('request_1'), 'the first request did not reach the native pipe')
                closed = time.monotonic()
                sock.close()
                sock = None

                def follow():
                    data = json.dumps({'messages': [{'role': 'user', 'content': 'hi'}],
                                       'max_tokens': 1, 'reasoning_effort': 'none'}).encode()
                    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=data,
                                                 headers={'Content-Type': 'application/json'})
                    try:
                        with urllib.request.urlopen(req, timeout=15) as response:
                            reply.append(json.load(response))
                    except Exception as exc:
                        errors.append(exc)

                follower = threading.Thread(target=follow)
                follower.start()
                deadline = closed + 2.5
                while not has('stop') and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(has('stop'), 'disconnect must send STOP before a PP line or 10 s heartbeat')
                follower.join(timeout=2.5)
                self.assertFalse(follower.is_alive(), 'the cancelled request did not release the next request')
                self.assertEqual(errors, [])
                self.assertEqual(reply[0]['choices'][0]['message']['content'], 'Z')
                self.assertTrue(engine.alive(), 'cancellation must preserve the resident engine')
                self.assertEqual(reply[0]['usage']['completion_tokens'], 1)
                self.assertEqual(service.history[-1]['engine_generated'], 1,
                                 'the next request must record its own DONE, not the cancelled request')
                if batch:
                    self.assertFalse(any(engine.slot_busy), 'the cancelled admission must release its slot')
                else:
                    self.assertEqual(engine.last['finish'], 'stop', 'the next request must read its own DONE')
            finally:
                if sock is not None:
                    sock.close()
                engine._send('STOP')
                if follower is not None:
                    follower.join(timeout=3)
                http.shutdown()
                http.server_close()
                engine.close()

    def test_anthropic_stream(self):
        self.check_hangup('anthropic', True, 0)

    def test_batch_solo_stream(self):
        # A batch-enabled engine runs an isolated request with more than one output token through GEN first.
        self.check_hangup('anthropic', True, 2, max_new=2)

    def test_other_http_modes(self):
        for api in ('openai', 'anthropic', 'responses'):
            for stream in (False, True):
                for batch in (0, 2):
                    if (api, stream, batch) == ('anthropic', True, 0):
                        continue
                    with self.subTest(api=api, stream=stream, batch=batch):
                        self.check_hangup(api, stream, batch)


class QuietWaitPolicy(unittest.TestCase):
    def test_expired_silence_is_not_a_heartbeat(self):
        engine = object.__new__(server.StrataEngine)
        engine.proc = mock.Mock()
        engine.proc.poll.return_value = None
        engine.proc.stdin = io.StringIO()
        engine.lines = mock.Mock()
        engine.can_stop, engine.silence_s, engine.log_path = True, 10, None
        now = [0.0]

        def clock():
            now[0] += .001                 # time also passes between calculating a wait and starting it
            return now[0]

        def get(timeout):
            now[0] += timeout
            raise queue.Empty

        engine.lines.get.side_effect = get
        with mock.patch.object(server.time, 'monotonic', clock):
            gen = engine.generate([], 10, {}, threading.Event())
            with self.assertRaises(server.EngineSilent):
                next(gen)                  # must fail before yielding an overdue heartbeat to a slow consumer
        engine.proc.kill.assert_called_once()
        self.assertNotIn('STOP', engine.proc.stdin.getvalue())

    def test_polling_keeps_ten_second_heartbeats(self):
        for control in (False, True):
            with self.subTest(control=control):
                engine = object.__new__(server.StrataEngine)
                engine.proc = mock.Mock()
                engine.proc.poll.return_value = None
                engine.proc.stdin = io.StringIO()
                engine.lines = mock.Mock()
                engine.can_stop, engine.silence_s = True, 0
                engine._ctl_mode = 'solo'
                engine._send = mock.Mock()
                now = [0.0]
                waits = []

                def get(timeout):
                    waits.append(timeout)
                    if now[0] >= 21.0:
                        return 'DONE 0 1 1 0 length'
                    now[0] += timeout
                    raise queue.Empty

                engine.lines.get.side_effect = get
                cancel = threading.Event()
                with mock.patch.object(server.time, 'monotonic', lambda: now[0]):
                    gen = engine._control(cancel, lambda t: None) if control else engine.generate([1], 1, {}, cancel)
                    heartbeats = list(gen)
                self.assertEqual(heartbeats, [None, None], 'a quiet 21 seconds emits two heartbeats, not 42')
                self.assertTrue(all(0 < wait <= .5 for wait in waits), 'cancellation uses bounded blocking waits')
                engine._send.assert_not_called()
                self.assertNotIn('STOP', engine.proc.stdin.getvalue())


if __name__ == '__main__':
    unittest.main()
