"""A stand-in for edge0-strata's serve/server.py (standard library only, no GPU).

It accepts the server's command line (--engine strata --config <json> --host --port
--ready-file), writes the ready file once it listens, and answers /health,
/v1/status, /v1/models and /v1/chat/completions (streaming and not, with
`timings`); /v1/responses and /v1/messages answer in their own shapes (the
Anthropic usage with cache fields) and echo the request. Like the real
engine it starts a child process (the "engine") in its own process group.

Behaviour is controlled with environment variables:
  FAKE_STRATA_RECORD       directory for records: config.json, process.json,
                           requests.jsonl, sent-<n>.bin, events.jsonl
  FAKE_STRATA_FAIL_START   exit with this code before becoming ready
  FAKE_STRATA_READY_DELAY  seconds to wait before listening
  FAKE_STRATA_EXIT_AFTER   exit after this many generation requests (code 3)
  FAKE_STRATA_CHUNK_DELAY  seconds between streamed chunks (default 0.02); every
                           stream sends the answer one word per text event
  FAKE_STRATA_BODY_DELAY   seconds to wait before a non-streaming answer (default 0)
  FAKE_STRATA_METRICS_STATUS  HTTP status of GET /metrics (default 200)
  FAKE_STRATA_DIE_ON_REQUEST  exit (code 4) on the first POST, after recording it and before any response
  FAKE_STRATA_TOP_OBJECT   JSON value put as the top-level "object" of every
                           /v1/responses and /v1/messages body and event (e.g. null)
  FAKE_STRATA_ANSWER       answer text (default below); FAKE_STRATA_REASONING
                           streams reasoning first unless thinking is off
Run with --self-test to check the fake itself.
"""
import argparse
import faulthandler
import http.client
import http.server
import json
import os
from pathlib import Path
import signal
import socketserver
import subprocess
import sys
import tempfile
import threading
import time

ANSWER = 'Hello from fake Strata, answering in several pieces.'

# Stand-in for edge0-strata's tools/mtp_gguf.py. The delegate launcher runs it with the engine Python
# when a Metal template's prepare output is missing. FAKE_MTP_MODE selects a failure:
#   fail   exit 1 with an error on stderr and no output file
#   exit0  exit 0 with no output file
#   sleep  record the call, then sleep (Ctrl-C must kill this process)
MTP_SCRIPT = '''"""Fake MTP conversion used by delegate-mode tests."""
import json
import os
import sys
import time
from pathlib import Path


def main():
    args = sys.argv[1:]
    out = None
    if "--out" in args:
        out = Path(args[args.index("--out") + 1])
    if out is not None:
        log = out.parent / "prepare-calls.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "argv": args,
                "cwd": os.getcwd(),
                "api_key": os.environ.get("STRATA_API_KEY"),
                "pid": os.getpid(),
            }) + "\\n")
    mode = os.environ.get("FAKE_MTP_MODE", "")
    if mode == "fail":
        sys.stderr.write("mtp conversion failed\\n")
        return 1
    if mode == "exit0":
        return 0
    if mode == "sleep":
        time.sleep(30)
        return 0
    if out is None:
        sys.stderr.write("missing --out\\n")
        return 2
    out.write_bytes(b"mtp-q2_0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


class Fake:
    def __init__(self, config, record):
        self.config = config
        self.record = Path(record) if record else None
        self.lock = threading.Lock()
        self.generations = 0
        self.sent = 0
        args = config.get('args', [])
        self.context = int(args[args.index('--max-context') + 1]) if '--max-context' in args else 4096
        self.exit_after = int(os.environ.get('FAKE_STRATA_EXIT_AFTER', '0') or 0)
        self.delay = float(os.environ.get('FAKE_STRATA_CHUNK_DELAY', '0.02'))
        self.body_delay = float(os.environ.get('FAKE_STRATA_BODY_DELAY', '0') or 0)
        self.answer = os.environ.get('FAKE_STRATA_ANSWER', ANSWER)
        self.reasoning = os.environ.get('FAKE_STRATA_REASONING', '')
        self.child = None

    def log(self, name, value):
        if self.record:
            with self.lock, open(self.record / name, 'a', encoding='utf-8') as out:
                out.write(json.dumps(value) + '\n')

    def save_sent(self, data):
        with self.lock:
            self.sent += 1
            index = self.sent
        if self.record:
            partial = self.record / f'sent-{index}.part'
            partial.write_bytes(data)
            partial.replace(self.record / f'sent-{index}.bin')

    def timings(self, prompt_tokens, generated):
        return {'cache_n': 2, 'prompt_n': prompt_tokens - 2, 'prompt_ms': 120.0, 'prompt_per_token_ms': 12.0,
                'prompt_per_second': round((prompt_tokens - 2) / 0.12, 1), 'predicted_n': generated,
                'predicted_ms': 400.0, 'predicted_per_token_ms': 400.0 / generated,
                'predicted_per_second': round(generated / 0.4, 1), 'draft_n': 4, 'draft_n_accepted': 3}

    def finished_generation(self):
        with self.lock:
            self.generations += 1
            count = self.generations
        if self.exit_after and count >= self.exit_after:
            def leave():
                time.sleep(0.2)
                self.stop_child()
                self.log('events.jsonl', {'event': 'exit_after', 'requests': count})
                print(f'fake strata: leaving after {count} requests', flush=True)
                os._exit(3)
            threading.Thread(target=leave, daemon=True).start()

    def stop_child(self):
        if self.child and self.child.poll() is None:
            self.child.kill()
            self.child.wait()


def prompt_tokens(request):
    return len(json.dumps(request.get('messages', request.get('input', ''))).split()) + 4


def dialect_body(path, request, answer):
    """A non-streaming /v1/responses or /v1/messages body shaped like edge0-strata's, plus the echoed request."""
    prompt, generated = prompt_tokens(request), len(answer.split(' '))
    if path == '/v1/responses':
        body = {'id': 'resp_fake', 'object': 'response', 'status': 'completed', 'model': request.get('model'),
                'output': [{'type': 'message', 'status': 'completed', 'content': [{'type': 'output_text', 'text': answer}]}],
                'usage': {'input_tokens': prompt, 'output_tokens': generated}}
    else:
        # Anthropic semantics: input_tokens counts only the uncached part of the input.
        body = {'id': 'msg_fake', 'type': 'message', 'role': 'assistant', 'model': request.get('model'),
                'content': [{'type': 'text', 'text': answer}], 'stop_reason': 'end_turn', 'stop_sequence': None,
                'usage': {'input_tokens': 1, 'output_tokens': generated, 'cache_creation_input_tokens': prompt - 3,
                          'cache_read_input_tokens': 2}}
    body.update({'echo': request, 'path': path})
    return with_top_object(body)


def with_top_object(value):
    if 'FAKE_STRATA_TOP_OBJECT' in os.environ:
        value = dict(value, object=json.loads(os.environ['FAKE_STRATA_TOP_OBJECT']))
    return value


def pieces_of(answer):
    """The answer one word per streamed text event, each but the last with its trailing space."""
    words = answer.split(' ')
    return [w + (' ' if i + 1 < len(words) else '') for i, w in enumerate(words)]


def dialect_events(path, request, answer):
    """The streamed form: Responses events, or Anthropic message_start ... message_delta ... message_stop."""
    prompt, generated = prompt_tokens(request), len(answer.split(' '))
    if path == '/v1/responses':
        response = {'id': 'resp_fake', 'object': 'response', 'model': request.get('model'), 'usage': None}
        values = [('response.created', {'response': dict(response, status='in_progress'), 'echo': request, 'path': path}),
                  *[('response.output_text.delta', {'delta': piece}) for piece in pieces_of(answer)],
                  ('response.completed', {'response': dict(response, status='completed',
                                                           usage={'input_tokens': prompt, 'output_tokens': generated})})]
    else:
        message = {'id': 'msg_fake', 'type': 'message', 'role': 'assistant', 'model': request.get('model'), 'content': [],
                   'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': prompt, 'output_tokens': 0}}
        values = [('message_start', {'message': message, 'echo': request, 'path': path}),
                  ('content_block_start', {'index': 0, 'content_block': {'type': 'text', 'text': ''}}),
                  *[('content_block_delta', {'index': 0, 'delta': {'type': 'text_delta', 'text': piece}}) for piece in pieces_of(answer)],
                  ('content_block_stop', {'index': 0}),
                  # Its input counts deliberately differ from message_start's, which is the one that counts.
                  ('message_delta', {'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
                                     'usage': {'input_tokens': 1, 'cache_creation_input_tokens': 5, 'cache_read_input_tokens': 0,
                                               'output_tokens': generated}}),
                  ('message_stop', {})]
    return [('event: %s\ndata: %s\n\n' % (kind, json.dumps(with_top_object(dict(value, type=kind,
            **({'sequence_number': 3 + i * 7} if path == '/v1/responses' else {})))))).encode()
            for i, (kind, value) in enumerate(values)]


class Server(http.server.ThreadingHTTPServer):
    """http.server's server without its reverse lookup of the bound address (socket.getfqdn): nothing here uses
    the host name, and on some machines (GitHub's macOS runners) that lookup takes tens of seconds."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def handler_for(fake):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def send_json(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return data

        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/health':
                self.send_json(200, {'status': 'ok', 'service': 'strata', 'loaded': True, 'model': fake.config.get('model_name')})
            elif path in ('/v1/status', '/metrics', '/props'):
                # The description endpoints a benchmark reads; what was sent is recorded for byte comparisons.
                if path == '/v1/status':
                    status, kind = 200, 'application/json'
                    value = {'service': 'strata', 'model': fake.config.get('model_name'), 'engine': 'fake-strata 0.0',
                             'context': {'native': fake.context, 'max_positions': fake.context},
                             'machine': {'gpu': {'name': 'Fake GPU', 'used_mib': 1000, 'total_mib': 24000},
                                         'ram': {'used_gib': 10.0, 'total_gib': 64.0}}}
                elif path == '/metrics':
                    status = int(os.environ.get('FAKE_STRATA_METRICS_STATUS', '200') or 200)
                    kind = 'application/json; charset=utf-8'
                    value = {'engine': {'gpu': 'Fake GPU \u00e9'}, 'history': [], 'totals': {'requests': fake.generations}}
                else:
                    status, kind = 200, 'application/json'
                    value = {'default_generation_settings': {'n_ctx': fake.context}, 'model_alias': fake.config.get('model_name')}
                data = json.dumps(value, ensure_ascii=False).encode()
                fake.log('get' + path.replace('/', '-') + '.json', {'status': status, 'content_type': kind, 'body': data.decode()})
                self.send_response(status)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path == '/v1/models':
                self.send_json(200, {'object': 'list', 'data': [{'id': fake.config.get('model_name'), 'object': 'model'}]})
            else:
                self.send_json(404, {'error': {'message': 'not found'}})

        def do_POST(self):
            path = self.path.split('?')[0]
            body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
            fake.log('requests.jsonl', {'path': path, 'target': self.path, 'body': body.decode('utf-8', 'replace'),
                                        'authorization': self.headers.get('Authorization'),
                                        'expect': self.headers.get('Expect'),
                                        'content_type': self.headers.get('Content-Type')})
            if os.environ.get('FAKE_STRATA_DIE_ON_REQUEST'):
                fake.stop_child()
                fake.log('events.jsonl', {'event': 'die_on_request'})
                os._exit(4)
            try:
                request = json.loads(body)
            except ValueError:
                self.send_json(400, {'error': {'message': 'invalid JSON', 'type': 'invalid_request_error'}})
                return
            if path == '/v1/messages/count_tokens':
                fake.save_sent(self.send_json(int(request.get('fake_status', 200)), {'input_tokens': 17}))
                return
            if request.get('fake_error'):
                if not request.get('stream'):
                    fake.save_sent(self.send_json(429, {'error': {'message': 'fixture busy', 'code': 'fixture_busy'}}))
                elif path == '/v1/chat/completions':
                    self.stream([b'data: {"error":{"message":"fixture failure","code":"fixture_error"}}\n\n'])
                else:
                    kind = 'response.failed' if path == '/v1/responses' else 'error'
                    value = {'type': kind, 'error': {'type': 'api_error', 'message': 'fixture failure'}}
                    if path == '/v1/responses':
                        value = {'type': kind, 'response': {'id': 'resp_fake', 'status': 'failed', 'error': {'code': 'fixture_error'}}}
                    self.stream([('event: ' + kind + '\ndata: ' + json.dumps(value) + '\n\n').encode()])
                return
            if request.get('fake_failure'):
                mode = request['fake_failure']
                streaming = request.get('stream', False)
                if streaming:
                    data = (b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                            if path == '/v1/chat/completions' else b''.join(dialect_events(path, request, fake.answer)[:3]))
                else:
                    data = b'{"partial":'
                if mode == 'before_event':
                    data = b''
                if mode == 'short_length':
                    data = b'{"valid_but_short":true}'
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream' if streaming else 'application/json')
                self.send_header('Connection', 'close')
                if mode == 'short_length':
                    self.send_header('Content-Length', str(len(data) + 50))
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()
                fake.save_sent(data)
                self.close_connection = True
                if mode == 'exit':
                    fake.stop_child()
                    os._exit(4)
                return
            if isinstance(request.get('max_tokens'), int) and request['max_tokens'] < 1:
                self.send_json(400, {'error': {'message': 'max_tokens must be at least 1', 'type': 'invalid_request_error',
                                               'param': 'max_tokens'}})
                return
            if path in ('/v1/responses', '/v1/messages'):
                if request.get('stream'):
                    self.stream(dialect_events(path, request, fake.answer))
                else:
                    time.sleep(fake.body_delay)
                    fake.save_sent(self.send_json(200, dialect_body(path, request, fake.answer)))
                fake.finished_generation()
                return
            if path != '/v1/chat/completions':
                self.send_json(404, {'error': {'message': 'not found'}})
                return
            prompt = sum(len(str(m.get('content', '')).split()) for m in request.get('messages', [])) + 4
            pieces = pieces_of(fake.answer)
            thinking = (request.get('chat_template_kwargs') or {}).get('enable_thinking', True)
            reasoning = fake.reasoning.split(' ') if fake.reasoning and thinking is not False else []
            usage = {'prompt_tokens': prompt, 'completion_tokens': len(pieces) + len(reasoning), 'total_tokens': prompt + len(pieces) + len(reasoning)}
            timings = fake.timings(prompt, len(pieces) + len(reasoning))
            model = request.get('model')
            if request.get('stream'):
                def event(delta, finish=None, extra=None):
                    chunk = {'id': 'chatcmpl-fake', 'object': 'chat.completion.chunk', 'created': 1, 'model': model,
                             'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
                    chunk.update(extra or {})
                    return ('data: ' + json.dumps(chunk) + '\n\n').encode()
                events = [event({'role': 'assistant', 'content': ''})]
                events += [event({'reasoning_content': r + ' '}) for r in reasoning]
                events += [event({'content': p}) for p in pieces]
                events.append(event({}, 'stop', {'usage': usage, 'timings': timings}))
                events.append(b'data: [DONE]\n\n')
                self.stream(events)
            else:
                time.sleep(fake.body_delay)
                fake.save_sent(self.send_json(200, {
                    'id': 'chatcmpl-fake', 'object': 'chat.completion', 'created': 1, 'model': model,
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': ''.join(pieces)}, 'finish_reason': 'stop'}],
                    'usage': usage, 'timings': timings}))
            fake.finished_generation()

        def stream(self, events):
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            sent = b''
            try:
                for data in events:
                    self.wfile.write(b'%x\r\n%s\r\n' % (len(data), data))
                    self.wfile.flush()
                    sent += data
                    time.sleep(fake.delay)
                # Recorded before the terminating chunk, so a client that has read the whole stream finds the record.
                fake.save_sent(sent)
                self.wfile.write(b'0\r\n\r\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                fake.log('events.jsonl', {'event': 'client_closed', 'sent_events': sent.count(b'\n\n')})
                self.close_connection = True

    return Handler


def serve(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8095)
    parser.add_argument('--ready-file')
    parser.add_argument('--gpu')
    if hasattr(signal, 'SIGUSR1'):
        faulthandler.register(signal.SIGUSR1)  # a test that waits too long asks where this process is stuck
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    record = os.environ.get('FAKE_STRATA_RECORD')
    if record:
        Path(record).mkdir(parents=True, exist_ok=True)
        (Path(record) / 'config.json').write_text(json.dumps(config, indent=1))
    fake = Fake(config, record)
    # The child holds none of this process's pipes: if it outlives an abrupt exit, whoever reads this
    # process's output (CTest, a test) still sees the end of it instead of waiting for the child.
    fake.child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3600)'],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fake.log('process.json', {'pid': os.getpid(), 'pgid': os.getpgid(0) if os.name != 'nt' else None,
                              'child': fake.child.pid, 'strata_api_key_present': 'STRATA_API_KEY' in os.environ, 'argv': argv, 'path': os.environ.get('PATH', ''), 'python': sys.executable})
    if os.environ.get('FAKE_STRATA_FAIL_START'):
        print('fake strata: failing on purpose', flush=True)
        fake.stop_child()
        return int(os.environ['FAKE_STRATA_FAIL_START'])
    time.sleep(float(os.environ.get('FAKE_STRATA_READY_DELAY', '0') or 0))
    server = Server((args.host, args.port), handler_for(fake))
    server.daemon_threads = True

    def terminate(*_):
        fake.log('events.jsonl', {'event': 'sigbreak' if os.name == 'nt' and _ and _[0] == signal.SIGBREAK else 'sigterm'})
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    if os.name == 'nt':
        signal.signal(signal.SIGBREAK, terminate)
    if args.ready_file:
        port = server.server_address[1]
        descriptor = os.open(args.ready_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as out:
            out.write(json.dumps({'schema_version': 1, 'pid': os.getpid(), 'api_url': f'http://127.0.0.1:{port}',
                                  'bound_host': args.host, 'bound_port': port, 'model_state': 'ready'}) + '\n')
    print(f'ready: http://127.0.0.1:{server.server_address[1]}/v1', flush=True)
    try:
        server.serve_forever()
    finally:
        fake.stop_child()
        if args.ready_file:
            Path(args.ready_file).unlink(missing_ok=True)
    return 0


def stuck(process):
    """A fake that is alive but not ready: have it print its threads' stacks (POSIX), then fail."""
    if hasattr(signal, 'SIGUSR1'):
        process.send_signal(signal.SIGUSR1)
        time.sleep(1)
    raise AssertionError('fake not ready after 30 s (its stacks are printed above on POSIX)')


def self_test():
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        config = root / 'config.json'
        config.write_text(json.dumps({'model_name': 'm', 'args': ['--max-context', '1234']}))
        ready = root / 'ready.json'
        env = dict(os.environ, FAKE_STRATA_RECORD=str(root / 'record'), FAKE_STRATA_REASONING='think hard',
                   FAKE_STRATA_CHUNK_DELAY='0')
        process = subprocess.Popen([sys.executable, __file__, '--engine', 'strata', '--config', str(config),
                                    '--port', '0', '--ready-file', str(ready)], env=env,
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        try:
            deadline = time.monotonic() + 30
            while not (ready.exists() and ready.read_text().endswith('\n')):
                assert process.poll() is None, f'fake exited before it was ready (code {process.returncode})'
                if time.monotonic() >= deadline:
                    stuck(process)
                time.sleep(0.05)
            port = json.loads(ready.read_text())['bound_port']

            def call(method, path, body=None):
                connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
                connection.request(method, path, None if body is None else json.dumps(body),
                                   {'Content-Type': 'application/json'})
                response = connection.getresponse()
                data = response.read()
                connection.close()
                return response.status, response.getheader('Content-Type'), data
            assert call('GET', '/health')[0] == 200
            status = json.loads(call('GET', '/v1/status')[2])
            assert status['context']['native'] == 1234 and status['machine']['gpu']['total_mib'] == 24000
            code, kind, data = call('POST', '/v1/chat/completions', {'model': 'm', 'messages': [{'role': 'user', 'content': 'hi'}]})
            value = json.loads(data)
            assert code == 200 and value['choices'][0]['message']['content'] == ANSWER and value['timings']['predicted_n'] > 0
            code, kind, data = call('POST', '/v1/chat/completions', {'model': 'm', 'stream': True,
                                                                    'messages': [{'role': 'user', 'content': 'hi'}]})
            assert code == 200 and kind == 'text/event-stream' and data.endswith(b'data: [DONE]\n\n')
            events = [json.loads(line[6:]) for line in data.decode().splitlines() if line.startswith('data: {')]
            assert ''.join(e['choices'][0]['delta'].get('content', '') for e in events) == ANSWER
            assert ''.join(e['choices'][0]['delta'].get('reasoning_content', '') for e in events) == 'think hard '
            assert 'timings' in events[-1] and events[-1]['choices'][0]['finish_reason'] == 'stop'
            assert (root / 'record' / 'sent-2.bin').read_bytes() == data
            code, _, data = call('POST', '/v1/chat/completions', {'model': 'm', 'stream': True, 'chat_template_kwargs': {'enable_thinking': False},
                                                                 'messages': [{'role': 'user', 'content': 'hi'}]})
            assert b'reasoning_content' not in data
            for path in ('/v1/responses', '/v1/messages'):
                code, _, data = call('POST', path, {'x': 1})
                value = json.loads(data)
                assert code == 200 and value['echo'] == {'x': 1} and value['path'] == path
            # The other two stream formats also send the answer in several text events.
            for path, kind, text in (('/v1/responses', 'response.output_text.delta', lambda e: e['delta']),
                                     ('/v1/messages', 'content_block_delta', lambda e: e['delta']['text'])):
                code, _, data = call('POST', path, {'stream': True, 'x': 1})
                deltas = [text(e) for e in (json.loads(line[6:]) for line in data.decode().splitlines() if line.startswith('data: {'))
                          if e['type'] == kind]
                assert code == 200 and len(deltas) == len(ANSWER.split(' ')) and ''.join(deltas) == ANSWER, (path, deltas)
            requests = [json.loads(line) for line in (root / 'record' / 'requests.jsonl').read_text().splitlines()]
            assert [r['path'] for r in requests] == ['/v1/chat/completions'] * 3 + ['/v1/responses', '/v1/messages'] * 2
            assert call('POST', '/v1/messages/count_tokens', {'x': 7})[2] == b'{"input_tokens": 17}'
            assert json.loads((root / 'record' / 'requests.jsonl').read_text().splitlines()[-1])['body'] == '{"x": 7}'
            for stream in (False, True):
                code, kind, data = call('POST', '/v1/chat/completions', {'stream': stream, 'fake_failure': 'disconnect'})
                assert code == 200 and b'partial' in data and b'[DONE]' not in data
                assert process.poll() is None
            assert call('POST', '/v1/responses', {'stream': True, 'fake_failure': 'before_event'})[2] == b''
            process_info = json.loads((root / 'record' / 'process.json').read_text())
            assert process_info['strata_api_key_present'] == ('STRATA_API_KEY' in env)
            child = process_info['child']
        finally:
            process.send_signal(signal.CTRL_BREAK_EVENT if os.name == 'nt' else signal.SIGTERM)
            try:
                code = process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(10)
                raise AssertionError('fake did not stop within 10 s of SIGTERM')
            assert code == 0, code
        assert not ready.exists()
        if os.name != 'nt':
            try:
                os.kill(child, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            assert not alive, 'engine child left behind'
        failing = subprocess.run([sys.executable, __file__, '--engine', 'strata', '--config', str(config), '--port', '0',
                                  '--ready-file', str(root / 'ready2.json')], env=dict(env, FAKE_STRATA_FAIL_START='5'),
                                 capture_output=True)
        assert failing.returncode == 5 and not (root / 'ready2.json').exists()
    # fake_failure=exit: the engine sends part of an answer and exits with code 4. What the client reads after
    # that depends on the platform (EOF, or a reset that can even drop the part already sent), so the check
    # uses what the fake itself recorded as sent, and its exit code; the client only has to end its read.
    for stream in (False, True):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text('{}')
            ready = root / 'ready.json'
            process = subprocess.Popen([sys.executable, __file__, '--engine', 'strata', '--config', str(root / 'config.json'),
                                        '--port', '0', '--ready-file', str(ready)],
                                       env=dict(os.environ, FAKE_STRATA_RECORD=str(root / 'record')),
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 30
                while not (ready.exists() and ready.read_text().endswith('\n')):
                    assert process.poll() is None, f'fake exited before it was ready (code {process.returncode})'
                    if time.monotonic() >= deadline:
                        stuck(process)
                    time.sleep(0.05)
                port = json.loads(ready.read_text())['bound_port']
                connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
                try:
                    connection.request('POST', '/v1/chat/completions', json.dumps({'stream': stream, 'fake_failure': 'exit'}))
                    connection.getresponse().read()
                except (OSError, http.client.HTTPException):
                    pass
                finally:
                    connection.close()
                assert process.wait(10) == 4
                sent = (root / 'record' / 'sent-1.bin').read_bytes()
                assert b'partial' in sent and b'[DONE]' not in sent, sent
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(10)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        script = root / 'mtp_gguf.py'
        script.write_text(MTP_SCRIPT)
        out = root / 'model' / 'mtp-q2_0.gguf'

        def run(mode):
            env = dict(os.environ)
            env.pop('STRATA_API_KEY', None)
            if mode:
                env['FAKE_MTP_MODE'] = mode
            else:
                env.pop('FAKE_MTP_MODE', None)
            return subprocess.run([sys.executable, str(script), '--rt', str(root / 'mtp'), '--out', str(out)],
                                  cwd=root, env=env, capture_output=True, text=True)

        ok = run('')
        assert ok.returncode == 0 and out.read_bytes() == b'mtp-q2_0', (ok.returncode, ok.stderr)
        recorded = json.loads((out.parent / 'prepare-calls.jsonl').read_text().splitlines()[0])
        assert Path(recorded['cwd']).resolve() == root.resolve() and recorded['pid'] > 0 and recorded['api_key'] is None
        out.unlink()
        failed = run('fail')
        assert failed.returncode == 1 and 'mtp conversion failed' in failed.stderr and not out.exists()
        silent = run('exit0')
        assert silent.returncode == 0 and not out.exists() and silent.stderr == ''
    print('PASS fake strata: command line, ready file, status, streaming with timings, echo, records, failure, child cleanup')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        self_test()
    else:
        sys.exit(serve(sys.argv[1:]))
