"""serve, inspect and the other subcommands for a delegated model, against the fake edge0-strata."""
import argparse
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strata_support import (ALIAS, MODEL_JSON, Running, assert_no_leftovers, environment, gone, make_engine, make_model,  # noqa: E402
                            process_record, strata_args, temp_dir)

ROOT = Path(__file__).resolve().parents[2]


def request(port, method, path, body=None, headers=None, host='127.0.0.1'):
    connection = http.client.HTTPConnection(host, port, timeout=30)
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    connection.request(method, path, data, {'Content-Type': 'application/json', **(headers or {})})
    response = connection.getresponse()
    payload = response.read()
    kind = response.getheader('Content-Type')
    connection.close()
    return response.status, kind, payload


def requests_seen(record):
    path = Path(record) / 'requests.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def sent(record, index, timeout=10):
    """What the fake sent for its index-th response; it records a stream after the last chunk is written."""
    path = Path(record) / f'sent-{index}.bin'
    deadline = time.time() + timeout
    while not path.exists() and time.time() < deadline:
        time.sleep(0.02)
    return path.read_bytes()


def events(record):
    path = Path(record) / 'events.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def status_table():
    document = (ROOT / 'docs/local-server.md').read_text()
    table = document.split('<!-- status-fields-begin -->')[1].split('<!-- status-fields-end -->')[0]
    fields = {}
    for line in table.splitlines():
        match = re.match(r'\| `([^`]+)` \| ([^|]+) \| (required|optional) \| (portal|mapped|recorded|n/a) \|', line)
        if match:
            fields[match[1]] = (set(match[2].strip().split(' or ')), match[3], match[4])
    assert len(fields) > 150, 'status table lost its Delegated column'
    return fields


def check_status_schema(snapshots):
    """Every field documented; types as documented; n/a fields null or absent; mapped/recorded may be null."""
    fields = status_table()
    seen = set()

    def kind(value):
        if value is None: return 'null'
        if isinstance(value, bool): return 'boolean'
        if isinstance(value, int): return 'integer'
        if isinstance(value, float): return 'number'
        if isinstance(value, str): return 'string'
        if isinstance(value, list): return 'array'
        return 'object'

    def walk(value, path=''):
        if path:
            assert path in fields, ('undocumented field', path)
            types, _, source = fields[path]
            actual = kind(value)
            if source == 'n/a':
                assert actual == 'null' and fields[path][1] == 'required', ('n/a field present with a value', path, value)
            elif actual == 'null':
                assert 'null' in types or source in ('mapped', 'recorded'), ('null not allowed', path)
            else:
                assert actual in types or actual == 'integer' and 'number' in types, (path, actual, types)
            seen.add(path)
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, path + '.' + key if path else key)
            prefix = path + '.' if path else ''
            for field, (_, presence, source) in fields.items():
                if presence != 'required' or not field.startswith(prefix): continue
                suffix = field[len(prefix):]
                if '.' not in suffix and '[' not in suffix and suffix != '*':
                    assert suffix in value, ('missing field', field)
        elif isinstance(value, list):
            for child in value:
                walk(child, path + '[]')
    for snapshot in snapshots:
        walk(snapshot)
    # Exactly the documented names: every field that is not n/a (and not under an n/a field) was produced.
    def applies(field):
        parts = field.replace('[]', '').split('.')
        return all(fields.get('.'.join(parts[:i]), (None, None, 'portal'))[2] != 'n/a' for i in range(1, len(parts) + 1)) \
            and fields[field][2] != 'n/a'
    expected = {field for field in fields if applies(field)}
    assert expected <= seen, ('documented fields not exercised', sorted(expected - seen))


def strata_options():
    # The delegate options as the argument parser accepts them, so a new one cannot be left out of the help.
    source = (Path(__file__).resolve().parents[2] / 'tools' / 'cli' / 'serve_delegate_registry.cpp').read_text()
    options = sorted(set(re.findall(r'key == "(--strata-[a-z-]+)"', source)))
    assert options, 'no --strata-* options found in the parser'
    return options


def refusals(cli, root):
    for command in ('serve', 'chat'):
        shown = subprocess.run([str(cli), command, '--help'], capture_output=True, text=True)
        assert shown.returncode == 0, shown.stderr
        missing = [o for o in strata_options() if o not in shown.stdout]
        assert not missing, f'{command} --help lacks {missing}'
    engine = make_engine(root / 'r-engine')
    model = make_model(root / 'r-complete')
    inspected = subprocess.run([str(cli), 'inspect', '--model', str(model), '--json'], capture_output=True, text=True)
    assert inspected.returncode == 0, inspected.stderr
    report = json.loads(inspected.stdout)
    assert report['adapter'] == 'edge0_145b' and report['ok'] is True and report['delegated'] is True
    from strata_support import FILES
    assert {f['filename'] for f in report['files']} == {*FILES, 'model.json', 'mtp/', 'tokenizer/', 'scales/'}
    assert all(f['exists'] and f['bytes'] > 0 for f in report['files'])
    assert sum(f['bytes'] for f in report['files']) == sum(p.stat().st_size for p in model.rglob('*') if p.is_file())
    partial = make_model(root / 'r-partial', skip=('labels.bin', 'tokenizer'))
    inspected = subprocess.run([str(cli), 'inspect', '--model', str(partial), '--json'], capture_output=True, text=True)
    report = json.loads(inspected.stdout)
    assert inspected.returncode == 1 and report['ok'] is False
    assert {f['filename'] for f in report['files'] if not f['exists']} == {'labels.bin', 'tokenizer/'}
    for command in ('run', 'bench', 'forward', 'describe', 'tokenize', 'prompt'):
        refused = subprocess.run([str(cli), command, '--model', str(model)], capture_output=True, text=True)
        assert refused.returncode == 2 and 'edge0-infer serve' in refused.stderr and 'edge0-infer chat' in refused.stderr, (command, refused.stderr)
    record = root / 'r-record'
    started = subprocess.run([str(cli), 'serve', '--model', str(partial), *strata_args(engine)], capture_output=True, text=True,
                             env=environment(record), timeout=60)
    assert started.returncode == 2 and 'missing: labels.bin, tokenizer/' in started.stderr, started.stderr
    assert not record.exists(), 'a process was started for an incomplete directory'
    # model.json alone makes the directory delegated: inspect lists every other file as missing, serve starts nothing.
    bare = root / 'r-bare'
    bare.mkdir()
    (bare / 'model.json').write_text(MODEL_JSON)
    inspected = subprocess.run([str(cli), 'inspect', '--model', str(bare), '--json'], capture_output=True, text=True)
    report = json.loads(inspected.stdout)
    assert inspected.returncode == 1 and report['delegated'] is True and report['ok'] is False
    assert [f['filename'] for f in report['files'] if f['exists']] == ['model.json']
    # Other architectures, and directories without model.json, go to the native library exactly as before.
    for name, text, category, message in (
            ('native', '{"architecture": "edge0_35b", "format_version": 1}', 'asset_invalid', 'assets.tsv'),
            ('foreign', '{"architecture": "edge0_8b", "format_version": 1}', 'unsupported', 'not compiled into this library'),
            ('old', '{"architecture":"edge0_145b","format_version":1,"model_name":"edge0-145b-q3k"}',
             'unsupported', 'not compiled into this library'),
            ('none', None, 'asset_invalid', 'model.json: cannot read')):
        directory = make_model(root / ('r-' + name))
        if text is None:
            (directory / 'model.json').unlink()
        else:
            (directory / 'model.json').write_text(text)
        native = subprocess.run([str(cli), 'inspect', '--model', str(directory), '--json'], capture_output=True, text=True)
        assert native.returncode != 0 and 'delegated' not in native.stdout, (name, native.stdout)
        assert category in native.stderr and message in native.stderr, (name, native.stderr)
        record = root / ('r-record-' + name)
        served = subprocess.run([str(cli), 'serve', '--model', str(directory), '--port', '0', *strata_args(engine)],
                                capture_output=True, text=True, env=environment(record), timeout=60)
        assert served.returncode != 0 and not record.exists(), (name, served.stderr)
    print('PASS refusals: inspect --json, partial directories, other subcommands')


def failures(cli, root):
    engine = make_engine(root / 'f-engine')
    model = make_model(root / 'f-model')

    def run(name, extra=(), engine_dir=engine, **fake):
        record = root / ('f-' + name)
        result = subprocess.run([str(cli), 'serve', '--model', str(model), '--port', '0', *strata_args(engine_dir), *extra],
                                capture_output=True, text=True, env=environment(record, **fake), timeout=90)
        assert result.returncode != 0 and 'Listening:' not in result.stderr, (name, result.stderr)
        return record, result.stderr
    record, text = run('missing-dir', engine_dir=root / 'nowhere')
    assert 'edge0-strata not found' in text and not record.exists()
    record, text = run('missing-python', ['--strata-python', str(root / 'no-python')])
    assert 'Python for edge0-strata not found' in text and not record.exists()
    other = make_engine(root / 'f-engine-2', contract='3')
    record, text = run('contract', engine_dir=other)
    assert 'version 3 is not supported' in text and 'supports version 4' in text and not record.exists()
    record, text = run('start-failure', fail_start=5)
    assert 'exited before it was ready (exit code 5)' in text, text
    assert_no_leftovers(record)
    record, text = run('timeout', ['--strata-ready-timeout', '1'], ready_delay=30)
    assert 'not ready within 1 s' in text, text
    assert_no_leftovers(record)
    # The engine directory can come from EDGE0_STRATA_DIR instead of --strata-dir.
    record = root / 'f-env'
    env = environment(record)
    env['EDGE0_STRATA_DIR'] = str(engine)
    server = subprocess.Popen([str(cli), 'serve', '--model', str(model), '--port', '0', '--strata-python', sys.executable],
                              stderr=subprocess.PIPE, text=True, env=env,
                              creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
    for line in server.stderr:
        if line.startswith('Listening:'): break
    server.send_signal(signal.CTRL_BREAK_EVENT if os.name == 'nt' else signal.SIGTERM)
    assert server.wait(60) == 0
    assert_no_leftovers(record)
    print('PASS failures: engine directory, Python, contract version, start failure, ready timeout; no leftovers')


def proxy(cli, root, snapshots):
    engine = make_engine(root / 'p-engine')
    model = make_model(root / 'p-model')
    record = root / 'p-record'
    server = Running(cli, model, engine, record, env=environment(record, ready_delay=1.5, metrics_status=500))
    try:
        process = process_record(record)
        assert server.listening_at - server.started >= 1.5, 'Listening printed before edge0-strata was ready'
        assert server.alias == ALIAS
        port = server.port
        # /health: Edge0's status with edge0-strata's own fields merged in, so its benchmark recognizes the proxy.
        code, _, body = request(port, 'GET', '/health')
        assert code == 200 and list(json.loads(body).items()) == [('status', 'ok'), ('service', 'strata'), ('loaded', True),
                                                                   ('model', ALIAS)], body
        # edge0-strata's description endpoints pass through: status, content type and body as the fake sent them.
        for path in ('/v1/status', '/metrics', '/props'):
            code, kind, body = request(port, 'GET', path)
            last = json.loads((record / ('get' + path.replace('/', '-') + '.json')).read_text().splitlines()[-1])
            assert (code, kind, body) == (last['status'], last['content_type'], last['body'].encode()), (path, code, kind, body)
        assert request(port, 'GET', '/metrics')[0] == 500 and 'charset=utf-8' in request(port, 'GET', '/metrics')[1]
        for path in ('/v1/models', '/v1/models/edge0_145b'):
            value = json.loads(request(port, 'GET', path)[2])
            item = value['data'][0] if 'data' in value else value
            assert item['id'] == ALIAS and item['context_length'] == 131072
            assert item['capabilities'] == {'thinking': True, 'thinking_default': True}
        assert request(port, 'GET', '/v1/models/other')[0] == 404
        initial = json.loads(request(port, 'GET', '/status')[2])
        snapshots.append(initial)
        assert initial['current'] is None and initial['last'] is None and initial['totals']['requests'] == 0
        assert initial['instance']['pid'] == server.process.pid and initial['listen'] == {'host': '127.0.0.1', 'port': port, 'auth': False}
        assert initial['engine']['version'] == 'fake-strata 0.0' and initial['model']['context_length'] == 131072
        assert initial['engine']['devices'][0]['name'] == 'cuda:0' and initial['engine']['devices'][0]['description'] == 'Fake GPU'
        assert initial['memory'] == {'ledger': None, 'resident': None, 'physical': None}

        # Non-streaming chat: the request and the response bodies arrive byte for byte.
        sent_body = b'{"model": "edge0_145b",  "messages": [{"role": "user", "content": "byte \\u00e9xact"}], "temperature": 0}'
        code, kind, body = request(port, 'POST', '/v1/chat/completions', sent_body, {'Authorization': 'Bearer local-anything'})
        assert code == 200 and kind == 'application/json'
        assert body == sent(record, 1)
        seen = requests_seen(record)
        assert seen[-1]['body'].encode() == sent_body and seen[-1]['authorization'] is None
        # Streaming chat: every byte of the event stream, timings included.
        stream_body = {'model': ALIAS, 'stream': True, 'messages': [{'role': 'user', 'content': 'hello'}]}
        code, kind, body = request(port, 'POST', '/v1/chat/completions', stream_body)
        assert code == 200 and kind == 'text/event-stream'
        assert body == sent(record, 2) and b'"timings"' in body and body.endswith(b'data: [DONE]\n\n')
        assert json.loads(requests_seen(record)[-1]['body']) == stream_body
        # The other two generation endpoints are forwarded as well.
        for index, path in ((3, '/v1/responses'), (4, '/v1/messages')):
            code, _, body = request(port, 'POST', path, {'model': ALIAS, 'input': path})
            value = json.loads(body)
            assert code == 200 and value['echo'] == {'model': ALIAS, 'input': path} and value['path'] == path
            assert body == sent(record, index)
        # A large body goes out at once: no "Expect: 100-continue" (Strata's HTTP/1.0 server never answers it).
        large = {'model': ALIAS, 'messages': [{'role': 'user', 'content': 'word ' * 4000}]}
        code, _, body = request(port, 'POST', '/v1/responses', large)
        assert code == 200 and body == sent(record, 5) and json.loads(body)['echo'] == large
        assert requests_seen(record)[-1]['expect'] is None
        code, _, body = request(port, 'POST', '/v1/chat/completions', b'{not json')
        assert code == 400 and b'invalid JSON' in body
        code, _, body = request(port, 'POST', '/v1/chat/completions', {'model': ALIAS, 'max_tokens': 0, 'messages': []})
        assert code == 400 and json.loads(body)['error']['param'] == 'max_tokens'

        # Streaming in progress: /status shows it; then the client disconnects and the upstream request is closed.
        connection = http.client.HTTPConnection('127.0.0.1', port)
        long_body = {'model': ALIAS, 'stream': True, 'messages': [{'role': 'user', 'content': 'long'}]}
        connection.request('POST', '/v1/chat/completions', json.dumps(long_body), {'Content-Type': 'application/json'})
        response = connection.getresponse()
        assert response.status == 200 and response.readline().startswith(b'data: ')
        live = json.loads(request(port, 'GET', '/status')[2])
        snapshots.append(live)
        assert live['current']['endpoint'] == '/v1/chat/completions' and live['current']['source'] == '127.0.0.1'
        # A stream: prefill until its first text fragment is forwarded, then decode with the fragment count.
        current = live['current']
        assert current['elapsed_seconds'] >= 0 and current['phase'] in ('prefill', 'decode'), current
        if current['phase'] == 'prefill':
            assert current['generated_tokens'] is None and 'first_token_seconds' not in current, current
        else:
            assert current['generated_tokens'] > 0 and current['first_token_seconds'] <= current['elapsed_seconds'], current
        connection.sock.shutdown(socket.SHUT_RDWR)
        connection.close()
        deadline = time.time() + 20
        while time.time() < deadline and not any(e['event'] == 'client_closed' for e in events(record)):
            time.sleep(0.05)
        assert any(e['event'] == 'client_closed' for e in events(record)), 'upstream request was not closed'

        final = json.loads(request(port, 'GET', '/status')[2])
        snapshots.append(final)
        records = final['recent_requests']
        assert [r['endpoint'] for r in records] == ['/v1/chat/completions'] * 3 + ['/v1/responses', '/v1/messages', '/v1/responses'] + \
            ['/v1/chat/completions'] * 2, records
        cancelled, bad_param, invalid, large_record, messages, responses, streamed, plain = records
        assert cancelled['stop_reason'] == 'cancelled' and invalid['http_status'] == 400 and 'stop_reason' not in invalid
        assert invalid['code'] == 'invalid_request_error' and invalid['message'] == 'invalid JSON' and 'param' not in invalid
        assert bad_param['http_status'] == 400 and bad_param['param'] == 'max_tokens' and 'stop_reason' not in bad_param
        plain_sent = json.loads(sent(record, 1))
        assert plain['stream'] is False and plain['http_status'] == 200 and plain['stop_reason'] == 'end_of_generation'
        assert plain['input_tokens'] == plain_sent['usage']['prompt_tokens'] and plain['generated_tokens'] == plain_sent['usage']['completion_tokens']
        assert plain['reused_tokens'] == plain_sent['timings']['cache_n'] and plain['first_token_seconds'] == plain_sent['timings']['prompt_ms'] / 1000
        assert plain['decode_tokens_per_second'] == plain_sent['timings']['predicted_per_second']
        assert plain['decode_seconds'] == plain_sent['timings']['predicted_ms'] / 1000 and 'computed_tokens' not in plain
        assert streamed['stream'] is True and streamed['stop_reason'] == 'end_of_generation' and streamed['input_tokens'] > 0
        totals = final['totals']
        assert [totals[k] for k in ('requests', 'rejected_busy', 'rejected_invalid', 'unauthorized')] == [6, 0, 2, 0]
        entered = [r for r in records if 'stop_reason' in r]
        assert totals['input_tokens'] == sum(r['input_tokens'] or 0 for r in entered)
        assert totals['generated_tokens'] == sum(r['generated_tokens'] or 0 for r in entered)
        assert final['last']['endpoint'] == '/v1/chat/completions' and final['last']['stop_reason'] == 'cancelled'
        assert final['current'] is None
        logs = [json.loads(line) for line in server.text().splitlines() if line.startswith('{') and '"http_status"' in line]
        assert list(reversed(logs)) == records
    finally:
        assert server.stop() == 0
    assert_no_leftovers(record)
    print('PASS proxy: ready line after readiness, byte-exact forwarding of three endpoints and SSE, disconnect, models, status')


def data_events(body):
    return [json.loads(line[5:]) for line in body.decode().splitlines() if line.startswith('data: {')]


def carries_text(event):
    """Whether a streamed event of any of the three formats carries text (the fake sends no tool calls)."""
    if 'choices' in event:
        delta = event['choices'][0].get('delta') or {}
        return bool(delta.get('content') or delta.get('reasoning_content'))
    if event.get('type') == 'response.output_text.delta':
        return bool(event['delta'])
    return event.get('type') == 'content_block_delta' and bool(event['delta'].get('text'))


def live_progress(cli, root, snapshots):
    """/status while a request is in flight: streams count forwarded text fragments; non-streaming is unknown."""
    engine = make_engine(root / 'v-engine')
    model = make_model(root / 'v-model')
    record = root / 'v-record'
    server = Running(cli, model, engine, record, env=environment(record, chunk_delay=0.15, body_delay=1.5))

    def status():
        value = json.loads(request(server.port, 'GET', '/status')[2])
        snapshots.append(value)
        return value

    def idle_status():
        deadline = time.time() + 20
        while time.time() < deadline:
            value = status()
            if value['current'] is None:
                return value
            time.sleep(0.05)
        raise AssertionError('the request stayed in current')
    try:
        content = {'model': ALIAS, 'messages': [{'role': 'user', 'content': 'count these few words'}]}
        index = 0
        for path in ('/v1/chat/completions', '/v1/responses', '/v1/messages'):
            connection = http.client.HTTPConnection('127.0.0.1', server.port, timeout=60)
            connection.request('POST', path, json.dumps(dict(content, stream=True)), {'Content-Type': 'application/json'})
            response = connection.getresponse()
            assert response.status == 200
            received, texts = b'', 0
            # Read until two text events have arrived: the proxy counts a chunk before it forwards it.
            while texts < 2:
                line = response.readline()
                assert line, (path, received)
                received += line
                if line.startswith(b'data: {'):
                    texts += carries_text(json.loads(line[6:]))
            first = status()['current']
            time.sleep(0.4)
            second = status()['current']
            for current in (first, second):
                assert current['endpoint'] == path and current['phase'] == 'decode', (path, current)
                assert current['generated_tokens'] >= 2 and current['first_token_seconds'] <= current['elapsed_seconds'], (path, current)
                assert current['computed_tokens'] is None and current['pending_tokens'] is None, (path, current)
            assert second['generated_tokens'] >= first['generated_tokens'] and second['first_token_seconds'] == first['first_token_seconds']
            if path == '/v1/messages':  # message_start carries the input count at the start
                start = data_events(received)[0]
                assert start['type'] == 'message_start' and first['input_tokens'] == start['message']['usage']['input_tokens']
            else:  # Chat's and Responses' counts come only with the last event
                assert first['input_tokens'] is None and first['reused_tokens'] is None, (path, first)
            body = received + response.read()
            connection.close()
            index += 1
            assert body == sent(record, index), path
            events_sent = data_events(body)
            assert second['generated_tokens'] <= sum(carries_text(e) for e in events_sent)
            # Once it ends: current is null and the record holds the engine's own numbers, not the forwarded count.
            final = idle_status()
            last = final['last']
            assert last['endpoint'] == path and last['stream'] is True and last['stop_reason'] == 'end_of_generation', last
            if path == '/v1/chat/completions':
                usage, timings = events_sent[-1]['usage'], events_sent[-1]['timings']
                assert (last['input_tokens'], last['generated_tokens'], last['reused_tokens']) == \
                    (usage['prompt_tokens'], usage['completion_tokens'], timings['cache_n']), last
                assert last['first_token_seconds'] == timings['prompt_ms'] / 1000 != second['first_token_seconds'], last
                assert last['decode_seconds'] == timings['predicted_ms'] / 1000
                assert last['decode_tokens_per_second'] == timings['predicted_per_second']
            elif path == '/v1/responses':
                usage = events_sent[-1]['response']['usage']
                assert (last['input_tokens'], last['generated_tokens']) == (usage['input_tokens'], usage['output_tokens']), last
                assert last['first_token_seconds'] is None and last['reused_tokens'] is None, last
            else:
                start, delta = events_sent[0], events_sent[-2]
                assert (last['input_tokens'], last['generated_tokens']) == \
                    (start['message']['usage']['input_tokens'], delta['usage']['output_tokens']), last
                assert last['first_token_seconds'] is None, last
        # Non-streaming: nothing is forwarded until the answer is complete, so phase and count are unknown.
        for path in ('/v1/chat/completions', '/v1/responses', '/v1/messages'):
            answer = {}
            worker = threading.Thread(target=lambda: answer.update(result=request(server.port, 'POST', path, content)))
            worker.start()
            deadline = time.time() + 20
            while time.time() < deadline:
                current = status()['current']
                if current is not None:
                    break
                time.sleep(0.05)
            assert current is not None and current['endpoint'] == path, path
            assert current['phase'] is None and current['generated_tokens'] is None and current['elapsed_seconds'] >= 0, current
            assert 'first_token_seconds' not in current and current['input_tokens'] is None, current
            worker.join(30)
            assert answer['result'][0] == 200
            index += 1
            assert answer['result'][2] == sent(record, index), path
            last = idle_status()['last']
            assert last['endpoint'] == path and last['stream'] is False and last['input_tokens'] > 0, last
        totals = idle_status()['totals']
        assert totals['requests'] == 6
    finally:
        assert server.stop() == 0
    assert_no_leftovers(record)
    print('PASS live progress: streamed fragments counted for Chat, Responses and Messages; non-streaming unknown; records unchanged')


def dialects(cli, root):
    """/status accounting of /v1/responses and /v1/messages, in the shapes edge0-strata answers with."""
    engine = make_engine(root / 'd-engine')
    model = make_model(root / 'd-model')
    record = root / 'd-record'
    server = Running(cli, model, engine, record)
    try:
        expected = []
        content = {'model': ALIAS, 'messages': [{'role': 'user', 'content': 'count these few words'}]}
        for path in ('/v1/responses', '/v1/messages'):
            for stream in (False, True):
                code, _, body = request(server.port, 'POST', path, dict(content, stream=stream))
                assert code == 200
                if path == '/v1/responses':
                    # Non-streaming: the finish state is the top-level status; streaming: response.status of the last event.
                    usage = data_events(body)[-1]['response']['usage'] if stream else json.loads(body)['usage']
                    expected.append((usage['input_tokens'], usage['output_tokens']))
                elif stream:
                    # The input count is message_start's; message_delta only brings output_tokens and stop_reason.
                    start, delta = data_events(body)[0], data_events(body)[-2]
                    assert start['type'] == 'message_start' and delta['type'] == 'message_delta'
                    expected.append((start['message']['usage']['input_tokens'], delta['usage']['output_tokens']))
                else:
                    usage = json.loads(body)['usage']
                    total = usage['input_tokens'] + usage['cache_creation_input_tokens'] + usage['cache_read_input_tokens']
                    assert total != usage['input_tokens']
                    expected.append((total, usage['output_tokens']))
        status = json.loads(request(server.port, 'GET', '/status')[2])
        records = list(reversed(status['recent_requests']))
        assert [(r['endpoint'], r['stream']) for r in records] == \
            [('/v1/responses', False), ('/v1/responses', True), ('/v1/messages', False), ('/v1/messages', True)], records
        for r, (input_tokens, generated) in zip(records, expected):
            assert r['stop_reason'] == 'end_of_generation', r
            assert (r['input_tokens'], r['generated_tokens']) == (input_tokens, generated), (r, input_tokens, generated)
        for r in records[2:]:  # Messages carry no timings
            assert r['first_token_seconds'] is None and r['decode_tokens_per_second'] is None and r['decode_seconds'] is None
        assert status['totals']['requests'] == 4
        assert status['totals']['input_tokens'] == sum(i for i, _ in expected)
        assert status['totals']['generated_tokens'] == sum(g for _, g in expected)
    finally:
        assert server.stop() == 0
    assert_no_leftovers(record)
    # Accounting never changes forwarding: a top-level "object" that is not a string (here null) is passed through
    # byte for byte, streaming or not, and every request is still recorded and leaves the active set.
    record = root / 'd-null-record'
    server = Running(cli, model, engine, record, env=environment(record, top_object='null'))
    try:
        index = 0
        for path in ('/v1/responses', '/v1/messages'):
            for stream in (False, True):
                code, kind, body = request(server.port, 'POST', path, dict(content, stream=stream))
                index += 1
                assert code == 200 and kind == ('text/event-stream' if stream else 'application/json'), (path, stream, code, body)
                assert body == sent(record, index), (path, stream)
                assert (data_events(body)[0] if stream else json.loads(body))['object'] is None
        status = json.loads(request(server.port, 'GET', '/status')[2])
        records = list(reversed(status['recent_requests']))
        assert [(r['endpoint'], r['stream'], r['http_status']) for r in records] == \
            [('/v1/responses', False, 200), ('/v1/responses', True, 200), ('/v1/messages', False, 200), ('/v1/messages', True, 200)], records
        assert status['current'] is None and status['totals']['requests'] == 4
    finally:
        assert server.stop() == 0
    assert_no_leftovers(record)
    print('PASS dialects: /status records Responses and Messages finish state and input tokens; malformed fields do not touch forwarding')


def non_loopback_address():
    addresses = socket.gethostbyname_ex(socket.gethostname())[2]
    address = next((value for value in addresses if not value.startswith(('127.', '169.254.'))), None)
    if not address:
        # Linux hosts often map their name to 127.0.1.1; ask the routing table instead (a UDP connect sends nothing).
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(('192.0.2.1', 9))
            address = probe.getsockname()[0]
    assert address and not address.startswith('127.'), 'No non-loopback IPv4 address'
    return address


def authentication(cli, root):
    engine = make_engine(root / 'a-engine')
    model = make_model(root / 'a-model')
    record = root / 'a-record'
    key = root / 'key'
    key.write_text(' SECRET_KEY_731 \n')
    address = non_loopback_address()
    server = Running(cli, model, engine, record, ['--host', '0.0.0.0', '--api-key-file', str(key)])
    try:
        body = {'model': ALIAS, 'messages': [{'role': 'user', 'content': 'hi'}]}
        # Every route, the generation endpoints and the passed-through GETs included, is checked before routing.
        for path in ('/v1/chat/completions', '/v1/responses', '/v1/messages'):
            assert request(server.port, 'POST', path, body, host=address)[0] == 401
            assert request(server.port, 'POST', path, body, {'Authorization': 'Bearer wrong'}, host=address)[0] == 401
        for path in ('/health', '/v1/models', f'/v1/models/{ALIAS}', '/status', '/v1/status', '/metrics', '/props'):
            assert request(server.port, 'GET', path, host=address)[0] == 401, path
        assert not requests_seen(record), 'an unauthorized request was forwarded'
        # Only the proxy's own start-up read of /v1/status reached edge0-strata; no unauthorized GET was passed through.
        assert len((record / 'get-v1-status.json').read_text().splitlines()) == 1
        assert not (record / 'get-metrics.json').exists() and not (record / 'get-props.json').exists()
        assert request(server.port, 'POST', '/v1/chat/completions', body, {'Authorization': 'Bearer SECRET_KEY_731'}, host=address)[0] == 200
        assert request(server.port, 'POST', '/v1/chat/completions', body)[0] == 200  # loopback needs no token
        assert requests_seen(record)[0]['authorization'] is None
        # The query string goes upstream unchanged (Anthropic's SDK sends /v1/messages?beta=true).
        assert request(server.port, 'POST', '/v1/messages?beta=true', {'model': ALIAS, 'input': 'q'})[0] == 200
        assert requests_seen(record)[-1]['target'] == '/v1/messages?beta=true' and requests_seen(record)[-1]['path'] == '/v1/messages'
        status = json.loads(request(server.port, 'GET', '/status')[2])
        assert status['listen']['auth'] is True and status['totals']['unauthorized'] == 13 and status['totals']['requests'] == 3
        code, _, payload = request(server.port, 'POST', '/v1/messages', body, host=address)
        assert code == 401 and json.loads(payload)['error']['type'] == 'authentication_error'
        assert request(server.port, 'POST', '/v1/messages', body, {'x-api-key': 'SECRET_KEY_731'}, host=address)[0] == 200
        before = len(requests_seen(record))
        code, _, payload = request(server.port, 'POST', '/v1/messages/count_tokens', body, {'x-api-key': 'SECRET_KEY_731'}, host=address)
        assert code == 200 and json.loads(payload) == {'input_tokens': 17}
        assert len(requests_seen(record)) == before + 1
        code, _, payload = request(server.port, 'POST', '/v1/messages/count_tokens', body, host=address)
        assert code == 401 and json.loads(payload)['error']['type'] == 'authentication_error'
        assert len(requests_seen(record)) == before + 1
        assert 'SECRET_KEY_731' not in json.dumps(status) + server.text()
    finally:
        assert server.stop() == 0
    unsafe = subprocess.run([str(cli), 'serve', '--model', str(model), '--host', '0.0.0.0', *strata_args(engine)],
                            capture_output=True, text=True, env=environment(root / 'a-unsafe'), timeout=60)
    assert unsafe.returncode != 0 and 'api-key-file' in unsafe.stderr and not (root / 'a-unsafe').exists()
    print('PASS authentication: non-loopback 401 before forwarding, token accepted, loopback exempt')


def lifecycle(cli, root):
    engine = make_engine(root / 'l-engine')
    model = make_model(root / 'l-model')
    # edge0-strata exits after serving: generation and health turn into 503.
    record = root / 'l-exit'
    server = Running(cli, model, engine, record, env=environment(record, exit_after=1))
    try:
        body = {'model': ALIAS, 'messages': [{'role': 'user', 'content': 'hi'}]}
        assert request(server.port, 'POST', '/v1/chat/completions', body)[0] == 200
        assert server.wait_for(lambda text: 'edge0-strata exited (exit code 3)' in text), server.text()
        code, _, payload = request(server.port, 'POST', '/v1/chat/completions', body)
        assert code == 503 and json.loads(payload)['error']['code'] == 'engine_exited'
        assert request(server.port, 'GET', '/health')[0] == 503
        for path in ('/v1/status', '/metrics', '/props'):
            code, _, payload = request(server.port, 'GET', path)
            assert code == 503 and json.loads(payload)['error']['code'] == 'engine_exited', path
        assert request(server.port, 'GET', '/status')[0] == 200
    finally:
        assert server.stop() == 0
    assert_no_leftovers(record)
    # edge0-strata exits after receiving a request but before answering: the client, the log line and /status all
    # carry the same 503.
    record = root / 'l-die'
    server = Running(cli, model, engine, record, env=environment(record, die_on_request=1))
    try:
        code, _, payload = request(server.port, 'POST', '/v1/chat/completions', {'model': ALIAS, 'messages': [{'role': 'user', 'content': 'hi'}]})
        assert code == 503 and json.loads(payload)['error']['code'] == 'engine_exited', (code, payload)
        assert len(requests_seen(record)) == 1, 'the request did not reach edge0-strata'
        last = json.loads(request(server.port, 'GET', '/status')[2])['last']
        logged = [json.loads(line) for line in server.text().splitlines() if line.startswith('{') and '"http_status"' in line]
        assert last['http_status'] == 503 and [r['http_status'] for r in logged] == [503], (last, logged)
    finally:
        assert server.stop() == 0
    assert_no_leftovers(record)
    # SIGTERM during a stream: the stream finishes first, then everything stops.
    record = root / 'l-term'
    server = Running(cli, model, engine, record, env=environment(record, chunk_delay=0.15))
    connection = http.client.HTTPConnection('127.0.0.1', server.port, timeout=60)
    connection.request('POST', '/v1/chat/completions', json.dumps({'model': ALIAS, 'stream': True,
                       'messages': [{'role': 'user', 'content': 'hi'}]}), {'Content-Type': 'application/json'})
    response = connection.getresponse()
    first = response.readline()
    assert first.startswith(b'data: ')
    server.process.send_signal(signal.CTRL_BREAK_EVENT if os.name == 'nt' else signal.SIGTERM)
    rest = response.read()
    connection.close()
    assert (first + rest).endswith(b'data: [DONE]\n\n') and b'"timings"' in rest
    assert server.stop() == 0
    text = server.text()
    assert text.index('stopping acceptance') < text.index('active requests finished') < text.index('edge0-strata stopped')
    assert any(e['event'] == ('sigbreak' if os.name == 'nt' else 'sigterm') for e in events(record))
    assert_no_leftovers(record)
    # SIGINT is handled the same way.
    record = root / 'l-int'
    if os.name == 'nt':
        from strata_console_ctrl_c import run_console
        run_console(cli, model, engine, record, 'serve')
    else:
        server = Running(cli, model, engine, record)
        assert server.stop(signal.SIGINT) == 0
        assert_no_leftovers(record)
    # edge0-strata itself killed outright (kill -9): the launcher passes the signal on as its own exit status
    # (128 + 9) instead of failing on it, and stopping afterwards leaves nothing.
    if os.name != 'nt':
        record = root / 'l-engine-kill'
        server = Running(cli, model, engine, record)
        try:
            os.kill(process_record(record)['pid'], signal.SIGKILL)
            assert server.wait_for(lambda text: 'edge0-strata exited (exit code ' in text), server.text()
            assert 'edge0-strata exited (exit code 137)' in server.text(), server.text()
            assert 'Traceback' not in server.text(), server.text()
        finally:
            assert server.stop() == 0
        assert_no_leftovers(record)
    # edge0-infer killed outright (kill -9): the launcher left in edge0-strata's process group stops it.
    for attempt in range(5):
        record = root / f'l-kill-{attempt}'
        server = Running(cli, model, engine, record)
        server.process.kill()
        server.process.wait(30)
        server.reader.join(10)
        try:
            assert_no_leftovers(record, temp_empty=False)  # killed outright: its private directory cannot be removed
        except AssertionError:
            pid = process_record(record)['pid']
            status = Path(f'/proc/{pid}/status')
            raise AssertionError(('edge0-strata outlived a killed edge0-infer', events(record), server.text(),
                                  status.read_text() if status.exists() else None))
        if os.name != 'nt':
            assert any(e['event'] == 'sigterm' for e in events(record)), 'edge0-strata was not asked to stop'
    print('PASS lifecycle: 503 after edge0-strata exits, stream drains before stop, SIGTERM/SIGINT/kill -9 leave no process')


def parameters(cli, root):
    engine = make_engine(root / 'o-engine')
    model = make_model(root / 'o-model')

    def config_of(extra):
        record = root / ('o-' + str(len(list(root.glob('o-run-*'))))) / 'record'
        record = root / f'o-run-{time.time_ns()}'
        server = Running(cli, model, engine, record, extra)
        try:
            return json.loads((record / 'config.json').read_text())
        finally:
            assert server.stop() == 0
    config = config_of([])
    assert config['gpu'] == 0 and config['model_name'] == ALIAS
    assert config['env']['STRATA_LATENT_DENSE'] == str(model / 'latent_dense_q8.bin') and config['env']['STRATA_LATENT_DENSE_FMT'] == '8'
    assert config['args'] == ['--pack', str(model), '--ple-io', 'direct', '--max-context', '131072']
    assert Path(config['exe']) == engine / 'build' / ('strata.exe' if os.name == 'nt' else 'strata') and config['cwd'] == str(engine)
    config = config_of(['--device', 'cuda:1', '--model-option', 'du=q4', '--model-option', 'ple-io=ram', '--context', '8192',
                        '--alias', 'big', '--prefill-device', 'cuda:1'])
    assert config['gpu'] == 1 and config['model_name'] == 'big'
    assert config['env']['STRATA_LATENT_DENSE'] == str(model / 'latent_dense_q4.bin') and config['env']['STRATA_LATENT_DENSE_FMT'] == '4'
    assert config['args'] == ['--pack', str(model), '--ple-io', 'ram', '--max-context', '8192']
    assert config_of(['--device', 'cuda'])['gpu'] == 0
    for extra, message in ((['--device', 'cpu'], 'CUDA GPU'), (['--device', 'metal'], 'CUDA GPU'), (['--device', 'vulkan'], 'CUDA GPU'),
                           (['--device', 'cuda:x'], 'CUDA GPU'), (['--prefill-device', 'cpu'], 'must equal --device'),
                           (['--model-option', 'du=q5'], 'q8|q4'), (['--model-option', 'ple-io=mmap'], 'direct|ram'),
                           (['--model-option', 'experts=resident'], 'unknown model option experts'),
                           (['--prefill-chunk', '64'], '--prefill-chunk'), (['--temperature', '0'], '--temperature'),
                           (['--max-tokens', '4'], '--max-tokens'), (['--no-warmup'], '--no-warmup'),
                           (['--session-idle-seconds', '5'], '--session-idle-seconds'), (['--threads', '4'], '--threads')):
        record = root / f'o-bad-{time.time_ns()}'
        result = subprocess.run([str(cli), 'serve', '--model', str(model), *strata_args(engine), *extra], capture_output=True,
                                text=True, env=environment(record), timeout=60)
        assert result.returncode == 2 and message in result.stderr, (extra, result.stderr)
        assert not record.exists(), ('a process was started', extra)
    print('PASS parameters: cuda devices, model options, context and alias reach the config; other options rejected first')


# Machine tuning words for the engine, one per --strata-arg; '-x' and 'direct' check that values are passed on as given.
TUNING = ['--pool-workers', '24', '--pcie-mode', 'dma', '--kv-resident', '32768', '-x', 'direct']


def tuning_args(words):
    return [item for word in words for item in ('--strata-arg', word)]


def passthrough(cli, root):
    engine = make_engine(root / 'pass-engine')
    model = make_model(root / 'pass-model')
    record = root / 'p-run'
    server = Running(cli, model, engine, record, ['--context', '8192', *tuning_args(TUNING)])
    try:
        config = json.loads((record / 'config.json').read_text())
    finally:
        assert server.stop() == 0
    assert_no_leftovers(record)
    assert config['args'] == ['--pack', str(model), '--ple-io', 'direct', '--max-context', '8192', *TUNING], config
    for name, words, message in (('empty', ['--pool-workers', ''], 'invalid_argument: --strata-arg needs a value'),
                                 ('template', ['--pool-workers', '24', '--ple-io', 'ram'],
                                  'invalid_argument: --strata-arg --ple-io: set by the launch template'),
                                 ('pack', ['--pack', 'elsewhere'], 'invalid_argument: --strata-arg --pack: set by the launch template'),
                                 ('context', ['--max-context', '4096'], 'invalid_argument: --strata-arg --max-context: use --context')):
        record = root / ('p-bad-' + name)
        result = subprocess.run([str(cli), 'serve', '--model', str(model), '--port', '0', *strata_args(engine), *tuning_args(words)],
                                capture_output=True, text=True, env=environment(record), timeout=60)
        assert result.returncode == 2 and message in result.stderr, (name, result.returncode, result.stderr)
        assert not record.exists(), ('a process was started', name)
        assert not list(temp_dir(record).iterdir()), ('temporary directories left behind', name)
    # inspect only checks files: --strata-arg is a usage error there.
    inspected = subprocess.run([str(cli), 'inspect', '--model', str(model), '--json', '--strata-arg', '--pool-workers'],
                               capture_output=True, text=True, timeout=60)
    assert inspected.returncode == 2 and not inspected.stdout and 'Usage: edge0-infer inspect' in inspected.stderr, (inspected.returncode, inspected.stdout, inspected.stderr)
    # A native model (the tiny fixture) treats it like any other unknown serve option.
    tiny = Path(os.environ.get('MOE_TINY', '/nonexistent')) / 'tiny-a'
    record = root / 'p-native'
    native = subprocess.run([str(cli), 'serve', '--model', str(tiny), '--device', 'cpu', '--port', '0', '--strata-arg', '--pool-workers'],
                            capture_output=True, text=True, env=environment(record), timeout=60)
    assert native.returncode != 0 and 'Unknown serve option: --strata-arg' in native.stderr and 'Listening:' not in native.stderr, \
        (native.returncode, native.stderr)
    print('PASS passthrough: --strata-arg words end the engine args; empty, template options and --max-context start nothing; '
          'inspect and native serve refuse it')


def model_formats(cli, root):
    from strata_support import TEMPLATE
    engine = make_engine(root / 'formats-engine')
    release = dict(TEMPLATE)
    release['args'] = [*TEMPLATE['args'], '--native', '${MODEL_DIR}/model-skeleton.gguf',
                       '--native-dense-gguf', '${MODEL_DIR}/model-skeleton.gguf',
                       '--expert-profile', '${MODEL_DIR}/profile.bin']
    for template in ('edge0-145b.json', 'edge0-120b.json'):
        (engine / 'edge0' / 'models' / template).write_text(json.dumps(release))
    for identifier, du in (('edge0_145b', 'q8'), ('edge0_120b', 'q4')):
        for experts in ('q3k', 'fp4'):
            name = identifier.replace('_', '-') + '-' + experts
            model = make_model(root / name, model_name=name)
            record = root / (name + '-record')
            server = Running(cli, model, engine, record)
            try:
                config = json.loads((record / 'config.json').read_text())
                assert config['model_name'] == identifier
                argv = config['args']
                for flag, filename in (('--pack', ''), ('--native', 'model-skeleton.gguf'),
                                       ('--native-dense-gguf', 'model-skeleton.gguf'),
                                       ('--expert-profile', 'profile.bin')):
                    assert Path(argv[argv.index(flag) + 1]) == model / filename
                assert Path(config['env']['STRATA_LATENT_A']) == model / 'experts.bin'
                assert Path(config['env']['STRATA_LATENT_LABELS']) == model / 'labels.bin'
                assert config['env']['STRATA_LAT_FUSED_Q3K'] == ('1' if experts == 'q3k' else '0')
                assert Path(config['env']['STRATA_LATENT_DENSE']) == model / f'latent_dense_{du}.bin'
                assert config['env']['STRATA_LATENT_DENSE_FMT'] == du[1:]
                assert 'log' not in config and config['exe'] != 'ignored'
            finally:
                assert server.stop() == 0
            assert_no_leftovers(record)
            base = [str(cli), 'inspect', '--model', str(model), '--json']
            result = subprocess.run(base, capture_output=True, text=True)
            report = json.loads(result.stdout)
            assert result.returncode == 0 and report['ok']
            assert ('scales/' in [f['filename'] for f in report['files']]) == (experts == 'q3k')
            for command in ('serve', 'chat', 'inspect'):
                record = root / (name + '-bad-' + command)
                extra = strata_args(engine) if command != 'inspect' else ['--json']
                result = subprocess.run([str(cli), command, '--model', str(model), *extra,
                                         '--model-option', 'experts=' + experts],
                                        env=environment(record), capture_output=True, text=True, timeout=60)
                assert result.returncode == 2 and 'unknown model option experts' in result.stderr
                assert not record.exists()
            (model / 'latent_dense_q4.bin').unlink()
            for command in ('serve', 'chat'):
                record = root / (name + '-missing-' + command)
                result = subprocess.run([str(cli), command, '--model', str(model), *strata_args(engine),
                                         '--model-option', 'du=q4'], env=environment(record),
                                        capture_output=True, text=True, timeout=60)
                assert result.returncode == 2 and 'missing: latent_dense_q4.bin' in result.stderr
                assert not record.exists()
    print('PASS release formats: four names, flat paths, selected kernels; obsolete options and missing D/U start nothing')


def inspect_defaults(cli, root):
    import shutil
    keys = {'adapter', 'delegated', 'engine', 'directory', 'ok', 'files'}
    file_keys = {'filename', 'required', 'exists', 'bytes'}
    for identifier, default_du, other_du in (('edge0_145b', 'q8', 'q4'), ('edge0_120b', 'q4', 'q8')):
        model = make_model(root / ('inspect-' + identifier), model_name=identifier.replace('_', '-') + '-q3k')
        base = [str(cli), 'inspect', '--model', str(model), '--json']

        def inspect(extra, ok):
            result = subprocess.run([*base, *extra], capture_output=True, text=True)
            report = json.loads(result.stdout)
            assert result.returncode == (0 if ok else 1) and report['ok'] == ok, (result.returncode, report)
            assert set(report) == keys
            assert report['adapter'] == identifier and report['delegated'] is True
            assert all(set(f) == file_keys and isinstance(f['bytes'], int) for f in report['files'])
            return result, report

        inspect([], True)
        (model / ('latent_dense_' + default_du + '.bin')).unlink()
        result, _ = inspect([], False)
        assert result.stderr.rstrip().endswith('use --model-option du=' + other_du + ' instead.'), result.stderr
        inspect(['--model-option', 'du=' + other_du], True)
        result, _ = inspect(['--model-option', 'du=' + default_du], False)
        suggested = result.stderr.rsplit(' use ', 1)[1].removesuffix(' instead.\n').split()
        assert suggested == ['--model-option', 'du=' + other_du]
        inspect(suggested, True)
        shutil.rmtree(model / 'scales')
        result, _ = inspect([], False)
        assert 'scales/' in result.stderr and '--model-option' not in result.stderr
        (model / 'scales').mkdir()
        shutil.rmtree(model / 'tokenizer')
        result, _ = inspect([], False)
        assert 'tokenizer/' in result.stderr and '--model-option' not in result.stderr
    print('PASS inspect defaults: per-model D/U, only complete alternatives, unchanged App JSON schema')


def platform_launch(cli, root):
    engine = make_engine(root / 'engine with spaces')
    model = make_model(root / 'model with spaces')
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(engine / '.venv')], check=True)
    record = root / 'platform-launch'
    cuda = root / 'CUDA toolkit'
    (cuda / 'bin' / 'x64').mkdir(parents=True)
    (cuda / 'bin' / 'x64' / 'cublas64_13.dll').write_bytes(b'fixture')
    env = environment(record)
    env['CUDA_PATH'] = str(cuda)
    server = Running(cli, model, engine, record, env=env, default_python=True)
    try:
        process = process_record(record)
        expected_python = engine / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        assert Path(process['python']) == expected_python, process
        print('PASS default Python: real venv without --strata-python')
        assert request(server.port, 'POST', '/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'hi'}]})[0] == 200
        print('PASS space paths: model and engine directories')
        config = json.loads((record / 'config.json').read_text())
        expected_dirs = [cuda / 'bin' / 'x64', cuda / 'bin'] if os.name == 'nt' else []
        assert [Path(p) for p in config['lib_dirs']] == expected_dirs, config
        prefix = ''.join(str(p) + os.pathsep for p in expected_dirs)
        assert process['path'] == prefix + env['PATH'], (process['path'], prefix + env['PATH'])
        print('PASS CUDA directories: configuration and child PATH')
    finally:
        assert server.stop() == 0
    assert_no_leftovers(record)


def interrupted_responses(cli, root):
    engine = make_engine(root / 'interrupt-engine')
    model = make_model(root / 'interrupt-model')
    for stream in (False, True):
        for mode in ('disconnect', 'exit'):
            for path in ('/v1/chat/completions', '/v1/responses', '/v1/messages'):
                record = root / ('interrupt-' + str(stream) + '-' + mode + '-' + path.rsplit('/', 1)[-1])
                server = Running(cli, model, engine, record)
                try:
                    body = {'model': ALIAS, 'stream': stream, 'fake_failure': mode}
                    code, _, wire = request(server.port, 'POST', path, body)
                    expected_status = 503 if mode == 'exit' else 502
                    expected_code = 'engine_exited' if mode == 'exit' else 'engine_unavailable'
                    if stream:
                        assert code == 200
                        prefix = sent(record, 1)
                        assert wire.startswith(prefix)
                        extra = data_events(wire[len(prefix):])
                        assert len(extra) == 1, wire
                        error = extra[0]
                        if path == '/v1/responses':
                            prior = data_events(prefix)
                            assert error['type'] == 'response.failed'
                            assert error['response']['id'] == prior[0]['response']['id']
                            assert error['sequence_number'] == max(e['sequence_number'] for e in prior) + 1
                            assert error['response']['error']['code'] == expected_code
                            assert b'event: response.failed\n' in wire
                        elif path == '/v1/messages':
                            assert error['type'] == 'error'
                            assert error['error']['type'] == ('overloaded_error' if mode == 'exit' else 'api_error')
                            assert b'event: error\n' in wire
                        else:
                            assert error['error']['code'] == expected_code and b'[DONE]' not in wire
                    else:
                        assert code == expected_status and b'partial' not in wire, (code, wire)
                        error = json.loads(wire)['error']
                        if path != '/v1/messages': assert error['code'] == expected_code
                        else: assert error['type'] == ('overloaded_error' if mode == 'exit' else 'api_error')
                    deadline = time.time() + 10
                    while True:
                        status = json.loads(request(server.port, 'GET', '/status')[2])
                        if status['current'] is None and status['last'] is not None: break
                        assert time.time() < deadline, status
                        time.sleep(0.02)
                    assert status['last']['http_status'] == expected_status and status['last']['code'] == expected_code, status['last']
                    assert status['last']['stop_reason'] == 'failed'
                    if mode == 'disconnect':
                        assert request(server.port, 'POST', path, {'model': ALIAS})[0] == 200
                    else:
                        code, _, wire = request(server.port, 'POST', '/v1/messages/count_tokens', {})
                        assert code == 503 and json.loads(wire)['error']['type'] == 'overloaded_error'
                finally:
                    assert server.stop() == 0
                assert_no_leftovers(record)
    record = root / 'interrupt-before-event'
    server = Running(cli, model, engine, record)
    try:
        code, _, wire = request(server.port, 'POST', '/v1/responses', {'stream': True, 'fake_failure': 'before_event'})
        events = data_events(wire)
        assert code == 200 and len(events) == 1
        assert events[0]['type'] == 'response.failed' and events[0]['sequence_number'] == 0
        assert events[0]['response']['id'].startswith('resp_') and events[0]['response']['error']['code'] == 'engine_unavailable'
        code, _, wire = request(server.port, 'POST', '/v1/chat/completions', {'fake_failure': 'short_length'})
        assert code == 502 and json.loads(wire)['error']['code'] == 'engine_unavailable', (code, wire)
        index = 2
        for path in ('/v1/chat/completions', '/v1/responses', '/v1/messages'):
            for stream in (False, True):
                code, _, wire = request(server.port, 'POST', path, {'stream': stream, 'fake_error': True})
                index += 1
                assert code == (200 if stream else 429) and wire == sent(record, index), (code, wire)
        # Wait for the provider's completion callback before comparing the history.
        deadline = time.time() + 10
        while json.loads(request(server.port, 'GET', '/status')[2])['current'] is not None:
            assert time.time() < deadline
            time.sleep(0.02)
        before = json.loads(request(server.port, 'GET', '/status')[2])
        for response_status in (200, 422):
            body = {'messages': [], 'fake_status': response_status}
            code, _, wire = request(server.port, 'POST', '/v1/messages/count_tokens?beta=true', body)
            assert code == response_status and wire == b'{"input_tokens": 17}'
            seen = requests_seen(record)[-1]
            assert seen['target'] == '/v1/messages/count_tokens?beta=true' and json.loads(seen['body']) == body
        after = json.loads(request(server.port, 'GET', '/status')[2])
        for key in ('current', 'last', 'recent_requests', 'totals'): assert before[key] == after[key], key
    finally:
        assert server.stop() == 0
    assert_no_leftovers(record)
    print('PASS interrupted responses: three protocols, disconnect/exit, before first event, short Content-Length, count_tokens')


def launch_handshake(cli, root):
    import shlex
    engine = make_engine(root / 'handshake-engine')
    model = make_model(root / 'handshake-model')
    record = root / 'handshake-environment'
    env = environment(record)
    env['STRATA_API_KEY'] = 'k'
    server = Running(cli, model, engine, record, env=env)
    try:
        assert not process_record(record)['strata_api_key_present'], 'STRATA_API_KEY leaked to engine'
        assert request(server.port, 'POST', '/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'hi'}]})[0] == 200
    finally:
        assert server.stop() == 0
    assert_no_leftovers(record)
    if os.name == 'nt':
        print('SKIP pipe handshake wrappers: Windows uses CreateProcessW PID')
        return
    wrapper = root / 'python-wrapper'
    # Exercise continuation reads too: the prefix inside a long unrelated line must not be accepted.
    noise = 'wrapper banner ' + 'x' * 300 + 'edge0-launcher-pid 1'
    wrapper.write_text('#!/bin/sh\nprintf "%s\\n" ' + shlex.quote(noise) + '\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
    wrapper.chmod(0o755)
    record = root / 'handshake-banner'
    server = Running(cli, model, engine, record, ['--strata-python', str(wrapper)])
    try:
        assert request(server.port, 'POST', '/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'hi'}]})[0] == 200
    finally:
        assert server.stop() == 0
    assert_no_leftovers(record)
    wrapper.write_text('#!/bin/sh\nprintf "wrapper did not start python\\n"\nexit 7\n')
    record = root / 'handshake-no-python'
    result = subprocess.run([str(cli), 'serve', '--model', str(model), '--port', '0',
                             '--strata-dir', str(engine), '--strata-python', str(wrapper)],
                            env=environment(record), capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'wrapper did not start python' in result.stderr, result.stderr
    assert 'Listening:' not in result.stderr and not record.exists()
    assert not list(temp_dir(record).iterdir())
    print('PASS launch handshake: environment cleared, banner skipped, missing PID reported, no leftovers')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        launch_handshake(args.cli, root)
        interrupted_responses(args.cli, root)
        inspect_defaults(args.cli, root)
        refusals(args.cli, root)
        failures(args.cli, root)
        snapshots = []
        proxy(args.cli, root, snapshots)
        live_progress(args.cli, root, snapshots)
        # Every /status read above against the field table: types, n/a fields null, every field exercised.
        check_status_schema(snapshots)
        dialects(args.cli, root)
        authentication(args.cli, root)
        platform_launch(args.cli, root)
        lifecycle(args.cli, root)
        parameters(args.cli, root)
        passthrough(args.cli, root)
        model_formats(args.cli, root)
    print('PASS serve strata')


if __name__ == '__main__':
    main()
