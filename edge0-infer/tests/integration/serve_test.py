"""Portal HTTP, state retention and lifecycle integration contract."""
import argparse
import http.client
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_support import Server, events, tiny


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    parser.add_argument('--backend', default='cpu')
    parser.add_argument('--case', default='all')
    args = parser.parse_args()
    if args.backend == 'metal':
        info = subprocess.run([args.cli, 'info'], text=True, encoding='utf-8', capture_output=True)
        if info.returncode == 0 and not any(
                line.startswith('device.') and 'backend=metal' in line
                and any(token.startswith('name=') and token != 'name=' for token in shlex.split(line))
                for line in info.stdout.splitlines()):
            print('Not executed: no Metal device', flush=True)
            return 77
    model = tiny()
    description = json.loads(subprocess.run([args.cli, 'describe', '--model', str(model), '--device', args.backend],
                                           check=True, capture_output=True, text=True).stdout)
    capabilities = {'thinking': description['thinking']['enabled_supported'],
                    'thinking_default': description['thinking']['default']}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        if args.case in ('all', 'status'):
            from serve_status_test import run
            run(args.cli, model, root, args.backend)
        if args.case in ('all', 'port'):
            with Server(args.cli, model, root / 'port-owner', args.backend) as server:
                second = subprocess.run(
                    [args.cli, 'serve', '--model', str(model), '--device', args.backend,
                     '--context', '4096', '--port', str(server.port), '--no-warmup'],
                    capture_output=True, text=True, timeout=30)
                assert second.returncode != 0, (second.stdout, second.stderr)
                assert 'Cannot bind server' in second.stderr, second.stderr
                assert 'Listening:' not in second.stderr, second.stderr
                assert server.process.poll() is None
                assert server.request('/health')[0] == 200
                assert server.request('/v1/chat/completions', server.chat())[0] == 200
            print('PASS exclusive server port')
        for protocol in (('chat', 'responses') if args.case in ('all', 'protocol') else ()):
            for stream in (False, True):
                with Server(args.cli, model, root / f'{protocol}-{stream}', args.backend) as server:
                    endpoint = '/v1/chat/completions' if protocol == 'chat' else '/v1/responses'
                    assert server.request('/health')[0] == 200
                    assert json.loads(server.request('/v1/models')[1])['data'][0]['id'] == 'test'
                    assert json.loads(server.request('/v1/models')[1])['data'][0]['capabilities'] == capabilities
                    assert json.loads(server.request('/v1/models/test')[1])['capabilities'] == capabilities
                    assert server.request('/v1/models/wrong')[0] == 404
                    messages = []
                    for turn, text in enumerate(('hello', 'again')):
                        messages.append({'role': 'user', 'content': text})
                        body = {'model': 'test', 'messages' if protocol == 'chat' else 'input': messages, 'stream': stream}
                        if protocol == 'chat' and stream:
                            body['stream_options'] = {'include_usage': True}
                        status, response = server.request(endpoint, body)
                        assert status == 200, response
                        if stream:
                            parts = events(response)
                            if protocol == 'chat':
                                assert response.endswith('data: [DONE]\n\n')
                                answer = ''.join(p['choices'][0]['delta'].get('content', '') for p in parts if p['choices'])
                                usage = parts[-1]['usage']
                                metrics = parts[-1]['metrics']
                            else:
                                assert [p['sequence_number'] for p in parts] == list(range(len(parts)))
                                assert parts[0]['type'] == 'response.created'
                                answer = ''.join(p['delta'] for p in parts if p['type'] == 'response.output_text.delta')
                                usage = parts[-1]['response']['usage']
                                metrics = parts[-1]['response']['metrics']
                        else:
                            full = json.loads(response)
                            answer = full['choices'][0]['message']['content'] if protocol == 'chat' else ''.join(p['content'][0]['text'] for p in full['output'] if p['type'] == 'message')
                            usage = full['usage']
                            metrics = full['metrics']
                        evidence = server.evidence()[-1]
                        for key in ('first_token_seconds', 'decode_tokens_per_second'):
                            expected_metric = evidence[1]['request_summary'][key]
                            if expected_metric is None:
                                assert key not in metrics
                            else:
                                assert metrics[key] == expected_metric, (metrics, evidence[1])
                        assert 'computed_tokens' not in metrics
                        assert metrics['memory_peak_bytes'] > 0
                        assert metrics['memory_peak_metric'] in ('physical', 'resident')
                        memory_kind = 'physical' if metrics['memory_peak_metric'] == 'physical' else 'rss'
                        assert metrics['memory_peak_bytes'] == evidence[1]['memory_snapshot'][memory_kind]['values']['peak_bytes']
                        prefix = 'prompt' if protocol == 'chat' else 'input'
                        assert usage[prefix + '_tokens'] == len(evidence[0]['input'])
                        assert usage[prefix + '_tokens_details']['cached_tokens'] == evidence[1]['reused_tokens']
                        if turn:
                            assert evidence[1]['reused_tokens'] > 0
                        messages.append({'role': 'assistant', 'content': answer})
                        if not turn:
                            before = len(server.evidence())
                            for patch in ({'unknown': 1}, {'n': 2}, {'previous_response_id': 'x'}, {'tool_choice': 'required'},
                                          {'tools': [{'type': 'function', 'function': {'name': 'f', 'strict': True}}]},
                                          {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': 'x'}]}]},
                                          {'messages': [{'role': 'user', 'content': 'hello ' * 10000}]}):
                                status, error = server.request('/v1/chat/completions', server.chat(**patch))
                                assert status == 400, error
                            marker = 'UNIQUE_REJECTED_PORTAL_MARKER_2817'
                            status, error = server.request(endpoint, {'model': 'test', 'messages' if protocol == 'chat' else 'input': [{'role': 'user', 'content': marker}], 'promptcachekey': 'x'})
                            error = json.loads(error)['error']
                            assert status == 400 and error['param'] == 'promptcachekey' and error['message'].endswith(': promptcachekey'), error
                            with server.changed:
                                assert server.changed.wait_for(lambda: any('"param":"promptcachekey"' in line for line in server.lines), timeout=30), ''.join(server.lines)
                            line = next(json.loads(line) for line in server.lines if '"param":"promptcachekey"' in line)
                            assert {key: line[key] for key in ('endpoint','http_status','code','message','param')} == {'endpoint': endpoint, 'http_status': 400, 'code': 'unsupported_parameter', 'message': error['message'], 'param': 'promptcachekey'}, line
                            assert marker not in ''.join(server.lines)
                            assert len(server.evidence()) == before
                    script = root / 'turns.json'
                    script.write_text(json.dumps({'turns': [{'user': 'hello'}, {'user': 'again'}]}))
                    output = root / f'oracle-{protocol}-{stream}'
                    subprocess.run([args.cli, 'run', '--model', str(model), '--device', args.backend,
                        '--context', '4096', '--temperature', '0', '--presence-penalty', '0', '--max-tokens', '4', '--turns', str(script), '--out', str(output)], check=True, capture_output=True)
                    for index, (tokens, result) in enumerate(server.evidence(), 1):
                        reference = json.loads((output / str(index) / 'tokens.json').read_text(encoding='utf-8'))
                        assert all(tokens[key] == reference[key] for key in ('input', 'generated')), (protocol, stream, index, tokens, reference)
                        expected = json.loads((output / str(index) / 'result.json').read_text(encoding='utf-8'))
                        assert result['reused_tokens'] == expected['reused_tokens']
                    first = server.evidence()[0][0]
                    for alias in ({'promptCacheKey': 'key'}, {'promptCacheKey': 'key', 'prompt_cache_key': 'key'}):
                        status, response = server.request(endpoint, {'model': 'test', 'messages' if protocol == 'chat' else 'input': [{'role': 'user', 'content': 'hello'}], **alias})
                        assert status == 200, response
                        tokens = server.evidence()[-1][0]
                        assert all(tokens[key] == first[key] for key in ('input', 'generated')), (alias, tokens, first)
                    print('PASS protocol, stream, two-turn exact tokens and invalid-input retention:', protocol, stream)
        if args.case in ('all', 'temperature'):
            with Server(args.cli, model, root / 'temperature', args.backend) as server:
                first = json.loads(server.request('/v1/chat/completions', server.chat())[1])
                status, _ = server.request('/v1/chat/completions', server.chat([
                    {'role': 'user', 'content': 'hello'}, first['choices'][0]['message'], {'role': 'user', 'content': 'again'}], temperature=0.5))
                assert status == 200
                result = server.evidence()[-1][1]
                assert result['configuration']['temperature'] == 0.5 and not result['cold']
                marker = 'UNIQUE_PRIVATE_PORTAL_MARKER_1739'
                server.request('/v1/chat/completions', server.chat([{'role': 'user', 'content': marker}]))
                assert marker not in ''.join(server.lines)
        for mode in (mode for mode in ('busy', 'disconnect', 'shutdown') if args.case in ('all', mode)):
            hold = root / f'hold-{mode}'
            hold.touch()
            with Server(args.cli, model, root / mode, args.backend, ['--hold-file', str(hold)]) as server:
                connection = http.client.HTTPConnection('127.0.0.1', server.port)
                connection.request('POST', '/v1/chat/completions', json.dumps(server.chat(stream=True)), {'Content-Type': 'application/json'})
                response = connection.getresponse()
                while True:
                    line = response.readline().decode()
                    assert line, 'stream ended before hold'
                    if line.startswith('data: ') and json.loads(line[6:])['choices'][0]['delta'].get('content'):
                        break
                if mode == 'busy':
                    assert server.request('/v1/chat/completions', server.chat())[0] == 429
                    assert server.request('/v1/models')[0] == 200
                    hold.unlink()
                    response.read()
                elif mode == 'disconnect':
                    # Close the actual transport even while HTTPResponse owns its file.
                    sock = response.fp.raw._sock
                    sock.shutdown(socket.SHUT_RDWR)
                    response.close()
                    connection.close()
                    while not server.summaries():
                        assert server.process.poll() is None
                        time.sleep(0.01)
                    assert server.summaries()[-1]['stop_reason'] == 'cancelled'
                    snapshot = json.loads(server.request('/status')[1])
                    assert snapshot['last']['stop_reason'] == 'cancelled' and snapshot['totals']['requests'] == 1
                    assert snapshot['recent_requests'][0] == server.summaries()[-1]
                    hold.unlink()
                    status, _ = server.request('/v1/chat/completions', server.chat())
                    assert status == 200
                    assert server.evidence()[-1][1]['reused_tokens'] > 0
                else:
                    log = server.stop()
                    stages = ['stopping acceptance', 'active requests finished', 'session released', 'model unloaded', 'runtime safely finished']
                    assert [log.index(s) for s in stages] == sorted(log.index(s) for s in stages)
                connection.close()
            print('PASS', mode)
        if args.case in ('all', 'alias'):
            with Server(args.cli, model, root / 'alias', args.backend, alias=None) as server:
                assert json.loads(server.request('/v1/models')[1])['data'][0]['id'] == 'edge0_35b'
                assert 'model=edge0_35b' in ''.join(server.lines)
                status, response = server.request('/v1/chat/completions', {**server.chat(), 'model': 'edge0_35b'})
                assert status == 200, response
            print('PASS alias')
        if args.case in ('all', 'idle'):
            with Server(args.cli, model, root / 'idle', args.backend, ['--session-idle-seconds', '1']) as server:
                server.request('/v1/chat/completions', server.chat())
                while 'session released' not in ''.join(server.lines):
                    time.sleep(0.01)
                server.request('/v1/chat/completions', server.chat())
                assert server.evidence()[-1][1]['cold']
                assert 'session created' in ''.join(server.lines)
        if args.case in ('all', 'write-block'):
            with Server(args.cli, model, root / 'write-block', args.backend, ['--write-timeout-seconds', '1']) as server:
                sock = socket.socket()
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
                sock.connect(('127.0.0.1', server.port))
                body = json.dumps(server.chat(stream=True, logprobs=True, top_logprobs=20, max_tokens=4096)).encode()
                sock.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body)
                while not server.summaries():
                    assert server.process.poll() is None
                    time.sleep(0.01)
                result = server.evidence()[-1][1]
                assert result['stop_reason'] == 'failed', result['stop_reason']
                assert server.summaries()[-1]['error_stage'] == 4
                sock.close()
                status, body = server.request('/v1/chat/completions', server.chat())
                assert status == 200, (status, body, ''.join(server.lines))
        if args.case in ('all', 'stop'):
            with Server(args.cli, model, root / 'stop', args.backend, ['--max-tokens', '16']) as server:
                # Derive the stop string from a separate run, not from an expected literal.
                baseline = root / 'stop-baseline'
                command = [args.cli, 'run', '--model', str(model), '--device', args.backend, '--context', '4096',
                           '--temperature', '0', '--presence-penalty', '0', '--max-tokens', '16']
                baseline_run = subprocess.run([*command, '--user', 'hello', '--out', str(baseline)], check=True, capture_output=True, text=True)
                answer = baseline_run.stdout
                stop = answer[len(answer)//2:len(answer)//2+3]
                assert stop
                first = json.loads(server.request('/v1/chat/completions', server.chat(stop=stop))[1])
                assert first['choices'][0]['finish_reason'] == 'stop'
                messages = [{'role':'user','content':'hello'}, first['choices'][0]['message'], {'role':'user','content':'again'}]
                assert server.request('/v1/chat/completions', server.chat(messages))[0] == 200
                data = server.evidence()
                assert data[-1][1]['reuse_method'] == 'restore'
                assert data[-1][1]['checkpoint_restored']
                assert data[-1][1]['reused_tokens'] == data[0][1]['checkpoint_boundary'] == len(data[0][0]['input']) - 1
                script = root / 'stop-turns.json'
                script.write_text(json.dumps({'turns':[{'user':'hello','stop':[stop]}, {'user':'again'}]}))
                oracle = root / 'stop-oracle'
                subprocess.run([*command, '--turns', str(script), '--out', str(oracle)], check=True, capture_output=True)
                expected = json.loads((oracle / '2/tokens.json').read_text(encoding='utf-8'))
                assert data[-1][0]['generated'] == expected['generated']
        print('PASS selected network cases:', args.case)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
