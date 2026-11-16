"""A short real session against Edge0 145B through edge0-infer: chat over stdin, then serve's protocols.

Usage: real_session.py --cli <edge0-infer> --model <145B directory> [--strata-dir DIR] [--strata-python PATH]
       [--out session.json] [--part chat|serve|both] [--min-expert-cache-mib 10848] [--device cuda:0]

The serve part runs first. Right after serve is ready, edge0-strata's `expert_cache_mib` is read through the proxy
(engine_info.py) and recorded; the chat part does the same on the engine port named in edge0-strata's log. If it is
below --min-expert-cache-mib (0 skips the check), another process held GPU memory while it loaded, and the session
stops there with exit code 3.

chat: `edge0-infer chat` driven over stdin: thinking off, two turns where the second refers to the first,
`/regen`, then `/think on` and one more question. The answers, reasoning and per-turn summary lines are
captured.

serve: `edge0-infer serve`, then one non-streaming and one streaming request each to /v1/responses and
/v1/messages, one streaming /v1/chat/completions request with thinking on
(`chat_template_kwargs.enable_thinking=true`), and finally /status. For each request the status, content
type, the order of event types and the first 200 characters of the answer and of the reasoning are kept.

Prints a summary and writes everything to --out. Exits 0 when every request returned 200 and produced
text; whether the answers read well is for a person to judge.
"""
import argparse
import http.client
import json
import re
import signal
import subprocess
import sys
import threading
import time

import engine_info

CHAT_INPUT = [
    '/think off',
    'My name is Lin and my favourite number is 7. Please remember both and reply with one short sentence.',
    'What is my name, and what is my favourite number doubled?',
    '/regen',
    '/think on',
    'Is 91 a prime number? Answer briefly.',
    '/quit',
]


def engine_args(args):
    return (['--strata-dir', args.strata_dir] if args.strata_dir else []) + \
        (['--strata-python', args.strata_python] if args.strata_python else []) + \
        (['--device', args.device] if args.device else [])


def clip(text, size=200):
    return text if len(text) <= size else text[:size] + '...'


def run_chat(args):
    """chat over stdin; once it is ready, edge0-strata's expert cache is checked on the port named in its log."""
    command = [args.cli, 'chat', '--model', args.model] + engine_args(args)
    started = time.time()
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    head, log = [], None
    for line in process.stderr:
        head.append(line)
        match = re.search(r'\(log: (.+)\)$', line.rstrip('\n'))
        if match:
            log = match.group(1)
        if line.startswith('Ready: load '):
            break
    info, ok = [], True
    if process.poll() is None and log and args.min_expert_cache_mib:
        port = re.findall(r'ready: http://127\.0\.0\.1:(\d+)/v1', open(log, encoding='utf-8', errors='replace').read())
        ok, info = engine_info.check(f'http://127.0.0.1:{port[-1]}', args.min_expert_cache_mib) if port else (False, ['no port in the log'])
        print('\n'.join(info), flush=True)
    script = CHAT_INPUT if ok else ['/quit']
    stdout, stderr = process.communicate('\n'.join(script) + '\n', timeout=3600)
    return {'command': command, 'input': script, 'exit_code': process.returncode, 'seconds': round(time.time() - started, 1),
            'engine_info': info, 'precondition_failed': not ok, 'stdout': stdout, 'stderr_tail': (''.join(head) + stderr)[-6000:]}


class Server:
    def __init__(self, args):
        command = [args.cli, 'serve', '--model', args.model, '--port', '0'] + engine_args(args)
        self.process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True)
        self.lines, self.port = [], None
        ready = threading.Event()

        def read():
            for line in self.process.stderr:
                self.lines.append(line.rstrip('\n'))
                match = re.match(r'Listening: http://127\.0\.0\.1:(\d+) model=(\S+)$', line.rstrip('\n'))
                if match:
                    self.port, self.alias = int(match.group(1)), match.group(2)
                    ready.set()
            ready.set()
        threading.Thread(target=read, daemon=True).start()
        if not ready.wait(1800) or self.port is None:
            raise SystemExit('serve did not become ready:\n' + '\n'.join(self.lines[-20:]))

    def stop(self):
        self.process.send_signal(signal.SIGTERM)
        return self.process.wait(240)


def call(port, method, path, body=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=1800)
    connection.request(method, path, None if body is None else json.dumps(body), {'Content-Type': 'application/json'})
    response = connection.getresponse()
    kind = response.getheader('Content-Type') or ''
    if 'text/event-stream' not in kind:
        data = response.read().decode('utf-8', 'replace')
        connection.close()
        return response.status, kind, data, None
    events, raw = [], []
    for line in response:
        line = line.decode('utf-8', 'replace').rstrip('\r\n')
        raw.append(line)
        if line.startswith('event:'):
            events.append(('event', line[6:].strip()))
        elif line.startswith('data:'):
            payload = line[5:].strip()
            events.append(('data', payload))
    connection.close()
    return response.status, kind, '\n'.join(raw), events


def summarize(path, status, kind, data, events):
    item = {'path': path, 'status': status, 'content_type': kind, 'raw_head': data[:1500], 'raw_tail': data[-800:]}
    text, reasoning, types = '', '', []
    if events is None:
        value = json.loads(data) if data.startswith('{') else {}
        if path == '/v1/responses':
            for output in value.get('output', []):
                for part in output.get('content', []) or []:
                    if output.get('type') == 'reasoning':
                        reasoning += part.get('text', '')
                    elif part.get('type') == 'output_text':
                        text += part.get('text', '')
            types = [value.get('object'), value.get('status')]
        elif path == '/v1/messages':
            for block in value.get('content', []):
                if block.get('type') == 'text':
                    text += block.get('text', '')
                elif block.get('type') == 'thinking':
                    reasoning += block.get('thinking', '')
            types = [value.get('type'), value.get('stop_reason')]
    else:
        for kind_of, payload in events:
            if kind_of == 'event':
                if not types or types[-1] != payload:
                    types.append(payload)
                continue
            if payload == '[DONE]':
                types.append('[DONE]')
                continue
            try:
                value = json.loads(payload)
            except ValueError:
                continue
            if path == '/v1/chat/completions':
                for choice in value.get('choices', []):
                    delta = choice.get('delta', {})
                    text += delta.get('content') or ''
                    reasoning += delta.get('reasoning_content') or ''
                    if choice.get('finish_reason'):
                        types.append('finish_reason=' + choice['finish_reason'])
            elif path == '/v1/responses':
                if value.get('type') == 'response.output_text.delta':
                    text += value.get('delta', '')
                elif value.get('type', '').startswith('response.reasoning') and value.get('type', '').endswith('.delta'):
                    reasoning += value.get('delta', '')
            elif path == '/v1/messages':
                delta = value.get('delta', {})
                text += delta.get('text', '') if delta.get('type') == 'text_delta' else ''
                reasoning += delta.get('thinking', '') if delta.get('type') == 'thinking_delta' else ''
    item.update({'event_types': types, 'answer': clip(text), 'reasoning': clip(reasoning), 'answer_chars': len(text),
                 'reasoning_chars': len(reasoning)})
    return item


def run_serve(args):
    server = Server(args)
    alias, port = server.alias, server.port
    ok, info = engine_info.check(f'http://127.0.0.1:{port}', args.min_expert_cache_mib or None)
    print('\n'.join(info), flush=True)
    if not ok:
        server.stop()
        return {'alias': alias, 'engine_info': info, 'precondition_failed': True}
    requests = [
        ('/v1/responses', {'model': alias, 'input': 'Name the three primary colours of light in one sentence.',
                           'max_output_tokens': 2048}),
        ('/v1/responses', {'model': alias, 'input': 'Give one short tip for writing clear code.', 'max_output_tokens': 2048,
                           'stream': True}),
        ('/v1/messages', {'model': alias, 'max_tokens': 2048,
                          'messages': [{'role': 'user', 'content': 'What is the boiling point of water at sea level in Celsius? One sentence.'}]}),
        ('/v1/messages', {'model': alias, 'max_tokens': 2048, 'stream': True,
                          'messages': [{'role': 'user', 'content': 'Translate "good morning" into French and Spanish.'}]}),
        ('/v1/chat/completions', {'model': alias, 'stream': True, 'max_tokens': 2048,
                                  'chat_template_kwargs': {'enable_thinking': True},
                                  'messages': [{'role': 'user', 'content': 'A train travels 120 km in 1.5 hours. What is its average speed?'}]}),
    ]
    results = []
    try:
        for path, body in requests:
            started = time.time()
            status, kind, data, events = call(port, 'POST', path, body)
            item = summarize(path, status, kind, data, events)
            item.update({'stream': bool(body.get('stream')), 'seconds': round(time.time() - started, 1), 'request': body})
            results.append(item)
            print(f"{path} stream={item['stream']} status={status} events={item['event_types'][:12]} answer={item['answer']!r}",
                  flush=True)
        status = json.loads(call(port, 'GET', '/status')[2])
    finally:
        code = server.stop()
    recent = status.get('recent_requests', [])
    return {'alias': alias, 'engine_info': info, 'requests': results, 'serve_exit_code': code,
            'status_recent_requests': recent, 'status_totals': status.get('totals'), 'status_last': status.get('last'),
            'serve_log_tail': server.lines[-30:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cli', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--strata-dir')
    parser.add_argument('--strata-python')
    parser.add_argument('--device')
    parser.add_argument('--part', choices=['chat', 'serve', 'both'], default='both')
    parser.add_argument('--out')
    parser.add_argument('--min-expert-cache-mib', type=int, default=10848)
    args = parser.parse_args()
    report, ok = {}, True
    if args.part in ('serve', 'both'):
        report['serve'] = run_serve(args)
        if report['serve'].get('precondition_failed'):
            if args.out:
                with open(args.out, 'w', encoding='utf-8') as out:
                    json.dump(report, out, ensure_ascii=False, indent=1)
            print('RESULT PRECONDITION FAILED: expert_cache_mib is missing or below the minimum '
                  '(a smaller cache means GPU memory was in use by another process while edge0-strata loaded)')
            sys.exit(3)
        ok &= all(item['status'] == 200 and item['answer_chars'] > 0 for item in report['serve']['requests'])
    if args.part in ('chat', 'both'):
        report['chat'] = run_chat(args)
        if report['chat']['precondition_failed']:
            if args.out:
                with open(args.out, 'w', encoding='utf-8') as out:
                    json.dump(report, out, ensure_ascii=False, indent=1)
            print('RESULT PRECONDITION FAILED (chat): expert_cache_mib is missing or below the minimum')
            sys.exit(3)
        ok &= report['chat']['exit_code'] == 0 and bool(report['chat']['stdout'].strip())
        print('chat exit', report['chat']['exit_code'], flush=True)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as out:
            json.dump(report, out, ensure_ascii=False, indent=1)
    print('RESULT', 'OK' if ok else 'CHECK')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
