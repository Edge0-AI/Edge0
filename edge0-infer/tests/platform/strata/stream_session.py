"""One recorded session of a Metal latent model through `edge0-infer serve`, for the SSD-streaming checks.

Usage: stream_session.py --cli <edge0-infer> --model <model directory> --strata-dir <edge0-strata> --out <dir>
       [--parts questions,long,chat] [--long-file library-report-prompt.txt] [--bench]
       [--wire-gib 128 --wire-min-free-gib 24 [--squeeze-min-free-gib 8] [--cache-max-percent 5]]
       [--watch] [--max-seconds 7200] [--config-env NAME=value ...]

Starts `edge0-infer serve --device metal` with this directory's engine_log_python.py as --strata-python, so that the
engine's own log is kept (<out>/engine.log); the environment (STRATA_METAL_STREAM, STRATA_METAL_ARENA_GB,
...) is passed on as it is; --config-env sets a variable in the server config's "env" instead (needed for
the ones the template sets, such as STRATA_METAL_PROMPT_CHUNK: the config's "env" wins over the environment). Then, through the proxy, greedily with thinking off
(temperature 0, top_k 1, `chat_template_kwargs.enable_thinking=false`, no streaming):

  questions  the four fixed questions (max_tokens 256), each answer with its draft counts (timings.draft_n,
             timings.draft_n_accepted);
  long       --long-file as one user message, max_tokens 64;
  chat       a three-turn conversation (each turn sends the earlier turns and answers), max_tokens 128.

--bench then runs edge0-strata's unchanged tools/strata_bench.py --url <proxy> --runs 3 --out <out>/bench (with
edge0-strata's .venv Python) and keeps its output.

--wire-gib first wires that much memory with wire_memory.py (stops at --wire-min-free-gib), then briefly wires more
down to --squeeze-min-free-gib (the file cache is pushed out) and checks with file_cache.py that experts.bin of
the model is cached at most --cache-max-percent (else it stops, exit code 4). --watch runs mem_watch.py on the
engine (the command line contains build-metal/strata) from before serve starts to after it stops. Whatever happens,
serve is stopped first, then the watch, then the wired memory is released.

Writes <out>/session.json (answers, timings, the wiring and cache numbers, the watch summary) and prints a short
summary. Exit 0 when every request returned 200.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS = [
    'What is 17 * 23? Answer with the number only.',
    '\u7528\u4e09\u53e5\u8bdd\u4ecb\u7ecd\u4e00\u4e0b\u957f\u57ce\u3002',  # Chinese: introduce the Great Wall in three sentences.
    'Write a Python function that checks whether a string is a palindrome.',
    # Chinese: list the eight planets of the solar system, one sentence each.
    '\u5217\u51fa\u592a\u9633\u7cfb\u7684\u516b\u5927\u884c\u661f\uff0c\u5e76\u5404\u7528\u4e00\u53e5\u8bdd\u8bf4\u660e\u7279\u70b9\u3002',
]
CHAT = [
    'My name is Lin and my favourite number is 7. Please remember both and reply with one short sentence.',
    'What is my name, and what is my favourite number doubled?',
    'Is 91 a prime number? Answer briefly.',
]


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def ask(port, alias, messages, max_tokens):
    body = {'model': alias, 'messages': messages, 'temperature': 0, 'top_k': 1, 'max_tokens': max_tokens,
            'stream': False, 'chat_template_kwargs': {'enable_thinking': False}}
    started = time.time()
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=3600)
    connection.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    if response.status != 200:
        return {'status': response.status, 'error': payload[:500].decode('utf-8', 'replace')}
    value = json.loads(payload)
    choice = value['choices'][0]
    timings = value.get('timings') or {}
    return {'status': 200, 'content': choice['message'].get('content'), 'finish_reason': choice.get('finish_reason'),
            'usage': value.get('usage'), 'draft_n': timings.get('draft_n'), 'draft_n_accepted': timings.get('draft_n_accepted'),
            'timings': timings, 'seconds': round(time.time() - started, 2)}


class Child:
    """A helper process whose stdout lines are kept; wait_for() waits for a line that matches."""

    def __init__(self, command, env=None):
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        self.lines = []
        self.changed = threading.Condition()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            with self.changed:
                self.lines.append(line.rstrip('\n'))
                self.changed.notify_all()
        with self.changed:
            self.changed.notify_all()

    def wait_for(self, pattern, seconds):
        deadline = time.time() + seconds
        with self.changed:
            while True:
                for line in self.lines:
                    if re.search(pattern, line):
                        return line
                if self.process.poll() is not None and not any(re.search(pattern, l) for l in self.lines):
                    return None
                left = deadline - time.time()
                if left <= 0:
                    return None
                self.changed.wait(min(left, 1.0))

    def stop(self, sig=signal.SIGTERM, seconds=300):
        if self.process.poll() is None:
            self.process.send_signal(sig)
            try:
                self.process.wait(seconds)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        return self.process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cli', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--strata-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--parts', default='questions,long,chat')
    parser.add_argument('--long-file')
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--wire-gib', type=float, default=0)
    parser.add_argument('--wire-min-free-gib', type=float, default=24)
    parser.add_argument('--squeeze-min-free-gib', type=float, default=8)
    parser.add_argument('--cache-max-percent', type=float, default=5)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--max-seconds', type=float, default=7200)
    parser.add_argument('--config-env', action='append', default=[], metavar='NAME=value')
    args = parser.parse_args()
    parts = [p for p in args.parts.split(',') if p]
    if 'long' in parts and not args.long_file:
        parser.error('--parts long needs --long-file')
    os.makedirs(args.out, exist_ok=True)
    python = sys.executable
    if any('=' not in item or ',' in item for item in args.config_env):
        parser.error('--config-env takes NAME=value (no commas)')
    result = {'model': args.model, 'parts': parts, 'env': {k: v for k, v in os.environ.items() if k.startswith('STRATA_')},
              'config_env': args.config_env,
              'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    wire = watch = server = None
    ok = True
    try:
        if args.wire_gib > 0:
            wire = Child([python, os.path.join(HERE, 'wire_memory.py'), '--gib', str(args.wire_gib), '--min-free-gib',
                          str(args.wire_min_free_gib), '--max-seconds', str(args.max_seconds)])
            ready = wire.wait_for(r'^READY ', 1800)
            log(f'wire: {ready}')
            if ready is None:
                raise SystemExit('wire_memory.py did not become ready:\n' + '\n'.join(wire.lines[-10:]))
            result['wire'] = list(wire.lines)
            squeeze = subprocess.run([python, os.path.join(HERE, 'wire_memory.py'), '--gib', '64', '--min-free-gib',
                                      str(args.squeeze_min_free_gib), '--step-gib', '2', '--max-seconds', '5'],
                                     capture_output=True, text=True)
            result['squeeze'] = squeeze.stdout.splitlines()
            log('squeeze: ' + ' | '.join(result['squeeze'][-3:]))
        experts = os.path.join(args.model, 'experts.bin')
        cache = subprocess.run([python, os.path.join(HERE, 'file_cache.py'), '--json', experts],
                               capture_output=True, text=True, check=True)
        result['experts_cached_percent'] = json.loads(cache.stdout)[experts]['cached_percent']
        log(f'experts.bin cached {result["experts_cached_percent"]}%')
        if args.wire_gib > 0 and result['experts_cached_percent'] > args.cache_max_percent:
            result['stopped'] = 'experts.bin is still cached'
            raise SystemExit(4)
        if args.watch:
            watch = Child([python, os.path.join(HERE, 'mem_watch.py'), '--match', 'build-metal/strata', '--interval', '2',
                           '--out', os.path.join(args.out, 'watch.json'), '--max-seconds', str(args.max_seconds)])
        env = dict(os.environ)
        env['EDGE0_TEST_ENGINE_LOG'] = os.path.join(os.path.abspath(args.out), 'engine.log')
        env['EDGE0_TEST_CONFIG_ENV'] = ','.join(args.config_env)
        command = [args.cli, 'serve', '--model', args.model, '--device', 'metal', '--port', '0', '--strata-dir',
                   args.strata_dir, '--strata-python', os.path.join(HERE, 'engine_log_python.py')]
        started = time.time()
        server = Child(command, env=env)
        line = server.wait_for(r'^Listening: http://[^:]+:\d+ model=\S+', 3600)
        if line is None:
            raise SystemExit('serve did not become ready:\n' + '\n'.join(server.lines[-20:]))
        port, alias = re.match(r'Listening: http://[^:]+:(\d+) model=(\S+)', line).groups()
        port = int(port)
        result['load_seconds'] = round(time.time() - started, 1)
        log(f'serve ready on {port} ({alias}) in {result["load_seconds"]} s')
        if 'questions' in parts:
            result['questions'] = []
            for q in QUESTIONS:
                a = ask(port, alias, [{'role': 'user', 'content': q}], 256)
                result['questions'].append({'question': q, **a})
                log(f'question: {a.get("status")} drafts {a.get("draft_n_accepted")}/{a.get("draft_n")} {a.get("seconds")} s')
        if 'long' in parts:
            with open(args.long_file, encoding='utf-8') as f:
                text = f.read()
            a = ask(port, alias, [{'role': 'user', 'content': text}], 64)
            result['long'] = a
            log(f'long: {a.get("status")} prompt {(a.get("usage") or {}).get("prompt_tokens")} tokens, {a.get("seconds")} s')
        if 'chat' in parts:
            messages, turns = [], []
            for q in CHAT:
                messages.append({'role': 'user', 'content': q})
                a = ask(port, alias, list(messages), 128)
                turns.append({'user': q, **a})
                messages.append({'role': 'assistant', 'content': a.get('content') or ''})
                log(f'chat: {a.get("status")} drafts {a.get("draft_n_accepted")}/{a.get("draft_n")}')
            result['chat'] = turns
        if args.bench:
            venv_python = os.path.join(args.strata_dir, '.venv', 'bin', 'python')
            bench = subprocess.run([venv_python, os.path.join(args.strata_dir, 'tools', 'strata_bench.py'), '--url',
                                    f'http://127.0.0.1:{port}', '--runs', '3', '--out', os.path.join(os.path.abspath(args.out), 'bench')],
                                   capture_output=True, text=True,
                                   cwd=args.strata_dir)
            result['bench'] = {'exit_code': bench.returncode, 'stdout': bench.stdout[-12000:], 'stderr': bench.stderr[-4000:]}
            log(f'bench: exit {bench.returncode}')
    finally:
        if server:
            result['serve_exit'] = server.stop(signal.SIGTERM, 300)
            result['serve_tail'] = server.lines[-40:]
        if watch:
            watch.stop(signal.SIGTERM, 60)
            summary = [l for l in watch.lines if l.startswith('SUMMARY ')]
            result['watch'] = json.loads(summary[-1][8:]) if summary else None
        if wire:
            wire.stop(signal.SIGTERM, 120)
            result['wire_release'] = wire.lines[-1:]
        with open(os.path.join(args.out, 'session.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
    statuses = [q.get('status') for q in result.get('questions', [])] + [t.get('status') for t in result.get('chat', [])]
    if 'long' in result:
        statuses.append(result['long'].get('status'))
    ok = all(s == 200 for s in statuses) and (not args.bench or result['bench']['exit_code'] == 0)
    log(f'done: {len(statuses)} requests, all 200: {ok}; watch {json.dumps(result.get("watch"))}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
