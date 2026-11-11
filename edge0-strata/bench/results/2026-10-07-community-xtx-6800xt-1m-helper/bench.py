#!/usr/bin/env python3
"""Reproduce one recorded request fixture against an already-running Strata API."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

HERE = Path(__file__).resolve().parent


def prompt_for(case, rep):
    if case in ('coding', 'prose'):
        prompts = json.loads((HERE / 'prompts.json').read_text())
        name = 'agentcode' if case == 'coding' else 'longprose'
        extra = (' Include all code, tests and detailed explanations with examples; write at least 3500 words.'
                 if case == 'coding' else ' Write at least 3500 words of developed prose; keep the story moving.')
        return f'[fn100-sustained-{name}-{rep}]\n' + prompts[name] + extra, 8192, 'medium', 1.0
    name = 'full-window' if case == 'full' else 'depth-' + case[5:]
    text = gzip.decompress((HERE / f'{name}-prompt.txt.gz').read_bytes()).decode()
    return text, 504 if case == 'full' else 768, 'none', 0.0


def measure(url, model, prompt, budget, thinking, temperature):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': budget,
            'reasoning_effort': thinking, 'temperature': temperature, 'top_p': .95 if temperature else 1.0,
            'top_k': 20 if temperature else 1, 'min_p': 0.0, 'presence_penalty': 0.0, 'seed': 314159,
            'stream': True, 'stream_options': {'include_usage': True}}
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('STRATA_BENCH_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['STRATA_BENCH_API_KEY']
    request = urllib.request.Request(url.rstrip('/') + '/v1/chat/completions',
                                     json.dumps(body).encode(), headers)
    start = time.monotonic()
    first = last = None
    content, reasoning = [], []
    timings = usage = finish = None
    with urllib.request.urlopen(request, timeout=7200) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            payload = line[5:].strip()
            if payload == '[DONE]':
                break
            event = json.loads(payload)
            if event.get('error'):
                raise RuntimeError(event['error'])
            timings = event.get('timings') or timings
            usage = event.get('usage') or usage
            for choice in event.get('choices') or []:
                finish = choice.get('finish_reason') or finish
                delta = choice.get('delta') or {}
                visible, hidden = delta.get('content') or '', delta.get('reasoning_content') or ''
                if visible or hidden:
                    now = time.monotonic()
                    first = now if first is None else first
                    last = now
                    content.append(visible)
                    reasoning.append(hidden)
    if not timings or timings.get('predicted_n', 0) <= 0:
        raise RuntimeError('Missing successful engine generation timings')
    row = {'measured_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'thinking': thinking,
           'sampling': {k: body[k] for k in ('temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'seed')},
           'max_tokens': budget, 'wall_s': time.monotonic() - start,
           'ttft_s': first - start if first is not None else None,
           'timings': timings, 'usage': usage, 'finish': finish,
           'client_tg': (usage['completion_tokens'] - 1) / (last - first)
                        if usage and first is not None and last > first else None}
    return row, ''.join(reasoning) + '\n' + ''.join(content)


def check_recorded():
    records = json.loads((HERE / 'measurements.json').read_text())['arms']
    coding = [r for r in records['q8il-combo-w15-s8'] if r['case'] == 'sustained-agentcode']
    assert len(coding) == 3 and all(r['timings']['predicted_n'] == 8192 for r in coding)
    assert sorted(r['timings']['predicted_per_second'] for r in coding) == [96.7, 97.0, 97.2]
    for f, expected in json.loads((HERE / 'fixture-hashes.json').read_text()).items():
        assert hashlib.sha256(gzip.decompress((HERE / f).read_bytes())).hexdigest() == expected
    for r in json.loads((HERE / 'full-context.json').read_text()).values():
        assert r['input_tokens'] + r['generated_tokens'] + r['api_reserved_tokens'] == 1048576
        assert all(r['markers_found'].values()) and r['status'] == 'passed'
    print('Recorded coding counts/rates, exact fixture hashes and full-context capacity checks passed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--model', default='default-big')
    parser.add_argument('--case', choices=['coding', 'prose', 'depth50000', 'depth130000', 'full'], default='coding')
    parser.add_argument('--reps', type=int, default=1)
    parser.add_argument('--out', type=Path, default=Path('new-measurements'))
    parser.add_argument('--check-recorded', action='store_true')
    args = parser.parse_args()
    if args.check_recorded:
        check_recorded()
        return
    if args.reps <= 0:
        parser.error('--reps must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    for rep in range(1, args.reps + 1):
        prompt, budget, thinking, temp = prompt_for(args.case, rep)
        try:
            row, text = measure(args.url, args.model, prompt, budget, thinking, temp)
            row.update(case=args.case, rep=rep)
            row['prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
            (args.out / f'{args.case}-{rep}.json').write_text(json.dumps(row, indent=2) + '\n')
            (args.out / f'{args.case}-{rep}.txt').write_text(text)
            print(args.case, rep, 'TG', row['timings']['predicted_per_second'],
                  'PP', row['timings']['prompt_per_second'], 'counts', row['usage'])
            if args.case not in ('coding', 'prose') and row['timings'].get('cache_n', 0):
                raise RuntimeError('Prefix reuse: this request is not a cold PP comparison')
        except Exception as error:
            (args.out / f'{args.case}-{rep}-failure.json').write_text(json.dumps({'error': str(error)}, indent=2) + '\n')
            raise


if __name__ == '__main__':
    main()
