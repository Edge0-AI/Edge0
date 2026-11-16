#!/usr/bin/env python3
"""Matched fresh/follow-up prefill probe against an otherwise idle local Strata server.

Does not start/stop services. Use a dedicated server and its engine log, restart
between build/configuration arms, and keep model/template/settings identical.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time
import urllib.request

PATTERN = re.compile(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([0-9.]+) tok/s\), (\d+) generated in (\d+) ms \(([0-9.]+) tok/s\)')
FIELDS = ('prompt_tokens', 'reused', 'fresh', 'prefill_ms', 'prefill_tps', 'generated', 'decode_ms', 'decode_tps')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--model', required=True)
    p.add_argument('--engine-log', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--label', required=True)
    args = p.parse_args()
    results = []

    def request(messages, kind, trial):
        offset = args.engine_log.stat().st_size
        body = dict(model=args.model, messages=messages, max_tokens=128,
                    temperature=0, top_k=1, top_p=1, min_p=0, seed=42,
                    reasoning_effort='none')
        headers = {'Content-Type': 'application/json'}
        if os.environ.get('STRATA_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['STRATA_API_KEY']
        req = urllib.request.Request(args.url.rstrip('/') + '/v1/chat/completions',
                                     data=json.dumps(body).encode(), headers=headers)
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=360) as response:
            data = json.load(response)
        elapsed = time.monotonic() - start
        # The API response may arrive just before stderr's completion line is flushed.
        deadline = time.monotonic() + 2
        while True:
            with args.engine_log.open('rb') as log:
                log.seek(offset)
                matches = PATTERN.findall(log.read().decode(errors='replace'))
            if matches or time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        if len(matches) != 1:
            raise RuntimeError('Expected exactly one completed engine timing line; check idle server and log path')
        metrics = dict(zip(FIELDS, map(float, matches[0])))
        if kind == 'fresh' and metrics['reused'] != 0:
            raise RuntimeError('Fresh prompt reused cached tokens; restart this arm before comparison')
        choice = data['choices'][0]
        results.append(dict(label=args.label, kind=kind, trial=trial, wall_s=elapsed,
                            metrics=metrics, usage=data.get('usage'),
                            finish_reason=choice.get('finish_reason'), message=choice['message']))
        args.output.write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(results[-1]), flush=True)
        return choice['message']

    request([{'role': 'user', 'content': 'Reply READY.'}], 'warmup', 0)
    for trial, n in enumerate((140, 280, 140, 280), 1):
        code = '\n'.join(f'export function rule{i}(x) {{ return x === {i} ? x + {i+1} : x - {i}; }}' for i in range(n))
        messages = [{'role': 'user', 'content': f'CASE {trial}: Review this source and describe its behavior precisely in a paragraph.\n' + code}]
        reply = request(messages, 'fresh', trial)
        messages += [reply, {'role': 'user', 'content': 'Explain the most relevant boundary case and the smallest useful regression test. ' + 'Focus on integer equality, zero, negative input, unexpected types, and caller assumptions. ' * 5}]
        request(messages, 'followup', trial)


if __name__ == '__main__':
    main()
