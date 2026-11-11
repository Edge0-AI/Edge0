"""Opt-in Windows direct API trials. Never loads a model, edits config, or runs tools."""
import argparse
import json
import os
import time
from pathlib import Path
import benchmark as b

QUOTE = 'An office quote includes four visits at $120 each, three floor jobs at $85 each, and $45 materials. Apply 10% discount to the subtotal then 13% tax. Return only JSON with numeric subtotal, discounted, tax and total.'
VALIDATION = 'Accept only JSON numbers which are finite and nonnegative (zero is valid); reject null, strings and negatives. Classify [0, null, "4", -2, 1.5, 3]. Return only JSON with arrays accepted and rejected, preserving their relative order.'
TOOL_PROMPT = 'Use the read_file tool to inspect package.json before suggesting a test command. Do not guess its contents. Call the tool exactly once; no other actions.'
TOOLS = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file in the current project', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path'], 'additionalProperties': False}}}]


def recall(n):
    filler = 'This is irrelevant background about routine office cleaning schedules. Ignore it when extracting marked keys.\n'
    return 'START_KEY=CLENEX-4181\n' + filler * n + 'MIDDLE_KEY=STRATA-8081\n' + filler * n + 'END_KEY=LONDON-9060\nReturn only JSON with start, middle and end holding the three marked key values.'


def wait_idle(base):
    for _ in range(30):
        if not any(s.get('is_processing') for s in b.get_json(base, '/slots')):
            return
        time.sleep(0.5)
    raise RuntimeError('Engine busy; no request sent')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True, help='Your locally accessible server URL, without /v1')
    p.add_argument('--model', required=True)
    p.add_argument('--mode', choices=('prefill', 'recall120k', 'medium'), required=True)
    p.add_argument('--prefill', type=int, default=8192, help='Label only; configure server separately')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output exists; choose a new path. No requests were sent.')
    if os.environ.get('STRATA_API_KEY'):
        b.AUTH_HEADERS['Authorization'] = 'Bearer ' + os.environ['STRATA_API_KEY']
    status = b.get_json(a.base, '/v1/status')
    if not status.get('loaded') or status.get('model') != a.model or status.get('context', {}).get('max_positions') != 122880:
        p.error('Expected the selected loaded model at 122880 context. No benchmark request sent.')
    wait_idle(a.base)
    result = {'mode': a.mode, 'declared_server_prefill': a.prefill, 'results': []}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == 'prefill':
        result['warmup'] = b.request(a.base, a.model, 'quote_math', QUOTE, 0)
        a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        if not b.validate(result['warmup']):
            raise RuntimeError('Warmup failed; stopping')
        cases = [(label, recall(n), run, None, None, 'low', 1024)
                 for label, n in [('recall_8k', 220), ('recall_28k', 780)] for run in (1, 2, 3)]
    elif a.mode == 'recall120k':
        cases = [('recall_120k', recall(3150), 1, None, None, 'low', 1024)]
    else:
        cases = [('quote_math', QUOTE, 1, None, {'type': 'json_object'}, 'medium', 8192),
                 ('validation_logic', VALIDATION, 1, None, {'type': 'json_object'}, 'medium', 8192),
                 ('tool_call', TOOL_PROMPT, 1, TOOLS, None, 'medium', 8192)]
    for label, prompt, run, tools, fmt, effort, allowance in cases:
        wait_idle(a.base)
        if b.available_ram() <= 4:
            raise RuntimeError('Less than 4 GiB available physical RAM; no next request sent')
        row = b.request(a.base, a.model, label, prompt, run, tools, response_format=fmt,
                        reasoning_effort=effort, max_tokens=allowance)
        row['semantic_pass'] = b.validate(row)
        row['strict_pass'] = b.validate(row, strict=True)
        result['results'].append(row)
        a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({k: row[k] for k in ('case', 'run', 'wall_seconds', 'semantic_pass', 'strict_pass', 'error')}), flush=True)
        if row['error'] or not row['semantic_pass'] or row['timings'].get('cache_n') != 0:
            raise RuntimeError('Failed or cache-reused trial preserved; stopping without retry')


if __name__ == '__main__':
    main()
