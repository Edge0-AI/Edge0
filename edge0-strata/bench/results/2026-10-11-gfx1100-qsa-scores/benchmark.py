"""One fresh prompt and one cached continuation; restart the server between invocations.

Use the same engine/config with STRATA_SCORES_THREAD=0 or 1 at server startup.
Model loading is excluded. This measures request latency, not streaming TTFT or coding quality.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_tokenizer(folder):
    spec = importlib.util.spec_from_file_location('qsa_bench_tokenizer', REPO / 'tools/strata_tokenizer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vocab = json.loads((folder / 'vocab.json').read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    return module.Tokenizer(tokens, (folder / 'merges.txt').read_text().split('\n'),
                            json.loads((folder / 'token_type.json').read_text()))


def make_requests(tokenizer, source, length, model, follow_tokens):
    ids = tokenizer.encode(source)
    if not ids or not 0 <= follow_tokens <= len(ids):
        raise ValueError('Empty source or invalid continuation size')
    text = tokenizer.decode((ids * (length // len(ids) + 1))[:length - 128])
    body = dict(model=model, temperature=0, top_p=1, top_k=1, seed=6300,
                reasoning_effort='none', max_tokens=1, messages=[
        dict(role='system', content='You are a precise coding assistant.'),
        dict(role='user', content=text + '\nReview the numerical implementation and synchronization in detail.')])
    follow = json.loads(json.dumps(body))
    follow['messages'].extend([
        dict(role='assistant', content='I will inspect the numerical implementation.'),
        dict(role='user', content='Continue with a detailed review of synchronization and memory lifetime.')])
    if follow_tokens:
        follow['messages'][-1]['content'] += '\nADDITIONAL_SOURCE\n' + tokenizer.decode(ids[-follow_tokens:])
    return [('fresh-' + str(length), body), ('follow-' + str(length), follow)]


def fetch(url, body=None):
    headers = {'Content-Type': 'application/json'}
    key = os.environ.get('STRATA_API_KEY')
    if key:
        headers['Authorization'] = 'Bearer ' + key
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--model', default='strata-qwen38-iq3s-uncensored')
    parser.add_argument('--length', type=int, choices=[131072, 240000], required=True)
    parser.add_argument('--follow-tokens', type=int, default=2048)
    parser.add_argument('--source', type=Path, default=HERE / 'prompt-source.txt.gz')
    parser.add_argument('--variant', choices=['off', 'on'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    source = gzip.decompress(args.source.read_bytes()) if args.source.suffix == '.gz' else args.source.read_bytes()
    requests = make_requests(load_tokenizer(args.tokenizer), source.decode(), args.length,
                             args.model, args.follow_tokens)
    receipt = dict(variant=args.variant, length=args.length, output_cap=1, seed=6300,
                   temperature=0, reasoning_effort='none', follow_source_tokens=args.follow_tokens,
                   source_sha256=digest(source), script_sha256=digest(Path(__file__).read_bytes()),
                   prepare_only=args.prepare_only, rows=[], status='prepared')
    # Refuse to overwrite previous observations, including failed attempts.
    with args.output.open('x') as output:
        def save():
            output.seek(0)
            json.dump(receipt, output, indent=2)
            output.write('\n')
            output.truncate()
            output.flush()
        try:
            save()
            if args.prepare_only:
                receipt['request_hashes'] = {case: digest(json.dumps(body).encode()) for case, body in requests}
                save()
                return
            health = fetch(args.url.rstrip('/') + '/health')
            if not health.get('loaded') or health.get('max_context') != 262144:
                raise RuntimeError('Requires an already loaded 262144-context server')
            for case, body in requests:
                started = time.monotonic()
                response = fetch(args.url.rstrip('/') + '/v1/chat/completions', body)
                elapsed = time.monotonic() - started
                metrics = fetch(args.url.rstrip('/') + '/metrics?requests=all')
                native = max(metrics['requests'], key=lambda row: row['time'])
                choice = response['choices'][0]
                message = choice.get('message', {})
                row = dict(case=case, seconds=elapsed, usage=response.get('usage'),
                           finish_reason=choice['finish_reason'],
                           request_sha256=digest(json.dumps(body).encode()),
                           output_sha256=digest(((message.get('reasoning_content') or '') + '\n' +
                                                 (message.get('content') or '')).encode()),
                           native={k: native.get(k) for k in (
                               'prompt_tokens', 'reused', 'prompt_read', 'output_tokens', 'engine_generated',
                               'prompt_ms', 'decode_ms', 'drafts_offered', 'drafts_accepted',
                               'hit_rate', 'ram_blobs', 'file_blobs', 'file_mb')})
                receipt['rows'].append(row)
                save()
                if choice['finish_reason'] not in ('length', 'stop'):
                    raise RuntimeError('Request did not complete normally')
                if case.startswith('fresh-') and native.get('reused') != 0:
                    raise RuntimeError('Fresh prompt reused a prefix: restart the server and use a NEW output file')
            receipt['status'] = 'complete'
            save()
        except Exception as error:
            receipt['status'] = 'failed'
            receipt['error'] = str(error)
            if isinstance(error, urllib.error.HTTPError):
                receipt['http_status'] = error.code
            save()
            raise


if __name__ == '__main__':
    main()
