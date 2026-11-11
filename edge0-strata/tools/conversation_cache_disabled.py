"""Compare untouched upstream with cache-disabled engines; dry-run by default.

Private engines run sequentially in an exclusive GPU window. This is a correctness
gate, not a throughput benchmark: state fingerprinting adds diagnostic overhead.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from conversation_cache_parity import load_tokenizer, require
from conversation_cache_soak import answer_text
from serve.frontend import ChatTemplate
from serve.server import StrataEngine, child_env

NAMES = ('short', 'A', 'A-live', 'A-checkpoint', 'B', 'A-return', 'B-return', 'A-return-again')
# (not 'version': the gate also compares a new release with the previous one, where only the version differs)
SETTINGS = ('context', 'kv', 'kv_resident', 'expert_slots', 'spec',
            'mtp_max', 'lookup', 'cvec', 'pcie_frac', 'spec_min_p', 'pool_workers')
# Untouched upstream does not emit the indexer `dead` and `pooled_full` fields
# added by this PR; `pooled` is the completed-row extent in both engines.
# Draft scratch and unused page padding are not comparable authoritative state.
STATE_KEYS = ('L', 'gdn', 'ple', 'tail', 'pooled', 'kv', 'ple_prev')


def state_hashes(text, *, candidate=False):
    hashes = []
    for line in text.splitlines():
        if 'STATE_HASH L=' in line:
            fields = dict(re.findall(r'(\w+)=([0-9a-f,-]+)', line))
            require(set(STATE_KEYS) <= fields.keys(), 'missing upstream-compatible state fields')
            if candidate:
                # Evidence that the candidate is a build with this change: it
                # also fingerprints the spare row (compared by the cache-on gates).
                require('pooled_full' in fields, 'candidate lacks the pooled_full fingerprint')
            hashes.append({key: fields[key] for key in STATE_KEYS})
    return hashes


def common_args(cfg, spec):
    # Upstream has no conversation-cache options. Omit them from EVERY arm so
    # the candidates exercise their disabled default with identical arguments.
    args, i = [], 0
    while i < len(cfg['args']):
        arg = cfg['args'][i]
        if arg.startswith('--conversation-cache-'):
            require(i + 1 < len(cfg['args']), 'cache option lacks a value')
            i += 2
        else:
            args.append(arg)
            i += 1
    return args + ['--prompt-cache', '6', '--adapt-swaps', '0', '--spec', str(max(2, spec)),
                   '--mtp-max-t', str(spec), '--suffix-draft', '0', '--spec-min-p', '0']


def verify(results):
    arms = results['arms']
    require(len(arms) >= 2 and arms[0]['name'] == 'upstream', 'missing comparison arms')
    baseline = arms[0]
    requests = results['requests']
    require([r['name'] for r in requests] == list(NAMES), 'incomplete request sequence')
    require(all(r['ids'] and r['max_new'] > 0 for r in requests), 'empty request')
    digest = hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()
    require(digest == results['request_digest'], 'request evidence changed')
    for arm in arms:
        require(arm['args'] == baseline['args'], 'engine arguments differ')
        require(arm['request_digest'] == results['request_digest'], 'input requests differ')
        require([r['name'] for r in arm['records']] == list(NAMES), 'incomplete output sequence')
        require(not arm['cache_events'], 'disabled cache performed snapshot operations')
        for key in SETTINGS:
            require(key in arm['info'] and arm['info'][key] == baseline['info'][key],
                    f'{arm["name"]}: resolved setting differs: {key}')
        if arm['name'] != 'upstream':
            require(arm['info'].get('conversation_cache_mib') == 0, 'RAM cache not disabled')
            require(arm['info'].get('conversation_disk_mib', 0) == 0, 'disk cache not disabled')
        for request, reference, record in zip(requests, baseline['records'], arm['records']):
            require(record['ids'] and record['finish'] in ('stop', 'length'), 'incomplete generation')
            require(set(STATE_KEYS) <= record['state'].keys(), 'incomplete state fingerprint')
            require(record['ids'] == reference['ids'], f'{arm["name"]}/{record["name"]}: output differs')
            require(record['state'] == reference['state'], f'{arm["name"]}/{record["name"]}: state differs')
            require(record['reused'] == reference['reused'], 'prefix reuse differs')
            require(record['prompt_tokens'] == len(request['ids']), 'wrong prompt length')
            if request.get('expected'):
                require(answer_text(record['text']) == request['expected'].casefold(), 'known answer is wrong')
        records = {r['name']: r for r in arm['records']}
        require(records['A-live']['reused'] > 0 and records['A-checkpoint']['reused'] > 0,
                'continuation/checkpoint path not exercised')
        require(records['A-return']['reused'] == records['A-return-again']['reused'] == 0,
                'conversation switching did not force cold fallback')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, action='append', required=True)
    ap.add_argument('--output', type=Path, required=True, help='new directory; existing paths refused')
    ap.add_argument('--paragraphs', type=int, default=128)
    ap.add_argument('--spec', type=int, choices=(1, 4), default=1)
    ap.add_argument('--run', action='store_true')
    a = ap.parse_args()
    if a.paragraphs < 1:
        ap.error('paragraphs must be positive')
    if not a.run:
        print('Dry run: upstream and disabled candidates; identical inputs/settings, known answers and state parity.')
        return
    cfg = json.loads(a.config.read_text(encoding='utf-8'))
    tok_path = Path(cfg['tokenizer'])
    tok = load_tokenizer(tok_path)
    tpl_path = tok_path / 'chat_template.jinja'
    tpl = ChatTemplate(tpl_path if tpl_path.exists() else ROOT / 'serve/chat_template.jinja')
    def encode(text):
        return tok.encode(text, parse_special=True)
    def prompt(text):
        return encode(tpl.render([{'role': 'user', 'content': text}], enable_thinking=False))
    text = 'Conversation A. Remember the exact code AZURE-314159.\n'
    text += '\n'.join(f'Record {i}: blue square, green triangle, red circle.' for i in range(a.paragraphs))
    A = prompt(text + '\nThe exact code to remember is AZURE-314159. Reply OK.')
    B = prompt('Unrelated conversation B. Remember BRONZE-271828. Reply OK.')
    suffix = encode('<|im_end|>\n<|im_start|>user\nWhat exact code did I ask you to remember? '
                    'Reply with only that code.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    args = common_args(cfg, a.spec)
    env = child_env(cfg)
    env.pop('STRATA_SNAPSHOT_VERIFY', None)
    env['STRATA_STATE_HASH'] = '1'
    env['STRATA_MTP_BATCH'] = '1'
    a.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    results = {'config': cfg, 'paragraphs': a.paragraphs, 'spec': a.spec, 'arms': [], 'requests': [],
               'compared_state_fields': STATE_KEYS,
               'environment': {k: v for k, v in env.items() if k.startswith('STRATA_')}}
    for index, exe in enumerate([a.upstream, *a.candidate]):
        name = 'upstream' if index == 0 else f'candidate-{index}'
        log = a.output / f'{name}.log'
        print(f'START {name}: {exe}', flush=True)
        started = time.monotonic()
        engine = StrataEngine(str(exe.resolve()), args, cwd=cfg.get('cwd'), log=str(log), env=env)
        arm = {'name': name, 'exe': str(exe.resolve()), 'args': args, 'info': dict(engine.info),
               'startup_seconds': time.monotonic() - started, 'records': []}
        with exe.open('rb') as source:
            arm['exe_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
        def generate(request):
            ids = [t for t in engine.generate(request['ids'], request['max_new'], {'temperature': 0},
                                             threading.Event()) if t is not None]
            arm['records'].append({'name': request['name'], 'ids': ids, **engine.last,
                                   'text': tok.decode(ids)})
            return ids
        try:
            if index == 0:
                def add(name, ids, max_new, expected=None):
                    request = dict(name=name, ids=ids, max_new=max_new, expected=expected)
                    results['requests'].append(request)
                    return generate(request)
                add('short', prompt('What is 2 + 2? Reply with only the number.'), 16, '4')
                head = add('A', A, 1)
                require(len(head) == 1, 'A must produce one token')
                continuation = A + head + suffix
                add('A-live', continuation, 32, 'AZURE-314159')
                add('A-checkpoint', continuation, 32, 'AZURE-314159')
                add('B', B, 1)
                add('A-return', continuation, 32, 'AZURE-314159')
                add('B-return', B, 1)
                add('A-return-again', continuation, 32, 'AZURE-314159')
                payload = json.dumps(results['requests'], sort_keys=True).encode()
                results['request_digest'] = hashlib.sha256(payload).hexdigest()
                (a.output / 'requests.json').write_bytes(payload)
            else:
                for request in results['requests']:
                    generate(request)
        finally:
            engine.close()
            engine.proc.wait(timeout=30)
            engine.log.close()
        arm['request_digest'] = hashlib.sha256(json.dumps(results['requests'], sort_keys=True).encode()).hexdigest()
        log_text = log.read_text(encoding='utf-8')
        hashes = state_hashes(log_text, candidate=index != 0)
        require(len(hashes) == len(arm['records']), f'{name}: missing state hashes')
        for record, fingerprint in zip(arm['records'], hashes):
            record['state'] = fingerprint
        arm['cache_events'] = [line for line in log_text.splitlines() if any(mark in line for mark in
                               ('conversation cache: parked', 'disk cache: spilled', 'SNAPSHOT_VERIFY'))]
        results['arms'].append(arm)
        (a.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(f'DONE {name}', flush=True)
    verify(results)
    (a.output / 'passed.json').write_text(json.dumps({'passed': True, 'request_digest': results['request_digest']}) + '\n')
    print('PASS: untouched upstream versus disabled engines; identical tokens, main state, reuse and known answers')


if __name__ == '__main__':
    main()
