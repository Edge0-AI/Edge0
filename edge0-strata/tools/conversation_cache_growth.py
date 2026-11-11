"""Compare full and incremental snapshot capture during growth and rewind.

Private engines run sequentially; dry-run is the default. Both arms enable the
same conversation cache. The reference forces full capture for every parking.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from conversation_cache_parity import STATE_KEYS, load_tokenizer, require, state_hashes
from conversation_cache_disabled import SETTINGS, common_args
from conversation_cache_soak import answer_text
from serve.frontend import ChatTemplate
from serve.server import StrataEngine, child_env

NAMES = ['A', 'B-0', 'A-grow-0', 'B-1', 'A-grow-1', 'B-2', 'A-grow-2',
         'B-rewind', 'A-rewind', 'B-branch', 'A-branch', 'B-final']
SECRET = 'SAPPHIRE'


def verify(results):
    full, incremental = results['arms']
    require(full['name'] == 'full' and incremental['name'] == 'incremental', 'incorrect comparison arms')
    requests = results['requests']
    require([r['name'] for r in requests] == NAMES, 'incomplete growth/rewind requests')
    require(all(r['ids'] and r['max_new'] > 0 for r in requests), 'empty growth request')
    digest = hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()
    require(digest == results['request_digest'], 'request evidence changed')
    for arm in (full, incremental):
        require(arm['request_digest'] == digest and arm['args'] == full['args'], 'inputs or arguments differ')
        require(arm['exe_sha256'] == full['exe_sha256'], 'executable changed between comparison arms')
        require([r['name'] for r in arm['records']] == NAMES, 'incomplete growth/rewind outputs')
        for key in (*SETTINGS, 'conversation_cache_mib', 'conversation_cache_slots'):
            require(key in arm['info'] and arm['info'][key] == full['info'][key], f'engine setting differs: {key}')
        require(arm['info']['conversation_cache_mib'] == results['cache_mib'] > 0, 'cache budget differs')
        budget = arm['info']['conversation_cache_mib'] * 1024 * 1024
        require(arm['parks'] and all(0 < p['snapshot_bytes'] <= p['bytes'] <= budget and
                0 <= p['reused_kv_bytes'] <= p['snapshot_bytes'] for p in arm['parks']), 'invalid allocation/reuse evidence')
        require(arm['draft_verifications'] >= 4, 'missing restored draft read-back')
        for request, reference, record in zip(requests, full['records'], arm['records']):
            require(record['ids'] and record['finish'] in ('stop', 'length'), 'incomplete generation')
            require(record['ids'] == reference['ids'], f'{record["name"]}: output differs')
            require(set(STATE_KEYS) <= record['state'].keys() and record['state'] == reference['state'],
                    f'{record["name"]}: state differs or is incomplete')
            require(record['reused'] == reference['reused'], 'prefix reuse differs')
            require(record['prompt_tokens'] == len(request['ids']), 'prompt length differs')
            if record['name'].startswith('A-'):
                require(answer_text(record['text']) == SECRET.casefold(), 'known answer differs')
                require(record['reused'] > 0, 'conversation was not restored')
    require(all(p['reused_kv_bytes'] == 0 for p in full['parks']), 'reference did not force full captures')
    require(sum(p['reused_kv_bytes'] > 0 for p in incremental['parks']) >= 3, 'repeated parking did not reuse KV bytes')
    sizes = {r['name']: len(r['ids']) for r in requests}
    require(sizes['A'] < sizes['A-grow-0'] < sizes['A-grow-1'] < sizes['A-grow-2'], 'conversation did not grow')
    require(sizes['A-rewind'] == sizes['A-grow-0'] < sizes['A-grow-2'], 'missing rewind')
    require(sizes['A-branch'] > sizes['A-rewind'], 'missing growth after rewind')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='new output directory')
    ap.add_argument('--paragraphs', type=int, default=128)
    ap.add_argument('--spec', type=int, choices=(1, 4), default=1)
    ap.add_argument('--cache-mib', type=int, default=8192)
    ap.add_argument('--run', action='store_true')
    a = ap.parse_args()
    if a.paragraphs < 1 or a.cache_mib < 1:
        ap.error('paragraphs and cache-mib must be positive')
    if not a.run:
        print('Dry run: paired full/incremental captures; growth, interruption, rewind and branch; no model loaded.')
        return
    cfg = json.loads(a.config.read_text(encoding='utf-8'))
    tok_path = Path(cfg['tokenizer'])
    tok = load_tokenizer(tok_path)
    template = ChatTemplate(tok_path / 'chat_template.jinja')
    def encode(text):
        return tok.encode(text, parse_special=True)
    def prompt(text):
        return encode(template.render([{'role': 'user', 'content': text}], enable_thinking=False))
    text = f'Remember the secret code {SECRET}.\n'
    text += '\n'.join(f'Record {i}: blue square, green triangle, red circle.' for i in range(a.paragraphs))
    initial = prompt(text + f'\nThe secret code is {SECRET}. Reply OK.')
    other = prompt('Unrelated worker. Reply HELLO.')
    end = encode('<|im_end|>')
    def continuation(ids, output, step):
        close = [] if output[-len(end):] == end else end
        filler = '\n'.join(f'Additional record {step}-{i}: blue square and green triangle.' for i in range(24))
        return ids + output + close + encode('\n<|im_start|>user\n' + filler +
            '\nWhat is the secret code? Reply with only that word.<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n')
    args = common_args(cfg, a.spec) + ['--conversation-cache-mib', str(a.cache_mib), '--conversation-cache-slots', '4']
    env = child_env(cfg)
    env['STRATA_STATE_HASH'] = '1'
    env['STRATA_SNAPSHOT_VERIFY'] = '1'
    env['STRATA_MTP_BATCH'] = '1'
    a.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    results = {'requests': [], 'arms': [], 'spec': a.spec, 'paragraphs': a.paragraphs, 'cache_mib': a.cache_mib}
    for name in ('full', 'incremental'):
        current_env = dict(env)
        if name == 'full':
            current_env['STRATA_SNAPSHOT_FULL_CAPTURE'] = '1'
        else:
            current_env.pop('STRATA_SNAPSHOT_FULL_CAPTURE', None)
        print('START', name, flush=True)
        log = a.output / (name + '.log')
        engine = StrataEngine(str(a.engine.resolve()), args, cwd=cfg.get('cwd'), log=str(log), env=current_env)
        arm = {'name': name, 'args': args, 'info': dict(engine.info), 'records': []}
        with a.engine.open('rb') as source:
            arm['exe_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
        def generate(request):
            ids = [x for x in engine.generate(request['ids'], request['max_new'], {'temperature': 0},
                                               threading.Event()) if x is not None]
            arm['records'].append({'name': request['name'], 'ids': ids, 'text': tok.decode(ids), **engine.last})
            return ids
        try:
            if name == 'full':
                def add(label, ids, count, expected=False):
                    request = {'name': label, 'ids': ids, 'max_new': count, 'expected': expected}
                    results['requests'].append(request)
                    return generate(request)
                ids = initial
                output = add('A', ids, 1)
                first_growth = None
                for step in range(3):
                    add(f'B-{step}', other, 1)
                    ids = continuation(ids, output, step)
                    if step == 0:
                        first_growth = list(ids)
                    output = add(f'A-grow-{step}', ids, 16, True)
                add('B-rewind', other, 1)
                output = add('A-rewind', first_growth, 16, True)
                add('B-branch', other, 1)
                add('A-branch', continuation(first_growth, output, 'branch'), 16, True)
                add('B-final', other, 1)
                results['request_digest'] = hashlib.sha256(json.dumps(results['requests'], sort_keys=True).encode()).hexdigest()
            else:
                for request in results['requests']:
                    generate(request)
        finally:
            engine.close()
            engine.proc.wait(timeout=30)
            engine.log.close()
        log_text = log.read_text(encoding='utf-8')
        hashes = state_hashes(log_text)
        require(len(hashes) == len(arm['records']), 'missing state fingerprints')
        for record, state in zip(arm['records'], hashes):
            record['state'] = state
        parks = re.findall(r'conversation cache: parked (\d+) tokens in ([\d.]+) ms; parked=\d+ bytes=(\d+) '
                           r'evictions=\d+ snapshot_bytes=(\d+) reused_kv_bytes=(\d+)', log_text)
        arm['parks'] = [dict(tokens=int(p[0]), ms=float(p[1]), bytes=int(p[2]), snapshot_bytes=int(p[3]),
                             reused_kv_bytes=int(p[4])) for p in parks]
        arm['draft_verifications'] = log_text.count('SNAPSHOT_VERIFY draft=')
        arm['request_digest'] = hashlib.sha256(json.dumps(results['requests'], sort_keys=True).encode()).hexdigest()
        results['arms'].append(arm)
        (a.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print('DONE', name, flush=True)
    verify(results)
    (a.output / 'passed.json').write_text(json.dumps({'passed': True, 'request_digest': results['request_digest']}) + '\n')
    print('PASS: growth and rewind preserve output/state; repeated captures reuse unchanged KV bytes')


if __name__ == '__main__':
    main()
