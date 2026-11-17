"""Full-model A/B/A parity gate. Loads private engines sequentially, never uses an existing server.

Dry-run by default. --run requires an available GPU/model-loading window.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from serve.server import StrataEngine, child_env
from serve.frontend import ChatTemplate
import strata_tokenizer as ST

# pooled: the completed indexer rows (the 0.1.29 extent); pooled_full: those plus the spare row
STATE_KEYS = ('L', 'gdn', 'ple', 'tail', 'dead', 'pooled', 'pooled_full', 'kv', 'ple_prev')

def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_tokenizer(path):
    vocab = json.loads((path / 'vocab.json').read_text(encoding='utf-8'))
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    return ST.Tokenizer(tokens, (path / 'merges.txt').read_text(encoding='utf-8').split('\n'),
                        json.loads((path / 'token_type.json').read_text(encoding='utf-8')))


def state_hashes(text):
    # Padding and the drafter's final uncomputed cell aren't main-model state.
    hashes = []
    for line in text.splitlines():
        if 'STATE_HASH L=' in line:
            fields = dict(re.findall(r'(\w+)=([0-9a-f,-]+)', line))
            hashes.append({key: fields[key] for key in STATE_KEYS})
    return hashes


def engine_args(cfg, budget, spec):
    # Native IQ packs require verifier capacity >= 2, even for one-token decode.
    return list(cfg['args']) + [
        '--conversation-cache-mib', str(budget), '--conversation-cache-slots', '4',
        '--prompt-cache', '6', '--adapt-swaps', '0', '--spec', str(max(2, spec)),
        '--mtp-max-t', str(spec), '--suffix-draft', '0', '--spec-min-p', '0']


def pressure_budget_hint(results, budget_mib):
    sizes = [p['snapshot_bytes'] for p in results['pressure']['parks'] if p.get('snapshot_bytes', 0) > 0]
    message = f'Configured --cache-mib {budget_mib}.'
    if not sizes:
        return message + ' No snapshot sizes recorded; use an engine that reports snapshot_bytes.'
    mib = 1024 * 1024
    message += ' Observed snapshots: ' + ', '.join(f'{n / mib:.2f} MiB ({n} bytes)' for n in sizes) + '.'
    if len(sizes) < 2:
        return message + ' Need at least two captured snapshots to determine a pressure budget.'
    # Fit every individual image but no adjacent pair in the A/B/C sequence.
    low = (max(sizes) + mib - 1) // mib
    high = (min(a + b for a, b in zip(sizes, sizes[1:])) - 1) // mib
    if low <= high:
        message += f' Use --cache-mib {low}..{high} to fit each snapshot but not two adjacent snapshots.'
    else:
        message += ' No whole-MiB budget fits every snapshot while excluding each adjacent pair; use similarly sized prompts.'
    return message


def verify_pressure(results, budget_mib, oversized=False):
    """Require actual byte-pressure evidence, not just correct output on misses."""
    baseline, candidate = results['baseline'], results['candidate']
    names = ['A', 'B', 'C', 'A-again']
    for records in (baseline, candidate):
        require([r['name'] for r in records] == names, 'incomplete pressure sequence')
    for before, after in zip(baseline, candidate):
        require(len(before['ids']) == len(after['ids']) == 1, 'missing pressure output')
        require(before['finish'] in ('length', 'stop') and after['finish'] in ('length', 'stop'),
                'pressure request did not finish normally')
        require(before['reused'] == after['reused'] == 0,
                'pressure did not force a cache miss; this run did not exercise the required fallback. ' +
                pressure_budget_hint(results, budget_mib))
        require(before['ids'] == after['ids'], 'pressure output differs')
        keys = set(STATE_KEYS)
        require(keys <= before['state'].keys() and keys <= after['state'].keys(), 'missing pressure state')
        require(before['state'] == after['state'], 'pressure state differs')
    for key in ('expert_slots', 'kv', 'kv_resident', 'context', 'spec', 'mtp_max', 'lookup', 'cvec'):
        require(results['engine_info']['baseline'][key] == results['engine_info']['candidate'][key],
                f'pressure engine setting differs: {key}')
    info = results['engine_info']['candidate']
    require(info['conversation_cache_mib'] == budget_mib and info['conversation_cache_slots'] == 4,
            'pressure cache configuration differs')
    evidence = results['pressure']
    parks = evidence['parks']
    require(all(p['bytes'] <= budget_mib * 1024 * 1024 for p in parks), 'byte budget exceeded')
    if oversized:
        require(evidence['skips'] == 3 and not parks, 'oversized snapshots were not all skipped')
    else:
        require(evidence['skips'] == 0 and len(parks) == 3, 'snapshots must fit individually')
        require(all(0 < p['parked'] < 4 for p in parks), 'slot pressure confounds byte-pressure test')
        require(parks[-1]['evictions'] > 0, 'no byte-pressure eviction observed. ' +
                pressure_budget_hint(results, budget_mib))


def verify_results(results, prompt_tokens, spec):
    """Fail closed on incomplete evidence, even under python -O."""
    baseline, candidate = results['baseline'], results['candidate']
    require([r['name'] for r in baseline] == ['A', 'A+'], 'incomplete baseline')
    require([r['name'] for r in candidate] == ['A', 'B', 'A+', 'B-again', 'A+-checkpoint'], 'incomplete candidate')
    state_keys = set(STATE_KEYS)
    for record in baseline + candidate:
        require(bool(record['ids']), 'missing generated tokens')
        require(record['finish'] in ('length', 'stop'), 'request did not finish normally')
        require(state_keys <= record['state'].keys(), 'incomplete state fingerprint')
    require(len(baseline[0]['ids']) == len(candidate[0]['ids']) == 1, 'invalid initial A output length')
    require(baseline[0]['ids'] == candidate[0]['ids'], 'initial A output differs')
    require(candidate[2]['reused'] >= prompt_tokens, 'A was not restored after B')
    require(candidate[4]['reused'] > 0, 'parked checkpoint was not restored')
    for key in ('expert_slots', 'kv', 'kv_resident', 'context', 'spec', 'mtp_max', 'lookup', 'cvec'):
        require(results['engine_info']['baseline'][key] == results['engine_info']['candidate'][key],
                f'baseline/candidate engine setting differs: {key}')
    for record in (candidate[2], candidate[4]):
        require(record['ids'] == baseline[1]['ids'], 'restored continuation output differs')
        if spec == 1:
            require(record['state'] == baseline[1]['state'], 'restored main-model state differs')


def verify_exchange(results, prompt_tokens, spec, budget_mib):
    verify_results(results, prompt_tokens, spec)
    require(results['engine_info']['candidate']['conversation_cache_mib'] == budget_mib,
            'exchange cache budget differs')
    evidence = results['pressure']
    require(evidence['skips'] == 2 and len(evidence['parks']) == 2,
            'expected two parks and two outgoing skips while an incoming snapshot is held')
    require(all(p['parked'] == 1 and 0 < p['bytes'] <= budget_mib * 1024 * 1024
                for p in evidence['parks']), 'invalid exchange snapshot budget')


def verify_admission(results, budget_mib, floor_mib):
    evidence = results['pressure']
    require(evidence['memory_skips'] == 3 and evidence['skips'] == 0 and not evidence['parks'],
            'expected physical-memory denial, not oversized-budget fallback')
    require(results['engine_info']['candidate']['conversation_cache_min_free_mib'] == floor_mib,
            'physical RAM floor differs')
    # The same four cold requests must preserve output and complete state. Reuse
    # the oversized gate's no-parking assertions after checking the actual cause.
    equivalent = {**results, 'pressure': {**evidence, 'skips': evidence['memory_skips']}}
    verify_pressure(equivalent, budget_mib, oversized=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='new private directory; existing paths refused')
    ap.add_argument('--cache-mib', type=int, default=8192)
    ap.add_argument('--paragraphs', type=int, default=128)
    ap.add_argument('--scenario', choices=('reuse', 'pressure', 'oversized', 'exchange', 'admission'), default='reuse',
                    help='pressure requires snapshots fitting individually but not together; oversized requires none to fit')
    ap.add_argument('--min-free-mib', type=int, default=2560,
                    help='physical RAM floor; for admission denial choose a value above available system RAM')
    ap.add_argument('--spec', type=int, default=1, choices=range(1, 9),
                    help='decode window cap; 1 uses engine --spec 2 --mtp-max-t 1 for native IQ packs')
    ap.add_argument('--run', action='store_true')
    a = ap.parse_args()
    if a.cache_mib <= 0 or a.paragraphs < 1 or not 0 <= a.min_free_mib <= (2**63 - 1) // (1024 * 1024):
        ap.error('cache-mib/paragraphs must be positive and min-free-mib must fit the engine range')
    if not a.run:
        print(f'Dry run: {a.scenario}; paired baseline/candidate; fixed residency, greedy output.')
        print('No model loaded. Use --run only with a separately available GPU/test window.')
        return
    cfg = json.loads(a.config.read_text(encoding='utf-8'))
    p = Path(cfg['tokenizer'])
    tok = load_tokenizer(p)
    tpl = ChatTemplate(p / 'chat_template.jinja')
    def encode(text):
        return tok.encode(text, parse_special=True)
    def prompt(label):
        text = label + ': remember this list.\n' + '\n'.join(
            f'Record {i}: blue square, green triangle, red circle.' for i in range(a.paragraphs))
        return encode(tpl.render([{'role': 'user', 'content': text}], enable_thinking=False))
    A, B, C = prompt('Conversation A'), prompt('Unrelated conversation B'), prompt('Distinct conversation C')
    suffix = encode('<|im_end|>\n<|im_start|>user\nName a color from the list.<|im_end|>\n'
                    '<|im_start|>assistant\n<think>\n\n</think>\n\n')
    a.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    env = child_env(cfg)
    env['STRATA_STATE_HASH'] = '1'
    results = {'engine_info': {}}
    for label, budget in [('baseline', 0), ('candidate', a.cache_mib)]:
        log = a.output / f'{label}.log'
        args = engine_args(cfg, budget, a.spec)
        args += ['--conversation-cache-min-free-mib', str(a.min_free_mib)]
        engine = StrataEngine(str(a.engine.resolve()), args, cwd=cfg.get('cwd'), log=str(log), env=env)
        results['engine_info'][label] = dict(engine.info)
        records = []
        def generate(ids, count, name):
            out = [t for t in engine.generate(ids, count, {'temperature': 0}, threading.Event()) if t is not None]
            records.append({'name': name, 'ids': out, **engine.last})
            return out
        try:
            # One output token leaves exactly A's prompt as the live prefix,
            # avoiding the pre-existing accepted-draft output-cap overshoot.
            if a.scenario in ('pressure', 'oversized', 'admission'):
                generate(A, 1, 'A')
                generate(B, 1, 'B')
                generate(C, 1, 'C')
                generate(A, 1, 'A-again')
            else:
                head = generate(A, 1, 'A')
                continuation = A + head + suffix
                if budget:
                    generate(B, 1, 'B')
                generate(continuation, 8, 'A+')
                if budget:
                    generate(B, 1, 'B-again')
                    generate(continuation, 8, 'A+-checkpoint')
        finally:
            engine.close()
        hashes = state_hashes(log.read_text(encoding='utf-8'))
        require(len(hashes) == len(records), 'missing state hashes')
        for record, fingerprint in zip(records, hashes):
            record['state'] = fingerprint
        results[label] = records
        if label == 'candidate' and a.scenario != 'reuse':
            log_text = log.read_text(encoding='utf-8')
            parks = re.findall(r'conversation cache: parked \d+ tokens .*?parked=(\d+) bytes=(\d+) evictions=(\d+)(?: snapshot_bytes=(\d+))?', log_text)
            results['pressure'] = {
                'parks': [dict(zip(('parked', 'bytes', 'evictions', 'snapshot_bytes'),
                                  (int(value) if value else 0 for value in p))) for p in parks],
                'skips': log_text.count('conversation cache: skip parking (snapshot '),
                'memory_skips': log_text.count('conversation cache: skip parking (physical RAM admission;')}
    (a.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    if a.scenario == 'pressure':
        print(pressure_budget_hint(results, a.cache_mib), flush=True)
    if a.scenario == 'admission':
        verify_admission(results, a.cache_mib, a.min_free_mib)
        print('PASS: physical-memory admission denial, output and byte-exact main-model state')
    elif a.scenario == 'exchange':
        verify_exchange(results, len(A), a.spec, a.cache_mib)
        print('PASS: bounded incoming/outgoing exchange, A/B/A output and checkpoint reuse')
    elif a.scenario == 'reuse':
        verify_results(results, len(A), a.spec)
        print('PASS: A/B/A output and checkpoint reuse' + (', byte-exact main-model state' if a.spec == 1 else ''))
    else:
        verify_pressure(results, a.cache_mib, a.scenario == 'oversized')
        print(f'PASS: {a.scenario} fallback, output and byte-exact main-model state')
    print(f'Results: {a.output / "results.json"}')


if __name__ == '__main__':
    main()
