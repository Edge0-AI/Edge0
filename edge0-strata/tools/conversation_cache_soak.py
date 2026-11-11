"""Private-engine known-answer A/B/A soak; dry-run by default.

Loads paired engines sequentially. Requires an exclusive model/GPU window, not
an existing HTTP server. This is a synthetic cache gate, not a real-agent benchmark.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from conversation_cache_parity import STATE_KEYS, engine_args, load_tokenizer, require, state_hashes
from serve.server import StrataEngine, child_env
from serve.frontend import ChatTemplate


def answer_text(text):
    return text.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip().casefold()


def verify(results):
    contexts, cycles = results['contexts'], results['cycles']
    count = results['requested_cycles']
    require(count >= 30 and len(cycles) == count, 'incomplete 30-cycle soak')
    require(len(contexts) >= 3, 'need at least three context lengths')
    require(len({c['prompt_tokens'] for c in contexts}) == len(contexts), 'context lengths are not distinct')
    baseline, candidate = results['engine_info']['baseline'], results['engine_info']['candidate']
    for key in ('expert_slots', 'kv', 'kv_resident', 'context', 'spec', 'mtp_max', 'lookup', 'cvec'):
        require(baseline[key] == candidate[key], f'engine setting differs: {key}')
    require(baseline['conversation_cache_mib'] == 0 and candidate['conversation_cache_mib'] > 0,
            'invalid baseline/candidate cache configuration')
    lengths = [c['prompt_tokens'] for c in contexts]
    require(candidate['kv_resident'] > 0 and max(lengths) > candidate['kv_resident'], 'no streamed-KV coverage')
    require(max(lengths) >= .9 * candidate['context'], 'longest prompt is not near the context limit')
    budget = candidate['conversation_cache_mib'] * 1024 * 1024
    for c in contexts:
        require(c['baseline']['ids'] and set(STATE_KEYS) <= c['baseline']['state'].keys(), 'incomplete baseline')
        require(c['baseline']['finish'] in ('length', 'stop'), 'baseline did not finish normally')
        require(answer_text(c['baseline']['text']) == c['expected'].casefold(), 'baseline known answer is wrong')
    for i, record in enumerate(cycles):
        require(record['cycle'] == i and record['context'] == i % len(contexts), 'missing or reordered cycle')
        c = contexts[record['context']]
        require(record['finish'] in ('length', 'stop') and record['ids'], 'soak request failed')
        require(answer_text(record['text']) == c['expected'].casefold(), 'restored known answer is wrong')
        require(record['ids'] == c['baseline']['ids'], 'continuation token parity failed')
        require(record['reused'] >= c['prompt_tokens'], 'return did not restore the long prefix')
        require(set(STATE_KEYS) <= record['state'].keys() and record['state'] == c['baseline']['state'],
                'restored main-model state differs')
        require(0 <= record['cache_bytes'] <= budget, 'parked byte budget exceeded')
    # Compare steady-state retained snapshot payload, not allocator RSS or VRAM.
    # Sampled RSS is diagnostic: allocator high-water marks need not shrink,
    # and request-end samples do not establish a transient peak bound.
    for index in range(len(contexts)):
        tail = [r['cache_bytes'] for r in cycles if r['context'] == index][-3:]
        require(len(tail) == 3 and len(set(tail)) == 1, 'retained payload did not reach a stable plateau')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='new directory; existing paths refused')
    ap.add_argument('--prompt-tokens', default='2048,40000,120000', help='three increasing approximate prompt lengths')
    ap.add_argument('--cycles', type=int, default=30)
    ap.add_argument('--cache-mib', type=int, default=8192)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    try:
        lengths = [int(n) for n in args.prompt_tokens.split(',')]
    except ValueError:
        ap.error('prompt-tokens must contain three integers')
    if len(lengths) != 3 or lengths != sorted(set(lengths)) or min(lengths) < 256 or args.cycles < 30 or args.cache_mib <= 0:
        ap.error('need three distinct increasing lengths >= 256, cycles >= 30 and cache-mib > 0')
    if not args.run:
        print(f'Dry run: {args.cycles} known-answer A/B/A cycles; approximate lengths {lengths}.')
        print('No model loaded or files created. Use --run only in an exclusive GPU test window.')
        return
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    path = Path(cfg['tokenizer'])
    tok = load_tokenizer(path)
    template = ChatTemplate(path / 'chat_template.jinja')
    def encode(text):
        return tok.encode(text, parse_special=True)
    def prompt(text):
        return encode(template.render([{'role': 'user', 'content': text}], enable_thinking=False))
    def long_prompt(target, secret, index):
        paragraphs = max(1, target // 17)
        for _ in range(6):
            lines = [f'Record {i}: blue square, green triangle, red circle.' for i in range(paragraphs)]
            lines[paragraphs // 3] = f'The secret access code for conversation {index} is {secret}.'
            ids = prompt(f'Conversation {index}. Remember its secret access code for my next question.\n' + '\n'.join(lines))
            delta = target - len(ids)
            if abs(delta) < 64:
                return ids
            paragraphs = max(1, paragraphs + delta // 17)
        return ids
    secrets = ['MANGO', 'CEDAR', 'RAVEN']
    prompts = [long_prompt(n, secret, i) for i, (n, secret) in enumerate(zip(lengths, secrets))]
    suffix = encode('<|im_end|>\n<|im_start|>user\nWhat is this conversation\'s secret access code? '
                    'Reply with only that one word, no punctuation.<|im_end|>\n'
                    '<|im_start|>assistant\n<think>\n\n</think>\n\n')
    other = prompt('Unrelated worker conversation. Reply with HELLO and nothing else.')
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    env = child_env(cfg)
    env['STRATA_STATE_HASH'] = '1'
    results = {'requested_cycles': args.cycles, 'engine_info': {}, 'contexts': [], 'cycles': []}
    heads, continuations = [], []
    for label, budget in [('baseline', 0), ('candidate', args.cache_mib)]:
        log = args.output / f'{label}.log'
        engine = StrataEngine(str(args.engine.resolve()), engine_args(cfg, budget, 1),
                              cwd=cfg.get('cwd'), log=str(log), env=env)
        results['engine_info'][label] = dict(engine.info)
        hash_count = 0
        def generate(ids, count):
            nonlocal hash_count
            output = [t for t in engine.generate(ids, count, {'temperature': 0}, threading.Event()) if t is not None]
            text = log.read_text(encoding='utf-8')
            hashes = state_hashes(text)
            require(len(hashes) == hash_count + 1 and int(hashes[-1]['L']) >= len(ids), 'missing current state fingerprint')
            hash_count += 1
            state = hashes[-1]
            occupancy = re.findall(r'parked=\d+ bytes=(\d+)', text)
            rss = None
            try:
                match = re.search(r'^VmRSS:\s+(\d+) kB', Path(f'/proc/{engine.proc.pid}/status').read_text(encoding='utf-8'), re.M)
                if match:
                    rss = int(match[1]) * 1024
            except OSError:
                pass
            return {'ids': output, 'text': tok.decode(output), **engine.last, 'state': state,
                    'cache_bytes': int(occupancy[-1]) if occupancy else 0, 'sampled_rss_bytes': rss}
        try:
            require(max(map(len, prompts)) + len(suffix) + 64 < engine.max_context, 'test prompt exceeds context')
            if label == 'baseline':
                for i, ids in enumerate(prompts):
                    head = generate(ids, 1)
                    require(len(head['ids']) == 1, 'baseline initial output is missing')
                    heads.append(head['ids'])
                    continuations.append(ids + head['ids'] + suffix)
                    answer = generate(continuations[-1], 16)
                    results['contexts'].append({'expected': secrets[i], 'prompt_tokens': len(ids), 'baseline': answer})
                    require(answer_text(answer['text']) == secrets[i].casefold(), 'baseline known answer is wrong')
                    print(f'baseline length={len(ids)} answer={answer_text(answer["text"])}', flush=True)
            else:
                primed = set()
                for cycle in range(args.cycles):
                    index = cycle % len(prompts)
                    if index not in primed:
                        head = generate(prompts[index], 1)
                        require(head['ids'] == heads[index], 'initial candidate output differs')
                        primed.add(index)
                    generate(other, 1)
                    answer = generate(continuations[index], 16)
                    results['cycles'].append({'cycle': cycle, 'context': index, **answer})
                    print(f'cycle={cycle + 1}/{args.cycles} length={len(prompts[index])} '
                          f'reused={answer["reused"]} answer={answer_text(answer["text"])}', flush=True)
        finally:
            engine.close()
            # Keep partial evidence on failure; verify rejects incomplete runs.
            (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    verify(results)
    print('PASS: 30+ known-answer A/B/A cycles, streamed/near-limit KV, exact state/output, stable retained payload')


if __name__ == '__main__':
    main()
