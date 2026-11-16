"""#528: decode speed after a conversation-cache restore, against the same turns kept live.

Two private engines run one after the other (never an existing server):
  baseline   --conversation-cache-mib 0: conversation A, turn after turn, continues live
  candidate  --conversation-cache-mib N: between A's turns an unrelated conversation B parks A, so every later
             turn of A is restored from its parked snapshot
Each A turn's greedy output must be the same token for token in both, and its decode speed should be too (a
restored conversation decodes like a live one). --mode same: one conversation whose client re-renders each reply
(no parking needed); --mode pair: two long conversations take turns (a snapshot restored every turn). On a PC with
little RAM the cache's physical-RAM floor may refuse to park (the log says "skip parking"): that turn reads its
prompt again and is reported as not restored. Dry-run by default; --run needs the GPU (and its lock).

Measured on the RTX 5070 (0.1.38, #528): Q2_0 37K tokens --kv-resident 16384, IQ3_XXS 50-75K tokens
--kv-resident 32768 (switch, same and pair modes): restored turns decode at the live turns' speed, same tokens.

  python tools/conversation_restore_speed.py --config cfg.json --engine build-release/strata.exe \
      --output out-dir --tokens 40000 --set=--max-context=65536 --set=--kv-resident=16384 --run
"""
import argparse
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]


def override(args, pairs):
    """cfg args with each `--flag value` of pairs replacing the config's own (or appended)."""
    out = list(args)
    for flag, value in pairs:
        if flag in out:
            i = out.index(flag)
            if value is None:
                del out[i:i + 2]
            else:
                out[i + 1] = value
        elif value is not None:
            out += [flag, value]
    return out


def parse_sets(items):
    pairs = []
    for item in items or []:
        flag, _, value = item.partition('=')
        pairs.append((flag, value if value != '' else None))
    return pairs


def summarize(records):
    rows = []
    for r in records:
        tok_s = r['generated'] / (r['decode_ms'] / 1000) if r.get('decode_ms') else 0.0
        acc = f"{r.get('drafts_accepted', 0)}/{r.get('drafts_offered', 0)}"
        rows.append(f"  {r['name']:<8} reused={r.get('reused', 0):>6} gen={r['generated']:>4} "
                    f"decode={tok_s:6.1f} tok/s drafts={acc:<9} hits={r.get('hits', 0)}/{r.get('lookups', 0)}")
    return '\n'.join(rows)


def compare(results, tolerance):
    """The A turns: identical ids, restored (reused > 0) in the candidate, and decode within `tolerance`."""
    base = {r['name']: r for r in results['baseline']}
    problems = []
    for r in results['candidate']:
        if not r['name'].startswith('A'):
            continue
        b = base.get(r['name'])
        if b is None:
            problems.append(f"{r['name']}: no baseline turn")
            continue
        if r['ids'] != b['ids']:
            problems.append(f"{r['name']}: output differs from the live baseline")
        if r['name'] != 'A1' and not r.get('reused'):
            problems.append(f"{r['name']}: not restored (reused=0)")
        speed = lambda x: x['generated'] / (x['decode_ms'] / 1000) if x.get('decode_ms') else 0.0
        if r['name'] != 'A1' and speed(r) < (1 - tolerance) * speed(b):
            problems.append(f"{r['name']}: restored decode {speed(r):.1f} tok/s vs live {speed(b):.1f}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='new directory; an existing path is refused')
    ap.add_argument('--tokens', type=int, default=20000, help='about this many prompt tokens in A')
    ap.add_argument('--gen', type=int, default=160, help='tokens generated per A turn')
    ap.add_argument('--turns', type=int, default=3, help='A turns (the first is read fresh)')
    ap.add_argument('--cache-mib', type=int, default=8192)
    ap.add_argument('--set', action='append', metavar='FLAG=VALUE',
                    help='override an engine flag of the config (FLAG= drops it); repeatable')
    ap.add_argument('--tolerance', type=float, default=0.15, help='allowed decode shortfall of a restored turn')
    ap.add_argument('--only', choices=('baseline', 'candidate'), help='run one side only (no comparison)')
    ap.add_argument('--mode', choices=('switch', 'same', 'pair'), default='switch',
                    help='switch: B parks A between turns; same: one conversation whose client drops the last '
                         '--trim tokens of each reply (re-rendered history), as agent clients do; pair: two long '
                         'conversations A and C take turns (each turn restores a big snapshot, the cache holds two)')
    ap.add_argument('--trim', type=int, default=8)
    ap.add_argument('--run', action='store_true')
    a = ap.parse_args()
    if not a.run:
        print('Dry run: baseline (live turns) vs candidate (every turn after the first restored from a parked '
              'snapshot); greedy, fixed residency. Use --run with the GPU free.')
        return
    from serve.server import StrataEngine, child_env
    from serve.frontend import ChatTemplate
    from conversation_cache_parity import load_tokenizer
    cfg = json.loads(a.config.read_text(encoding='utf-8'))
    p = Path(cfg['tokenizer'])
    tok = load_tokenizer(p)
    tpl = ChatTemplate(p / 'chat_template.jinja')
    enc = lambda text: tok.encode(text, parse_special=True)
    colors = ('blue', 'green', 'red', 'amber', 'violet', 'teal', 'grey', 'white')
    shapes = ('square', 'triangle', 'circle', 'hexagon', 'star', 'ring')

    def doc(label, n):
        lines = [f'{label}: here is an inventory list to remember.']
        i = 0
        while len(lines) * 15 < n:
            lines.append(f'Record {i}: a {colors[i % 8]} {shapes[(i * 7) % 6]} stored in bin {(i * 37) % 997}.')
            i += 1
        return '\n'.join(lines)

    ask = 'Write a long, detailed story that uses as many of the records above as you can.'
    A = enc(tpl.render([{'role': 'user', 'content': doc('Conversation A', a.tokens) + '\n' + ask}],
                       enable_thinking=False))
    B = enc(tpl.render([{'role': 'user', 'content': doc('Unrelated conversation B', 2000) + '\nName one bin.'}],
                       enable_thinking=False))
    follow = [enc('<|im_end|>\n<|im_start|>user\nContinue the story, with new records.<|im_end|>\n'
                  '<|im_start|>assistant\n<think>\n\n</think>\n\n')]
    a.output.mkdir(parents=False, exist_ok=False)
    env = child_env(cfg)
    base_args = override(cfg['args'], parse_sets(a.set) + [('--adapt-swaps', '0'), ('--prompt-cache', '6')])
    results = {'prompt_tokens': len(A), 'args': base_args}
    sides = [('baseline', 0), ('candidate', a.cache_mib)]
    if a.only:
        sides = [s for s in sides if s[0] == a.only]
    for label, budget in sides:
        args = base_args + ['--conversation-cache-mib', str(budget), '--conversation-cache-slots', '4']
        engine = StrataEngine(str(a.engine.resolve()), args, cwd=cfg.get('cwd'), log=str(a.output / f'{label}.log'),
                              env=env)
        records = []

        def generate(ids, count, name):
            out = [t for t in engine.generate(ids, count, {'temperature': 0}, threading.Event()) if t is not None]
            records.append({'name': name, 'ids': out, **engine.last})
            print(f'{label} {summarize(records[-1:]).strip()}', flush=True)
            return out

        try:
            convo = list(A)
            other = enc(tpl.render([{'role': 'user', 'content': doc('Second long conversation C', a.tokens) + '\n' +
                                     ask}], enable_thinking=False)) if a.mode == 'pair' else None
            for turn in range(1, a.turns + 1):
                if other is not None and budget:   # C's turn first: A comes back from its snapshot after it
                    out_c = generate(other, a.gen if turn == 1 else 32, f'C{turn}')
                    other = other + out_c + follow[0]
                if turn > 1:
                    if budget and a.mode == 'switch':
                        generate(B, 1, f'B{turn}')   # parks A: this turn of A comes back from the snapshot
                    convo += follow[0]
                out = generate(convo, a.gen, f'A{turn}')
                convo += out[:-a.trim] if a.mode == 'same' and len(out) > a.trim else out
        finally:
            engine.close()
        results[label] = records
    (a.output / 'results.json').write_text(json.dumps(results, indent=1) + '\n')
    for label, _ in sides:
        print(f'{label}:\n{summarize(results[label])}')
    if a.only:
        return
    problems = compare(results, a.tolerance)
    for line in problems:
        print('FAIL ' + line)
    print('PASS: restored turns decode like live ones, same output' if not problems else 'FAILED')
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
