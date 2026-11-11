"""Compare two runs of Strata's tools/strata_bench.py (suite 1, results.json).

Usage: compare_bench.py <baseline results.json or directory> <candidate results.json or directory>
       [--ttft-limit-ms 50] [--ttft-limit-share-pct 5] [--engine-limit-pct 5] [--markdown]

Engine-timed items (medians over the suite's runs): short-chat decode, 4K and 16K prefill, and the
agent session's follow-up decode and prefill. Each must differ from the baseline by at most
--engine-limit-pct percent, in either direction. Client-measured first-token medians (short chat, 4K,
16K, agent follow-up turns) may grow by at most the larger of --ttft-limit-ms milliseconds and
--ttft-limit-share-pct percent of the baseline value (each row prints the allowed increase); with --ttft-limit-pct
(an A/A run of the same server twice) they are judged like the engine items instead, by relative
difference in either direction.

Prints one line per item and a verdict; exits 0 when every item is within its limit, 1 otherwise.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys


def load(path):
    path = Path(path)
    if path.is_dir():
        found = sorted(path.rglob('results.json'))
        if len(found) != 1:
            sys.exit(f'{path}: expected exactly one results.json, found {len(found)}')
        path = found[0]
    return json.loads(path.read_text())


def median(values, name):
    values = [v for v in values if v is not None]
    if not values:
        raise SystemExit(f'no values for {name}')
    return statistics.median(values)


def items(data):
    r = data['results']
    ok = lambda requests: [q for q in requests if q.get('ok')]  # noqa: E731
    chat = ok(r['chat']['requests'])
    prompts = {size: ok(r['prompts'][size]['requests']) for size in ('4K', '16K')}
    follow = [q for session in r['agent']['sessions'] for q in ok(session) if q.get('turn', 0) > 0]
    engine = {
        'chat decode tok/s': median([q['decode_tps'] for q in chat], 'chat decode'),
        '4K prefill tok/s': median([q['prompt_tps'] for q in prompts['4K']], '4K prefill'),
        '16K prefill tok/s': median([q['prompt_tps'] for q in prompts['16K']], '16K prefill'),
        'agent follow-up decode tok/s': median([q['decode_tps'] for q in follow], 'agent decode'),
        'agent follow-up prefill tok/s': median([q['prompt_tps'] for q in follow], 'agent prefill'),
    }
    ttft = {
        'chat first token s': median([q['ttft_s'] for q in chat], 'chat ttft'),
        '4K first token s': median([q['ttft_s'] for q in prompts['4K']], '4K ttft'),
        '16K first token s': median([q['ttft_s'] for q in prompts['16K']], '16K ttft'),
        'agent follow-up first token s': median([q['ttft_s'] for q in follow], 'agent ttft'),
    }
    return engine, ttft


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('baseline')
    parser.add_argument('candidate')
    parser.add_argument('--engine-limit-pct', type=float, default=5.0)
    parser.add_argument('--ttft-limit-ms', type=float, default=50.0)
    parser.add_argument('--ttft-limit-share-pct', type=float, default=5.0,
                        help='first-token increase may also reach this share of the baseline, whichever is larger')
    parser.add_argument('--ttft-limit-pct', type=float, help='judge first-token times by relative difference instead')
    parser.add_argument('--markdown', action='store_true', help='print a Markdown table')
    args = parser.parse_args()
    base_engine, base_ttft = items(load(args.baseline))
    cand_engine, cand_ttft = items(load(args.candidate))
    rows, passed = [], True
    for name, a in base_engine.items():
        b = cand_engine[name]
        pct = (b - a) / a * 100
        ok = abs(pct) <= args.engine_limit_pct
        passed &= ok
        rows.append((name, f'{a:.1f}', f'{b:.1f}', f'{pct:+.1f} %', f'|diff| <= {args.engine_limit_pct:g} %', ok))
    for name, a in base_ttft.items():
        b = cand_ttft[name]
        ms = (b - a) * 1000
        if args.ttft_limit_pct is not None:
            pct = (b - a) / a * 100
            ok = abs(pct) <= args.ttft_limit_pct
            rows.append((name, f'{a:.3f}', f'{b:.3f}', f'{ms:+.0f} ms ({pct:+.1f} %)', f'|diff| <= {args.ttft_limit_pct:g} %', ok))
        else:
            allowed = max(args.ttft_limit_ms, a * 1000 * args.ttft_limit_share_pct / 100)
            ok = ms <= allowed
            rows.append((name, f'{a:.3f}', f'{b:.3f}', f'{ms:+.0f} ms', f'increase <= {allowed:.0f} ms', ok))
        passed &= ok
    if args.markdown:
        print('| Item | Baseline | Candidate | Difference | Limit | Within |')
        print('| --- | ---: | ---: | ---: | --- | --- |')
        for name, a, b, diff, limit, ok in rows:
            print(f'| {name} | {a} | {b} | {diff} | {limit} | {"yes" if ok else "NO"} |')
    else:
        for name, a, b, diff, limit, ok in rows:
            print(f'{"PASS" if ok else "FAIL"} {name}: {a} -> {b} ({diff}; {limit})')
    print('VERDICT', 'PASS' if passed else 'FAIL')
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
