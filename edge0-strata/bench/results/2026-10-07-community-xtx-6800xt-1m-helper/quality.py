#!/usr/bin/env python3
"""Paired multiple-choice quality check through the chat API (no logprobs needed, so it runs on any
OpenAI-compatible server): the first N HellaSwag and Winogrande items, reasoning off, greedy, answer = one letter.
Run it on two engines serving the same weights, then compare with --compare A_DIR B_DIR.

Usage: strata-quality-eval.py LABEL OUTDIR BASE_URL [--conc 1] [--hs 400] [--wg 400]
       strata-quality-eval.py --compare DIR_A DIR_B
"""
import argparse, concurrent.futures as cf, json, math, os, re, sys, time, urllib.request

HS = 'hellaswag_val_full.txt'
WG = 'winogrande-debiased-eval.csv'


def hs_items(n):
    L = open(HS, encoding='utf-8').read().split('\n')
    out = []
    for i in range(n):
        ctx, lab, *ends = L[6 * i:6 * i + 6]
        q = (f'Which ending is the most plausible continuation of the text?\n\nText: {ctx}\n\n' +
             '\n'.join(f'{"ABCD"[k]}) {e}' for k, e in enumerate(ends)) +
             '\n\nAnswer with the letter only.')
        out.append({'id': f'hs{i}', 'prompt': q, 'gold': 'ABCD'[int(lab)], 'choices': 'ABCD'})
    return out


def wg_items(n):
    import csv
    out = []
    with open(WG, encoding='utf-8') as f:
        for i, row in enumerate(csv.reader(f)):
            if i >= n: break
            _, sent, o1, o2, ans = row
            q = (f'Which option correctly fills the blank (_)?\n\nSentence: {sent}\n\nA) {o1}\nB) {o2}\n\n'
                 'Answer with the letter only.')
            out.append({'id': f'wg{i}', 'prompt': q, 'gold': 'AB'[int(ans) - 1], 'choices': 'AB'})
    return out


def ask(base, it):
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': it['prompt']}], 'max_tokens': 32,
            'temperature': 0.0, 'top_k': 1, 'top_p': 1.0, 'reasoning_effort': 'none'}
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                j = json.loads(r.read())
            break
        except Exception as e:
            if attempt == 2: return {**it, 'answer': None, 'error': str(e)}
            time.sleep(5)
    msg = j['choices'][0]['message']
    text = (msg.get('content') or '').strip()
    m = re.search(r'\b([%s])\b' % it['choices'], text) or re.search(r'([%s])' % it['choices'], text)
    if not m:   # thinking leaked into reasoning_content only
        m = re.search(r'\b([%s])\b' % it['choices'], (msg.get('reasoning_content') or '')[-200:])
    return {'id': it['id'], 'gold': it['gold'], 'answer': m.group(1) if m else None, 'text': text[:80]}


def compare(da, db):
    A = {json.loads(l)['id']: json.loads(l) for l in open(os.path.join(da, 'answers.jsonl'))}
    B = {json.loads(l)['id']: json.loads(l) for l in open(os.path.join(db, 'answers.jsonl'))}
    for task in ('hs', 'wg'):
        ids = sorted(k for k in A if k.startswith(task) and k in B)
        if not ids: continue
        ca = sum(A[k]['answer'] == A[k]['gold'] for k in ids); cb = sum(B[k]['answer'] == B[k]['gold'] for k in ids)
        agree = sum(A[k]['answer'] == B[k]['answer'] for k in ids)
        a_only = sum(A[k]['answer'] == A[k]['gold'] and B[k]['answer'] != B[k]['gold'] for k in ids)
        b_only = sum(B[k]['answer'] == B[k]['gold'] and A[k]['answer'] != A[k]['gold'] for k in ids)
        n = a_only + b_only   # exact two-sided McNemar (binomial on the discordant pairs)
        k = min(a_only, b_only)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
        inv_a = sum(A[k2]['answer'] is None for k2 in ids); inv_b = sum(B[k2]['answer'] is None for k2 in ids)
        print(f'{task}: n={len(ids)}  {os.path.basename(da)} {100*ca/len(ids):.1f}%  {os.path.basename(db)} '
              f'{100*cb/len(ids):.1f}%  diff {100*(cb-ca)/len(ids):+.1f} pp  same answer {100*agree/len(ids):.1f}%  '
              f'right only A/B {a_only}/{b_only}  McNemar p={p:.2f}  invalid {inv_a}/{inv_b}')


ap = argparse.ArgumentParser()
ap.add_argument('label', nargs='?'); ap.add_argument('outdir', nargs='?'); ap.add_argument('base', nargs='?')
ap.add_argument('--conc', type=int, default=1); ap.add_argument('--hs', type=int, default=400)
ap.add_argument('--wg', type=int, default=400); ap.add_argument('--compare', nargs=2)
ap.add_argument('--hellaswag', default=HS)
ap.add_argument('--winogrande', default=WG)
ap.add_argument('--model', default='default-big')
a = ap.parse_args()
HS, WG, MODEL = a.hellaswag, a.winogrande, a.model
if a.compare:
    compare(*a.compare); sys.exit(0)
OUT = os.path.join(a.outdir, a.label); os.makedirs(OUT, exist_ok=True)
items = hs_items(a.hs) + wg_items(a.wg)
t0 = time.time()
with cf.ThreadPoolExecutor(a.conc) as ex, open(os.path.join(OUT, 'answers.jsonl'), 'w') as f:
    for r in ex.map(lambda it: ask(a.base, it), items):
        f.write(json.dumps(r) + '\n'); f.flush()
print(f'{a.label}: {len(items)} items in {time.time() - t0:.0f} s', flush=True)
for task in ('hs', 'wg'):
    rows = [json.loads(l) for l in open(os.path.join(OUT, 'answers.jsonl')) if json.loads(l)['id'].startswith(task)]
    if rows:
        print(f'{a.label} {task}: acc {100*sum(r["answer"] == r["gold"] for r in rows)/len(rows):.1f}% '
              f'(invalid {sum(r["answer"] is None for r in rows)})', flush=True)
