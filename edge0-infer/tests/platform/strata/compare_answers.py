"""Compare the answers of stream_session.py runs, word for word.

Usage: compare_answers.py <session.json | compare_outputs.py output> <session.json | compare_outputs.py output>
       [--drafts]

Each side is a session.json written by stream_session.py (its questions, long and chat answers) or the printed output
of compare_outputs.py (its "--- question N: ..." blocks, the proxy answers). Only the parts both sides have are
compared. --drafts also compares the accepted draft counts (timings.draft_n_accepted) of each answer, which only
session.json files have. Prints one line per answer and VERDICT PASS when every compared answer is the same.
"""
from __future__ import annotations

import argparse
import json
import re
import sys


def load(path):
    """{key: (answer, accepted drafts or None)} with keys q1..q4, long, chat1..chat3."""
    with open(path, encoding='utf-8') as f:
        text = f.read()
    if path.endswith('.json'):
        value = json.loads(text)
        out = {}
        for i, q in enumerate(value.get('questions', []), 1):
            out[f'q{i}'] = (q.get('content'), q.get('draft_n_accepted'))
        if 'long' in value:
            out['long'] = (value['long'].get('content'), value['long'].get('draft_n_accepted'))
        for i, t in enumerate(value.get('chat', []), 1):
            out[f'chat{i}'] = (t.get('content'), t.get('draft_n_accepted'))
        return out
    out = {}
    blocks = re.split(r'^--- question (\d+): .*\n', text, flags=re.M)
    for i in range(1, len(blocks), 2):
        body = blocks[i + 1].split('\n', 1)[1]            # the line after the header is the status line
        body = re.split(r'^VERDICT ', body, flags=re.M)[0]
        if body.endswith('\n'):
            body = body[:-1]                              # print() added it
        out[f'q{blocks[i]}'] = (body, None)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('left')
    parser.add_argument('right')
    parser.add_argument('--drafts', action='store_true')
    args = parser.parse_args()
    left, right = load(args.left), load(args.right)
    keys = [k for k in left if k in right]
    ok = bool(keys)
    for k in keys:
        same = left[k][0] == right[k][0]
        line = f'{k}: answer {"same" if same else "DIFFERENT"}'
        if args.drafts:
            drafts = left[k][1] == right[k][1]
            same = same and drafts
            line += f'; accepted drafts {left[k][1]} / {right[k][1]} {"same" if drafts else "DIFFERENT"}'
        if left[k][0] != right[k][0]:
            a, b = left[k][0] or '', right[k][0] or ''
            n = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
            line += f' (first difference at character {n}: {a[n:n + 30]!r} / {b[n:n + 30]!r})'
        ok = ok and same
        print(line)
    print(f'compared {len(keys)} answers: VERDICT {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
