#!/usr/bin/env python3
"""Replace this machine's private paths in the report's data files before publishing.

The report rules say to remove credentials and private paths. Every file this
report ships is rewritten in place:

    <home>/Projects/Strata-data  -> <data root>
    <home>/Projects/Strata       -> <strata checkout>
    <home>/.local/opt            -> <local opt>
    <home>                       -> <home>

Nothing else is touched: numbers, flags and log lines stay exactly as the
engine wrote them. Run it once, after every measurement is in data/.

Usage: sanitize.py [--data DIR] [--dry-run]
"""
import argparse
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data'
HOME = Path.home()
# This machine's layout: where the checkout, its data root and the toolchain are.
CHECKOUT = HOME / 'Projects' / 'Strata'
DATA_ROOT = HOME / 'Projects' / 'Strata-data'
LOCAL_OPT = HOME / '.local' / 'opt'
REPLACEMENTS = [
    (str(DATA_ROOT), '<data root>'),
    (str(CHECKOUT), '<strata checkout>'),
    (str(LOCAL_OPT), '<local opt>'),
    (str(HOME), '<home>'),
]
SUFFIXES = {'.log', '.out', '.json', '.txt', '.csv', '.sh', '.py', '.md'}


def sanitize(text):
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    files = [f for f in sorted(args.data.rglob('*'))
             if f.is_file() and f.suffix in SUFFIXES]
    readme = args.data.parent / 'README.md'
    if readme.exists():
        files.append(readme)
    changed = 0
    for f in files:
        before = f.read_text(encoding='utf-8', errors='replace')
        after = sanitize(before)
        if after != before:
            changed += 1
            hits = sum(before.count(old) for old, _ in REPLACEMENTS)
            print(f'{f.name}: {hits} path(s)')
            if not args.dry_run:
                f.write_text(after, encoding='utf-8')
    print(f'{changed} file(s) {"would be " if args.dry_run else ""}rewritten')


if __name__ == '__main__':
    sys.exit(main())
