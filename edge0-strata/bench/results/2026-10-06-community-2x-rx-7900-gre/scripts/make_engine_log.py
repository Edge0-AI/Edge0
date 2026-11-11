#!/usr/bin/env python3
"""Assemble data/engine.log from the engine log slices this report collected.

The raw slices are one long block per server start (about 500 lines each, one
per configuration and per interleaved request), which is too much for a PR.
This keeps what the report is about from every slice:

  * the lines about the layer split, the PCIe probes, the expert caches,
    the resident stage, --batch and --pipeline-windows,
  * every completion line (`prompt N tokens = ...`) - the numbers in README,
  * the lines that say a feature was turned off or adjusted.

and prefixes each section with the slice's name and how many lines it had.

Usage: make_engine_log.py [--data DIR]
"""
import argparse
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data'

KEEP = re.compile(
    r'layer split|PCIe probe|pipeline-windows|--batch|resident|'
    r'expert cache|token graph|room for experts|prompt \d+ tokens =|'
    r'loads the dense weights|engine started|off:|WARNING|'
    r'session is up|KV streaming|GPU \d+:|decode expert cache hit|'
    r'strata mtp|strata batch|batch window|batch windows|no progress|\bnan\b', re.I)

ORDER = ['engine-interleave.log', 'engine-ab-pipeline-off.log',
         'engine-ab-stage-trim-off.log', 'engine-ab-resident.log',
         'soak-engine.log', 'engine-ab-batch2.log', 'engine-ab-batch4.log',
         'batch-concurrent-batch2.engine.log',
         'batch-concurrent-batch4.engine.log']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args()
    out = args.out or (args.data / 'engine.log')
    parts = ['= engine.log: the lines of each slice this report kept =\n',
             'built by scripts/make_engine_log.py; the raw slices sit beside it\n']
    names = [n for n in ORDER if (args.data / n).exists()]
    names += sorted(n.name for n in args.data.glob('engine-*.log')
                    if n.name not in names and n.name != 'engine.log')
    for name in names:
        text = (args.data / name).read_text(encoding='utf-8', errors='replace')
        lines = text.splitlines()
        kept = [ln for ln in lines if KEEP.search(ln)]
        parts.append(f'\n=== {name}: {len(lines)} lines, {len(kept)} kept ===\n')
        parts.extend(ln + '\n' for ln in kept)
    out.write_text(''.join(parts), encoding='utf-8')
    print(f'{out}: {out.stat().st_size} bytes from {len(names)} slice(s)')


if __name__ == '__main__':
    main()
