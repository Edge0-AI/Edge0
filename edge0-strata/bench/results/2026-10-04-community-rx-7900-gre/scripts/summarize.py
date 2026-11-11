#!/usr/bin/env python3
"""Aggregate the runs in ../data into medians, ranges and memory peaks.

    python summarize.py            # print the tables
    python summarize.py --write   # also write ../data/summary.json

Throughput comes from the engine timing lines that tools/hip/bench_prefill.py
records per request. Memory peaks come from the sampler CSVs in ../data/memory.
"""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data'
QUANTS = ['coder-iq1_m', 'iq2_xs', 'iq3_xxs']


def runs(quant):
    out = []
    for n in (1, 2, 3):
        out.extend(json.load((DATA / f'{quant}-run{n}.json').open()))
    return out


def agg(values):
    return dict(median=round(statistics.median(values), 1), min=round(min(values), 1),
                max=round(max(values), 1), n=len(values))


def peak_vram(quant, suffix):
    path = DATA / 'memory' / f'{quant}-mem-{suffix}.csv'
    rows = list(csv.DictReader(path.open()))
    cols = [c for c in rows[0] if c.endswith('_vram_used')]
    return dict(samples=len(rows),
                vram_peak_gib=round(max(max(int(r[c]) for r in rows) for c in cols) / 2 ** 30, 2),
                ram_used_peak_gib=round(max(float(r['ram_used_mib']) for r in rows) / 1024, 2))


def engine_facts(quant):
    text = (DATA / f'{quant}.engine.log').read_text(errors='replace')
    facts = {}
    m = re.search(r'expert cache (\d+) slots, ([\d.]+) GiB of VRAM', text)
    if m:
        facts['expert_cache_slots'] = int(m.group(1))
        facts['expert_cache_vram_gib'] = float(m.group(2))
    m = re.search(r'PCIe probe: ([\d.]+) GB/s', text)
    if m:
        facts['pcie_probe_gb_s'] = float(m.group(1))
    m = re.search(r'profile \S+: (\d+) ranked pairs', text)
    if m:
        facts['profile_ranked_pairs'] = int(m.group(1))
    accepted = drafted = hits = lookups = 0
    for m in re.finditer(r'drafts accepted (\d+) of (\d+)', text):
        accepted += int(m.group(1))
        drafted += int(m.group(2))
    for m in re.finditer(r'decode expert cache hit rate: [\d.]+% \((\d+) hits / (\d+) lookups\)', text):
        if int(m.group(2)) > 20000:  # the per-request windows, not the short warm-up ones
            hits += int(m.group(1))
            lookups += int(m.group(2))
    facts['drafts_accepted'] = accepted
    facts['drafts_drafted'] = drafted
    facts['draft_acceptance_pct'] = round(100 * accepted / drafted, 1)
    facts['decode_cache_hit_pct'] = round(100 * hits / lookups, 1)
    return facts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write', action='store_true')
    args = p.parse_args()
    summary = {}
    for q in QUANTS:
        rs = runs(q)
        entry = {'engine': engine_facts(q),
                 'memory': {s: peak_vram(q, s) for s in ('run1', 'run2', 'run3', 'needle')},
                 'needles': json.load((DATA / f'{q}-needles.json').open())}
        for kind in ('fresh', 'followup'):
            got = [r for r in rs if r['kind'] == kind]
            entry[kind] = {
                'prefill_tps': agg([r['metrics']['prefill_tps'] for r in got]),
                'decode_tps': agg([r['metrics']['decode_tps'] for r in got]),
                'wall_s': agg([r['wall_s'] for r in got]),
                'reused_tokens': sorted({int(r['metrics']['reused']) for r in got}),
                'prompt_tokens': sorted({int(r['metrics']['prompt_tokens']) for r in got}),
                'generated_tokens': sorted({int(r['metrics']['generated']) for r in got}),
            }
        summary[q] = entry
        print('=' * 72)
        print(q)
        for kind in ('fresh', 'followup'):
            for field in ('prefill_tps', 'decode_tps', 'wall_s'):
                a = entry[kind][field]
                print(f"  {kind:9s} {field:12s} median {a['median']:8.1f}  [{a['min']:.1f}-{a['max']:.1f}]  n={a['n']}")
        print(f"  needles {sum(1 for x in entry['needles'] if x['found'])}/{len(entry['needles'])} found")
        print(f"  engine {entry['engine']}")
    if args.write:
        (DATA / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('wrote', DATA / 'summary.json')


if __name__ == '__main__':
    main()
