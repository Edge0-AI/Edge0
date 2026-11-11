#!/usr/bin/env python3
"""Validate three repetitions and summarize native timings without mixing tasks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

METRICS = ('prompt_tps', 'decode_tps', 'ttft_s', 'post_first_tps', 'total_s')
def stats(values):
    return {'median': statistics.median(values), 'minimum': min(values), 'maximum': max(values)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', type=Path)
    ap.add_argument('output', type=Path)
    a = ap.parse_args()
    rows_path = a.input / 'requests.json'
    if not rows_path.exists():
        rows_path = a.input / 'results.json'
    rows = json.loads(rows_path.read_text())
    runs = json.loads((a.input / 'runs.json').read_text())
    result = json.loads((a.input / 'result.json').read_text())
    assert result['failure'] is None
    assert len(runs) == 3 and all(r['native_exit'] == 0 for r in runs)
    assert all(r['valid'] for r in rows)
    measured = [r for r in rows if r['measured']]
    assert len(measured) == 30
    for r in rows:
        n = r['native']
        assert r['native_done_line'].startswith('DONE ')
        assert n['generated'] == r['output_tokens'] and n['prompt_tokens'] == r['input_tokens']
        assert n['reused'] == 0 and n['prompt_read'] == r['input_tokens']
        assert abs(r['prompt_tps'] - n['prompt_read'] * 1000 / n['prompt_ms']) < 1e-10
        assert abs(r['decode_tps'] - n['generated'] * 1000 / n['decode_ms']) < 1e-10
        # Independent stderr format: timings/rates are rounded more coarsely
        # than DONE; validate counts and allow only that documented rounding.
        assert len(r['native_timing_lines']) == 1
        line = re.search(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms '
                         r'\(([\d.]+) tok/s\), (\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\)',
                         r['native_timing_lines'][0])
        assert line is not None
        assert (int(line[1]), int(line[2]), int(line[3]), int(line[6])) == (n['prompt_tokens'], n['reused'], n['prompt_read'], n['generated'])
        assert abs(float(line[4]) - n['prompt_ms']) <= 1 and abs(float(line[7]) - n['decode_ms']) <= 1
        assert abs(float(line[5]) - r['prompt_tps']) <= 0.11 and abs(float(line[8]) - r['decode_tps']) <= 0.11
    per_task = []
    for length in (512, 7000):
        for i in range(5):
            label = f'bench-{length}-{i}'
            rr = [r for r in measured if r['label'] == label]
            assert len(rr) == 3 and sorted(r['run'] for r in rr) == [1, 2, 3]
            assert len({r['input_ids_sha256'] for r in rr}) == 1
            assert all(r['output_tokens'] == 640 and r['native']['finish'] == 'length' for r in rr)
            same = len({r['output_ids_sha256'] for r in rr}) == 1
            assert same, f'greedy stream changed: {label}'
            accepted = sum(r['native']['drafts_accepted'] for r in rr)
            offered = sum(r['native']['drafts_offered'] for r in rr)
            per_task.append({'label': label, 'input_tokens': length, 'task_index': i,
                             'runs': 3, 'generated_tokens': 640, 'reused_tokens': 0,
                             'all_output_ids_equal': same, 'draft_acceptance_pooled': accepted / offered,
                             **{k: stats([r[k] for r in rr]) for k in METRICS}})
    per_run = []
    for run in (1, 2, 3):
        for length in (512, 7000):
            rr = [r for r in measured if r['run'] == run and r['input_tokens'] == length]
            assert len(rr) == 5
            per_run.append({'run': run, 'input_tokens': length, 'tasks': 5,
                            **{k: statistics.median(r[k] for r in rr) for k in METRICS}})
    across_run_medians = []
    for length in (512, 7000):
        rr = [r for r in per_run if r['input_tokens'] == length]
        across_run_medians.append({'input_tokens': length, 'runs': 3, 'tasks_per_run': 5,
                                  **{k: stats([r[k] for r in rr]) for k in METRICS}})
    summary = {'native_timing_definition': 'prompt_read*1000/prompt_ms; generated*1000/decode_ms from DONE',
               'per_task_three_repetitions': per_task, 'five_task_medians_per_run': per_run,
               'range_of_three_five_task_medians': across_run_medians,
               'successful_measured_requests': 30, 'failed_requests': 0,
               'excluded_warmups_and_checks': len(rows) - 30,
               'all_greedy_output_ids_equal_across_three_runs': True,
               'all_native_exits_zero': True}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    # Raw JSON retains excluded checks and warmups, not just selected successes.
    (a.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    fields = ['run', 'label', 'measured', 'input_tokens', 'output_tokens', 'prompt_ms', 'decode_ms',
              'reused_tokens', 'drafts_accepted', 'drafts_offered', *METRICS, 'valid']
    with (a.output / 'results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for r in rows:
            n = r['native']
            w.writerow({**{k: r[k] for k in fields if k in r},
                        'prompt_ms': n['prompt_ms'], 'decode_ms': n['decode_ms'],
                        'reused_tokens': n['reused'], 'drafts_accepted': n['drafts_accepted'],
                        'drafts_offered': n['drafts_offered']})
    lines = []
    for r in rows:
        lines.append(f"# run={r['run']} label={r['label']} measured={r['measured']}")
        lines.append(r['native_done_line'].strip())
        lines.extend(r['native_timing_lines'])
    (a.output / 'native-timings.log').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'validated_measured_requests': 30, 'native_exits': [r['native_exit'] for r in runs],
                      'output_hashes_identical': True, 'summary': across_run_medians}))

if __name__ == '__main__':
    main()
