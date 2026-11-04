"""Benchmark window, workload, summary, and validation through the public CLI."""
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

cli = sys.argv[1]
backend = sys.argv[2] if len(sys.argv) > 2 else "cpu"
reference = Path(os.environ.get('MOE_TINY', '/nonexistent'))
if not (reference / 'tiny-a').exists():
    raise SystemExit(77)
model = str(reference / 'tiny-a')
case = Path(__file__).resolve().parents[2] / 'tests/fixtures/answers' / 'tiny-base-cpu-flash-split/chat-intro/tokens.json'
fixture = Path(__file__).resolve().parents[1] / 'fixtures/bench/edge0-long-prompt.txt'


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, f'Duplicate JSON key: {key}'
        result[key] = value
    return result


def platform_runtime():
    # The default thread counts are platform specific (macOS pins {4, 4}; Windows
    # derives them from the core count), so read the expected values from the CLI's
    # own 'info' rather than hardcoding them. The exact rule is covered elsewhere.
    out = subprocess.run([cli, 'info'], capture_output=True, text=True).stdout
    decode = int(re.search(r'runtime\.cpu_threads: (\d+)', out).group(1))
    prefill = int(re.search(r'runtime\.cpu_prefill_threads: (\d+)', out).group(1))
    return {'cpu_threads': decode, 'cpu_prefill_threads': prefill,
            'cpu_threads_source': 'platform_default', 'cpu_prefill_threads_source': 'platform_default'}


def json_lines(text):
    rows = [json.loads(line, object_pairs_hook=unique_object) for line in text.splitlines()]
    for row in rows:
        if row['type'] in ('run', 'summary'):
            assert row['runtime'] == EXPECTED_RUNTIME
    return rows


def call(args):
    result = subprocess.run([cli, *args, "--device", backend], capture_output=True, text=True)
    if result.returncode == 77:
        raise SystemExit(77)
    assert result.returncode == 0, result.stderr
    return result


EXPECTED_RUNTIME = platform_runtime()
for source in [['--tokens', str(case)], ['--user-file', str(fixture)]]:
    output = call(['bench', '--model', model, *source, '--warmup', '2', '--ntok', '5', '--runs', '2'])
    rows = json_lines(output.stdout)
    assert len(rows) == 3 and rows[-1]['type'] == 'summary'
    run_input = source if source[0] == '--tokens' else ['--user', fixture.read_text()]
    run = json.loads(call(['run', '--model', model, *run_input, '--temperature', '0', '--max-tokens', '1']).stderr)
    for index, row in enumerate(rows[:2], 1):
        assert row['type'] == 'run' and row['run'] == index
        assert row['input_count'] == run['input_count']
        assert row['accepted_tokens'] == 8 and row['stop_reason'] == 'length'
        assert row['decode_forwards'] == 5
        timing = row['request_summary']
        assert 0 < row['prefill_seconds'] <= row['marker_seconds'] <= timing['total_seconds']
        assert 0 < row['decode_seconds'] <= timing['decode_seconds'] <= timing['total_seconds']
        assert math.isclose(row['decode_seconds'], row['prefill_seconds'] + timing['decode_seconds'] - row['marker_seconds'])
        assert math.isclose(row['decode_tokens_per_second'], 5 / row['decode_seconds'])
        assert math.isclose(row['prefill_tokens_per_second'], row['input_count'] / row['prefill_seconds'])
        assert row['generated_count'] + row['end_token_count'] == row['accepted_tokens']
        assert (row['first_end_token_position'] is None) == (row['end_token_count'] == 0)
        assert set(row['memory_snapshot']) == {'ledger', 'rss', 'physical'}
    assert rows[-1]['decode_forwards'] == 10
    engine = rows[-1]['engine']
    assert re.fullmatch(r'([0-9a-f]{40})?', engine['source_commit']) and engine['source_modified'] in ('yes', 'no', 'unknown')
    assert engine['ggml_version'] and engine['version']
    assert rows[-1]['model']['assets'] and rows[-1]['model']['assets_verified'] is False
    assert math.isclose(rows[-1]['decode_seconds'], sum(row['decode_seconds'] for row in rows[:2]))
chunked = call(['bench', '--model', model, '--tokens', str(case), '--warmup', '0', '--ntok', '1', '--runs', '1', '--prefill-chunk', '3'])
assert json_lines(chunked.stdout)[0]['accepted_tokens'] == 2

for extra in [['--ntok', '0'], ['--runs', '0'], ['--warmup', '-1'], ['--warmup', '9223372036854775807'], ['--ntok', '4096'], ['--ntok', '1x'], ['--user-file', str(fixture)]]:
    result = subprocess.run([cli, 'bench', '--model', model, '--tokens', str(case), *extra], capture_output=True, text=True)
    assert result.returncode == 1, result.stdout
print('bench two runs, exact window, input parity, memory, EOG accounting and invalid workloads passed')

for warmup, ntok in [(2, 5), (10, 200)]:
    detailed = call(['bench', '--model', model, '--tokens', str(case), '--warmup', str(warmup), '--ntok', str(ntok), '--runs', '1', '--breakdown'])
    row = json_lines(detailed.stdout)[0]
    timing = row['breakdown']
    assert timing['window_first_acceptance'] == warmup + 1
    assert timing['window_last_acceptance'] == warmup + ntok + 1
    for name, distribution in timing['fields'].items():
        assert distribution['samples'] == ntok
        assert 0 <= distribution['median'] <= distribution['p95'] <= distribution['maximum']
    assert timing['fields']['segments']['median'] > 0
    assert timing['fields']['binding_seconds']['median'] > 0
    assert timing['fields']['readback_bytes']['median'] > 0
    assert timing['submission_overhead']['samples'] == 200
    assert 0 < timing['submission_overhead']['median_seconds'] <= timing['submission_overhead']['p95_seconds']
    assert timing['estimated_lower_bound_seconds']['samples'] == ntok
    assert 'sampling_seconds' not in timing['fields']
sampled = call(['bench', '--model', model, '--tokens', str(case), '--warmup', '2', '--ntok', '5', '--runs', '1',
                '--breakdown', '--temperature', '.8', '--seed', '7'])
timing = json_lines(sampled.stdout)[0]['breakdown']['fields']['sampling_seconds']
assert timing['samples'] == 5
assert 0 <= timing['median'] <= timing['p95'] <= timing['maximum']
print('sampling timing present only for host sampling and nonnegative passed')


# Conversation reuse: bench --turns against run --turns on the same conversation and configuration.
conversation = {'turns': [{'user': 'Hello there, tell me about the sea.'}, {'user': 'And the sky?'}]}
with tempfile.TemporaryDirectory(prefix='moe-bench-turns-') as directory:
    turns_file = Path(directory) / 'turns.json'
    turns_file.write_text(json.dumps(conversation))
    out = Path(directory) / 'run-out'
    run = call(['run', '--model', model, '--turns', str(turns_file), '--max-tokens', '6', '--temperature', '0',
                '--presence-penalty', '0', '--out', str(out)])
    bench = call(['bench', '--model', model, '--turns', str(turns_file), '--max-tokens', '6'])
    rows = json_lines(bench.stdout)
    turns, summary = rows[:-1], rows[-1]
    assert [row['type'] for row in turns] == ['turn', 'turn'] and summary['type'] == 'summary' and summary['turns'] == 2
    for index, row in enumerate(turns, 1):
        expected = json.loads((out / str(index) / 'tokens.json').read_text())
        assert row['turn'] == index and row['thinking'] is False
        assert row['input'] == expected['input'] and row['generated'] == expected['generated'], index
        assert row['input_count'] == len(row['input']) and row['generated_count'] == len(row['generated']) == 6
        assert row['stop_reason'] == 'length' and row['first_token_seconds'] > 0
        # computed_tokens counts every forward of the request: new input plus decode steps.
        assert row['reused_tokens'] + row['computed_tokens'] == row['input_count'] + row['generated_count'] - 1
        # The cold reference is a different, fresh session: nothing reused, the whole input computed.
        cold = row['cold']
        assert cold['reuse_method'] == 'first' and cold['reused_tokens'] == 0 and cold['computed_tokens'] == row['input_count']
        assert cold['first_token_seconds'] > 0 and cold['stop_reason'] == 'length'
    assert turns[0]['reuse_method'] == 'first' and turns[0]['reused_tokens'] == 0
    assert turns[1]['reuse_method'] != 'first' and turns[1]['reused_tokens'] > 0
    assert turns[1]['input'][:turns[1]['reused_tokens']] == turns[0]['input'][:turns[1]['reused_tokens']]
    assert summary['engine']['version'] and summary['model']['assets'] and summary['model']['assets_verified'] is False
    thinking_file = Path(directory) / 'thinking.json'
    thinking_file.write_text(json.dumps({'turns': [{'user': 'Hello.', 'thinking': True}]}))
    thinking = json_lines(call(['bench', '--model', model, '--turns', str(thinking_file), '--max-tokens', '2']).stdout)
    assert thinking[0]['thinking'] is True
    for bad in [['--turns', str(turns_file), '--user-file', str(fixture)], ['--turns', str(turns_file), '--tokens', str(case)],
                ['--turns', str(turns_file), '--warmup', '1'], ['--turns', str(turns_file), '--breakdown'],
                ['--turns', str(turns_file), '--max-tokens', '0'], ['--tokens', str(case), '--max-tokens', '4']]:
        result = subprocess.run([cli, 'bench', '--model', model, *bad], capture_output=True, text=True)
        assert result.returncode == 1 and not result.stdout, bad
    for invalid in [{'turns': []}, {'turns': [{'user': 'x', 'extra': 1}]}, {'turns': [{'thinking': True}]}, {'turns': [{'user': 'x'}], 'other': 1}]:
        broken = Path(directory) / 'broken.json'
        broken.write_text(json.dumps(invalid))
        result = subprocess.run([cli, 'bench', '--model', model, '--turns', str(broken), '--device', backend], capture_output=True, text=True)
        assert result.returncode == 1, invalid
print('bench --turns tokens match run --turns, reuse accounting, cold reference on a fresh session, and invalid arguments passed')
