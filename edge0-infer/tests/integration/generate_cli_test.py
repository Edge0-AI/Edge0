"""Public CLI validation and strict comparison failure paths."""
import math
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

cli = sys.argv[1]
backend = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
reference = Path(os.environ.get('MOE_TINY', '/nonexistent'))
if not (reference / 'tiny-a').exists():
    raise SystemExit(77)
help_result = subprocess.run([cli], capture_output=True, text=True)
assert help_result.returncode == 2 and 'run' in help_result.stderr and '--help' in help_result.stderr
run_help = subprocess.run([cli, 'run', '--help'], capture_output=True, text=True)
assert run_help.returncode == 0 and '--verbose' in run_help.stdout and '--context' in run_help.stdout, run_help.stdout
case = Path(__file__).resolve().parents[2] / 'tests/fixtures/answers' / f'tiny-base-{backend}-flash-split/chat-intro'
base = [cli, 'run', '--model-option', 'routing=actual', '--device', backend, '--model', str(reference / 'tiny-a'), '--tokens', str(case / 'tokens.json'), '--model-option', 'lora=off', '--max-tokens', '16']
result = subprocess.run(base, capture_output=True, text=True)
if result.returncode == 77:
    raise SystemExit(77)
assert result.returncode == 0, result.stderr
default = json.loads(result.stderr)
assert default['configuration']['temperature'] == .7 and default['configuration']['top_k'] == 20
assert default['configuration']['top_p'] == .8 and default['configuration']['seed_used']
assert default['sources']['temperature'] == 'model'
run_usage = subprocess.run([cli, 'run', '--user', 'x'], capture_output=True, text=True)
assert run_usage.returncode != 0 and '--model-option' in run_usage.stderr, run_usage.stderr
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    output = root / 'out'
    result = subprocess.run(base + ['--temperature', '0', '--presence-penalty', '0', '--out', str(output), '--check', str(case)], capture_output=True, text=True)
    assert result.returncode == 0 and 'check: 16/16 tokens match' in result.stderr, result.stderr
    tokens = json.loads((output / 'tokens.json').read_text())
    summary = json.loads((output / 'result.json').read_text())
    assert summary['checkpoint_saved'] is True
    assert json.loads(next(line for line in result.stderr.splitlines() if line.startswith('{')))['checkpoint_saved'] is True
    assert tokens['logits_shape'] == [16, 248320]
    assert (output / 'logits.f32').stat().st_size == 16 * 248320 * 4
    assert summary['stop_reason'] == tokens['stop_reason'] == 'length'
    required = ['e', 'input_count', 'generated_count', 'tail_token', 'state_valid', 'reused_tokens', 'cold',
                'decode_plan_builds', 'decode_plan_reuses', 'configuration', 'sources', 'budget_peak', 'budget_current']
    assert all(key in summary for key in required)
    timing = summary['request_summary']
    assert 0 < timing['first_token_seconds'] <= timing['total_seconds']
    assert 0 < timing['decode_seconds'] <= timing['total_seconds']
    assert timing['decode_forwards'] == len(tokens['generated']) - 1
    assert math.isclose(timing['decode_tokens_per_second'], timing['decode_forwards'] / timing['decode_seconds'])
    activity = summary['weight_activity']
    assert activity['decode']['demand'] == 16 * 8 * 4
    assert activity['decode']['hits'] + activity['decode']['loads'] == activity['decode']['demand']
    assert activity['decode']['bytes'] == activity['decode']['loads'] * 7680
    assert json.loads(next(line for line in result.stderr.splitlines() if line.startswith('{')))['weight_activity'] == activity
    rejected = subprocess.run(base + ['--temperature', '0', '--presence-penalty', '0', '--model-option', 'expert_slots=3'], capture_output=True, text=True)
    assert rejected.returncode != 0 and 'expert_slots' in rejected.stderr
    repeated = subprocess.run(base + ['--temperature', '0', '--presence-penalty', '0', '--model-option', 'expert_slots=4', '--model-option', 'expert_slots=8'], capture_output=True, text=True)
    assert repeated.returncode != 0 and 'more than once' in repeated.stderr, repeated.stderr
    resident = root / 'resident'
    control = subprocess.run(base + ['--temperature', '0', '--presence-penalty', '0', '--model-option', 'experts=resident', '--out', str(resident)], capture_output=True, text=True)
    assert control.returncode == 0, control.stderr
    assert json.loads((resident / 'tokens.json').read_text()) == tokens
    assert json.loads((resident / 'result.json').read_text())['weight_activity'] is None
    forward_tokens = root / 'forward-tokens.json'
    forward_tokens.write_text(json.dumps({'input': tokens['input'], 'generated': tokens['generated'][:-1]}))
    forward = root / 'forward'
    checked = subprocess.run([cli, 'forward', '--model-option', 'routing=actual', '--device', backend, '--model', str(reference / 'tiny-a'), '--tokens', str(forward_tokens), '--model-option', 'lora=off', '--split-last', '--out', str(forward), '--model-option', 'experts=slots'], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    assert json.loads((forward / 'report.json').read_text())['weight_activity'] == activity
    resident_forward = root / 'resident-forward'
    checked_resident = subprocess.run([cli, 'forward', '--model-option', 'routing=actual', '--device', backend, '--model', str(reference / 'tiny-a'), '--tokens', str(forward_tokens), '--model-option', 'lora=off', '--split-last', '--out', str(resident_forward), '--model-option', 'experts=resident'], capture_output=True, text=True)
    assert checked_resident.returncode == 0, checked_resident.stderr
    assert json.loads((resident_forward / 'report.json').read_text())['weight_activity'] is None
    describe = subprocess.run([cli, 'describe', '--model', str(reference / 'tiny-a'), '--device', backend, '--model-option', 'lora=off', '--model-option', 'routing=actual'], capture_output=True, text=True)
    assert describe.returncode == 0, describe.stderr
    description = json.loads(describe.stdout)
    assert description['loading'] == summary['loading']
    default_describe = subprocess.run([cli, 'describe', '--model', str(reference / 'tiny-a'), '--device', backend], capture_output=True, text=True)
    default_run = subprocess.run([cli, 'run', '--model', str(reference / 'tiny-a'), '--device', backend,
                                  '--tokens', str(case / 'tokens.json'), '--temperature', '0', '--presence-penalty', '0', '--max-tokens', '1'], capture_output=True, text=True)
    assert default_describe.returncode == default_run.returncode == 0, default_describe.stderr + default_run.stderr
    default_summary = json.loads(default_run.stderr)
    assert json.loads(default_describe.stdout)['loading'] == default_summary['loading']
    assert default_summary['loading']['sources']['lora'] == 'model'
    assert default_summary['loading']['model_options']['prefill_read'] == 'auto'
    assert default_summary['loading']['model_options']['prefill_ahead_min_tokens'] == '3000'
    assert default_summary['loading']['sources']['prefill_ahead_min_tokens'] == 'model'
    assert default_summary['loading']['prefill_chunk'] == 4096
    assert default_summary['loading']['sources']['prefill_chunk'] == 'platform'
    chunk_run = subprocess.run(base + ['--temperature', '0', '--presence-penalty', '0', '--prefill-chunk', '3'], capture_output=True, text=True)
    assert chunk_run.returncode == 0, chunk_run.stderr
    chunk_description = subprocess.run([cli, 'describe', '--model', str(reference / 'tiny-a'),
                                        '--device', backend, '--prefill-chunk', '3', '--model-option', 'prefill_ahead_min_tokens=3'], capture_output=True, text=True)
    assert chunk_description.returncode == 0, chunk_description.stderr
    described = json.loads(chunk_description.stdout)['loading']
    assert described['model_options']['prefill_ahead_min_tokens'] == '3'
    assert described['sources']['prefill_ahead_min_tokens'] == 'request'
    for loading in [json.loads(chunk_run.stderr)['loading'], json.loads(chunk_description.stdout)['loading']]:
        assert loading['prefill_chunk'] == 3 and loading['sources']['prefill_chunk'] == 'request'
    one = default_summary['request_summary']
    assert one['decode_tokens_per_second'] is None and one['decode_seconds'] is None
    assert one['decode_forwards'] is None and one['decode_reason'] == 'fewer_than_two_tokens'
    assert 0 < one['first_token_seconds'] <= one['total_seconds']
    assert description['recommended']['temperature'] == .7
    assert description['recommended']['top_k'] == 20
    assert description['recommended']['top_p'] == .8
    assert description['thinking'] == dict(supported=True, default=False, enabled_supported=True)
    assert description['tools_supported'] is True
    memory = summary['memory_snapshot']
    assert memory['ledger']['current_bytes'] == summary['budget_current']
    assert memory['ledger']['peak_bytes'] == summary['budget_peak']
    assert memory['ledger']['scope'] == 'runtime_lifetime'
    for name in ['rss', 'physical']:
        assert memory[name]['scope'] == 'process_lifetime'
        if sys.platform == 'darwin':
            assert memory[name]['values']['peak_bytes'] >= memory[name]['values']['current_bytes'] > 0
    quiet = subprocess.run(base + ['--temperature', '0', '--presence-penalty', '0'], capture_output=True, text=True)
    assert quiet.returncode == 0, quiet.stderr
    assert json.loads(next(line for line in quiet.stderr.splitlines() if line.startswith('{')))['checkpoint_saved'] is True
    verbose_output = root / 'verbose'
    verbose = subprocess.run(base + ['--temperature', '0', '--presence-penalty', '0', '--verbose', '--out', str(verbose_output)], capture_output=True, text=True)
    assert verbose.returncode == 0, verbose.stderr
    assert verbose.stdout == quiet.stdout == result.stdout
    assert json.loads((verbose_output / 'tokens.json').read_text()) == tokens
    if backend == 'metal':
        assert 'ggml_metal_' in verbose.stderr, verbose.stderr
    tampered = root / 'tampered'
    tampered.mkdir()
    expected = json.loads((case / 'tokens.json').read_text())
    expected['generated'][3] = (expected['generated'][3] + 1) % 248320
    (tampered / 'tokens.json').write_text(json.dumps(expected))
    result = subprocess.run(base + ['--temperature', '0', '--presence-penalty', '0', '--check', str(tampered)], capture_output=True, text=True)
    assert result.returncode == 1 and 'first difference at 3' in result.stderr
    print('checkpoint fields, verbose text/token equality and tampered-token rejection passed', flush=True)
    assert not any(line.startswith('ggml_') for line in quiet.stderr.splitlines()), quiet.stderr
    assert json.loads(quiet.stderr)['checkpoint_saved'] is True
print('run usage, recommended defaults, output schema, exact check and tampered token rejection passed')

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    for name, turns in [('tokens', [{'input':[1,2,3]},{'continue':[4,5]}]),
                        ('messages', [{'user':'Hello'},{'user':'Next'}])]:
        turn_file=root/(name+'.json');turn_file.write_text(json.dumps({'turns':turns}))
        output=root/name
        command=[cli,'run','--model',str(reference/'tiny-a'),'--device',backend,'--temperature','0','--max-tokens','2','--turns',str(turn_file)]
        result=subprocess.run(command+['--out',str(output)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        for index in [1,2]:
            folder=output/str(index)
            for file in ['tokens.json','logits.f32','result.json','blocks.json']:
                assert (folder/file).is_file()
            summary=json.loads((folder/'result.json').read_text())
            assert {'reuse_method','common_prefix','checkpoint_restored','checkpoint_boundary','checkpoint_save_seconds'}<=summary.keys()
        assert (output/'blocks.json').is_file()
        for extra in [['--user','x'],['--tokens',str(case/'tokens.json')]]:
            rejected=subprocess.run(command+extra,capture_output=True,text=True)
            assert rejected.returncode!=0 and 'Usage:' in rejected.stderr
print('two-turn token/message schemas and mutually exclusive inputs passed')
