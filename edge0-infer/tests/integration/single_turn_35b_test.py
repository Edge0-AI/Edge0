"""35B fixed CPU judgement, Metal repeatability and recorded cross-backend differences."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_metric import process_peak
import numpy as np

ANSWERS = Path(__file__).resolve().parents[2] / 'tests/fixtures/answers'
sys.path.insert(0, str(ANSWERS.parents[1] / 'answers'))
from judge import judge, judge_generation, judge_suite, load_answer, observed_routes, own_logits
from answer_format import read_routes


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 77:
        print(result.stderr, flush=True)
        raise SystemExit(77)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def differences(output, observed, answer):
    """Record (without a threshold) how a run differs from an answer of another backend."""
    identity, expected, rows = load_answer(answer, identity_backend(answer))
    actual = json.loads((output / 'tokens.json').read_text())
    assert actual['input'] == expected['input']
    a = own_logits(output, actual).astype(np.float64)
    start = len(expected['input']) - 1
    saved = rows[start:start + len(a)]
    assert len(saved) == len(a)
    prefix = 0
    for own, reference in zip(actual['generated'], expected['generated']):
        if own != reference:
            break
        prefix += 1
    probes = np.asarray(identity['probes'])
    top = np.take_along_axis(a, saved['tokens'].astype(np.int64), axis=1)
    row_max = np.maximum(np.abs(top - saved['values']).max(axis=1), np.abs(a[:, probes] - saved['probes']).max(axis=1))
    selected = []
    for i, row in enumerate(saved):
        token = expected['generated'][i] if i < len(expected['generated']) else None
        listed = dict(zip(row['tokens'].tolist(), row['values'].tolist()))
        selected.append(None if token not in listed else abs(float(a[i, token]) - listed[token]))
    different = total = ratio = None
    if observed is not None:
        expected_routes = read_routes(answer / 'routes.bin', identity, [tuple(s) for s in expected['route_steps']])
        actual_routes = observed_routes(observed)
        assert expected_routes.keys() <= actual_routes.keys()
        different, total = 0, 0
        for key, ids in sorted(expected_routes.items()):
            own_ids = actual_routes[key][0]
            assert own_ids.shape == ids.shape
            changed = np.any(np.sort(ids, axis=-1) != np.sort(own_ids, axis=-1), axis=-1)
            different += int(changed.sum())
            total += changed.size
        ratio = different / total
    return dict(token_prefix_match=prefix, expected_tokens=len(expected['generated']),
                route_different_positions=different, route_positions=total, route_difference_ratio=ratio,
                observation='executed' if observed is not None else 'not executed: exceeds iOS app memory limit',
                selected_token_logit_abs_difference=selected, row_logits_max_abs=row_max.tolist(),
                numerical_threshold=None, note='recorded only; histories diverge after the first differing token')


def identity_backend(answer):
    return json.loads((Path(answer).parent / 'identity.json').read_text())['backend']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    parser.add_argument('--backend', choices=['cpu', 'metal'], default='cpu')
    parser.add_argument('--mode', choices=['numerical', 'memory', 'describe'], default='numerical')
    parser.add_argument("--cross-kernel", action="store_true", help="Use explicit cross-kernel answer tolerances")
    parser.add_argument('--record-only', action='store_true', help='Record numerical-mode answer differences')
    parser.add_argument('--skip-observe', action='append', default=[],
                        choices=['chat-intro', 'plain-mixed', 'long-mixed'],
                        help='Skip route observation for a named case exceeding iOS app memory')
    args = parser.parse_args()
    if args.skip_observe and (args.backend != 'metal' or args.mode != 'numerical'):
        parser.error('--skip-observe requires Metal numerical mode')
    if args.record_only and args.mode != 'numerical':
        parser.error('--record-only requires --mode numerical')
    assets = Path(os.environ.get('MOE_ASSETS_35B', '/nonexistent'))
    reference = ANSWERS / '35b-lora-cpu-flash-split'
    if not (assets / 'assets.tsv').is_file():
        print('SKIP: missing MOE_ASSETS_35B')
        return 77
    common = ['--model', str(assets), '--device', args.backend]
    if args.mode == 'describe':
        result = json.loads(run([args.cli, 'describe', *common]).stdout)
        assert result['adapter'] == 'edge0_35b' and result['context'] == 16384
        assert result['context_limit'] >= 4096
        assert result['recommended']['temperature'] == .7
        assert result['recommended']['top_k'] == 20 and result['recommended']['top_p'] == .8
        assert result['thinking'] == dict(supported=True, default=False, enabled_supported=True)
        assert result['tools_supported'] is True
        assert result['loading']['model_options']['experts'] == 'slots'
        assert result['loading']['model_options']['routing'] == 'predicted'
        result['directory'] = '$MOE_ASSETS_35B'
        result['loading']['directory'] = '$MOE_ASSETS_35B'
        print(json.dumps(result), flush=True)
        return 0
    with tempfile.TemporaryDirectory() as temp:
        workspace = Path(temp)
        reports = []
        names = ['chat-intro'] if args.mode == 'memory' else ['chat-intro', 'plain-mixed', 'long-mixed']
        for name in names:
            answer = reference / name
            expected = json.loads((answer / 'tokens.json').read_text())
            output = workspace / name
            command = [args.cli, 'run', *common, '--model-option', 'routing=actual', '--tokens', str(answer / 'tokens.json'), '--temperature', '0', '--presence-penalty', '0',
                       '--max-tokens', str(len(expected['generated']))]
            run(command + ['--out', str(output)])
            summary = json.loads((output / 'result.json').read_text())
            if args.mode == 'memory':
                memory = summary['memory_snapshot']
                assert summary['generated_count'] == 16
                assert memory['ledger']['peak_bytes'] <= 4 * 1024**3
                metric, source, peak = process_peak(memory)
                print(f'memory verdict: {metric} peak ({source}) = {peak}', flush=True)
                assert peak <= 4 * 1024**3
                assert summary['loading']['sources']['context'] == 'model'
                print(json.dumps(dict(backend=args.backend, memory_snapshot=memory, weight_activity=summary['weight_activity'])), flush=True)
                continue
            forward, observed = workspace / (name+'-forward'), workspace / (name+'-observed')
            base = [args.cli, 'forward', *common, '--model-option', 'routing=actual', '--tokens', str(answer / 'tokens.json'), '--split-last']
            if name in args.skip_observe:
                print(f'{name}: --observe not executed: exceeds iOS app memory limit', flush=True)
                observed = None
            else:
                run(base + ['--observe', '--out', str(observed)])
            if args.backend == 'cpu':
                run(base + ['--out', str(forward)])
                report = judge(forward, observed, answer, 'cpu', assets, cross_kernel=args.cross_kernel)
                own_logits(observed, json.loads((observed / 'tokens.json').read_text()))
                own_logits(output, json.loads((output / 'tokens.json').read_text()))
                generation = judge_generation(output, answer, 'cpu', assets, cross_kernel=args.cross_kernel, record_only=args.record_only)
                reports.append(report)
                print(json.dumps(dict(case=name, record_only=args.record_only, forward_status=report['status'], tie_case=report['tie_case'],
                    logits=report['logits'], route_count=len(report['routes']), route_mismatches=sum(item['mismatches'] for item in report['routes']),
                    generation=generation, first_failure=report['first_failure'])), flush=True)
                if not args.record_only:
                    assert report['status'] == generation['status'] == 'passed'
                    assert generation['tie_cases'] == 0
            else:
                repeated = workspace / (name+'-repeat')
                run(command + ['--out', str(repeated)])
                for filename in ['tokens.json', 'logits.f32']:
                    assert (output / filename).read_bytes() == (repeated / filename).read_bytes(), (name, filename)
                print(json.dumps(dict(case=name, repeated_bitwise=True, **differences(output, observed, answer))), flush=True)
        if reports:
            suite = judge_suite(reports)
            if not args.record_only:
                assert suite['status'] == 'passed'
            print(json.dumps(dict(status=suite['status'], routing_tie_cases=suite['tie_cases'])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
