"""Sampling reproducibility, exact readback accounting and multi-turn logits replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    parser.add_argument('--backend', choices=['cpu', 'metal'], default='cpu')
    args = parser.parse_args()
    model = Path(os.environ.get('MOE_TINY', '/nonexistent')) / 'tiny-a'
    answer = Path(__file__).resolve().parents[2] / 'tests/fixtures/answers' / f'tiny-base-{args.backend}-flash-split/chat-intro'
    if not (model / 'assets.tsv').is_file():
        return 77
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        base = [args.cli, 'run', '--model', str(model), '--device', args.backend, '--max-tokens', '16',
                '--model-option', 'lora=off', '--model-option', 'routing=actual']

        def call(extra):
            result = subprocess.run(base + extra, capture_output=True, text=True)
            if result.returncode == 77:
                raise SystemExit(77)
            assert result.returncode == 0, result.stderr
            rows = [json.loads(line) for line in result.stderr.splitlines() if line.startswith('{')]
            return result, rows

        def run(name, options, export=True):
            extra = ['--tokens', str(answer / 'tokens.json'), *options]
            if export:
                extra += ['--out', str(root / name)]
            result, rows = call(extra)
            tokens = json.loads((root / name / 'tokens.json').read_text())['generated'] if export else None
            return result, rows[0], tokens

        accounting_errors = []
        def account(actual, expected, context):
            if actual != expected:
                accounting_errors.append((context, actual, expected))

        options = ['--temperature', '.8', '--seed', '7']
        a, first, tokens = run('a', options)
        b, second, again = run('b', options)
        assert tokens == again and a.stdout == b.stdout
        n = len(tokens)
        assert first['random_draws'] == second['random_draws'] == n
        plain, measured, _ = run('plain', options, False)
        assert plain.stdout == a.stdout
        v = measured['vocabulary']
        assert measured['host_sampled_steps'] == measured['random_draws'] == n
        account(measured['sampling_readback_bytes'], 4 * v * n, 'host')
        account(first['sampling_readback_bytes'], 4 * v * n, 'host export')
        a, unseeded, tokens = run('unseeded', ['--temperature', '.8'])
        seed = unseeded['configuration']['seed']
        assert isinstance(seed, int) and unseeded['configuration']['seed_used']
        b, _, again = run('reseeded', ['--temperature', '.8', '--seed', str(seed)])
        assert tokens == again and a.stdout == b.stdout
        a, exported, tokens = run('greedy', ['--temperature', '0', '--presence-penalty', '0', '--check', str(answer)])
        b, measured, _ = run('greedy-plain', ['--temperature', '0', '--presence-penalty', '0', '--check', str(answer)], False)
        assert a.stdout == b.stdout
        assert tokens == json.loads((answer / 'tokens.json').read_text())['generated']
        n = len(tokens)
        assert measured['random_draws'] == measured['host_sampled_steps'] == 0
        account(measured['sampling_readback_bytes'], 8 * n, 'greedy')
        account(exported['sampling_readback_bytes'], (8 + 4 * v) * n, 'greedy export')
        a, penalty, tokens = run('penalty', ['--temperature', '0', '--repetition-penalty', '1.3'])
        b, measured, _ = run('penalty-plain', ['--temperature', '0', '--repetition-penalty', '1.3'], False)
        assert a.stdout == b.stdout
        assert measured['random_draws'] == 0 and measured['host_sampled_steps'] == len(tokens)
        account(measured['sampling_readback_bytes'], 4 * v * len(tokens), 'penalized greedy')
        assert penalty['configuration']['seed'] is None
        turns = root / 'turns.json'
        turns.write_text(json.dumps({'turns': [{'input': [1, 2, 3, 4, 5]}, {'continue': [6, 7]}, {'continue': [8, 9]}]}))
        _, reports = call(['--turns', str(turns), '--out', str(root / 'turns'), *options])
        assert len(reports) == 3 and reports[1]['reuse_method'] == reports[2]['reuse_method'] == 'keep'
        for index in range(1, 4):
            folder = root / 'turns' / str(index)
            replay = root / f'replay-{index}'
            call(['--replay', str(folder / 'blocks.json'), '--out', str(replay)])
            original = json.loads((folder / 'tokens.json').read_text())
            blocks = json.loads((folder / 'blocks.json').read_text())['blocks']
            selected = [i for i, block in enumerate(blocks) if block['sample']][-len(original['generated']):]
            width = v * 4
            all_rows = (replay / 'logits.f32').read_bytes()
            cold = b''.join(all_rows[i*width:(i+1)*width] for i in selected)
            actual = (folder / 'logits.f32').read_bytes()
            assert len(actual) == len(cold) == len(original['generated']) * width
            assert actual == cold, (args.backend, index, 'replay logits differ')
        print('Reproducibility and three-turn replay passed', flush=True)
        assert not accounting_errors, accounting_errors
        print(json.dumps({'backend': args.backend, 'passed': True, 'vocabulary': v,
                          'seed_reproducible': True, 'exact_readback_bytes': True, 'replay_turns': 3}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
