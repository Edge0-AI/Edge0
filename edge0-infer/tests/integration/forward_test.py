"""CLI format and determinism checks, independent of the numerical answer judgement."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    parser.add_argument('--backend', default='cpu', choices=['cpu', 'metal'])
    args = parser.parse_args()
    root = Path(os.environ.get('MOE_TINY', '/nonexistent'))
    if not (root / 'tiny-a/assets.tsv').exists():
        print('SKIP: missing MOE_TINY tiny-a')
        return 77
    answer = Path(__file__).resolve().parents[2] / f'tests/fixtures/answers/tiny-lora-{args.backend}-flash/plain-mixed'
    tokens = json.loads((answer / 'tokens.json').read_text())
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        for name in ['first', 'second', 'observed']:
            command = [args.cli, 'forward', '--model', str(root / 'tiny-a'), '--device', args.backend,
                       '--tokens', str(answer / 'tokens.json'), '--out', str(work / name)]
            if name == 'observed':
                command.append('--observe')
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 77:
                print(result.stderr)
                return 77
            assert result.returncode == 0, result.stdout + result.stderr
            report = json.loads((work / name / 'report.json').read_text())
            assert report['budget_source'] == 'device_default'
            assert 0 < report['budget_peak'] <= report['budget_limit']
            assert report['budget_current'] == 0
            assert report['weight_registrations'] == [255] * (1 + len(tokens['generated']))
            metadata = json.loads((work / name / 'tokens.json').read_text())
            assert metadata['row_semantics'].startswith('prefill positions')
            assert {k: metadata[k] for k in ['input', 'generated', 'logits_shape']} == \
                {k: tokens[k] for k in ['input', 'generated', 'logits_shape']}
            assert 'chunking' not in metadata
            assert (work / name / 'logits.f32').stat().st_size == 4 * tokens['logits_shape'][0] * tokens['logits_shape'][1]
        assert (work / 'first/logits.f32').read_bytes() == (work / 'second/logits.f32').read_bytes()
        actual = json.loads((work / 'observed/layers/index.json').read_text())
        assert all(r.keys() == {'name', 'file', 'step', 'position', 'type', 'shape'} for r in actual)
        records = {(r['step'], r['name']): r for r in actual}
        layers = {r['name'].rsplit('-', 1)[1] for r in actual if r['name'].startswith('ffn_moe_topk-')}
        assert layers
        # The recorded routing steps and their position counts are those the answer stores.
        for step, positions in tokens['route_steps']:
            for layer in layers:
                ids, probs = records[(step, f'ffn_moe_topk-{layer}')], records[(step, f'ffn_moe_probs-{layer}')]
                assert ids['type'] == 26 and probs['type'] == 0
                assert ids['shape'][1] == probs['shape'][1] == positions, (step, layer)
        assert {step for step, _ in records} == {step for step, _ in tokens['route_steps']}
        usage = subprocess.run([args.cli], capture_output=True, text=True)
        assert usage.returncode == 2 and 'forward' in usage.stderr
    print('forward schema, deterministic logits, weight registrations and cleanup passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
