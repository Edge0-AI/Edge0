"""Independent F32 head oracle using only GGUF weights and observed features."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/assets'))
from format_gguf import Reader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    parser.add_argument('--profile', choices=['tiny', '35b'], default='tiny')
    parser.add_argument('--backend', choices=['cpu', 'metal'], default='cpu')
    args = parser.parse_args()
    reference = Path(os.environ.get('MOE_TINY', '/nonexistent'))
    model = Path(os.environ.get('MOE_ASSETS_35B', '/nonexistent')) if args.profile == '35b' else reference / 'tiny-a'
    if not (model / 'prerouter.gguf').exists():
        return 77
    reader = Reader(model / 'prerouter.gguf')
    owners = reader.metadata['prerouter.owners'].data
    selected = [6, 22, 38] if args.profile == '35b' else owners
    steps = 4 if args.profile == '35b' else 8
    experts = reader.tensors[f'blk.{owners[0]}.prerouter_fc2.weight'].shape[1]
    top = reader.metadata['prerouter.top_k'].data
    weights = {}
    for layer in selected:
        for name in ['fc1', 'fc2', 'linear_init']:
            key = f'blk.{layer}.prerouter_{name}.weight'
            weights[layer, name] = np.frombuffer(reader.data(key), dtype='<f2').astype(np.float32).reshape(reader.tensors[key].shape[::-1])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tokens = root / 'tokens.json'
        tokens.write_text(json.dumps(dict(input=[1, 2, 3], generated=list(range(10, 10+steps)))))
        command = [args.cli, 'forward', '--model', str(model), '--device', args.backend, '--tokens', str(tokens),
                   '--out', str(root / 'out'), '--model-option', 'routing=predicted', '--observe', '--observe-steps', str(steps)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 77:
            return 77
        assert result.returncode == 0, result.stderr
        records = json.loads((root / 'out/layers/index.json').read_text())
        def tensor(step, name):
            record = next(r for r in records if r['step'] == step and r['name'] == name)
            dtype = '<i4' if record['type'] == 26 else '<f4'
            return np.fromfile(root / 'out/layers' / record['file'], dtype=dtype)
        assert not any(r['step'] < 0 and r['name'].startswith('prerouter_') for r in records)
        maximum = 0.0
        for step in range(steps):
            for layer in selected:
                x = tensor(step, f'prerouter_in-{layer}')
                current = np.zeros(experts, np.float32)
                def executed(t, owner):
                    name = 'executed' if t > 0 and owner > owners[0] else 'ffn_moe_topk'
                    return tensor(t, f'{name}-{owner}')
                current[executed(step, layer)] = 1
                previous = np.zeros(experts, np.float32)
                if step:
                    previous[executed(step-1, layer)] = 1
                np.testing.assert_array_equal(x[-2*experts:-experts], current)
                np.testing.assert_array_equal(x[-experts:], previous)
                hidden = weights[layer, 'fc1'] @ x
                erf = np.array([math.erf(float(v) / math.sqrt(2)) for v in hidden], dtype=np.float32)
                expected = weights[layer, 'fc2'] @ (0.5 * hidden * (1+erf)) + weights[layer, 'linear_init'] @ x
                actual = tensor(step, f'prerouter_out-{layer}')
                error = float(np.max(np.abs(actual-expected)))
                assert error <= 2e-2, (step, layer, error)
                maximum = max(maximum, error)
                ordered = np.argsort(-expected)
                if expected[ordered[top-1]] - expected[ordered[top]] >= 1e-2:
                    assert set(ordered[:top]) == set(np.argsort(-actual)[:top]), (step, layer)
        for step in range(1, steps):
            for owner in owners:
                np.testing.assert_array_equal(tensor(step, f'executed-{owner+1}'), tensor(step-1, f'prerouter_ids-{owner}'))
                np.testing.assert_array_equal(tensor(step, f'executed_scores-{owner+1}'), tensor(step-1, f'prerouter_scores-{owner}'))
        print(json.dumps(dict(routing="predicted", profile=args.profile, backend=args.backend, steps=steps, owners=selected, max_abs=maximum)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
