"""35B long-input bitwise checks and explicit 64-step route trace."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    model = Path(os.environ.get('MOE_ASSETS_35B', '/nonexistent'))
    if not (model / 'prerouter.gguf').exists():
        return 77
    with tempfile.TemporaryDirectory() as temp:
        root = args.out or Path(temp)
        root.mkdir(parents=True, exist_ok=True)
        for label, mode in [('predicted', 'predicted'), ('repeat', 'predicted')]:
            command = [args.cli, 'run', '--model', str(model), '--device', 'metal', '--user-file',
                       'tests/fixtures/bench/library-report-prompt.txt', '--temperature', '0', '--presence-penalty', '0', '--max-tokens', '65',
                       '--out', str(root / label), '--model-option', f'routing={mode}']
            if label == 'predicted':
                command += ['--route-trace', str(root / 'route.jsonl')]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 77:
                return 77
            assert result.returncode == 0, result.stderr
        for a, b in [('predicted', 'repeat')]:
            for name in ['tokens.json', 'logits.f32']:
                assert (root/a/name).read_bytes() == (root/b/name).read_bytes(), (a, b, name)
        rows = [json.loads(line) for line in (root/'route.jsonl').read_text().splitlines()]
        assert len(rows) == 64
        print('35B long input: 65 generated tokens / 64 decode steps; predicted trace on/off=repeat bytewise; 64 trace rows')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
