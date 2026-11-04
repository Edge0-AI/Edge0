"""Explicit CLI budget admission and three-part weight requirements."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    parser.add_argument('--backend', choices=['cpu', 'metal'], default='cpu')
    args = parser.parse_args()
    model = Path(os.environ.get('MOE_TINY', '/nonexistent')) / 'tiny-a'
    if not (model / 'assets.tsv').is_file():
        return 77
    base = [args.cli, 'run', '--model', str(model), '--device', args.backend, '--user', 'hello', '--max-tokens', '1']
    for budget in ('1', '0', '18446744073709551615'):
        result = subprocess.run(base + ['--memory-budget', budget], capture_output=True, text=True)
        assert result.returncode != 0, result.stdout
        if budget == '1':
            assert 'budget_exceeded' in result.stderr, result.stderr
            for part in ('resident=', 'pools=', 'temporary='):
                assert part in result.stderr, result.stderr
        else:
            assert 'invalid_argument' in result.stderr, result.stderr
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
