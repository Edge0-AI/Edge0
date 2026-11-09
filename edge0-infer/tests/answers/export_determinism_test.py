"""Export the same tiny suite twice and require byte-identical hash lists."""
import argparse
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export import Unavailable, export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    parser.add_argument('--backend', default='cpu', choices=['cpu', 'metal'])
    args = parser.parse_args()
    model = Path(os.environ.get('MOE_TINY', '/nonexistent')) / 'tiny-a'
    if not (model / 'assets.tsv').is_file():
        print('SKIP: missing MOE_TINY tiny-a')
        return 77
    suite = f'tiny-lora-{args.backend}-flash-split'
    with tempfile.TemporaryDirectory() as directory:
        roots = [Path(directory) / name / suite for name in ('first', 'second')]
        try:
            for root in roots:
                export(args.cli, model, suite, root)
        except Unavailable as exc:
            print(f'SKIP: {exc}')
            return 77
        first, second = [(root / 'SHA256SUMS').read_bytes() for root in roots]
        print(first.decode(), end='')
        assert first == second, 'repeated export produced different hash lists'
        assert len(first.splitlines()) == 1 + 3 * 3
        for root in roots:
            leftovers = [p.name for p in root.parent.iterdir() if p.name != suite]
            assert not leftovers, leftovers
    print(f'{suite}: two exports have identical SHA256SUMS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
