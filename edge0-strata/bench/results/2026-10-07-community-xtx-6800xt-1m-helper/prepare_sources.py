#!/usr/bin/env python3
"""Create pinned source worktrees; does not install packages, build or start a model."""
import argparse
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=HERE.parents[2])
    args = parser.parse_args()
    repo, root = args.repo.resolve(), args.root.resolve()
    layout = json.loads((HERE / 'source-layout.json').read_text())
    for spec in layout.values():
        if not isinstance(spec, dict) or 'commit' not in spec:
            continue
        dst = root / spec['path']
        if dst.exists():
            actual = subprocess.check_output(['git', '-C', str(dst), 'rev-parse', 'HEAD'], text=True).strip()
            if actual != spec['commit']:
                raise RuntimeError(f'{dst} exists at a different commit; choose another --root')
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if spec.get('repository'):
            subprocess.run(['git', 'clone', '--no-checkout', spec['repository'], str(dst)], check=True)
            subprocess.run(['git', '-C', str(dst), 'checkout', '--detach', spec['commit']], check=True)
        else:
            subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '--detach', str(dst), spec['commit']], check=True)
    print('Pinned sources prepared. Follow README for packages, artifacts, builds and API startup.')


if __name__ == '__main__':
    main()
