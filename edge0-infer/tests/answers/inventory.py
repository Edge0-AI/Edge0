#!/usr/bin/env python3
"""The answer suites in use; the answer directory must hold exactly these."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import answer_format as fmt

ANSWERS = Path(__file__).resolve().parents[1] / 'fixtures/answers'
IN_USE = (
    'tiny-base-cpu-flash',
    'tiny-lora-cpu-flash',
    'tiny-base-metal-flash',
    'tiny-lora-metal-flash',
    'tiny-base-cpu-flash-split',
    'tiny-lora-cpu-flash-split',
    'tiny-base-metal-flash-split',
    'tiny-lora-metal-flash-split',
    '35b-lora-cpu-flash-split',
)
ROOT_FILES = {'cases.json', 'README.md'}


def problems(root: Path = ANSWERS) -> list[str]:
    found = []
    if not root.is_dir():
        return [f'answer directory not found: {root}']
    names = {p.name for p in root.iterdir()}
    for name in sorted(names - set(IN_USE) - ROOT_FILES):
        found.append(f'not in the in-use list: {name}')
    for name in IN_USE:
        suite = root / name
        if not suite.is_dir():
            found.append(f'missing suite: {name}')
            continue
        try:
            identity = fmt.suite_identity(suite)
        except (ValueError, OSError) as exc:
            found.append(f'{name}: {exc}')
            continue
        if identity['suite'] != name:
            found.append(f'{name}: identity names suite {identity["suite"]}')
        expected = set(identity['cases']) | {'identity.json', 'SHA256SUMS', 'run.json'}
        if {p.name for p in suite.iterdir()} != expected:
            found.append(f'{name}: unexpected entries {sorted({p.name for p in suite.iterdir()} ^ expected)}')
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ANSWERS)
    args = parser.parse_args()
    found = problems(args.root)
    for item in found:
        print(item)
    print('answer inventory ' + ('failed' if found else f'passed: {len(IN_USE)} suites'))
    return 1 if found else 0


if __name__ == '__main__':
    raise SystemExit(main())
