#!/usr/bin/env python3
"""Validate asset catalogs and compare reproducible tensor payloads.

The framework checks the catalog columns, file sizes and digests, unlisted
files and the structure of `model.json`, then dispatches on its
`architecture` to the model check registered in MODELS. A model check module
provides ROLES (role -> file name), GGUF_FILES, TEXT_FILES and
check(root, catalog_rows, report), which sets report['structure'] once the
roles match.
"""

import argparse
import csv
import importlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from asset_common import AssetError, fail, read_json, sha256
from format_gguf import Reader

# Registered models: architecture identifier -> model check module.
MODELS = {'edge0_35b': 'edge0_35b.check'}

CATALOG_COLUMNS = ['role', 'bytes', 'sha256', 'filename']


def model_module(architecture: str):
    if architecture not in MODELS:
        fail('unsupported_model', f'{architecture!r} is not registered; registered: {sorted(MODELS)}')
    return importlib.import_module(MODELS[architecture])


def catalog(root: Path) -> dict:
    """Framework catalog check: columns, unique plain file names, sizes, digests, no unlisted files."""
    if not (root / 'assets.tsv').is_file():
        fail('missing_manifest', 'assets.tsv')
    with (root / 'assets.tsv').open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if reader.fieldnames != CATALOG_COLUMNS:
            fail('invalid_manifest', 'invalid catalog columns')
        rows = list(reader)
    result, names = {}, set()
    for row in rows:
        role, name = row['role'], row['filename']
        if not role or role in result or not name or name in names or Path(name).name != name or name == 'assets.tsv':
            fail('invalid_manifest', f'invalid or duplicate catalog entry: {role}')
        result[role] = row
        names.add(name)
    actual = {p.name for p in root.iterdir()}
    if actual - names - {'assets.tsv'}:
        fail('unlisted_file', str(sorted(actual - names - {'assets.tsv'})))
    for row in result.values():
        path = root / row['filename']
        if not path.is_file() or path.is_symlink():
            fail('missing_file', row['filename'])
        try:
            size = int(row['bytes'])
        except (ValueError, TypeError):
            fail('invalid_manifest', row['filename'])
        if size != path.stat().st_size:
            fail('size_mismatch', row['filename'])
        if row['sha256'] != sha256(path):
            fail('digest_mismatch', row['filename'])
    return result


def model_description(root: Path) -> dict:
    path = root / 'model.json'
    if not path.is_file():
        fail('missing_file', 'model.json')
    value = read_json(path)
    if not isinstance(value, dict) or value.get('format_version') != 1 or not isinstance(value.get('architecture'), str):
        fail('invalid_manifest', 'model.json needs format_version 1 and an architecture string')
    return value


def write_catalog(root: Path, roles: dict | None = None) -> None:
    """Write assets.tsv in role order; roles default to those of the registered model."""
    if roles is None:
        roles = model_module(model_description(root)['architecture']).ROLES
    with (root / 'assets.tsv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(CATALOG_COLUMNS)
        for role, name in roles.items():
            p = root / name
            writer.writerow([role, p.stat().st_size, sha256(p), name])


def check(root: str | Path) -> dict:
    root = Path(root)
    report = {'status': 'failed', 'model': None, 'structure': 'not_executed', 'integrity': 'not_executed',
              'tensors': 'not_executed', 'numerical': {'status': 'not_executed', 'reason': 'No verified numerical reference supplied.'}, 'issues': []}
    try:
        rows = catalog(root)
        report['integrity'] = 'passed'
        architecture = model_description(root)['architecture']
        module = model_module(architecture)
        report['model'] = architecture
        module.check(root, rows, report)
        report['tensors'] = report['status'] = 'passed'
    except AssetError as exc:
        report['issues'].append({'category': exc.category, 'detail': str(exc)})
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        report['issues'].append({'category': 'invalid_asset', 'detail': str(exc)})
    return report


def compare(a: str | Path, b: str | Path) -> dict:
    a, b = Path(a), Path(b)
    checks = [check(a), check(b)]
    if any(r['status'] != 'passed' for r in checks):
        return {'status': 'failed', 'checks': checks}
    if checks[0]['model'] != checks[1]['model']:
        return {'status': 'failed', 'differences': [{'kind': 'model', 'models': [r['model'] for r in checks]}]}
    module = model_module(checks[0]['model'])
    differences = []
    for name in module.GGUF_FILES:
        left, right = Reader(a / name), Reader(b / name)
        if left.metadata != right.metadata:
            differences.append({'file': name, 'kind': 'metadata'})
        for tensor in left.tensors:
            if tensor not in right.tensors or left.tensors[tensor].shape != right.tensors[tensor].shape or left.tensors[tensor].type != right.tensors[tensor].type or left.digest(tensor) != right.digest(tensor):
                differences.append({'file': name, 'kind': 'tensor', 'tensor': tensor})
        for tensor in sorted(set(right.tensors) - set(left.tensors)):
            differences.append({'file': name, 'kind': 'tensor', 'tensor': tensor})
    for name in module.TEXT_FILES:
        if (a / name).read_bytes() != (b / name).read_bytes():
            differences.append({'file': name, 'kind': 'text'})
    return {'status': 'failed' if differences else 'passed', 'differences': differences,
            'whole_file_equal': {n: sha256(a / n) == sha256(b / n) for n in module.ROLES.values()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check').add_argument('assets', type=Path)
    c = sub.add_parser('compare')
    c.add_argument('left', type=Path)
    c.add_argument('right', type=Path)
    args = parser.parse_args()
    report = check(args.assets) if args.command == 'check' else compare(args.left, args.right)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
