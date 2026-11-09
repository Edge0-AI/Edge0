#!/usr/bin/env python3
"""Build 35B assets with the conversion scripts kept in this repository."""

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asset_common import read_json, sha256, snapshot, verify_source, write_json
from check_assets import check, write_catalog
from edge0_35b import prerouter

REPOSITORY = Path(__file__).resolve().parents[3]
CONVERTER_DIR = Path(__file__).resolve().with_name('edge0_convert')
CONVERTER = CONVERTER_DIR / 'convert_mlx_to_gguf.py'
TEXT_FILES = ['tokenizer.json', 'chat_template.jinja', 'generation_config.json']


def converter_identity() -> dict:
    origin = read_json(CONVERTER_DIR / 'ORIGIN.json')
    files = {p.name: sha256(p) for p in sorted(CONVERTER_DIR.iterdir()) if p.is_file()}
    return {'origin': origin, 'files_sha256': files}


class MakeError(ValueError):
    """A failure after the unfinished directory was created; the message names it."""


def keep_hint(partial: Path) -> str:
    return f'unfinished output kept at {partial}; delete it and rerun'


def recorded_path(path: Path, partial: Path) -> str:
    """Machine-independent spelling of a path for provenance.json."""
    path = Path(path).absolute()
    if path.is_relative_to(partial):
        return '<partial>/' + path.relative_to(partial).as_posix()
    if path.resolve().is_relative_to(REPOSITORY):
        return path.resolve().relative_to(REPOSITORY).as_posix()
    return '<outside-repository>/' + path.name


def make_assets(source: Path, out: Path, lock: dict | None = None, converter: Path = CONVERTER) -> None:
    if out.exists() or out.is_symlink():
        raise ValueError(f'output already exists: {out}')
    partial = out.with_name(out.name + '.partial')
    if partial.exists() or partial.is_symlink():
        raise ValueError(f'an earlier run left {partial}; delete it and rerun')
    lock = lock if lock is not None else read_json(Path(__file__).with_name('source.lock.json'))
    verify_source(source, lock)
    before = snapshot(source)
    partial.mkdir(parents=True, exist_ok=False)
    try:
        _convert(source, out, partial, lock, converter, before)
    except Exception as exc:
        raise MakeError(f'{exc}\n{keep_hint(partial)}') from exc
    shutil.rmtree(partial)


def _convert(source: Path, out: Path, partial: Path, lock: dict, converter: Path, before: dict) -> None:
    # The converter detects the model tier from the name of its --dir directory
    # ("edge0-35b"), so the staging directory name must stay as is.
    staging = partial / 'staging/edge0-35b'
    staging.mkdir(parents=True)
    converted = partial / 'converted'
    assets = partial / 'assets'
    try:
        # The converter input stays exactly as before the prerouter was packed.
        for name in lock['files']:
            if name != prerouter.SOURCE:
                (staging / name).symlink_to((source / name).resolve())
        command = [sys.executable, str(converter), '--dir', str(staging), '--out', str(converted)]
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        with out.with_name(out.name + '.log').open('wb') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        assets.mkdir()
        shutil.move(converted / 'edge0-35b.gguf', assets / 'base.gguf')
        shutil.move(converted / 'lora_edge0_35b-gguf.gguf', assets / 'lora.gguf')
        for name in TEXT_FILES:
            shutil.copyfile(source / name, assets / name)
        packed = prerouter.pack(source / prerouter.SOURCE, assets / 'prerouter.gguf')
        provenance = {
            'source': lock, 'converter': converter_identity(),
            'converter_manifest': read_json(converted / 'manifest.json'),
            'python_version': platform.python_version(),
            'dependencies': {'numpy': importlib.metadata.version('numpy')},
            'arguments': [recorded_path(Path(a), partial) if i in (0, 2, 4) else a for i, a in enumerate(command[1:])],
            'source_snapshot_before': before,
            'checks': {'status': 'pending'}, 'prerouter': packed,
        }
        write_json(assets / 'model.json', {'format_version': 1, 'architecture': 'edge0_35b'})
        write_json(assets / 'provenance.json', provenance)
        write_catalog(assets)
        report = check(assets)
        if report['status'] != 'passed':
            raise ValueError(f'asset check failed: {report}')
        provenance['checks'] = report
        write_json(assets / 'provenance.json', provenance)
        write_catalog(assets)
        if check(assets)['status'] != 'passed':
            raise ValueError('final asset check failed')
    finally:
        if snapshot(source) != before:
            raise ValueError('source directory changed during conversion')
    if out.exists():
        raise ValueError(f'output appeared during conversion: {out}')
    os.rename(assets, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='directory with the registered source files')
    parser.add_argument('--out', type=Path, required=True, help='new output directory')
    args = parser.parse_args()
    try:
        make_assets(args.source.resolve(), args.out.absolute())
    except (ValueError, OSError) as exc:
        parser.exit(1, f'error: {exc}\n')
    print(f'Converted and checked {args.out}')


if __name__ == '__main__':
    main()
