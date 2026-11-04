"""Shared deterministic file and source-identity helpers."""

import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    # LF on every host: these files are asset payload and their digests are compared
    # across platforms (text mode would write CRLF on Windows).
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + '\n',
                    encoding='utf-8', newline='\n')


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}


def verify_source(source: Path, lock: dict, names=None) -> None:
    for name in names if names is not None else lock['files']:
        path = source / name
        if not path.is_file():
            raise ValueError(f'missing source file: {name}')
        if sha256(path) != lock['files'][name]:
            raise ValueError(f'source digest mismatch: {name}')


class AssetError(ValueError):
    """A checker finding with a stable category."""

    def __init__(self, category: str, detail: str):
        super().__init__(detail)
        self.category = category


def fail(category: str, detail: str):
    raise AssetError(category, detail)
