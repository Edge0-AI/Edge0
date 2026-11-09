"""On-disk format of the repository's own single-turn answers.

A suite directory holds `identity.json`, `SHA256SUMS`, an unhashed `run.json`
and one directory per case with `tokens.json`, `logits.bin` and `routes.bin`.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
TOP = 32
PROBE_COUNT = 32
PROBE_SEED = 20261021
CASES = ('chat-intro', 'plain-mixed', 'long-mixed')
UNHASHED = {'run.json', 'SHA256SUMS'}
IDENTITY_FIELDS = (
    'format_version', 'suite', 'assets_sha256', 'lora', 'backend', 'chunking', 'model_options',
    'generation', 'engine_commit', 'engine_tracked_clean', 'exporter_sha256', 'top', 'probe_seed',
    'probes', 'cases', 'cases_sha256', 'vocabulary', 'route_layers', 'route_top_k', 'route_experts',
)


def row_dtype(top: int = TOP, probes: int = PROBE_COUNT) -> np.dtype:
    return np.dtype([('tokens', '<i4', (top,)), ('values', '<f4', (top,)), ('probes', '<f4', (probes,))])


def probe_positions(vocabulary: int, seed: int = PROBE_SEED, count: int = PROBE_COUNT) -> list[int]:
    chosen = np.random.default_rng(seed).choice(vocabulary, size=count, replace=False)
    return sorted(int(i) for i in chosen)


def top_tokens(row: np.ndarray, top: int = TOP) -> np.ndarray:
    """The `top` largest values, ordered by value descending and token ascending."""
    threshold = np.partition(row, -top)[-top]
    candidates = np.flatnonzero(row >= threshold)
    order = np.lexsort((candidates, -row[candidates].astype(np.float64)))
    return candidates[order[:top]]


def encode_logits(logits: np.ndarray, probes: list[int], top: int = TOP) -> np.ndarray:
    rows = np.zeros(len(logits), dtype=row_dtype(top, len(probes)))
    for i, row in enumerate(logits):
        tokens = top_tokens(row, top)
        rows[i]['tokens'] = tokens
        rows[i]['values'] = row[tokens]
        rows[i]['probes'] = row[probes]
    return rows


def read_logits(path: Path, identity: dict, rows: int) -> np.ndarray:
    dtype = row_dtype(identity['top'], len(identity['probes']))
    data = np.fromfile(path, dtype=dtype)
    if len(data) != rows:
        raise ValueError(f'answer logits row count differs: {len(data)} != {rows}')
    for field in ('values', 'probes'):
        if not np.isfinite(data[field]).all():
            raise ValueError('answer logits are not finite')
    if np.any(data['tokens'] < 0) or np.any(data['tokens'] >= identity['vocabulary']):
        raise ValueError('answer token outside vocabulary')
    return data


def read_routes(path: Path, identity: dict, steps: list) -> dict:
    """Return {(step, layer): ids[positions, top_k]} in the stored record order."""
    layers, k = identity['route_layers'], identity['route_top_k']
    data = np.fromfile(path, dtype=np.uint8)
    expected = sum(n for _, n in steps) * layers * k
    if data.size != expected:
        raise ValueError(f'answer routes size differs: {data.size} != {expected}')
    if np.any(data >= identity['route_experts']):
        raise ValueError('answer expert outside expert count')
    result, offset = {}, 0
    for step, positions in steps:
        for layer in range(layers):
            size = positions * k
            result[(step, layer)] = data[offset:offset + size].reshape(positions, k).astype(np.int64)
            offset += size
    return result


def encode_routes(records: dict, steps: list, layers: int) -> bytes:
    parts = []
    for step, _ in steps:
        for layer in range(layers):
            ids = records[(step, layer)]
            if np.any(ids < 0) or np.any(ids > 255):
                raise ValueError('expert ID does not fit in one byte')
            parts.append(ids.astype(np.uint8).tobytes())
    return b''.join(parts)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sums(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.relative_to(root).as_posix() not in UNHASHED}


def write_sums(root: Path) -> None:
    (root / 'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name, digest in sums(root).items()))


def read_sums(root: Path) -> dict[str, str]:
    path = root / 'SHA256SUMS'
    if not path.is_file():
        raise ValueError(f'missing hash list: {path}')
    result = {}
    for line in path.read_text().splitlines():
        digest, name = line.split('  ', 1)
        if name in result or Path(name).is_absolute() or '..' in Path(name).parts or name in UNHASHED:
            raise ValueError('invalid hash list')
        result[name] = digest
    if result != sums(root):
        raise ValueError(f'hash list does not match files: {root}')
    return result


def validate_identity(identity: dict) -> None:
    missing = [key for key in IDENTITY_FIELDS if identity.get(key) is None]
    if missing:
        raise ValueError(f'answer identity is missing fields: {missing}')
    if identity['format_version'] != FORMAT_VERSION:
        raise ValueError('unsupported answer format version')
    if identity['backend'] not in ('cpu', 'metal') or identity['chunking'] not in ('whole', 'split-last'):
        raise ValueError('invalid answer backend or chunking')
    if len(identity['probes']) != PROBE_COUNT or identity['top'] != TOP:
        raise ValueError('invalid answer top or probe count')
    if not identity['assets_sha256'] or not identity['engine_commit']:
        raise ValueError('answer identity has empty asset or engine fields')


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def suite_identity(suite: Path) -> dict:
    read_sums(suite)
    identity = read_json(suite / 'identity.json')
    validate_identity(identity)
    return identity


# Files that describe how the assets were made rather than what they compute. Re-making the
# assets from the same source weights changes them (timestamps, paths) but not the payload.
NON_PAYLOAD = ('assets.tsv', 'provenance.json')


def catalog_digests(model: Path) -> dict[str, str]:
    """Payload digests recorded by the asset catalog (every file except NON_PAYLOAD)."""
    lines = (model / 'assets.tsv').read_text(encoding='utf-8').splitlines()
    if not lines or lines[0].split('\t') != ['role', 'bytes', 'sha256', 'filename']:
        raise ValueError('invalid asset catalog')
    result = {}
    for line in lines[1:]:
        _, _, digest, name = line.split('\t')
        if name not in NON_PAYLOAD:
            result[name] = digest
    return result


def check_assets(identity: dict, model: Path) -> None:
    if catalog_digests(Path(model)) != identity['assets_sha256']:
        raise ValueError(f'answer asset identity differs from {model}')
