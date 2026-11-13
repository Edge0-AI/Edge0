#!/usr/bin/env python3
"""Pack the Edge0 35B prerouter heads into prerouter.gguf.

Source layout (Edge0 c7ee73d docs/prerouter.md, windows/serve/prefetch.h): one
safetensors file with F16 tensors `layers.<owner>.{fc1,fc2,linear_init}.weight`
and string metadata kind/model/K/r/alpha/owners. Payload bytes are copied
unchanged; only names move to `blk.<owner>.prerouter_*.weight` and shapes to
GGML order. `add` attaches the file to an existing asset directory without
touching any other payload.
"""

import argparse
import json
import os
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asset_common import read_json, sha256, verify_source, write_json
from check_assets import check, write_catalog
from format_gguf import F16, Value, write

SOURCE = 'prerouter_edge0_35b.safetensors'
PARTS = ['fc1', 'fc2', 'linear_init']
NAME = re.compile(r'layers\.(\d+)\.(fc1|fc2|linear_init)\.weight')
HEADER_LIMIT = 16 << 20


def read_safetensors(path: Path) -> tuple[dict, dict]:
    """Return (string metadata, {name: (shape, bytes)}) for an all-F16 file."""
    # Read each payload into its final buffer instead of retaining a whole-file
    # copy and another copy of its data section alongside the returned tensors.
    with path.open('rb') as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError(f'{path.name}: truncated safetensors')
        n = struct.unpack('<Q', prefix)[0]
        size = os.fstat(file.fileno()).st_size
        if n > HEADER_LIMIT or 8 + n > size:
            raise ValueError(f'{path.name}: invalid header length')
        encoded = file.read(n)
        if len(encoded) != n:
            raise ValueError(f'{path.name}: truncated header')
        header = json.loads(encoded)
        metadata = header.pop('__metadata__', {})
        body_size = size - 8 - n
        tensors, end = {}, 0
        for name, spec in sorted(header.items(), key=lambda kv: kv[1]['data_offsets'][0]):
            start, stop = spec['data_offsets']
            if spec['dtype'] != 'F16':
                raise ValueError(f'{name}: expected F16, got {spec["dtype"]}')
            count = 1
            for d in spec['shape']:
                count *= d
            if start != end or stop < start or stop > body_size or stop - start != 2 * count:
                raise ValueError(f'{name}: inconsistent data offsets')
            payload = file.read(stop - start)
            if len(payload) != stop - start:
                raise ValueError(f'{path.name}: truncated payload')
            tensors[name] = (tuple(spec['shape']), payload)
            end = stop
        if end != body_size or file.read(1):
            raise ValueError(f'{path.name}: trailing or missing payload')
    return metadata, tensors


def convert(source: Path) -> tuple[dict[str, Value], list, dict]:
    """Map the safetensors heads to GGUF metadata and tensors (payload unchanged)."""
    metadata, tensors = read_safetensors(source)
    # The published file nests its JSON metadata as one string under "__metadata__".
    if set(metadata) == {'__metadata__'}:
        metadata = json.loads(metadata['__metadata__'])
    if metadata.get('kind') != 'prerouter' or metadata.get('format_version') != '1':
        raise ValueError('not a format_version 1 prerouter file')
    owners = json.loads(metadata['owners'])
    found = {}
    for name in tensors:
        m = NAME.fullmatch(name)
        if not m:
            raise ValueError(f'unexpected tensor: {name}')
        found.setdefault(int(m.group(1)), set()).add(m.group(2))
    if sorted(found) != owners or any(parts != set(PARTS) for parts in found.values()):
        raise ValueError('tensor set does not match owners')
    hidden = tensors[f'layers.{owners[0]}.fc1.weight'][0][0]
    gguf_metadata = {
        'general.architecture': Value(8, 'qwen3next'),
        'general.type': Value(8, 'prerouter'),
        'prerouter.top_k': Value(4, int(metadata['K'])),
        'prerouter.feed_forward_length': Value(4, hidden),
        'prerouter.owners': Value(9, owners, 4),
        'prerouter.lora_rank': Value(4, int(metadata['r'])),
        'prerouter.lora_alpha': Value(6, float(metadata['alpha'])),
    }
    out = []
    for owner in owners:
        for part in PARTS:
            shape, payload = tensors[f'layers.{owner}.{part}.weight']
            out.append((f'blk.{owner}.prerouter_{part}.weight', tuple(reversed(shape)), F16, payload))
    return gguf_metadata, out, metadata


def pack(source: Path, out: Path) -> dict:
    """Write prerouter.gguf and return the provenance record for it."""
    gguf_metadata, tensors, metadata = convert(source)
    write(out, gguf_metadata, tensors)
    return {'source_file': source.name, 'source_sha256': sha256(source), 'source_metadata': metadata,
            'packer_sha256': sha256(Path(__file__)), 'tensors': len(tensors)}


def add(assets: Path, source: Path, lock: dict | None = None) -> None:
    """Attach prerouter.gguf to an existing, otherwise unchanged asset directory."""
    lock = lock if lock is not None else read_json(Path(__file__).with_name('source.lock.json'))
    verify_source(source, lock, [SOURCE])
    target = assets / 'prerouter.gguf'
    if target.exists():
        raise ValueError(f'already present: {target}')
    before = {p.name: sha256(p) for p in assets.iterdir() if p.name not in ('assets.tsv', 'provenance.json')}
    saved = {name: (assets / name).read_bytes() for name in ('assets.tsv', 'provenance.json')}
    try:
        record = pack(source / SOURCE, target)
        provenance = read_json(assets / 'provenance.json')
        provenance['prerouter'] = record
        write_json(assets / 'provenance.json', provenance)
        write_catalog(assets)
        report = check(assets)
        if report['status'] != 'passed':
            raise ValueError(f'asset check failed: {report}')
    except BaseException:
        target.unlink(missing_ok=True)
        for name, data in saved.items():
            (assets / name).write_bytes(data)
        raise
    after = {p.name: sha256(p) for p in assets.iterdir() if p.name in before}
    if after != before:
        raise ValueError('existing payloads changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='directory with the registered source files')
    parser.add_argument('--assets', type=Path, required=True, help='existing asset directory')
    args = parser.parse_args()
    try:
        add(args.assets.resolve(), args.source.resolve())
    except (ValueError, OSError) as exc:
        parser.exit(1, f'error: {exc}\n')
    print(f'Added and checked {args.assets / "prerouter.gguf"}')


if __name__ == '__main__':
    main()
