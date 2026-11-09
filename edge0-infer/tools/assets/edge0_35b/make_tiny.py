#!/usr/bin/env python3
"""Generate the deterministic tiny Edge0 35B asset fixture (qwen3next GGUF)."""

import argparse
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from asset_common import read_json, sha256, verify_source, write_json
from check_assets import check, write_catalog
from format_gguf import F16, F32, Q4_1, pack_q4_1, write
from edge0_35b.manifest import expand, expand_prerouter, prerouter_profile
from edge0_35b.metadata import lora_metadata, model_metadata, prerouter_metadata, tokenizer_metadata

TEXT_FILES = ['tokenizer.json', 'chat_template.jinja', 'generation_config.json']


def make_tiny(source: Path, out: Path) -> None:
    if out.exists() or out.is_symlink():
        raise ValueError(f'output already exists: {out}')
    config = read_json(Path(__file__).with_name('tiny_config.json'))
    lock = read_json(Path(__file__).with_name('source.lock.json'))
    verify_source(source, lock, TEXT_FILES)
    partial = out.with_name(out.name + '.partial')
    partial.mkdir(parents=True, exist_ok=False)
    p = config['parameters']
    rng = np.random.default_rng(config['seed'])
    for role in ['base', 'lora']:
        tensors = []
        for name, spec in expand(p, role).items():
            shape = spec['shape']
            array = rng.normal(0, config['weight_std'] if role == 'base' else config['lora_std'], tuple(reversed(shape))).astype(np.float32)
            if role == 'base':
                if 'norm' in name:
                    array.fill(1)
                elif name.endswith('ssm_a'):
                    array.fill(-1)
                elif name.endswith('ssm_dt.bias'):
                    array.fill(-2)
            elif 'ssm_ba' in name and name.endswith('.lora_b'):
                # Preserve two merged paths and their rank/alpha compensation.
                rank = p['lora_rank']
                for group in range(p['n_key_head']):
                    array[4 * group:4 * group + 2, rank:] = 0
                    array[4 * group + 2:4 * group + 4, :rank] = 0
                array *= 2
            payload = pack_q4_1(array) if spec['type'] == Q4_1 else array.astype('<f4' if spec['type'] == F32 else '<f2').tobytes()
            tensors.append((name, shape, spec['type'], payload))
        metadata = model_metadata(p) | tokenizer_metadata(source, p['n_vocab']) if role == 'base' else lora_metadata(p['lora_rank'])
        write(partial / f'{role}.gguf', metadata, tensors)
    # Drawn after base and LoRA so their bytes do not depend on the prerouter.
    q = p | prerouter_profile('tiny')
    tensors = []
    for name, spec in expand_prerouter(q).items():
        array = rng.normal(0, config['weight_std'], tuple(reversed(spec['shape'])))
        tensors.append((name, spec['shape'], F16, array.astype('<f2').tobytes()))
    write(partial / 'prerouter.gguf', prerouter_metadata(q, config['lora_alpha']), tensors)
    for name in TEXT_FILES:
        shutil.copyfile(source / name, partial / name)
    write_json(partial / 'model.json', {'format_version': 1, 'architecture': 'edge0_35b'})
    write_json(partial / 'provenance.json', {
        'generator': 'make_tiny.py', 'config': config, 'numpy_version': np.__version__,
        'text_source': {name: lock['files'][name] for name in TEXT_FILES},
        'numerical_reference': 'not_executed',
    })
    write_catalog(partial)
    report = check(partial)
    if report['status'] != 'passed':
        raise ValueError(f'asset check failed: {report}')
    if out.exists():
        raise ValueError(f'output appeared during generation: {out}')
    os.rename(partial, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        make_tiny(args.source, args.out)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'error: {exc}\n')
    print(f'Generated and checked {args.out}')


if __name__ == '__main__':
    main()
