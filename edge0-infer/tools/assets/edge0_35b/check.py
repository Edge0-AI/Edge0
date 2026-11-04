"""Model check for edge0_35b assets: roles, model description, GGUF metadata and tensor contract."""

from pathlib import Path

from asset_common import fail, read_json
from format_gguf import Reader
from edge0_35b import manifest

ROLES = {
    'base': 'base.gguf', 'lora': 'lora.gguf', 'tokenizer': 'tokenizer.json',
    'chat_template': 'chat_template.jinja', 'generation_config': 'generation_config.json',
    'model': 'model.json', 'provenance': 'provenance.json', 'prerouter': 'prerouter.gguf',
}
GGUF_FILES = ['base.gguf', 'lora.gguf', 'prerouter.gguf']
TEXT_FILES = ['tokenizer.json', 'chat_template.jinja', 'generation_config.json', 'model.json']
MODEL = {'format_version': 1, 'architecture': 'edge0_35b'}


def roles(rows: dict) -> None:
    if set(ROLES) - set(rows):
        fail('missing_file', f'missing roles: {sorted(set(ROLES) - set(rows))}')
    for role, row in rows.items():
        if role not in ROLES or row['filename'] != ROLES[role]:
            fail('invalid_manifest', f'invalid catalog role: {role}')


def check(root: Path, rows: dict, report: dict) -> None:
    roles(rows)
    report['structure'] = 'passed'
    if read_json(root / 'model.json') != MODEL:
        fail('hyperparameter_mismatch', 'model.json')
    read_json(root / 'provenance.json')
    base = Reader(root / 'base.gguf')
    lora = Reader(root / 'lora.gguf')
    prerouter = Reader(root / 'prerouter.gguf')
    contract = manifest.load()
    try:
        p = manifest.metadata_parameters(base.metadata)
        profiles = [k for k, v in contract['profiles'].items() if all(p[k] == v[k] for k in p)]
        if len(profiles) != 1:
            fail('hyperparameter_mismatch', 'metadata does not match a declared profile')
        p = contract['profiles'][profiles[0]] | manifest.prerouter_profile(profiles[0])
        required = {
            'general.architecture': 'qwen3next',
            'qwen3next.attention.value_length': p['head_dim'],
            'qwen3next.ssm.inner_size': p['state_dim'] * p['n_value_head'],
            'qwen3next.attention.recurrent_layers': [(i + 1) % p['full_attention_interval'] != 0 for i in range(p['n_layer'])],
        }
        for key, value in required.items():
            if base.metadata[key].data != value:
                fail('hyperparameter_mismatch', key)
        for key, value in {'general.architecture': 'qwen3next', 'general.type': 'adapter', 'adapter.type': 'lora', 'adapter.lora.alpha': float(p['lora_rank'] * 2)}.items():
            if lora.metadata[key].data != value:
                fail('hyperparameter_mismatch', f'lora: {key}')
        owners = list(range(p['prerouter_first_owner'], p['n_layer'] - 1))
        for key, value in {'general.architecture': 'qwen3next', 'general.type': 'prerouter', 'prerouter.top_k': p['top_k'],
                           'prerouter.feed_forward_length': p['prerouter_hidden'], 'prerouter.owners': owners}.items():
            if prerouter.metadata[key].data != value:
                fail('hyperparameter_mismatch', f'prerouter: {key}')
    except (KeyError, ValueError, TypeError) as exc:
        if hasattr(exc, 'category'):
            raise
        fail('hyperparameter_mismatch', str(exc))
    for role, reader in [('base', base), ('lora', lora), ('prerouter', prerouter)]:
        expected = manifest.expand_prerouter(p) if role == 'prerouter' else manifest.expand(p, role)
        if set(expected) - set(reader.tensors):
            fail('missing_tensor', f'{role}: {sorted(set(expected) - set(reader.tensors))}')
        if set(reader.tensors) - set(expected):
            fail('extra_tensor', f'{role}: {sorted(set(reader.tensors) - set(expected))}')
        for name, spec in expected.items():
            tensor = reader.tensors[name]
            if tensor.type != spec['type']:
                fail('representation_mismatch', f'{role}: {name}')
            if tensor.shape != spec['shape']:
                fail('shape_mismatch', f'{role}: {name}')
            if reader.alignment != spec['alignment'] or (reader.data_offset + tensor.offset) % spec['alignment']:
                fail('unaligned_tensor', f'{role}: {name}')
