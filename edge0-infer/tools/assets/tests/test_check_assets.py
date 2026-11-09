from copy import deepcopy
import shutil
import struct

import pytest

from asset_common import write_json
from check_assets import MODELS, check, compare, write_catalog
from edge0_35b import check as edge0_check
from format_gguf import Reader, Value, tensor_size, write
from edge0_35b import manifest
from edge0_35b.metadata import lora_metadata, model_metadata, prerouter_metadata

pytestmark = pytest.mark.l1


@pytest.fixture
def asset(tmp_path, monkeypatch):
    # Only the test contract reduces vocabulary. Production profiles remain fixed.
    contract = deepcopy(manifest.load())
    p = contract['profiles']['tiny']
    p['n_vocab'] = 32
    monkeypatch.setattr(manifest, 'load', lambda: contract)
    root = tmp_path / 'asset'
    root.mkdir()
    for role, meta in [('base', model_metadata(p) | {'tokenizer.ggml.tokens': Value(9, [str(i) for i in range(32)], 8)}), ('lora', lora_metadata(p['lora_rank']))]:
        tensors = [(name, v['shape'], v['type'], bytes(tensor_size(v['shape'], v['type']))) for name, v in manifest.expand(p, role).items()]
        write(root / f'{role}.gguf', meta, tensors)
    q = p | manifest.prerouter_profile('tiny')
    tensors = [(name, v['shape'], v['type'], bytes(tensor_size(v['shape'], v['type']))) for name, v in manifest.expand_prerouter(q).items()]
    write(root / 'prerouter.gguf', prerouter_metadata(q, 8.0), tensors)
    for name in ['tokenizer.json', 'generation_config.json', 'provenance.json']:
        write_json(root / name, {})
    write_json(root / 'model.json', {'format_version': 1, 'architecture': 'edge0_35b'})
    (root / 'chat_template.jinja').write_text('test')
    write_catalog(root)
    return root


@pytest.mark.parametrize('identity', ['qwen3next', 'edge0_8b'])
def test_rejects_unregistered_model(asset, identity):
    assert Reader(asset / 'base.gguf').metadata['general.architecture'].data == 'qwen3next'
    write_json(asset / 'model.json', {'format_version': 1, 'architecture': identity})
    write_catalog(asset, edge0_check.ROLES)
    report = check(asset)
    assert report['status'] == 'failed' and report['model'] is None
    assert report['issues'][0]['category'] == 'unsupported_model'
    assert identity in report['issues'][0]['detail'] and "'edge0_35b'" in report['issues'][0]['detail']
    assert report['tensors'] == 'not_executed'


def test_registered_model_is_reported(asset):
    report = check(asset)
    assert report['model'] == 'edge0_35b' and report['structure'] == report['integrity'] == 'passed'
    assert sorted(MODELS) == ['edge0_35b']


@pytest.mark.parametrize('description', [{'architecture': 'edge0_35b'}, {'format_version': 1}, ['edge0_35b']])
def test_rejects_malformed_model_description(asset, description):
    write_json(asset / 'model.json', description)
    write_catalog(asset, edge0_check.ROLES)
    assert check(asset)['issues'][0]['category'] == 'invalid_manifest'


def test_model_description_with_extra_fields_fails_model_check(asset):
    write_json(asset / 'model.json', {'format_version': 1, 'architecture': 'edge0_35b', 'extra': 1})
    write_catalog(asset)
    assert check(asset)['issues'][0]['category'] == 'hyperparameter_mismatch'


def rewrite(root, mutate):
    path = root / 'base.gguf'
    reader = Reader(path)
    tensors = [(t.name, t.shape, t.type, reader.data(t.name)) for t in reader.tensors.values()]
    mutate(reader.metadata, tensors)
    path.unlink()
    write(path, reader.metadata, tensors)
    write_catalog(root)


def test_pass_and_compare(asset, tmp_path):
    assert check(asset)['status'] == 'passed'
    assert check(asset)['numerical']['status'] == 'not_executed'
    other = tmp_path / 'other'
    shutil.copytree(asset, other)
    assert compare(asset, other)['status'] == 'passed'
    reader = Reader(other / 'base.gguf')
    first = next(iter(reader.tensors))
    with (other / 'base.gguf').open('r+b') as f:
        f.seek(reader.data_offset)
        f.write(b'\x01')
    write_catalog(other)
    result = compare(asset, other)
    assert result['status'] == 'failed'
    assert result['differences'] == [{'file': 'base.gguf', 'kind': 'tensor', 'tensor': first}]


@pytest.mark.parametrize('category', ['missing_manifest', 'size_mismatch', 'digest_mismatch', 'unlisted_file', 'missing_tensor', 'extra_tensor', 'representation_mismatch', 'shape_mismatch', 'unaligned_tensor', 'hyperparameter_mismatch'])
def test_reject(asset, category):
    if category == 'missing_manifest':
        (asset / 'assets.tsv').unlink()
    elif category == 'size_mismatch':
        (asset / 'chat_template.jinja').write_text('longer text')
    elif category == 'digest_mismatch':
        (asset / 'chat_template.jinja').write_text('xxxx')
    elif category == 'unlisted_file':
        (asset / 'surprise').write_text('extra')
    elif category == 'missing_tensor':
        rewrite(asset, lambda m, t: t.pop())
    elif category == 'extra_tensor':
        rewrite(asset, lambda m, t: t.append(('extra', (1,), 0, bytes(4))))
    elif category == 'representation_mismatch':
        def change(m, t):
            name, shape, _, _ = t[0]
            t[0] = name, shape, 0, bytes(tensor_size(shape, 0))
        rewrite(asset, change)
    elif category == 'shape_mismatch':
        def change(m, t):
            name, _, type, _ = t[0]
            t[0] = name, (32, 64), type, bytes(tensor_size((32, 64), type))
        rewrite(asset, change)
    elif category == 'unaligned_tensor':
        path = asset / 'base.gguf'
        reader = Reader(path)
        last = list(reader.tensors.values())[-1]
        blob = bytearray(path.read_bytes())
        needle = struct.pack('<Q', len(last.name)) + last.name.encode()
        pos = blob.index(needle) + len(needle) + 4 + len(last.shape) * 8 + 4
        struct.pack_into('<Q', blob, pos, last.offset + 1)
        blob.append(0)
        path.write_bytes(blob)
        write_catalog(asset)
    elif category == 'hyperparameter_mismatch':
        rewrite(asset, lambda m, t: m.update({'qwen3next.expert_used_count': Value(4, 3)}))
    result = check(asset)
    assert result['status'] == 'failed'
    assert result['issues'][0]['category'] == category, result


@pytest.mark.parametrize('value', [Value(5, -1), Value(6, 4.0), Value(7, True)])
def test_reject_invalid_hyperparameter_type_or_sign(asset, value):
    rewrite(asset, lambda m, t: m.update({'qwen3next.expert_used_count': value}))
    result = check(asset)
    assert result['status'] == 'failed'
    assert result['issues'][0]['category'] == 'hyperparameter_mismatch'


def rewrite_prerouter(root, mutate):
    path = root / 'prerouter.gguf'
    reader = Reader(path)
    tensors = [(t.name, t.shape, t.type, reader.data(t.name)) for t in reader.tensors.values()]
    mutate(reader.metadata, tensors)
    path.unlink()
    write(path, reader.metadata, tensors)
    write_catalog(root)


@pytest.mark.parametrize('mutate, category', [
    (lambda m, t: m.__setitem__('prerouter.owners', Value(9, [1, 2, 3], 4)), 'hyperparameter_mismatch'),
    (lambda m, t: m.__setitem__('prerouter.top_k', Value(4, 8)), 'hyperparameter_mismatch'),
    (lambda m, t: t.pop(), 'missing_tensor'),
    (lambda m, t: t.__setitem__(0, (t[0][0], (t[0][1][1], t[0][1][0]), t[0][2], t[0][3])), 'shape_mismatch'),
])
def test_reject_prerouter(asset, mutate, category):
    rewrite_prerouter(asset, mutate)
    report = check(asset)
    assert report['status'] == 'failed'
    assert report['issues'][0]['category'] == category


def test_prerouter_is_required(asset):
    (asset / 'prerouter.gguf').unlink()
    assert check(asset)['issues'][0]['category'] == 'missing_file'
    write_catalog_without = [line for line in (asset / 'assets.tsv').read_text().splitlines() if 'prerouter' not in line]
    (asset / 'assets.tsv').write_text('\n'.join(write_catalog_without) + '\n')
    assert check(asset)['issues'][0]['category'] == 'missing_file'
