from copy import deepcopy
from collections import Counter
import shutil
import struct

import pytest

from asset_common import write_json
import check_assets
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


@pytest.mark.parametrize('metadata_type', [5, 10, 11])
def test_rejects_non_uint32_alignment(asset, metadata_type):
    # Keep the value and file layout valid; only the metadata's encoded type is wrong.
    rewrite(asset, lambda m, t: m.update({'fixture.alignment': Value(metadata_type, 32)}))
    path = asset / 'base.gguf'
    path.write_bytes(path.read_bytes().replace(b'fixture.alignment', b'general.alignment'))
    write_catalog(asset)
    report = check(asset)
    assert report['status'] == 'failed' and report['issues'][0]['category'] == 'invalid_asset', report
    assert 'alignment' in report['issues'][0]['detail']


@pytest.mark.parametrize('element_type', [13, 0xFFFFFFFF])
def test_rejects_unknown_empty_array_type(asset, element_type):
    key = 'fixture.empty_array'
    rewrite(asset, lambda m, t: m.update({key: Value(9, [], 4)}))
    path = asset / 'base.gguf'
    blob = bytearray(path.read_bytes())
    struct.pack_into('<I', blob, blob.index(key.encode()) + len(key) + 4, element_type)
    path.write_bytes(blob)
    write_catalog(asset)
    report = check(asset)
    assert report['status'] == 'failed' and report['issues'][0]['category'] == 'invalid_asset', report
    assert 'array' in report['issues'][0]['detail']


@pytest.mark.parametrize('description', [{'architecture': 'edge0_35b'}, {'format_version': 1}, ['edge0_35b']])
def test_rejects_malformed_model_description(asset, description):
    write_json(asset / 'model.json', description)
    write_catalog(asset, edge0_check.ROLES)
    assert check(asset)['issues'][0]['category'] == 'invalid_manifest'


@pytest.mark.parametrize('version', [True, 1.0])
def test_format_version_requires_a_json_integer(asset, version):
    write_json(asset / 'model.json', {'format_version': version, 'architecture': 'edge0_35b'})
    write_catalog(asset, edge0_check.ROLES)
    report = check(asset)
    assert report['status'] == 'failed'
    assert report['issues'][0]['category'] == 'invalid_manifest'


@pytest.mark.parametrize('case', ['extra', 'empty_extra', 'blank', 'quoted_role',
                                'plus_bytes', 'spaced_bytes', 'underscored_bytes', 'unicode_bytes'])
def test_rejects_catalog_spelling_the_native_loader_cannot_read(asset, case):
    path = asset / 'assets.tsv'
    lines = path.read_text().splitlines()
    fields = lines[1].split('\t')
    if case == 'extra':
        fields.append('ignored')
    elif case == 'empty_extra':
        fields.append('')
    elif case == 'blank':
        lines.append('')
    elif case == 'quoted_role':
        fields[0] = '"' + fields[0] + '"'
    elif case == 'plus_bytes':
        fields[1] = '+' + fields[1]
    elif case == 'spaced_bytes':
        fields[1] = ' ' + fields[1] + ' '
    elif case == 'underscored_bytes':
        fields[1] = '_'.join(fields[1])
    else:
        fields[1] = fields[1].translate(str.maketrans('0123456789', '\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669'))
    lines[1] = '\t'.join(fields)
    path.write_text('\n'.join(lines) + '\n')
    assert check(asset)['status'] == 'failed'


def test_catalog_keeps_valid_decimal_leading_zeros(asset):
    path = asset / 'assets.tsv'
    lines = path.read_text().splitlines()
    fields = lines[1].split('\t')
    fields[1] = '000' + fields[1]
    lines[1] = '\t'.join(fields)
    path.write_text('\n'.join(lines) + '\n')
    assert check(asset)['status'] == 'passed'


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


def test_compare_equal_assets_reads_each_payload_once(asset, tmp_path, monkeypatch):
    other = tmp_path / 'other'
    shutil.copytree(asset, other)
    hashes, tensors = Counter(), Counter()
    original_hash, original_digest = check_assets.sha256, Reader.digest

    def counted_hash(path):
        hashes[path] += 1
        return original_hash(path)

    def counted_digest(reader, name):
        tensors[reader.path] += 1
        return original_digest(reader, name)

    monkeypatch.setattr(check_assets, 'sha256', counted_hash)
    monkeypatch.setattr(Reader, 'digest', counted_digest)
    result = compare(asset, other)
    assert result['status'] == 'passed' and all(result['whole_file_equal'].values())
    assert len(hashes) == 2 * len(edge0_check.ROLES)
    assert set(hashes.values()) == {1}
    assert not tensors


@pytest.mark.parametrize('change', ['metadata', 'text', 'padding', 'corrupt'])
def test_compare_still_checks_different_and_corrupt_files(asset, tmp_path, change):
    other = tmp_path / 'other'
    shutil.copytree(asset, other)
    if change == 'metadata':
        rewrite(other, lambda m, t: m.update({'test.note': Value(8, 'different')}))
    elif change == 'text':
        (other / 'chat_template.jinja').write_text('different template')
    else:
        with (other / 'base.gguf').open('ab') as file:
            file.write(b'\0' * 32)
    if change != 'corrupt':
        write_catalog(other)
    result = compare(asset, other)
    if change == 'corrupt':
        assert result['status'] == 'failed'
        assert result['checks'][1]['issues'][0]['category'] == 'size_mismatch'
    else:
        name = 'chat_template.jinja' if change == 'text' else 'base.gguf'
        assert result['whole_file_equal'][name] is False
        assert result['status'] == ('passed' if change == 'padding' else 'failed')
        assert result['differences'] == ([] if change == 'padding' else [{'file': name, 'kind': change}])


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


@pytest.mark.parametrize('key', ['qwen3next.rope.freq_base', 'qwen3next.attention.layer_norm_rms_epsilon'])
@pytest.mark.parametrize('value', [None, Value(6, 0.0), Value(6, -1.0), Value(6, float('nan')),
                                  Value(6, float('inf')), Value(6, float('-inf')), Value(12, 1.0),
                                  Value(4, 1), Value(7, True), Value(8, '1.0')])
def test_reject_invalid_runtime_float(asset, key, value):
    def change(metadata, tensors):
        if value is None:
            metadata.pop(key)
        else:
            metadata[key] = value
    rewrite(asset, change)
    report = check(asset)
    assert report['status'] == 'failed'
    assert report['issues'][0]['category'] == 'hyperparameter_mismatch'
    assert key in report['issues'][0]['detail']


@pytest.mark.parametrize('value', [None, Value(4, 0), Value(5, -1), Value(5, 4096),
                                  Value(10, 4096), Value(6, 4096.0), Value(7, True)])
def test_reject_invalid_context_length(asset, value):
    key = 'qwen3next.context_length'
    rewrite(asset, lambda metadata, tensors: metadata.pop(key) if value is None else metadata.update({key: value}))
    report = check(asset)
    assert report['status'] == 'failed'
    assert report['issues'][0]['category'] == 'hyperparameter_mismatch'
    assert key in report['issues'][0]['detail']


@pytest.mark.parametrize('context, frequency, epsilon', [(1, 1.0, 1e-8), (4096, 1e6, 1e-5), (2**32 - 1, 1e7, 1e-6)])
def test_accept_valid_runtime_hyperparameters(asset, context, frequency, epsilon):
    rewrite(asset, lambda metadata, tensors: metadata.update({
        'qwen3next.context_length': Value(4, context),
        'qwen3next.rope.freq_base': Value(6, frequency),
        'qwen3next.attention.layer_norm_rms_epsilon': Value(6, epsilon),
    }))
    assert check(asset)['status'] == 'passed'


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
