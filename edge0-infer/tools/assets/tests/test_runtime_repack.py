import importlib.util
import json
from pathlib import Path
import struct
import sys

import pytest

from converter_fixture import make_source

pytestmark = pytest.mark.l1
CONVERT = Path(__file__).resolve().parents[1] / 'edge0_35b/edge0_convert'


def runtime_module(monkeypatch):
    monkeypatch.syspath_prepend(str(CONVERT))
    spec = importlib.util.spec_from_file_location('tested_runtime_repack', CONVERT / 'repack_r3.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_runtime(monkeypatch, module, source, out, verify='all'):
    monkeypatch.setattr(sys, 'argv', [str(CONVERT / 'repack_r3.py'), '--dir', str(source),
                                    '--out', str(out), '--verify', verify])
    return module.main()


def test_runtime_q41_batches_preserve_complete_gguf(tmp_path, monkeypatch):
    source = tmp_path / 'edge0-35b'
    make_source(source)
    module = runtime_module(monkeypatch)
    original = module.pack_q41_rows
    calls = []

    def counted(q, scales, biases, where='?'):
        calls.append((where, len(q)))
        return original(q, scales, biases, where)

    monkeypatch.setattr(module, 'pack_q41_rows', counted)
    # Three rows deliberately leave a non-aligned chunk boundary and ragged tail.
    monkeypatch.setattr(module.core, 'CHUNK', 3)
    small = tmp_path / 'small'
    assert run_runtime(monkeypatch, module, source, small) == 0
    assert max(rows for name, rows in calls if name != 'ssm_ba') <= 3
    embedding = [rows for name, rows in calls if name == 'language_model.model.embed_tokens.weight']
    assert sum(embedding) == 64 and embedding[-1] == 1
    monkeypatch.setattr(module.core, 'CHUNK', 8192)
    whole = tmp_path / 'whole'
    assert run_runtime(monkeypatch, module, source, whole) == 0
    assert (small / 'edge0-35b.gguf').read_bytes() == (whole / 'edge0-35b.gguf').read_bytes()


def audit_result(out):
    return json.loads((out / 'edge0-35b-audit.json').read_text())


def set_embedding_scale(source, value):
    path = source / 'model.safetensors'
    data = bytearray(path.read_bytes())
    header_size = struct.unpack_from('<Q', data)[0]
    header = json.loads(data[8:8 + header_size])
    offset = header['language_model.model.embed_tokens.scales']['data_offsets'][0]
    struct.pack_into('<f', data, 8 + header_size + offset, value)
    path.write_bytes(data)


@pytest.mark.parametrize('verify', ['all', '64'])
@pytest.mark.parametrize('row', [1, 4])
def test_runtime_verification_detects_nonfirst_row_mismatch(tmp_path, monkeypatch, verify, row):
    source = tmp_path / 'edge0-35b'
    make_source(source)
    module = runtime_module(monkeypatch)
    original = module.pack_q41_rows
    embedding_offset = 0

    def corrupt(q, scales, biases, where='?'):
        nonlocal embedding_offset
        blocks, exempt = original(q, scales, biases, where)
        if where == 'language_model.model.embed_tokens.weight':
            if embedding_offset <= row < embedding_offset + len(q):
                blocks['qs'][row - embedding_offset, 0, 0] ^= 1
            embedding_offset += len(q)
        return blocks, exempt

    monkeypatch.setattr(module, 'pack_q41_rows', corrupt)
    monkeypatch.setattr(module.core, 'CHUNK', 3)
    out = tmp_path / 'out'
    assert run_runtime(monkeypatch, module, source, out, verify) == 1
    report = audit_result(out)
    assert report['verdict'] == 'R3-artifact RED'
    assert any(gate.get('ok') is False and gate['t'] == 'token_embd.weight'
               for gate in report['audit']['gates'])


def test_runtime_verifies_interleaved_ba_values(tmp_path, monkeypatch):
    source = tmp_path / 'edge0-35b'
    make_source(source)
    module = runtime_module(monkeypatch)
    original = module.pack_q41_rows

    def corrupt(q, scales, biases, where='?'):
        blocks, exempt = original(q, scales, biases, where)
        if where == 'ssm_ba':
            blocks['qs'][1, 0, 0] ^= 1
        return blocks, exempt

    monkeypatch.setattr(module, 'pack_q41_rows', corrupt)
    out = tmp_path / 'out'
    assert run_runtime(monkeypatch, module, source, out) == 1
    assert any(gate.get('ok') is False and gate['t'] == 'blk.0.ssm_ba.weight'
               for gate in audit_result(out)['audit']['gates'])


@pytest.mark.parametrize('corrupt_other_block', [False, True])
def test_runtime_exemption_covers_only_its_own_block(tmp_path, monkeypatch, corrupt_other_block):
    source = tmp_path / 'edge0-35b'
    make_source(source, hidden=128)
    # The first 64-wide group has an established FP16 rounding exemption.
    set_embedding_scale(source, 1e-7)
    module = runtime_module(monkeypatch)
    original = module.pack_q41_rows

    def corrupt(q, scales, biases, where='?'):
        blocks, exempt = original(q, scales, biases, where)
        if corrupt_other_block and where == 'language_model.model.embed_tokens.weight':
            # Block 2 belongs to the second 64-wide group, which is exact.
            blocks['qs'][0, 2, 0] ^= 1
        return blocks, exempt

    monkeypatch.setattr(module, 'pack_q41_rows', corrupt)
    out = tmp_path / 'out'
    assert run_runtime(monkeypatch, module, source, out) == int(corrupt_other_block)
    if not corrupt_other_block:
        assert any(gate['t'] == 'token_embd.weight' and gate.get('sentinel') == 'EXEMPT'
                   for gate in audit_result(out)['audit']['gates'])


@pytest.mark.parametrize('row,expected', [(0, 1), (1, 0)])
def test_runtime_zero_verify_keeps_first_row_sentinel(tmp_path, monkeypatch, row, expected):
    source = tmp_path / 'edge0-35b'
    make_source(source)
    module = runtime_module(monkeypatch)
    original = module.pack_q41_rows

    def corrupt(q, scales, biases, where='?'):
        blocks, exempt = original(q, scales, biases, where)
        if where == 'language_model.model.embed_tokens.weight':
            blocks['qs'][row, 0, 0] ^= 1
        return blocks, exempt

    monkeypatch.setattr(module, 'pack_q41_rows', corrupt)
    assert run_runtime(monkeypatch, module, source, tmp_path / 'out', '0') == expected


@pytest.mark.parametrize('verify', ['-1', 'invalid'])
def test_runtime_rejects_invalid_verify_before_conversion(tmp_path, monkeypatch, verify):
    source = tmp_path / 'edge0-35b'
    make_source(source)
    module = runtime_module(monkeypatch)
    out = tmp_path / 'out'
    with pytest.raises(SystemExit) as error:
        run_runtime(monkeypatch, module, source, out, verify)
    assert error.value.code == 2
    assert not out.exists()
