import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from format_gguf import F16, F32, Value, write

pytestmark = pytest.mark.l1

CONVERT = Path(__file__).resolve().parents[1] / 'edge0_35b/edge0_convert'
FILES = {'convert_mlx_to_gguf.py', 'repack_mlx_to_gguf.py', 'repack_r3.py', 'lora_mlx_to_gguf.py'}


def lora_module():
    spec = importlib.util.spec_from_file_location('copied_lora', CONVERT / 'lora_mlx_to_gguf.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gguf(path, tensors):
    write(path, {'general.architecture': Value(8, 'qwen3next')},
          [(name, shape, F32, bytes(4 * shape[0] * shape[1])) for name, shape in tensors])
    return path


def test_origin_lists_the_copied_files():
    origin = json.loads((CONVERT / 'ORIGIN.json').read_text())
    assert origin['repository'] == 'https://github.com/Edge0-AI/Edge0'
    assert origin['commit'] == 'c7ee73d5dfb3729de452bdb1fec7df7aebe42806'
    assert origin['path'] == 'windows/tools'
    assert set(origin['upstream_sha256']) == FILES
    assert all(len(d) == 64 for d in origin['upstream_sha256'].values())
    assert set(origin['modifications']) == {'M1', 'M2', 'M3', 'M4'}
    assert FILES <= {p.name for p in CONVERT.iterdir()}


def test_entry_rejects_non_35b_before_converting(tmp_path):
    source = tmp_path / 'edge0-8b'
    source.mkdir()
    out = tmp_path / 'out'
    result = subprocess.run([sys.executable, str(CONVERT / 'convert_mlx_to_gguf.py'),
                             '--dir', str(source), '--out', str(out)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'E-TIER' in result.stderr
    assert not out.exists()


def test_cross_check_accepts_matching_adapter(tmp_path, capsys):
    base = gguf(tmp_path / 'base.gguf', [('blk.0.attn_q.weight', (64, 128)), ('token_embd.weight', (16, 100))])
    adapter = gguf(tmp_path / 'lora.gguf', [
        ('blk.0.attn_q.weight.lora_a', (64, 8)), ('blk.0.attn_q.weight.lora_b', (8, 128)),
        ('token_embd.weight.lora_a', (8, 100)), ('token_embd.weight.lora_b', (100, 16))])
    lora_module().verify_against_base(str(adapter), str(base))
    assert 'GREEN: 2 pairs' in capsys.readouterr().out


def test_cross_check_rejects_name_missing_from_base(tmp_path):
    base = gguf(tmp_path / 'base.gguf', [('blk.0.attn_q.weight', (64, 128))])
    adapter = gguf(tmp_path / 'lora.gguf', [
        ('blk.0.attn_k.weight.lora_a', (64, 8)), ('blk.0.attn_k.weight.lora_b', (8, 128))])
    with pytest.raises(SystemExit, match='miss=1'):
        lora_module().verify_against_base(str(adapter), str(base))


@pytest.mark.parametrize('a_shape,b_shape', [
    pytest.param((64, 8), (8, 96), id='output-dimension'),
    pytest.param((32, 8), (8, 128), id='input-dimension'),
    pytest.param((64, 8), (16, 128), id='rank-equality'),
])
def test_cross_check_rejects_shape_violation(tmp_path, a_shape, b_shape):
    base = gguf(tmp_path / 'base.gguf', [('blk.0.attn_q.weight', (64, 128))])
    adapter = gguf(tmp_path / 'lora.gguf', [
        ('blk.0.attn_q.weight.lora_a', a_shape), ('blk.0.attn_q.weight.lora_b', b_shape)])
    with pytest.raises(SystemExit, match='shape_bad=1'):
        lora_module().verify_against_base(str(adapter), str(base))


def test_cross_check_does_not_need_gguf_py(tmp_path, monkeypatch):
    # A poisoned 'gguf' module on the path must not be imported by the gate.
    poison = tmp_path / 'poison'
    poison.mkdir()
    (poison / 'gguf.py').write_text("raise RuntimeError('gguf-py must not be imported')\n")
    monkeypatch.syspath_prepend(str(poison))
    sys.modules.pop('gguf', None)
    base = gguf(tmp_path / 'base.gguf', [('blk.0.attn_q.weight', (64, 128))])
    adapter = gguf(tmp_path / 'lora.gguf', [
        ('blk.0.attn_q.weight.lora_a', (64, 8)), ('blk.0.attn_q.weight.lora_b', (8, 128))])
    lora_module().verify_against_base(str(adapter), str(base))
    assert 'gguf' not in sys.modules
