import importlib.util
from pathlib import Path
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
