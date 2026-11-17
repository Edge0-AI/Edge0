import json
from pathlib import Path
import subprocess
import sys

import pytest

from converter_fixture import make_source

pytestmark = pytest.mark.l1
ENTRY = Path(__file__).resolve().parents[1] / 'edge0_35b/edge0_convert/convert_mlx_to_gguf.py'


@pytest.mark.parametrize('custom_output', [False, True])
def test_underscore_source_produces_canonical_artifacts(tmp_path, custom_output):
    source = tmp_path / 'edge0_35b'
    make_source(source)
    command = [sys.executable, str(ENTRY), '--dir', str(source)]
    out = tmp_path / 'custom-output' if custom_output else tmp_path / 'edge0_35b-gguf'
    if custom_output:
        command.extend(['--out', str(out)])
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr + (out / 'convert.log').read_text()
    assert '[self-check] edge0-35b.gguf: tensors=703' in result.stdout
    assert '[self-check] adapter tensors=100 OK' in result.stdout
    assert (out / 'edge0-35b.gguf').is_file()
    assert (out / 'lora_edge0_35b-gguf.gguf').is_file()
    assert not (out / 'edge0_35b.gguf').exists()
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['model'] == 'edge0_35b' and manifest['tier'] == '35b'
    before = (out / 'edge0-35b.gguf').stat().st_mtime_ns
    repeated = subprocess.run(command, text=True, capture_output=True)
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert 'already converted' in repeated.stdout
    assert (out / 'edge0-35b.gguf').stat().st_mtime_ns == before
