import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import pytest

from converter_fixture import make_source

pytestmark = pytest.mark.l1
ASSETS = Path(__file__).resolve().parents[1]
CONVERT = ASSETS / 'edge0_35b/edge0_convert'
ENTRY = CONVERT / 'convert_mlx_to_gguf.py'


def run_converter(source, out, *extra, entry=ENTRY):
    return subprocess.run([sys.executable, str(entry), '--dir', str(source), '--out', str(out), *extra],
                          capture_output=True, text=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def converted(tmp_path):
    source, out = tmp_path / 'edge0-35b', tmp_path / 'out'
    make_source(source)
    result = run_converter(source, out)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '[self-check] edge0-35b.gguf: tensors=703' in result.stdout
    return source, out


@pytest.mark.parametrize('changed', ['weights', 'config', 'tokenizer', 'template'])
def test_cached_conversion_matches_fresh_conversion_after_input_changes(converted, tmp_path, changed):
    source, out = converted
    previous = sha(out / 'edge0-35b.gguf')
    if changed == 'weights':
        path = source / 'model.safetensors'
        data = bytearray(path.read_bytes())
        data[8 + struct.unpack('<Q', data[:8])[0]] ^= 1
        path.write_bytes(data)
    elif changed == 'config':
        path = source / 'config.json'
        value = json.loads(path.read_text())
        value['text_config']['max_position_embeddings'] = 512
        path.write_text(json.dumps(value))
    elif changed == 'tokenizer':
        path = source / 'tokenizer.json'
        value = json.loads(path.read_text())
        value['model']['vocab']['renamed'] = value['model']['vocab'].pop('a')
        path.write_text(json.dumps(value))
    else:
        (source / 'chat_template.jinja').write_text('new template: {{ messages }}')
    rerun = run_converter(source, out)
    fresh_out = tmp_path / 'fresh'
    fresh = run_converter(source, fresh_out)
    assert rerun.returncode == fresh.returncode == 0, rerun.stderr + fresh.stderr
    assert sha(out / 'edge0-35b.gguf') == sha(fresh_out / 'edge0-35b.gguf')
    assert sha(out / 'edge0-35b.gguf') != previous


def test_cached_conversion_tracks_converter_code(converted, tmp_path):
    source, out = converted
    copy = tmp_path / 'tools'
    copied_convert = copy / 'edge0_35b/edge0_convert'
    shutil.copytree(CONVERT, copied_convert, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['format_gguf.py', 'asset_common.py']:
        shutil.copyfile(ASSETS / name, copy / name)
    runtime = copied_convert / 'repack_r3.py'
    original = runtime.read_text()
    changed = original.replace('w.kv_s("general.name", "edge0-35b")',
                               'w.kv_s("general.name", "updated converter identity")')
    assert changed != original
    runtime.write_text(changed)
    entry = copied_convert / 'convert_mlx_to_gguf.py'
    rerun = run_converter(source, out, entry=entry)
    fresh_out = tmp_path / 'fresh'
    fresh = run_converter(source, fresh_out, entry=entry)
    assert rerun.returncode == fresh.returncode == 0, rerun.stderr + fresh.stderr
    assert sha(out / 'edge0-35b.gguf') == sha(fresh_out / 'edge0-35b.gguf')


def test_unchanged_inputs_keep_the_verified_cache(converted):
    source, out = converted
    names = ['edge0-35b.gguf', 'lora_edge0_35b-gguf.gguf', 'manifest.json']
    before = {name: ((out / name).stat().st_mtime_ns, sha(out / name)) for name in names}
    result = run_converter(source, out)
    assert result.returncode == 0, result.stderr
    assert 'already converted' in result.stdout
    assert {name: ((out / name).stat().st_mtime_ns, sha(out / name)) for name in names} == before


def test_cache_hit_still_checks_requested_comparison(converted, tmp_path):
    source, out = converted
    reference = tmp_path / 'reference'
    reference.mkdir()
    for name in ['edge0-35b.gguf', 'lora_edge0_35b-gguf.gguf']:
        shutil.copyfile(out / name, reference / name)
    path = reference / 'edge0-35b.gguf'
    data = bytearray(path.read_bytes())
    data[-64] ^= 1
    path.write_bytes(data)
    result = run_converter(source, out, '--compare-against', str(reference))
    assert result.returncode != 0
    assert 'determinism mismatch' in result.stderr


def test_cache_hit_accepts_matching_reference_without_rebuilding(converted):
    source, out = converted
    before = (out / 'edge0-35b.gguf').stat().st_mtime_ns
    result = run_converter(source, out, '--compare-against', str(out))
    assert result.returncode == 0, result.stderr
    assert '[determinism] vs baselines: ALL MATCH OK' in result.stdout
    assert 'already converted' in result.stdout
    assert (out / 'edge0-35b.gguf').stat().st_mtime_ns == before


def test_legacy_manifest_without_tool_identity_is_revalidated(converted):
    source, out = converted
    path = out / 'manifest.json'
    old = json.loads(path.read_text())
    old['converter'].pop('files_sha256')
    path.write_text(json.dumps(old))
    result = run_converter(source, out)
    assert result.returncode == 0, result.stderr
    assert 'already converted' not in result.stdout
    assert json.loads(path.read_text())['converter']['files_sha256']
