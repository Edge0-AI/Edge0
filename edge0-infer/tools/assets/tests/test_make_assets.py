import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from asset_common import sha256
from edge0_35b import make_assets as maker

pytestmark = pytest.mark.l1

REPO = Path(__file__).resolve().parents[3]

FAILING = '''import argparse, json, os, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--dir")
p.add_argument("--out")
a = p.parse_args()
(Path(a.dir) / "generated").write_text("created")
Path(os.environ["PROBE_FILE"]).write_text(json.dumps({"sys_path": sys.path, "pythonpath": os.environ.get("PYTHONPATH")}))
raise SystemExit(7)
'''


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source, out = tmp_path / 'source', tmp_path / 'out'
    source.mkdir()
    for name in ['config.json', 'lora_edge0_35b.safetensors']:
        (source / name).write_text('{}')
    converter = tmp_path / 'converter/convert_mlx_to_gguf.py'
    converter.parent.mkdir()
    converter.write_text(FAILING)
    probe = tmp_path / 'probe.json'
    monkeypatch.setenv('PROBE_FILE', str(probe))
    lock = {'files': {p.name: sha256(p) for p in source.iterdir()}}
    return source, out, lock, converter, probe


def test_digest_stops_before_conversion(setup):
    source, out, lock, converter, probe = setup
    (source / 'config.json').write_text('changed')
    with pytest.raises(ValueError, match='source digest mismatch'):
        maker.make_assets(source, out, lock, converter)
    assert not out.with_name('out.partial').exists()
    assert not probe.exists()


def test_missing_lora_does_not_publish(setup):
    source, out, lock, converter, probe = setup
    (source / 'lora_edge0_35b.safetensors').unlink()
    with pytest.raises(ValueError, match='missing source file'):
        maker.make_assets(source, out, lock, converter)
    assert not out.exists()
    assert not out.with_name('out.partial').exists()


def test_converter_failure_preserves_partial_and_input(setup):
    source, out, lock, converter, probe = setup
    partial = out.with_name('out.partial')
    with pytest.raises(maker.MakeError) as failure:
        maker.make_assets(source, out, lock, converter)
    assert isinstance(failure.value.__cause__, subprocess.CalledProcessError)
    assert f'unfinished output kept at {partial}; delete it and rerun' in str(failure.value)
    assert not out.exists()
    assert (partial / 'staging/edge0-35b/generated').is_file()
    assert not (source / 'generated').exists()
    assert not list(partial.rglob('assets.tsv'))
    # A plain rerun refuses because of the kept directory and gives the same hint.
    with pytest.raises(ValueError, match=re.escape(f'left {partial}; delete it and rerun')):
        maker.make_assets(source, out, lock, converter)
    assert (partial / 'staging/edge0-35b/generated').is_file()


def test_refuse_existing_output(setup):
    source, out, lock, converter, probe = setup
    out.mkdir()
    (out / 'keep').write_text('preserve')
    with pytest.raises(ValueError, match='output already exists'):
        maker.make_assets(source, out, lock, converter)
    assert (out / 'keep').read_text() == 'preserve'


def test_converter_sees_only_repository_and_python_paths(setup, monkeypatch, tmp_path):
    source, out, lock, converter, probe = setup
    foreign = tmp_path / 'foreign-code'
    foreign.mkdir()
    monkeypatch.setenv('PYTHONPATH', str(foreign))
    with pytest.raises(maker.MakeError):
        maker.make_assets(source, out, lock, converter)
    seen = json.loads(probe.read_text())
    assert seen['pythonpath'] is None
    allowed = [REPO, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(), converter.parent.resolve()]
    for entry in seen['sys_path']:
        path = Path(entry or os.getcwd()).resolve()
        assert any(path.is_relative_to(root) for root in allowed), entry


def test_default_converter_is_in_repository():
    assert maker.CONVERTER.is_file()
    assert maker.CONVERTER.resolve().is_relative_to(REPO)


def test_command_line_takes_no_foreign_code():
    result = subprocess.run([sys.executable, '-I', str(Path(maker.__file__)), '--help'],
                            capture_output=True, text=True, check=True)
    assert '--source' in result.stdout and '--out' in result.stdout
    assert 'edge0' not in result.stdout.lower() and 'llama' not in result.stdout.lower()


SUCCEEDING = '''import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--dir")
p.add_argument("--out")
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True)
(out / "edge0-35b.gguf").write_bytes(b"base")
(out / "lora_edge0_35b-gguf.gguf").write_bytes(b"lora")
(out / "manifest.json").write_text(json.dumps({"converter": "stand-in"}))
'''
TEXT = ['tokenizer.json', 'chat_template.jinja', 'generation_config.json']


@pytest.fixture
def succeeding(tmp_path, monkeypatch):
    """A stand-in converter, packer and checker so that a build publishes."""
    source = tmp_path / 'source'
    source.mkdir()
    for name in ['config.json', *TEXT]:
        (source / name).write_text(name)
    converter = tmp_path / 'converter/convert_mlx_to_gguf.py'
    converter.parent.mkdir()
    converter.write_text(SUCCEEDING)

    def pack(_, target):
        target.write_bytes(b'prerouter')
        return {'source_file': 'prerouter.safetensors'}
    monkeypatch.setattr(maker.prerouter, 'pack', pack)
    monkeypatch.setattr(maker, 'check', lambda root: {'status': 'passed'})
    lock = {'files': {p.name: sha256(p) for p in source.iterdir()}}
    return source, lock, converter


def test_success_removes_partial_and_records_no_local_paths(succeeding, tmp_path):
    source, lock, converter = succeeding
    records = []
    for place in ['first', 'second/deeper']:
        out = tmp_path / place / 'assets'
        maker.make_assets(source, out, lock, converter)
        assert (out / 'assets.tsv').is_file()
        assert not out.with_name('assets.partial').exists()
        text = (out / 'provenance.json').read_text()
        assert str(tmp_path) not in text and str(tmp_path.resolve()) not in text
        records.append(json.loads(text))
    assert records[0] == records[1]
    assert records[0]['arguments'] == ['<outside-repository>/convert_mlx_to_gguf.py', '--dir', '<partial>/staging/edge0-35b',
                                       '--out', '<partial>/converted']


def test_repository_converter_is_recorded_relative(tmp_path):
    partial = tmp_path / 'out.partial'
    assert maker.recorded_path(maker.CONVERTER, partial) == 'tools/assets/edge0_35b/edge0_convert/convert_mlx_to_gguf.py'
    assert maker.recorded_path(partial / 'staging/edge0-35b', partial) == '<partial>/staging/edge0-35b'
