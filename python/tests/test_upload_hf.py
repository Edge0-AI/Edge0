"""Upload file selection and CLI wiring, with no MLX or network."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def upload_hf(monkeypatch):
    # The script patches its dependencies and environment on import. Use
    # private stubs so those changes cannot affect other tests or the network.
    class FakeTransport:
        def handle_request(self, request):
            raise AssertionError("network access is out of scope")

    class FakeApi:
        def upload_file(self, **kwargs):
            raise AssertionError("real uploads are out of scope")

    httpx = ModuleType("httpx")
    httpx.HTTPTransport = FakeTransport
    hub = ModuleType("huggingface_hub")
    lfs = ModuleType("huggingface_hub.lfs")
    lfs.fix_hf_endpoint_in_url = lambda url, endpoint: url
    hub.lfs = lfs
    hf_api = ModuleType("huggingface_hub.hf_api")
    hf_api.HfApi = FakeApi
    for name, module in (
        ("httpx", httpx),
        ("huggingface_hub", hub),
        ("huggingface_hub.lfs", lfs),
        ("huggingface_hub.hf_api", hf_api),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("HF_ENDPOINT", "HF_HUB_DISABLE_XET", "NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("HF_ENDPOINT", "https://example.invalid")

    spec = importlib.util.spec_from_file_location(
        "upload_hf", ROOT / "scripts" / "upload_hf.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


def test_files_to_upload_includes_readme_and_sorts(tmp_path, upload_hf):
    for name in ("model.safetensors", "config.json", "README.md"):
        (tmp_path / name).write_text("x")
    assert upload_hf.files_to_upload(str(tmp_path)) == [
        "README.md", "config.json", "model.safetensors"]


def test_files_to_upload_excludes_dotfiles_and_vision_backups(tmp_path, upload_hf):
    for name in ("README.md", "model.safetensors",
                 "model.safetensors.bak_vision", ".DS_Store"):
        (tmp_path / name).write_text("x")
    assert upload_hf.files_to_upload(str(tmp_path)) == [
        "README.md", "model.safetensors"]


def test_files_to_upload_excludes_directories(tmp_path, upload_hf):
    (tmp_path / "README.md").write_text("x")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("x")
    assert upload_hf.files_to_upload(str(tmp_path)) == ["README.md"]


def test_files_to_upload_empty_directory(tmp_path, upload_hf):
    assert upload_hf.files_to_upload(str(tmp_path)) == []


@pytest.mark.parametrize("tier,env_name,repo_id", [
    ("edge0-8b", "EDGE0_8B_MODEL", "Edge0/Edge0-8B-A1B-preview"),
    ("edge0-35b", "EDGE0_35B_MODEL", "Edge0/Edge0-35B-A3B-preview"),
])
def test_main_passes_readme_to_upload_api(
    tmp_path, upload_hf, monkeypatch, tier, env_name, repo_id,
):
    for name in ("README.md", "config.json", ".DS_Store",
                 "model.safetensors.bak_vision"):
        (tmp_path / name).write_text("x")
    (tmp_path / "subdir").mkdir()
    uploads = []

    class RecordingApi:
        def upload_file(self, **kwargs):
            uploads.append(kwargs)
            return "uploaded"

    monkeypatch.setattr(upload_hf, "HfApi", RecordingApi)
    monkeypatch.setenv(env_name, str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["upload_hf.py", tier])
    upload_hf.main()
    assert uploads == [
        {
            "path_or_fileobj": str(tmp_path / name),
            "path_in_repo": name,
            "repo_id": repo_id,
            "repo_type": "model",
        }
        for name in ("README.md", "config.json")
    ]
