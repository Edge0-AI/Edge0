"""scripts/upload_hf.py file-selection logic.

Pure filesystem logic, no MLX and no network — the upload itself
(``HfApi.upload_file``) is out of scope for unit tests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_upload_hf():
    """Import scripts/upload_hf.py without running main() or touching
    the network — the module patches httpx at import time, which is
    fine (pure monkeypatch, no I/O)."""
    spec = importlib.util.spec_from_file_location(
        "upload_hf", ROOT / "scripts" / "upload_hf.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_files_to_upload_includes_readme(tmp_path):
    for name in ("README.md", "config.json", "model.safetensors"):
        (tmp_path / name).write_text("x")
    mod = _load_upload_hf()
    assert mod.files_to_upload(str(tmp_path)) == [
        "README.md", "config.json", "model.safetensors"]


def test_files_to_upload_excludes_dotfiles_and_vision_backups(tmp_path):
    for name in ("README.md", "model.safetensors",
                 "model.safetensors.bak_vision", ".DS_Store"):
        (tmp_path / name).write_text("x")
    mod = _load_upload_hf()
    assert mod.files_to_upload(str(tmp_path)) == [
        "README.md", "model.safetensors"]
