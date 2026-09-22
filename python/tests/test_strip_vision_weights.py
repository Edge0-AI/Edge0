"""scripts/strip_vision_weights.py: index bookkeeping after stripping.

Builds a minimal two-tensor safetensors shard (one vision, one text) by
hand so the test needs neither MLX nor a real checkpoint.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_strip_vision_weights():
    spec = importlib.util.spec_from_file_location(
        "strip_vision_weights", ROOT / "scripts" / "strip_vision_weights.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_shard(path: Path, hdr: dict, data: bytes) -> None:
    hdr_bytes = json.dumps(hdr).encode()
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    hdr_bytes += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        f.write(data)


def _make_checkpoint(model_dir: Path) -> None:
    # 4 bytes of vision tensor, 4 bytes of text tensor.
    hdr = {
        "__metadata__": {"format": "pt"},
        "vision_tower.a": {"dtype": "F32", "shape": [1],
                            "data_offsets": [0, 4]},
        "model.layers.0.w": {"dtype": "F32", "shape": [1],
                              "data_offsets": [4, 8]},
    }
    _write_shard(model_dir / "model.safetensors", hdr, b"\x00" * 8)
    idx = {
        "metadata": {"total_size": 8},
        "weight_map": {
            "vision_tower.a": "model.safetensors",
            "model.layers.0.w": "model.safetensors",
        },
    }
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(idx))


def test_total_size_shrinks_by_dropped_bytes(tmp_path, monkeypatch):
    _make_checkpoint(tmp_path)
    mod = _load_strip_vision_weights()
    monkeypatch.setattr(
        sys, "argv", ["strip_vision_weights.py", str(tmp_path)])
    assert mod.main() == 0

    idx = json.loads(
        (tmp_path / "model.safetensors.index.json").read_text())
    assert "vision_tower.a" not in idx["weight_map"]
    assert idx["metadata"]["total_size"] == 4  # 8 - 4 dropped vision bytes


def test_no_vision_weights_leaves_total_size_untouched(tmp_path, monkeypatch):
    hdr = {"model.layers.0.w": {"dtype": "F32", "shape": [1],
                                 "data_offsets": [0, 4]}}
    _write_shard(tmp_path / "model.safetensors", hdr, b"\x00" * 4)
    idx = {
        "metadata": {"total_size": 4},
        "weight_map": {"model.layers.0.w": "model.safetensors"},
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(idx))

    mod = _load_strip_vision_weights()
    monkeypatch.setattr(
        sys, "argv", ["strip_vision_weights.py", str(tmp_path)])
    assert mod.main() == 0

    idx_after = json.loads(
        (tmp_path / "model.safetensors.index.json").read_text())
    assert idx_after == idx  # untouched: nothing to strip
