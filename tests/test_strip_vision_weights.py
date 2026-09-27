"""Regression test for scripts/strip_vision_weights.py (#18).

Pure stdlib: builds a tiny synthetic safetensors shard + index, runs the
script as a subprocess, and checks that ``metadata.total_size`` shrinks
by the bytes actually dropped. Imports nothing from ``edge0`` so it runs
on any platform (including CI without MLX wheels).
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "strip_vision_weights.py"


def _write_shard(path: Path, tensors: dict) -> None:
    header = {}
    off = 0
    for name, blob in tensors.items():
        header[name] = {"dtype": "U8", "shape": [len(blob)],
                        "data_offsets": [off, off + len(blob)]}
        off += len(blob)
    header["__metadata__"] = {"format": "pt"}
    hdr = json.dumps(header).encode()
    hdr += b" " * ((8 - len(hdr) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for blob in tensors.values():
            f.write(blob)


def _read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def test_total_size_updated_after_strip(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    idx_path = tmp_path / "model.safetensors.index.json"
    tensors = {
        "visual.patch": b"v" * 100,
        "layers.0.mlp": b"a" * 40,
        "layers.1.mlp": b"b" * 60,
    }
    _write_shard(shard, tensors)
    total = sum(len(v) for v in tensors.values())
    idx_path.write_text(json.dumps({
        "metadata": {"total_size": total},
        "weight_map": {k: shard.name for k in tensors},
    }))

    r = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), "--no-backup"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr

    new_idx = json.loads(idx_path.read_text())
    assert "visual.patch" not in new_idx["weight_map"]
    assert len(new_idx["weight_map"]) == 2
    # total_size must shrink by exactly the dropped tensor bytes
    assert new_idx["metadata"]["total_size"] == total - 100
    # shard header survives: kept tensors + non-tensor metadata entry
    hdr = _read_header(shard)
    assert "visual.patch" not in hdr
    assert "layers.0.mlp" in hdr and "layers.1.mlp" in hdr
    assert hdr["__metadata__"] == {"format": "pt"}
