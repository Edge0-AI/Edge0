#!/usr/bin/env python3
"""Strip the unused vision-tower weights from an edge0-35b checkpoint.

The qwen35 checkpoint ships a 333-tensor / ~0.9GB vision tower
(``vision_tower.*``) that the text-only engine never loads — the model's
sanitize() already skips these keys, so removing them from the
safetensors shards is pure disk/memory savings.  Rewrites shard 1 in
place (backup first), updates model.safetensors.index.json.

Usage: python scripts/strip_vision_weights.py <model_dir> [--no-backup]
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path


def read_shard(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        data_start = 8 + n
    return hdr, data_start


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()
    d = Path(args.model_dir)
    idx_path = d / "model.safetensors.index.json"
    idx = json.load(open(idx_path))
    wm = idx["weight_map"]
    vs = {k for k in wm if "visual" in k or "vision" in k}
    if not vs:
        print("no vision weights found — nothing to strip")
        return 0

    shards = sorted({wm[k] for k in vs})
    for shard_name in shards:
        path = d / shard_name
        hdr, data_start = read_shard(path)
        drop = {k for k in vs if wm[k] == shard_name}
        keep_hdr = {k: v for k, v in hdr.items()
                    if k not in drop and "data_offsets" in v}
        # rewrite: new offsets (contiguous from original data_start)
        out = bytearray()
        new_hdr = {}
        off = 0
        with open(path, "rb") as f:
            f.seek(data_start)
            for k in sorted(keep_hdr, key=lambda k: keep_hdr[k]["data_offsets"][0]):
                e = dict(keep_hdr[k])
                b0, b1 = e["data_offsets"]
                f.seek(data_start + b0)
                blob = f.read(b1 - b0)
                out += blob
                e["data_offsets"] = [off, off + len(blob)]
                off += len(blob)
                new_hdr[k] = e
        # non-tensor entries (e.g. __metadata__) pass through
        for k, v in hdr.items():
            if k not in keep_hdr and "data_offsets" not in v and k not in drop:
                new_hdr[k] = v
        hdr_bytes = json.dumps(new_hdr).encode()
        pad = (8 - (len(hdr_bytes) % 8)) % 8
        hdr_bytes += b" " * pad
        if not args.no_backup:
            shutil.copy2(path, path.with_suffix(".safetensors.bak_vision"))
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hdr_bytes)))
            f.write(hdr_bytes)
            f.write(out)
        saved = sum(hdr[k]["data_offsets"][1] - hdr[k]["data_offsets"][0]
                    for k in drop)
        print(f"{shard_name}: dropped {len(drop)} tensors, "
              f"{saved/1e9:.2f} GB, new size {path.stat().st_size/1e9:.2f} GB")

    # index: remove vision keys
    idx["weight_map"] = {k: v for k, v in wm.items() if k not in vs}
    if not args.no_backup:
        shutil.copy2(idx_path, idx_path.with_suffix(".json.bak_vision"))
    json.dump(idx, open(idx_path, "w"), indent=2)
    print(f"index: {len(idx['weight_map'])} keys remain "
          f"({len(vs)} vision keys removed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
