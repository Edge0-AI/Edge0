#!/usr/bin/env python3
"""One-shot legacy adapter converter: npz -> safetensors.

The training pipelines exported adapters as ``.npz`` (qwen round-7 and
the training side).  edge0 consumes safetensors ONLY — this script converts each
legacy pair once into ``artifacts/`` (gitignored), writing the
conversion provenance (model, kind, K, r/alpha, source md5, format
version) into the safetensors ``__metadata__``.  The npz sources are
NOT deleted (see the conversion note in README); they are only needed
to regenerate the artifacts.

Usage:
    python scripts/convert_adapters_legacy.py [--force]

Idempotent: existing artifacts are skipped unless ``--force``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

# (npz path, tier, kind) — every legacy source this script understands.
# Training-side npz exports discovered per machine; pass --npz-dir (or
# edit this list) to point at your own exports.
NPZ_DIRS = [
    Path(os.environ.get("EDGE0_NPZ_DIR", "/path/to/npz/exports")),
]
SOURCES = [
    ("prerouter_<family>.npz", "edge0-35b", "prerouter"),
    ("lora_<family>.npz", "edge0-35b", "lora"),
    ("prerouter_<family>.npz", "edge0-8b", "prerouter"),
    ("lora_<family>.npz", "edge0-8b", "lora"),
]


def _find_source(name: str) -> Path | None:
    for d in NPZ_DIRS:
        p = d / name
        if p.is_file():
            return p
    return None


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rewrite_key(key: str, tier: str, kind: str) -> str:
    """Map a legacy npz key to the edge0 artifact key.

    prerouter heads are normalized to ``layers.<N>.<part>.weight``;
    qwen lora gains the ``language_model.`` prefix (the checkpoint
    namespace the adapter is applied into).
    """
    if kind == "prerouter":
        # Some families use layers.N.*; others namespace the layer index
        # with a prefix segment (<ns>.<N>.<part>.weight).  Both are
        # normalized to layers.<N>.<part>.weight.
        parts = key.split(".")
        if (len(parts) >= 4 and parts[0] != "layers"
                and parts[1].isdigit()):
            return f"layers.{parts[1]}.{parts[2]}.{parts[3]}"
        return key
    if kind == "lora" and tier == "edge0-35b" and key.startswith("model.layers."):
        return "language_model." + key
    return key


def convert(source: Path, tier: str, kind: str, force: bool = False) -> Path:
    name = {
        ("edge0-35b", "prerouter"): "prerouter_edge0_35b_k4.safetensors",
        ("edge0-35b", "lora"): "lora_edge0_35b_k4.safetensors",
        ("edge0-8b", "prerouter"): "prerouter_edge0_8b.safetensors",
        ("edge0-8b", "lora"): "lora_edge0_8b.safetensors",
    }[(tier, kind)]
    out = ARTIFACTS / name
    if out.exists() and not force:
        print(f"skip  {out.name} (exists)")
        return out

    data = np.load(source)
    tensors = {}
    for key in data.files:
        tensors[rewrite_key(key, tier, kind)] = data[key]

    # provenance: rank from the A matrix, alpha from the training run
    # (32 for both shipped pairs), K from the tier profile.
    meta = {
        "model": tier,
        "kind": kind,
        "K": "4" if tier == "edge0-35b" else "8",
        "r": "16", "alpha": "32",
        "source": str(source),
        "source_md5": md5(source),
        "converted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "format_version": "1",
    }
    if kind == "prerouter":
        owners = sorted({int(k.split(".")[1])
                         for k in tensors if k.startswith("layers.")})
        meta["owners"] = json.dumps(owners)

    from safetensors.numpy import save_file as save_np
    save_np(tensors, str(out), metadata={"__metadata__": json.dumps(meta)})
    print(f"write {out.name} ({len(tensors)} tensors, "
          f"{out.stat().st_size >> 20} MiB)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="reconvert even if the artifact exists")
    ap.add_argument("--npz-dir", default=None,
                    help="directory holding the legacy npz exports "
                         "(default: $EDGE0_NPZ_DIR)")
    args = ap.parse_args(argv)

    if args.npz_dir:
        NPZ_DIRS.insert(0, Path(args.npz_dir))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    missing = []
    resolved = []
    for name, tier, kind in SOURCES:
        p = _find_source(name)
        if p is None:
            missing.append(name)
        else:
            resolved.append((p, tier, kind))
    if missing:
        print("missing legacy sources (skip their conversion):")
        for m in missing:
            print("  -", m)
    for path, tier, kind in resolved:
        try:
            convert(path, tier, kind, force=args.force)
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"FAIL {path.name}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
