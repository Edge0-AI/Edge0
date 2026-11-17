#!/usr/bin/env python3
"""stream_experts.py: the routed experts of a GGUF re-laid for SSD streaming - every expert's gate, up and down rows
next to each other, so the Metal engine reads a missed expert with one request instead of three (the M4 Mac mini's
SSD: 2.97 GB/s for one 1.75 MB read per expert against 2.30 GB/s for three 0.58 MB reads, 6 threads, F_NOCACHE).

    python3 metal/tools/stream_experts.py --gguf <model shard 1 or the repacked -metal.gguf> [--out <gguf>.experts] [--verify]

The file: a 64-byte header ("STRX", version 1, 48 layers, 512 experts, the source GGUF's size), then per layer its
offset in this file and the three slice sizes (4 x u64 each), then the experts, layer by layer, expert by expert:
gate slice, up slice, down slice - each byte for byte the GGUF's.  --verify compares every expert with the GGUF.
The engine (metal/src/engine/stream.cpp) uses it when it sits beside the GGUF as <gguf>.experts and its header
matches; otherwise it reads the GGUF as before.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metal_repack as R  # noqa: E402

NL, NE = 48, 512
ALIGN = 16384


def plan(gguf: Path):
    """The reader, each layer's (gate, up, down, slice sizes), each layer's offset in the file, the file's size, the
    first data offset."""
    rd = R.Reader(gguf)
    tab = {t["name"]: t for t in rd.tensors}
    lay = []
    for l in range(NL):
        g, u, d = (tab[f"blk.{l}.ffn_{k}_exps.weight"] for k in ("gate", "up", "down"))
        lay.append((g, u, d, g["bytes"] // NE, u["bytes"] // NE, d["bytes"] // NE))
    head_len = 64 + NL * 32
    data0 = (head_len + ALIGN - 1) // ALIGN * ALIGN
    offs, at = [], data0
    for g, u, d, gs, us, ds in lay:
        offs.append(at)
        at += NE * (gs + us + ds)
        at = (at + ALIGN - 1) // ALIGN * ALIGN
    return rd, lay, offs, at, data0


def size_bytes(gguf: Path) -> int:
    """The size the file for this GGUF will have (setup checks the free disk with it)."""
    rd, _, _, at, _ = plan(gguf)
    rd.f.close()
    return at


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    out = a.out or a.gguf.with_name(a.gguf.name + ".experts")
    rd, lay, offs, at, data0 = plan(a.gguf)
    head = struct.pack("<4sIIIQ", b"STRX", 1, NL, NE, os.path.getsize(a.gguf)).ljust(64, b"\0")
    for (g, u, d, gs, us, ds), o in zip(lay, offs):
        head += struct.pack("<QQQQ", o, gs, us, ds)
    tmp = out.with_name(out.name + ".part")
    src = rd.f
    with open(tmp, "wb") as f:
        f.write(head.ljust(data0, b"\0"))
        for l, ((g, u, d, gs, us, ds), o) in enumerate(zip(lay, offs)):
            src.seek(g["abs"]); G = src.read(g["bytes"])
            src.seek(u["abs"]); U = src.read(u["bytes"])
            src.seek(d["abs"]); D = src.read(d["bytes"])
            f.seek(o)
            f.write(b"".join(G[e * gs:(e + 1) * gs] + U[e * us:(e + 1) * us] + D[e * ds:(e + 1) * ds] for e in range(NE)))
            print(f"\r  layer {l + 1} / {NL}", end="", flush=True)
        f.truncate(at)
    print()
    if a.verify:
        with open(tmp, "rb") as f:
            for l, ((g, u, d, gs, us, ds), o) in enumerate(zip(lay, offs)):
                h1, h2 = hashlib.sha256(), hashlib.sha256()
                for t in (g, u, d):
                    src.seek(t["abs"])
                    h1.update(src.read(t["bytes"]))
                f.seek(o)
                blob = f.read(NE * (gs + us + ds))
                step = gs + us + ds
                for part, size, base in ((0, gs, 0), (1, us, gs), (2, ds, gs + us)):
                    h2.update(b"".join(blob[e * step + base:e * step + base + size] for e in range(NE)))
                if h1.digest() != h2.digest():
                    tmp.unlink()
                    sys.exit(f"verify FAILED at layer {l} (output removed)")
        print(f"verify: {NL} layers x {NE} experts identical to the GGUF")
    os.replace(tmp, out)
    print(f"{out}: {at / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
