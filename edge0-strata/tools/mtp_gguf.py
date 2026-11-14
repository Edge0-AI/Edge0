"""Rebuild the Metal engine's MTP GGUF (mtp-q2_0.gguf) from the CUDA runtime directory (mtp/).

The runtime directory is written by Strata's tools/mtp_rt.py from mtp-q2_0.gguf. This script inverts it:

  experts.bin   Q2_0 blobs relaid out back to the fused gate_up_proj / down_proj tensors (byte-exact).
  dense.bin     bf16 entries are copied as BF16; f32 entries (RMSNorm weights stored as 1 + w) are written
                as w; q8_0 entries are written as Q8_0 tensors. The Metal engine quantizes BF16 matrices to
                Q8_0 when it loads the drafter and uses Q8_0 tensors as they are.

    python mtp_gguf.py --rt <model>/mtp --out <model>/mtp-q2_0.gguf
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

H, FF, NE = 2560, 640, 512
Q2_BLOCK, Q2_BYTES = 64, 18
GU_ROWS, GU_BLOCKS = 2 * FF, H // Q2_BLOCK      # gate_up: 1280 rows of 2560 inputs
DN_ROWS, DN_BLOCKS = H, FF // Q2_BLOCK          # down: 2560 rows of 640 inputs
BLOB = (GU_ROWS * GU_BLOCKS + DN_ROWS * DN_BLOCKS) * Q2_BYTES
ALIGN = 32

T_F32, T_Q8_0, T_BF16, T_Q2_0 = 0, 8, 30, 42
GGUF_STRING, GGUF_UINT32 = 8, 4


def experts(rt: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.fromfile(rt / "experts.bin", dtype=np.uint8)
    if raw.size != NE * BLOB:
        raise ValueError(f"experts.bin: {raw.size} bytes, expected {NE * BLOB}")
    raw = raw.reshape(NE, BLOB)
    gu_codes = GU_ROWS * GU_BLOCKS * 16
    dn_codes = DN_ROWS * DN_BLOCKS * 16
    gu_scales = GU_ROWS * GU_BLOCKS * 2
    o = 0
    parts = []
    for n in (gu_codes, dn_codes, gu_scales):
        parts.append(raw[:, o:o + n])
        o += n
    parts.append(raw[:, o:])
    gu = np.empty((NE, GU_ROWS, GU_BLOCKS, Q2_BYTES), dtype=np.uint8)
    dn = np.empty((NE, DN_ROWS, DN_BLOCKS, Q2_BYTES), dtype=np.uint8)
    inter = np.empty_like(gu)
    inter[..., 2:] = parts[0].reshape(NE, GU_ROWS, GU_BLOCKS, 16)
    inter[..., :2] = parts[2].reshape(NE, GU_ROWS, GU_BLOCKS, 2)
    dn[..., 2:] = parts[1].reshape(NE, DN_ROWS, DN_BLOCKS, 16)
    dn[..., :2] = parts[3].reshape(NE, DN_ROWS, DN_BLOCKS, 2)
    gu[:, :FF] = inter[:, 0::2]                  # interleaved rows: 2r = gate r, 2r + 1 = up r
    gu[:, FF:] = inter[:, 1::2]
    return gu, dn


def full_name(short: str) -> str:
    top = ("fc_embedding.", "fc_hidden.", "hyper_connection_mixer.", "pre_fc_norm_")
    return "mtp." + short if short.startswith(top) else "mtp.layers.0." + short


def dense(rt: Path):
    blob = (rt / "dense.bin").read_bytes()
    for line in (rt / "dense.txt").read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        short, kind, rows, cols, off, size = line.split()
        rows, cols, off, size = int(rows), int(cols), int(off), int(size)
        data = blob[off:off + size]
        if kind == "f32":
            plus_one = np.frombuffer(data, dtype="<f4")
            w = (plus_one - np.float32(1.0)).astype(np.float32)
            # The checkpoint's norms are BF16. Snap w to the nearest BF16 value where that keeps 1 + w, which
            # recovers small weights whose low bits the addition dropped.
            u = w.view(np.uint32)
            snapped = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)
            w = np.where(snapped + np.float32(1.0) == plus_one, snapped, w).astype(np.float32)
            if not np.array_equal(w + np.float32(1.0), plus_one):
                raise ValueError(f"{short}: 1 + w does not round-trip")
            yield full_name(short), T_F32, [cols], w.tobytes()
        elif kind == "bf16":
            if size != rows * cols * 2:
                raise ValueError(f"{short}: {size} bytes for bf16 {rows}x{cols}")
            yield full_name(short), T_BF16, [cols, rows], data
        elif kind == "q8_0":
            if cols % 32 or size != rows * cols // 32 * 34:
                raise ValueError(f"{short}: {size} bytes for q8_0 {rows}x{cols}")
            yield full_name(short), T_Q8_0, [cols, rows], data
        else:
            raise ValueError(f"{short}: unknown kind {kind}")


def gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def write_gguf(path: Path, tensors: list[tuple[str, int, list[int], bytes]]) -> None:
    kv = [("general.architecture", "qwen4exp-mtp"),
          ("strata.mtp.source", "Edge0 release mtp/ runtime files (tools/mtp_rt.py output), inverted"),
          ("strata.mtp.expert_format", "q2_0")]
    tensors = sorted(tensors, key=lambda t: t[0])
    head = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(kv) + 1))
    head += gguf_string("general.alignment") + struct.pack("<II", GGUF_UINT32, ALIGN)
    for k, v in kv:
        head += gguf_string(k) + struct.pack("<I", GGUF_STRING) + gguf_string(v)
    off = 0
    for name, ty, dims, data in tensors:
        head += gguf_string(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        head += struct.pack("<IQ", ty, off)
        off += len(data) + (-len(data)) % ALIGN
    head += b"\0" * ((-len(head)) % ALIGN)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.partial")
    try:
        with open(tmp, "wb") as f:
            f.write(head)
            for _, _, _, data in tensors:
                f.write(data)
                f.write(b"\0" * ((-len(data)) % ALIGN))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rt", required=True, help="the model's mtp/ directory (experts.bin, dense.bin, dense.txt)")
    ap.add_argument("--out", required=True, help="output GGUF, e.g. <model>/mtp-q2_0.gguf")
    a = ap.parse_args()
    rt = Path(a.rt)
    gu, dn = experts(rt)
    tensors = list(dense(rt))
    tensors.append(("mtp.layers.0.mlp.experts.gate_up_proj", T_Q2_0, [H, GU_ROWS, NE], gu.tobytes()))
    tensors.append(("mtp.layers.0.mlp.experts.down_proj", T_Q2_0, [FF, DN_ROWS, NE], dn.tobytes()))
    if len(tensors) != 31:
        raise ValueError(f"{len(tensors)} tensors, expected 31")
    write_gguf(Path(a.out), tensors)
    print(f"{a.out}: {len(tensors)} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
