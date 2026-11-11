"""tools/mtp_pack.py - plan v0.3 P6 prep: the MTP draft block as a Strata GGUF, from the BF16 tensors.

llama.cpp's qwen4exp converter drops the MTP head (`supports_mtp_export = False`), and the flyweight MTP GGUF is
no longer on this PC, so Strata packs its own from the 31 `mtp.*` tensors fetched by tools/mtp_fetch.py:

    python tools/mtp_pack.py --src Desktop/Strata/mtp-bf16 --experts q2_0 --out mtp-q2_0.gguf
    python tools/mtp_pack.py --src ... --experts q4_0 --out mtp-q4_0.gguf       (acceptance comparison arm)

Layout: dense tensors (attention, indexer, hyper-connections, shared expert, router, fc/norms) stay BF16 (F32 for
1-D norms), ~0.18 GB. The routed experts keep the checkpoint's fused layout - `gate_up_proj` [512, 1280, 2560] and
`down_proj` [512, 2560, 640], quantized along the last (input) axis - in one of:

    q2_0   64-element blocks, grid {-1, 0, 1, 2} x d. The scale is chosen per block to MINIMIZE squared error over
           that grid (the ggml reference sets d = max|w| and never uses the +2 level). Same format as the main
           model's experts, so Strata's CPU VNNI kernel and GPU hit kernel serve it unchanged. ~0.71 GB.
    q4_0   ggml reference rounding. ~1.42 GB.       q8_0   ggml reference. ~2.67 GB.

This is round-to-nearest, not GSQ: the plan picks the expert format by MEASURED draft acceptance (P0.3/P6), not
by this file's reconstruction error, which is reported per tensor only as a sanity check. No model runs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
import gguf  # noqa: E402  (the pinned llama.cpp gguf-py, which knows Q2_0 = type 42)

EXPERT_TENSORS = ("mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj")


class _Experts:
    """Index an expert out of a BF16 memmap as float32 on demand."""
    def __init__(self, raw):
        self.raw = raw

    def __getitem__(self, e):
        return (np.asarray(self.raw[e]).astype(np.uint32) << 16).view(np.float32)


def load_bf16(path: Path, shape: list[int]) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    if raw.size != int(np.prod(shape)):
        raise ValueError(f"{path}: {raw.size} values, expected shape {shape}")
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)


# ------------------------------------------------------------------------------------------------ quantizers
def q2_0(w: np.ndarray) -> np.ndarray:
    """[..., n] float32 -> bytes of Q2_0 blocks (fp16 d, 16 bytes of 2-bit codes, code = q + 1, 4 per byte)."""
    x = w.reshape(-1, 64).astype(np.float32)
    amax = np.abs(x).max(axis=1, keepdims=True)
    best_err = np.full((x.shape[0], 1), np.inf, dtype=np.float32)
    best_d = np.zeros_like(amax)
    # Candidate scales span amax/2 (the +2 level reaches the max) to amax (the ggml reference); 17 steps.
    for f in np.linspace(0.5, 1.0, 17, dtype=np.float32):
        d = amax * f
        inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
        q = np.clip(np.rint(x * inv), -1, 2)
        err = ((q * d - x) ** 2).sum(axis=1, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_d = np.where(better, d, best_d)
    d16 = best_d.astype(np.float16)
    d = d16.astype(np.float32)                                   # quantize against the STORED scale
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    codes = (np.clip(np.rint(x * inv), -1, 2) + 1).astype(np.uint8)          # 0..3
    c = codes.reshape(-1, 16, 4)
    packed = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)
    out = np.empty((x.shape[0], 18), dtype=np.uint8)
    out[:, :2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.reshape(-1)


def q4_0(w: np.ndarray) -> np.ndarray:
    """ggml quantize_row_q4_0_ref: d = max / -8 (signed max), code = clamp(round(x/d + 8.5) truncated, 0, 15)."""
    x = w.reshape(-1, 32).astype(np.float32)
    idx = np.abs(x).argmax(axis=1)
    mx = x[np.arange(x.shape[0]), idx][:, None]
    d = mx / -8.0
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0)
    q = np.minimum(15, np.trunc(x * inv + 8.5)).astype(np.uint8)
    packed = (q[:, :16] | (q[:, 16:] << 4)).astype(np.uint8)
    out = np.empty((x.shape[0], 18), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.reshape(-1)


def q8_0(w: np.ndarray) -> np.ndarray:
    x = w.reshape(-1, 32).astype(np.float32)
    d = np.abs(x).max(axis=1, keepdims=True) / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    q = np.rint(x * inv).astype(np.int8)
    out = np.empty((x.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.reshape(-1)


def dequant(kind: str, blob: np.ndarray, n: int) -> np.ndarray:
    if kind == "q2_0":
        b = blob.reshape(-1, 18)
        d = b[:, :2].copy().view(np.float16).astype(np.float32)
        qs = b[:, 2:]
        codes = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], axis=2).reshape(-1, 64).astype(np.float32)
        return ((codes - 1.0) * d).reshape(-1)[:n]
    if kind == "q4_0":
        b = blob.reshape(-1, 18)
        d = b[:, :2].copy().view(np.float16).astype(np.float32)
        qs = b[:, 2:]
        q = np.concatenate([qs & 15, qs >> 4], axis=1).astype(np.float32)
        return ((q - 8.0) * d).reshape(-1)[:n]
    b = blob.reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)[:n]


QUANT = {"q2_0": (q2_0, gguf.GGMLQuantizationType.Q2_0, 64, 18),
         "q4_0": (q4_0, gguf.GGMLQuantizationType.Q4_0, 32, 18),
         "q8_0": (q8_0, gguf.GGMLQuantizationType.Q8_0, 32, 34)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory written by tools/mtp_fetch.py fetch")
    ap.add_argument("--experts", choices=sorted(QUANT), default="q2_0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-experts", type=int, default=8, help="experts per tensor used for the error report")
    a = ap.parse_args()
    src = Path(a.src)
    manifest = json.loads((src / "mtp-manifest.json").read_text())
    fn, qtype, block, block_bytes = QUANT[a.experts]
    w = gguf.GGUFWriter(a.out, "qwen4exp-mtp")
    w.add_string("strata.mtp.source", "Qwen/Qwen3.8-Flash-Next BF16 checkpoint, mtp.* tensors")
    w.add_string("strata.mtp.source_sha256", hashlib.sha256(
        json.dumps({t["name"]: t["sha256"] for t in manifest}, sort_keys=True).encode()).hexdigest())
    w.add_string("strata.mtp.expert_format", a.experts)
    w.add_string("strata.mtp.expert_quantizer", "per-block MSE scale search" if a.experts == "q2_0" else "ggml reference")
    report = []
    for t in sorted(manifest, key=lambda t: t["name"]):
        name, shape = t["name"], t["shape"]
        t0 = time.time()
        if name in EXPERT_TENSORS:
            # Streamed one expert at a time from a read-only view: the fused gate_up tensor alone is 6.7 GB as
            # float32, so it is never materialized whole.
            raw = np.memmap(src / t["file"], dtype="<u2", mode="r", shape=tuple(shape))
            x = _Experts(raw)
            if shape[-1] % block:
                raise ValueError(f"{name}: inner dim {shape[-1]} not a multiple of {block}")
            parts = [fn(x[e]) for e in range(shape[0])]            # one expert at a time bounds memory
            blob = np.concatenate(parts)
            errs = []
            for e in range(min(a.check_experts, shape[0])):
                ref = x[e].reshape(-1)
                got = dequant(a.experts, parts[e], ref.size)
                errs.append(float(np.sqrt(((got - ref) ** 2).mean()) / (np.sqrt((ref ** 2).mean()) + 1e-12)))
            # gguf-py takes the quantized bytes with the ELEMENT shape; ggml order is innermost-first.
            w.add_tensor(name, blob, raw_shape=list(shape[:-1]) + [shape[-1] // block * block_bytes], raw_dtype=qtype)
            report.append({"tensor": name, "format": a.experts, "bytes": int(blob.size),
                           "relative_rms_error_first_experts": round(float(np.mean(errs)), 5),
                           "seconds": round(time.time() - t0, 1)})
        elif len(shape) == 1:
            x = load_bf16(src / t["file"], shape)
            w.add_tensor(name, x.astype(np.float32))
            report.append({"tensor": name, "format": "F32", "bytes": x.size * 4})
        else:
            bf = np.fromfile(src / t["file"], dtype="<u2").reshape(shape)
            w.add_tensor(name, bf, raw_dtype=gguf.GGMLQuantizationType.BF16)
            report.append({"tensor": name, "format": "BF16", "bytes": bf.size * 2})
        print(f"{name}: {report[-1]['format']} {report[-1]['bytes'] / 1e6:.1f} MB", flush=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    total = sum(r["bytes"] for r in report)
    Path(a.out + ".report.json").write_text(json.dumps({"total_bytes": total, "tensors": report}, indent=1))
    print(f"wrote {a.out}: {total / 1e9:.3f} GB of tensor data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
