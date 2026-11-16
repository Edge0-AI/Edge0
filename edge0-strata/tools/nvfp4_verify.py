"""tools/nvfp4_verify.py - prove the ModelOpt -> ggml NVFP4 repack is lossless on a real checkpoint.

    python tools/nvfp4_verify.py --model <ModelOpt NVFP4 dir> [--experts 4] [--scan-all]

llama.cpp's converter repacks ModelOpt NVFP4 by moving bits: nibbles into ggml's split-half order, and each
E4M3 block scale into a UE4M3 byte by stripping its sign bit. That is lossless only while every scale is a
finite, non-negative E4M3 - a negative scale would silently flip, a NaN would propagate. So this checks two
things independently of the converter's own code path:

  1. every weight_scale byte in the checkpoint (--scan-all): no sign bit set, no NaN encoding (0x7F / 0xFF);
  2. for a few experts: dequantize ModelOpt directly (E2M1 value x E4M3 scale x weight_scale_2) and compare
     with gguf-py's dequantization of the repacked block_nvfp4 bytes times the same weight_scale_2. They must
     agree bit for bit in float32.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from safetensors import safe_open

LLAMA = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "llama.cpp"
sys.path.insert(0, str(LLAMA / "gguf-py"))
sys.path.insert(0, str(LLAMA))
from gguf import GGMLQuantizationType as Q, quants  # noqa: E402
from conversion.base import ModelBase  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def modelopt_dequant(weight: torch.Tensor, scale: torch.Tensor, scale2: torch.Tensor) -> np.ndarray:
    """The ModelOpt definition, written out independently of the converter."""
    rows = weight.shape[0]
    lo = (weight & 0x0F).long()
    hi = (weight >> 4).long()
    nib = torch.stack([lo, hi], dim=-1).reshape(rows, -1)          # element 2j low, 2j+1 high
    vals = E2M1[nib]
    s = scale.float().repeat_interleave(16, dim=1)
    return (vals * s * scale2.float()).numpy().astype(np.float32)


def check_scales(model: pathlib.Path) -> int:
    index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name, shard in index.items():
        if name.endswith(".weight_scale"):
            by_shard.setdefault(shard, []).append(name)
    n_bytes = n_sign = n_nan = n_tensors = 0
    worst = []
    for i, (shard, names) in enumerate(sorted(by_shard.items())):
        with safe_open(model / shard, framework="pt") as f:
            for name in names:
                t = f.get_tensor(name)
                if t.dtype != torch.float8_e4m3fn or t.ndim != 2:
                    continue                                          # FP8 per-channel scales are not NVFP4
                b = t.view(torch.uint8)
                sign = int((b & 0x80).count_nonzero())
                nan = int(((b & 0x7F) == 0x7F).sum())
                n_bytes += b.numel()
                n_sign += sign
                n_nan += nan
                n_tensors += 1
                if sign or nan:
                    worst.append((name, sign, nan))
        if i % 40 == 0:
            print("  scanned %3d/%d shards, %d scale tensors" % (i + 1, len(by_shard), n_tensors), flush=True)
    print("scale bytes %.2f GiB in %d tensors: sign bit set %d, NaN %d" % (n_bytes / 2**30, n_tensors, n_sign, n_nan))
    for w in worst[:10]:
        print("  BAD", *w)
    return 1 if (n_sign or n_nan) else 0


def check_experts(model: pathlib.Path, n: int) -> int:
    index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    picks = [name for name in sorted(index)
             if ".experts." in name and name.endswith(".weight") and ".mtp." not in name
             and name.replace(".weight", ".weight_scale") in index][:: max(1, 50000 // max(1, n))][:n * 3]
    bad = 0
    for name in picks:
        shard = model / index[name]
        with safe_open(shard, framework="pt") as f:
            w = f.get_tensor(name)
            s = f.get_tensor(name.replace(".weight", ".weight_scale"))
            s2_name = name.replace(".weight", ".weight_scale_2")
            s2 = f.get_tensor(s2_name) if s2_name in f.keys() else torch.tensor(1.0)
        ref = modelopt_dequant(w, s, s2)
        raw, shape = ModelBase._nvfp4_pack(w, s)
        got = quants.dequantize(np.ascontiguousarray(raw), Q.NVFP4).reshape(shape).astype(np.float32)
        got = (torch.from_numpy(got) * s2.float()).numpy()
        # E2M1 code 8 is -0.0; ggml's kvalues_fp4 stores both zeros as integer 0, so the repacked weights give
        # +0.0 where ModelOpt gives -0.0. That is the same value in every sum (x + -0 == x), so normalise zeros
        # (adding +0.0 turns -0.0 into +0.0) and keep the comparison bitwise for everything else.
        neg_zero = int(np.count_nonzero((ref == 0) & np.signbit(ref)))
        same = np.array_equal((ref + np.float32(0)).view(np.uint32), (got + np.float32(0)).view(np.uint32))
        diff = float(np.abs(ref - got).max())
        print("  %-70s %-12s bitwise %s  max|diff| %.3g  (-0.0 in source: %d)"
              % (name[-70:], tuple(shape), "OK " if same else "NO ", diff, neg_zero))
        bad += not same
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--experts", type=int, default=4, help="how many experts to compare (3 projections each)")
    ap.add_argument("--scan-all", action="store_true", help="check every weight_scale byte in the checkpoint")
    a = ap.parse_args()
    model = pathlib.Path(a.model)
    rc = check_experts(model, a.experts)
    if a.scan_all:
        rc |= check_scales(model)
    print("RESULT:", "lossless" if rc == 0 else "NOT lossless")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
