"""Regenerate quant fixtures with the pinned llama.cpp C library.

Build ggml-base from 6c84c7d5d8833c6e0df69628f75a0f599797934e, then:
  PYTHONPATH=src python scripts/gguf_quant_reference.py /path/libggml-base.dylib
"""
import argparse
import ctypes as ct
from pathlib import Path

import numpy as np

from edge0.checkpoints.gguf import QUANT_SIZES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    parser.add_argument("--output", default="tests/fixtures/gguf_quant_reference.npz")
    args = parser.parse_args()
    lib = ct.CDLL(args.library)
    names = {8: "q8_0", 12: "q4_K", 13: "q5_K", 14: "q6_K", 16: "iq2_xxs", 19: "iq1_s", 20: "iq4_nl"}
    rng = np.random.default_rng(381)
    fixture = {}
    for kind, (block, size) in QUANT_SIZES.items():
        raw = rng.integers(0, 256, (32, size), dtype=np.uint8)
        if kind == 0:
            raw = rng.normal(size=32).astype("<f4").view(np.uint8).reshape(32, 4)
            decoded = raw.copy().view("<f4").reshape(-1)
        elif kind == 30:
            # Finite BF16 values; conversion through upstream ggml_bf16_to_fp32_row.
            raw = (rng.normal(size=32).astype("<f4").view("<u4") >> 16).astype("<u2").view(np.uint8).reshape(32, 2)
            decoded = np.empty(32, dtype=np.float32)
            fn = lib.ggml_bf16_to_fp32_row
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int64]
            fn(raw.ctypes.data, decoded.ctypes.data, 32)
        else:
            # Keep scales finite while exercising every random bit of grids,
            # signs, subscales, and high/low quantized values.
            at = size - 2 if kind == 14 else 0
            raw[:, at:at + 2] = rng.uniform(0.001, 0.2, 32).astype("<f2").view(np.uint8).reshape(32, 2)
            if kind in (12, 13):
                raw[:, 2:4] = rng.uniform(0.001, 0.2, 32).astype("<f2").view(np.uint8).reshape(32, 2)
            decoded = np.empty(32 * block, dtype=np.float32)
            fn = getattr(lib, "dequantize_row_" + names[kind])
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int64]
            fn(raw.ctypes.data, decoded.ctypes.data, decoded.size)
        fixture[f"raw_{kind}"] = raw.reshape(-1)
        fixture[f"decoded_{kind}"] = decoded
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **fixture)
    print(path)


if __name__ == "__main__":
    main()
