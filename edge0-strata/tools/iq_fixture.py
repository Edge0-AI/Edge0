"""tools/iq_fixture.py - deterministic i-quant fixtures for src/kernels/iq_parity.cpp (TODO 24).

For each of the ten types iq_parity tests, writes `<out>/<name>.bin` (int32 header: the KERNEL type id,
rows, cols - then the raw GGUF block bytes) and `<out>/<name>.f32` (rows*cols float32 reference values,
dequantized by the same library the comparison treats as ground truth):

    python tools/iq_fixture.py --out build/iq_fixture
    python tools/iq_fixture.py --out build/iq_fixture --seed 7 --rows 32 --cols 256

Generation is deterministic: an explicit seed and fixed dimensions, and every random choice comes from
numpy's RandomState under that seed.  The blocks are synthetic on purpose: the i-quant formats have no
published quantizer in gguf-py (it dequantizes only), so each block combines sane half-precision scales
(written at the layout's scale offsets, see SCALE_OFFSETS below) with seeded index/sub-scale bytes - the
dequantized reference covers real codebook values either way, which is what the parity compares.

Dependencies: python3 and numpy, plus the repository's VENDORED gguf-py (third_party/llama.cpp/gguf-py -
no pip install; it is put on sys.path relative to this file).  Q2_0 is this repository's own format
(GGML type 42, which gguf-py does not carry): its codec comes from tools/gguf_writer.py.

The kernel ids in the .bin headers are the repo's i-quant dispatch table (iq_kernels.cu is_iq /
iq_row_bytes) and match the vendored gguf-py's enum ids one for one - the layout table below is the
same GGML_QUANT_SIZES both sides compile in.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))
sys.path.insert(0, str(ROOT / "tools"))

try:
    import gguf
    from gguf.quants import dequantize as gguf_dequantize
    from gguf_writer import dequantize_q2_0, quantize_q2_0
except ImportError as e:  # a missing numpy or vendored gguf-py must be explicit, not a traceback
    sys.stderr.write(f"iq_fixture: missing dependency: {e}\n"
                     "numpy and the repository's vendored gguf-py (third_party/llama.cpp/gguf-py) are "
                     "required.  The vendored copy comes with setup's llama.cpp download (the pinned "
                     "LLAMA_CPP_COMMIT zip); on a fresh checkout run setup once, or fetch that zip and "
                     "unpack its gguf-py there.\n")
    sys.exit(3)

QT = gguf.GGMLQuantizationType
# name -> (block values, block bytes, the fp16 scale offsets inside a block).  The layouts mirror
# gguf-py's dequantize_blocks implementations: every one of these formats keeps its half-precision
# scale(s) at fixed offsets (IQ*_X*: offset 0; Q3_K: after the 32-byte hmask, 64-byte qs and the
# 12 scale bytes) - the rest of the block is indices and sub-scales, which are seeded random bytes.
LAYOUT = {   # (block values, block bytes) == the vendored gguf-py's GGML_QUANT_SIZES == the kernels' sizes
    "IQ2_XXS": (256,  66),
    "IQ2_XS":  (256,  74),
    "IQ2_S":   (256,  82),
    "IQ3_XXS": (256,  98),
    "IQ3_S":   (256, 110),
    "IQ1_M":   (256,  56),
    "IQ4_NL":   (32,  18),
    "IQ4_XS":  (256, 136),
    "Q3_K":    (256, 110),
}
HALF_ONE = struct.pack("<e", 1.0)      # fp16 1.0 = 0x3c00: a tame, exactly-representable scale


def repair_scales(name: str, raw: np.ndarray) -> None:
    """Overwrite the fp16 scale bits of every block with a sane value, in place.  The index and
    sub-scale bytes stay seeded-random (their values are bounded by construction).  Layouts mirror
    gguf-py's dequantize_blocks: IQ*_X* and IQ4_* keep the fp16 scale at block offset 0; Q3_K keeps
    it after the 32-byte hmask, 64-byte qs and 12 scale bytes; IQ1_M is "the only one which stores
    the f16 scale in multiple parts" - the fp16 bits are the HIGH NIBBLES of the four uint16s at
    bytes 48..56, so a 1.0 scale is nibbles 3, C, 0, 0 on bytes 49, 51, 54, 55."""
    if name == "Q3_K":
        raw[:, 108:110] = np.frombuffer(HALF_ONE, dtype=np.uint8)
        return
    if name == "IQ1_M":
        raw[:, 48:56] &= np.uint8(0x0F)            # clear the four scale nibbles
        for byte, nib in ((49, 0x30), (51, 0xC0), (54, 0x00), (55, 0x00)):
            raw[:, byte] |= np.uint8(nib)
        return
    raw[:, 0:2] = np.frombuffer(HALF_ONE, dtype=np.uint8)
# name -> (the KERNEL type id written into the .bin header, identical to the gguf-py enum id)
KERNEL_IDS = {"IQ2_XXS": 16, "IQ2_XS": 17, "IQ2_S": 22, "IQ3_XXS": 18, "IQ3_S": 21,
              "IQ1_M": 29, "IQ4_NL": 20, "IQ4_XS": 23, "Q2_0": 42, "Q3_K": 11}


def build(name: str, rows: int, cols: int, seed: int):
    """The .bin body (header + raw blocks) and the .f32 reference for one type."""
    kernel_type = KERNEL_IDS[name]
    n = rows * cols
    rng = np.random.RandomState((seed, hash(name) & 0xFFFF) if False else (seed, KERNEL_IDS[name]))
    if name == "Q2_0":                                 # the repository's own codec, gguf-py has no type 42
        w = rng.standard_normal(n).astype(np.float32)
        raw = np.frombuffer(quantize_q2_0(w), dtype=np.uint8)
        ref = dequantize_q2_0(raw.tobytes()).astype(np.float32)
        flat = raw
    else:
        block_values, block_bytes = LAYOUT[name]
        blocks = n // block_values
        raw = rng.randint(0, 256, (blocks, block_bytes), dtype=np.uint8)
        repair_scales(name, raw)                       # sane fp16 scales; indices stay seeded-random
        flat = raw.reshape(-1)
        ref = np.asarray(gguf_dequantize(flat, getattr(QT, name)), dtype=np.float32)
    if not np.isfinite(ref).all() or float(np.abs(ref).max()) > 1e4:
        raise RuntimeError(f"{name}: the dequantized reference is not finite/tame - the layout table "
                           f"(block bytes, scale offsets) does not match gguf-py's dequantizer")
    header = struct.pack("<3i", kernel_type, rows, cols)
    return header + flat.tobytes(), ref


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="logs/iq_fixture", help="the fixture directory (created; may live in the "
                                                             "build directory - the source tree is not written)")
    ap.add_argument("--seed", type=int, default=7, help="the fixture RNG seed (deterministic generation)")
    ap.add_argument("--rows", type=int, default=32, help="weight rows (rows*cols stays a multiple of 256)")
    ap.add_argument("--cols", type=int, default=256, help="the reduction dimension, a multiple of 256")
    args = ap.parse_args()
    if args.cols % 256 or args.rows * args.cols % 256:
        ap.error("--cols must be a multiple of 256 and rows*cols a multiple of 256")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in KERNEL_IDS:
        blob, ref = build(name, args.rows, args.cols, args.seed)
        (out / f"{name}.bin").write_bytes(blob)
        (out / f"{name}.f32").write_bytes(np.ascontiguousarray(ref, dtype="<f4").tobytes())
        print(f"  {name:<8} type {KERNEL_IDS[name]:>2}  {args.rows} x {args.cols}  "
              f"{len(blob) - 12} block bytes, {ref.size} reference floats")
    print(f"iq_fixture: {len(KERNEL_IDS)} fixtures in {out} (seed {args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
