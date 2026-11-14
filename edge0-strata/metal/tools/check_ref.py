"""metal/tools/check_ref.py - checks metal/src/ref_quant.cpp's dequantizers against gguf-py's (and Q2_0 against
tools/gguf_writer.py's codec) on the files tests/ref_dump wrote:

    python metal/tools/check_ref.py <dir>

Prints the max |difference| per type; the reference passes when every one is 0 (bitwise: both decode to float32).
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gguf.quants import dequantize  # noqa: E402
from gguf.constants import GGMLQuantizationType as Q  # noqa: E402

ROWS, COLS = 16, 2560
d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
bad = 0
for name in ("Q2_0", "Q4_0", "Q5_0", "Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS",
             "IQ3_S", "IQ4_NL", "IQ4_XS", "BF16", "F16"):
    raw = np.fromfile(d / f"{name}.bin", dtype=np.uint8)
    ref = np.fromfile(d / f"{name}.f32", dtype=np.float32).reshape(ROWS, COLS)
    if name == "Q2_0":
        from gguf_writer import dequantize_q2_0  # noqa: E402
        got = np.asarray(dequantize_q2_0(raw.tobytes()), dtype=np.float32).reshape(ROWS, COLS)
    else:
        got = dequantize(raw.reshape(ROWS, -1), Q[name]).astype(np.float32).reshape(ROWS, COLS)
    diff = float(np.max(np.abs(got - ref)))
    scale = float(np.max(np.abs(got)))
    print(f"{name:8s} max |ref_quant - gguf-py| = {diff:.3e}  (max |w| {scale:.3e})  {'ok' if diff == 0 else 'DIFF'}")
    bad += diff != 0
sys.exit(1 if bad else 0)
