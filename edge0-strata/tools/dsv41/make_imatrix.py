#!/usr/bin/env python3
"""make_imatrix.py CAL_DIR OUT.dat : llama.cpp's legacy imatrix (.dat) for requant_gguf.py from the per-expert calibration sums the
gate-3 run dumped (CAL_DIR/L<l>_gu.f64 [384][5120] sums of x^2 of the gate/up input, L<l>_dn.f64 [384][2304] of the down input,
L<l>_cnt.u64 [384] picks).  One entry per expert tensor: blk.<l>.ffn_{gate,up,down}_exps.weight with 384 x ne0 values (E[x^2] per
expert and input channel; an expert with few picks is shrunk toward the layer mean with a prior of 64 picks, as gate 3 did)."""
import struct
import sys

import numpy as np

cal, out = sys.argv[1], sys.argv[2]
PRIOR = 64.0
ents = []
for l in range(40):
    gu = np.fromfile("%s/L%d_gu.f64" % (cal, l), dtype=np.float64).reshape(384, 5120)
    dn = np.fromfile("%s/L%d_dn.f64" % (cal, l), dtype=np.float64).reshape(384, 2304)
    cnt = np.fromfile("%s/L%d_cnt.u64" % (cal, l), dtype=np.uint64).astype(np.float64)
    tot = max(cnt.sum(), 1.0)
    for names, s in ((("gate", "up"), gu), (("down",), dn)):
        mean = s.sum(0) / tot
        h = (s + PRIOR * mean) / (cnt[:, None] + PRIOR)
        h = np.maximum(h, 1e-8).astype(np.float32)
        for nm in names:
            ents.append(("blk.%d.ffn_%s_exps.weight" % (l, nm), h.reshape(-1)))
with open(out, "wb") as f:
    f.write(struct.pack("<i", len(ents)))
    for name, v in ents:
        b = name.encode()
        f.write(struct.pack("<i", len(b)) + b + struct.pack("<ii", 1, v.size) + v.tobytes())
print("wrote", out, len(ents), "entries")
