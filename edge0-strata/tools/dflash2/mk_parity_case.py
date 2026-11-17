"""Builds the input of src/core/dflash2_parity.cpp and the torch reference answer.

    python mk_parity_case.py <drafter.safetensors> <drafter_config.json> <out.bin> [n_ctx=2300] [vp=20000] [seed=1]

The case: n_ctx random tap rows (positions 0 .. n_ctx-1), an anchor token at position n_ctx, a head of `vp` random rows (Q8_0, the draft vocabulary
subset path: dvocab maps a row to a token id) and the target embedding rows of the anchor and the mask token.  The torch port (dflash2_ref.py,
equal to z-lab's module) gives the reference: the final normalized block rows, the walk's tokens and their probabilities.  A context longer than the
2048-token window exercises the ring wrap and the window mask.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import dflash2_ref as R


class Embed:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, ids):
        return torch.stack([self.rows[int(i)] for i in ids])


def quant_q8_0(w):
    rows, cols = w.shape
    b = w.reshape(rows, cols // 32, 32)
    amax = np.abs(b).max(axis=-1)
    d = amax / 127.0
    inv = np.where(d != 0, 1.0 / np.where(d == 0, 1, d), 0.0)
    q = np.rint(b * inv[..., None]).astype(np.int8)
    out = np.empty((rows, cols // 32, 34), np.uint8)
    out[..., :2] = d.astype(np.float16).view(np.uint8).reshape(rows, cols // 32, 2)
    out[..., 2:] = q.view(np.uint8)
    return out.tobytes(), (q.astype(np.float32) * d.astype(np.float16).astype(np.float32)[..., None]).reshape(rows, cols)


def main():
    st, cfgp, out = sys.argv[1:4]
    n_ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 2300
    vp = int(sys.argv[5]) if len(sys.argv) > 5 else 20000
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    rng = np.random.default_rng(seed)
    H, V = 2560, 248320
    mask_id = 248077
    anchor = int(rng.integers(1000, 200000))
    taps = (rng.standard_normal((n_ctx, 5 * H)) * 1.0).astype(np.float32)
    emb = (rng.standard_normal((2, H)) * 1.0).astype(np.float32)
    dvocab = np.sort(rng.choice(V, size=vp, replace=False)).astype(np.int32)
    head = (rng.standard_normal((vp, H)) * 0.05).astype(np.float32)
    head_blob, head_dq = quant_q8_0(head)
    embed = Embed({anchor: torch.from_numpy(emb[0]), mask_id: torch.from_numpy(emb[1])})
    d = R.Drafter(st, cfgp, embed, torch.from_numpy(head_dq), threads=6)
    d.set_context(torch.from_numpy(taps))
    path, _ = d.draft(n_ctx, anchor, 8)           # runs the forward; the tail below redoes the selector with the token map
    fn = d.last_all.clone()                       # [8][H]
    out_rows = fn[1:]
    logits = out_rows @ torch.from_numpy(head_dq).T
    unary, idx = torch.topk(logits, 16, dim=-1)
    cand = torch.from_numpy(dvocab.astype(np.int64))[idx]
    hp = out_rows @ d.w["candidate_selector.hidden_projection.weight"].T
    A, B = d.w["candidate_selector.predecessor_codebook"], d.w["candidate_selector.successor_codebook"]
    pred = anchor
    toks, probs = [], []
    for i in range(7):
        sc = unary[i] + (B[cand[i]].float() @ (A[pred].float() * hp[i]))
        j = int(torch.argmax(sc))
        probs.append(float(torch.softmax(sc, dim=-1)[j]))
        pred = int(cand[i][j])
        toks.append(pred)
    with open(out, "wb") as f:
        f.write(struct.pack("<4i", n_ctx, anchor, vp, 0))
        f.write(taps.tobytes())
        f.write(emb.tobytes())
        f.write(dvocab.tobytes())
        f.write(head_blob)
        f.write(struct.pack("<7i", *toks))
        f.write(struct.pack("<7f", *probs))
        f.write(fn.numpy().astype(np.float32).tobytes())
    print("case written:", out, "n_ctx", n_ctx, "anchor", anchor, "tokens", toks, "probs", [round(p, 3) for p in probs])


if __name__ == "__main__":
    main()
