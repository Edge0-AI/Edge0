"""gyro_column_profile.py - is the unevenness of agentionai's per-input-column error a property of the layer (an
activation-aware objective shared by all experts that read the same input) or noise?  Compares the per-column squared-error
profile (rotated basis) between experts and between gate and up of one layer, for their file and for our plain encode.
"""
from __future__ import annotations

import argparse
import mmap
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import gguf_reader  # noqa: E402
import hf_slice  # noqa: E402
import tqenc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--repo", default="Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--experts", default="0,1,2,3")
    a = ap.parse_args()
    g = gguf_reader.GGUFFile(a.gguf)
    fh = open(a.gguf, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    by = {t.name: t for t in g.tensors}
    sw = [int(x) for x in g.metadata["prism.hadamard.sign_widths"]]
    sv = np.array(g.metadata["prism.hadamard.sign_values"], dtype=np.float32)
    signs, at = {}, 0
    for w in sw:
        signs[w] = sv[at:at + w]
        at += w
    repo = hf_slice.Repo(a.repo)
    pre = "model.language_model.layers.%d.mlp.experts." % a.layer
    prof = {}   # (who, role, expert) -> per-column mean squared error / mean
    for e in [int(x) for x in a.experts.split(",")]:
        gu = repo.tensor(pre + "gate_up_proj", e)[0]
        for role, W in (("gate", gu[:640]), ("up", gu[640:])):
            ti = by["blk.%d.ffn_%s_exps.weight" % (a.layer, role)]
            t = tqenc.TYPE_BY_NAME[ti.type_name]
            n_in, n_out = ti.shape[0], ti.shape[1]
            bb = tqenc.block_bytes(t)
            rb = n_in // 128 * bb
            raw = np.frombuffer(mm, dtype=np.uint8, count=n_out * rb, offset=g.data_start + ti.offset + e * n_out * rb)
            Wd = tqenc.decode(raw, t).reshape(n_out, n_in)
            Wr = tqenc.rotate(W, signs[n_in])
            packed, _, _, _ = tqenc.encode(Wr, t, iters=1, threads=8)
            Wo = tqenc.decode(packed, t).reshape(n_out, n_in)
            for who, Wq in (("theirs", Wd), ("ours", Wo)):
                c = ((Wr - Wq).astype(np.float64) ** 2).mean(axis=0)
                # remove the part explained by the signal's own column energy (weights of large columns err more)
                prof[(who, role, e)] = c / c.mean()
            print("expert %d %s fetched and encoded" % (e, role), flush=True)
    es = [int(x) for x in a.experts.split(",")]
    for who in ("theirs", "ours"):
        cc = [np.corrcoef(prof[(who, "gate", es[0])], prof[(who, "gate", e)])[0, 1] for e in es[1:]]
        cu = [np.corrcoef(prof[(who, "gate", e)], prof[(who, "up", e)])[0, 1] for e in es]
        print("%-7s corr of the per-column error profile: gate expert %d vs other experts' gate %s | gate vs up of the same expert %s" % (
            who, es[0], " ".join("%.3f" % x for x in cc), " ".join("%.3f" % x for x in cu)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
