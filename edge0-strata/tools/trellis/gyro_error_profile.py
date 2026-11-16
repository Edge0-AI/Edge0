"""gyro_error_profile.py - how is agentionai's quantization error distributed, compared with a plain-MSE encode?

For a few expert matrices of a Gyro file: W' = rotated original (hf_slice), Wd = the file's decoded weights, Wo = our plain
encode.  Reports, per column of the (rotated) input axis and per row, how uneven the squared error is (a plain-MSE encoder
spreads it evenly; an activation-weighted one concentrates it where the weights matter less), the scale ratio of their
blocks to the least-squares scale of their own path, and the correlation of their error with ours.
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
    ap.add_argument("--layers", default="10")
    ap.add_argument("--experts", default="0")
    ap.add_argument("--roles", default="gate,down")
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
    for layer in [int(x) for x in a.layers.split(",")]:
        pre = "model.language_model.layers.%d.mlp.experts." % layer
        for e in [int(x) for x in a.experts.split(",")]:
            gu = repo.tensor(pre + "gate_up_proj", e)[0]
            dn = repo.tensor(pre + "down_proj", e)[0]
            mats = {"gate": gu[:640], "up": gu[640:], "down": dn}
            for role in a.roles.split(","):
                W = mats[role]
                ti = by["blk.%d.ffn_%s_exps.weight" % (layer, role)]
                t = tqenc.TYPE_BY_NAME[ti.type_name]
                n_in, n_out = ti.shape[0], ti.shape[1]
                bb = tqenc.block_bytes(t)
                rb = n_in // 128 * bb
                raw = np.frombuffer(mm, dtype=np.uint8, count=n_out * rb, offset=g.data_start + ti.offset + e * n_out * rb)
                Wd = tqenc.decode(raw, t).reshape(n_out, n_in)
                Wr = tqenc.rotate(W, signs[n_in])
                packed, _, _, _ = tqenc.encode(Wr, t, iters=3, ncand=3, threads=8)
                Wo = tqenc.decode(packed, t).reshape(n_out, n_in)
                Ed, Eo = (Wr - Wd).astype(np.float64), (Wr - Wo).astype(np.float64)
                print("L%d e%d %s %s" % (layer, e, role, ti.type_name))
                for name, E in (("agentionai", Ed), ("ours (plain MSE)", Eo)):
                    col = (E ** 2).mean(axis=0)
                    row = (E ** 2).mean(axis=1)
                    blk = (E ** 2).reshape(n_out, n_in // 128, 128).mean(axis=2)
                    pos = (E ** 2).reshape(-1, 128).mean(axis=0)
                    print("  %-18s rel MSE %.4f | per-column error: cv %.3f, max/mean %.2f | per-row: cv %.3f | per-128-block: cv %.3f | "
                          "position-in-block cv %.3f | corr(error, |w|) %.3f" % (
                              name, (E ** 2).sum() / (Wr.astype(np.float64) ** 2).sum(), col.std() / col.mean(), col.max() / col.mean(),
                              row.std() / row.mean(), blk.std() / blk.mean(), pos.std() / pos.mean(),
                              np.corrcoef(np.abs(Wr).ravel()[::7], np.abs(E).ravel()[::7])[0, 1]))
                # their scale vs the least-squares scale for their own path
                dv = raw.reshape(-1, bb)[:, :2].copy().view(np.float16).astype(np.float64).ravel()
                cw = tqenc.decode(np.concatenate([np.zeros((raw.size // bb, 2), np.uint8), raw.reshape(-1, bb)[:, 2:]], axis=1), t)  # d = 0 -> zeros
                # unit-scale path values: decode with d = 1.0 (fp16 0x3c00)
                one = raw.reshape(-1, bb).copy()
                one[:, 0] = 0x00
                one[:, 1] = 0x3c
                cu = tqenc.decode(one, t).reshape(-1, 128).astype(np.float64)
                wb = Wr.reshape(-1, 128).astype(np.float64)
                ls = (wb * cu).sum(axis=1) / (cu * cu).sum(axis=1)
                print("  their scale / LS-optimal scale for their own path: median %.3f  p10 %.3f  p90 %.3f" % (
                    np.median(dv / ls), np.percentile(dv / ls, 10), np.percentile(dv / ls, 90)))
                print("  error correlation (theirs, ours): %.3f   energy of their error that is orthogonal to ours: %.3f" % (
                    np.corrcoef(Ed.ravel()[::5], Eo.ravel()[::5])[0, 1], 1 - (Ed * Eo).sum() ** 2 / ((Ed ** 2).sum() * (Eo ** 2).sum())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
