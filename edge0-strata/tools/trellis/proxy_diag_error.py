"""proxy_diag_error.py - the diagonal-Hessian output error of the swapped layers, no engine needed.

For some experts of the chosen layers: W' = the ORIGINAL weights in the file's rotated basis; shipped = agentionai's decoded
blocks; ours = plain-MSE encode, and activation-weighted encodes (h = E[x^2]^p from a STRATA_ACT_DUMP directory).
Reported: the unweighted relative error ||W' - Q||^2 / ||W'||^2 and the diagonal-Hessian error
sum_i h_i ||W'_:,i - Q_:,i||^2 / sum_i h_i ||W'_:,i||^2 (the expected squared output error under independent channels),
per layer and role, the mean over experts, in dB.
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
import tqenc  # noqa: E402
from build_swap import bf16, load_h  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--layers", default="4,16,28,40")
    ap.add_argument("--experts", default="0,37,111,200,301,420")
    ap.add_argument("--powers", default="1.0,0.5")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--gpu", action="store_true")
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
    powers = [float(x) for x in a.powers.split(",")]
    experts = [int(x) for x in a.experts.split(",")]
    res = {}
    for l in [int(x) for x in a.layers.split(",")]:
        gu_raw = np.memmap(pathlib.Path(a.orig) / ("L%d_gate_up.bf16" % l), dtype=np.uint16, mode="r").reshape(512, 1280, 2560)
        dn_raw = np.memmap(pathlib.Path(a.orig) / ("L%d_down.bf16" % l), dtype=np.uint16, mode="r").reshape(512, 2560, 640)
        hs = {"gu": load_h(a.dump_dir, l, "gu", 1.0)[0], "dn": load_h(a.dump_dir, l, "dn", 1.0)[0]}
        for e in experts:
            gu = bf16(np.asarray(gu_raw[e]))
            dn = bf16(np.asarray(dn_raw[e]))
            for role, W, kind in (("gate", gu[:640], "gu"), ("up", gu[640:], "gu"), ("down", dn, "dn")):
                ti = by["blk.%d.ffn_%s_exps.weight" % (l, role)]
                t = tqenc.TYPE_BY_NAME[ti.type_name]
                n_in, n_out = ti.shape[0], ti.shape[1]
                rb = n_in // 128 * tqenc.block_bytes(t)
                raw = np.frombuffer(mm, dtype=np.uint8, count=n_out * rb, offset=g.data_start + ti.offset + e * n_out * rb)
                ship = tqenc.decode(raw, t).reshape(n_out, n_in)
                Wr = tqenc.rotate(W, signs[n_in])
                h = hs[kind]
                cands = {"shipped": ship}
                kw = dict(iters=1, ncand=1)
                if a.gpu:
                    kw["gpu"] = True
                else:
                    kw["threads"] = a.threads
                pk, _, _, _ = tqenc.encode(Wr, t, **kw)
                cands["plain"] = tqenc.decode(pk, t).reshape(n_out, n_in)
                for p in powers:
                    hp = (h ** p / (h ** p).mean()).astype(np.float32)
                    pk, _, _, _ = tqenc.encode(Wr, t, h=np.broadcast_to(hp, Wr.shape).copy(), **kw)
                    cands["diag p=%g" % p] = tqenc.decode(pk, t).reshape(n_out, n_in)
                en = float(np.sum(Wr.astype(np.float64) ** 2))
                enh = float(np.sum((Wr.astype(np.float64) ** 2) * h))
                for name, Q in cands.items():
                    d = Wr.astype(np.float64) - Q
                    res.setdefault((l, role, name), []).append((float(np.sum(d * d) / en), float(np.sum(d * d * h) / enh)))
            print("layer %d expert %d done" % (l, e), flush=True)
    print("\nrelative error in dB (higher = better), mean over experts: unweighted | diagonal-Hessian")
    names = ["shipped", "plain"] + ["diag p=%g" % p for p in powers]
    tot = {n: [[], []] for n in names}
    for l in [int(x) for x in a.layers.split(",")]:
        for role in ("gate", "up", "down"):
            line = "L%-2d %-5s" % (l, role)
            for n in names:
                v = np.array(res[(l, role, n)])
                line += "  %-11s %5.2f | %5.2f" % (n, -10 * np.log10(v[:, 0].mean()), -10 * np.log10(v[:, 1].mean()))
                tot[n][0].append(v[:, 0].mean())
                tot[n][1].append(v[:, 1].mean())
            print(line)
    print("ALL      " + "  ".join("%-11s %5.2f | %5.2f" % (n, -10 * np.log10(np.mean(tot[n][0])), -10 * np.log10(np.mean(tot[n][1]))) for n in names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
