"""proxy_ldl.py - the Hessian-weighted output error of plain, block-feedback (LDL) and agentionai's encodes, no engine needed.

For some experts of the chosen layers: W' = ORIGINAL weights in the file's rotated basis, shipped = agentionai's decoded blocks,
plain = our plain-MSE encode, ldl = block feedback with the second-moment matrix H of --dump-dir.  Errors are measured with
H_test (--test-dump-dir, a different part of the calibration set; default = H itself) as
    tr(E H E^T) / tr(W H W^T)     (the expected squared output error for inputs with that second moment)
and unweighted ||E||^2 / ||W||^2, in dB, mean over experts.
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
import blockldl  # noqa: E402
import gguf_reader  # noqa: E402
import tqenc  # noqa: E402
from build_swap import bf16, load_hessian  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--test-dump-dir", default=None)
    ap.add_argument("--layers", default="4,16,28,40")
    ap.add_argument("--experts", default="0,111,301")
    ap.add_argument("--roles", default="gate,down")
    ap.add_argument("--damps", default="0.01")
    ap.add_argument("--threads", type=int, default=4)
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
    damps = [float(x) for x in a.damps.split(",")]
    res = {}
    for l in [int(x) for x in a.layers.split(",")]:
        gu_raw = np.memmap(pathlib.Path(a.orig) / ("L%d_gate_up.bf16" % l), dtype=np.uint16, mode="r").reshape(512, 1280, 2560)
        dn_raw = np.memmap(pathlib.Path(a.orig) / ("L%d_down.bf16" % l), dtype=np.uint16, mode="r").reshape(512, 2560, 640)
        for role in a.roles.split(","):
            kind = "dn" if role == "down" else "gu"
            H = load_hessian(a.dump_dir, l, kind)
            Ht = load_hessian(a.test_dump_dir, l, kind) if a.test_dump_dir else H
            fbs = {d: blockldl.feedback_matrices(blockldl.hessian_factor(H, d)) for d in damps}
            for e in [int(x) for x in a.experts.split(",")]:
                W = bf16(np.asarray(gu_raw[e][:640] if role == "gate" else gu_raw[e][640:] if role == "up" else dn_raw[e]))
                ti = by["blk.%d.ffn_%s_exps.weight" % (l, role)]
                t = tqenc.TYPE_BY_NAME[ti.type_name]
                n_in, n_out = ti.shape[0], ti.shape[1]
                rb = n_in // 128 * tqenc.block_bytes(t)
                raw = np.frombuffer(mm, dtype=np.uint8, count=n_out * rb, offset=g.data_start + ti.offset + e * n_out * rb)
                Wr = tqenc.rotate(W, signs[n_in]).astype(np.float64)
                cands = {"shipped": tqenc.decode(raw, t).reshape(n_out, n_in).astype(np.float64)}
                kw = dict(threads=a.threads) if not a.gpu else dict(gpu=True)
                pk, _, _, _ = tqenc.encode(Wr.astype(np.float32), t, iters=1, **kw)
                cands["plain"] = tqenc.decode(pk, t).reshape(n_out, n_in).astype(np.float64)
                for d in damps:
                    _, dec = blockldl.encode_ldl(Wr.astype(np.float32), t, fbs[d], gpu=a.gpu, threads=a.threads)
                    cands["ldl damp=%g" % d] = dec.astype(np.float64)
                den_u = float(np.sum(Wr ** 2))
                den_h = float(np.trace(Wr @ Ht @ Wr.T))
                for name, Q in cands.items():
                    E = Wr - Q
                    res.setdefault((l, role, name), []).append((float(np.sum(E ** 2) / den_u), float(np.trace(E @ Ht @ E.T) / den_h)))
                print("layer %d %s expert %d done" % (l, role, e), flush=True)
    names = ["shipped", "plain"] + ["ldl damp=%g" % d for d in damps]
    print("\nrelative error in dB (higher is better): unweighted | H-weighted (%s)" % ("held-out H" if a.test_dump_dir else "same H"))
    tot = {n: [[], []] for n in names}
    for l in [int(x) for x in a.layers.split(",")]:
        for role in a.roles.split(","):
            line = "L%-2d %-5s" % (l, role)
            for n in names:
                v = np.array(res[(l, role, n)])
                line += "  %-13s %5.2f | %5.2f" % (n, -10 * np.log10(v[:, 0].mean()), -10 * np.log10(v[:, 1].mean()))
                tot[n][0].append(v[:, 0].mean())
                tot[n][1].append(v[:, 1].mean())
            print(line)
    print("ALL        " + "  ".join("%-13s %5.2f | %5.2f" % (n, -10 * np.log10(np.mean(tot[n][0])), -10 * np.log10(np.mean(tot[n][1]))) for n in names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
