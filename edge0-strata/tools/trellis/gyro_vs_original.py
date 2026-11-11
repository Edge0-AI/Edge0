"""gyro_vs_original.py - the fair comparison: agentionai's Gyro quantization of Qwen3.8-Flash-Next against OUR encoder, both
measured against the original BF16 expert weights (fetched tensor by tensor with HTTP range requests, ~10 MB per expert).

For a layer and expert: W = original (gate, up from the fused gate_up_proj, down), rotated into the file's basis with the
file's own sign vectors (prism.hadamard.sign_values, widths 2560 / 640): W' = H (s * w) per row and 128-block.
  agentionai: Wd  = decode of the file's blocks
  ours:       Wo  = decode(encode(W'))   at the same type (TQK6 gate/up, TQK7 down), several search settings
Reported: relative MSE and SQNR of each against W'.

  python3 tools/trellis/gyro_vs_original.py --gguf <Gyro-S> --layers 0,20,39 --experts 0,5 --cfg fast,default,rot3
"""
from __future__ import annotations

import argparse
import mmap
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import gguf_reader  # noqa: E402
import hf_slice  # noqa: E402
import tqenc  # noqa: E402

CFGS = {
    "fast": dict(iters=0, ncand=1, scale_tries=1),
    "default": dict(iters=1, ncand=1, scale_tries=1),
    "iters3": dict(iters=3, ncand=1, scale_tries=1),
    "rot3": dict(iters=3, ncand=3, scale_tries=1),
    "scale3": dict(iters=3, ncand=3, scale_tries=3),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--repo", default="Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("--layers", default="0,20,39")
    ap.add_argument("--experts", default="0")
    ap.add_argument("--cfg", default="fast,default,rot3")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

    g = gguf_reader.GGUFFile(a.gguf)
    fh = open(a.gguf, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    by_name = {t.name: t for t in g.tensors}
    sw = [int(x) for x in g.metadata["prism.hadamard.sign_widths"]]
    sv = np.array(g.metadata["prism.hadamard.sign_values"], dtype=np.float32)
    signs, at = {}, 0
    for w in sw:
        signs[w] = sv[at:at + w]
        at += w
    print("sign widths", sw)
    repo = hf_slice.Repo(a.repo)
    cfgs = a.cfg.split(",")
    tot = {k: [0.0, 0.0] for k in ["agentionai"] + cfgs}
    tot_role = {}
    for layer in [int(x) for x in a.layers.split(",")]:
        pre = "model.language_model.layers.%d.mlp.experts." % layer
        for e in [int(x) for x in a.experts.split(",")]:
            t0 = time.time()
            gu = repo.tensor(pre + "gate_up_proj", e)[0]      # [1280, 2560]
            dn = repo.tensor(pre + "down_proj", e)[0]         # [2560, 640]
            print("layer %d expert %d fetched in %.1f s: gate_up %s down %s" % (layer, e, time.time() - t0, gu.shape, dn.shape))
            mats = {"gate": gu[:640], "up": gu[640:], "down": dn}
            for role, W in mats.items():
                ti = by_name["blk.%d.ffn_%s_exps.weight" % (layer, role)]
                t = tqenc.TYPE_BY_NAME[ti.type_name]
                n_in, n_out = ti.shape[0], ti.shape[1]
                assert W.shape == (n_out, n_in), (W.shape, ti.shape)
                bb = tqenc.block_bytes(t)
                row_bytes = n_in // 128 * bb
                off = g.data_start + ti.offset + e * n_out * row_bytes
                raw = np.frombuffer(mm, dtype=np.uint8, count=n_out * row_bytes, offset=off)
                wd = tqenc.decode(raw, t).reshape(n_out, n_in)
                Wr = tqenc.rotate(W, signs[n_in])
                en = float(np.sum(Wr.astype(np.float64) ** 2))
                ea = float(np.sum((Wr.astype(np.float64) - wd) ** 2))
                # sanity: the fused gate_up order (gate first) - if swapped the SQNR collapses
                line = ["agentionai %s %.2f dB" % (ti.type_name, 10 * np.log10(en / ea))]
                key = "%s" % role
                tot["agentionai"][0] += ea; tot["agentionai"][1] += en
                tr = tot_role.setdefault(key, {k: [0.0, 0.0] for k in ["agentionai"] + cfgs})
                tr["agentionai"][0] += ea; tr["agentionai"][1] += en
                for c in cfgs:
                    t1 = time.time()
                    packed, err, totr, st = tqenc.encode(Wr, t, threads=a.threads, **CFGS[c])
                    sec = time.time() - t1
                    line.append("%s %.2f dB (%.0fk w/s)" % (c, 10 * np.log10(en / totr), Wr.size / sec / 1e3))
                    tot[c][0] += totr; tot[c][1] += en
                    tr[c][0] += totr; tr[c][1] += en
                print("  L%d e%d %-4s  " % (layer, e, role) + " | ".join(line))
    print("\nTOTAL SQNR (dB) against the rotated original, all tested matrices")
    for k, (er, en) in tot.items():
        print("  %-10s %.3f dB  (rel MSE %.4e)" % (k, 10 * np.log10(en / er), er / en))
    for role, d in tot_role.items():
        print("  %-5s " % role + "  ".join("%s %.3f" % (k, 10 * np.log10(v[1] / v[0])) for k, v in d.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
