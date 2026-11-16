"""pilot.py - gate-3 block-quality pilot: a few experts from several layers of a source checkpoint, encoded at TQK4X / TQK5X /
TQK6 / TQK7 with and without the input-only block-128 Hadamard, and measured.

Sources:  --gguf <shard1 or shard dir files...>  (UD-Q8_K_XL: MXFP4 / Q8_0 / BF16 / F16 tensors, decoded exactly: the decoded
          source is the teacher, as the plan says)   or   --hf Qwen/...  (range requests; test of the pipeline)

Metrics per (layer, role, expert, rate, transform), over a random row subset of the projection (--rows):
  mse     relative weight MSE ||W - W^||^2 / ||W||^2 in the original basis (the Hadamard is orthonormal: the same in either basis)
  act_iso ||(W - W^) X||^2 / ||W X||^2 with X i.i.d. N(0,1) columns - PLACEHOLDER; in expectation equal to mse
  act_syn ||(W - W^) X||^2 / ||W X||^2 with X whose input channels have heavy-tailed scales (a few outlier channels,
          log-normal rest) - SYNTHETIC PLACEHOLDER for real calibration activations, labelled as such in every output
Real activation-weighted numbers need calibration activations from the reference runtime (not done here).
"""
from __future__ import annotations

import argparse
import json
import mmap
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import gguf_reader  # noqa: E402
import tqenc  # noqa: E402

MXFP4_LUT = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)
RATES = {"K4": tqenc.TQK4X, "K5": tqenc.TQK5X, "K6": tqenc.TQK6, "K7": tqenc.TQK7, "K4t": tqenc.TQK4T, "K5t": tqenc.TQK5T}   # K4 / K5: computed codebook (v2); K4t / K5t: point table (v1)
CFGS = {
    "fast": dict(iters=0, ncand=1, scale_tries=1),
    "default": dict(iters=1, ncand=1, scale_tries=1),
    "rot3": dict(iters=3, ncand=3, scale_tries=1),
}


def decode_mxfp4(raw: np.ndarray, n_in: int) -> np.ndarray:
    """raw: [rows * n_in/32 * 17] bytes -> float32 [rows, n_in].  ggml block_mxfp4: e8m0 scale, 16 bytes, low nibble =
    element j, high nibble = element j + 16, value = kvalues[nibble] * 2^(e - 128) (the e8m0 half scale)."""
    b = raw.reshape(-1, 17)
    e = b[:, 0].astype(np.int32)
    d = np.where(e == 0, np.float32(2.0 ** -127), np.ldexp(np.float32(1.0), e - 128)).astype(np.float32)
    q = b[:, 1:]
    vals = np.concatenate([MXFP4_LUT[q & 15], MXFP4_LUT[q >> 4]], axis=1) * d[:, None]
    return vals.reshape(-1, n_in)


def decode_q8_0(raw: np.ndarray, n_in: int) -> np.ndarray:
    b = raw.reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    q = b[:, 2:].copy().view(np.int8).astype(np.float32)
    return (q * d).reshape(-1, n_in)


class Source:
    """Yields float32 [rows, n_in] row subsets of one expert's projection.  Local GGUF files (mmap) or, with
    remote=(repo, rev, [shard paths]), the same files read by HTTP range requests (headers and the few experts needed)."""

    def __init__(self, gguf_paths=None, remote=None, tmpdir="/tmp"):
        self.tensors = {}
        if gguf_paths:
            for p in gguf_paths:
                g = gguf_reader.GGUFFile(p)
                fh = open(p, "rb")
                mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
                for t in g.tensors:
                    self.tensors[t.name] = (g, (lambda off, n, mm=mm: np.frombuffer(mm, dtype=np.uint8, count=n, offset=off)), t)
        if remote:
            import urllib.request
            repo, rev, shards = remote
            for sh in shards:
                url = "https://huggingface.co/%s/resolve/%s/%s" % (repo, rev, sh)

                def rd(off, n, url=url):
                    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (off, off + n - 1)})
                    with urllib.request.urlopen(req, timeout=300) as r:
                        return np.frombuffer(r.read(), dtype=np.uint8)
                hdr = rd(0, 6 << 20)
                tp = pathlib.Path(tmpdir) / ("hdr_%s" % pathlib.Path(sh).name)
                tp.write_bytes(hdr.tobytes())
                g = gguf_reader.GGUFFile(tp)
                for t in g.tensors:
                    self.tensors[t.name] = (g, rd, t)

    def names(self, needle):
        return sorted(n for n in self.tensors if needle in n)

    def expert_rows(self, name: str, expert: int, rows: np.ndarray) -> np.ndarray:
        g, rd, ti = self.tensors[name]
        n_in, n_out = ti.shape[0], ti.shape[1]
        if ti.type_name == "MXFP4":
            rb = n_in // 32 * 17
        elif ti.type_name == "Q8_0":
            rb = n_in // 32 * 34
        elif ti.type_name in ("BF16", "F16"):
            rb = n_in * 2
        elif ti.type_name == "F32":
            rb = n_in * 4
        else:
            raise ValueError("%s: tensor type %s is not readable by this pilot" % (name, ti.type_name))
        base = g.data_start + ti.offset + expert * n_out * rb
        whole = rd(base, n_out * rb)     # one expert's projection (6.3 MB for MXFP4 2304 x 5120), then the row subset
        out = np.empty((len(rows), n_in), dtype=np.float32)
        for k, r in enumerate(rows):
            raw = whole[int(r) * rb:(int(r) + 1) * rb]
            if ti.type_name == "MXFP4":
                out[k] = decode_mxfp4(raw, n_in)[0]
            elif ti.type_name == "Q8_0":
                out[k] = decode_q8_0(raw, n_in)[0]
            elif ti.type_name == "BF16":
                out[k] = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif ti.type_name == "F16":
                out[k] = raw.view(np.float16).astype(np.float32)
            else:
                out[k] = raw.view(np.float32)
        return out


def synth_x(n_in: int, T: int, seed: int) -> np.ndarray:
    """SYNTHETIC activations: channel scale exp(N(0, 0.5)), 1% of the channels x (10..40)."""
    r = np.random.default_rng(seed)
    s = np.exp(0.5 * r.standard_normal(n_in)).astype(np.float32)
    out = r.choice(n_in, size=max(1, n_in // 100), replace=False)
    s[out] *= r.uniform(10, 40, size=out.size).astype(np.float32)
    return (r.standard_normal((n_in, T)).astype(np.float32) * s[:, None])


def act_err(W, What, X):
    D = (W - What).astype(np.float32) @ X
    Y = W.astype(np.float32) @ X
    return float(np.sum(D.astype(np.float64) ** 2) / max(np.sum(Y.astype(np.float64) ** 2), 1e-300))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", nargs="+", default=None)
    ap.add_argument("--remote", default=None, help="REPO@REV: read the shards named in --shards by HTTP range requests")
    ap.add_argument("--shards", nargs="+", default=[])
    ap.add_argument("--layers", default="1,7,14,21,30,38")
    ap.add_argument("--experts", default="0,97,200,383")
    ap.add_argument("--roles", default="gate,up,down")
    ap.add_argument("--rates", default="K4,K5,K6,K7")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=256, help="columns of the placeholder X")
    ap.add_argument("--cfg", default="default")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True, help="jsonl results")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pattern", default="blk.{L}.ffn_{R}_exps.weight")
    ap.add_argument("--gpu", action="store_true", help="encode on the GPU (run under the box locks, HIP_VISIBLE_DEVICES)")
    ap.add_argument("--act-diag-dir", default=None,
                    help="directory with L<layer>_<role>.npy = E[x^2] per input channel of that projection (calibration "
                         "second moments, diagonal).  Adds act_diag (the exact diagonal-Hessian error) and an importance-weighted "
                         "encode (h = E[x^2]; in the Hadamard basis h = the block average of it)")
    a = ap.parse_args()

    if a.remote:
        repo, rev = a.remote.split("@")
        src = Source(remote=(repo, rev, a.shards))
    else:
        src = Source(a.gguf)
    rng = np.random.default_rng(a.seed)
    cfg = CFGS[a.cfg]
    sign_cache, x_cache = {}, {}
    out = open(a.out, "a")
    for layer in [int(x) for x in a.layers.split(",")]:
        for role in a.roles.split(","):
            name = a.pattern.format(L=layer, R=role)
            if name not in src.tensors:
                print("no tensor", name, "(have e.g. %s)" % src.names("exps")[:3])
                continue
            g, rd, ti = src.tensors[name]
            n_in, n_out, n_exp = ti.shape[0], ti.shape[1], ti.shape[2]
            for e in [int(x) for x in a.experts.split(",")]:
                if e >= n_exp:
                    continue
                rows = np.sort(rng.choice(n_out, size=min(a.rows, n_out), replace=False))
                W = src.expert_rows(name, e, rows)
                en = float(np.sum(W.astype(np.float64) ** 2))
                if n_in not in sign_cache:
                    sign_cache[n_in] = tqenc.make_signs(n_in, 1000 + n_in)
                    x_cache[n_in] = (np.random.default_rng(7).standard_normal((n_in, a.tokens)).astype(np.float32),
                                     synth_x(n_in, a.tokens, 9))
                Xi, Xs = x_cache[n_in]
                diag = None
                if a.act_diag_dir:
                    dp = pathlib.Path(a.act_diag_dir) / ("L%d_%s.npy" % (layer, role))
                    if dp.exists():
                        diag = np.load(dp).astype(np.float64)
                        assert diag.shape == (n_in,), (dp, diag.shape)
                for had in (False, True):
                    sg = sign_cache[n_in] if had else None
                    Wr = tqenc.rotate(W, sg) if had else W
                    for rname in a.rates.split(","):
                        t = RATES[rname]
                        t0 = time.time()
                        packed, err, tot, st = tqenc.encode(Wr, t, threads=a.threads, gpu=a.gpu, **cfg)
                        sec = time.time() - t0
                        dec = tqenc.decode(packed, t).reshape(Wr.shape)
                        What = tqenc.unrotate(dec, sg) if had else dec
                        rec = dict(layer=layer, role=role, expert=e, rate=rname, had=had, n_in=n_in, rows=len(rows),
                                   mse=float(np.sum((W.astype(np.float64) - What) ** 2) / en),
                                   act_iso=act_err(W, What, Xi), act_syn=act_err(W, What, Xs),
                                   bpw=tqenc.bpw(t), seconds=sec, zero_blocks=st["zero_blocks"], cfg=a.cfg,
                                   x_kind="placeholder: iso gaussian / synthetic outlier channels")
                        if diag is not None:
                            hv = np.maximum(diag, 1e-30).astype(np.float32)
                            rec["act_diag"] = float(np.sum(((W - What).astype(np.float64) ** 2) * hv) / np.sum((W.astype(np.float64) ** 2) * hv))
                            hb = hv if not had else np.repeat(hv.reshape(-1, 128).mean(axis=1), 128)
                            hw = np.broadcast_to(hb, Wr.shape).astype(np.float32)
                            packed2, _, _, _ = tqenc.encode(Wr, t, h=hw, threads=a.threads, gpu=a.gpu, **cfg)
                            dec2 = tqenc.decode(packed2, t).reshape(Wr.shape)
                            What2 = tqenc.unrotate(dec2, sg) if had else dec2
                            rec["act_diag_weighted_encode"] = float(np.sum(((W - What2).astype(np.float64) ** 2) * hv) / np.sum((W.astype(np.float64) ** 2) * hv))
                            rec["mse_weighted_encode"] = float(np.sum((W.astype(np.float64) - What2) ** 2) / en)
                        out.write(json.dumps(rec) + "\n")
                        out.flush()
                        print("L%-3d %-4s e%-3d %s %-5s mse %.4f (%.2f dB)  act_iso %.4f  act_syn %.4f  %.1fs" % (
                            layer, role, e, rname, "had" if had else "plain", rec["mse"], -10 * np.log10(rec["mse"]),
                            rec["act_iso"], rec["act_syn"], sec), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
