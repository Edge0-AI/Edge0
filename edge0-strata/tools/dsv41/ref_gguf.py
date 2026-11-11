# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
# Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
"""tools/ds4/ref_gguf.py - tools/ds4/ref.py's oracle over a GGUF (tools/ds4/pack_gguf.py's pack): the same forward,
each weight dequantised in torch from the GGUF's own blocks (BF16, Q8_0, MXFP4, MXFP8) - a check of maya-ds4's GGUF
path that shares none of its code.

    python tools/ds4/ref_gguf.py --pack <model>/maya-ds4 --ids 0,671,6102 [--dump out.npz] [--gen 8]

The token ids come from the server's tokenizer (maya-ds4 --serve, or the engine's own --ids runs); --dump writes
what tools/ds4/compare.py reads.
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import struct
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ref as R  # noqa: E402

BLOCK = {"Q8_0": 34, "MXFP4": 17, "MXFP8": 33}
E2M1 = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=torch.float32)


class Pack:
    """pack_gguf.py's tensors.txt over the GGUF's files (memory-mapped), and ds4.txt's geometry."""

    def __init__(self, d: pathlib.Path):
        self.kv = {}
        for line in (d / "ds4.txt").read_text().splitlines():
            k, _, v = line.partition(" ")
            self.kv[k] = v
        model = pathlib.Path(self.kv["model_dir"])
        self.mm, self.t = {}, {}
        for line in (d / "tensors.txt").read_text().splitlines():
            f = line.split()
            if f[0] == "file":
                self.mm[int(f[1])] = np.memmap(model / f[2], dtype=np.uint8, mode="r")
            elif f[0] == "t":
                self.t[f[1]] = (f[2], int(f[3]), int(f[4]), int(f[5]), tuple(int(x) for x in f[6:]))
        self._embed = None

    def has(self, name):
        return name in self.t

    def raw(self, name) -> np.ndarray:
        dt, fi, off, nb, _ = self.t[name]
        return self.mm[fi][off:off + nb]

    def num(self, k, d=0.0):
        return float(self.kv.get(k, d))

    def ints(self, k):
        return [int(x) for x in self.kv.get(k, "").split(",") if x]


def dequant(p: Pack, name: str, dev) -> torch.Tensor:
    """A tensor as F32 on dev, shaped as stored ([out][in] for a matrix)."""
    dt, _, _, _, shape = p.t[name]
    r = torch.from_numpy(np.array(p.raw(name))).to(dev)
    if dt == "F32":
        v = r.view(torch.float32)
    elif dt == "F16":
        v = r.view(torch.float16).float()
    elif dt == "BF16":
        v = r.view(torch.bfloat16).float()
    elif dt == "Q8_0":
        b = r.view(-1, 34)
        v = (b[:, :2].contiguous().view(torch.float16).float() * b[:, 2:].view(torch.int8).float()).reshape(-1)
    elif dt == "MXFP4":
        b = r.view(-1, 17)
        s = torch.ldexp(torch.ones(1, device=dev), b[:, :1].int() - 128)          # 2^(e - 128)
        q = b[:, 1:].long()
        lut = E2M1.to(dev)
        v = (torch.cat([lut[q & 15], lut[q >> 4]], 1) * s).reshape(-1)          # values 0..15, then 16..31
    elif dt == "MXFP8":
        b = r.view(-1, 33)
        s = torch.ldexp(torch.ones(1, device=dev), b[:, :1].int() - 127)
        v = (b[:, 1:].contiguous().view(torch.float8_e4m3fn).float() * s).reshape(-1)
    else:   # the low-bit expert types (IQ2_XXS .. Q4_K): gguf-py's numpy dequantisers (ggml's dequantize_row_*)
        v = torch.from_numpy(gq_values(np.array(p.raw(name)), dt, name)).to(dev)
    return v.reshape(shape)


def gq_values(raw: np.ndarray, dt: str, name: str) -> np.ndarray:
    """ggml blocks of type dt as F32 (flat) through gguf-py's dequantisers - the installed package, else a llama.cpp
    checkout's ($GGUF_PY, ~/llama.cpp/gguf-py)."""
    try:
        try:
            import gguf
        except ImportError:
            for d in (os.environ.get("GGUF_PY", ""), os.path.expanduser("~/llama.cpp/gguf-py")):
                if d and os.path.isdir(d):
                    sys.path.insert(0, d)
            import gguf
        from gguf.quants import dequantize as gq_dequant
        qt = gguf.GGMLQuantizationType[dt]
    except (ImportError, KeyError):
        raise SystemExit("%s: dtype %s (no gguf-py with a dequantiser for it: GGUF_PY=<llama.cpp>/gguf-py)" % (name, dt))
    return np.ascontiguousarray(gq_dequant(raw, qt), dtype=np.float32).reshape(-1)


class Hasher:
    """engram_hash.Hasher from the pack's engram.bin (the token map, multipliers, primes and offsets)."""

    def __init__(self, path: pathlib.Path):
        b = path.read_bytes()
        vocab, nl, ngram, heads, hd, pad = struct.unpack_from("<6i", b, 4)
        cols = (ngram - 1) * heads
        o = 28
        self.tm = np.frombuffer(b, np.int32, vocab, o).astype(np.int64)
        o += 4 * vocab
        self.mult = np.frombuffer(b, np.int64, nl * ngram, o).reshape(nl, ngram)
        o += 8 * nl * ngram
        self.primes = np.frombuffer(b, np.int64, nl * cols, o).reshape(nl, ngram - 1, heads)
        o += 8 * nl * cols
        self.off = np.frombuffer(b, np.int64, nl * cols, o).reshape(nl, cols)
        self.ngram, self.pad, self.hd = ngram, pad, hd

    def ids(self, tokens) -> np.ndarray:   # engram_hash.Hasher.ids
        c = self.tm[np.asarray(tokens, dtype=np.int64)]
        T = len(c)
        pos = np.arange(T)
        toks, blocked = [], np.zeros(T, dtype=bool)
        for s in range(self.ngram):
            src = c[np.clip(pos - s, 0, None)]
            blocked = blocked | (pos < s)
            toks.append(np.where(blocked, self.pad, src))
        toks = np.stack(toks, axis=-1)
        prod = toks[:, None, :] * self.mult[None]
        rolling, out = prod[..., 0], []
        for i in range(1, self.ngram):
            rolling = np.bitwise_xor(rolling, prod[..., i])
            out.append(rolling[..., None] % self.primes[None, :, i - 1])
        return np.concatenate(out, axis=-1) + self.off[None]


def q6k_rows(rows: np.ndarray, name: str) -> np.ndarray:
    """Engram rows of Q6_K superblocks as F32 (ggml's dequantize_row_q6_K, through gguf-py)."""
    return gq_values(rows.reshape(-1), "Q6_K", name)


class RefGguf(R.Ref):
    def __init__(self, pack: pathlib.Path, dev: str = "cuda:0", max_seq: int = 8192):
        self.dir = pack
        self.st = p = Pack(pack)
        g = p.num
        t = {k: int(g(k)) for k in ("hidden_size", "hc_mult", "num_hidden_layers", "candidate_source_layer_id",
                                    "qk_rope_head_dim", "num_experts_per_tok", "sliding_window", "o_groups",
                                    "num_attention_heads", "index_topk", "index_n_heads", "index_head_dim", "head_dim",
                                    "candidate_topk_blocks", "candidate_block_size")}
        t.update(hc_eps=g("hc_eps"), hc_sinkhorn_iters=int(g("hc_sinkhorn_iters")),
                 rms_norm_eps=g("rms_norm_eps"), routed_scaling_factor=g("routed_scaling_factor"),
                 swiglu_limit=g("swiglu_limit"), rope_theta=g("rope_theta"), compress_rope_theta=g("compress_rope_theta"),
                 compress_ratios=p.ints("compress_ratios"), kv_source_layer_ids=p.ints("kv_source_layer_ids"),
                 index_source_layer_ids=p.ints("index_source_layer_ids"),
                 rope_scaling={"factor": g("rope_scaling.factor"), "beta_fast": g("rope_scaling.beta_fast"),
                               "beta_slow": g("rope_scaling.beta_slow"),
                               "original_max_position_embeddings": int(g("rope_scaling.original_max_position_embeddings"))})
        self.cfg = {"text_config": t}
        self.t = t
        self.dev = dev
        self.dim, self.hc = t["hidden_size"], t["hc_mult"]
        self.eps = t["rms_norm_eps"]
        self.nl = t["num_hidden_layers"]
        self.ratios = t["compress_ratios"]
        self.kv_src = set(t["kv_source_layer_ids"])
        self.ix_src = set(t["index_source_layer_ids"])
        self.cand_src = t["candidate_source_layer_id"]
        rs, rd = t["rope_scaling"], t["qk_rope_head_dim"]
        self.fc_window = R.freqs_cis(rd, max_seq, 0, t["rope_theta"], rs["factor"], rs["beta_fast"], rs["beta_slow"], dev)
        self.fc_comp = R.freqs_cis(rd, max_seq, rs["original_max_position_embeddings"], t["compress_rope_theta"],
                                   rs["factor"], rs["beta_fast"], rs["beta_slow"], dev)
        self.hasher = Hasher(pack / "engram.bin")
        self.lay = SimpleNamespace(layer_ids=tuple(p.ints("engram_layer_ids")), head_dim=self.hasher.hd)
        self.timing = {}
        self._embed = None

    def W(self, prefix: str) -> torch.Tensor:
        return dequant(self.st, prefix + ".weight", self.dev)

    def V(self, name: str) -> torch.Tensor:
        return dequant(self.st, name, self.dev).float()

    def engram(self, li, x, hash_ids):
        P = "layers.%d.engram." % li
        T, cols = hash_ids.shape
        hd = self.lay.head_dim
        dt = self.st.t[P + "embed.weight"][0]
        rb = hd // 256 * 210 if dt == "Q6_K" else hd // 32 * 33        # (a row's bytes: Q6_K or MXFP8 blocks)
        tab = self.st.raw(P + "embed.weight").reshape(-1, rb)
        flat = hash_ids.reshape(-1)
        order = np.argsort(flat)
        rows = np.empty((flat.size, rb), np.uint8)
        rows[order] = tab[flat[order]]
        if dt == "Q6_K":
            e = torch.from_numpy(q6k_rows(rows, P)).to(self.dev).view(T, cols * hd)
        else:
            b = torch.from_numpy(rows).to(self.dev).view(-1, 33)
            s = torch.ldexp(torch.ones(1, device=self.dev), b[:, :1].int() - 127)
            e = (b[:, 1:].contiguous().view(torch.float8_e4m3fn).float() * s).view(T, cols * hd)
        e = e.to(torch.bfloat16).float()
        kv = e @ self.W(P + "wkv").T
        d, hc = self.dim, self.hc
        key, value = kv[:, :hc * d].view(T, hc, d), kv[:, hc * d:]
        w = self.V(P + "gate.weight")                                     # q_weight * k_weight (folded)
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * w * key).sum(-1) * rstd * d ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return h + gate.unsqueeze(-1) * value.unsqueeze(1)

    def moe(self, li, x):
        t = self.t
        M = "layers.%d.ffn." % li
        T = x.shape[0]
        gw = self.V(M + "gate.weight")
        scores = F.softplus(x.float() @ gw.T).sqrt()
        idx = (scores + self.V(M + "gate.bias")).topk(t["num_experts_per_tok"], dim=-1)[1]
        wts = scores.gather(1, idx)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * t["routed_scaling_factor"]
        lim = t["swiglu_limit"]
        y = torch.zeros(T, self.dim, device=self.dev)
        for e in torch.unique(idx).tolist():
            P = M + "experts.%d." % e
            rows, slot = torch.where(idx == e)
            xs = x[rows]
            h = F.silu((xs @ self.W(P + "w1").T).clamp(max=lim)) * (xs @ self.W(P + "w3").T).clamp(-lim, lim)
            y.index_add_(0, rows, (h * wts[rows, slot, None]) @ self.W(P + "w2").T)
        S = M + "shared_experts."
        h = F.silu((x @ self.W(S + "w1").T).clamp(max=lim)) * (x @ self.W(S + "w3").T).clamp(-lim, lim)
        y += h @ self.W(S + "w2").T
        self.last_route = (idx, wts)
        return y

    @torch.inference_mode()
    def forward(self, ids, dump=None):
        # ref.py's forward reads the embedding rows as BF16: a Q8_0 table is dequantised for the rows asked
        dt = self.st.t["embed.weight"][0]
        if dt != "BF16":
            full = self.st.raw("embed.weight").reshape(-1, self.dim // 32 * 34)
            sel = torch.from_numpy(np.array(full[np.asarray(ids)])).to(self.dev).view(-1, 34)
            v = (sel[:, :2].contiguous().view(torch.float16).float() * sel[:, 2:].view(torch.int8).float())
            rows = v.reshape(len(ids), self.dim).to(torch.bfloat16).view(torch.uint16).cpu().numpy()
            raw0 = self.st.raw
            emb = np.zeros((max(ids) + 1, self.dim), np.uint16)
            emb[np.asarray(ids)] = rows
            self.st.raw = lambda n, e=emb, r=raw0: e.view(np.uint8) if n == "embed.weight" else r(n)
            try:
                return super().forward(ids, dump)
            finally:
                self.st.raw = raw0
        return super().forward(ids, dump)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="pack_gguf.py's pack (<model>/maya-ds4)")
    ap.add_argument("--ids", required=True, help="comma-separated token ids")
    ap.add_argument("--gen", type=int, default=0, help="greedy tokens to append (a whole forward each)")
    ap.add_argument("--dump", help="write the per-layer residual streams, the logits and the ids (.npz)")
    ap.add_argument("--no-hs", action="store_true", help="--dump without the per-layer residual streams")
    ap.add_argument("--dev", default="cuda:0")
    a = ap.parse_args()
    ids = [int(x) for x in a.ids.split(",")]
    r = RefGguf(pathlib.Path(a.pack).expanduser(), a.dev)
    print("tokens:", len(ids), ids[:32], flush=True)
    dump = [] if a.dump and not a.no_hs else None
    t0 = time.time()
    logits = r.forward(ids, dump)
    print("forward %.1fs" % (time.time() - t0), flush=True)
    lp = torch.log_softmax(logits.float(), -1)
    if len(ids) > 1:
        tgt = torch.tensor(ids[1:], device=lp.device)
        nll = -lp[:-1].gather(1, tgt[:, None]).squeeze(1)
        print("ppl over %d tokens: %.4f" % (len(ids) - 1, math.exp(nll.mean().item())))
    top = lp[-1].topk(8)
    print("next:", list(zip(top.indices.tolist(), [round(v, 3) for v in top.values.tolist()])))
    if a.dump:
        np.savez(a.dump, ids=np.array(ids), logits=logits.cpu().numpy(), hs=np.stack(dump) if dump else np.zeros(0))
    out = list(ids)
    for _ in range(a.gen):
        nxt = int(lp[-1].argmax())
        out.append(nxt)
        print("gen ids:", out[len(ids):], flush=True)
        if nxt == 1:
            break
        lp = torch.log_softmax(r.forward(out).float(), -1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
