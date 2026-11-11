# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
# Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
"""tools/ds4/dspark_ref.py - DeepSeek-V4.1's DSpark drafter in torch, straight from DeepSeek's checkpoint, against what
the engine's drafter computed (the engine's oracle for the drafter; ref.py / ref_gguf.py cover the trunk only).

    DS4_DSPARK_DUMP=d.bin maya-ds4 --pack <pack> ... --n 64        (the engine: its drafter's inputs and outputs)
    python tools/ds4/dspark_ref.py --dump d.bin --shards <dir with model-00044..46-of-00048> --pack <pack>
                                   [--rounds 4] [--head]

It follows DeepSeek's inference/model.py (MIT) - DSparkBlock / DSparkAttention / forward_spec: the taps of target
layers 37..39 (the streams' mean entering them) through main_proj / main_norm into every block's window ring (wkv,
kv_norm, rope, FP8 rounding), the block [y, noise x 4] at the positions after the ring's through the three blocks
(Hyper-Connections, window attention over the ring and the whole block, the 128-expert MoE with 3 picks), the final
collapse and, with --head, the trunk's head plus the markov bias (the drafts) - with every weight decoded here from
the checkpoint's own FP8 / FP4 / BF16 tensors (not from drafter_pack.py's dspark.bin), activations F32.  Per round
it prints how far the engine's hidden rows and confidences are from it, and with --head whether the drafts agree.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F

SHARDS = ("model-00044-of-00048.safetensors", "model-00045-of-00048.safetensors", "model-00046-of-00048.safetensors")


def e4m3_table() -> np.ndarray:
    t = np.zeros(256, np.float32)
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 15, c & 7
        v = np.nan if (e == 15 and m == 7) else (m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
        t[c] = -v if s else v
    return t


E4M3 = e4m3_table()
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


class Ckpt:
    """The drafter's shards: each tensor decoded to F32 on request (FP8 with its 32 x 32 E8M0 block scales; FP4 codes
    two to a byte, the first in the low nibble, an E8M0 scale per 32 along K; BF16; F32)."""

    def __init__(self, d: pathlib.Path):
        self.t = {}
        for name in SHARDS:
            raw = np.memmap(d / name, np.uint8, mode="r")
            n = struct.unpack("<Q", bytes(raw[:8]))[0]
            head = json.loads(bytes(raw[8:8 + n]))
            head.pop("__metadata__", None)
            for k, e in head.items():
                self.t[k] = (raw, 8 + n, e)

    def raw(self, k):
        raw, base, e = self.t[k]
        a, b = e["data_offsets"]
        return raw[base + a:base + b], e["dtype"], e["shape"]

    def get(self, k) -> torch.Tensor:
        w, dt, sh = self.raw(k)
        if dt == "F32":
            v = np.frombuffer(w, np.float32).reshape(sh)
        elif dt == "BF16":
            v = (np.frombuffer(w, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(sh)
        elif dt == "F8_E4M3":
            s, _, ssh = self.raw(k[:-len("weight")] + "scale")
            r, c = sh
            sc = np.exp2(np.asarray(s, np.float32).reshape(ssh) - 127.0)
            v = E4M3[np.asarray(w).reshape(r, c)] * np.repeat(np.repeat(sc, 32, 0), 32, 1)[:r, :c]
        elif dt == "I8":
            s, _, ssh = self.raw(k[:-len("weight")] + "scale")
            r, c2 = sh
            p = np.asarray(w).reshape(r, c2)
            codes = np.empty((r, c2 * 2), np.uint8)
            codes[:, 0::2] = p & 15
            codes[:, 1::2] = p >> 4
            sc = np.exp2(np.asarray(s, np.float32).reshape(ssh) - 127.0)
            v = E2M1[codes] * np.repeat(sc, 32, 1)
        else:
            raise SystemExit("%s: dtype %s" % (k, dt))
        return torch.from_numpy(np.ascontiguousarray(v, np.float32))


def fp8_round(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    sh = x.shape
    v = x.float().reshape(*sh[:-1], sh[-1] // block, block)
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    y = (v / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
    return y.reshape(sh)


def rms(x, w, eps):
    x = x.float()
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return y * w if w is not None else y


def freqs(dim, n, base):
    f = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    a = torch.outer(torch.arange(n, dtype=torch.float32), f)
    return torch.polar(torch.ones_like(a), a)


def rope(x, fc, inverse=False):
    c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2).contiguous())
    if inverse:
        fc = fc.conj()
    if x.ndim == 3:
        fc = fc[:, None, :]
    return torch.view_as_real(c * fc).flatten(-2)


def pack_conf(pack: pathlib.Path):
    cfg = {}
    for line in (pack / "ds4.txt").read_text().splitlines():
        k, _, v = line.partition(" ")
        cfg[k] = v
    files, tens = {}, {}
    for line in (pack / "tensors.txt").read_text().splitlines():
        p = line.split()
        if p[0] == "file":
            files[int(p[1])] = p[2]
        elif p[0] == "t":
            tens[p[1]] = (p[2], int(p[3]), int(p[4]), int(p[5]), [int(x) for x in p[6:]])
    return cfg, files, tens


def gguf_rows(pack, cfg, files, tens, name):
    dt, fi, off, nb, sh = tens[name]
    if dt != "BF16":
        raise SystemExit("%s: %s, the oracle reads a BF16 one" % (name, dt))
    path = pathlib.Path(cfg["model_dir"]) / files[fi]
    return np.memmap(path, np.uint16, mode="r", offset=off, shape=tuple(sh))


def bf16_rows(m, idx):
    return torch.from_numpy((np.asarray(m[idx]).astype(np.uint32) << 16).view(np.float32))


class Drafter:
    def __init__(self, ck: Ckpt, cfg: dict):
        self.ck, self.cfg = ck, cfg
        self.D = int(cfg["hidden_size"])
        self.hc = int(cfg["hc_mult"])
        self.eps = float(cfg["rms_norm_eps"])
        self.hc_eps = float(cfg["hc_eps"])
        self.sink_iters = int(cfg["hc_sinkhorn_iters"])
        self.H, self.HD, self.rd = int(cfg["num_attention_heads"]), int(cfg["head_dim"]), int(cfg["qk_rope_head_dim"])
        self.G = int(cfg["o_groups"])
        self.win = int(cfg["sliding_window"])
        self.k = int(cfg["dspark_num_experts_per_tok"])
        self.scale = float(cfg["routed_scaling_factor"])
        self.lim = float(cfg["swiglu_limit"])
        self.fc = freqs(self.rd, 1 << 16, float(cfg["rope_theta"]))
        self.nb = 3
        self.cache = {}
        self.ring = [dict() for _ in range(self.nb)]   # position -> main KV row, per block

    def W(self, k):
        if k not in self.cache:
            self.cache[k] = self.ck.get(k)
        return self.cache[k]

    def seed(self, p0, taps):   # taps [n, 3 * D]
        main_x = rms(taps @ self.W("mtp.0.main_proj.weight").T, self.W("mtp.0.main_norm.weight"), self.eps)
        n = taps.shape[0]
        fc = self.fc[p0:p0 + n]
        for b in range(self.nb):
            A = "mtp.%d.attn." % b
            kv = rms(main_x @ self.W(A + "wkv.weight").T, self.W(A + "kv_norm.weight"), self.eps)
            kv = torch.cat([kv[:, :-self.rd], rope(kv[:, -self.rd:], fc)], -1)
            kv = fp8_round(kv)
            for i in range(n):
                self.ring[b][p0 + i] = kv[i]

    def hc_mixes(self, x, P):
        hc = self.hc
        fn, scale, base = self.W(P + "_fn"), self.W(P + "_scale"), self.W(P + "_base")
        xf = x.flatten(1).float()
        r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        mixes = (xf @ fn.T) * r
        pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + self.hc_eps
        post = 2 * torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
        comb = (mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]).view(-1, hc, hc)
        eps = self.hc_eps
        comb = comb.softmax(-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(self.sink_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb

    def attention(self, b, x, p0):
        A = "mtp.%d.attn." % b
        T = x.shape[0]
        H, D, rd = self.H, self.HD, self.rd
        fc = self.fc[p0:p0 + T]
        qr = rms(x @ self.W(A + "wq_a.weight").T, self.W(A + "q_norm.weight"), self.eps)
        q = (qr @ self.W(A + "wq_b.weight").T).view(T, H, D)
        q = torch.cat([q[..., :-rd], rope(q[..., -rd:], fc)], -1)
        kv = rms(x @ self.W(A + "wkv.weight").T, self.W(A + "kv_norm.weight"), self.eps)
        kv = torch.cat([kv[:, :-rd], rope(kv[:, -rd:], fc)], -1)
        kv = fp8_round(kv)
        ring = [self.ring[b][p] for p in range(max(0, p0 - self.win), p0) if p in self.ring[b]]
        allkv = torch.cat([torch.stack(ring), kv], 0) if ring else kv
        sink = self.W(A + "attn_sink")
        s = torch.einsum("thd,kd->thk", q, allkv) * D ** -0.5
        m = torch.maximum(s.amax(-1), sink[None, :])
        pr = torch.exp(s - m.unsqueeze(-1))
        den = pr.sum(-1) + torch.exp(sink[None, :] - m)
        o = torch.einsum("thk,kd->thd", pr, allkv) / den.unsqueeze(-1)
        o = torch.cat([o[..., :-rd], rope(o[..., -rd:], fc, inverse=True)], -1)
        o = fp8_round(o.reshape(T, self.G, -1))
        wo_a = self.W(A + "wo_a.weight").view(self.G, -1, o.shape[-1])
        oa = torch.einsum("tgd,grd->tgr", o, wo_a)
        return oa.flatten(1) @ self.W(A + "wo_b.weight").T

    def moe(self, b, x):
        M = "mtp.%d.ffn." % b
        T = x.shape[0]
        scores = F.softplus(x.float() @ self.W(M + "gate.weight").T).sqrt()
        idx = (scores + self.W(M + "gate.bias")).topk(self.k, dim=-1)[1]
        wts = scores.gather(1, idx)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * self.scale
        lim = self.lim
        y = torch.zeros(T, self.D)
        for e in torch.unique(idx).tolist():
            P = M + "experts.%d." % e
            rows, slot = torch.where(idx == e)
            xs = x[rows]
            w1, w3 = self.ck.get(P + "w1.weight"), self.ck.get(P + "w3.weight")
            h = F.silu((xs @ w1.T).clamp(max=lim)) * (xs @ w3.T).clamp(-lim, lim)
            del w1, w3
            w2 = self.ck.get(P + "w2.weight")
            y.index_add_(0, rows, (h * wts[rows, slot, None]) @ w2.T)
            del w2
        S = M + "shared_experts."
        h = F.silu((x @ self.W(S + "w1.weight").T).clamp(max=lim)) * (x @ self.W(S + "w3.weight").T).clamp(-lim, lim)
        return y + h @ self.W(S + "w2.weight").T, idx

    def block_forward(self, emb, p0):   # emb [T, D] -> the collapsed hidden rows [T, D], each block's stream
        T = emb.shape[0]
        h = emb.unsqueeze(1).repeat(1, self.hc, 1)
        pre = torch.zeros(T, self.hc)
        pre[:, 0] = 1.0
        routes = []
        streams = []
        for b in range(self.nb):
            B = "mtp.%d." % b
            res = h
            a_pre, a_post, a_comb = self.hc_mixes(h, B + "hc_attn")
            x = rms((pre.unsqueeze(-1) * h).sum(1), self.W(B + "attn_norm.weight"), self.eps)
            y = self.attention(b, x, p0)
            h = a_post.unsqueeze(-1) * y.unsqueeze(1) + (a_comb.unsqueeze(-1) * res.unsqueeze(2)).sum(1)
            res = h
            f_pre, f_post, f_comb = self.hc_mixes(h, B + "hc_ffn")
            x = rms((a_pre.unsqueeze(-1) * h).sum(1), self.W(B + "ffn_norm.weight"), self.eps)
            y, idx = self.moe(b, x)
            routes.append(idx)
            h = f_post.unsqueeze(-1) * y.unsqueeze(1) + (f_comb.unsqueeze(-1) * res.unsqueeze(2)).sum(1)
            pre = f_pre
            streams.append(h)
        return (pre.unsqueeze(-1) * h).sum(1), routes, streams


def records(path, nb, hc, win, hd):
    """The engine's dump: 'S' seeds (first position, the taps), 'D' draft rounds (the block's position, y, the drafts,
    their confidences, the collapsed hidden rows, then per block the ring it read (BF16) and the stream it left)."""
    with open(path, "rb") as f:
        while True:
            hb = f.read(32)
            if len(hb) < 32:
                return
            kind, a, b, c = struct.unpack("<4q", hb)
            if kind == ord("S"):
                n, w = b, c
                taps = np.frombuffer(f.read(n * w * 4), np.float32).reshape(n, w)
                yield "S", a, taps
            elif kind == ord("D"):
                T = c
                D = int(DIM[0])
                toks = np.frombuffer(f.read(T * 4), np.int32)
                conf = np.frombuffer(f.read(T * 4), np.float32)
                x = np.frombuffer(f.read(T * 4 * D), np.float32).reshape(T, D)
                rings, streams = [], []
                for _ in range(nb):
                    r = np.frombuffer(f.read(win * hd * 2), np.uint16).reshape(win, hd)
                    rings.append((r.astype(np.uint32) << 16).view(np.float32))
                    streams.append(np.frombuffer(f.read(T * hc * D * 4), np.float32).reshape(T, hc, D))
                yield "D", (a, b, T, toks, conf, x, rings, streams), None
            else:
                raise SystemExit("dump: record %d" % kind)


DIM = [5120]


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--shards", required=True, help="the folder with DeepSeek's model-00044..46-of-00048")
    ap.add_argument("--pack", required=True, help="the maya-ds4 pack (ds4.txt, tensors.txt: the embedding, the head)")
    ap.add_argument("--rounds", type=int, default=4, help="draft rounds to check (from --skip)")
    ap.add_argument("--skip", type=int, default=0, help="draft rounds to pass over first")
    ap.add_argument("--head", action="store_true", help="also the drafts: the trunk's head + the markov bias")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    pack = pathlib.Path(a.pack)
    cfg, files, tens = pack_conf(pack)
    DIM[0] = int(cfg["hidden_size"])
    ck = Ckpt(pathlib.Path(a.shards))
    dr = Drafter(ck, cfg)
    emb = gguf_rows(pack, cfg, files, tens, "embed.weight")
    head = gguf_rows(pack, cfg, files, tens, "head.weight") if a.head else None
    noise = int(cfg["dspark_noise_token_id"])
    seen = done = 0
    worst_x = worst_c = 0.0
    agree = total = 0
    for kind, v, taps in records(a.dump, dr.nb, dr.hc, dr.win, dr.HD):
        if kind == "S":
            dr.seed(v, torch.from_numpy(taps.copy()))
            continue
        p0, y, T, toks, conf, xd, rings, streams = v
        seen += 1
        if seen <= a.skip:
            continue
        ex = torch.from_numpy(xd.copy())
        ids = [y] + [noise] * (T - 1)
        xh, routes, ref_streams = dr.block_forward(bf16_rows(emb, ids), p0)
        for b in range(dr.nb):   # the ring each block read (the positions the reference holds), the stream it left
            ps = [p for p in range(max(0, p0 - dr.win), p0) if p in dr.ring[b]]
            er = torch.from_numpy(rings[b][[p % dr.win for p in ps]].copy())
            rr = torch.stack([dr.ring[b][p] for p in ps])
            print("  block %d: ring rel err %.4f (%d rows), stream rel err %.4f, routes %s" % (
                b, rel(er, rr), len(ps), rel(torch.from_numpy(streams[b].copy()), ref_streams[b]),
                routes[b].tolist()), flush=True)
        prev = [y] + [int(t) for t in toks[:-1]]
        me = dr.W("mtp.2.markov_head.embed.weight")[prev]
        cw = dr.W("mtp.2.confidence_head.proj.weight")[0]
        cf = torch.sigmoid(torch.cat([xh, me], -1) @ cw)
        rerr = ((ex - xh).norm(dim=-1) / xh.norm(dim=-1)).tolist()
        cos = F.cosine_similarity(ex, xh, dim=-1).tolist()
        dc = (torch.from_numpy(conf.copy()) - cf).abs().tolist()
        worst_x = max(worst_x, max(rerr))
        worst_c = max(worst_c, max(dc))
        line = "round %d (pos %d, y %d): hidden rel err %s, cos %s; conf engine %s ref %s" % (
            seen, p0, y, " ".join("%.4f" % r for r in rerr), " ".join("%.5f" % c for c in cos),
            " ".join("%.3f" % c for c in conf.tolist()), " ".join("%.3f" % c for c in cf.tolist()))
        if head is not None:
            xn = rms(xh, dr.W("mtp.2.norm.weight"), dr.eps)
            lg = torch.zeros(T, head.shape[0])
            for c0 in range(0, head.shape[0], 16384):
                c1 = min(head.shape[0], c0 + 16384)
                lg[:, c0:c1] = xn @ bf16_rows(head, slice(c0, c1)).T
            mh = dr.W("mtp.2.markov_head.head.weight")
            ref_t = []
            p = y
            for i in range(T):   # greedy chain on the reference's own drafts
                row = lg[i] + mh @ dr.W("mtp.2.markov_head.embed.weight")[p]
                p = int(row.argmax())
                ref_t.append(p)
            same = sum(int(r == int(t)) for r, t in zip(ref_t, toks))
            agree += same
            total += T
            line += "; drafts engine %s ref %s" % (list(map(int, toks)), ref_t)
        print(line, flush=True)
        done += 1
        if done >= a.rounds:
            break
    print("checked %d rounds: worst hidden rel err %.4f, worst conf diff %.4f%s" % (
        done, worst_x, worst_c, (", drafts agree %d/%d" % (agree, total)) if head is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
