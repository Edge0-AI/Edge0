# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
# Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
"""tools/ds4/ref.py - a whole-sequence DeepSeek-V4.1 forward in torch over an EXL3 checkpoint (the engine's oracle).

    python tools/ds4/ref.py --model ~/models/DSV4 --text "The capital of France is" [--dump out.npz] [--gen 8]

It follows DeepSeek's inference/model.py (MIT) at start_pos 0 - the prompt path, every position at once - with each
layer's weights decoded from EXL3 when the layer runs (tools/ds4/exl3_torch.py) and freed after it, so one GPU of
24 GB carries the 211 GB model: Hyper-Connections with the lagged pre-mix, sliding-window plus compressed sparse
attention with the two-level indexer and its candidate blocks, the cross-layer KV / indexer sharing, Engram, the
sqrt-softplus router, and the FP8 / FP4 activation rounding of the reference kernels (simulated, as their `inplace`
paths do).  Activations are F32 throughout (the reference keeps BF16 between ops).

--dump writes every layer's residual stream ([T, hc, dim] F32 after each block), the logits and the token ids, for
the engine's parity checks.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engram_hash as EH  # noqa: E402
import exl3_torch as ET  # noqa: E402
from st import SafeTensors  # noqa: E402


# ------------------------------------------------------------------ the reference kernels' roundings
def fp8_round(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """act_quant(..., scale_fmt='ue8m0', inplace=True): per `block` values a power-of-two scale, E4M3 values."""
    sh = x.shape
    v = x.float().reshape(*sh[:-1], sh[-1] // block, block)
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    y = (v / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
    return y.reshape(sh)


_FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _e2m1(v: torch.Tensor) -> torch.Tensor:
    """Round to the E2M1 grid, nearest, ties to the even mantissa (values already in [-6, 6])."""
    a = v.abs()
    g = _FP4.to(v.device)
    d = (a.unsqueeze(-1) - g).abs()
    best = d.argmin(-1)
    # ties: argmin takes the lower index; the even mantissa is the lower one at 0.25, 1.25, 2.5, 5 and the upper one
    # at 0.75, 1.75, 3.5 (grid indices: mantissa bit = index & 1 for index >= 1, 0 at index 0)
    lo = best
    hi = (best + 1).clamp(max=7)
    tie = (d.gather(-1, lo.unsqueeze(-1)).squeeze(-1) == d.gather(-1, hi.unsqueeze(-1)).squeeze(-1)) & (hi != lo)
    take_hi = tie & ((lo & 1) == 1)
    idx = torch.where(take_hi, hi, lo)
    return torch.copysign(g[idx], v)


def fp4_round(x: torch.Tensor, block: int, e4m3_scale: bool) -> torch.Tensor:
    """fp4_act_quant(..., inplace=True): E2M1 values with an E8M0 (power of two) or an E4M3 scale per block."""
    sh = x.shape
    v = x.float().reshape(*sh[:-1], sh[-1] // block, block)
    amax = v.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        amax = amax.clamp_min(6 * 2.0 ** -9)
        s = (amax / 6.0).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2.0 ** -126)
        s = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))
    y = _e2m1((v / s).clamp(-6.0, 6.0)) * s
    return y.reshape(sh)


def rms(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    x = x.float()
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return y * w if w is not None else y


def freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, dev) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corr(rot):
            return dim * math.log(original_seq_len / (rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corr(beta_fast)), 0)
        high = min(math.ceil(corr(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    f = torch.outer(torch.arange(seqlen, dtype=torch.float32), freqs)
    return torch.polar(torch.ones_like(f), f).to(dev)


def rope(x: torch.Tensor, fc: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Adjacent pairs as complex numbers.  x [T, d] or [T, h, d]; fc [T, d/2]."""
    c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2).contiguous())
    if inverse:
        fc = fc.conj()
    if x.ndim == 3:
        fc = fc[:, None, :]
    return torch.view_as_real(c * fc).flatten(-2)


# ------------------------------------------------------------------ the model
class Ref:
    def __init__(self, model: pathlib.Path, dev: str = "cuda:0", max_seq: int = 8192):
        self.dir = model
        self.cfg = json.loads((model / "config.json").read_text())
        self.t = t = self.cfg["text_config"]
        self.st = SafeTensors(model)
        self.dev = dev
        self.dim, self.hc = t["hidden_size"], t["hc_mult"]
        self.eps = t["rms_norm_eps"]
        self.nl = t["num_hidden_layers"]
        self.ratios = t["compress_ratios"]
        self.kv_src = set(t["kv_source_layer_ids"])
        self.ix_src = set(t["index_source_layer_ids"])
        self.cand_src = t["candidate_source_layer_id"]
        rs = t["rope_scaling"]
        rd = t["qk_rope_head_dim"]
        self.fc_window = freqs_cis(rd, max_seq, 0, t["rope_theta"], rs["factor"], rs["beta_fast"], rs["beta_slow"], dev)
        self.fc_comp = freqs_cis(rd, max_seq, rs["original_max_position_embeddings"], t["compress_rope_theta"],
                                 rs["factor"], rs["beta_fast"], rs["beta_slow"], dev)
        self.lay = EH.Layout.from_config(t)
        tm, n = EH.compressed_token_map(model / "tokenizer.json")
        if n != t["engram_compressed_vocab_size"]:
            raise SystemExit("engram: compressed vocab %d, the config says %d" % (n, t["engram_compressed_vocab_size"]))
        self.hasher = EH.Hasher(self.lay, tm, EH.multipliers(self.lay, n), tm[t["engram_pad_token_id"]])
        self.timing: dict[str, float] = {}

    # weights
    def W(self, prefix: str) -> torch.Tensor:
        return ET.weight(self.st, prefix, self.dev)

    def V(self, name: str) -> torch.Tensor:
        return torch.from_numpy(np.array(self.st.f32(name))).to(self.dev)

    # pieces
    def hc_mixes(self, x, fn, scale, base):
        hc = self.hc
        xf = x.flatten(1).float()
        r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        mixes = (xf @ fn.T) * r                                           # [T, (2 + hc) hc]
        pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + self.t["hc_eps"]
        post = 2 * torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
        comb = (mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]).view(-1, hc, hc)
        eps = self.t["hc_eps"]
        comb = comb.softmax(-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(self.t["hc_sinkhorn_iters"] - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb

    @staticmethod
    def hc_pre(x, mix):                                                     # [T, hc, d], [T, hc] -> [T, d]
        return (mix.unsqueeze(-1) * x).sum(1)

    @staticmethod
    def hc_post(y, res, post, comb):                                        # y [T, d], res [T, hc, d]
        return post.unsqueeze(-1) * y.unsqueeze(1) + (comb.unsqueeze(-1) * res.unsqueeze(2)).sum(1)

    def engram(self, li: int, x: torch.Tensor, hash_ids: np.ndarray) -> torch.Tensor:
        """x [T, hc, d]; hash_ids [T, cols] absolute rows."""
        P = "layers.%d.engram." % li
        T, cols = hash_ids.shape
        flat = hash_ids.reshape(-1)
        wt = self.st.raw(P + "embed.weight").view(np.uint8).reshape(-1, self.lay.head_dim)
        sc = self.st.raw(P + "embed.scale").view(np.uint8).reshape(-1, self.lay.head_dim // 32)
        order = np.argsort(flat)                                          # sequential reads
        rows = np.empty((flat.size, self.lay.head_dim), np.uint8)
        srow = np.empty((flat.size, self.lay.head_dim // 32), np.uint8)
        rows[order] = wt[flat[order]]
        srow[order] = sc[flat[order]]
        v = torch.from_numpy(rows).to(self.dev).view(torch.float8_e4m3fn).float()
        s = torch.from_numpy(srow).to(self.dev).view(torch.float8_e8m0fnu).float()
        e = (v.view(-1, self.lay.head_dim // 32, 32) * s.unsqueeze(-1)).view(T, cols * self.lay.head_dim)
        e = e.to(torch.bfloat16).float()                                  # ParallelEngramEmbedding returns BF16
        kv = e @ self.W(P + "wkv").T                                      # [T, (hc + 1) d]
        d, hc = self.dim, self.hc
        key, value = kv[:, :hc * d].view(T, hc, d), kv[:, hc * d:]
        w = self.V(P + "q_weight") * self.V(P + "k_weight")             # [hc, d]
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * w * key).sum(-1) * rstd * d ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return h + gate.unsqueeze(-1) * value.unsqueeze(1)

    def sparse_attn(self, q, kv, sink, idx, scale):
        """q [T, H, D], kv [N, D], sink [H], idx [T, K] int (-1 none) -> [T, H, D]."""
        T, H, D = q.shape
        out = torch.empty_like(q)
        for a in range(0, T, 256):
            b = min(T, a + 256)
            ii = idx[a:b]
            valid = ii >= 0
            g = kv[ii.clamp_min(0)]                                       # [t, K, D]
            s = torch.einsum("thd,tkd->thk", q[a:b], g) * scale
            s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))
            m = torch.maximum(s.amax(-1), sink[None, :])                  # [t, H]
            m = torch.maximum(m, torch.full_like(m, -1e30))
            p = torch.exp(s - m.unsqueeze(-1))
            den = p.sum(-1) + torch.exp(sink[None, :] - m)
            out[a:b] = torch.einsum("thk,tkd->thd", p, g) / den.unsqueeze(-1)
        return out

    @staticmethod
    def candidates(logits, compress_lens, topk_blocks, block):
        width = logits.size(-1)
        scores = F.pad(logits, (0, -width % block), value=float("-inf"))
        scores = scores.unflatten(-1, (-1, block)).amax(-1)
        nb = scores.size(-1)
        last = (compress_lens - 1) // block
        scores = scores.masked_fill(torch.arange(nb, device=logits.device) == last, float("inf"))
        top = scores.topk(min(topk_blocks, nb), dim=-1)
        keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
        return keep.repeat_interleave(block, dim=-1)[..., :width]

    def attention(self, li: int, x: torch.Tensor, S: dict) -> torch.Tensor:
        t = self.t
        A = ("layers.%d.attn." % li) if li < self.nl else ("mtp.%d.attn." % (li - self.nl))
        T = x.shape[0]
        H, D, rd = t["num_attention_heads"], t["head_dim"], t["qk_rope_head_dim"]
        ratio = self.ratios[li]
        fc = (self.fc_comp if ratio else self.fc_window)[:T]
        qr = rms(x @ self.W(A + "wq_a").T, self.V(A + "q_norm.weight"), self.eps)
        q = (qr @ self.W(A + "wq_b").T).view(T, H, D)
        q = torch.cat([q[..., :-rd], rope(q[..., -rd:], fc)], -1)
        kv = rms(x @ self.W(A + "wkv").T, self.V(A + "kv_norm.weight"), self.eps)
        kv = torch.cat([kv[:, :-rd], rope(kv[:, -rd:], fc)], -1)
        kv = fp8_round(kv)
        win = t["sliding_window"]
        end = torch.arange(T, device=self.dev).unsqueeze(1)
        widx = (end - win + 1).clamp(0) + torch.arange(min(T, win), device=self.dev)
        widx = torch.where(widx > end, -1, widx)
        all_kv, all_idx = kv, widx
        if ratio:
            is_src = li < self.nl and li in self.kv_src
            latent = None
            if is_src:
                C = A + "compressor."
                if ratio == 1:
                    latent = rms(x @ self.W(C + "wkv").T, self.V(C + "norm.weight"), self.eps)
                elif T >= ratio:
                    cut = T - T % ratio
                    kvc = x[:cut] @ self.W(C + "wkv").T
                    sc = x[:cut] @ self.W(C + "wgate").T
                    kvc, sc = kvc.view(-1, ratio, D), sc.view(-1, ratio, D)
                    latent = rms((kvc * sc.softmax(1)).sum(1), self.V(C + "norm.weight"), self.eps)
            # the indexer (index sources) before the latent is rotated
            if li < self.nl and li in self.ix_src:
                S["topk"] = self.indexer(li, A, x, qr, latent, ratio, T, fc, S)
            if latent is not None:
                lat = torch.cat([latent[:, :-rd], rope(latent[:, -rd:], fc[:T - T % ratio:ratio])], -1)
                S["ckv"] = fp4_round(lat, 16, True)
            ckv = S["ckv"][:T // ratio] if S.get("ckv") is not None else kv.new_zeros(0, D)
            all_kv = torch.cat([kv, ckv], 0)
            all_idx = torch.cat([widx, S["topk"]], -1)
        o = self.sparse_attn(q, all_kv, self.V(A + "attn_sink"), all_idx, D ** -0.5)
        o = torch.cat([o[..., :-rd], rope(o[..., -rd:], fc, inverse=True)], -1)
        G = t["o_groups"]
        o = fp8_round(o.reshape(T, G, -1))
        parts = [o[:, g] @ self.W(A + "wo_a.slice.%d" % g).T for g in range(G)]
        return torch.cat(parts, -1) @ self.W(A + "wo_b").T

    def indexer(self, li, A, x, qr, latent, ratio, T, fc, S):
        t = self.t
        I = A + "indexer."
        nh, hd, rd = t["index_n_heads"], t["index_head_dim"], t["qk_rope_head_dim"]
        clen = T // ratio
        if clen == 0:
            return torch.empty(T, 0, dtype=torch.long, device=self.dev)
        if li in self.kv_src and latent is not None:                      # this layer owns the index keys
            k = rms(latent @ self.W(I + "wk").T, self.V(I + "k_norm.weight"), self.eps)
            k = torch.cat([k[:, :-rd], rope(k[:, -rd:], fc[:T - T % ratio:ratio])], -1)
            S["ik"] = fp4_round(k, 32, False)
        q = (qr @ self.W(I + "wq_b").T).view(T, nh, hd)
        q = torch.cat([q[..., :-rd], rope(q[..., -rd:], fc)], -1)
        q = fp4_round(q, 32, False)
        ik = S["ik"][:clen]
        w = (x @ self.V(I + "weights_proj.weight").T) * (hd ** -0.5 * nh ** -0.5)   # [T, nh]
        score = torch.zeros(T, clen, device=self.dev)
        for a in range(0, T, 512):
            b = min(T, a + 512)
            s = torch.einsum("thd,kd->thk", q[a:b], ik).relu_()
            score[a:b] = (s * w[a:b].unsqueeze(-1)).sum(1)
        lens = (torch.arange(1, T + 1, device=self.dev) // ratio).unsqueeze(-1)
        score = score.masked_fill(torch.arange(clen, device=self.dev) >= lens, float("-inf"))
        if li == self.cand_src:
            S["cand"] = self.candidates(score, lens, t["candidate_topk_blocks"], t["candidate_block_size"])
        elif 0 <= self.cand_src < li:
            score = score.masked_fill(~S["cand"], float("-inf"))
        topk = min(t["index_topk"], clen)
        idx = score.topk(topk, dim=-1, sorted=False).indices.sort(-1).values
        return torch.where(idx < lens, idx + T, -1)

    def moe(self, li: int, x: torch.Tensor) -> torch.Tensor:
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

        def expert(P, xs):
            g = (xs @ self.W(P + "w1").T).clamp(max=lim)
            u = (xs @ self.W(P + "w3").T).clamp(-lim, lim)
            return F.silu(g) * u, P

        uniq = torch.unique(idx).tolist()
        for a in range(0, len(uniq), 24):                                   # 24 experts decoded at once
            batch = uniq[a:a + 24]
            Ps = [M + "experts.%d." % e for e in batch]
            w1 = ET.decode_many(self.st, [P + "w1" for P in Ps], self.dev)
            w3 = ET.decode_many(self.st, [P + "w3" for P in Ps], self.dev)
            w2 = ET.decode_many(self.st, [P + "w2" for P in Ps], self.dev)
            for i, e in enumerate(batch):
                rows, slot = torch.where(idx == e)
                xs = x[rows]
                h = F.silu((xs @ w1[i].T).clamp(max=lim)) * (xs @ w3[i].T).clamp(-lim, lim)
                y.index_add_(0, rows, (h * wts[rows, slot, None]) @ w2[i].T)
            del w1, w2, w3
        h, _ = expert(M + "shared_experts.", x)
        y += h @ self.W(M + "shared_experts.w2").T
        self.last_route = (idx, wts)
        return y

    @torch.inference_mode()
    def forward(self, ids: list[int], dump: list | None = None) -> torch.Tensor:
        T = len(ids)
        hashes = self.hasher.ids(ids)                                     # [T, n_engram, cols]
        emb = self.st.raw("embed.weight").view(np.uint16).reshape(-1, self.dim)
        h = torch.from_numpy(np.array(emb[np.asarray(ids)])).to(self.dev).view(torch.bfloat16).float()
        h = h.unsqueeze(1).repeat(1, self.hc, 1)
        pre = torch.zeros(T, self.hc, device=self.dev)
        pre[:, 0] = 1.0
        S: dict = {}
        for li in range(self.nl):
            t0 = time.time()
            B = "layers.%d." % li
            if li in self.lay.layer_ids:
                h = self.engram(li, h, hashes[:, self.lay.layer_ids.index(li)])
            res = h
            a_pre, a_post, a_comb = self.hc_mixes(h, self.V(B + "hc_attn_fn"), self.V(B + "hc_attn_scale"),
                                                  self.V(B + "hc_attn_base"))
            x = rms(self.hc_pre(h, pre), self.V(B + "attn_norm.weight"), self.eps)
            h = self.hc_post(self.attention(li, x, S), res, a_post, a_comb)
            res = h
            f_pre, f_post, f_comb = self.hc_mixes(h, self.V(B + "hc_ffn_fn"), self.V(B + "hc_ffn_scale"),
                                                  self.V(B + "hc_ffn_base"))
            x = rms(self.hc_pre(h, a_pre), self.V(B + "ffn_norm.weight"), self.eps)
            h = self.hc_post(self.moe(li, x), res, f_post, f_comb)
            pre = f_pre
            if dump is not None:
                dump.append(h.cpu().numpy())
            self.timing[li] = time.time() - t0
        x = rms(self.hc_pre(h, pre), self.V("norm.weight"), self.eps)
        head = self.W("head")
        return x @ head.T                                                  # [T, vocab]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(pathlib.Path("~/models/DSV4").expanduser()))
    ap.add_argument("--text", help="raw text (tokenized as is, no BOS added)")
    ap.add_argument("--ids", help="comma-separated token ids")
    ap.add_argument("--chat", help="a user message, as DeepSeek-V4.1's chat (non-thinking) prompt")
    ap.add_argument("--gen", type=int, default=0, help="greedy tokens to append (a whole forward each)")
    ap.add_argument("--dump", help="write the per-layer residual streams, the logits and the ids (.npz)")
    ap.add_argument("--no-hs", action="store_true", help="--dump without the per-layer residual streams")
    ap.add_argument("--dev", default="cuda:0")
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    if a.ids:
        ids = [int(x) for x in a.ids.split(",")]
    elif a.chat:
        ids = tok.encode("<｜begin▁of▁sentence｜><｜User｜>" + a.chat + "<｜Assistant｜></think>",
                         add_special_tokens=False).ids
    else:
        ids = tok.encode(a.text or "The capital of France is", add_special_tokens=False).ids
    r = Ref(model, a.dev)
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
    print("next:", [(tok.decode([i]), round(v, 3)) for v, i in zip(top.values.tolist(), top.indices.tolist())])
    if a.dump:
        np.savez(a.dump, ids=np.array(ids), logits=logits.cpu().numpy(),
                 hs=np.stack(dump) if dump else np.zeros(0))
    out = list(ids)
    for _ in range(a.gen):
        nxt = int(lp[-1].argmax())
        out.append(nxt)
        print("gen:", repr(tok.decode(out[len(ids):], skip_special_tokens=False)), flush=True)
        if nxt == 1:
            break
        logits = r.forward(out)
        lp = torch.log_softmax(logits.float(), -1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
