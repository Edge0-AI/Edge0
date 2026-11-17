"""A plain-torch port of the DFlash2 drafter (z-lab/dflash model.py, DFlash2DraftModel) for the offline acceptance experiment.

Everything is fp32 on the CPU. The target's embedding table and LM head are the dequantized tensors of OUR GGUF (IQ4_XS embedding,
Q6_K head), the context features are the hidden states dumped by the engine's STRATA_TAP_FILE hook.
"""
import json, math, struct, sys
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, r"C:\Users\AI-Server\Desktop\Strata\Public\Engine\third_party\llama.cpp\gguf-py")
import gguf
from gguf.quants import dequantize


def rms(x, w, eps=1e-6):
    v = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(v + eps)) * w


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def load_dequant(path, rows, cols, qtype):
    raw = np.fromfile(path, dtype=np.uint8)
    bs, ts = gguf.GGML_QUANT_SIZES[qtype]
    raw = raw.reshape(rows, cols // bs * ts)
    return torch.from_numpy(dequantize(raw, qtype).astype(np.float32).reshape(rows, cols))


class Drafter:
    def __init__(self, st_path, cfg_path, embed, head, threads=6):
        torch.set_num_threads(threads)
        cfg = json.load(open(cfg_path))
        dc = cfg["dflash_config"]
        self.H = cfg["hidden_size"]
        self.L = cfg["num_hidden_layers"]
        self.NH = cfg["num_attention_heads"]
        self.NKV = cfg["num_key_value_heads"]
        self.HD = cfg["head_dim"]
        self.win = cfg["sliding_window"]
        self.theta = cfg["rope_parameters"]["rope_theta"]
        self.eps = cfg["rms_norm_eps"]
        self.gs = dc["conv_group_size"]
        self.ks = dc["conv_kernel_size"]
        self.mask_id = dc["mask_token_id"]
        self.top_k = dc["selector_top_k"]
        self.taps = dc["target_layer_ids"]
        self.embed = embed
        self.head = head
        self.w = {}
        with safe_open(st_path, "pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                # the two codebooks stay bf16 (only gathered rows are used); everything else fp32
                self.w[k] = t if "codebook" in k else t.float()
        self.inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.HD, 2, dtype=torch.float32) / self.HD))
        self.Kc = None
        self.fast_head = False
        self.head_bf16 = None

    def enable_fast_head(self):
        self.head_bf16 = self.head.to(torch.bfloat16)
        self.fast_head = True

    def cos_sin(self, pos):
        fr = pos.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat([fr, fr], dim=-1)
        return emb.cos(), emb.sin()

    # ---- context: fc + hidden_norm over the tapped features, then every layer's K (norm + rope) and V
    def set_context(self, feats):
        """feats: [n, taps*H] float32, row j = the features of position j."""
        w = self.w
        h = rms(feats @ w["fc.weight"].T, w["hidden_norm.weight"], self.eps)
        n = h.shape[0]
        cos, sin = self.cos_sin(torch.arange(n))
        self.Kc, self.Vc = [], []
        for l in range(self.L):
            p = "layers.%d.self_attn." % l
            k = (h @ w[p + "k_proj.weight"].T).view(n, self.NKV, self.HD)
            v = (h @ w[p + "v_proj.weight"].T).view(n, self.NKV, self.HD)
            k = rms(k, w[p + "k_norm.weight"], self.eps)
            k = k * cos[:, None, :] + rotate_half(k) * sin[:, None, :]
            self.Kc.append(k.permute(1, 0, 2).contiguous())   # [NKV, n, HD]
            self.Vc.append(v.permute(1, 0, 2).contiguous())

    def conv(self, x, dyn, base):
        """_grouped_dynamic_convolve over a block x [b, H]; dyn [b, ks, groups]; base [ks, H]."""
        b, H = x.shape
        groups = H // self.gs
        blocks = x.view(b, groups, self.gs)
        out = torch.zeros_like(blocks)
        for o in range(base.shape[0]):
            vals = blocks if o == 0 else F.pad(blocks[:-o], (0, 0, 0, 0, o, 0))
            kern = base[o].float().view(1, groups, self.gs)
            out = out + kern * vals
            out = out + dyn[:, o, :, None] * vals
        return out.view(b, H)

    def prepare(self, x, p):
        w = self.w
        groups = self.H // self.gs
        dyn = (x @ w[p + "kernel_projection.weight"].T).view(x.shape[0], 2, self.ks, groups)
        return self.conv(x, dyn[:, 0], w[p + "base_kernel"][0]), dyn[:, 1]

    def finish(self, x, dyn, p):
        return self.conv(x, dyn, self.w[p + "base_kernel"][1])

    # ---- one block: anchor token at position s (s >= 1), block size b; returns the drafted path (b-1 tokens), the candidate lists
    def draft(self, s, anchor, b, greedy=True):
        w = self.w
        H, NH, NKV, HD = self.H, self.NH, self.NKV, self.HD
        ids = torch.tensor([anchor] + [self.mask_id] * (b - 1))
        h = self.embed[ids]                                  # [b, H]
        pos = torch.arange(s, s + b)
        cos, sin = self.cos_sin(pos)
        # keys: context positions max(0, s-win+1) .. s-1 (a block row i sees ctx j when s+i-j < win), then the block
        lo = max(0, s - self.win + 1)
        for l in range(self.L):
            p = "layers.%d." % l
            r = h
            x = rms(h, w[p + "input_layernorm.weight"], self.eps)
            x, kdyn = self.prepare(x, p + "attention_conv.")
            q = (x @ w[p + "self_attn.q_proj.weight"].T).view(b, NH, HD)
            q = rms(q, w[p + "self_attn.q_norm.weight"], self.eps)
            kn = rms((x @ w[p + "self_attn.k_proj.weight"].T).view(b, NKV, HD), w[p + "self_attn.k_norm.weight"], self.eps)
            vn = (x @ w[p + "self_attn.v_proj.weight"].T).view(b, NKV, HD)
            q = q * cos[:, None, :] + rotate_half(q) * sin[:, None, :]
            kn = kn * cos[:, None, :] + rotate_half(kn) * sin[:, None, :]
            Kc = self.Kc[l][:, lo:s, :]                      # [NKV, c, HD]
            Vc = self.Vc[l][:, lo:s, :]
            K = torch.cat([Kc, kn.permute(1, 0, 2)], dim=1)  # [NKV, c+b, HD]
            V = torch.cat([Vc, vn.permute(1, 0, 2)], dim=1)
            c = Kc.shape[1]
            kpos = torch.cat([torch.arange(lo, s), pos])
            vis = (pos[:, None] - kpos[None, :]).abs() < self.win          # [b, c+b]
            rep = NH // NKV
            Kx = K.repeat_interleave(rep, dim=0)             # [NH, c+b, HD]
            Vx = V.repeat_interleave(rep, dim=0)
            sc = torch.einsum("bhd,hkd->hbk", q, Kx) * (HD ** -0.5)
            sc = sc.masked_fill(~vis[None], float("-inf"))
            att = torch.softmax(sc, dim=-1)
            o = torch.einsum("hbk,hkd->bhd", att, Vx).reshape(b, NH * HD)
            o = o @ w[p + "self_attn.o_proj.weight"].T
            o = self.finish(o, kdyn, p + "attention_conv.")
            h = r + o
            r = h
            x = rms(h, w[p + "post_attention_layernorm.weight"], self.eps)
            x, mdyn = self.prepare(x, p + "mlp_conv.")
            m = (F.silu(x @ w[p + "mlp.gate_proj.weight"].T) * (x @ w[p + "mlp.up_proj.weight"].T)) @ w[p + "mlp.down_proj.weight"].T
            m = self.finish(m, mdyn, p + "mlp_conv.")
            h = r + m
        self.last_all = rms(h, w["norm.weight"], self.eps)
        out = self.last_all[1:]                              # rows 1..b-1
        if self.fast_head:
            # candidates from a bf16 pass over the table (7x faster on this CPU), then the exact fp32 logits of the 32 best decide the top 16
            lb = (out.to(torch.bfloat16) @ self.head_bf16.T)
            c32 = torch.topk(lb, 32, dim=-1, sorted=False).indices            # [b-1, 32]
            ex = torch.einsum("rkh,rh->rk", self.head[c32], out)               # [b-1, 32] exact
            unary, j16 = torch.topk(ex, self.top_k, dim=-1, sorted=False)
            cand = c32.gather(-1, j16)
            logits = None
        else:
            logits = out @ self.head.T                       # [b-1, V]
            unary, cand = torch.topk(logits, self.top_k, dim=-1, sorted=False)
        hp = out @ w["candidate_selector.hidden_projection.weight"].T   # [b-1, rank]
        A = w["candidate_selector.predecessor_codebook"]
        B = w["candidate_selector.successor_codebook"]
        pred = anchor
        path, probs = [], []
        for i in range(b - 1):
            sc = unary[i] + (B[cand[i]].float() @ (A[pred].float() * hp[i]))
            j = int(torch.argmax(sc))
            probs.append(float(torch.softmax(sc, dim=-1)[j]))   # the selector's own probability of the pick (a confidence)
            pred = int(cand[i][j])
            path.append(pred)
        top1 = (cand.gather(-1, unary.argmax(-1, keepdim=True))[:, 0] if logits is None else logits.argmax(-1)).tolist()
        self.last_probs = probs
        return path, top1


def read_taps(path):
    f = open(path, "rb")
    magic, ver, N, nt = struct.unpack("<4i", f.read(16))
    assert magic == 0x50415453 and ver == 1, (hex(magic), ver)
    layers = struct.unpack("<%di" % nt, f.read(4 * nt))
    rec = 8 + 4 * nt * N
    data = f.read()
    n = len(data) // rec
    pos = np.empty(n, np.int32)
    tok = np.empty(n, np.int32)
    taps = np.empty((n, nt * N), np.float32)
    for i in range(n):
        o = i * rec
        pos[i], tok[i] = struct.unpack_from("<2i", data, o)
        taps[i] = np.frombuffer(data, np.float32, nt * N, o + 8)
    return layers, pos, tok, taps
