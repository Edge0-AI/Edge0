# SPDX-License-Identifier: Apache-2.0
"""Torch port of the vendored Ling 3.0 backbone (``model_type: bailing_hybrid``).

A line-by-line port of ``backends/mlx/_impl/bailing_hybrid.py`` -- see that
file for the architecture and its references -- with the same module tree,
parameter names and call signatures, so ``engine/ling.py``, the engine hooks
and the prerouter stager drive it unchanged. It replaces the checkpoint's
own ``modeling_bailing_moe_v3.py`` (remote code, ``fla``/Triton, per-expert
``ModuleList``) on the torch backend.

Parameters are named as stored in the checkpoint, which differs from the
MLX module in one place: the MLX ``sanitize`` moves ``*_conv1d.weight``
([C, 1, k], torch's depthwise layout) to ``*_conv1d.conv.weight`` in MLX's
[C, k, 1] layout; here the torch layout is used as stored.

Checked against the MLX module on the real edge0-8b checkpoint, layer by
layer, in ``tests/test_backend_parity.py``.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _rope_interleave_torch(x: torch.Tensor, positions: torch.Tensor,
                           theta: float) -> torch.Tensor:
    """``apply_rotary_pos_emb_interleave`` of ``modeling_bailing_moe_v3``
    (same formula as the MLX port): rotate consecutive pairs of the last
    axis by ``positions * 1 / theta ** (arange(0, d, 2) / d)``."""
    B, H, L, D = x.shape
    freqs = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float32,
                                          device=x.device) / D))
    angles = torch.outer(positions.to(torch.float32), freqs)
    emb = torch.cat([angles, angles], dim=-1)
    cos = torch.cos(emb)[None, None]
    sin = torch.sin(emb)[None, None]
    xi = x.reshape(B, H, L, D // 2, 2).transpose(-1, -2).reshape(B, H, L, D)
    h = D // 2
    rh = torch.cat([-xi[..., h:], xi[..., :h]], dim=-1)
    return xi * cos + rh * sin


def _rms_norm(x, weight, eps):
    """``mx.fast.rms_norm``: normalize in float32, output in x's dtype."""
    xf = x.to(torch.float32)
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.to(torch.float32)).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dims))

    def forward(self, x):
        return _rms_norm(x, self.weight, self.eps)


def _linear(i, o, bias=False):
    # the facade's Linear: accepts plain-tensor weight assignment, which
    # prerouter/install.py relies on
    from edge0.backends.cuda.nn import Linear
    return Linear(i, o, bias=bias)


@dataclass
class ModelArgs:
    model_type: str = "bailing_hybrid"
    hidden_size: int = 1536
    num_hidden_layers: int = 24
    intermediate_size: int = 4608
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 157184
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = False
    layer_group_size: int = 4
    first_k_dense_replace: int = 1
    short_conv_kernel_size: int = 4
    no_kda_lora: bool = True
    kda_safe_gate: bool = True
    kda_lower_bound: float = -5.0
    q_lora_rank: int | None = 256
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float = 6000000.0
    rope_interleave: bool = True
    rope_scaling: dict | None = None
    use_qkv_bias: bool = False
    gated_attention_proj_granularity_type: str | None = "head_wise"
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    moe_intermediate_size: int = 512
    moe_shared_expert_intermediate_size: int = 512
    n_group: int = 8
    topk_group: int = 4
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    moe_router_enable_expert_bias: bool = True
    prerouter_enabled: bool = False
    prerouter_start_layer: int = 7
    prerouter_hidden: int = 512

    @classmethod
    def from_dict(cls, params: dict) -> "ModelArgs":
        """Keep only this class's fields, like mlx-lm's BaseModelArgs."""
        names = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in params.items() if k in names})

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def is_mla_layer(self, idx: int) -> bool:
        g = self.layer_group_size
        full = self.num_hidden_layers // g * g
        return (idx + 1) % g == 0 or idx >= full


# ---- caches (mlx-lm KVCache / ArraysCache, the parts the model uses) --------

class KVCache:
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = torch.cat([self.keys, keys], dim=2)
            self.values = torch.cat([self.values, values], dim=2)
        self.offset += keys.shape[2]
        return self.keys, self.values

    def make_mask(self, N: int):
        return None if N == 1 else "causal"

    def size(self):
        return self.offset


class ArraysCache:
    def __init__(self, size: int):
        self.cache = [None] * size

    def __setitem__(self, idx, value):
        self.cache[idx] = value

    def __getitem__(self, idx):
        return self.cache[idx]


def create_attention_mask(h, cache=None):
    N = h.shape[1]
    if cache is not None and hasattr(cache, "make_mask"):
        return cache.make_mask(N)
    return None if N == 1 else "causal"


def _sdpa(queries, keys, values, scale, mask):
    """``mx.fast.scaled_dot_product_attention``. Its "causal" mask is
    aligned to the END of the keys (query i of L sees keys up to
    Lk - L + i); torch's ``is_causal`` aligns to the start, which is wrong
    once a cache holds earlier tokens -- so build the mask explicitly."""
    if isinstance(mask, str) and mask == "causal":
        L, Lk = queries.shape[-2], keys.shape[-2]
        q_pos = torch.arange(Lk - L, Lk, device=queries.device)[:, None]
        mask = torch.arange(Lk, device=queries.device)[None, :] <= q_pos
    return F.scaled_dot_product_attention(queries, keys, values,
                                          attn_mask=mask, scale=scale)


# ---- KDA ---------------------------------------------------------------------

class ShortConv1d(nn.Module):
    """Causal depthwise conv with silu and a rolling cache; weight in the
    checkpoint's torch layout [channels, 1, ksize]."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.zeros(channels, 1, kernel_size))

    def forward(self, x, state=None):
        B, T, C = x.shape
        if state is None:
            state = torch.zeros(B, self.kernel_size - 1, C, dtype=x.dtype,
                                device=x.device)
        conv_input = torch.cat([state, x], dim=1)
        out = F.conv1d(conv_input.transpose(1, 2),
                       self.weight.to(conv_input.dtype), groups=C)
        out = F.silu(out.transpose(1, 2))
        if self.kernel_size == 1:
            new_state = conv_input[:, :0, :]
        else:
            new_state = conv_input[:, -(self.kernel_size - 1):, :]
        return out, new_state


def _kda_gate(f, A_log, dt_bias, *, safe_gate: bool, lower_bound: float):  # noqa: N803
    f = f.to(torch.float32) + dt_bias.to(torch.float32)
    a = torch.exp(A_log.to(torch.float32))
    if safe_gate:
        return lower_bound * torch.sigmoid(a[..., None] * f)
    return -a[..., None] * F.softplus(f)


def _kda_update(q, k, v, g_log, beta, state):
    """mlx-lm ``gated_delta_ops`` with vectorized (per-channel) decay:
    q, k [B, T, H, Dk], v [B, T, H, Dv], g_log [B, T, H, Dk], beta
    [B, T, H], state [B, H, Dv, Dk] (float32). Sequential over T."""
    g = torch.exp(g_log)
    if state is None:
        B = q.shape[0]
        state = torch.zeros(B, v.shape[-2], v.shape[-1], k.shape[-1],
                            dtype=torch.float32, device=q.device)
    ys = []
    for t in range(q.shape[1]):
        qt, kt, vt, gt, bt = q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t]
        state = state * gt[..., None, :]
        kv_mem = (state * kt[..., None, :]).sum(dim=-1)          # [B, H, Dv]
        delta = (vt - kv_mem) * bt[..., None]
        state = state + kt[..., None, :] * delta[..., None]
        ys.append((state * qt[..., None, :]).sum(dim=-1))
    return torch.stack(ys, dim=1), state


class BailingKDA(nn.Module):
    """Kimi Delta Attention with the Ling V3 safe gate."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.proj_dim = self.num_heads * self.head_dim
        self.conv_kernel = args.short_conv_kernel_size
        self.safe_gate = args.kda_safe_gate
        self.lower_bound = float(args.kda_lower_bound)
        self.no_kda_lora = args.no_kda_lora
        self.scale = float(self.head_dim) ** -0.5

        hidden = args.hidden_size
        self.q_proj = _linear(hidden, self.proj_dim)
        self.k_proj = _linear(hidden, self.proj_dim)
        self.v_proj = _linear(hidden, self.proj_dim)
        self.q_conv1d = ShortConv1d(self.proj_dim, self.conv_kernel)
        self.k_conv1d = ShortConv1d(self.proj_dim, self.conv_kernel)
        self.v_conv1d = ShortConv1d(self.proj_dim, self.conv_kernel)
        if self.no_kda_lora:
            self.f_proj = _linear(hidden, self.proj_dim)
            self.g_proj = _linear(hidden, self.proj_dim)
        else:
            self.f_a_proj = _linear(hidden, self.head_dim)
            self.f_b_proj = _linear(self.head_dim, self.proj_dim)
            self.g_a_proj = _linear(hidden, self.head_dim)
            self.g_b_proj = _linear(self.head_dim, self.proj_dim)
        self.b_proj = _linear(hidden, self.num_heads)
        self.A_log = nn.Parameter(torch.zeros(self.num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.proj_dim))
        self.o_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = _linear(self.proj_dim, hidden)

    def forward(self, x, mask=None, cache: Any | None = None):
        B, T, _ = x.shape
        dtype = x.dtype
        if cache is not None:
            q_state, k_state, v_state, ssm_state = cache.cache
        else:
            q_state = k_state = v_state = ssm_state = None

        q_conv, q_state = self.q_conv1d(self.q_proj(x), q_state)
        k_conv, k_state = self.k_conv1d(self.k_proj(x), k_state)
        v_conv, v_state = self.v_conv1d(self.v_proj(x), v_state)
        if cache is not None:
            cache[0], cache[1], cache[2] = q_state, k_state, v_state

        q = q_conv.reshape(B, T, self.num_heads, self.head_dim)
        k = k_conv.reshape(B, T, self.num_heads, self.head_dim)
        v = v_conv.reshape(B, T, self.num_heads, self.head_dim)
        qf, kf = q.to(torch.float32), k.to(torch.float32)
        q = self.scale * qf / (torch.linalg.norm(qf, dim=-1, keepdim=True) + 1e-6)
        k = kf / (torch.linalg.norm(kf, dim=-1, keepdim=True) + 1e-6)

        if self.no_kda_lora:
            f = self.f_proj(x)
            gate = self.g_proj(x)
        else:
            f = self.f_b_proj(self.f_a_proj(x))
            gate = self.g_b_proj(self.g_a_proj(x))
        f = f.reshape(B, T, self.num_heads, self.head_dim)
        g = _kda_gate(f, self.A_log,
                      self.dt_bias.reshape(self.num_heads, self.head_dim),
                      safe_gate=self.safe_gate, lower_bound=self.lower_bound)
        beta = torch.sigmoid(self.b_proj(x).to(torch.float32))

        out, ssm_state = _kda_update(q, k, v.to(torch.float32), g, beta,
                                     ssm_state)
        if cache is not None:
            cache[3] = ssm_state

        gate = gate.reshape(B, T, self.num_heads, self.head_dim)
        out = self.o_norm(out.to(dtype)) * torch.sigmoid(gate)
        return self.o_proj(out.reshape(B, T, -1))


# ---- MLA ---------------------------------------------------------------------

class BailingMLA(nn.Module):
    """DeepSeek-style MLA plus the V3 head-wise output gate."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.q_lora_rank = args.q_lora_rank
        self.scale = self.qk_head_dim ** -0.5
        self.gate_kind = args.gated_attention_proj_granularity_type

        hidden = args.hidden_size
        bias = args.use_qkv_bias
        if self.q_lora_rank is None:
            self.q_proj = _linear(hidden, self.num_heads * self.qk_head_dim)
        else:
            self.q_a_proj = _linear(hidden, self.q_lora_rank, bias)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
            self.q_b_proj = _linear(self.q_lora_rank,
                                    self.num_heads * self.qk_head_dim)
        self.kv_a_proj_with_mqa = _linear(
            hidden, self.kv_lora_rank + self.qk_rope_head_dim, bias)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        self.kv_b_proj = _linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim))
        if self.gate_kind == "head_wise":
            self.g_proj = _linear(hidden, self.num_heads)
        elif self.gate_kind == "element_wise":
            self.g_proj = _linear(hidden, self.num_heads * self.v_head_dim)
        self.dense = _linear(self.num_heads * self.v_head_dim, hidden, bias)
        if args.rope_scaling:
            raise NotImplementedError(
                "bailing_hybrid: rope_scaling is not supported "
                f"(got {args.rope_scaling!r})")
        self.rope_theta = args.rope_theta

    def forward(self, x, mask=None, cache: Any | None = None):
        B, L, _ = x.shape
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(B, L, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed = self.kv_a_proj_with_mqa(x)
        kv_latent, k_pe = torch.split(
            compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv = self.kv_b_proj(kv_latent)
        kv = kv.reshape(B, L, self.num_heads,
                        self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, values = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        offset = cache.offset if cache is not None else 0
        positions = torch.arange(L, device=x.device) + offset
        q_pe = _rope_interleave_torch(q_pe, positions, self.rope_theta)
        k_pe = _rope_interleave_torch(k_pe, positions, self.rope_theta)
        k_pe = k_pe.expand(B, self.num_heads, L, self.qk_rope_head_dim)

        # RoPE computes in float32; MLX's concatenate promotes the same way
        queries = torch.cat([q_nope.to(q_pe.dtype), q_pe], dim=-1)
        keys = torch.cat([k_nope.to(k_pe.dtype), k_pe], dim=-1)
        values = values.to(keys.dtype)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        out = _sdpa(queries, keys, values, self.scale, mask)
        out = out.transpose(1, 2).reshape(B, L, -1)
        if self.gate_kind == "head_wise":
            gate = torch.sigmoid(self.g_proj(x))
            out = out.reshape(B, L, self.num_heads, self.v_head_dim)
            out = (out * gate[..., None]).reshape(B, L, -1)
        elif self.gate_kind == "element_wise":
            out = out * torch.sigmoid(self.g_proj(x))
        return self.dense(out)


# ---- MLP / MoE -----------------------------------------------------------------

class BailingMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate: int):
        super().__init__()
        self.gate_proj = _linear(args.hidden_size, intermediate)
        self.up_proj = _linear(args.hidden_size, intermediate)
        self.down_proj = _linear(intermediate, args.hidden_size)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _group_topk(scores, select, a: ModelArgs):
    """Group-limited top-k shared by BailingGate and the prerouter path
    (moe/routing.py's law): drop the n_group - topk_group groups with the
    lowest top-2 sums, then take the top-k by selection score; the weights
    are the raw scores of the chosen experts."""
    B = select.shape[:-1]
    k_drop = a.n_group - a.topk_group
    if k_drop > 0:
        grouped = select.reshape(*B, a.n_group, a.num_experts // a.n_group)
        group_scores = torch.topk(grouped, 2, dim=-1).values.sum(dim=-1)
        drop = torch.argsort(group_scores, dim=-1, stable=True)[..., :k_drop]
        masked = grouped.clone()
        masked.scatter_(-2, drop[..., None].expand(*drop.shape, grouped.shape[-1]),
                        float("-inf"))
        select = masked.reshape(*B, a.num_experts)
    k = a.num_experts_per_tok
    idx = torch.argsort(-select, dim=-1, stable=True)[..., :k]
    w = torch.take_along_dim(scores, idx, dim=-1)
    if a.norm_topk_prob:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return idx, w * a.routed_scaling_factor


class BailingGate(nn.Module):
    """Sigmoid-score router with noaux_tc expert-bias group selection."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.weight = nn.Parameter(torch.zeros(args.num_experts, args.hidden_size))
        if args.moe_router_enable_expert_bias:
            self.expert_bias = nn.Parameter(torch.zeros(args.num_experts))

    def forward(self, x):
        scores = torch.sigmoid((x @ self.weight.to(x.dtype).T).to(torch.float32))
        select = scores
        if self.args.moe_router_enable_expert_bias:
            select = scores + self.expert_bias.to(torch.float32)
        return _group_topk(scores, select, self.args)


class BailingPrerouter(nn.Module):
    """One-MoE-block-ahead expert selector (weights from the prerouter file,
    installed by prerouter/install.py)."""

    def __init__(self, args: ModelArgs, hidden: int | None = None,
                 input_dim: int | None = None):
        super().__init__()
        self.args = args
        self.hidden = hidden or args.prerouter_hidden
        self.input_dim = input_dim or args.hidden_size
        self.fc1 = _linear(self.input_dim, self.hidden)
        self.fc2 = _linear(self.hidden, args.num_experts)
        self.linear_init = _linear(self.input_dim, args.num_experts)
        with torch.no_grad():
            self.linear_init.weight.zero_()

    def forward(self, x, topk_oh=None, prev_oh=None):
        dtype = self.fc1.weight.dtype
        feats = [x.to(dtype)]
        if topk_oh is not None:
            feats.append(topk_oh.to(dtype))
        if prev_oh is not None:
            feats.append(prev_oh.to(dtype))
        xi = torch.cat(feats, dim=-1)
        h = F.gelu(self.fc1(xi))
        # MLX promotes mixed dtypes (e.g. fp16 heads + a float32 zero
        # linear_init) to the wider type; torch Linear needs one dtype.
        a = self.fc2(h)
        b = self.linear_init(xi.to(self.linear_init.weight.dtype))
        out = torch.promote_types(a.dtype, b.dtype)
        return a.to(out) + b.to(out)


class BailingSparseMoE(nn.Module):
    def __init__(self, args: ModelArgs, prerouter: BailingPrerouter | None = None,
                 use_prerouter: bool = False):
        super().__init__()
        self.args = args
        self.gate = BailingGate(args)
        self.prerouter = prerouter
        self.use_prerouter = use_prerouter
        self.last_topk = None
        self.prev_topk_oh = None
        self.feature_topk = os.environ.get("PREROUTER_FEATURE_TOPK", "teacher")
        self.intra_mode = os.environ.get("PREROUTER_INTRA", "0") == "1"
        # Routed experts come from the streaming installer
        # (install_streaming_experts swaps a StreamingSwitchGLU in here);
        # the checkpoint's stacked expert tensors are never made resident.
        self.experts = None
        self.shared_experts = BailingMLP(
            args, args.moe_shared_expert_intermediate_size * args.num_shared_experts)

    def _select_from_logits(self, logits):
        scores = torch.sigmoid(logits.to(torch.float32))
        return _group_topk(scores, scores, self.args)

    def forward(self, x, prev_prerouter_logits=None):
        if self.use_prerouter and prev_prerouter_logits is not None:
            idx, w = self._select_from_logits(prev_prerouter_logits)
            if not self.intra_mode and self.feature_topk == "teacher":
                feat_idx, _ = self.gate(x)
            else:
                feat_idx = idx
        else:
            idx, w = self.gate(x)
            feat_idx = idx
        self.last_topk = feat_idx if not self.intra_mode else None
        if self.experts is None:
            raise RuntimeError(
                "routed experts not installed: run install_streaming_experts")
        routed = self.experts(x, idx)
        out = (routed * w[..., None].to(routed.dtype)).sum(dim=-2)
        return out + self.shared_experts(x)


class BailingDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_mla = args.is_mla_layer(layer_idx)
        self.attention = BailingMLA(args) if self.is_mla else BailingKDA(args, layer_idx)
        self.prerouter_enabled = bool(getattr(args, "prerouter_enabled", False))
        self.prerouter_start_layer = int(getattr(args, "prerouter_start_layer", 7))
        self.prerouter_logits = None
        self.m_in_cache = None
        self.after_attention_cb = None
        self.has_prerouter = (
            self.prerouter_enabled
            and layer_idx >= self.prerouter_start_layer - 1
            and layer_idx < args.num_hidden_layers - 1)
        self.use_prerouter = (
            self.prerouter_enabled
            and layer_idx >= self.prerouter_start_layer
            and layer_idx < args.num_hidden_layers)
        if layer_idx >= args.first_k_dense_replace:
            self.mlp = BailingSparseMoE(
                args,
                prerouter=BailingPrerouter(args) if self.has_prerouter else None,
                use_prerouter=self.use_prerouter)
        else:
            self.mlp = BailingMLP(args, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(self, x, mask=None, cache=None, prev_prerouter_logits=None):
        h = x + self.attention(self.input_layernorm(x), mask, cache)
        m_in = self.post_attention_layernorm(h)
        self.m_in_cache = m_in if self.has_prerouter else None
        self.prerouter_logits = None
        if self.after_attention_cb is not None:
            self.after_attention_cb(self.layer_idx, m_in)
        if isinstance(self.mlp, BailingSparseMoE):
            out = self.mlp(m_in, prev_prerouter_logits=prev_prerouter_logits)
        else:
            out = self.mlp(m_in)
        return h + out


class BailingModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList(
            [BailingDecoderLayer(args, i) for i in range(args.num_hidden_layers)])
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.first_mla_idx = next(
            i for i in range(args.num_hidden_layers) if args.is_mla_layer(i))

    def forward(self, inputs, cache=None, input_embeddings=None,
                after_layer_cb=None, before_layer_cb=None,
                async_eval_per_layer: bool = False, prerouter_cache=None):
        h = input_embeddings if input_embeddings is not None \
            else self.word_embeddings(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mla_mask = create_attention_mask(h, cache[self.first_mla_idx])
        clip_v = float(os.environ.get("LING_HIDDEN_CLIP", "0"))
        for li, (layer, c) in enumerate(zip(self.layers, cache)):
            if before_layer_cb is not None:
                before_layer_cb(li)
            mask = mla_mask if layer.is_mla else None
            prev = None
            if (getattr(layer, "use_prerouter", False)
                    and prerouter_cache is not None and li in prerouter_cache):
                prev = prerouter_cache[li]
            h = layer(h, mask, c, prev)
            if clip_v > 0:
                h = torch.where(torch.isnan(h), torch.zeros_like(h),
                                h.clamp(-clip_v, clip_v))
            if after_layer_cb is not None:
                after_layer_cb(li, h)
            # async_eval_per_layer: an MLX lazy-graph hint; torch is eager.
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = BailingModel(args)
        self.tie_word_embeddings = args.tie_word_embeddings
        if not self.tie_word_embeddings:
            self.lm_head = _linear(args.hidden_size, args.vocab_size)

    def forward(self, inputs, cache=None, input_embeddings=None,
                after_layer_cb=None):
        h = self.model(inputs, cache, input_embeddings, after_layer_cb)
        if self.tie_word_embeddings:
            return F.linear(h, self.model.word_embeddings.weight)
        return self.lm_head(h)

    def sanitize(self, weights: dict) -> dict:
        """Drop MTP heads, like the MLX sanitize. Experts are stacked on
        disk already, and conv weights are used in their stored layout."""
        return {k: v for k, v in weights.items()
                if ".mtp_" not in k and "mtp." not in k}

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() if layer.is_mla else ArraysCache(size=4)
                for layer in self.model.layers]
