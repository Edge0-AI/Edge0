"""Torch port of the vendored Qwen3.5-MoE backbone (edge0-35b).

A line-by-line port of ``backends/mlx/_impl/qwen3_5_moe.py`` plus the
``qwen3_5.py`` / ``qwen3_next.py`` pieces it builds on (mlx-lm 0.31.0),
with the same module tree, parameter names and call signatures, so
``engine/qwen.py`` and the prerouter patch drive it unchanged.

It computes the MLX way on the MLX checkpoint layout: RMSNorm weights are
the absolute multiplier (MLX's sanitize stores ``w + 1`` for the
zero-centred transformers norms) and ``conv1d.weight`` is [C, k, 1], so the
published edge0-35b tensors are used exactly as stored -- no conversion on
load. Checked against the MLX module in ``tests/test_backend_parity.py``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from edge0.backends.cuda._impl.bailing_hybrid import (
    ArraysCache,
    KVCache,
    RMSNorm,
    _linear,
    _rms_norm,
    _sdpa,
    create_attention_mask,
)


def create_ssm_mask(h, cache=None):
    """mlx-lm: only padded batches get an SSM mask; edge0 serves B=1."""
    return None


def _rope(x, dims: int, base: float, offset: int):
    """``mx.fast.rope(traditional=False)``: rotate the first ``dims`` of the
    last axis as two halves (i, i + dims/2) by ``pos * base**(-2i/dims)``;
    the rest passes through."""
    L = x.shape[-2]
    half = dims // 2
    inv_freq = base ** (-torch.arange(0, half, dtype=torch.float32,
                                      device=x.device) * 2 / dims)
    pos = torch.arange(offset, offset + L, dtype=torch.float32, device=x.device)
    ang = pos[:, None] * inv_freq[None, :]
    cos, sin = torch.cos(ang), torch.sin(ang)
    xf = x.to(torch.float32)
    x1, x2, rest = xf[..., :half], xf[..., half:dims], xf[..., dims:]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, rest], dim=-1)
    return out.to(x.dtype)


@dataclass
class TextModelArgs:
    model_type: str = ""
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    linear_num_value_heads: int = 64
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 192
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = None
    full_attention_interval: int = 4
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default", "mrope_section": [11, 11, 10],
            "rope_theta": 100000, "partial_rotary_factor": 0.25})
    partial_rotary_factor: float = 0.25
    rope_theta: float = 100000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    @classmethod
    def from_dict(cls, params: dict):
        names = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in params.items() if k in names})

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_parameters:
            if ("type" not in self.rope_parameters
                    and "rope_type" in self.rope_parameters):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")
            self.partial_rotary_factor = self.rope_parameters.get(
                "partial_rotary_factor", 0.25)
            self.rope_theta = self.rope_parameters.get("rope_theta", 100000.0)
            self.rope_scaling = self.rope_parameters


class RMSNormGated(nn.Module):
    """``Qwen3NextRMSNormGated``: rms_norm(x) * silu(gate), the gating in
    float32 (``_precise_swiglu``), result in the input dtype."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states, gate=None):
        x = _rms_norm(hidden_states, self.weight, self.eps)
        if gate is None:
            return x.to(hidden_states.dtype)
        g = F.silu(gate.to(torch.float32))
        return (g * x.to(torch.float32)).to(hidden_states.dtype)


class RoPE(nn.Module):
    """``nn.RoPE(dims, traditional=False, base)``: the only kind the
    edge0-35b config asks for (rope_type "default")."""

    def __init__(self, dims: int, base: float, scale: float = 1.0):
        super().__init__()
        if scale != 1.0:
            raise NotImplementedError("scaled RoPE is not ported")
        self.dims, self.base = dims, base

    def forward(self, x, offset: int = 0):
        return _rope(x, self.dims, self.base, offset)


def _initialize_rope(dims, base, scaling_config):
    rope_type = "default"
    if scaling_config is not None:
        rope_type = scaling_config.get("type") or scaling_config.get(
            "rope_type", "default")
    if rope_type in ("default", "mrope"):
        return RoPE(dims, base)
    raise NotImplementedError(f"RoPE type {rope_type!r} is not ported")


class Attention(nn.Module):
    """``Qwen3NextAttention``: gated full attention with GQA, q/k norms and
    partial RoPE."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        bias = args.attention_bias
        self.q_proj = _linear(args.hidden_size,
                              self.num_attention_heads * self.head_dim * 2, bias)
        self.k_proj = _linear(args.hidden_size,
                              self.num_key_value_heads * self.head_dim, bias)
        self.v_proj = _linear(args.hidden_size,
                              self.num_key_value_heads * self.head_dim, bias)
        self.o_proj = _linear(self.num_attention_heads * self.head_dim,
                              args.hidden_size, bias)
        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = _initialize_rope(
            int(self.head_dim * args.partial_rotary_factor), args.rope_theta,
            args.rope_scaling)

    def forward(self, x, mask=None, cache: Optional[Any] = None):
        B, L, _ = x.shape
        q_out = self.q_proj(x).reshape(B, L, self.num_attention_heads, -1)
        queries, gate = torch.split(q_out, q_out.shape[-1] // 2, dim=-1)
        gate = gate.reshape(B, L, -1)
        keys, values = self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(queries).transpose(1, 2)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(1, 2)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(1, 2)
        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        # GQA as mx.fast.sdpa does it: query head h reads kv head h // rep
        rep = self.num_attention_heads // self.num_key_value_heads
        if rep > 1:
            keys = keys.repeat_interleave(rep, dim=1)
            values = values.repeat_interleave(rep, dim=1)
        out = _sdpa(queries, keys, values, self.scale, mask)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out * torch.sigmoid(gate))


class MLP(nn.Module):
    """``Qwen3NextMLP``."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = _linear(dim, hidden_dim)
        self.down_proj = _linear(hidden_dim, dim)
        self.up_proj = _linear(dim, hidden_dim)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None):  # noqa: N803
    """mlx-lm ``gated_delta_update`` on its ops path: beta = sigmoid(b),
    g = exp(-exp(A_log) * softplus(a + dt_bias)) cast to a's dtype, one
    decay scalar per value head, state in q's dtype, k heads repeated
    (consecutively) to the value heads."""
    beta = torch.sigmoid(b)
    g = torch.exp(-torch.exp(A_log.to(torch.float32))
                  * F.softplus(a + dt_bias)).to(a.dtype)
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = torch.zeros(B, Hv, Dv, Dk, dtype=q.dtype, device=q.device)
    if Hv // Hk > 1:
        q = q.repeat_interleave(Hv // Hk, dim=-2)
        k = k.repeat_interleave(Hv // Hk, dim=-2)
    ys = []
    for t in range(T):
        qt, kt, vt, gt, bt = q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t]
        state = state * gt[..., None, None]
        kv_mem = (state * kt[..., None, :]).sum(dim=-1)
        delta = (vt - kv_mem) * bt[..., None]
        state = state + kt[..., None, :] * delta[..., None]
        ys.append((state * qt[..., None, :]).sum(dim=-1))
    return torch.stack(ys, dim=1), state


class GatedDeltaNet(nn.Module):
    """``qwen3_5.GatedDeltaNet`` (split in_proj_qkv / z / b / a)."""

    def __init__(self, config: TextModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError("num_v_heads must be divisible by num_k_heads")
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = _DepthwiseConv(self.conv_dim, self.conv_kernel_size)
        self.in_proj_qkv = _linear(self.hidden_size, self.key_dim * 2 + self.value_dim)
        self.in_proj_z = _linear(self.hidden_size, self.value_dim)
        self.in_proj_b = _linear(self.hidden_size, self.num_v_heads)
        self.in_proj_a = _linear(self.hidden_size, self.num_v_heads)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = _linear(self.value_dim, self.hidden_size)

    def forward(self, inputs, mask=None, cache: Optional[Any] = None):
        B, S, _ = inputs.shape
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = torch.zeros(B, self.conv_kernel_size - 1, self.conv_dim,
                                     dtype=inputs.dtype, device=inputs.device)
        if mask is not None:
            qkv = torch.where(mask[..., None], qkv, 0)
        conv_input = torch.cat([conv_state, qkv], dim=1)
        if cache is not None:
            cache[0] = conv_input[:, -(self.conv_kernel_size - 1):]
        conv_out = F.silu(self.conv1d(conv_input))
        q, k, v = [
            t.reshape(B, S, h, d) for t, h, d in zip(
                torch.split(conv_out, [self.key_dim, self.key_dim, self.value_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim])]
        state = cache[1] if cache is not None else None
        inv_scale = k.shape[-1] ** -0.5
        ones = torch.ones(q.shape[-1], dtype=q.dtype, device=q.device)
        q = (inv_scale ** 2) * _rms_norm(q, ones, 1e-6)
        k = inv_scale * _rms_norm(k, ones, 1e-6)
        out, state = _gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, state)
        if cache is not None:
            cache[1] = state
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


class _DepthwiseConv(nn.Module):
    """MLX ``nn.Conv1d(groups=C, padding=0)`` on [B, T, C]; the weight in
    MLX's layout [C, k, 1] as stored in the checkpoint."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels, kernel_size, 1))

    def forward(self, x):
        w = self.weight.transpose(1, 2).to(x.dtype)            # [C, 1, k]
        return F.conv1d(x.transpose(1, 2), w, groups=x.shape[-1]).transpose(1, 2)


class SparseMoeBlock(nn.Module):
    """``Qwen3NextSparseMoeBlock``: softmax top-k router (precise softmax),
    routed experts from the streaming installer, gated shared expert. The
    35b prerouter patches this class's ``__call__`` (prerouter/install.py),
    as it does the MLX class."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.gate = _linear(dim, self.num_experts)
        self.switch_mlp = None     # install_streaming_experts swaps the twin in
        self.shared_expert = MLP(dim, args.shared_expert_intermediate_size)
        self.shared_expert_gate = _linear(dim, 1)

    def forward(self, x):
        gates = self.gate(x)
        gates = torch.softmax(gates.to(torch.float32), dim=-1).to(gates.dtype)
        k = self.top_k
        inds = torch.argsort(gates, dim=-1, stable=True)[..., -k:]
        scores = torch.take_along_dim(gates, inds, dim=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(dim=-1, keepdim=True)
        if self.switch_mlp is None:
            raise RuntimeError(
                "routed experts not installed: run install_streaming_experts")
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(dim=-2)
        shared_y = self.shared_expert(x)
        shared_y = torch.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if args.num_experts > 0:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def forward(self, x, mask=None, cache=None):
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen3_5TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(args, i) for i in range(args.num_hidden_layers)])
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def forward(self, inputs, cache=None, input_embeddings=None,
                before_layer_cb=None, after_layer_cb=None,
                async_eval_per_layer: bool = False):
        hidden_states = input_embeddings if input_embeddings is not None \
            else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])
        for li, (layer, c) in enumerate(zip(self.layers, cache)):
            if before_layer_cb is not None:
                before_layer_cb(li)
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)
            if after_layer_cb is not None:
                after_layer_cb(li, hidden_states)
            # async_eval_per_layer: an MLX lazy-graph hint; torch is eager.
        return self.norm(hidden_states)


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3_5TextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = _linear(args.hidden_size, args.vocab_size)

    def forward(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            return F.linear(out, self.model.embed_tokens.weight)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache()
                for l in self.layers]

    def sanitize(self, weights: dict) -> dict:
        """The MLX sanitize, on torch tensors: drop MTP; a raw transformers
        checkpoint (conv1d [C, 1, k]) is moved to the MLX layout and its
        zero-centred norms shifted by +1. The published edge0 checkpoints
        are already in MLX form, so for them this changes nothing."""
        has_mtp = any("mtp." in k for k in weights)
        unsanitized = any("conv1d.weight" in k and v.shape[-1] != 1
                          for k, v in weights.items())
        shift = has_mtp or unsanitized
        weights = {k: v for k, v in weights.items() if "mtp." not in k}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        norm_keys = (".input_layernorm.weight", ".post_attention_layernorm.weight",
                     "model.norm.weight", ".q_norm.weight", ".k_norm.weight")
        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.transpose(1, 2).contiguous()
            if shift and k.endswith(norm_keys) and v.ndim == 1:
                weights[k] = v + 1.0
        return weights


@dataclass
class ModelArgs:
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params: dict):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return cls(model_type=params["model_type"], text_config=params["text_config"])


class Model(nn.Module):
    """``qwen3_5_moe.Model`` (via ``qwen3_5.Model``)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))

    def forward(self, inputs, cache=None, input_embeddings=None):
        return self.language_model(inputs, cache=cache,
                                   input_embeddings=input_embeddings)

    def sanitize(self, weights: dict) -> dict:
        new = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif not key.startswith("language_model."):
                key = "language_model." + key
            new[key] = value
        for l in range(self.language_model.args.num_hidden_layers):
            prefix = f"language_model.model.layers.{l}.mlp"
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key in new:
                gate_up = new.pop(gate_up_key)
                mid = gate_up.shape[-2] // 2
                new[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
                new[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]
                new[f"{prefix}.switch_mlp.down_proj.weight"] = new.pop(
                    f"{prefix}.experts.down_proj")
        return self.language_model.sanitize(new)

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()
