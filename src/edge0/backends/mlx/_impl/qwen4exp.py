"""Qwen3.8 Flash Next text graph over bounded GGUF weight operations.

Mathematical reference: llama.cpp src/models/qwen4exp.cpp at
6c84c7d5d8833c6e0df69628f75a0f599797934e (MIT, checkpoints/LICENSE.ggml).
GGUF already folds zero-centered normalization weights and tiles GDN V
heads; neither transformation may be applied again here.
"""
import math
import os

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

from edge0.streaming.gguf import GGUFExperts


def rms(x, weight, eps):
    return mx.fast.rms_norm(x, weight, eps)


def silu(x):
    return x * mx.sigmoid(x)


def ple_rows(ids, history, ngram, per_gram, eos, multipliers, offsets, sizes):
    """Exact uint64 hash with EOS padding; returns rows and bounded history."""
    history = list(history)
    result = []
    for token in ids:
        context = [int(token)]
        cut = False
        for j in range(1, ngram):
            previous = history[-j] if j <= len(history) else eos
            cut = cut or previous == eos
            context.append(eos if cut else previous)
        rows = []
        mixed = (context[0] * multipliers[0]) & ((1 << 64) - 1)
        for n in range(2, ngram + 1):
            mixed ^= (context[n - 1] * multipliers[n - 1]) & ((1 << 64) - 1)
            for g in range(per_gram):
                h = (n - 2) * per_gram + g
                rows.append(mixed % sizes[h] + offsets[h])
        result.append(rows)
        history.append(int(token))
        history = history[-(ngram - 1):]
    return result, history


class Model:
    def __init__(self, weights, metadata):
        self.w = weights
        self.metadata = metadata
        self.arch = metadata["general.architecture"]
        if self.arch != "qwen4exp":
            raise ValueError(f"text adapter unavailable for {self.arch!r}")
        self.h = self.required("embedding_length")
        self.n = self.required("block_count")
        self.hc = self.required("hyper_connection.count")
        self.lr = self.required("hyper_connection.low_rank")
        self.eps = self.required("attention.layer_norm_rms_epsilon")
        self.nh = self.required("attention.head_count")
        self.nkv = self.required("attention.head_count_kv")
        self.d = self.required("attention.key_length")
        self.dk = self.required("ssm.state_size")
        self.nk = self.required("ssm.group_count")
        self.nv = self.required("ssm.time_step_rank")
        self.conv = self.required("ssm.conv_kernel")
        self.ne = self.required("expert_count")
        self.topk = self.required("expert_used_count")
        self.ff = self.required("expert_feed_forward_length")
        self.shared = self.required("expert_shared_feed_forward_length")
        self.rot = self.required("rope.dimension_count")
        self.theta = self.required("rope.freq_base")
        self.idx_heads = self.required("attention.indexer.head_count")
        self.idx_dim = self.required("attention.indexer.key_length")
        self.idx_k = self.required("attention.indexer.top_k")
        self.ratios = self.required("attention.compress_ratios")
        self.recurrent = metadata.get(self.arch + ".attention.recurrent_layers")
        if self.recurrent is None:
            interval = self.required("full_attention_interval")
            if interval < 1:
                raise ValueError("full_attention_interval must be positive")
            self.recurrent = [(i + 1) % interval != 0 for i in range(self.n)]
        self.ple_layers = metadata.get(self.arch + ".ple.layers", [])
        if len(self.ple_layers) > 1:
            raise ValueError("qwen4exp supports one PLE layer")
        self.validate()
        self.experts = [GGUFExperts(weights, i, self.ne) for i in range(self.n)]
        self.history = []
        self.trace_directory = os.environ.get("EDGE0_GGUF_TRACE_DIR")

    def trace(self, name, value):
        if self.trace_directory:
            import numpy as np
            from pathlib import Path
            np.save(Path(self.trace_directory) / (name + ".npy"), np.array(value))

    def required(self, key):
        name = self.arch + "." + key
        if name not in self.metadata:
            raise ValueError(f"missing GGUF architecture metadata {name!r}")
        return self.metadata[name]

    def validate(self):
        """Validate all text tensor names and shapes before any weight reads."""
        dims = (self.h, self.n, self.hc, self.lr, self.nh, self.nkv, self.d,
                self.dk, self.nk, self.nv, self.conv, self.ne, self.topk,
                self.ff, self.shared, self.idx_heads, self.idx_dim, self.idx_k)
        if any(type(v) is not int or v <= 0 for v in dims):
            raise ValueError("qwen4exp dimensions must be positive integers")
        if self.nh % self.nkv or self.nv % self.nk or self.topk > self.ne:
            raise ValueError("invalid qwen4exp attention or routing head counts")
        if self.required("ssm.inner_size") != self.nv * self.dk:
            raise ValueError("ssm.inner_size disagrees with value head dimensions")
        if self.required("attention.value_length") != self.d:
            raise ValueError("qwen4exp requires equal attention key/value dimensions")
        if len(self.ratios) != self.n or len(self.recurrent) != self.n:
            raise ValueError("attention layer metadata length disagrees with block_count")
        if self.rot % 2 or not 0 < self.rot <= min(self.d, self.idx_dim):
            raise ValueError("invalid rotary dimension")
        if self.metadata.get(self.arch + ".rope.scaling.type", "none") not in ("none", "default"):
            raise ValueError("scaled RoPE is not implemented for custom qwen4exp")
        shapes = {}
        def add(name, shape):
            shapes[name] = shape
        vocab = len(self.metadata.get("tokenizer.ggml.tokens", []))
        token_info = self.w.source.tensors.get("token_embd.weight")
        if token_info is None or len(token_info.shape) != 2:
            raise ValueError("missing or invalid token_embd.weight")
        vocab = vocab or token_info.shape[0]
        add("token_embd.weight", (vocab, self.h))
        if "output.weight" in self.w.source.tensors:
            add("output.weight", (vocab, self.h))
        for suffix, shape in (("norm", (self.h * self.hc,)),
                              ("down", (self.lr, self.h * self.hc)),
                              ("up", (self.h * self.hc, self.lr))):
            add(f"output_hc_{suffix}.weight", shape)
        for i in range(self.n):
            p = f"blk.{i}."
            for module in ("attn", "ffn"):
                for suffix, shape in (("norm", (self.h * self.hc,)),
                    ("down", (self.lr, self.h * self.hc)),
                    ("up", (self.h * self.hc, self.lr)),
                    ("inject", (self.hc, self.h * self.hc))):
                    add(f"{p}hc_{module}_{suffix}.weight", shape)
            for proj, shape in (("gate", (self.ne, self.ff, self.h)),
                                ("up", (self.ne, self.ff, self.h)),
                                ("down", (self.ne, self.h, self.ff))):
                add(p + f"ffn_{proj}_exps.weight", shape)
            for proj, shape in (("gate", (self.shared, self.h)), ("up", (self.shared, self.h)),
                                ("down", (self.h, self.shared))):
                add(p + f"ffn_{proj}_shexp.weight", shape)
            add(p + "ffn_gate_inp.weight", (self.ne, self.h))
            add(p + "ffn_gate_inp_shexp.weight", (self.h,))
            if self.recurrent[i]:
                channels = (2 * self.nk + self.nv) * self.dk
                for name, shape in {
                    "attn_qkv.weight": (channels, self.h), "attn_gate.weight": (self.nv * self.dk, self.h),
                    "ssm_conv1d.weight": (channels, self.conv), "ssm_a": (self.nv,),
                    "ssm_dt.bias": (self.nv,), "ssm_alpha.weight": (self.nv, self.h),
                    "ssm_beta.weight": (self.nv, self.h), "ssm_norm.weight": (self.dk,),
                    "ssm_out.weight": (self.h, self.nv * self.dk)}.items():
                    add(p + name, shape)
                if self.ratios[i] != 0:
                    raise ValueError("recurrent layer cannot have indexer compression")
            else:
                if type(self.ratios[i]) is not int or self.ratios[i] < 1:
                    raise ValueError("full attention requires a positive compression ratio")
                for name, shape in {
                    "attn_q.weight": (self.nh * self.d * 2, self.h),
                    "attn_k.weight": (self.nkv * self.d, self.h),
                    "attn_v.weight": (self.nkv * self.d, self.h),
                    "attn_output.weight": (self.h, self.nh * self.d),
                    "attn_q_norm.weight": (self.d,), "attn_k_norm.weight": (self.d,),
                    "indexer.q_proj.weight": (self.idx_heads * self.idx_dim, self.h),
                    "indexer.k_proj.weight": (self.idx_dim, self.h),
                    "indexer.q_norm.weight": (self.idx_dim,), "indexer.k_norm.weight": (self.idx_dim,)}.items():
                    add(p + name, shape)
        if self.ple_layers:
            i = self.ple_layers[0]
            if type(i) is not int or not 0 <= i < self.n:
                raise ValueError("invalid PLE layer index")
            gram, heads = self.required("ple.ngram_size"), self.required("ple.heads_per_ngram")
            dim = self.required("embedding_length_per_layer_input")
            if not 2 <= gram <= 4 or heads < 1 or (gram - 1) * heads * dim != self.h:
                raise ValueError("PLE head dimensions disagree with embedding_length")
            offsets, sizes = self.required("ple.head_offsets"), self.required("ple.head_vocab_sizes")
            if len(self.required("ple.layer_multipliers")) != gram or len(offsets) != (gram - 1) * heads or len(sizes) != len(offsets):
                raise ValueError("invalid PLE hash constants")
            table = self.w.source.tensors.get("per_layer_token_embd.weight")
            if table is None or len(table.shape) != 2 or table.shape[1] != dim:
                raise ValueError("missing or invalid PLE embedding table")
            if any(o < 0 or s < 1 or o + s > table.shape[0] for o, s in zip(offsets, sizes)):
                raise ValueError("PLE head range exceeds embedding table")
            add(table.name, table.shape)
            p = f"blk.{i}.ple_"
            add(p + "key.weight", (self.h * self.hc, self.h))
            add(p + "value.weight", (self.h, self.h))
            for name in ("norm_key", "norm_query", "norm_conv"):
                add(p + name + ".weight", (self.h * self.hc,))
            add(p + "conv1d.weight", (self.h * self.hc, self.required("ple.conv_kernel")))
        for name, shape in shapes.items():
            tensor = self.w.source.tensors.get(name)
            if tensor is None or tensor.shape != shape:
                raise ValueError(f"required text tensor {name!r}: expected {shape}, found {None if tensor is None else tensor.shape}")
        unknown = set(self.w.source.tensors) - shapes.keys()
        # This exporter omits vision and MTP. Reject extra tensors until their
        # schema has been explicitly recognized, rather than ignoring text weights.
        if unknown:
            raise ValueError(f"unrecognized GGUF tensors: {sorted(unknown)[:8]}")

    def make_cache(self):
        # conv, recurrent state, attention K/V, indexer raw K, PLE conv.
        return [ArraysCache(6) for _ in range(self.n)]

    def _matrix(self, name):
        shape = self.w.source.tensors[name].shape
        # Only small convolution kernels; large projections use linear().
        return self.w.rows(name, range(shape[0]))

    def mix(self, x, prefix, inject=True):
        normalized = rms(x, None, self.eps).reshape(x.shape[0], -1)
        normalized = normalized * self.w.vector(prefix + "norm.weight")
        lo = silu(self.w.linear(prefix + "down.weight", normalized) / self.hc)
        gates = mx.sigmoid(self.w.linear(prefix + "up.weight", lo))
        mixed = (normalized * gates).reshape(-1, self.hc, self.h).mean(axis=1)
        weights = self.w.linear(prefix + "inject.weight", normalized) if inject else None
        return mixed, weights

    def combine(self, residual, output, inject):
        return residual + output[:, None, :] * (2 * mx.sigmoid(inject / self.hc))[:, :, None]

    def rope(self, x, positions):
        # Text positions are identical in all IMRoPE sections; equivalent to
        # split-half RoPE on the checkpoint's rotary subdimension.
        freq = self.theta ** (-mx.arange(0, self.rot, 2, dtype=mx.float32) / self.rot)
        angles = mx.array(positions, dtype=mx.float32)[:, None, None] * freq[None, None, :]
        a, b = x[..., :self.rot // 2], x[..., self.rot // 2:self.rot]
        y = mx.concatenate([a * mx.cos(angles) - b * mx.sin(angles),
                            a * mx.sin(angles) + b * mx.cos(angles), x[..., self.rot:]], axis=-1)
        return y

    def linear_attention(self, x, i, cache):
        p = f"blk.{i}."
        t = x.shape[0]
        qkv = self.w.linear(p + "attn_qkv.weight", x)
        if i == 0:
            self.trace("linear_attn_qkv_mixed-0", qkv)
        z = self.w.linear(p + "attn_gate.weight", x).reshape(t, self.nv, self.dk)
        beta = mx.sigmoid(self.w.linear(p + "ssm_beta.weight", x))
        alpha = self.w.linear(p + "ssm_alpha.weight", x) + self.w.vector(p + "ssm_dt.bias")
        decay = mx.exp(mx.logaddexp(alpha, mx.zeros_like(alpha)) * self.w.vector(p + "ssm_a"))
        history = cache[0]
        if history is None:
            history = mx.zeros((self.conv - 1, qkv.shape[-1]))
        padded = mx.concatenate([history, qkv], axis=0)
        kernel = self._matrix(p + "ssm_conv1d.weight")
        conv = sum(padded[k:k + t] * kernel[:, k] for k in range(self.conv))
        cache[0] = padded[-(self.conv - 1):] if self.conv > 1 else padded[:0]
        q, k, v = mx.split(silu(conv), [self.nk * self.dk, 2 * self.nk * self.dk], axis=-1)
        q, k = q.reshape(t, self.nk, self.dk), k.reshape(t, self.nk, self.dk)
        q = q / mx.maximum(mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True)), self.eps)
        k = k / mx.maximum(mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True)), self.eps)
        q = mx.tile(q, (1, self.nv // self.nk, 1)) / math.sqrt(self.dk)
        k = mx.tile(k, (1, self.nv // self.nk, 1))
        v = v.reshape(t, self.nv, self.dk)
        state = cache[1]
        if state is None:
            state = mx.zeros((self.nv, self.dk, self.dk))  # K,V
        outputs = []
        for j in range(t):
            state = state * decay[j, :, None, None]
            prediction = (state * k[j, :, :, None]).sum(axis=-2)
            delta = (v[j] - prediction) * beta[j, :, None]
            state = state + k[j, :, :, None] * delta[:, None, :]
            outputs.append((state * q[j, :, :, None]).sum(axis=-2))
            mx.eval(state, outputs[-1])
        cache[1] = state
        if i == 0:
            self.trace("output_predelta-0", mx.stack(outputs))
        out = rms(mx.stack(outputs), self.w.vector(p + "ssm_norm.weight"), self.eps) * mx.sigmoid(z)
        return self.w.linear(p + "ssm_out.weight", out.reshape(t, -1))

    def attention(self, x, i, cache, offset):
        p = f"blk.{i}."
        t = x.shape[0]
        positions = list(range(offset, offset + t))
        qfull = self.w.linear(p + "attn_q.weight", x).reshape(t, self.nh, 2 * self.d)
        q, gate = mx.split(qfull, 2, axis=-1)
        k = self.w.linear(p + "attn_k.weight", x).reshape(t, self.nkv, self.d)
        v = self.w.linear(p + "attn_v.weight", x).reshape(t, self.nkv, self.d)
        q = self.rope(rms(q, self.w.vector(p + "attn_q_norm.weight"), self.eps), positions)
        k = self.rope(rms(k, self.w.vector(p + "attn_k_norm.weight"), self.eps), positions)
        ik = self.w.linear(p + "indexer.k_proj.weight", x)
        iq = self.w.linear(p + "indexer.q_proj.weight", x).reshape(t, self.idx_heads, self.idx_dim)
        iq = self.rope(rms(iq, self.w.vector(p + "indexer.q_norm.weight"), self.eps), positions)
        if cache[2] is not None:
            k, v, ik = [mx.concatenate([cache[j], value], axis=0) for j, value in ((2, k), (3, v), (4, ik))]
        cache[2], cache[3], cache[4] = k, v, ik
        ratio = self.ratios[i]
        outputs = []
        for j, pos in enumerate(positions):
            visible = pos + 1
            width = min(visible, self.idx_k + ratio - 1)
            if visible > width:
                complete = visible // ratio
                pooled = ik[:complete * ratio].reshape(complete, ratio, self.idx_dim).mean(axis=1)
                pooled = rms(pooled, self.w.vector(p + "indexer.k_norm.weight"), self.eps)
                pooled = self.rope(pooled[:, None, :], list(range(0, complete * ratio, ratio)))[:, 0, :]
                score = mx.maximum(iq[j] @ pooled.T, 0).sum(axis=0)
                expanded = mx.concatenate([mx.repeat(score, ratio), mx.full((visible % ratio,), 1e9)])
                selected = mx.argsort(-expanded)[:width]
            else:
                selected = mx.arange(visible)
            kj = mx.repeat(k[selected], self.nh // self.nkv, axis=1).transpose(1, 0, 2)
            vj = mx.repeat(v[selected], self.nh // self.nkv, axis=1).transpose(1, 0, 2)
            scores = (q[j, :, None, :] * kj).sum(axis=-1) / math.sqrt(self.d)
            out = (mx.softmax(scores, axis=-1)[:, :, None] * vj).sum(axis=1)
            outputs.append(out * mx.sigmoid(gate[j]))
        return self.w.linear(p + "attn_output.weight", mx.stack(outputs).reshape(t, -1))

    def ple(self, x, rows, i, cache):
        t = x.shape[0]
        p = f"blk.{i}.ple_"
        emb = self.w.rows("per_layer_token_embd.weight", [r for row in rows for r in row]).reshape(t, self.h)
        key = self.w.linear(p + "key.weight", emb).reshape(t, self.hc, self.h)
        value = self.w.linear(p + "value.weight", emb)
        def norm(value, name):
            return rms(value, None, self.eps) * self.w.vector(p + name + ".weight").reshape(self.hc, self.h)
        dot = (norm(key, "norm_key") * norm(x, "norm_query")).sum(axis=-1) / math.sqrt(self.h)
        gate = mx.sigmoid(mx.sign(dot) * mx.sqrt(mx.maximum(mx.abs(dot), 1e-6)))
        gated = value[:, None, :] * gate[..., None]
        normalized = norm(gated, "norm_conv").reshape(t, -1)
        kernel = self._matrix(p + "conv1d.weight")
        dilation = self.required("ple.ngram_size")
        hist = (kernel.shape[1] - 1) * dilation
        previous = cache[5] if cache[5] is not None else mx.zeros((hist, self.hc * self.h))
        padded = mx.concatenate([previous, normalized], axis=0)
        out = sum(padded[k * dilation:k * dilation + t] * kernel[:, k] for k in range(kernel.shape[1]))
        cache[5] = padded[-hist:] if hist else padded[:0]
        return x + gated + silu(out).reshape(t, self.hc, self.h)

    def __call__(self, ids, cache, offset=0, before_layer_cb=None, after_layer_cb=None):
        if not ids:
            raise ValueError("cannot forward an empty token sequence")
        if offset + len(ids) > self.required("context_length"):
            raise ValueError("request exceeds checkpoint context length")
        x = mx.repeat(self.w.rows("token_embd.weight", ids)[:, None, :], self.hc, axis=1)
        rows = None
        if self.ple_layers:
            rows, self.history = ple_rows(ids, self.history, self.required("ple.ngram_size"),
                self.required("ple.heads_per_ngram"), self.required("ple.eos_token_id"),
                self.required("ple.layer_multipliers"), self.required("ple.head_offsets"), self.required("ple.head_vocab_sizes"))
        for i in range(self.n):
            if before_layer_cb:
                before_layer_cb(i)
            if i in self.ple_layers:
                x = self.ple(x, rows, i, cache[i])
            p = f"blk.{i}."
            mixed, inject = self.mix(x, p + "hc_attn_")
            if i == 0:
                self.trace("hc_mixed_attn-0", mixed)
            output = (self.linear_attention(mixed, i, cache[i]) if self.recurrent[i]
                      else self.attention(mixed, i, cache[i], offset))
            x = self.combine(x, output, inject)
            if i == 0:
                self.trace("linear_attn_out-0", output)
                self.trace("hc_combine_attn-0", x)
            mixed, inject = self.mix(x, p + "hc_ffn_")
            scores = mx.softmax(self.w.linear(p + "ffn_gate_inp.weight", mixed), axis=-1)
            indices = mx.argsort(-scores, axis=-1)[:, :self.topk]
            selected = mx.take_along_axis(scores, indices, axis=-1)
            selected = selected / selected.sum(axis=-1, keepdims=True)
            output = self.experts[i](mixed, indices, selected)
            shared = silu(self.w.linear(p + "ffn_gate_shexp.weight", mixed)) * self.w.linear(p + "ffn_up_shexp.weight", mixed)
            shared = self.w.linear(p + "ffn_down_shexp.weight", shared)
            shared = shared * mx.sigmoid(self.w.linear(p + "ffn_gate_inp_shexp.weight", mixed))
            x = self.combine(x, output + shared, inject)
            mx.eval(x, cache[i].state)
            self.trace(f"l_last-{i}", x)
            if after_layer_cb:
                after_layer_cb(i)
        mixed, _ = self.mix(x[-1:], "output_hc_", inject=False)
        output_name = "output.weight" if "output.weight" in self.w.source.tensors else "token_embd.weight"
        return self.w.linear(output_name, mixed)[0]

    def close(self):
        self.w.close()
