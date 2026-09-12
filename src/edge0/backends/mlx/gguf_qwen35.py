"""GGUF tensor adapter for the existing Qwen3.5 MoE text backbone.

GGUF conversion folds RMS gamma offsets and tiles linear-attention value
heads. Projection wrappers invert head tiling at the existing backbone's
boundary, without converting expert weights or the full checkpoint.
Real-checkpoint acceptance is required before advertising this adapter.
"""
import mlx.core as mx
import mlx.nn as nn

from edge0.backends.mlx._impl.qwen3_5 import TextModel, TextModelArgs
from edge0.streaming.gguf import GGUFExperts


class Linear(nn.Module):
    def __init__(self, weights, name, before=None, after=None):
        super().__init__()
        self._weights, self._name = weights, name
        self._before, self._after = before, after

    def __call__(self, x):
        if self._before:
            x = self._before(x)
        y = self._weights.linear(self._name, x)
        return self._after(y) if self._after else y


class Embedding(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self._weights = weights

    def __call__(self, ids):
        result = self._weights.rows("token_embd.weight", ids.reshape(-1).tolist())
        return result.reshape(*ids.shape, -1)

    def as_linear(self, x):
        return self._weights.linear("token_embd.weight", x)


class Experts(nn.Module):
    def __init__(self, expert):
        super().__init__()
        self._expert = expert

    def __call__(self, x, indices):
        shape = indices.shape
        y = self._expert(x.reshape(-1, x.shape[-1]), indices.reshape(-1, shape[-1]))
        return y.reshape(*shape, x.shape[-1])


class Model:
    def __init__(self, weights, metadata):
        self.w, self.metadata = weights, metadata
        self.history = []
        def get(key):
            name = "qwen35moe." + key
            if name not in metadata:
                raise ValueError(f"missing GGUF architecture metadata {name!r}")
            return metadata[name]
        self.get = get
        h, n, nk, nv, d = [get(k) for k in ("embedding_length", "block_count", "ssm.group_count", "ssm.time_step_rank", "ssm.state_size")]
        ne, topk, ff, shared = [get(k) for k in ("expert_count", "expert_used_count", "expert_feed_forward_length", "expert_shared_feed_forward_length")]
        nh, nkv, hd, conv = [get(k) for k in ("attention.head_count", "attention.head_count_kv", "attention.key_length", "ssm.conv_kernel")]
        interval = get("full_attention_interval")
        if any(type(v) is not int or v < 1 for v in (h, n, nk, nv, d, ne, topk, ff, shared, nh, nkv, hd, conv, interval)):
            raise ValueError("Qwen3.5 GGUF dimensions must be positive integers")
        if nv % nk or nh % nkv or topk > ne or n < interval or interval < 2:
            raise ValueError("invalid Qwen3.5 attention layout or routing dimensions")
        if get("ssm.inner_size") != nv * d or get("attention.value_length") != hd:
            raise ValueError("inconsistent Qwen3.5 attention dimensions")
        if metadata.get("qwen35moe.nextn_predict_layers", 0):
            raise ValueError("Qwen3.5 GGUF with appended MTP layers is not implemented")
        recurrent = metadata.get("qwen35moe.attention.recurrent_layers")
        if recurrent is not None and recurrent != [(i + 1) % interval != 0 for i in range(n)]:
            raise ValueError("Qwen3.5 backbone requires regular full_attention_interval")
        if metadata.get("qwen35moe.rope.scaling.type", "none") not in ("none", "default"):
            raise ValueError("scaled RoPE is not implemented for Qwen3.5 GGUF")
        embedding = weights.source.tensors.get("token_embd.weight")
        if embedding is None or len(embedding.shape) != 2 or embedding.shape[1] != h:
            raise ValueError("invalid or missing token_embd.weight")
        vocab = embedding.shape[0]
        shapes = {"token_embd.weight": (vocab, h), "output_norm.weight": (h,)}
        tied = "output.weight" not in weights.source.tensors
        if not tied:
            shapes["output.weight"] = (vocab, h)
        for i in range(n):
            p = f"blk.{i}."
            shapes[p + "attn_norm.weight"] = (h,)
            shapes[p + "post_attention_norm.weight"] = (h,)
            shapes[p + "ffn_gate_inp.weight"] = (ne, h)
            shapes[p + "ffn_gate_inp_shexp.weight"] = (h,)
            for proj, shape in (("gate", (ne, ff, h)), ("up", (ne, ff, h)), ("down", (ne, h, ff))):
                shapes[p + f"ffn_{proj}_exps.weight"] = shape
            for proj, shape in (("gate", (shared, h)), ("up", (shared, h)), ("down", (h, shared))):
                shapes[p + f"ffn_{proj}_shexp.weight"] = shape
            if (i + 1) % interval:
                entries = {"attn_qkv.weight": ((nk * 2 + nv) * d, h), "attn_gate.weight": (nv * d, h),
                    "ssm_conv1d.weight": ((nk * 2 + nv) * d, conv), "ssm_a": (nv,), "ssm_dt.bias": (nv,),
                    "ssm_alpha.weight": (nv, h), "ssm_beta.weight": (nv, h), "ssm_norm.weight": (d,),
                    "ssm_out.weight": (h, nv * d)}
            else:
                entries = {"attn_q.weight": (nh * hd * 2, h), "attn_k.weight": (nkv * hd, h),
                    "attn_v.weight": (nkv * hd, h), "attn_output.weight": (h, nh * hd),
                    "attn_q_norm.weight": (hd,), "attn_k_norm.weight": (hd,)}
            shapes.update({p + k: v for k, v in entries.items()})
        for name, shape in shapes.items():
            found = weights.source.tensors.get(name)
            if found is None or found.shape != shape:
                raise ValueError(f"required text tensor {name!r}: expected {shape}, found {None if found is None else found.shape}")
        unknown = set(weights.source.tensors) - shapes.keys()
        if unknown:
            raise ValueError(f"unrecognized Qwen3.5 GGUF tensors: {sorted(unknown)[:8]}")
        args = TextModelArgs(model_type="qwen3_5_moe_text", hidden_size=h, num_hidden_layers=n,
            intermediate_size=ff, num_experts=ne, num_experts_per_tok=topk,
            moe_intermediate_size=ff, shared_expert_intermediate_size=shared, num_attention_heads=nh,
            num_key_value_heads=nkv, head_dim=hd, linear_num_key_heads=nk, linear_num_value_heads=nv,
            linear_key_head_dim=d, linear_value_head_dim=d, linear_conv_kernel_dim=conv,
            full_attention_interval=interval, rms_norm_eps=get("attention.layer_norm_rms_epsilon"),
            vocab_size=vocab, max_position_embeddings=get("context_length"), tie_word_embeddings=tied,
            rope_parameters={"type": "default", "rope_theta": get("rope.freq_base"),
                             "partial_rotary_factor": get("rope.dimension_count") / hd})
        # Lazy skeleton construction; streamed matrices replace random initializers
        # before anything can evaluate them.
        self.text = TextModel(args)
        self.text.model.embed_tokens = Embedding(weights)
        self.text.model.norm.weight = weights.vector("output_norm.weight")
        if not tied:
            self.text.lm_head = Linear(weights, "output.weight")
        self.experts = [GGUFExperts(weights, i, ne) for i in range(n)]
        def untile(x, dim=d):
            return x.reshape(*x.shape[:-1], nv // nk, nk, dim).swapaxes(-3, -2).reshape(x.shape)
        def tile(x):
            return x.reshape(*x.shape[:-1], nk, nv // nk, d).swapaxes(-3, -2).reshape(x.shape)
        def qkv_untile(x):
            return mx.concatenate([x[..., :2 * nk * d], untile(x[..., 2 * nk * d:])], axis=-1)
        for i, layer in enumerate(self.text.layers):
            p = f"blk.{i}."
            layer.input_layernorm.weight = weights.vector(p + "attn_norm.weight")
            layer.post_attention_layernorm.weight = weights.vector(p + "post_attention_norm.weight")
            block = layer.mlp
            block.gate = Linear(weights, p + "ffn_gate_inp.weight")
            block.shared_expert_gate = Linear(weights, p + "ffn_gate_inp_shexp.weight")
            for proj in ("gate", "up", "down"):
                setattr(block.shared_expert, proj + "_proj", Linear(weights, p + f"ffn_{proj}_shexp.weight"))
            block.switch_mlp = Experts(self.experts[i])
            if layer.is_linear:
                a = layer.linear_attn
                for attr, name, after in (("in_proj_qkv", "attn_qkv", qkv_untile),
                    ("in_proj_z", "attn_gate", untile), ("in_proj_a", "ssm_alpha", lambda x: untile(x, 1)),
                    ("in_proj_b", "ssm_beta", lambda x: untile(x, 1))):
                    setattr(a, attr, Linear(weights, p + name + ".weight", after=after))
                a.out_proj = Linear(weights, p + "ssm_out.weight", before=tile)
                a.dt_bias = untile(weights.vector(p + "ssm_dt.bias"), 1)
                a.A_log = mx.log(-untile(weights.vector(p + "ssm_a"), 1))
                a.norm.weight = weights.vector(p + "ssm_norm.weight")
                raw = weights.rows(p + "ssm_conv1d.weight", range((nk * 2 + nv) * d))
                a.conv1d.weight = qkv_untile(raw.T).T[:, :, None]
            else:
                a = layer.self_attn
                for attr, name in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
                    setattr(a, attr, Linear(weights, p + name + ".weight"))
                a.q_norm.weight = weights.vector(p + "attn_q_norm.weight")
                a.k_norm.weight = weights.vector(p + "attn_k_norm.weight")

    def make_cache(self):
        return self.text.make_cache()

    def __call__(self, ids, cache, offset=0, **kwargs):
        if offset + len(ids) > self.get("context_length"):
            raise ValueError("request exceeds checkpoint context length")
        h = self.text.model(mx.array(ids)[None, :], cache=cache, hidden_clip=0, **kwargs)
        if self.text.args.tie_word_embeddings:
            return self.text.model.embed_tokens.as_linear(h[0, -1])
        return self.text.lm_head(h[0, -1])

    def close(self):
        self.w.close()
