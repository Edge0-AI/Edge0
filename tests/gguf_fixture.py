"""Deterministic, tiny, complete Qwen4Exp fixture for cross-runtime tests."""
import numpy as np
from test_gguf import write_gguf


def tiny_qwen4exp(path):
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = list(byte_values)
    for b in range(256):
        if b not in byte_values:
            byte_values.append(b)
            chars.append(256 + len(chars) - 188)
    tokens = [chr(c) for c in chars] + ["<|im_end|>", "<|im_start|>"]
    m = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen35",
         "tokenizer.ggml.tokens": tokens, "tokenizer.ggml.token_type": [1] * 256 + [3, 3],
         "tokenizer.ggml.merges": [], "tokenizer.ggml.eos_token_id": 256,
         "tokenizer.ggml.bos_token_id": 256, "tokenizer.ggml.add_bos_token": False,
         "tokenizer.chat_template": "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"}
    params = {"embedding_length": 32, "block_count": 4, "hyper_connection.count": 2,
        "hyper_connection.low_rank": 8, "attention.layer_norm_rms_epsilon": 1e-6,
        "attention.head_count": 2, "attention.head_count_kv": 1, "attention.key_length": 16,
        "attention.value_length": 16, "ssm.state_size": 8, "ssm.group_count": 1,
        "ssm.time_step_rank": 2, "ssm.inner_size": 16, "ssm.conv_kernel": 4,
        "expert_count": 4, "expert_used_count": 2, "expert_feed_forward_length": 16,
        "expert_shared_feed_forward_length": 16, "rope.dimension_count": 8,
        "rope.dimension_sections": [2, 1, 1, 0], "rope.freq_base": 10000.0,
        "attention.indexer.head_count": 2, "attention.indexer.key_length": 8,
        "attention.indexer.top_k": 2, "attention.compress_ratios": [0, 2, 0, 2],
        "full_attention_interval": 2, "context_length": 128,
        "ple.layers": [0], "ple.ngram_size": 3, "ple.heads_per_ngram": 2,
        "ple.conv_kernel": 3, "ple.eos_token_id": 256, "embedding_length_per_layer_input": 8,
        "ple.layer_multipliers": [23703573157769, 20109073645365, 8052911324071],
        "ple.head_offsets": [0, 23, 52, 83], "ple.head_vocab_sizes": [23, 29, 31, 37]}
    m.update({"qwen4exp." + k: v for k, v in params.items()})
    rng = np.random.default_rng(38)
    tensors = []
    def add(name, shape):
        data = rng.normal(0, 0.08, shape).astype(np.float32)
        if "norm" in name:
            data += 1
        if name.endswith("ssm_a"):
            data = -np.exp(data)
        tensors.append((name, shape, 0, data.tobytes()))
    add("token_embd.weight", (258, 32))
    add("output.weight", (258, 32))
    add("output_hc_norm.weight", (64,))
    add("output_hc_down.weight", (8, 64))
    add("output_hc_up.weight", (64, 8))
    add("per_layer_token_embd.weight", (120, 8))
    for i in range(4):
        p = f"blk.{i}."
        for module in ("attn", "ffn"):
            for suffix, shape in (("norm", (64,)), ("down", (8, 64)), ("up", (64, 8)), ("inject", (2, 64))):
                add(p + f"hc_{module}_{suffix}.weight", shape)
        for proj, shape in (("gate", (4, 16, 32)), ("up", (4, 16, 32)), ("down", (4, 32, 16))):
            add(p + f"ffn_{proj}_exps.weight", shape)
        for proj, shape in (("gate", (16, 32)), ("up", (16, 32)), ("down", (32, 16))):
            add(p + f"ffn_{proj}_shexp.weight", shape)
        add(p + "ffn_gate_inp.weight", (4, 32))
        add(p + "ffn_gate_inp_shexp.weight", (32,))
        if i % 2 == 0:
            for name, shape in {"attn_qkv.weight": (32, 32), "attn_gate.weight": (16, 32),
                "ssm_conv1d.weight": (32, 4), "ssm_a": (2,), "ssm_dt.bias": (2,),
                "ssm_alpha.weight": (2, 32), "ssm_beta.weight": (2, 32), "ssm_norm.weight": (8,),
                "ssm_out.weight": (32, 16)}.items():
                add(p + name, shape)
        else:
            for name, shape in {"attn_q.weight": (64, 32), "attn_k.weight": (16, 32), "attn_v.weight": (16, 32),
                "attn_output.weight": (32, 32), "attn_q_norm.weight": (16,), "attn_k_norm.weight": (16,),
                "indexer.q_proj.weight": (16, 32), "indexer.k_proj.weight": (8, 32),
                "indexer.q_norm.weight": (8,), "indexer.k_norm.weight": (8,)}.items():
                add(p + name, shape)
        if i == 0:
            for name, shape in {"key": (64, 32), "value": (32, 32), "norm_key": (64,),
                "norm_query": (64,), "norm_conv": (64,), "conv1d": (64, 3)}.items():
                add(p + "ple_" + name + ".weight", shape)
    return write_gguf(path, tensors, m)


def tiny_qwen35(path):
    from edge0.checkpoints.gguf import GGUFSource
    source_path = tiny_qwen4exp(path.with_suffix(".qwen4.gguf"))
    with GGUFSource(source_path) as source:
        metadata = {k.replace("qwen4exp.", "qwen35moe."): v for k, v in source.metadata.items()
                    if not any(part in k for part in ("hyper_connection", "indexer", "compress_ratios", "ple.", "embedding_length_per_layer_input"))}
        metadata["general.architecture"] = "qwen35moe"
        tensors = [(n, t.shape, t.encoding, source.read_bytes(n, 0, t.nbytes)) for n, t in source.tensors.items()
                   if not any(part in n for part in ("hc_", "ple_", "indexer", "per_layer_token"))]
    for name in ["output_norm.weight"] + [f"blk.{i}.{part}.weight" for i in range(4) for part in ("attn_norm", "post_attention_norm")]:
        tensors.append((name, (32,), 0, np.ones(32, dtype=np.float32).tobytes()))
    return write_gguf(path, tensors, metadata)
