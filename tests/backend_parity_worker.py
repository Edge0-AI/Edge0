"""Run one shared-code case under whichever backend EDGE0_BACKEND selects.

Invoked by tests/test_backend_parity.py as a subprocess, once per backend,
so the same framework code (routing, sampling, streaming) is exercised on
MLX and on the torch backend with identical inputs:

    EDGE0_BACKEND=cuda python tests/backend_parity_worker.py CASE IN.npz OUT.npz

Inputs and outputs are numpy arrays in .npz files; bf16 never crosses the
boundary (cases upcast to float32 before returning).
"""

from __future__ import annotations

import sys

import numpy as np

from edge0.backends import core


# .tolist() is an array method on both backends (mlx.core has no tolist
# function, despite the contract listing one).
def _np(a):
    return np.array(core.astype(a, core.float32).tolist(), dtype=np.float32)


def _np_int(a):
    return np.array(a.tolist(), dtype=np.int64)


def _sorted_pair(inds, scores):
    """Router outputs as (indices, scores) sorted by expert id per row --
    argpartition leaves the order within the top-k unspecified."""
    i, s = _np_int(inds), _np(scores)
    order = np.argsort(i, axis=-1)
    return np.take_along_axis(i, order, -1), np.take_along_axis(s, order, -1)


def case_select_from_logits(inp):
    from edge0.moe.routing import select_from_logits
    inds, scores = select_from_logits(core.array(inp["logits"]), top_k=4)
    i, s = _sorted_pair(inds, scores)
    return {"inds": i, "scores": s}


def case_group_select_from_logits(inp):
    from edge0.moe.routing import group_select_from_logits
    inds, scores = group_select_from_logits(
        core.array(inp["logits"]), top_k=8, n_group=8, topk_group=4,
        routed_scaling=2.5, expert_bias=core.array(inp["expert_bias"]))
    i, s = _sorted_pair(inds, scores)
    return {"inds": i, "scores": s}


def case_mask_logits(inp):
    from edge0.sampling import _mask_logits
    out = {}
    for name, kw in {
        "topk": dict(temperature=0.7, top_k=20, top_p=None),
        "topp": dict(temperature=0.9, top_k=None, top_p=0.8),
        "both": dict(temperature=1.0, top_k=50, top_p=0.9),
        "greedy": dict(temperature=0.0, top_k=None, top_p=None),
    }.items():
        out[name] = _np(_mask_logits(core.array(inp["logits"]), **kw))
    return out


def _streaming_layer(model_dir, **opts):
    """StreamingSwitchGLU over layer 1 of the real edge0-8b checkpoint."""
    import os
    from edge0.moe.spec import MoESpec, QuantSpec, RouterKind, WeightLayout
    from edge0.streaming.layer import StreamingSwitchGLU
    from edge0.streaming.mmap import SafetensorsMmap
    from edge0.streaming.options import LayerOptions
    spec = MoESpec(
        num_experts=128, top_k=8, intermediate_size=512,
        router=RouterKind.SIGMOID_GROUP, norm_topk_prob=True,
        routed_scaling=2.5, n_group=8, topk_group=4, shared_experts=1,
        quant=QuantSpec(bits=4, group_size=64, mode="affine"),
        layout=WeightLayout.SEPARATE,
        key_template="model.layers.{layer}.mlp.experts",
        block_path="model.layers.{layer}.mlp",
        layer_path="model.layers.{layer}")
    shards = [SafetensorsMmap(os.path.join(model_dir, "model.safetensors"))]
    return StreamingSwitchGLU(shards, 1, spec, options=LayerOptions(**opts))


def case_streaming(inp):
    """Every StreamingSwitchGLU.__call__ path on real expert weights."""
    model_dir = str(inp["model_dir"])
    x1 = core.astype(core.array(inp["x1"]), core.bfloat16)      # [1, H]
    x16 = core.astype(core.array(inp["x16"]), core.bfloat16)    # [16, H]
    i1 = core.array(inp["i1"], dtype=core.int32)                # [1, 8]
    i16 = core.array(inp["i16"], dtype=core.int32)              # [16, 8]
    out = {}
    for compiled in (True, False):
        tag = "c" if compiled else "e"
        layer = _streaming_layer(model_dir, use_compile=compiled)
        out[f"exact_t1_{tag}"] = _np(layer(x1, i1))             # unsorted
        out[f"exact_t16_{tag}"] = _np(layer(x16, i16))          # sorted
        layer.load_full_layer()
        out[f"full_t16_{tag}"] = _np(layer(x16, i16))
        layer.clear_full_layer()
        # Hot stack holding every even expert: odd ones are misses and go
        # through the exact scatter-add correction (core.index_add).
        layer._hot_counts = {e: 1.0 for e in range(0, 128, 2)}
        layer.load_hot_layer(n_hot=64)
        layer.materialize_hot()
        out[f"hot_t16_{tag}"] = _np(layer(x16, i16))
        layer.clear_hot_layer()
        layer.close()
        # Staged decode over a partial set: experts outside it are dropped.
        staged = _streaming_layer(model_dir, use_compile=compiled,
                                  staged=True, staged_n=8, staged_trigger=8)
        staged.stage_experts([int(e) for e in inp["staged_set"]])
        staged.wait_staged()
        out[f"staged_t1_{tag}"] = _np(staged(x1, i1))
        staged.close()
    return out


def case_install_qwen35_tiny(inp):
    """torch backend only: a small transformers Qwen3_5MoeForCausalLM whose
    experts are MLX-quantized into a real safetensors shard (named like the
    edge0-35b checkpoint), streamed in with install_streaming_experts and
    compared against the same model running its own dense experts."""
    import dataclasses

    import mlx.core as mx
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    from edge0.backends.cuda.model_specs import QWEN35_MOE_SPEC
    from edge0.backends.cuda.moe_blocks import TransformersExpertsAdapter
    from edge0.streaming.install import install_streaming_experts
    from edge0.streaming.mmap import SafetensorsMmap

    E, K, H, I = 8, 2, 128, 64
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=128, hidden_size=H, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        moe_intermediate_size=I, shared_expert_intermediate_size=I,
        num_experts=E, num_experts_per_tok=K,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16)
    torch.manual_seed(0)
    model = Qwen3_5MoeForCausalLM(cfg).eval()

    def quantize(w):
        """MLX 4-bit affine, scales/biases rounded to bf16 as on disk;
        returns the on-disk tensors and the weight they dequantize to."""
        wq, s, b = mx.quantize(mx.array(w.detach().numpy()), group_size=64,
                               bits=4)
        s, b = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
        deq = mx.dequantize(wq, s, b, group_size=64, bits=4).astype(
            mx.float32)
        bits16 = lambda a: torch.from_numpy(
            np.array(a.view(mx.uint16))).view(torch.bfloat16)
        return ({"weight": torch.from_numpy(np.array(wq)),
                 "scales": bits16(s), "biases": bits16(b)},
                torch.from_numpy(np.array(deq)))

    tensors = {}
    for li in range(cfg.num_hidden_layers):
        ex = model.model.layers[li].mlp.experts
        prefix = f"language_model.model.layers.{li}.mlp.switch_mlp"
        deq = {}
        for proj, w in (("gate_proj", ex.gate_up_proj[:, :I]),
                        ("up_proj", ex.gate_up_proj[:, I:]),
                        ("down_proj", ex.down_proj)):
            disk, deq[proj] = quantize(w)
            for part, t in disk.items():
                tensors[f"{prefix}.{proj}.{part}"] = t.contiguous()
        # the dense reference runs on exactly what the shard encodes
        with torch.no_grad():
            ex.gate_up_proj.copy_(torch.cat([deq["gate_proj"],
                                             deq["up_proj"]], dim=1))
            ex.down_proj.copy_(deq["down_proj"])
    shard_path = str(inp["shard_path"])
    save_file(tensors, shard_path)

    spec = dataclasses.replace(QWEN35_MOE_SPEC, num_experts=E, top_k=K,
                               intermediate_size=I)
    out = {}
    ids = {name: torch.from_numpy(inp[name].astype(np.int64))
           for name in ("ids6", "ids40")}   # 12 pairs: unsorted; 80: sorted
    with torch.no_grad():
        for name, x in ids.items():
            out[f"ref_{name}"] = model(x).logits.float().numpy()
        twins = install_streaming_experts(
            model, [SafetensorsMmap(shard_path)], spec,
            wrap=TransformersExpertsAdapter)
        out["n_twins"] = np.array(sum(t is not None for t in twins))
        out["experts_type"] = np.array(
            type(model.model.layers[0].mlp.experts).__name__)
        for name, x in ids.items():
            out[f"got_{name}"] = model(x).logits.float().numpy()
    for t in twins:
        t.close()
    return out


CASES = {name[len("case_"):]: fn for name, fn in globals().items()
         if name.startswith("case_")}


def main(argv):
    case, inp_path, out_path = argv[1:4]
    inp = dict(np.load(inp_path))
    np.savez(out_path, **CASES[case](inp))


if __name__ == "__main__":
    main(sys.argv)
