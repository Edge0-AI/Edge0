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


CASES = {name[len("case_"):]: fn for name, fn in globals().items()
         if name.startswith("case_")}


def main(argv):
    case, inp_path, out_path = argv[1:4]
    inp = dict(np.load(inp_path))
    np.savez(out_path, **CASES[case](inp))


if __name__ == "__main__":
    main(sys.argv)
