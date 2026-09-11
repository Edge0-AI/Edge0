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


def case_bailing_port_parity(inp):
    """torch backend only: the torch port of bailing_hybrid against the
    vendored MLX model (run on the MLX CPU device -- full-precision float32;
    some Apple GPUs run float32 matmul at ~7e-4), both loaded from the real
    edge0-8b checkpoint, float32. Every torch layer is fed MLX's input to
    that layer (per-layer error in isolation), and separately the whole
    torch model runs free on the token ids with its own caches. A chunked
    prefill (second chunk over a non-empty cache) and decode steps follow.
    """
    import mlx.core as mx
    import torch

    from edge0.backends.cuda._impl import bailing_hybrid as tb
    from edge0.backends.cuda.io import load_model as t_load
    from edge0.backends.cuda.model_specs import BAILING_V3_MOE_SPEC
    from edge0.backends.mlx._impl import bailing_hybrid as mb
    from edge0.backends.mlx.io import load_model as m_load
    from edge0.streaming.install import install_streaming_experts
    from edge0.streaming.mmap import SafetensorsMmap

    path = str(inp["model_dir"])
    over = {"model_type": "bailing_hybrid"}
    mx.set_default_device(mx.cpu)
    mm, _ = m_load(path, lazy=False, strict=False, model_config=over,
                   get_model_classes=lambda config: (mb.Model, mb.ModelArgs))
    mm.set_dtype(mx.float32)
    tm, _ = t_load(path, strict=False, model_config=over,
                   get_model_classes=lambda config: (tb.Model, tb.ModelArgs),
                   dtype=torch.float32)
    rep = tm._edge0_load_report
    twins = install_streaming_experts(
        tm, [SafetensorsMmap(f"{path}/model.safetensors")],
        BAILING_V3_MOE_SPEC, num_layers=len(tm.model.layers))

    def f32(a):
        return np.array(a.astype(mx.float32)) if isinstance(a, mx.array) \
            else a.detach().float().numpy()

    def rel(a, b):
        a, b = f32(a), f32(b)
        return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))

    ids = [int(i) for i in inp["ids"]]
    steps = [ids[:7], ids[7:]] + [None] * int(inp["n_decode"])
    n = len(tm.model.layers)
    m_cache, t_cache, t_free = mm.make_cache(), tm.make_cache(), tm.make_cache()
    layer_err = np.zeros((len(steps), n))
    logit_err, m_arg, t_arg = [], [], []
    for s, chunk in enumerate(steps):
        chunk = chunk if chunk is not None else [m_arg[-1]]
        h = mm.model.word_embeddings(mx.array(chunk)[None])
        t_emb = tm.model.word_embeddings(torch.tensor(chunk)[None])
        m_mask = mb.create_attention_mask(h, m_cache[mm.model.first_mla_idx])
        t_mask = tb.create_attention_mask(t_emb, t_cache[tm.model.first_mla_idx])
        with torch.no_grad():
            for li in range(n):
                ml, tl = mm.model.layers[li], tm.model.layers[li]
                x_in = h
                h = ml(x_in, m_mask if ml.is_mla else None, m_cache[li], None)
                mx.eval(h)
                t_out = tl(torch.from_numpy(f32(x_in)),
                           t_mask if tl.is_mla else None, t_cache[li], None)
                layer_err[s, li] = rel(t_out, h)
            mo = mm.lm_head(mm.model.norm(h))[0, -1]
            to = tm.lm_head(tm.model(torch.tensor(chunk)[None], cache=t_free))[0, -1]
        logit_err.append(rel(to, mo))
        m_arg.append(int(mx.argmax(mo).item()))
        t_arg.append(int(torch.argmax(to).item()))
    for t in twins:
        if t is not None:
            t.close()
    return {
        "layer_err": layer_err, "logit_err": np.array(logit_err),
        "m_argmax": np.array(m_arg), "t_argmax": np.array(t_arg),
        "is_mla": np.array([l.is_mla for l in tm.model.layers]),
        "n_missing": np.array(len(rep["missing"])),
        "unexpected": np.array(sorted({k.split(".")[3] + "." + k.split(".")[4]
                                       for k in rep["unexpected"]}), dtype=str),
    }


def case_engine_guard(inp):
    """Every entry point imports without pulling MLX in, and the MLX-only
    engines refuse another backend up front."""
    import importlib
    for mod in ("edge0", "edge0.engine", "edge0.engine.qwen",
                "edge0.engine.ling", "edge0.cli", "edge0.prerouter.install",
                "edge0.adapters.lora", "edge0.streaming.install"):
        importlib.import_module(mod)
    loaded_mlx = sorted(m for m in sys.modules
                        if m == "mlx" or m.startswith(("mlx.", "mlx_lm")))
    errors = []
    for mod in ("edge0.engine.qwen", "edge0.engine.ling"):
        try:
            importlib.import_module(mod).load_installed("unused", None)
            errors.append("no error")
        except NotImplementedError as e:
            errors.append(str(e))
    return {"loaded_mlx": np.array(loaded_mlx, dtype=str),
            "errors": np.array(errors, dtype=str)}


def _tiny_qwen35():
    """A 4-layer transformers Qwen3.5-MoE (3 linear-attention layers, 1 full)
    with every quantizable width a multiple of 64. Returns (config, model)."""
    import torch
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=128, hidden_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        moe_intermediate_size=64, shared_expert_intermediate_size=64,
        num_experts=8, num_experts_per_tok=2,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16)
    torch.manual_seed(0)
    return cfg, Qwen3_5MoeForCausalLM(cfg).eval()


def _mlx_quantize(w, bits=4):
    """MLX affine quantization (group 64), scales/biases rounded to bf16 as
    on disk. Returns the on-disk tensors and the float32 weight they
    dequantize to."""
    import mlx.core as mx
    import torch
    wq, s, b = mx.quantize(mx.array(w.detach().float().numpy()),
                           group_size=64, bits=bits)
    s, b = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
    # bf16-valued scales, float32 arithmetic: with bf16 scales MLX would
    # also round the dequantized weight to bf16
    deq = mx.dequantize(wq, s.astype(mx.float32), b.astype(mx.float32),
                        group_size=64, bits=bits)

    def bits16(a):
        return torch.from_numpy(np.array(a.view(mx.uint16))).view(torch.bfloat16)
    return ({"weight": torch.from_numpy(np.array(wq)),
             "scales": bits16(s), "biases": bits16(b)},
            torch.from_numpy(np.array(deq)))


def case_load_model_qwen35_tiny(inp):
    """torch backend only: write a small Qwen3.5-MoE checkpoint in the exact
    on-disk format of the published edge0-35b -- ConditionalGeneration
    config with text_config, language_model. prefix, every Linear and
    Embedding 4-bit, the router 8-bit, experts as switch_mlp, norms stored
    as w + 1, conv1d in MLX [C, k, 1] layout, the rest bf16 -- then
    load_model + install_streaming_experts it and compare logits with the
    source model running on the same effective weights."""
    import dataclasses
    import json
    import os

    import torch
    from safetensors.torch import save_file

    from edge0.backends.cuda import nn as cnn
    from edge0.backends.cuda.io import _QWEN35_SHIFTED_NORMS, load_model
    from edge0.backends.cuda.model_specs import QWEN35_MOE_SPEC
    from edge0.backends.cuda.moe_blocks import TransformersExpertsAdapter
    from edge0.streaming.install import install_streaming_experts
    from edge0.streaming.mmap import SafetensorsMmap

    cfg, ref = _tiny_qwen35()
    I = cfg.moe_intermediate_size
    ckpt = str(inp["ckpt_dir"])
    os.makedirs(ckpt, exist_ok=True)
    P = "language_model."
    disk, dense = {}, {}          # on-disk tensors; reference weights

    def put(name, t):
        disk[P + name] = t.contiguous()

    for name, t in ref.state_dict().items():
        t = t.detach().float()
        mod_path, _, leaf = name.rpartition(".")
        mod = ref.get_submodule(mod_path)
        if name.endswith("mlp.experts.gate_up_proj"):
            base = mod_path.replace(".experts", ".switch_mlp")
            g, dg = _mlx_quantize(t[:, :I])
            u, du = _mlx_quantize(t[:, I:])
            for proj, q in (("gate_proj", g), ("up_proj", u)):
                for part, v in q.items():
                    put(f"{base}.{proj}.{part}", v)
            dense[name] = torch.cat([dg, du], dim=1)
        elif name.endswith("mlp.experts.down_proj"):
            base = mod_path.replace(".experts", ".switch_mlp")
            q, dense[name] = _mlx_quantize(t)
            for part, v in q.items():
                put(f"{base}.down_proj.{part}", v)
        elif leaf == "weight" and (
                isinstance(mod, (torch.nn.Linear, torch.nn.Embedding))
                or name.endswith("mlp.gate.weight")):
            bits = 8 if name.endswith("mlp.gate.weight") else 4
            q, dense[name] = _mlx_quantize(t, bits=bits)
            for part, v in q.items():
                put(f"{mod_path}.{part}", v)
        else:
            stored = t.to(torch.bfloat16)
            dense[name] = stored.float()
            if name.endswith(_QWEN35_SHIFTED_NORMS):
                stored = (stored.float() + 1.0).to(torch.bfloat16)
            if name.endswith("conv1d.weight"):
                stored = stored.transpose(1, 2)          # MLX [C, k, 1]
            put(name, stored)
    save_file(disk, os.path.join(ckpt, "model.safetensors"))
    with open(os.path.join(ckpt, "config.json"), "w") as f:
        json.dump({"architectures": ["Qwen3_5MoeForConditionalGeneration"],
                   "text_config": json.loads(cfg.to_json_string()),
                   "quantization": {"group_size": 64, "bits": 4,
                                    "mode": "affine"}}, f)
    ref.load_state_dict(dense)

    got = load_model(ckpt, strict=True, dtype=torch.float32)
    spec = dataclasses.replace(QWEN35_MOE_SPEC, num_experts=cfg.num_experts,
                               top_k=cfg.num_experts_per_tok,
                               intermediate_size=I)
    twins = install_streaming_experts(
        got, [SafetensorsMmap(os.path.join(ckpt, "model.safetensors"))],
        spec, wrap=TransformersExpertsAdapter)
    out = {
        "embed_type": np.array(type(got.model.embed_tokens).__name__),
        "q_proj_type": np.array(
            type(got.model.layers[3].self_attn.q_proj).__name__),
        "lm_head_type": np.array(type(got.lm_head).__name__),
        "router_dtype": np.array(str(got.model.layers[0].mlp.gate.weight.dtype)),
        "router_bits": np.array(0),
        "n_meta_params": np.array(sum(
            p.is_meta for n, p in got.named_parameters())),
        "n_quantized": np.array(sum(
            isinstance(m, (cnn.QuantizedLinear, cnn.QuantizedEmbedding))
            for m in got.modules())),
    }
    with torch.no_grad():
        for key in ("ids6", "ids40"):
            x = torch.from_numpy(inp[key].astype(np.int64))
            out[f"ref_{key}"] = ref(x).logits.float().numpy()
            out[f"got_{key}"] = got(x).logits.float().numpy()
    for t in twins:
        t.close()
    return out


def case_install_qwen35_tiny(inp):
    """torch backend only: a small transformers Qwen3_5MoeForCausalLM whose
    experts are MLX-quantized into a real safetensors shard (named like the
    edge0-35b checkpoint), streamed in with install_streaming_experts and
    compared against the same model running its own dense experts."""
    import dataclasses

    import torch
    from safetensors.torch import save_file

    from edge0.backends.cuda.model_specs import QWEN35_MOE_SPEC
    from edge0.backends.cuda.moe_blocks import TransformersExpertsAdapter
    from edge0.streaming.install import install_streaming_experts
    from edge0.streaming.mmap import SafetensorsMmap

    cfg, model = _tiny_qwen35()
    E, K, I = cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size
    quantize = _mlx_quantize

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
