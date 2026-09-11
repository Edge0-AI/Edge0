"""CUDA backend: model / tokenizer / tensor-store loading.

Contract (documented): ``load_model, load_tokenizer, open_tensor_store``.
Two more names are used in practice by call sites that import
``edge0.backends.mlx.io`` DIRECTLY instead of going through the generic
facade -- ``load_safetensors`` (``prerouter/install.py``,
``adapters/lora.py``) and ``open_shards`` (``engine/qwen.py``). Those
call sites need editing to import from ``edge0.backends.io`` once a
backend is selected; until then this module still implements both so
the CUDA-side equivalents exist and are testable on their own.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import torch

from edge0.backends.base import TensorStore
from edge0.backends.cuda.core import DEVICE, _as_tensor

_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool,
}
_NP_DTYPES = {  # for the raw numpy view before the device transfer
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
    "U32": np.uint32,  # MLX-quantized payloads (packed codes)
}


class SafeTensorsStore(TensorStore):
    """mmap-backed safetensors store; ``get`` returns a device tensor.

    Structurally identical to ``backends/mlx/io.py::SafeTensorsStore``
    (same header parsing) -- the only change is the tail of ``get``:
    numpy view -> device tensor instead of numpy view -> mx.array. BF16
    has no numpy dtype on most builds, so it is read as raw uint16 and
    bit-cast via torch (``.view(torch.bfloat16)``), same trick the MLX
    version uses via mlx's native bfloat16.
    """

    def __init__(self, path: str):
        import mmap
        self._path = path
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_len)
        header = json.loads(header_bytes)
        self._metadata = header.pop("__metadata__", {})
        self._entries = {
            name: {
                "offset": meta["data_offsets"][0] + 8 + header_len,
                "size": meta["data_offsets"][1] - meta["data_offsets"][0],
                "dtype": meta["dtype"],
                "shape": tuple(meta["shape"]),
            }
            for name, meta in header.items()
        }
        self._keys = sorted(self._entries)
        self._file = open(path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    @property
    def path(self) -> str:
        return self._path

    def keys(self) -> list[str]:
        return list(self._keys)

    def metadata(self) -> dict:
        return dict(self._metadata)

    def get(self, name: str):
        e = self._entries.get(name)
        if e is None:
            raise KeyError(f"{self.path}: no tensor {name!r}")
        buf = np.frombuffer(self._mm, dtype=np.uint8, count=e["size"],
                             offset=e["offset"])
        if e["dtype"] == "BF16":
            u16 = np.frombuffer(buf, dtype=np.uint16,
                                 count=int(np.prod(e["shape"])))
            t = torch.from_numpy(u16.copy()).view(torch.bfloat16)
            t = t.reshape(e["shape"])
        else:
            arr = np.frombuffer(buf, dtype=_NP_DTYPES[e["dtype"]],
                                 count=int(np.prod(e["shape"])))
            t = torch.from_numpy(arr.copy()).reshape(e["shape"])
        return t.to(DEVICE, non_blocking=True)

    def close(self):
        self._mm.close()
        self._file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def open_shards(model_dir: str) -> list:
    """Same contract as ``backends/mlx/io.py::open_shards`` -- reuses
    ``streaming.mmap.SafetensorsMmap`` as-is, since that module is
    already backend-agnostic (pure ``mmap`` + ``numpy``)."""
    import glob
    import os
    from edge0.streaming.mmap import SafetensorsMmap
    shards = []
    for path in sorted(glob.glob(os.path.join(
            os.fspath(model_dir), "model*.safetensors"))):
        shards.append(SafetensorsMmap(path))
    if not shards:
        raise FileNotFoundError(
            f"no model*.safetensors shards under {model_dir}")
    return shards


def load_safetensors(path: str, dtype=None) -> dict:
    """Load every tensor of a (small) safetensors file as device tensors
    (adapter / prerouter weight files)."""
    store = SafeTensorsStore(path)
    try:
        out = {}
        for name in store.keys():
            t = store.get(name)
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype)
            out[name] = t
        return out
    finally:
        store.close()


def load_tokenizer(model_path):
    """Identical to the MLX backend's implementation -- this already
    goes through ``transformers.AutoTokenizer`` with no MLX dependency,
    so it is genuinely backend-agnostic; duplicated here rather than
    imported cross-backend so ``edge0.backends.cuda`` never imports
    ``edge0.backends.mlx`` (keeps the two backends independently
    installable)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True)


_EXPERT_KEY_MARKERS = (".mlp.experts.", ".switch_mlp.")
"""Tensor-name substrings for the quantized MoE expert weights (the ones
streaming/layer.py streams from disk on demand — never meant to be
resident). Matches both tiers' REAL on-disk key naming, checked against
the actual published checkpoints (safetensors index/header), not
assumed: edge0-35b uses ``...mlp.switch_mlp.{gate,up,down}_proj``,
edge0-8b uses ``...mlp.experts.{gate,up,down}_proj`` — both already
pre-stacked ``[num_experts, ...]`` tensors (``WeightLayout.SEPARATE``),
regardless of what either tier's REFERENCE implementation does with
experts in memory (transformers' Bailing class uses a per-expert
``nn.ModuleList`` internally, but that in-memory shape is irrelevant
here since these keys are never loaded into it — see ``load_model``).
"""

_KEY_PREFIX_STRIP = {
    # architectures-string -> checkpoint key prefix to strip before
    # matching transformers' own state_dict names. Confirmed against
    # the REAL published checkpoints' safetensors index/header:
    # edge0-35b's keys all carry "language_model." (MLX's own
    # post-sanitize naming, from _impl/qwen3_5_moe.py wrapping
    # TextModel under self.language_model) -- transformers'
    # Qwen3_5MoeForCausalLM has no such wrapper, so every dense weight
    # would silently fail to match without this. edge0-8b's keys
    # already match transformers.BailingMoeV3's own naming with no
    # prefix at all -- confirmed via the same check, not assumed
    # identical by default (see _resolve_model_class's docstring).
    "Qwen3_5MoeForConditionalGeneration": "language_model.",
}


def _resolve_model_class(model_path, raw_config: dict):
    """``config.json`` (+ directory, for trust_remote_code) -> a
    ``(constructed transformers config, model_cls, needs_trust_remote_code,
    key_prefix)`` tuple.

    Dispatches on ``architectures``/``auto_map``, NOT ``model_type`` --
    checked against the real ``Edge0/Edge0-8B-A1B-preview`` config.json
    (downloaded directly, not assumed): it has no top-level
    ``model_type`` field at all. A first version of this function keyed
    on ``model_type`` would have silently mis-dispatched on it.

    * edge0-35b: the REAL published config.json's ``architectures`` is
      ``["Qwen3_5MoeForConditionalGeneration"]`` -- the vision+text
      wrapper, not plain ``Qwen3_5MoeForCausalLM`` (a first version of
      this function assumed the latter). Checked directly: it has both
      ``text_config`` AND ``vision_config`` keys. edge0 strips vision
      entirely (confirmed in ``_impl/qwen3_5_moe.py::sanitize()``), so
      this still resolves to ``Qwen3_5MoeForCausalLM`` (config class
      ``Qwen3_5MoeTextConfig``) built from JUST ``config["text_config"]``
      -- the vision-wrapping class is never constructed. Every
      checkpoint key carries a ``language_model.`` prefix (confirmed via
      the real safetensors index) that the plain text-only class's own
      attribute names don't have -- stripped via ``_KEY_PREFIX_STRIP``.
    * edge0-8b (``BailingMoeV3ForCausalLM``): ships its OWN
      ``modeling_bailing_moe_v3.py``/``configuration_bailing_moe_v3.py``
      co-located in the checkpoint repo (confirmed via the HF API file
      listing, then downloaded and actually imported -- needs einops,
      fla, triton as extra deps, not currently in edge0's own
      dependency list). Experts are ``nn.ModuleList`` of per-expert MLP
      modules in the reference implementation's OWN in-memory layout,
      but the actual PUBLISHED CHECKPOINT stores them pre-stacked
      (confirmed via the real safetensors header) -- the
      ``nn.ModuleList`` is never populated from checkpoint data, it
      gets swapped out by the streaming installer instead. No key
      prefix needed -- confirmed via the same check, not assumed
      identical to the 35b tier by default.
      SECURITY NOTE, not a footnote: ``trust_remote_code=True`` runs
      third-party Python shipped inside the checkpoint directory. That
      is a real code-execution surface, not a formality -- worth a
      deliberate decision (pin+review the exact modeling file once,
      vendor it, or accept the risk per-checkpoint) before this path
      is used unattended, e.g. in `edge0 serve`.
    """
    archs = raw_config.get("architectures", [])
    auto_map = raw_config.get("auto_map", {})

    if any(a.startswith("Qwen3_5Moe") for a in archs):
        from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
        text_cfg_dict = raw_config.get("text_config", raw_config)
        config = Qwen3_5MoeTextConfig(**text_cfg_dict)
        key_prefix = next(
            (p for a, p in _KEY_PREFIX_STRIP.items() if a in archs), "")
        return config, Qwen3_5MoeForCausalLM, False, key_prefix

    if any("Bailing" in a for a in archs) or "bailing" in str(auto_map).lower():
        from transformers import AutoConfig, AutoModelForCausalLM
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return config, AutoModelForCausalLM, True, ""

    raise NotImplementedError(
        f"cuda backend: unrecognized architectures={archs!r} -- no known "
        f"transformers class for it (see this function's docstring for "
        f"the two paths currently resolved)")


_QWEN35_SHIFTED_NORMS = (
    ".input_layernorm.weight", ".post_attention_layernorm.weight",
    "model.norm.weight", ".q_norm.weight", ".k_norm.weight")
"""Norms that edge0's MLX ``qwen3_5.py::sanitize()`` stores as ``w + 1``:
transformers' Qwen3.5 RMSNorm computes ``x * (1 + w)``, the MLX one
``x * w``. Checked on the published edge0-35b bytes: input_layernorm mean
1.03, q_norm 1.33, model.norm 2.63 -- versus ~0 for a transformers
checkpoint. The gated ``linear_attn.norm`` is not shifted (mean 0.88)."""


def _undo_mlx_qwen35_sanitize(state: dict) -> None:
    """Invert edge0's MLX Qwen3.5 sanitize on a loaded state dict, in
    place -- only if the checkpoint went through it, detected the way the
    sanitize itself detects the opposite: conv1d weights in MLX layout
    ``[C, k, 1]`` instead of torch's ``[C, 1, k]``."""
    convs = [k for k in state if k.endswith("conv1d.weight")]
    if not convs or not all(state[k].shape[-1] == 1 and state[k].shape[1] != 1
                            for k in convs):
        return
    for k in convs:
        state[k] = state[k].transpose(1, 2).contiguous()
    for k in list(state):
        if k.endswith(_QWEN35_SHIFTED_NORMS) and state[k].ndim == 1:
            state[k] = state[k] - 1.0


def _read_tensor(shard, meta, name):
    import torch
    raw = shard.raw(name)
    count = int(np.prod(meta["shape"]))
    if meta["dtype"] == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16, count=count)
        t = torch.from_numpy(u16.copy()).view(torch.bfloat16)
    else:
        arr = np.frombuffer(raw, dtype=_NP_DTYPES[meta["dtype"]], count=count)
        t = torch.from_numpy(arr.copy())
    return t.reshape(meta["shape"])


def _params_on_meta():
    """Build a module with its PARAMETERS on the meta device (no memory)
    but buffers real: non-persistent buffers such as rotary ``inv_freq``
    are computed at init and never come from the checkpoint, so building
    everything under ``torch.device('meta')`` would leave them unusable."""
    import contextlib

    import torch

    @contextlib.contextmanager
    def ctx():
        orig = torch.nn.Module.register_parameter

        def register_parameter(self, name, param):
            if param is not None and param.device.type != "meta":
                param = torch.nn.Parameter(param.to("meta"),
                                           requires_grad=param.requires_grad)
            orig(self, name, param)

        torch.nn.Module.register_parameter = register_parameter
        try:
            yield
        finally:
            torch.nn.Module.register_parameter = orig
    return ctx()


def _install_quantized(model, state: dict, dtype) -> set:
    """Replace every module whose weight is MLX-quantized in ``state``
    (``<path>.scales`` present) with the torch equivalent, consuming the
    three tensors. Linear / Embedding get quantized modules; anything else
    (e.g. transformers' Qwen3.5 router, a custom module holding a plain
    ``weight`` parameter) gets its weight dequantized in place. Returns the
    module paths replaced (their tensors are already in place)."""
    import torch

    from edge0.backends.cuda import nn as cnn
    from edge0.backends.cuda.quant import _dequantize

    replaced = set()
    for skey in [k for k in state if k.endswith(".scales")]:
        path = skey[: -len(".scales")]
        try:
            mod = model.get_submodule(path)
        except AttributeError:
            continue                      # not part of this model: stays unexpected
        w = state.pop(f"{path}.weight")
        s = state.pop(skey)
        b = state.pop(f"{path}.biases")
        owner, _, attr = path.rpartition(".")
        parent = model.get_submodule(owner) if owner else model
        if isinstance(mod, torch.nn.Linear):
            bias = state.pop(f"{path}.bias", None)
            new = cnn.QuantizedLinear(w, s, b, mod.in_features, bias=bias)
        elif isinstance(mod, torch.nn.Embedding):
            new = cnn.QuantizedEmbedding(w, s, b, mod.embedding_dim,
                                         dtype=dtype)
        else:
            in_features = mod.weight.shape[-1]
            bits, group_size = cnn._quant_params(w, s, in_features)
            dense = _dequantize(w, s, b, group_size, bits)
            state[f"{path}.weight"] = dense.to(dtype or s.dtype)
            continue
        setattr(parent, attr, new.to(DEVICE))
        replaced.add(path)
    return replaced


def load_model(model_path, lazy=True, strict=False, model_config=None,
                get_model_classes=None, dtype=None):
    """Load an edge0 checkpoint into its transformers model class.

    The published checkpoints are MLX checkpoints: quantized throughout
    (embeddings, lm_head, attention, shared experts, routers -- 392
    non-expert tensors for edge0-35b, 236 for edge0-8b) and, for edge0-35b,
    run through MLX's sanitize. So:

    * modules whose weight has ``.scales`` become ``QuantizedLinear`` /
      ``QuantizedEmbedding`` (4-bit payload stays resident); other
      quantized params are dequantized in place;
    * the MLX sanitize is undone where it was applied (Qwen3.5: conv1d
      layout, ``w + 1`` norms -- see ``_undo_mlx_qwen35_sanitize``);
    * routed-expert tensors (``_EXPERT_KEY_MARKERS``) are not loaded: the
      streaming installer serves them from disk, and those parameters stay
      on the meta device until it replaces them.

    ``dtype`` casts the dense floating-point tensors (default: as stored,
    bf16). ``strict`` raises on non-expert parameters the checkpoint does
    not provide and on checkpoint tensors the model has no place for.
    ``get_model_classes``/``model_config`` exist for signature parity with
    the MLX backend and are unused: the class comes from config.json.
    """
    import json
    import os

    import torch

    with open(os.path.join(os.fspath(model_path), "config.json")) as f:
        raw_config = json.load(f)

    hf_config, model_cls, needs_trust_remote_code, key_prefix = \
        _resolve_model_class(model_path, raw_config)

    with _params_on_meta():
        model = model_cls.from_config(hf_config, trust_remote_code=True) \
            if needs_trust_remote_code else model_cls(hf_config)

    state = {}
    skipped_expert_keys = []
    for shard in open_shards(model_path):
        for name, meta in shard.entries.items():
            if any(m in name for m in _EXPERT_KEY_MARKERS):
                skipped_expert_keys.append(name)
                continue
            mapped = name[len(key_prefix):] if key_prefix and \
                name.startswith(key_prefix) else name
            state[mapped] = _read_tensor(shard, meta, name)
        shard.close()

    if model_cls.__name__.startswith("Qwen3_5Moe"):
        _undo_mlx_qwen35_sanitize(state)
    quantized = _install_quantized(model, state, dtype)
    for k, t in state.items():
        if dtype is not None and t.is_floating_point():
            state[k] = t.to(dtype)
        state[k] = state[k].to(DEVICE)

    missing, unexpected = model.load_state_dict(state, strict=False,
                                                assign=True)
    missing = [k for k in missing
               if not any(m in k for m in _EXPERT_KEY_MARKERS)
               and k.rpartition(".")[0] not in quantized]
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"load_model: {len(missing)} missing, {len(unexpected)} "
            f"unexpected non-expert tensors: missing={missing[:5]} "
            f"unexpected={unexpected[:5]}")
    model.eval()
    model._edge0_skipped_expert_keys = skipped_expert_keys  # for the streaming hook
    return model
