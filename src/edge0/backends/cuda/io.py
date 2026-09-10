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
    already backend-agnostic (pure ``mmap`` + ``numpy``, see the
    mapping doc)."""
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
    installable -- MLX has no Linux wheels, see the mapping doc's CI
    finding)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True)


def load_model(model_path, lazy=True, strict=False, model_config=None,
                get_model_classes=None):
    """NOT IMPLEMENTED -- deliberately, not silently.

    This is the one function in the contract that is NOT a thin
    reshape of existing logic. ``backends/mlx/io.py::load_model``
    delegates to ``mlx_lm.utils.load_model`` plus a vendored model
    class pair (``_impl/qwen3_5_moe.py``, ``_impl/bailing_hybrid.py``)
    hand-written against ``mlx.nn``. Two different answers for the two
    shipped tiers, found by checking what upstream already has:

    * edge0-35b (Qwen3.5-MoE): ``transformers`` (checked: 5.17.0)
      ships a complete, maintained PyTorch implementation
      (``transformers.models.qwen3_5_moe``, ``Qwen3_5MoeForCausalLM``,
      2288 lines) -- including the GatedDeltaNet hybrid-attention
      layers. Depend on it rather than hand-porting
      ``_impl/qwen3_5.py``/``qwen3_next.py``'s gated-delta recurrence;
      re-deriving that kernel by hand, with no MLX available to check
      against, is a correctness risk with no way to catch it here.
    * edge0-8b (Ling 3.0 / Bailing hybrid): no ``transformers`` match
      for "bailing"/"ling" as of 5.17.0. ``_impl/bailing_hybrid.py``
      has its OWN gated-recurrence variant (``BailingKDA``,
      ``_kda_update``) plus ``BailingMLA``, distinct from Qwen3-Next's.
      The checkpoint may ship its own PyTorch modeling code via HF's
      ``trust_remote_code``/``auto_map`` mechanism (``mlx-lm``'s
      tokenizer loader has a comment noting the checkpoint carries an
      ``auto_map`` -- see ``backends/mlx/io.py``); check that on the
      actual `Edge0/Edge0-8B-A1B-preview` repo files before deciding
      whether to depend on it or hand-port ``BailingKDA``/``BailingMLA``.

    Wiring either path in is the next concrete unit of work, not this
    one -- raising here instead of returning something that looks
    loaded but silently isn't.
    """
    raise NotImplementedError(
        "cuda backend: load_model has no implementation yet -- see this "
        "function's docstring for the two different paths the 35b and "
        "8b tiers need (transformers dependency vs. hand-port)")
