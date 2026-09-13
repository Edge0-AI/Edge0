"""MLX backend: model / tokenizer / tensor-store loading."""

from __future__ import annotations

import json
import mmap
import struct

import numpy as np
import mlx.core as mx
from mlx_lm.utils import load_model as _load_model

from edge0.backends.base import TensorStore


def load_model(model_path, lazy=True, strict=False, model_config=None,
               get_model_classes=None):
    """Load a checkpoint skeleton via mlx-lm.

    ``get_model_classes`` (callable config -> ``(Model, ModelArgs)``)
    selects a vendored class pair instead of mlx-lm's registry; its
    ``ModelArgs`` fields can be overridden through ``model_config`` —
    the hook used by the edge0 model adapters (e.g. prerouter wiring
    flags).  ``lazy=True`` keeps weights as mmap-backed lazy arrays so
    the skeleton occupies no RAM (only the small non-MoE parts are ever
    materialized).
    """
    from pathlib import Path
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    return _load_model(model_path, lazy=lazy, strict=strict,
                       model_config=model_config,
                       get_model_classes=get_model_classes)


def load_tokenizer(model_path):
    """Load a HuggingFace tokenizer from a LOCAL directory only.

    mlx-lm 0.31's ``load_tokenizer`` funnels the path through its
    ``_download`` (hf hub snapshot logic), which prompts for
    ``trust_remote_code`` when the checkpoint carries an ``auto_map`` —
    undesirable for a local-only serving stack.  The transformers
    ``AutoTokenizer`` loads a local directory without any hub round trip.
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True)


def open_shards(model_dir: str) -> list:
    """Open every ``model*.safetensors`` shard as a byte-range mmap
    (qwen-style multi-shard checkpoints; a single-file checkpoint yields
    a one-element list)."""
    from edge0.checkpoints.source import SafetensorsSource
    return _SourceShards(SafetensorsSource(model_dir))


class _SourceShards(list):
    """Legacy streaming view over a checkpoint source, with explicit ownership."""
    def __init__(self, source):
        super().__init__(source.shards)
        self.source = source

    def close(self):
        self.source.close()
        self.clear()


def load_safetensors(path: str, dtype=None) -> dict[str, "mx.array"]:
    """Load every tensor of a (small) safetensors file as mx arrays.

    Used for adapter / prerouter weight files; dtype converts (e.g.
    mx.float16) without copying when it matches the stored dtype.
    """
    import mlx.core as mx
    store = SafeTensorsStore(path)
    try:
        out = {}
        for name in store.keys():
            arr = store.get(name)
            if dtype is not None and arr.dtype != dtype:
                arr = arr.astype(dtype)
            out[name] = arr
        return out
    finally:
        store.close()


class SafeTensorsStore(TensorStore):
    """mmap-backed safetensors store with per-tensor lazy access.

    Tensors are returned as ``mx.array`` views (zero copy).  ``get`` never
    materializes the whole file.
    """

    def __init__(self, path: str):
        import mlx.core as mx
        self._mx = mx
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
        arr = np.frombuffer(buf, dtype=_DTYPES[e["dtype"]],
                            count=int(np.prod(e["shape"])))
        return self._mx.array(arr.reshape(e["shape"]))

    def close(self):
        self._mm.close()
        self._file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# Some numpy builds lack ``np.bfloat16`` (absent on this machine's
# 2.5.3); mlx has a native bfloat16 we fall back to — the array comes
# back as an mlx array either way.
_BF16 = getattr(np, "bfloat16", None) or mx.bfloat16
_DTYPES = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "BF16": _BF16, "I64": np.int64, "I32": np.int32,
    "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_,
}
