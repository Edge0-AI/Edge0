"""Parallel (unmerged) LoRA adapters, safetensors-native.

Port of the deployment's ``lora_apply.py`` scheme (both qwen and ling
tiers): each adapter target is wrapped as a *parallel delta path*::

    y = base(x) + scale * ((x @ A.T) @ B.T)      # scale = alpha / r

The base weights (including 4bit quantized linears) stay byte-identical —
no dequant / merge / requant — and the LoRA lives only in tiny fp16 A/B
tensors.  Mathematically identical to a merged weight.

Safetensors key naming (framework contract)::

    <target>.lora_A    [r, in]
    <target>.lora_B    [out, r]

where ``<target>`` is the module path inside the loaded model (e.g.
``model.layers.3.attention.q_proj`` or
``language_model.model.layers.0.linear_attn.in_proj_qkv``).  The legacy
npz files used the same A/B naming, so conversion is a pure format
migration (see ``scripts/convert_adapters_legacy.py``).
"""

from __future__ import annotations

import os

from edge0.backends import core
from edge0.backends import nn

from edge0.backends.mlx.io import load_safetensors


class LoraLinear(nn.Module):
    """Wraps a Linear/QuantizedLinear and adds the low-rank delta."""

    def __init__(self, base: nn.Module, lora_a: core.array, lora_b: core.array,
                 scale: float):
        super().__init__()
        self.base = base
        self.lora_A = lora_a      # [r, in]
        self.lora_B = lora_b      # [out, r]
        self.lora_scale = scale

    def __call__(self, x: core.array) -> core.array:
        y = self.base(x)
        ad = self.lora_A.dtype
        xd = x if x.dtype == ad else x.astype(ad)
        # delta = (B @ A) applied on x:  ((x @ A.T) @ B.T), shape [..., out]
        d = (xd @ self.lora_A.T) @ self.lora_B.T
        return y + (self.lora_scale * d).astype(y.dtype)


def _resolve(model: nn.Module, key: str):
    """Resolve ``model.layers.N.<...>`` or a bare dotted path to the
    module inside the loaded model."""
    parts = key.split(".")
    if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
        node = model.model.layers[int(parts[2])]
        rest = parts[3:]
    elif len(parts) >= 3 and parts[0] == "language_model" and parts[1] == "model":
        node = model.language_model.model.layers[int(parts[3])]
        rest = parts[4:]
    else:
        node = model
        rest = parts
    for p in rest:
        node = getattr(node, p)
    return node


def install_lora(model: nn.Module, adapters_path: str,
                 r: int = 16, alpha: float = 32.0,
                 dtype=core.float16, strict: bool = True) -> dict:
    """Attach every LoRA pair in ``adapters_path`` (safetensors) as a
    parallel delta path.

    Returns ``{applied: [...], not_found: [...], skipped: [...],
    scale: float, dtype: str}``.
    """
    if not os.path.isfile(adapters_path):
        raise FileNotFoundError(
            f"LoRA adapter file not found: {adapters_path}\n"
            "edge0 tiers are shipped with trained LoRA + prerouter "
            "adapters; download them for this tier (README -> 'Getting "
            "the models & adapters') and place them in the model "
            "directory, or disable LoRA with lora="" / --no-lora.")
    lora = load_safetensors(adapters_path)
    if not lora:
        raise ValueError("adapters has no lora tensors")

    # group by target key
    targets: dict[str, dict] = {}
    for k, v in lora.items():
        if k.endswith(".lora_A"):
            targets.setdefault(k[: -len(".lora_A")], {})["A"] = v
        elif k.endswith(".lora_B"):
            targets.setdefault(k[: -len(".lora_B")], {})["B"] = v
        else:
            raise ValueError(f"unexpected lora key: {k}")

    scale = alpha / r
    applied, not_found, skipped = [], [], []

    for key, d in sorted(targets.items()):
        if "A" not in d or "B" not in d:
            skipped.append((key, "missing A or B"))
            continue
        mod = _resolve(model, key)
        if mod is None:
            not_found.append(key)
            if strict:
                raise KeyError(f"lora target not found in MLX model: {key}")
            continue
        if not hasattr(mod, "weight") and not hasattr(mod, "base"):
            # not a linear layer we can wrap (e.g. RMSNorm / gate)
            skipped.append((key, type(mod).__name__))
            if strict:
                raise TypeError(
                    f"lora target {key} is {type(mod).__name__}, not Linear")
            continue
        a = d["A"].astype(dtype)
        b = d["B"].astype(dtype)
        # re-bind the module attribute (e.g. attention.q_proj = LoraLinear)
        parts = key.split(".")
        owner = _resolve(model, ".".join(parts[:-1]))
        attr = parts[-1]
        setattr(owner, attr, LoraLinear(mod, a, b, scale))
        applied.append(key)

    print(f"[lora] applied={len(applied)} not_found={len(not_found)} "
          f"skipped={len(skipped)} scale={scale}", flush=True)
    return {"applied": applied, "not_found": not_found, "skipped": skipped,
            "scale": scale, "dtype": str(dtype)}
