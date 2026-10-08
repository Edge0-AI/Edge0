#!/usr/bin/env python3
"""GGUF MoE helper for the edge0 Windows shell (standard library only).

Every subcommand prints one JSON object on stdout and exits 0 on success, 1 on error.

  inspect <file.gguf> [--vram-gb F] [--ram-gb F] [--ctx N] [--headroom-gb F]
      Read the GGUF header (all shards of a split model) and report the architecture,
      MoE shape, quantisation and family support. With --vram-gb, also plan expert
      offload for that GPU budget.

  scan <dir>
      Inspect every model under <dir> (first shard of each split model).

  registry-check --llama-src <vendor/llama.cpp>
      Confirm each arch in moe_families.json is registered in the pinned engine's
      src/llama-arch.cpp.

Offload semantics match llama.cpp at the pin: `--n-cpu-moe N` (-ncmoe) keeps the expert
tensors of layers 0..N-1 on the CPU, and `--cpu-moe` (-cmoe) keeps all of them there.
EXPERT_RE is the engine's LLM_FFN_EXPS_REGEX (common/common.h), so the byte counts here
line up with the engine's own overrides.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

GIB = 1024 ** 3
GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (2, 3)
HERE = Path(__file__).resolve().parent
REGISTRY_PATH = HERE / "moe_families.json"

# Largest array we keep in memory. Bigger arrays (tokenizer vocab) are skipped.
KEEP_ARRAY_MAX = 4096

# gguf metadata value types
_SCALAR_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
_T_STRING = 8
_T_ARRAY = 9

# ggml_type ids at the pinned commit (ggml/include/ggml.h). Gaps 4, 5, 31-33 and 36-38
# are removed types and never appear in a file.
GGML_TYPES: Dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
}

# llama.cpp LLM_FFN_EXPS_REGEX (common/common.h)
EXPERT_RE = re.compile(r"\.ffn_(up|down|gate|gate_up)_(ch|)exps")
BLK_RE = re.compile(r"^blk\.(\d+)\.")
SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")
# Entries of LLM_ARCH_NAMES in src/llama-arch.cpp look like: { LLM_ARCH_LLAMA, "llama" },
ARCH_RE = re.compile(r'\{\s*LLM_ARCH_[A-Z0-9_]+\s*,\s*"([^"]+)"\s*\}')


class GGUFError(ValueError):
    """Raised for files that are not readable GGUF (or are truncated/inconsistent)."""


# --------------------------------------------------------------------------- header


def _read_exact(f, n: int) -> bytes:
    buf = f.read(n)
    if len(buf) != n:
        raise GGUFError("truncated GGUF header")
    return buf


def _unpack(f, fmt: str):
    return struct.unpack(fmt, _read_exact(f, struct.calcsize(fmt)))[0]


def _read_string(f) -> str:
    n = _unpack(f, "<Q")
    if n > (1 << 30):
        raise GGUFError("implausible GGUF string length")
    return _read_exact(f, n).decode("utf-8", "replace")


def _read_value(f, vtype: int, keep: bool) -> Any:
    if vtype in _SCALAR_FMT:
        return _unpack(f, _SCALAR_FMT[vtype])
    if vtype == _T_STRING:
        return _read_string(f)
    if vtype == _T_ARRAY:
        etype = _unpack(f, "<I")
        count = _unpack(f, "<Q")
        if etype == _T_ARRAY:
            raise GGUFError("nested GGUF arrays are not supported")
        if etype == _T_STRING:
            if keep and count <= KEEP_ARRAY_MAX:
                return [_read_string(f) for _ in range(count)]
            for _ in range(count):
                f.seek(_unpack(f, "<Q"), os.SEEK_CUR)
            return {"len": count}
        if etype in _SCALAR_FMT:
            if keep and count <= KEEP_ARRAY_MAX:
                return [_unpack(f, _SCALAR_FMT[etype]) for _ in range(count)]
            f.seek(count * struct.calcsize(_SCALAR_FMT[etype]), os.SEEK_CUR)
            return {"len": count}
        raise GGUFError(f"unknown GGUF array element type {etype}")
    raise GGUFError(f"unknown GGUF value type {vtype}")


def read_header(path: str, keep_arrays: bool = True) -> Dict[str, Any]:
    """Parse one GGUF file's header. Tensor data is never read."""
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise GGUFError(f"not a GGUF file: {Path(path).name}")
        version = _unpack(f, "<I")
        if version not in SUPPORTED_VERSIONS:
            raise GGUFError(f"unsupported GGUF version {version} in {Path(path).name}")
        n_tensors = _unpack(f, "<Q")
        n_kv = _unpack(f, "<Q")
        kv: Dict[str, Any] = {}
        for _ in range(n_kv):
            key = _read_string(f)
            vtype = _unpack(f, "<I")
            kv[key] = _read_value(f, vtype, keep_arrays)
        tensors: List[Dict[str, Any]] = []
        for _ in range(n_tensors):
            name = _read_string(f)
            ndim = _unpack(f, "<I")
            if ndim > 8:
                raise GGUFError(f"tensor {name} claims {ndim} dimensions")
            dims = [_unpack(f, "<Q") for _ in range(ndim)]
            ttype = _unpack(f, "<I")
            offset = _unpack(f, "<Q")
            tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})
        header_end = f.tell()
    alignment = kv.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment <= 0:
        alignment = 32
    data_start = -(-header_end // alignment) * alignment
    if data_start > file_size:
        raise GGUFError(f"truncated GGUF: {Path(path).name}")
    return {
        "path": str(path),
        "version": version,
        "kv": kv,
        "tensors": tensors,
        "data_start": data_start,
        "file_size": file_size,
    }


def tensor_byte_sizes(header: Dict[str, Any]) -> List[int]:
    """Byte size of each tensor: distance to the next tensor's offset (or end of data).

    Includes the alignment padding after each tensor, so this slightly overestimates.
    """
    ts = header["tensors"]
    order = sorted(range(len(ts)), key=lambda i: ts[i]["offset"])
    data_len = header["file_size"] - header["data_start"]
    sizes = [0] * len(ts)
    for pos, i in enumerate(order):
        end = ts[order[pos + 1]]["offset"] if pos + 1 < len(order) else data_len
        sizes[i] = max(0, end - ts[i]["offset"])
    return sizes


def shard_paths(path: str) -> List[str]:
    """All shards of a split model (name-00001-of-000NN.gguf), or the file itself."""
    p = Path(path)
    m = SHARD_RE.match(p.name)
    if not m:
        return [str(p)]
    total = int(m.group("total"))
    found: List[str] = []
    missing: List[str] = []
    for i in range(1, total + 1):
        q = p.with_name(f"{m.group('prefix')}-{i:05d}-of-{total:05d}.gguf")
        (found if q.exists() else missing).append(str(q))
    if missing:
        raise GGUFError("missing GGUF shards: " + ", ".join(Path(x).name for x in missing))
    return found


def is_first_shard(name: str) -> bool:
    m = SHARD_RE.match(name)
    return m is None or int(m.group("idx")) == 1


# --------------------------------------------------------------------------- registry


def load_registry() -> Dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def match_family(arch: str, expert_count: int, registry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for fam in registry["families"]:
        if arch not in fam["archs"]:
            continue
        if fam.get("requires_experts") and expert_count <= 0:
            continue
        return fam
    return None


# --------------------------------------------------------------------------- analysis


def _meta_getter(kv: Dict[str, Any], arch: str):
    def meta(key: str, default: Any = None) -> Any:
        return kv.get(f"{arch}.{key}", default)

    return meta


def _dominant(d: Dict[str, int]) -> Optional[str]:
    if not d:
        return None
    return max(d.items(), key=lambda item: item[1])[0]


def kv_bytes_per_token(meta, layers: int):
    """f16 K+V bytes per token over all layers. Returns (bytes, note).

    This is an upper bound: sliding-window layers keep less than the full context.
    """
    kv_lora = meta("attention.kv_lora_rank")
    if kv_lora:
        # MLA stores one latent vector (kv_lora_rank) plus the RoPE part per layer.
        rope = int(meta("rope.dimension_count", 64) or 64)
        return 2 * (int(kv_lora) + rope) * layers, "KV estimate uses the MLA latent cache"
    head_kv = meta("attention.head_count_kv")
    key_len = meta("attention.key_length")
    val_len = meta("attention.value_length")
    if key_len is None:
        emb = meta("embedding_length")
        head = meta("attention.head_count")
        if emb and head:
            key_len = int(emb) // int(head)
    if val_len is None:
        val_len = key_len
    if head_kv is None or key_len is None or val_len is None:
        return 0, "KV size unknown (missing head/key metadata); estimate excludes KV cache"
    if isinstance(head_kv, list):
        heads_total = sum(int(h) for h in head_kv)
    else:
        heads_total = int(head_kv) * layers
    return 2 * heads_total * (int(key_len) + int(val_len)), None


def analyze(path: str, registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read every shard header and summarise the model. Internal fields start with '_'."""
    registry = registry if registry is not None else load_registry()
    shards = shard_paths(path)
    headers = [read_header(s) for s in shards]
    # Metadata normally lives in shard 1, but merge all shards in case a writer splits it.
    kv: Dict[str, Any] = {}
    for h in headers:
        for key, value in h["kv"].items():
            kv.setdefault(key, value)
    arch = kv.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        raise GGUFError("GGUF has no general.architecture")
    meta = _meta_getter(kv, arch)

    block_count = int(meta("block_count", 0) or 0)
    expert_count = int(meta("expert_count", 0) or 0)
    expert_used = int(meta("expert_used_count", 0) or 0)

    total_bytes = 0
    expert_bytes = 0
    expert_by_layer: Dict[int, int] = {}
    expert_type_bytes: Dict[str, int] = {}
    type_bytes: Dict[str, int] = {}
    tensor_count = 0
    unknown_types: List[str] = []
    for h in headers:
        for t, size in zip(h["tensors"], tensor_byte_sizes(h)):
            tensor_count += 1
            total_bytes += size
            tname = GGML_TYPES.get(t["type"])
            if tname is None:
                tname = f"type{t['type']}"
                if tname not in unknown_types:
                    unknown_types.append(tname)
            type_bytes[tname] = type_bytes.get(tname, 0) + size
            if EXPERT_RE.search(t["name"]):
                expert_bytes += size
                expert_type_bytes[tname] = expert_type_bytes.get(tname, 0) + size
                m = BLK_RE.match(t["name"])
                if m:
                    layer = int(m.group(1))
                    expert_by_layer[layer] = expert_by_layer.get(layer, 0) + size

    layers = block_count or (max(expert_by_layer) + 1 if expert_by_layer else 0)
    per_layer = [expert_by_layer.get(i, 0) for i in range(max(layers, max(expert_by_layer, default=-1) + 1))]
    layers = max(layers, len(per_layer))

    kv_tok, kv_note = kv_bytes_per_token(meta, layers)
    is_moe = expert_count > 0 and expert_bytes > 0
    family = match_family(arch, expert_count, registry)

    warnings: List[str] = []
    if unknown_types:
        warnings.append("unknown tensor types: " + ", ".join(unknown_types))
    if not is_moe:
        warnings.append("dense model: expert offload does not apply")
    if family is None and is_moe:
        warnings.append(
            f"MoE architecture '{arch}' is not listed in moe_families.json; the pinned engine may not load it"
        )
    if not kv.get("tokenizer.chat_template"):
        warnings.append("no embedded chat template; pass --chat-template or use a model that ships one")
    if len(shards) > 1:
        warnings.append(f"split model: {len(shards)} shards")

    quant = _dominant(expert_type_bytes) if expert_bytes else _dominant(type_bytes)
    name = kv.get("general.name") or Path(shards[0]).stem
    return {
        "path": str(path),
        "shards": [Path(s).name for s in shards],
        "name": name,
        "architecture": arch,
        "is_moe": is_moe,
        "layers": layers,
        "experts": expert_count,
        "experts_used": expert_used,
        "context_length": meta("context_length"),
        "embedding_length": meta("embedding_length"),
        "quant": quant,
        "quant_by_bytes": dict(sorted(type_bytes.items(), key=lambda kv_: -kv_[1])),
        "tensor_count": tensor_count,
        "file_bytes": sum(h["file_size"] for h in headers),
        "tensor_bytes": total_bytes,
        "expert_bytes": expert_bytes,
        "has_chat_template": bool(kv.get("tokenizer.chat_template")),
        "family": (
            {"id": family["id"], "label": family["label"], "archs": family["archs"]}
            if family
            else None
        ),
        "kv_bytes_per_token": kv_tok,
        "kv_note": kv_note,
        "warnings": warnings,
        "_expert_by_layer": per_layer,
    }


# --------------------------------------------------------------------------- planning


def plan_offload(
    a: Dict[str, Any],
    *,
    vram_gb: float,
    ram_gb: Optional[float],
    ctx: int,
    headroom_gb: float,
) -> Dict[str, Any]:
    """Choose the smallest -ncmoe N that fits the GPU budget (or --cpu-moe if none does).

    GPU use = dense tensors + expert tensors of layers N..L-1 + KV cache for `ctx`.
    The dense total includes token embeddings, which llama.cpp keeps on the CPU, so the
    estimate is conservative.
    """
    notes: List[str] = []
    if a.get("kv_note"):
        notes.append(a["kv_note"])
    kv_bytes = a["kv_bytes_per_token"] * ctx
    budget = vram_gb * GIB - headroom_gb * GIB - kv_bytes
    layers = a["layers"]
    if layers <= 0:
        return {
            "fits": None,
            "mode": "unknown",
            "args": ["-ngl", "99"],
            "notes": notes + ["layer count unknown; cannot plan expert offload"],
        }

    def out(fits: bool, mode: str, n_cpu: int, cpu_bytes: int, gpu_bytes: int, args: List[str]) -> Dict[str, Any]:
        if ram_gb is not None and cpu_bytes > 0.8 * ram_gb * GIB:
            notes.append(
                f"CPU-resident weights ({cpu_bytes / GIB:.1f} GiB) exceed ~80% of {ram_gb:.0f} GiB RAM; expect paging"
            )
        return {
            "fits": fits,
            "mode": mode,
            "n_cpu_moe": n_cpu,
            "cpu_moe": n_cpu >= layers and a["is_moe"] and mode != "gpu",
            "args": args,
            "gpu_gib_est": round(gpu_bytes / GIB, 2),
            "cpu_gib_est": round(cpu_bytes / GIB, 2),
            "kv_gib_est": round(kv_bytes / GIB, 2),
            "budget_gib": round(budget / GIB, 2),
            "notes": notes,
        }

    if not a["is_moe"]:
        total = a["tensor_bytes"]
        if total <= budget:
            return out(True, "gpu", 0, 0, total, ["-ngl", "99"])
        per_layer = total / layers
        n_gpu = int(budget // per_layer) if budget > 0 else 0
        notes.append("dense model larger than the GPU budget; the remaining layers run on the CPU")
        return out(n_gpu > 0, "partial", 0, total - int(n_gpu * per_layer), int(n_gpu * per_layer), ["-ngl", str(n_gpu)])

    per_layer = a["_expert_by_layer"]
    dense = a["tensor_bytes"] - a["expert_bytes"]
    suffix = [0] * (layers + 1)
    for i in range(layers - 1, -1, -1):
        suffix[i] = suffix[i + 1] + per_layer[i]

    if dense + suffix[0] <= budget:
        return out(True, "gpu", 0, 0, dense + suffix[0], ["-ngl", "99"])
    for n in range(1, layers + 1):
        if dense + suffix[n] <= budget:
            gpu = dense + suffix[n]
            cpu = suffix[0] - suffix[n]
            if n >= layers:
                return out(True, "cpu-experts", n, cpu, gpu, ["-ngl", "99", "--cpu-moe"])
            return out(True, "cpu-experts", n, cpu, gpu, ["-ngl", "99", "--n-cpu-moe", str(n)])
    # Every expert on CPU and the dense part still does not fit.
    notes.append("dense tensors and KV cache exceed the GPU budget; lower --ctx or pick a smaller quant")
    return out(False, "does-not-fit", layers, suffix[0], dense, ["-ngl", "99", "--cpu-moe"])


def inspect_model(
    path: str,
    *,
    ctx: int = 8192,
    vram_gb: Optional[float] = None,
    ram_gb: Optional[float] = None,
    headroom_gb: float = 1.5,
    registry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    a = analyze(path, registry=registry)
    result: Dict[str, Any] = {"ok": True, **{k: v for k, v in a.items() if not k.startswith("_")}}
    result["ctx"] = ctx
    if vram_gb is not None:
        result["plan"] = plan_offload(a, vram_gb=vram_gb, ram_gb=ram_gb, ctx=ctx, headroom_gb=headroom_gb)
    return result


def scan_dir(root: str, registry: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    registry = registry if registry is not None else load_registry()
    rows: List[Dict[str, Any]] = []
    for p in sorted(Path(root).rglob("*.gguf")):
        if not is_first_shard(p.name):
            continue
        try:
            a = analyze(str(p), registry=registry)
        except (GGUFError, OSError) as exc:
            rows.append({"path": str(p), "ok": False, "error": str(exc)})
            continue
        rows.append(
            {
                "ok": True,
                "path": a["path"],
                "name": a["name"],
                "shards": a["shards"],
                "architecture": a["architecture"],
                "is_moe": a["is_moe"],
                "experts": a["experts"],
                "family": a["family"]["id"] if a["family"] else None,
                "family_label": a["family"]["label"] if a["family"] else None,
                "quant": a["quant"],
                "file_bytes": a["file_bytes"],
            }
        )
    return rows


def registry_check(llama_src: str, registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    src = Path(llama_src) / "src" / "llama-arch.cpp"
    if not src.is_file():
        raise GGUFError(f"not found: {src}")
    registered = set(ARCH_RE.findall(src.read_text(encoding="utf-8", errors="replace")))
    registry = registry if registry is not None else load_registry()
    wanted: List[str] = []
    for fam in registry["families"]:
        for arch in fam["archs"]:
            if arch not in wanted:
                wanted.append(arch)
    missing = [a for a in wanted if a not in registered]
    return {"ok": not missing, "checked": len(wanted), "missing": missing, "source": str(src)}


# --------------------------------------------------------------------------- CLI


def _emit(obj: Dict[str, Any]) -> None:
    # ASCII-only output: the Windows shell reads stdout through the console code page.
    json.dump(obj, sys.stdout, indent=2, ensure_ascii=True)
    sys.stdout.write("\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="gguf_tool.py", description="GGUF MoE helper (JSON output)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="inspect one model (and plan offload)")
    p_inspect.add_argument("path")
    p_inspect.add_argument("--vram-gb", type=float, default=None)
    p_inspect.add_argument("--ram-gb", type=float, default=None)
    p_inspect.add_argument("--ctx", type=int, default=8192)
    p_inspect.add_argument("--headroom-gb", type=float, default=1.5)

    p_scan = sub.add_parser("scan", help="inspect every model under a directory")
    p_scan.add_argument("dir")

    p_reg = sub.add_parser("registry-check", help="check the family registry against a llama.cpp tree")
    p_reg.add_argument("--llama-src", required=True)

    args = ap.parse_args(argv)
    try:
        if args.cmd == "inspect":
            _emit(
                inspect_model(
                    args.path,
                    ctx=args.ctx,
                    vram_gb=args.vram_gb,
                    ram_gb=args.ram_gb,
                    headroom_gb=args.headroom_gb,
                )
            )
            return 0
        if args.cmd == "scan":
            _emit({"ok": True, "models": scan_dir(args.dir)})
            return 0
        res = registry_check(args.llama_src)
        _emit(res)
        return 0 if res["ok"] else 1
    except (GGUFError, OSError) as exc:
        _emit({"ok": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
