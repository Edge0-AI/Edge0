#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# From https://github.com/1872183316/quantscope (commit bfdf205), used by tools/strata_inspect.py and setup's --inspect.
"""quantscope: find the real quantization of a model without downloading it.

Reads only file headers (a few MB per file, via HTTP range requests) of GGUF or safetensors
models on ModelScope, Hugging Face, any URL, or local disk, and reports the actual storage
type and bits per weight of every tensor group (routed experts, attention, embeddings, ...).
File names such as "IQ4_XS" or "Q4_K_XL" are recipe labels; the header is the ground truth.

Examples:
  python quantscope.py ms:unsloth/Qwen3.8-Flash-Next-GGUF              # list variants
  python quantscope.py ms:unsloth/Qwen3.8-Flash-Next-GGUF UD-Q4_K_XL   # one variant
  python quantscope.py ms:unsloth/Qwen3.8-Flash-Next-GGUF --all        # compare all variants
  python quantscope.py hf:Qwen/Qwen3-30B-A3B-GPTQ-Int4
  python quantscope.py models/model-00001-of-00003.gguf
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import re
import struct
import sys
import urllib.request
from collections import defaultdict

# ---------------------------------------------------------------- sources --------------------

HUBS = {
    "ms": ("https://modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true",
           "https://modelscope.cn/models/{repo}/resolve/master/{path}"),
    "hf": ("https://huggingface.co/api/models/{repo}/tree/main?recursive=true",
           "https://huggingface.co/{repo}/resolve/main/{path}"),
}
if os.environ.get("HF_ENDPOINT"):
    base = os.environ["HF_ENDPOINT"].rstrip("/")
    HUBS["hf"] = (base + "/api/models/{repo}/tree/main?recursive=true",
                  base + "/{repo}/resolve/main/{path}")


def http(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {"User-Agent": "quantscope"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=60) as r:
                return r.read()
        except Exception as error:  # noqa: BLE001 - network retries
            if attempt == 4:
                raise RuntimeError(f"{url}: {error}") from error
    raise AssertionError


class Source:
    """A file readable by byte ranges (remote or local) with a known size."""

    def __init__(self, name: str, size: int, url: str | None = None, path: str | None = None):
        self.name, self.size, self.url, self.path = name, size, url, path

    def read(self, start: int, length: int) -> bytes:
        length = max(0, min(length, self.size - start))
        if length == 0:
            return b""
        if self.path:
            with open(self.path, "rb") as f:
                f.seek(start)
                return f.read(length)
        return http(self.url, start, start + length - 1)


def list_hub(kind: str, repo: str) -> list[Source]:
    list_url, file_url = HUBS[kind]
    data = json.loads(http(list_url.format(repo=repo)))
    out = []
    if kind == "ms":
        for f in data["Data"]["Files"]:
            if f.get("Type") == "blob":
                out.append(Source(f["Path"], int(f.get("Size") or 0),
                                  url=file_url.format(repo=repo, path=f["Path"])))
    else:
        for f in data:
            if f.get("type") == "file":
                size = int((f.get("lfs") or {}).get("size") or f.get("size") or 0)
                out.append(Source(f["path"], size, url=file_url.format(repo=repo, path=f["path"])))
    return out


def resolve(spec: str) -> list[Source]:
    m = re.match(r"^(ms|hf):([^/]+/[^/:]+)$", spec)
    if m:
        return list_hub(m.group(1), m.group(2))
    if spec.startswith(("http://", "https://")):
        head = urllib.request.Request(spec, method="HEAD", headers={"User-Agent": "quantscope"})
        with urllib.request.urlopen(head, timeout=60) as r:
            size = int(r.headers.get("Content-Length") or 0)
        return [Source(spec.rsplit("/", 1)[-1], size, url=spec)]
    paths = [spec] if os.path.isfile(spec) else sorted(
        glob.glob(os.path.join(spec, "**", "*"), recursive=True))
    return [Source(os.path.relpath(p, spec) if os.path.isdir(spec) else os.path.basename(p),
                   os.path.getsize(p), path=p) for p in paths if os.path.isfile(p)]


def variant_of(name: str) -> str:
    """Groups the shards of one model file: strips -00001-of-00004 and the extension."""
    return re.sub(r"(-\d{5}-of-\d{5})?\.(gguf|safetensors|ninfer)$", "", name)


# ---------------------------------------------------------------- GGUF -----------------------

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
    17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
    42: "Q2_0",  # ISTA-DASLab GSQ-RCO 2-bit (Strata, llama.cpp fork type)
    143: "PTQ1_0", 144: "TQ2_T", 145: "TQK6", 146: "TQK7",  # agentionai Gyro (tools/gguf_reader.py has the block sizes)
}


class Buffered:
    """Sequential reader over a remote header that fetches more bytes on demand."""

    def __init__(self, src: Source, first: int = 4 << 20):
        self.src, self.buf, self.pos = src, src.read(0, first), 0

    def take(self, n: int) -> bytes:
        while self.pos + n > len(self.buf):
            if len(self.buf) >= self.src.size:
                raise EOFError(f"{self.src.name}: truncated header")
            self.buf += self.src.read(len(self.buf), max(len(self.buf), 4 << 20))
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def u32(self) -> int: return struct.unpack("<I", self.take(4))[0]
    def u64(self) -> int: return struct.unpack("<Q", self.take(8))[0]
    def string(self) -> str: return self.take(self.u64()).decode("utf-8", "replace")


SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_value(r: Buffered, vtype: int):
    if vtype == 8:
        return r.string()
    if vtype == 9:
        itype, n = r.u32(), r.u64()
        if itype == 8:
            items = [r.string() for _ in range(n)]  # must be consumed either way
            return items if n <= 64 else None
        if itype == 9:
            raise ValueError("nested GGUF arrays are not supported")
        size = SCALAR[itype]
        r.take(size * n)
        return None
    raw = r.take(SCALAR[vtype])
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}[vtype]
    return struct.unpack(fmt, raw)[0]


def read_gguf(src: Source) -> tuple[dict, list[dict]]:
    r = Buffered(src)
    if r.take(4) != b"GGUF":
        raise ValueError(f"{src.name}: not a GGUF file")
    r.u32()  # version
    n_tensors, n_kv = r.u64(), r.u64()
    meta = {}
    for _ in range(n_kv):
        key = r.string()
        meta[key] = gguf_value(r, r.u32())
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.u64() for _ in range(r.u32())]
        ttype, offset = r.u32(), r.u64()
        count = 1
        for d in dims:
            count *= d
        tensors.append({"name": name, "type": GGML_TYPES.get(ttype, f"type#{ttype}"),
                        "elements": count, "offset": offset})
    align = int(meta.get("general.alignment") or 32)
    data_start = (r.pos + align - 1) // align * align
    # Actual stored bytes from the offsets: exact for every type, including unknown ones.
    order = sorted(tensors, key=lambda t: t["offset"])
    for a, b in zip(order, order[1:] + [None]):
        end = b["offset"] if b else src.size - data_start
        a["bytes"] = end - a["offset"]
    return meta, tensors


# ---------------------------------------------------------------- safetensors ----------------

ST_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8,
            "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E8M0": 1, "U32": 4}
PACKED = re.compile(r"(qweight|weight_packed|\.blocks$|_blocks$)")


def read_safetensors(src: Source, bits: int | None) -> list[dict]:
    n = struct.unpack("<Q", src.read(0, 8))[0]
    header = json.loads(src.read(8, n))
    out = []
    for name, t in header.items():
        if name == "__metadata__":
            continue
        count = 1
        for d in t["shape"]:
            count *= d
        nbytes = count * ST_BYTES.get(t["dtype"], 1)
        elements = count
        if PACKED.search(name):
            # Packed integer codes: logical weights = stored bits / code bits.
            elements = nbytes * 8 // (bits or 4)
        elif re.search(r"(scales|zeros|qzeros|g_idx|scale$|_scale|zero_point)", name):
            elements = 0  # metadata of a quantized weight: counts as bytes only
        out.append({"name": name, "type": t["dtype"], "elements": elements, "bytes": nbytes})
    return out


# ---------------------------------------------------------------- NInfer v3 ------------------

NINFER_MAGIC = b"NINFER" + bytes([0, 3])


def read_ninfer(src: Source) -> list[dict]:
    """Logical parameters of a NInfer v3 .ninfer entry, from its directory JSON only.

    A stored object may hold several logical parameters (for example an expert bank); each
    binding's share of the object's stored bytes is proportional to its element range.
    """
    head = src.read(0, 32)
    magic, json_bytes, _artifact_id = struct.unpack("<8sQ16s", head)
    if magic != NINFER_MAGIC:
        raise ValueError(f"{src.name}: not a NInfer v3 entry")
    directory = json.loads(src.read(32, json_bytes))
    objects = {o["id"]: o for o in directory["objects"] if o.get("kind", "tensor") == "tensor"}
    elements = {}
    for oid, o in objects.items():
        count = 1
        for d in o["shape"]:
            count *= d
        elements[oid] = count
    out = []
    for name, binding in directory["bindings"].items():
        parts = ([{"object": binding["object"], "range": [0, elements.get(binding["object"], 0)]}]
                 if "object" in binding else binding.get("parts", []))
        count = nbytes = 0
        fmt = None
        for part in parts:
            o = objects.get(part["object"])
            if o is None:
                continue
            n = part["range"][1] - part["range"][0]
            count += n
            nbytes += o["bytes"] * n / max(elements[part["object"]], 1)
            fmt = o["format"]
        if count:
            out.append({"name": name, "type": (fmt or "?").upper(), "elements": count,
                        "bytes": nbytes})
    return out


# ---------------------------------------------------------------- summary --------------------

CATEGORIES = [
    ("norms", r"(norm)"),
    ("hyper-connections", r"(hc_|_hc\.|hyper_connection)"),
    ("routed experts", r"(_exps\b|\.experts\.|experts\.\d+|block_sparse_moe\.experts)"),
    ("shared experts", r"(_shexp|shared_expert|moe\.shared\.)"),
    ("router", r"(ffn_gate_inp|\.gate\.weight$|\.router\b|shared_score)"),
    ("n-gram/per-layer embeddings", r"(ngram|per_layer|ple_|\.ple\.)"),
    ("linear attn/SSM", r"(ssm_|linear_attn|mamba|conv1d|\.gdn\.)"),
    # GGUF also names GatedDeltaNet input projections attn_qkv/attn_gate.
    ("attention (incl. GDN qkv/gate)", r"(attn|attention|self_attn|indexer|\.q_proj|\.k_proj|"
                                       r"\.v_proj|\.o_proj)"),
    ("dense FFN", r"(ffn_|\.mlp\.)"),
    ("token embedding", r"(token_embd|embed_tokens|token_embedding|wte)"),
    ("output head", r"(^output\.|lm_head|output_head)"),
    ("MTP / nextn", r"(nextn|mtp)"),
]


def category(name: str) -> str:
    n = name.lower().replace("/", ".")  # NInfer logical names use slashes
    for label, pattern in CATEGORIES:
        if re.search(pattern, n):
            return label
    return "other"


def q_class(bpw: float) -> str:
    for limit, label in [(1.9, "Q1 class"), (2.8, "Q2 class"), (3.7, "Q3 class"),
                         (4.9, "Q4 class"), (5.9, "Q5 class"), (7.5, "Q6 class"),
                         (10.0, "Q8 class"), (17.0, "16-bit")]:
        if bpw < limit:
            return label
    return "32-bit"


def summarize(title: str, tensors: list[dict]) -> dict:
    groups: dict[str, dict] = defaultdict(lambda: {"elements": 0, "bytes": 0,
                                                  "types": defaultdict(int)})
    for t in tensors:
        g = groups[category(t["name"])]
        g["elements"] += t["elements"]
        g["bytes"] += t["bytes"]
        g["types"][t["type"]] += t["bytes"]
    total_e = sum(g["elements"] for g in groups.values())
    total_b = sum(g["bytes"] for g in groups.values())
    print(f"\n=== {title} ===")
    print(f"total {total_b / 1e9:.2f} GB, {total_e / 1e9:.2f} B weights, "
          f"{8 * total_b / max(total_e, 1):.2f} bits/weight overall")
    print(f"{'group':30s} {'weights':>9s} {'GB':>8s} {'bits/w':>7s}  types (share of bytes)")
    for label, g in sorted(groups.items(), key=lambda kv: -kv[1]["bytes"]):
        bpw = 8 * g["bytes"] / g["elements"] if g["elements"] else float("nan")
        types = ", ".join(f"{k} {100 * v / g['bytes']:.0f}%"
                          for k, v in sorted(g["types"].items(), key=lambda kv: -kv[1]))
        print(f"{label:30s} {g['elements'] / 1e9:8.2f}B {g['bytes'] / 1e9:8.2f} {bpw:7.2f}  {types}")
    experts = groups.get("routed experts")
    verdict = None
    if experts and experts["elements"]:
        verdict = 8 * experts["bytes"] / experts["elements"]
        print(f"-> routed experts are {verdict:.2f} bits/weight ({q_class(verdict)})")
    return {"title": title, "gb": total_b / 1e9, "bpw": 8 * total_b / max(total_e, 1),
            "experts_bpw": verdict}


def scan(label: str, files: list[Source], meta_config: dict | None) -> dict:
    tensors: list[dict] = []
    for src in files:
        if src.name.endswith(".gguf"):
            _meta, ts = read_gguf(src)
        elif src.name.endswith(".ninfer"):
            ts = read_ninfer(src)
        else:
            bits = None
            if meta_config:
                q = meta_config.get("quantization_config") or {}
                bits = q.get("bits") or q.get("w_bit")
            ts = read_safetensors(src, bits)
        tensors += ts
        print(f"  read header of {src.name}: {len(ts)} tensors", file=sys.stderr)
    return summarize(label, tensors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="ms:owner/repo, hf:owner/repo, URL, file or directory")
    parser.add_argument("variant", nargs="?", help="variant name or glob (e.g. UD-Q4_K_XL)")
    parser.add_argument("--all", action="store_true", help="scan and compare every variant")
    args = parser.parse_args()

    files = [f for f in resolve(args.source)
             if f.name.endswith((".gguf", ".safetensors", ".ninfer"))]
    config = None
    for f in resolve(args.source) if args.source.startswith(("ms:", "hf:")) else []:
        if f.name == "config.json":
            config = json.loads(http(f.url))
            if config.get("quantization_config"):
                print("config.json quantization_config:",
                      json.dumps(config["quantization_config"])[:400])
    variants: dict[str, list[Source]] = defaultdict(list)
    for f in files:
        variants[variant_of(f.name)].append(f)
    if not variants:
        sys.exit("no .gguf, .safetensors or .ninfer files found")
    if len(variants) > 1 and not args.variant and not args.all:
        print("variants (pass one as the second argument, or --all):")
        for name, fs in sorted(variants.items()):
            print(f"  {name:70s} {sum(f.size for f in fs) / 1e9:8.2f} GB  {len(fs)} file(s)")
        return
    chosen = {k: v for k, v in variants.items()
              if args.all or not args.variant or args.variant in k
              or fnmatch.fnmatch(k, args.variant)}
    results = [scan(name, sorted(fs, key=lambda f: f.name), config)
               for name, fs in sorted(chosen.items())]
    if len(results) > 1:
        print("\n=== comparison ===")
        print(f"{'variant':60s} {'GB':>8s} {'bits/w':>7s} {'experts bits/w':>15s}")
        for r in results:
            e = f"{r['experts_bpw']:.2f} ({q_class(r['experts_bpw'])})" if r["experts_bpw"] else "-"
            print(f"{r['title']:60s} {r['gb']:8.2f} {r['bpw']:7.2f} {e:>15s}")


if __name__ == "__main__":
    main()
