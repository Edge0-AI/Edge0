#!/usr/bin/env python3
"""Per-layer tensor formats of a (sharded) GGUF, read from the headers only (docs/QUANTS.md).

    python tools/quant_formats.py PATH-OR-GLOB [...]           local shards (any shard; all are read)
    python tools/quant_formats.py --hf REPO REV DIR [--json OUT] Hugging Face: HTTP range requests on each shard's
                                                                header, nothing else is downloaded

Prints, per role (routed gate / up / down, shared expert, attention, SSM, hyper-connections, indexer, PLE, embedding,
head): the formats and the layers each one covers, and the bytes; then the routed experts layer by layer.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import pathlib
import re
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf_reader as G  # noqa: E402

ROLES = [
    ("routed gate", r"ffn_gate_exps"), ("routed up", r"ffn_up_exps"), ("routed down", r"ffn_down_exps"),
    ("shared expert", r"ffn_(gate|up|down)_shexp\.weight"), ("router", r"ffn_gate_inp"),
    ("attention", r"attn_(q|k|v|qkv|gate|output)\.weight"), ("SSM", r"ssm_(alpha|beta|out)\.weight"),
    ("hyper-connections", r"hc_(attn|ffn)_(up|down|inject)\.weight"), ("indexer", r"indexer\.(q|k)_proj"),
    ("PLE key/value", r"ple_(key|value)\.weight"), ("PLE table", r"^per_layer_token_embd"),
    ("token embedding", r"^token_embd"), ("output head", r"^output(_hc_(up|down))?\.weight"),
]


class RangeFile(io.RawIOBase):
    """A read-only, forward-reading file over HTTP range requests (4 MiB at a time)."""

    CHUNK = 4 << 20

    def __init__(self, url: str):
        self.url, self.pos, self.buf, self.buf_at = url, 0, b"", 0

    def _fill(self, need_to: int) -> None:
        while self.buf_at + len(self.buf) < need_to:
            start = self.buf_at + len(self.buf)
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + self.CHUNK - 1}"})
            data = urllib.request.urlopen(req, timeout=120).read()
            if not data:
                raise EOFError(self.url)
            self.buf += data

    def read(self, n: int = -1) -> bytes:
        self._fill(self.pos + n)
        off = self.pos - self.buf_at
        out = self.buf[off:off + n]
        self.pos += len(out)
        return out

    def tell(self) -> int:
        return self.pos


class Header(G.GGUFFile):
    def __init__(self, name: str, fh):  # noqa: D107 (reuses GGUFFile's parser on any file-like)
        self.path = pathlib.Path(name)
        self.metadata, self.tensors, self.version, self.alignment = {}, [], 0, 32
        self._parse(fh)


def hf_headers(repo: str, rev: str, d: str) -> list[Header]:
    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/{rev}/{d}", timeout=60))
    out = []
    for e in sorted(tree, key=lambda e: e["path"]):
        if e["type"] == "file" and e["path"].endswith(".gguf"):
            url = f"https://huggingface.co/{repo}/resolve/{rev}/{e['path']}"
            out.append(Header(e["path"], RangeFile(url)))
    return out


def ranges(layers: list[int]) -> str:
    layers = sorted(layers)
    out, i = [], 0
    while i < len(layers):
        j = i
        while j + 1 < len(layers) and layers[j + 1] == layers[j] + 1:
            j += 1
        out.append(str(layers[i]) if i == j else f"{layers[i]}-{layers[j]}")
        i = j + 1
    return ", ".join(out)


def summarize(heads: list[Header]) -> dict:
    roles: dict[str, dict[str, dict]] = {}
    per_layer: dict[int, dict[str, str]] = {}
    for h in heads:
        for t in h.tensors:
            m = re.match(r"blk\.(\d+)\.", t.name)
            layer = int(m.group(1)) if m else -1
            for role, pat in ROLES:
                if re.search(pat, t.name):
                    r = roles.setdefault(role, {}).setdefault(t.type_name, {"layers": set(), "bytes": 0, "n": 0})
                    r["layers"].add(layer)
                    r["bytes"] += t.expected_bytes() or 0
                    r["n"] += 1
                    if role.startswith("routed"):
                        per_layer.setdefault(layer, {})[role.split()[1]] = t.type_name
                    break
    return {"roles": roles, "per_layer": per_layer}


def render(name: str, s: dict) -> str:
    lines = [f"### {name}", "", "| Role | Format | Layers | Tensors | GiB |", "|---|---|---|---:|---:|"]
    for role, _ in ROLES:
        for ty, r in sorted(s["roles"].get(role, {}).items(), key=lambda kv: -kv[1]["bytes"]):
            lay = "-" if r["layers"] == {-1} else ranges([x for x in r["layers"] if x >= 0])
            lines.append(f"| {role} | {ty} | {lay} | {r['n']} | {r['bytes'] / 2**30:.2f} |")
    exp = sum(r["bytes"] for role in ("routed gate", "routed up", "routed down")
              for r in s["roles"].get(role, {}).values())
    lines += ["", f"Routed experts: {exp / 2**30:.1f} GiB.", ""]
    combos: dict[tuple, list[int]] = {}
    for layer, d in s["per_layer"].items():
        combos.setdefault((d.get("gate"), d.get("up"), d.get("down")), []).append(layer)
    lines += ["| Routed gate / up / down | Layers |", "|---|---|"]
    for (g, u, dn), lay in sorted(combos.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {g} / {u} / {dn} | {ranges(lay)} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--hf", nargs=3, metavar=("REPO", "REV", "DIR"))
    ap.add_argument("--name")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.hf:
        heads = hf_headers(*a.hf)
        name = a.name or a.hf[2]
    else:
        files = sorted({p for g in a.paths for p in glob.glob(g)})
        heads = [G.GGUFFile(pathlib.Path(p)) for p in files]
        name = a.name or (pathlib.Path(files[0]).parent.name if files else "?")
    s = summarize(heads)
    sys.stdout.write(render(name, s))
    if a.json:
        js = {"roles": {r: {t: {"layers": sorted(v["layers"]), "bytes": v["bytes"], "n": v["n"]}
                            for t, v in d.items()} for r, d in s["roles"].items()},
              "per_layer": {str(k): v for k, v in sorted(s["per_layer"].items())}}
        pathlib.Path(a.json).write_text(json.dumps(js, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
