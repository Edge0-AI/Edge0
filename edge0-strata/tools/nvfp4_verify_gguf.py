"""tools/nvfp4_verify_gguf.py - check a converted NVFP4 GGUF against its ModelOpt source, and a PLE GGUF against it.

    python tools/nvfp4_verify_gguf.py --model <ModelOpt dir> --gguf <converted.gguf> [--ple-gguf <IQ4_NL PLE gguf>]

nvfp4_verify.py proves the repack function is lossless. This checks what was actually written, where the
converter's own bookkeeping could go wrong without the function being wrong:

  * expert order: 512 experts are sorted and stacked per (layer, projection); expert e of the stacked tensor
    must be the source's expert e - checked bit for bit (signed zeros normalised) on sampled (layer, expert);
  * scale order: `.scale[e]` must be that expert's weight_scale_2;
  * the dense policy: every tensor's type against the policy, and every non-PLE tensor name of the reference
    layout present;
  * the PLE table (with --ple-gguf): the engine will read another file's IQ4_NL table, which is only valid if it
    is the same table. Sampled rows of it are compared with this checkpoint's FP8 rows by cosine similarity -
    two quantizations of one table agree to ~0.99, different tables do not.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import numpy as np
import torch
from safetensors import safe_open

LLAMA = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "llama.cpp"
sys.path.insert(0, str(LLAMA / "gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType as Q, quants  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nvfp4_verify import modelopt_dequant  # noqa: E402

PROJ = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}


def tensor_map(reader: GGUFReader) -> dict:
    return {t.name: t for t in reader.tensors}


def check_experts(model: pathlib.Path, T: dict, samples: int, rng: random.Random) -> int:
    index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    bad = 0
    for _ in range(samples):
        layer, expert, role = rng.randrange(48), rng.randrange(512), rng.choice(list(PROJ))
        g = T["blk.%d.ffn_%s_exps.weight" % (layer, role)]
        s = T["blk.%d.ffn_%s_exps.scale" % (layer, role)]
        if g.tensor_type != Q.NVFP4:
            print("  blk.%d.ffn_%s_exps is %s, not NVFP4" % (layer, role, g.tensor_type.name))
            bad += 1
            continue
        raw = np.asarray(g.data).reshape(512, -1)[expert]                 # one expert's block_nvfp4 bytes
        rows = int(g.shape[1])                                             # ggml shape: [cols, rows, experts]
        cols = int(g.shape[0])
        got = quants.dequantize(np.ascontiguousarray(raw), Q.NVFP4).reshape(rows, cols).astype(np.float32)
        scale_file = float(np.asarray(s.data).reshape(-1)[expert])
        src = "model.language_model.layers.%d.mlp.experts.%d.%s.weight" % (layer, expert, PROJ[role])
        with safe_open(model / index[src], framework="pt") as f:
            w = f.get_tensor(src)
            sc = f.get_tensor(src.replace(".weight", ".weight_scale"))
            s2 = f.get_tensor(src.replace(".weight", ".weight_scale_2"))
        ref = modelopt_dequant(w, sc, s2)
        got = got * np.float32(scale_file)
        same = np.array_equal((ref + np.float32(0)).view(np.uint32), (got + np.float32(0)).view(np.uint32))
        scale_ok = scale_file == float(s2.float())
        print("  blk.%-2d expert %-3d %-4s  weights %s  scale %s (%.6g)"
              % (layer, expert, role, "bitwise" if same else "DIFFER ", "ok" if scale_ok else "WRONG", scale_file))
        bad += (not same) + (not scale_ok)
    return bad


def check_policy(T: dict, reference: pathlib.Path | None) -> int:
    bf16 = ("hc_attn_down", "hc_attn_up", "hc_attn_inject", "hc_ffn_down", "hc_ffn_up", "hc_ffn_inject",
            "ssm_alpha", "ssm_beta", "indexer.q_proj", "indexer.k_proj", "ple_value", "ple_key",
            "ffn_gate_inp", "ffn_gate_inp_shexp", "output_hc_down", "output_hc_up")
    bad = 0
    counts: dict[str, int] = {}
    for name, t in T.items():
        typ = t.tensor_type.name
        counts[typ] = counts.get(typ, 0) + 1
        stem = name.rsplit(".weight", 1)[0].split(".", 2)[-1] if name.startswith("blk.") else name.rsplit(".weight", 1)[0]
        if name.endswith("_exps.weight") and typ != "NVFP4":
            bad += 1; print("  policy: %s is %s" % (name, typ))
        elif any(name.endswith(b + ".weight") for b in bf16) and typ != "BF16":
            bad += 1; print("  policy: %s is %s, expected BF16" % (name, typ))
        elif name in ("token_embd.weight", "output.weight") and typ != "Q8_0":
            bad += 1; print("  policy: %s is %s, expected Q8_0" % (name, typ))
    print("  types: " + ", ".join("%s x%d" % kv for kv in sorted(counts.items(), key=lambda kv: -kv[1])))
    if reference and reference.exists():
        want = {line.rsplit(":", 1)[0].strip() for line in reference.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#") and ":" in line}
        missing = sorted(want - set(T))
        extra = sorted(n for n in set(T) - want if not n.endswith((".scale", ".input_scale")))
        print("  reference layout: %d names, missing here %d, unexpected here %d"
              % (len(want), len(missing), len(extra)))
        for n in (missing + extra)[:10]:
            print("   ", "missing" if n in missing else "extra  ", n)
        bad += len(missing) + len(extra)
    return bad


def check_ple(model: pathlib.Path, ple: pathlib.Path, samples: int, rng: random.Random) -> int:
    reader = GGUFReader(ple)
    t = {x.name: x for x in reader.tensors}.get("per_layer_token_embd.weight")
    if t is None:
        print("  PLE: per_layer_token_embd.weight not in", ple.name)
        return 1
    width, rows = int(t.shape[0]), int(t.shape[1])
    print("  PLE file: %s %s, %d rows x %d" % (ple.name, t.tensor_type.name, rows, width))
    if t.tensor_type != Q.IQ4_NL:
        print("  PLE: the engine reads IQ4_NL only")
        return 1
    table = np.asarray(t.data).reshape(rows, -1)
    index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    shard_rows = 2500012
    worst, sims = 1.0, []
    for _ in range(samples):
        r = rng.randrange(rows)
        shard, off = divmod(r, shard_rows)
        name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_%d.weight" % shard
        with safe_open(model / index[name], framework="pt") as f:
            src = f.get_slice(name)[off:off + 1].float().numpy().reshape(-1)
            sname = name.replace(".weight", ".weight_scale")
            if sname in f.keys():
                src = src * float(f.get_tensor(sname).float())
        mine = quants.dequantize(np.ascontiguousarray(table[r]), Q.IQ4_NL).reshape(-1)
        cos = float(np.dot(src, mine) / (np.linalg.norm(src) * np.linalg.norm(mine) + 1e-30))
        sims.append(cos)
        worst = min(worst, cos)
    print("  PLE rows vs FP8 source: cosine mean %.4f, worst %.4f over %d random rows"
          % (float(np.mean(sims)), worst, samples))
    return 0 if worst > 0.95 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ple-gguf")
    ap.add_argument("--reference", help="an rco-allocation.txt listing the reference tensor names")
    ap.add_argument("--samples", type=int, default=12)
    a = ap.parse_args()
    rng = random.Random(20260929)
    model = pathlib.Path(a.model)
    T = tensor_map(GGUFReader(a.gguf))
    print("GGUF: %d tensors" % len(T))
    print("experts, sampled:")
    bad = check_experts(model, T, a.samples, rng)
    print("dense policy:")
    bad += check_policy(T, pathlib.Path(a.reference) if a.reference else None)
    if a.ple_gguf:
        print("PLE table:")
        bad += check_ple(model, pathlib.Path(a.ple_gguf), 64, rng)
    print("RESULT:", "OK" if bad == 0 else "%d problems" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
