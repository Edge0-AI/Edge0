#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lora_mlx_to_gguf.py — edge0 LoRA (MLX safetensors, peft shape) → llama-adapter GGUF.

Basis (verified against fork @7ab4ee7):
  naming law: adapter tensor names must == base weight names + ".lora_a/.lora_b"
              (llama-adapter.cpp:333 hard lookup)
  shape law:  A declared dims = reversed(torch [r,K]) = ggml ne [K,r]; B = [r,N]
              (:365-370 checks; torch row-major bytes untouched, only the dims
              declaration flips)
  scale law:  runtime scale = adapter.lora.alpha ? s·alpha/rank : s
              (llama-adapter.h:53, rank = B.ne0)
              source-of-truth scale = alpha/r = 32/16 = 2.0 ⇒ write
              adapter.lora.alpha=32.0 so plain `--lora FILE` (default s=1.0) is the same expression
  ssm_ba special tensor: the model side interleaves+merges in_proj_b/a
              (qwen3next.cpp:421-438) and the two source LoRAs have different A matrices
              ⇒ block-diagonal merge: A=[32,2048] (top 16 rows = A_b, bottom 16 = A_a),
              B=[64,32] with rows placed per the interleave; rank=32 ⇒ default
              scale=32/32=1.0 while the source needs 2.0 per path ⇒ **B entries ×2
              compensation** (linear value scaling, NOT requantization).
  arch law:   general.architecture must match the base (:212 hard compare).
"""
import argparse, os, re, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repack_mlx_to_gguf import StIndex, GgufWriter, KV, T_F16, bf16_to_f32
from repack_r3 import ssm_ba_perm

RENAME = [
    (re.compile(r"^language_model\.model\.layers\.(\d+)\.linear_attn\.in_proj_qkv$"), r"blk.\1.attn_qkv.weight"),
    (re.compile(r"^language_model\.model\.layers\.(\d+)\.linear_attn\.in_proj_z$"), r"blk.\1.attn_gate.weight"),
    (re.compile(r"^language_model\.model\.layers\.(\d+)\.linear_attn\.out_proj$"), r"blk.\1.ssm_out.weight"),
    (re.compile(r"^language_model\.model\.layers\.(\d+)\.self_attn\.q_proj$"), r"blk.\1.attn_q.weight"),
    (re.compile(r"^language_model\.model\.layers\.(\d+)\.self_attn\.k_proj$"), r"blk.\1.attn_k.weight"),
    (re.compile(r"^language_model\.model\.layers\.(\d+)\.self_attn\.v_proj$"), r"blk.\1.attn_v.weight"),
    (re.compile(r"^language_model\.model\.layers\.(\d+)\.self_attn\.o_proj$"), r"blk.\1.attn_output.weight"),
    (re.compile(r"^language_model\.model\.layers\.(\d+)\.mlp\.shared_expert\.gate_proj$"), r"blk.\1.ffn_gate_shexp.weight"),
    (re.compile(r"^language_model\.model\.layers\.(\d+)\.mlp\.shared_expert\.up_proj$"), r"blk.\1.ffn_up_shexp.weight"),
    (re.compile(r"^language_model\.model\.layers\.(\d+)\.mlp\.shared_expert\.down_proj$"), r"blk.\1.ffn_down_shexp.weight"),
]
BA = re.compile(r"^language_model\.model\.layers\.(\d+)\.linear_attn\.in_proj_([ab])$")


def f16(idx, name):
    a = idx.load(name)
    return np.ascontiguousarray(a, dtype=np.float16)


RENAME_8B = [
    (re.compile(r"^model\.layers\.(\d+)\.attention\.q_proj$"), r"blk.\1.attn_q.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.attention\.k_proj$"), r"blk.\1.attn_k.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.attention\.v_proj$"), r"blk.\1.attn_v.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.attention\.o_proj$"), r"blk.\1.attn_output.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.attention\.dense$"), r"blk.\1.attn_output.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.attention\.f_proj$"), r"blk.\1.ssm_f_a.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.attention\.b_proj$"), r"blk.\1.ssm_beta.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.attention\.q_a_proj$"), r"blk.\1.attn_q_a.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.attention\.q_b_proj$"), r"blk.\1.attn_q_b.weight"),
    # g_proj routes by layer type (KDA→ssm_g_a / MLA→attn_gate), decided via FULL() in main
    (re.compile(r"^model\.layers\.(\d+)\.attention\.g_proj$"), r"G"),
    # kv_b_proj targets a 3D split tensor (unimplemented upstream-side); never touched by --only experiments; plain conversions skip it and report
    (re.compile(r"^model\.layers\.(\d+)\.attention\.kv_b_proj$"), r"SKIP:kv_b_3d"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate_proj$"), r"blk.\1.ffn_gate.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.up_proj$"), r"blk.\1.ffn_up.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj$"), r"blk.\1.ffn_down.weight"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--alpha", type=float, default=32.0)   # = alpha baked in the source artifact; plain --lora (default scale 1.0) then reproduces the same expression
    ap.add_argument("--only", default=None, help="generate only families whose base name matches this regex (single-variable experiments)")
    ap.add_argument("--out-suffix", default="-gguf", help="infix in the output name")
    a = ap.parse_args()
    stem = os.path.basename(a.dir.rstrip("\\/"))
    is8 = "8b" in stem
    sfx = stem.replace("-", "_")          # artifact names use underscores: lora_edge0_35b.safetensors
    lora_path = os.path.join(a.dir, f"lora_{sfx}.safetensors")
    out = os.path.join(a.dir, f"lora_{sfx}{a.out_suffix}.gguf")
    idx = StIndex([lora_path])
    cfg = __import__("json").load(open(os.path.join(a.dir, "config.json"), encoding="utf8"))
    tc = cfg.get("text_config", cfg)
    n_v = tc.get("linear_num_value_heads", 32)
    n_k = tc.get("linear_num_key_heads", 16)
    only = re.compile(a.only) if a.only else None
    skipped = []

    simple, ba_pairs = {}, {}
    for name in idx.t:
        base = re.sub(r"\.lora_[AB]$", "", name)
        if only and not only.search(base):
            skipped.append(base)
            continue
        ab = "a" if name.endswith(".lora_A") else "b"
        if not is8:
            m = BA.match(base)
            if m:
                ba_pairs.setdefault((int(m.group(1)), m.group(2)), {})[ab] = name
                continue
        for rx, tpl in (RENAME_8B if is8 else RENAME):
            mm = rx.match(base)
            if mm:
                if tpl.startswith("SKIP:"):
                    skipped.append(base + " [3D split tensor unimplemented]")
                    break
                if tpl == "G":                  # 8b g_proj routes by layer type
                    il = int(mm.group(1))
                    dst = (f"blk.{il}.attn_gate.weight" if (il + 1) % 4 == 0
                           else f"blk.{il}.ssm_g_a.weight")
                else:
                    dst = rx.sub(tpl, base)
                simple.setdefault(dst, {})[ab] = name
                break
        else:
            if "pregate" not in base and "mlp.gate" not in base:
                raise SystemExit(f"unmapped LoRA family: {base}")
    if skipped:
        print("skip:", sorted(set(skipped))[:6], f"({len(skipped)} entries)")

    entries = []   # (model_name, A_f16 [r,K] torch, B_f16 [N,r] torch)
    for mname, d in sorted(simple.items()):
        A = idx.load(d["a"]); B = idx.load(d["b"])
        assert A.shape[0] == B.shape[1], (mname, A.shape, B.shape)
        entries.append((mname, A, B))
    # ssm_ba block-diagonal merge (with alpha/rank compensation: B ×(32/alpha·alpha/16→2))
    comp = 2.0     # target scale per path = 2.0; merged rank=32 ⇒ get_scale=alpha·s/32=1.0 ⇒ B needs ×2
    for il in sorted({k[0] for k in ba_pairs}):
        db, da = ba_pairs[(il, "b")], ba_pairs[(il, "a")]
        Ab, Bb = idx.load(db["a"]), idx.load(db["b"])
        Aa, Ba = idx.load(da["a"]), idx.load(da["b"])
        r = Ab.shape[0]
        A = np.concatenate([Ab, Aa], 0)                       # [2r, K]
        B = np.zeros((2 * n_v, 2 * r), dtype=np.float32)
        for out_row, (tag, j) in enumerate(ssm_ba_perm(n_k, n_v)):
            src_mat, src_row = (Bb, j) if tag == "b" else (Ba, j)
            cols = slice(0, r) if tag == "b" else slice(r, 2 * r)
            B[out_row, cols] = np.asarray(src_mat[src_row], dtype=np.float32) * comp
        entries.append((f"blk.{il}.ssm_ba.weight", A, B.astype(np.float16)))

    w = GgufWriter(out)
    w.kv_s("general.architecture", "bailingmoe3" if is8 else "qwen3next")
    w.kv_s("general.type", "adapter")
    w.kv_s("general.name", f"{stem}-lora (repack of lora_{stem}.safetensors)")
    w.kv_s("adapter.type", "lora")
    w.kv_s("adapter.name", f"edge0-{stem}")
    w.kv("adapter.lora.alpha", KV["f32"], struct.pack("<f", a.alpha))
    for mname, A, B in entries:
        w.register(mname + ".lora_a", T_F16, (A.shape[1], A.shape[0]), A.size * 2)
        w.register(mname + ".lora_b", T_F16, (B.shape[1], B.shape[0]), B.size * 2)
    w.finalize()
    for mname, A, B in entries:
        w.write(np.ascontiguousarray(A, np.float16).tobytes()); w.align_pad()
        w.write(np.ascontiguousarray(B, np.float16).tobytes()); w.align_pad()
    w.close(); idx.close()
    print(f"adapter: {out}  tensors={2*len(entries)}  pairs={len(entries)} (incl. ssm_ba merge ×{comp} compensation)")
    verify_against_base(out, "edge0-8b" if is8 else "edge0-35b")


def verify_against_base(adapter_path, base_stem):
    """Cross-check gate: right after writing, verify (1) adapter names ⊆ base names
    (2) the upstream shape three-law (llama-adapter.cpp:358-371: a.ne0=base.ne0 ∧
    b.ne1=base.ne1 ∧ a.ne1=b.ne0; token_embd flip is a special case).
    Any miss/shape violation = hard throw, preventing the 'loads without error but
    never enters the graph' failure mode from recurring."""
    # gguf-py comes from the pinned vendored upstream (monorepo layout or bootstrap depot);
    # standalone checkouts fall back to an installed `gguf` package on sys.path.
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(_here)
    for _p in [os.path.join(x, "vendor", "llama.cpp", "gguf-py") for x in
               ([os.environ["EDGE0_DEPOT"]] if os.environ.get("EDGE0_DEPOT") else [])
               + [os.path.join(_repo, os.pardir), os.path.join(_repo, ".edge0")]]:
        if os.path.isdir(_p):
            sys.path.insert(0, _p)
            break
    from gguf import GGUFReader
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.join(root, "models", f"{base_stem}-gguf", f"{base_stem}.gguf")
    bt = {str(t.name): [int(x) for x in t.shape]
          for t in GGUFReader(base_path).tensors}
    ad = GGUFReader(adapter_path)
    A = {str(t.name)[:-7]: [int(x) for x in t.shape] for t in ad.tensors if str(t.name).endswith(".lora_a")}
    B = {str(t.name)[:-7]: [int(x) for x in t.shape] for t in ad.tensors if str(t.name).endswith(".lora_b")}
    miss, shape_bad = [], []
    for stem in sorted(A.keys() & B.keys()):
        b = bt.get(stem)
        if b is None:
            miss.append(stem); continue
        a, bb = A[stem], B[stem]
        if stem.startswith("token_embd"):
            ok = b[0] == bb[1] and b[1] == a[1] and a[1] == bb[0]   # token_embd flip special case
        else:
            ok = b[0] == a[0] and b[1] == bb[1] and a[1] == bb[0]
        if not ok:
            shape_bad.append((stem, f"base={b} a={a} b={bb}"))
    if miss or shape_bad:
        raise SystemExit(f"cross-check gate RED: miss={len(miss)} {miss[:5]} shape_bad={len(shape_bad)} {shape_bad[:5]}")
    print(f"cross-check gate GREEN: {len(A.keys() & B.keys())} pairs all hit base {os.path.basename(base_path)} + shape three-law satisfied")


if __name__ == "__main__":
    main()
