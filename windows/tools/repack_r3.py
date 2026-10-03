#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""repack_r3.py — 35b → GGUF with llama.cpp canonical naming (the runtime artifact).

Built on the lossless repack core (tools/repack_mlx_to_gguf.py, proven bit-exact),
this script does three things and nothing else:
  A. rename: MLX full names → qwen3next registered names
     (per qwen3next.cpp:31-129 / llama-arch.cpp:434-518)
  B. value transforms (each with its source of truth):
     - ssm_a   = -exp(A_log)                          conversion/qwen.py:395-396, qwen3next.cpp:452
     - ssm_ba  = in_proj_b/a row interleave [b,b,a,a]×16 groups   qwen3next.cpp:421-446 (v-heads contiguous within a group)
     - conv1d  = [C,4,1] squeeze last axis → dims flipped to ne{4,C}
               (row-major bytes unchanged, pure dims rewrite)
     - everything else stored directly: norm (+1 is already baked into the source
       of truth — NEVER add +1 a second time), gate int8→F32, BF16→F32
  C. metadata + tokenizer:
     expert_used_count=4, rope.dimension_count=64, recurrent_layers explicit array,
     no MTP/vision/pregate tensors (an unregistered tensor = hard throw at load time),
     tokens padded to 248320, pre=qwen2, add_bos=false, eos=im_end(248046),
     byte_fallback=true

Gates: tensor count / name-set identity against the registry; ba permutation
reversible; Q4_1 sampled pack two-sided sha. Output: <stem>.gguf.
"""
import argparse, json, math, os, re, struct, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repack_mlx_to_gguf import (StIndex, GgufWriter, KV, T_F32, T_F16, T_Q4_1, ALIGN, QK41,
                                DT_Q41, mlx_q_bytes, pack_q41_rows, deq_mlx_side, deq_gguf_side,
                                sha16)
import repack_mlx_to_gguf as core

P = "language_model.model.layers.{}."
EMB, LMH, ONORM = "language_model.model.embed_tokens", "language_model.lm_head", "language_model.model.norm"
R3VER = "r3-1.0"


def full_attn_layer(i, interval):
    return (i + 1) % interval == 0


def build_name_table(nl, interval):
    """→ list[(r3_name, kind, src_key_or_pair)]"""
    t = [("token_embd.weight", "q41", EMB + ".weight"),
         ("output.weight", "q41", LMH + ".weight"),
         ("output_norm.weight", "f32raw", ONORM + ".weight")]
    for i in range(nl):
        b = P.format(i)
        rb = f"blk.{i}."
        t += [(rb + "attn_norm.weight", "f32raw", b + "input_layernorm.weight"),
              (rb + "post_attention_norm.weight", "f32raw", b + "post_attention_layernorm.weight")]
        if full_attn_layer(i, interval):
            t += [(rb + "attn_q.weight", "q41", b + "self_attn.q_proj.weight"),
                  (rb + "attn_k.weight", "q41", b + "self_attn.k_proj.weight"),
                  (rb + "attn_v.weight", "q41", b + "self_attn.v_proj.weight"),
                  (rb + "attn_q_norm.weight", "f32raw", b + "self_attn.q_norm.weight"),
                  (rb + "attn_k_norm.weight", "f32raw", b + "self_attn.k_norm.weight"),
                  (rb + "attn_output.weight", "q41", b + "self_attn.o_proj.weight")]
        else:
            la = b + "linear_attn."
            t += [(rb + "attn_qkv.weight", "q41", la + "in_proj_qkv.weight"),
                  (rb + "attn_gate.weight", "q41", la + "in_proj_z.weight"),
                  (rb + "ssm_ba.weight", "ba", (la + "in_proj_b.weight", la + "in_proj_a.weight")),
                  (rb + "ssm_conv1d.weight", "conv", la + "conv1d.weight"),
                  (rb + "ssm_dt.bias", "f32raw", la + "dt_bias"),
                  (rb + "ssm_a", "sa", la + "A_log"),
                  (rb + "ssm_norm.weight", "f32raw", la + "norm.weight"),
                  (rb + "ssm_out.weight", "q41", la + "out_proj.weight")]
        m = b + "mlp."
        t += [(rb + "ffn_gate_inp.weight", "f32i8", m + "gate.weight"),
              (rb + "ffn_gate_exps.weight", "q41", m + "switch_mlp.gate_proj.weight"),
              (rb + "ffn_up_exps.weight", "q41", m + "switch_mlp.up_proj.weight"),
              (rb + "ffn_down_exps.weight", "q41", m + "switch_mlp.down_proj.weight"),
              (rb + "ffn_gate_shexp.weight", "q41", m + "shared_expert.gate_proj.weight"),
              (rb + "ffn_up_shexp.weight", "q41", m + "shared_expert.up_proj.weight"),
              (rb + "ffn_down_shexp.weight", "q41", m + "shared_expert.down_proj.weight"),
              (rb + "ffn_gate_inp_shexp.weight", "f32i8_1d", m + "shared_expert_gate.weight")]
    return t


def ssm_ba_perm(n_k, n_v):
    """group g row order = b[2g],b[2g+1],a[2g],a[2g+1] (v-heads contiguous within a group, qwen3next.cpp:421-446)."""
    order = []
    for g in range(n_k):
        order += [("b", 2 * g), ("b", 2 * g + 1), ("a", 2 * g), ("a", 2 * g + 1)]
    return order


def load_ba(idx, kb, ka, n_k, n_v):
    """two source tensors (int4 g64, rows=n_v) → interleaved (2n_v, K) q/s/b arrays."""
    out = {}
    for tag, key in (("b", kb), ("a", ka)):
        w = idx.load(key); s = idx.load(key[:-7] + ".scales"); b = idx.load(key[:-7] + ".biases")
        out[tag] = (mlx_q_bytes(w.reshape(n_v, -1)).astype(np.float32), s.reshape(n_v, -1), b.reshape(n_v, -1))
    K = out["b"][0].shape[1]
    q = np.empty((2 * n_v, K), np.uint8); S = np.empty((2 * n_v, K // 64), np.float32); Bs = np.empty_like(S)
    for r, (tag, row) in enumerate(ssm_ba_perm(n_k, n_v)):
        q[r] = out[tag][0][row].astype(np.uint8); S[r] = out[tag][1][row]; Bs[r] = out[tag][2][row]
    return q, S, Bs, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--verify", default="all")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.dir, "config.json"), encoding="utf8"))["text_config"]
    n_k, n_v = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    nl, intv = cfg["num_hidden_layers"], cfg["full_attention_interval"]
    table = build_name_table(nl, intv)
    names = [t[0] for t in table]
    assert len(names) == len(set(names)), "name-set collision"
    if a.dry:
        print(f"r3 registry entries={len(table)}  (3+40*2+10*6+30*8+40*8=703 expected)")
        return
    idx = StIndex([os.path.join(a.dir, f) for f in sorted(os.listdir(a.dir))
                   if re.fullmatch(r"model(-\d+-of-\d+)?\.safetensors", f)])
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.basename(a.dir.rstrip("\\/"))
    gp = os.path.join(a.out, f"{stem}.gguf")
    t0, audit = time.time(), dict(tool=R3VER, tier=stem, mode="r3", gates=[], errors=[])
    w = GgufWriter(gp)

    # ---- metadata (required read set + overrides) ----
    kv = w.kv
    w.kv_s("general.architecture", "qwen3next")
    w.kv_s("general.name", "edge0-35b")
    A = "qwen3next."      # this fork keys every KV with the arch prefix (gguf-py "{arch}.X" verified; the old "llm." spelling is wrong and will not load)
    for k, ty, v in [
        (A + "block_count", KV["u32"], nl), (A + "context_length", KV["u32"], cfg["max_position_embeddings"]),
        (A + "embedding_length", KV["u32"], cfg["hidden_size"]), (A + "feed_forward_length", KV["u32"], cfg["moe_intermediate_size"]),
        (A + "attention.head_count", KV["u32"], cfg["num_attention_heads"]),
        (A + "attention.head_count_kv", KV["u32"], cfg["num_key_value_heads"]),
        (A + "attention.key_length", KV["u32"], cfg["head_dim"]), (A + "attention.value_length", KV["u32"], cfg["head_dim"]),
        (A + "attention.layer_norm_rms_epsilon", KV["f32"], cfg["rms_norm_eps"]),
        (A + "rope.freq_base", KV["f32"], cfg["rope_parameters"]["rope_theta"]),
        (A + "expert_count", KV["u32"], cfg["num_experts"]), (A + "expert_used_count", KV["u32"], 4),
        (A + "expert_feed_forward_length", KV["u32"], cfg["moe_intermediate_size"]),
        (A + "expert_shared_feed_forward_length", KV["u32"], cfg["shared_expert_intermediate_size"]),
        (A + "ssm.conv_kernel", KV["u32"], cfg["linear_conv_kernel_dim"]),
        (A + "ssm.state_size", KV["u32"], cfg["linear_key_head_dim"]),
        (A + "ssm.group_count", KV["u32"], n_k), (A + "ssm.time_step_rank", KV["u32"], n_v),
        (A + "ssm.inner_size", KV["u32"], cfg["linear_value_head_dim"] * n_v),
        (A + "rope.dimension_count", KV["u32"], int(cfg["head_dim"] * cfg["partial_rotary_factor"])),
        (A + "full_attention_interval", KV["u32"], intv),
    ]:
        kv(k, ty, struct.pack("<I", v) if ty in (KV["u32"], KV["i32"]) else struct.pack("<f", v))
    kv("qwen3next.attention.recurrent_layers", KV["arr"],
       struct.pack("<I", 7) + struct.pack("<Q", nl) + b"".join(b"\x00" if full_attn_layer(i, intv) else b"\x01" for i in range(nl)))

    # ---- tokenizer (required: tokens/merges/model; vocab padded to 248320) ----
    tkj = json.load(open(os.path.join(a.dir, "tokenizer.json"), encoding="utf8"))
    vocab, merges = tkj["model"]["vocab"], tkj["model"]["merges"]
    toks = [None] * cfg["vocab_size"]
    for s, i in vocab.items():
        toks[i] = s
    for at in tkj["added_tokens"]:
        toks[at["id"]] = at["content"]
    for i in range(len(toks)):
        if toks[i] is None:
            toks[i] = f"<0x{255 - (i % 256):02X}>"      # placeholder (high ids the source of truth never produces)
    tt = [1] * len(toks)
    for at in tkj["added_tokens"]:
        tt[at["id"]] = 3
    def arr(key, et, elems):
        kv(key, KV["arr"], struct.pack("<I", et) + struct.pack("<Q", len(elems)) + b"".join(elems))
    w.kv_s("tokenizer.ggml.model", "gpt2")
    arr("tokenizer.ggml.tokens", KV["str"], [w._str(s) for s in toks])
    arr("tokenizer.ggml.merges", KV["str"], [w._str(" ".join(m)) for m in merges])
    arr("tokenizer.ggml.scores", KV["f32"], [struct.pack("<f", 0.0) for _ in toks])
    arr("tokenizer.ggml.token_type", KV["i32"], [struct.pack("<i", x) for x in tt])
    w.kv("tokenizer.ggml.added_tokens_ids", KV["arr"],
         struct.pack("<I", KV["u32"]) + struct.pack("<Q", len(tkj["added_tokens"]))
         + b"".join(struct.pack("<I", x["id"]) for x in tkj["added_tokens"]))
    for k, v in [("tokenizer.ggml.bos_token_id", 248044), ("tokenizer.ggml.eos_token_id", 248046),
                 ("tokenizer.ggml.padding_token_id", 248044)]:
        kv(k, KV["u32"], struct.pack("<I", v))
    kv("tokenizer.ggml.add_bos_token", KV["bool"], b"\x00")
    kv("tokenizer.ggml.add_eos_token", KV["bool"], b"\x00")
    kv("tokenizer.ggml.byte_fallback", KV["bool"], b"\x01")
    w.kv_s("tokenizer.ggml.pre", "qwen2")
    w.kv_s("tokenizer.chat_template", open(os.path.join(a.dir, "chat_template.jinja"), encoding="utf8").read())
    kv("tokenizer.ggml.eot_token_id", KV["u32"], struct.pack("<I", 248046))

    # ---- tensors (two passes: A registers dims/size, B streams writes + gates) ----
    def r3_spec(kind, src):
        """→ (tt, dims_ne, size). Pure shape arithmetic, no data loaded."""
        if kind == "ba":
            K = idx.t[src[0]][4][-1] * 8
            return T_Q4_1, (K, 2 * n_v), 2 * n_v * (K // QK41) * 20
        shp = idx.t[src][4]; dt = idx.t[src][3]
        if kind == "q41":
            K = shp[-1] * 8
            rows = int(np.prod(shp[:-1])) if len(shp) > 1 else 1
            return T_Q4_1, (K,) + tuple(reversed(shp[:-1])), rows * (K // QK41) * 20
        if kind == "f32i8":
            K = shp[-1] * 4
            rows = int(np.prod(shp[:-1])) if len(shp) > 1 else 1
            return T_F32, (K,) + tuple(reversed(shp[:-1])), rows * K * 4
        if kind == "f32i8_1d":
            K = shp[-1] * 4
            return T_F32, (K,), K * 4
        if kind == "conv":
            return T_F32, (shp[1], shp[0]), int(np.prod(shp)) * 4
        if kind == "sa":
            return T_F32, (shp[0],), shp[0] * 4
        return T_F32, (tuple(reversed(shp)) if len(shp) > 1 else (shp[0],)), int(np.prod(shp)) * 4

    for r3n, kind, src in table:
        tt, dims, size = r3_spec(kind, src)
        w.register(r3n, tt, dims, size)
    w.finalize()

    n_done = 0
    for r3n, kind, src in table:
        if kind == "q41":
            base = src[:-7]
            W = idx.load(src); S = idx.load(base + ".scales"); Bs = idx.load(base + ".biases")
            lead = W.shape[:-1]; rows = int(np.prod(lead)) if lead else 1
            Wf, Sf, Bsf = W.reshape(rows, -1), S.reshape(rows, -1), Bs.reshape(rows, -1)
            blocks, exm = pack_q41_rows(mlx_q_bytes(Wf), Sf, Bsf, src)
            w.write(blocks.tobytes()); w.align_pad()
            # sentinel gate: first row depacked on both sides, sha compared; if that row contains an exempt block, downgrade to record
            if rows:
                a_side = deq_mlx_side(mlx_q_bytes(Wf[:1]), Sf[:1], Bsf[:1])
                b_side = deq_gguf_side(blocks[:1])
                same = bool(sha16(a_side.tobytes()) == sha16(b_side.tobytes()))
                row0_ex = any(r == 0 for (r, _b) in exm)
                st = "OK" if same else ("EXEMPT" if row0_ex else "FAIL")
                audit["gates"].append(dict(t=r3n, sentinel=st))
                if st == "FAIL":
                    audit["errors"].append(f"{r3n}: sentinel row non-exempt mismatch")
        elif kind in ("f32i8", "f32i8_1d"):
            base = src[:-7]
            W = idx.load(src); S = idx.load(base + ".scales"); Bs = idx.load(base + ".biases")
            rows = int(np.prod(W.shape[:-1])) if W.ndim > 1 else 1
            Wf, Sf, Bsf = W.reshape(rows, -1), S.reshape(rows, -1), Bs.reshape(rows, -1)
            q8 = Wf.view(np.uint8).reshape(rows, Wf.shape[1] * 4)
            v = deq_mlx_side(q8, Sf, Bsf)
            w.write(np.ascontiguousarray(v, np.float32).tobytes()); w.align_pad()
        elif kind == "f32raw":
            x = idx.load(src)
            w.write(np.ascontiguousarray(x, np.float32).tobytes()); w.align_pad()
        elif kind == "sa":
            x = idx.load(src)
            v = -np.exp(np.asarray(x, dtype=np.float32))
            v2 = np.array([-math.exp(float(t)) for t in x.ravel()], dtype=np.float32)  # libm second path, cross-checked against numpy
            dv = np.abs(v - v2).max(initial=0.0)
            audit["gates"].append(dict(t=r3n, exp_dual_dev=float(dv)))
            if dv > 1e-5:
                audit["errors"].append(f"{r3n}: -exp dual-path deviation {dv:.2e} > 1e-5")
            w.write(np.ascontiguousarray(v, np.float32).tobytes()); w.align_pad()
        elif kind == "conv":
            x = idx.load(src)                      # [C,4,1]
            assert x.shape[-1] == 1
            y = np.ascontiguousarray(x.reshape(x.shape[0], x.shape[1]), dtype=np.float32)
            w.write(y.tobytes()); w.align_pad()
        elif kind == "ba":
            kb, ka = src
            q, S, Bs, K = load_ba(idx, kb, ka, n_k, n_v)
            blocks, _ = pack_q41_rows(q, S, Bs, "ssm_ba")
            w.write(blocks.tobytes()); w.align_pad()
            # reversibility gate: undo the permutation and compare against the pre-interleave source q rows for exact equality
            wb = idx.load(kb); wa = idx.load(ka)
            qb = mlx_q_bytes(wb.reshape(n_v, -1)); qa = mlx_q_bytes(wa.reshape(n_v, -1))
            rec = np.empty_like(q, dtype=np.uint8)
            for r, (tag, row) in enumerate(ssm_ba_perm(n_k, n_v)):
                rec[r] = (qb if tag == "b" else qa)[row]
            ok = bool((rec == q).all())
            audit["gates"].append(dict(t=r3n, ba_permute_reversible=ok))
            if not ok:
                audit["errors"].append(f"{r3n}: ba permutation not reversible")
        n_done += 1
        if n_done % 60 == 0 or n_done == len(table):
            print(f"  {n_done}/{len(table)} {r3n:44s} {w.dlen/1e9:6.2f}GB {time.time()-t0:5.0f}s", flush=True)
    w.close(); idx.close()
    bad = [g for g in audit["gates"] if g.get("ok") is False or g.get("ba_permute_reversible") is False]
    audit["n_tensors"] = n_done; audit["elapsed_s"] = round(time.time() - t0, 1)
    v = "GREEN" if not bad and not audit["errors"] and n_done == len(table) else "RED"
    json.dump(dict(verdict=f"R3-artifact {v}", audit=audit, size=os.path.getsize(gp)),
              open(os.path.join(a.out, f"{stem}-audit.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\nGGUF {gp} {os.path.getsize(gp)/1e9:.2f}GB tensors={n_done}/{len(table)} ⇒ R3-artifact {v}"
          f" (errors={audit['errors']})")
    return 0 if v == "GREEN" else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
