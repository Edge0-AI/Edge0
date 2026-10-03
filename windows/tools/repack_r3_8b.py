#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""repack_r3_8b.py — 8b (BailingMoeV3/Ling-3) → bailingmoe3 canonical-naming GGUF.

The contract is the full reference table (conversion/bailingmoe3.py +
llama-arch.cpp + bailingmoe3.cpp):
  name strings (llama-arch.cpp table as-is): attn_q_a/attn_q_a_norm/attn_q_b/
    attn_kv_a_mqa/attn_kv_a_norm/attn_k_b/attn_v_b/attn_gate/
    ssm_conv1d_{q,k,v}/ssm_a(no .weight)/ssm_dt(.bias)/ssm_beta/ssm_f_a/ssm_g_a/
    ssm_norm/exp_probs_b(.bias)
  transforms (checked line-by-line against bailingmoe3.py):
    conv [C,1,4] → reshape(1,C,1,4) → ne{4,1,C,1}, bytes unchanged, F32
    ssm_a = torch.exp(A_log).reshape(-1,1)   ← NOTE +exp here (35b uses −exp; do not mix them up)
    dt_bias [2048] elementwise → ssm_dt.bias stored directly as F32
    kv_b_proj → split + transpose (view(16,256,512) → [128 noope | 128 v] → k transpose(1,2))
      k_b/v_b are stored as **F32, not Q4_1**: the transpose scatters the quantization
      grouping axis (dim0=128 spans 128 original rows), so lossless repack is
      physically impossible; 6 layers x 8MB = 48MB cost, negligible.
    f_proj→ssm_f_a; g_proj→(KDA)ssm_g_a /(MLA)attn_gate; dense(MLA)→attn_output
    gate BF16→F32 stored directly (8b gate is NOT quantized, unlike 35b int8);
    expert_bias→exp_probs_b.bias F32
    all norms stored directly (bailing_hybrid sanitize has no +1 and no moveaxis;
    the source of truth is the final value multiplied directly)
  metadata: head_count_kv is a 24-element array ((i+1)%4==0 →1 else 0, bailingmoe3.py:55-56);
    key_length=576(=kv_lora+qk_rope) value_length=128 key_length_mla=192 value_length_mla=128;
    expert_gating_func=2(SIGMOID, llama-hparams.h:19) expert_weights_scale=2.5
    (config routed_scaling_factor) expert_weights_norm=true; kda.head_dim=128
    kda.safe_gate=true kda.gate_lower_bound=-5; leading_dense_block_count=1;
    rope.dimension_count=64; **tokenizer.ggml.pre="bailingmoe2"** (verified by hashing
    tokenizer.json against the fork, NOT qwen2); architecture string
    general.architecture=bailingmoe3.
pregate x66 stripped (sidecar feeds the prefetch router). Gates: tensor-count
identity / sentinel sha / kv_b transform reversibility spot-check / exp dual-path.
"""
import argparse, json, math, os, re, struct, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repack_mlx_to_gguf import (StIndex, GgufWriter, KV, T_F32, T_F16, T_Q4_1, QK41, DT_Q41,
                                mlx_q_bytes, pack_q41_rows, deq_mlx_side, deq_gguf_side, sha16)

FULL = lambda i: (i + 1) % 4 == 0            # MLA layer (bailingmoe3.py:39-42 same rule)
NL, E, KDA_DIM, HEADS = 24, 128, 2048, 16
R3VER = "r3-8b-1.0"


def q41_src(idx, name):
    W = idx.load(name); base = name[:-7]
    return W, idx.load(base + ".scales"), idx.load(base + ".biases")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="models/edge0-8b")
    ap.add_argument("--out", default="models/edge0-8b-gguf")
    a = ap.parse_args()
    idx = StIndex([os.path.join(a.dir, f) for f in sorted(os.listdir(a.dir))
                   if re.fullmatch(r"model(-\d+-of-\d+)?\.safetensors", f)])
    cfg = json.load(open(os.path.join(a.dir, "config.json"), encoding="utf8"))
    tkj = json.load(open(os.path.join(a.dir, "tokenizer.json"), encoding="utf8"))
    A = "bailingmoe3."
    P = "model.layers.{}."
    ATT = P + "attention."
    MLP = P + "mlp."

    # ---- tensor table: (dst_name, kind, src); kind ∈ q41/f32raw/conv/sa/kvb ----
    table = [("token_embd.weight", "q41", "model.word_embeddings.weight"),
             ("output.weight", "q41", "lm_head.weight"),
             ("output_norm.weight", "f32raw", "model.norm.weight")]
    for i in range(NL):
        at_, mp = ATT.format(i), MLP.format(i)
        table += [(f"blk.{i}.attn_norm.weight", "f32raw", P.format(i) + "input_layernorm.weight"),
                  (f"blk.{i}.ffn_norm.weight", "f32raw", P.format(i) + "post_attention_layernorm.weight")]
        if FULL(i):
            table += [(f"blk.{i}.attn_q_a.weight", "q41", at_ + "q_a_proj.weight"),
                      (f"blk.{i}.attn_q_a_norm.weight", "f32raw", at_ + "q_a_layernorm.weight"),
                      (f"blk.{i}.attn_q_b.weight", "q41", at_ + "q_b_proj.weight"),
                      (f"blk.{i}.attn_kv_a_mqa.weight", "q41", at_ + "kv_a_proj_with_mqa.weight"),
                      (f"blk.{i}.attn_kv_a_norm.weight", "f32raw", at_ + "kv_a_layernorm.weight"),
                      (f"blk.{i}.attn_k_b.weight", "kvb_k", at_ + "kv_b_proj.weight"),
                      (f"blk.{i}.attn_v_b.weight", "kvb_v", at_ + "kv_b_proj.weight"),
                      (f"blk.{i}.attn_gate.weight", "q41", at_ + "g_proj.weight"),
                      (f"blk.{i}.attn_output.weight", "q41", at_ + "dense.weight")]
        else:
            table += [(f"blk.{i}.attn_q.weight", "q41", at_ + "q_proj.weight"),
                      (f"blk.{i}.attn_k.weight", "q41", at_ + "k_proj.weight"),
                      (f"blk.{i}.attn_v.weight", "q41", at_ + "v_proj.weight"),
                      (f"blk.{i}.ssm_conv1d_q.weight", "conv", at_ + "q_conv1d.weight"),
                      (f"blk.{i}.ssm_conv1d_k.weight", "conv", at_ + "k_conv1d.weight"),
                      (f"blk.{i}.ssm_conv1d_v.weight", "conv", at_ + "v_conv1d.weight"),
                      (f"blk.{i}.ssm_a", "sa", at_ + "A_log"),
                      (f"blk.{i}.ssm_dt.bias", "f32raw", at_ + "dt_bias"),
                      (f"blk.{i}.ssm_beta.weight", "q41", at_ + "b_proj.weight"),
                      (f"blk.{i}.ssm_f_a.weight", "q41", at_ + "f_proj.weight"),
                      (f"blk.{i}.ssm_g_a.weight", "q41", at_ + "g_proj.weight"),
                      (f"blk.{i}.ssm_norm.weight", "f32raw", at_ + "o_norm.weight"),
                      (f"blk.{i}.attn_output.weight", "q41", at_ + "o_proj.weight")]
        if i == 0:
            table += [(f"blk.0.ffn_gate.weight", "q41", mp + "gate_proj.weight"),
                      (f"blk.0.ffn_up.weight", "q41", mp + "up_proj.weight"),
                      (f"blk.0.ffn_down.weight", "q41", mp + "down_proj.weight")]
        else:
            table += [(f"blk.{i}.ffn_gate_inp.weight", "f32raw", mp + "gate.weight"),
                      (f"blk.{i}.exp_probs_b.bias", "f32raw", mp + "gate.expert_bias"),
                      (f"blk.{i}.ffn_gate_exps.weight", "q41", mp + "experts.gate_proj.weight"),
                      (f"blk.{i}.ffn_up_exps.weight", "q41", mp + "experts.up_proj.weight"),
                      (f"blk.{i}.ffn_down_exps.weight", "q41", mp + "experts.down_proj.weight"),
                      (f"blk.{i}.ffn_gate_shexp.weight", "q41", mp + "shared_experts.gate_proj.weight"),
                      (f"blk.{i}.ffn_up_shexp.weight", "q41", mp + "shared_experts.up_proj.weight"),
                      (f"blk.{i}.ffn_down_shexp.weight", "q41", mp + "shared_experts.down_proj.weight")]
    miss = [t for t in table if t[2] not in idx.t]
    assert not miss, f"source tensor missing: {miss[:5]}"
    names = [t[0] for t in table]
    assert len(names) == len(set(names))

    # ---- metadata ----
    os.makedirs(a.out, exist_ok=True)
    stem = "edge0-8b"
    gp = os.path.join(a.out, f"{stem}.gguf")
    w = GgufWriter(gp)
    w.kv_s("general.architecture", "bailingmoe3")
    w.kv_s("general.type", "model")
    w.kv_s("general.name", "edge0-8b")
    w.kv("general.alignment", KV["u32"], struct.pack("<I", 32))
    u = lambda v: struct.pack("<I", v)
    f32 = lambda v: struct.pack("<f", v)
    b1 = lambda v: b"\x01" if v else b"\x00"
    for k, ty, v in [
        (A + "block_count", KV["u32"], u(NL)), (A + "context_length", KV["u32"], u(cfg["max_position_embeddings"])),
        (A + "embedding_length", KV["u32"], u(cfg["hidden_size"])), (A + "feed_forward_length", KV["u32"], u(cfg["intermediate_size"])),
        (A + "attention.head_count", KV["u32"], u(HEADS)),
        (A + "attention.key_length", KV["u32"], u(cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"])),
        (A + "attention.value_length", KV["u32"], u(cfg["head_dim"])),
        (A + "attention.key_length_mla", KV["u32"], u(cfg["qk_head_dim"])),
        (A + "attention.value_length_mla", KV["u32"], u(cfg["v_head_dim"])),
        (A + "attention.q_lora_rank", KV["u32"], u(cfg["q_lora_rank"])),
        (A + "attention.kv_lora_rank", KV["u32"], u(cfg["kv_lora_rank"])),
        (A + "rope.freq_base", KV["f32"], f32(float(cfg["rope_theta"]))),
        (A + "rope.dimension_count", KV["u32"], u(cfg["qk_rope_head_dim"])),
        (A + "attention.layer_norm_rms_epsilon", KV["f32"], f32(cfg["rms_norm_eps"])),
        (A + "vocab_size", KV["u32"], u(cfg["vocab_size"])),
        (A + "ssm.conv_kernel", KV["u32"], u(cfg["short_conv_kernel_size"])),
        (A + "kda.head_dim", KV["u32"], u(cfg["head_dim"])),
        (A + "expert_count", KV["u32"], u(E)), (A + "expert_used_count", KV["u32"], u(cfg["num_experts_per_tok"])),
        (A + "expert_group_count", KV["u32"], u(cfg["n_group"])), (A + "expert_group_used_count", KV["u32"], u(cfg["topk_group"])),
        (A + "expert_gating_func", KV["u32"], u(2)),
        (A + "expert_weights_scale", KV["f32"], f32(cfg["routed_scaling_factor"])),
        (A + "expert_feed_forward_length", KV["u32"], u(cfg["moe_intermediate_size"])),
        (A + "expert_shared_feed_forward_length", KV["u32"], u(cfg.get("shared_expert_intermediate_size", cfg["moe_intermediate_size"]))),
        (A + "expert_shared_count", KV["u32"], u(1)),
        (A + "leading_dense_block_count", KV["u32"], u(cfg["first_k_dense_replace"])),
    ]:
        w.kv(k, ty, v)
    w.kv(A + "kda.safe_gate", KV["bool"], b1(True))
    w.kv(A + "kda.gate_lower_bound", KV["f32"], f32(cfg["kda_lower_bound"]))
    w.kv(A + "expert_weights_norm", KV["bool"], b1(cfg.get("norm_topk_prob", True)))
    w.kv(A + "attention.head_count_kv", KV["arr"], struct.pack("<I", KV["u32"]) + struct.pack("<Q", NL)
         + b"".join(u(1 if FULL(i) else 0) for i in range(NL)))
    w.kv_s("tokenizer.ggml.model", "gpt2")
    w.kv_s("tokenizer.ggml.pre", "bailingmoe2")

    # tokenizer arrays
    vocab = tkj["model"]["vocab"]; toks = [None] * cfg["vocab_size"]
    for s, i2 in vocab.items():
        toks[i2] = s
    for at in tkj["added_tokens"]:
        toks[at["id"]] = at["content"]
    tt = [1] * len(toks)
    for at in tkj["added_tokens"]:
        tt[at["id"]] = 3
    n_pad = sum(1 for x in toks if x is None)
    for j, x in enumerate(toks):
        if x is None:
            toks[j] = f"<0x{j:06X}>"

    def arr(key, et, elems):
        w.kv(key, KV["arr"], struct.pack("<I", et) + struct.pack("<Q", len(elems)) + b"".join(elems))
    arr("tokenizer.ggml.tokens", KV["str"], [w._str(s) for s in toks])
    arr("tokenizer.ggml.merges", KV["str"],
        [w._str(" ".join(m) if isinstance(m, list) else m) for m in tkj["model"]["merges"]])
    arr("tokenizer.ggml.scores", KV["f32"], [f32(0.0) for _ in toks])
    arr("tokenizer.ggml.token_type", KV["i32"], [struct.pack("<i", x) for x in tt])
    w.kv("tokenizer.ggml.bos_token_id", KV["u32"], u(156891))
    w.kv("tokenizer.ggml.eos_token_id", KV["u32"], u(cfg.get("eos_token_id", 156895)))
    w.kv("tokenizer.ggml.padding_token_id", KV["u32"], u(cfg.get("pad_token_id", 156892)))
    w.kv("tokenizer.ggml.add_bos_token", KV["bool"], b1(False))
    w.kv("tokenizer.ggml.add_eos_token", KV["bool"], b1(False))
    w.kv("tokenizer.ggml.byte_fallback", KV["bool"], b1(True))
    ct = os.path.join(a.dir, "chat_template.jinja")
    if os.path.isfile(ct):
        w.kv_s("tokenizer.chat_template", open(ct, encoding="utf8").read())

    # ---- two-pass tensors ----
    def spec(kind, src):
        shp = idx.t[src][4]
        if kind == "kvb_k":
            return T_F32, (cfg["qk_nope_head_dim"], cfg["kv_lora_rank"], HEADS), HEADS * cfg["kv_lora_rank"] * cfg["qk_nope_head_dim"] * 4
        if kind == "kvb_v":
            return T_F32, (cfg["kv_lora_rank"], cfg["v_head_dim"], HEADS), HEADS * cfg["kv_lora_rank"] * cfg["v_head_dim"] * 4
        if kind == "conv":
            return T_F32, (shp[2], 1, shp[0], 1), int(np.prod(shp)) * 4
        if kind == "sa":
            return T_F32, (1, shp[0]), shp[0] * 4          # bailingmoe3.cpp:89 create {1, n_head}
        if kind == "f32raw":
            return T_F32, (tuple(reversed(shp)) if len(shp) > 1 else (shp[0],)), int(np.prod(shp)) * 4
        K = shp[-1] * 8
        rows = int(np.prod(shp[:-1])) if len(shp) > 1 else 1
        return T_Q4_1, (K,) + tuple(reversed(shp[:-1])), rows * (K // QK41) * 20

    for dst, kind, src in table:
        tt_, dims, size = spec(kind, src)
        w.register(dst, tt_, dims, size)
    ds = w.finalize()
    audit = dict(tool=R3VER, gates=[], errors=[])
    t0 = time.time()
    for n, (dst, kind, src) in enumerate(table):
        if kind == "q41":
            W, S, Bs = q41_src(idx, src)
            rows = int(np.prod(W.shape[:-1])) if W.ndim > 1 else 1
            Wf, Sf, Bsf = W.reshape(rows, -1), S.reshape(rows, -1), Bs.reshape(rows, -1)
            blocks, exm = pack_q41_rows(mlx_q_bytes(Wf), Sf, Bsf, src)
            w.write(blocks.tobytes()); w.align_pad()
            a_side = deq_mlx_side(mlx_q_bytes(Wf[:1]), Sf[:1], Bsf[:1])
            same = sha16(a_side.tobytes()) == sha16(deq_gguf_side(blocks[:1]).tobytes())
            st = "OK" if same else ("EXEMPT" if any(r == 0 for r, _ in exm) else "FAIL")
            audit["gates"].append((dst, st))
            if st == "FAIL":
                audit["errors"].append(dst)
        elif kind == "kvb_k" or kind == "kvb_v":
            W, S, Bs = q41_src(idx, src)
            K = W.shape[-1] * 8
            q = mlx_q_bytes(W.reshape(4096, -1))
            v = deq_mlx_side(q, S.reshape(4096, -1), Bs.reshape(4096, -1))   # (4096,512)
            kv = v.reshape(HEADS, 256, K)
            kt = np.ascontiguousarray(kv[:, :128, :].transpose(0, 2, 1), np.float32)   # (16,512,128)
            vv = np.ascontiguousarray(kv[:, 128:, :], np.float32)                       # (16,128,512)
            payload = kt.tobytes() if kind == "kvb_k" else vv.tobytes()
            w.write(payload); w.align_pad()
        elif kind == "conv":
            x = idx.load(src)                          # [C,1,4]
            y = np.ascontiguousarray(x.reshape(1, x.shape[0], 1, x.shape[2]), np.float32)
            w.write(y.tobytes()); w.align_pad()
        elif kind == "sa":
            x = idx.load(src)
            v = np.exp(np.asarray(x, dtype=np.float32))
            v2 = np.array([math.exp(float(t)) for t in x.ravel()], np.float32)
            dv = float(np.abs(v - v2).max(initial=0.0))
            if dv > 1e-5:
                audit["errors"].append(f"{dst} exp dev {dv}")
            y = np.ascontiguousarray(v.reshape(-1, 1), np.float32)
            w.write(y.tobytes()); w.align_pad()         # ne {1,16} via reversed((16,1))
        else:  # f32raw
            x = idx.load(src)
            w.write(np.ascontiguousarray(x, np.float32).tobytes()); w.align_pad()
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(table)} {dst} {time.time()-t0:.0f}s", flush=True)
    w.close(); idx.close()
    ng = len(audit["gates"]); bad = [g for g in audit["gates"] if g[1] != "OK"]
    audit.update(n_tensors=len(table), pad_tokens=n_pad, elapsed_s=round(time.time() - t0, 1))
    json.dump(dict(audit=audit, size=os.path.getsize(gp),
                   verdict="R3-8b-artifact " + ("GREEN" if not bad and not audit["errors"] else "RED")),
              open(os.path.join(a.out, f"{stem}-audit.json"), "w"), ensure_ascii=False, indent=1)
    print(f"GGUF {gp} {os.path.getsize(gp)/1e9:.2f}GB tensors={len(table)} gate-mismatch={len(bad)} errors={audit['errors'][:5]}")
    print("sa dims check:", spec("sa", "model.layers.0.attention.A_log")[1])
    return 0 if not bad and not audit["errors"] else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
