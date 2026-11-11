
// src/core/layer.cpp - the GDN layer, composed.  See the header for the operation order and its traps.
#include "strata/core/layer.hpp"
#include "strata/core/pin_budget.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/fuse_decode.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_flash_attn.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#ifndef _WIN32
#include <sys/mman.h>
#endif
#include <memory>
#include <vector>
namespace strata::core {
namespace { bool g_shared_early = true; bool g_fused_gr = false; bool g_fast_attn = true; bool g_publish_kernel = true; bool g_fused_gdn = true; bool g_fast_select = true; }
namespace {constexpr int Q8K_BYTES_PER_BLOCK = 292;
constexpr int Q8K_ELEMS_PER_BLOCK = 256;
constexpr float RMS_EPS = 1e-6f;
const std::string EMBEDDING_NAME = "token_embd.weight";
// qwen4exp.attention.layer_norm_rms_epsilon, used for BOTH norms here
/// A Q8_K buffer needs `n` a multiple of 256 and `n/256` blocks of 292 bytes.
uint64_t q8k_bytes(int64_t n) { return (uint64_t) (n / Q8K_ELEMS_PER_BLOCK) * Q8K_BYTES_PER_BLOCK; }
/// `s_gemv_q8k` takes the canonical-form attributes; a `WeightRef` carries them, and a tensor that is NOT
/// quantized has none.  Returns false and names the tensor rather than building a form out of zeroes - which
/// would decode every code as `0 + bias` and produce a perfectly finite wrong answer.
bool sform_of(const WeightRef& r, strata::kernels::SForm& f, const std::string& name, std::string& err) {    if (!r.quantized()) {        err = name + " is not a quantized tensor, so it has no S-form";        return false;    }    f.code_bits = r.code_bits;    f.code_bias = r.code_bias;    f.group_elems = r.group_elems;    f.codebook = r.codebook_iq4nl ? strata::kernels::Codebook::Iq4Nl : strata::kernels::Codebook::Affine;    f.has_offset = r.has_offset;    f.act_kind = r.act_kind;
// carried, not derived - see the note on `SForm::act_kind`
return true;}
/// The three canonical planes of a quantized tensor, located INSIDE the loaded region.
//
//
// **THIS FUNCTION USED TO RE-DERIVE THE LAYOUT, AND THAT IS HOW A WIDTH BUG SURVIVED A ROUND.**  It computed
/// `n_groups = n_in / group_elems` and then `scales_bytes = n_out * n_groups * sizeof(float)` - 4 bytes per
/// scale for every tensor.  90 of the 303 quantized tensors hold fp16 scales in the pack (all 58 Q2_0 expert
/// tensors, plus Q4_0, Q5_0, Q8_0 and IQ4_NL), so for those the computed plane was TWICE the real one and the
/// three planes did not add up to the tensor.  The first diagnosis was "the manifest's `group_elems` must be
/// wrong, the real group is 64"; it is not, and an audit of all 303 tensors against `scales_fp16` found zero
/// inconsistencies.  **The unexamined input was the scale WIDTH, not the group COUNT.**
//
//
// The loader now widens fp16 scales to f32 on the way into the arena and records the resulting plane sizes in
/// the `WeightRef`, so this function has nothing left to derive - it reads what was loaded.  The redundancy is
/// deliberate: the sizes come from ONE place (the index, checked against the manifest by `pack_index.py` and
/// against the span by the loader), and this function only checks that they describe the tensor it was given.
//
//
// **THE PLANE KEY IS `offsets`, NOT `mins`, AND THAT COST 136 MiB.**  `tools/pack_index.py` and
/// `tools/pack_budget.py` both looked for `mins` and both therefore omitted the offset plane of 54 Q4_K/Q5_K
/// tensors - so the arena was sized without it AND the bytes were never copied.  Two tools agreeing is only
/// evidence when they do not share the assumption that is wrong.  This function refuses rather than guessing,
/// which is what caught it.
struct Planes {    const uint8_t* codes = nullptr;    const float* scales = nullptr;    const float* offset = nullptr;
///< null when the form has none
};
bool plane_ptrs(const WeightRef& r, const std::string& name, Planes& out, std::string& err) {
// S2, S4 AND S8 ALL SPLIT THE SAME WAY.  The plane LOCATION does not depend on the code width - the three
// sizes come from the index and are checked against the tensor below - so the guard is here to catch a
// tensor that is not quantized at all, not to pick a decoder.  WHICH KERNEL reads the planes is the
// caller's choice and the two differ: an S2 tensor's activation contract is Q8_0 (`s2_gemv_q8`) while a
// K-quant's is Q8_K (`s_gemv_q8k`).  This used to accept only 4 and 8, which refused `attn_q` - a Q2_0
// tensor and the reason the QSA layer could not be composed at all.
if (r.code_bits != 2 && r.code_bits != 4 && r.code_bits != 8) {        err = name + ": code_bits " + std::to_string(r.code_bits) + " is not an S2/S4/S8 form";        return false;    }    const uint64_t end = r.codes_bytes + r.scales_bytes + r.offset_bytes;    if (r.codes_bytes == 0 || r.scales_bytes == 0 || end != r.bytes) {        char buf[320];        std::snprintf(buf, sizeof buf,                      "%s: the planes add up to %llu B but the tensor is %llu B (codes %llu, scales %llu, "                      "offsets %llu) - what the loader recorded is not the layout that was loaded",                      name.c_str(), (unsigned long long) end, (unsigned long long) r.bytes,                      (unsigned long long) r.codes_bytes, (unsigned long long) r.scales_bytes,                      (unsigned long long) r.offset_bytes);        err = buf;        return false;    }    if (r.has_offset != (r.offset_bytes != 0)) {        err = name + ": has_offset is " + std::to_string(r.has_offset ? 1 : 0) + " but the offset plane is " +              std::to_string(r.offset_bytes) + " B";        return false;    }    const uint8_t* base = (const uint8_t*) r.data;    out.codes = base;    out.scales = (const float*) (base + r.codes_bytes);    out.offset = r.offset_bytes ? (const float*) (base + r.codes_bytes + r.scales_bytes) : nullptr;    return true;}
constexpr int TPR = 32;
bool native_bf16_projections = false;
bool native_flash_attn_short = false;

void project_bf16(const float* x, const uint16_t* x_bf16, const uint16_t* weights, float* out,
                  int64_t n_in, int64_t n_out, bool split, void* stream) {
    using namespace strata::kernels;
    if (native_bf16_projections) bf16_gemv_fp32_mmvf(x, weights, out, n_in, n_out, stream);
    else if (split) bf16_gemv_split(x_bf16, weights, out, n_in, n_out, TPR, stream);
    else bf16_gemv(x_bf16, weights, out, n_in, n_out, stream);
}
#if defined(STRATA_EXPERIMENTAL_SM60)
// TWO BF16 projections of the SAME x in ONE kernel (`ssm_alpha`/`ssm_beta`, `indexer.q`/`indexer.k`).
// `bf16_gemv_pair` reproduces the warp-per-row accumulation, which is what `bf16_gemv_split` at TPR 32
// always runs and what plain `bf16_gemv` runs at n_out >= 64 - below that the plain entry uses the naive
// one-thread-per-row sum and the pair would round differently, so those shapes keep two calls.
void project_bf16_pair(const float* x, const uint16_t* x_bf16, const uint16_t* w1, float* o1, int64_t n_out1,
                       const uint16_t* w2, float* o2, int64_t n_out2, int64_t n_in, bool split, void* stream) {
    using namespace strata::kernels;
    if (native_bf16_projections) {
        bf16_gemv_fp32_mmvf(x, w1, o1, n_in, n_out1, stream);
        bf16_gemv_fp32_mmvf(x, w2, o2, n_in, n_out2, stream);
    } else if (split || (n_out1 >= 64 && n_out2 >= 64)) {
        bf16_gemv_pair(x_bf16, w1, o1, n_out1, w2, o2, n_out2, n_in, stream);
    } else {
        bf16_gemv(x_bf16, w1, o1, n_in, n_out1, stream);
        bf16_gemv(x_bf16, w2, o2, n_in, n_out2, stream);
    }
}
#endif
///< threads per row for the row-split GEMVs.
/// **MEASURED, NOT ASSUMED, AND 64 IS NOT BETTER.**  `dense_pass.exe` drives the same split kernels at
/// threads_per_row 64 and reports 248.5 GB/s, so matching it looked like free performance.  It is not:
/// TPR=64 gives 11.75 tok/s against 32's 11.99 with experts, and 20.19 against 20.26 without - identical
/// inside noise and marginally worse on both.  That is also what the stage table predicts, because the GEMVs
/// are only ~16% of a token, so a few percent there cannot move the total.  Kept at 32; the sweep that would
/// actually settle it is per-tensor, not global, and belongs with the fusion work rather than before it.
///< threads per row for the row-split GEMVs
/// Writes the sequence number into MAPPED PINNED memory.  A KERNEL, not cudaEventRecord - round 199 found
/// that an event record inside a capture is silently dropped, while this is captured normally and the host can
/// read its result MID-GRAPH.  One thread: it is a store.
// (the ring kernel lives in src/kernels/cuda/elementwise.cu; this file is HOST code)
/// WHICH ACTIVATION A QUANTIZED WEIGHT WANTS, AND IT IS A PROPERTY OF THE **TENSOR**, NOT OF ITS ROLE.
//
//
// Rounds 205-215 ran `gdn_layer` and `qsa_layer` on the assumption that a name implies a family: `attn_qkv` is
/// a K-quant, `attn_q` is Q2_0, and so on.  **The pack does not work that way.**  The same name is quantized
/// per LAYER, and the spread is wide:
//
//
//     attn_qkv     IQ4_XS x13, Q3_K x18, Q2_0 x1,  Q4_K x4      (36 GDN layers)
///     attn_gate    Q3_K x25,  IQ4_XS x4, Q4_K x3,  Q2_0 x4
///     attn_q       Q3_K x6,   IQ4_XS x4, Q2_0 x2                (12 QSA layers)
//
//
// `docs/activation-contract.md` fixes the rule per TYPE, and it is short:
//
//
//     code_bits == 2           ->  Q8_0,  via `quantize_q8_0` + `s2_gemv_q8`
///     anything else quantized  ->  Q8_K,  via `quantize_q8_K` + `s_gemv_q8k_split`
//
//
// Getting it wrong is worth 0.6-1.4% on the GEMV - a plausible vector, not a broken one - which is exactly
/// what Gate C1 exists to find and what nothing before C1 would have.  `s_gemv_q8k`'s own guard is what caught
/// it here (`code_bits 2 has no Q8_K contract`).
//
//
// `x80` and `xq8k` are the two quantized images of the SAME activation; a caller produces both once and this
/// picks.  Producing only the one it thinks it needs is how the assumption gets baked in again.
bool gemv_quantized(const WeightRef& w, const Planes& p, const strata::kernels::SForm& f, const uint8_t* x80, const uint8_t* xq8k,                    float* y, int64_t n_in, int64_t n_out, const std::string& name, void* stream,                    std::string& err, const float* x_f32 = nullptr, bool x_q8_1_ready = false) {
    using namespace strata::kernels;
    if (w.native_data) {
        if (!x_f32 || !w.native_q8_1 || !stream || n_in != w.ne0 || n_out != w.ne1) {
            err = name + ": native projection requires matching FP32 input and session scratch";
            return false;
        }
        try {
            // Plan v0.3 P3: a caller whose previous native projection quantized the SAME x into the shared
            // scratch, with no native projection in between, passes x_q8_1_ready and the quantize is skipped.
            if (!x_q8_1_ready) native_quantize_q8_1(x_f32, w.native_q8_1, (int) n_in, 1, stream);
            native_mmvq(w.native_type, w.native_data, w.native_q8_1, y,
                        (int) n_in, (int) n_out, 1, stream);
        } catch (const std::exception& error) {
            err = name + ": " + error.what();
            return false;
        }
        return true;
    }
    if ((w.code_bits == 2 || !w.wants_q8k()) ? !x80 : !xq8k) {
        err = name + ": missing canonical quantized activation";
        return false;
    }
    if (w.code_bits == 2) {
// S2 carries no offset plane and no SForm: its attributes are fixed (2 bits, group 64, bias -1), which
// is why `s2_gemv_q8` takes neither.
if (w.has_offset || p.offset != nullptr) {            err = name + ": an S2 form must have no offset plane";            return false;        }        if (w.group_elems != 64 || w.code_bias != -1) {            err = name + ": an S2 form must be group 64 with bias -1, this one is group " +                  std::to_string(w.group_elems) + " with bias " + std::to_string(w.code_bias);            return false;        }        s2_gemv_q8(x80, p.codes, p.scales, y, n_in, n_out, TPR, stream);        return true;    }
// **A LEGACY 4/8-BIT FORM WANTS Q8_0, NOT Q8_K, AND `code_bits` CANNOT TELL YOU WHICH.**  This used to be
// `code_bits == 2 ? Q8_0 : Q8_K`, which is CORRECT ONLY BY LUCK: every legacy tensor in this pack apart
// from `ffn_down_shexp` is Q2_0, whose attributes happen to be S2's.  Q5_0 and Q5_K are both 8-bit with
// bias -16 and differ only in `has_offset`; IQ4_NL and IQ4_XS differ in NOTHING the S-form carries.  The
// kind therefore comes from the manifest's `source_type`, through the index (LEDGER L54/L55).
    if (((uintptr_t) p.codes < ((uintptr_t) 1 << 30)) || ((uintptr_t) p.scales < ((uintptr_t) 1 << 30))) {
        std::fprintf(stderr, "gemv_quantized SMALL PTR: %s codes=%p scales=%p off=%p n_in=%lld n_out=%lld cb=%d g=%d\n",
                     name.c_str(), (const void*) p.codes, (const void*) p.scales, (const void*) p.offset,
                     (long long) n_in, (long long) n_out, (int) w.code_bits, (int) w.group_elems);
        std::fflush(stderr);
    }
if (!w.wants_q8k()) {        s_gemv_q8_0_split(x80, p.codes, p.scales, p.offset, y, n_in, n_out, f, stream);        return true;    }    s_gemv_q8k_split(xq8k, p.codes, p.scales, p.offset, y, n_in, n_out, f, stream);    return true;}}
// namespace
void layer_set_native_bf16(bool enabled) { native_bf16_projections = enabled; }
void layer_set_native_flash_attn_short(bool enabled) { native_flash_attn_short = enabled; }
namespace { bool g_kv_int8 = false, g_kv_q4 = false, g_kv_hybrid = false, g_kv_int8_rot = false; }
void qsa_set_kv_int8(bool enabled) { g_kv_int8 = enabled; }
void qsa_set_kv_int8_rotate(bool enabled) { g_kv_int8_rot = enabled; }
bool qsa_kv_int8() { return g_kv_int8; }
void qsa_set_kv_q4(bool enabled) { g_kv_q4 = enabled; }
bool qsa_kv_q4() { return g_kv_q4; }
void qsa_set_kv_hybrid(bool enabled) { g_kv_hybrid = enabled; }
bool qsa_kv_hybrid() { return g_kv_hybrid; }
uint64_t gdn_buffers_bytes(const ModelGeometry& g) {    const int64_t C = g.ssm_conv_channels;    const int64_t V = g.ssm_value_dim;    const uint64_t parts[] = {        q8k_bytes(g.n_embd),
// x_q8k
(uint64_t) (g.n_embd / 32) * 34,
// x_q8_0
(uint64_t) g.n_embd * 2,
// x_bf16
(uint64_t) C * 4,
// qkv
(uint64_t) C * 4,
// conv_out
(uint64_t) C * 4,
// h
(uint64_t) g.ssm_v_heads * 4,
// alpha
(uint64_t) g.ssm_v_heads * 4,
// beta
(uint64_t) g.ssm_v_heads * 4,
// gate
(uint64_t) g.ssm_v_heads * g.ssm_state_size * 4,
// o
(uint64_t) V * 4,
// z
(uint64_t) V * 4,
// y
q8k_bytes(V),
// y_q8k
(uint64_t) (V / 32) * 34,
// y_q8_0
(uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size * 4,
// state
(uint64_t) C * (g.ssm_d_conv - 1) * 4,
// conv_state
};    uint64_t total = 0;    for (uint64_t v : parts) total += (v + 15) & ~(uint64_t) 15;    return total;}
uint64_t gdn_buffers_init(const ModelGeometry& g, void* base, GdnBuffers& b) {    const int64_t C = g.ssm_conv_channels;    const int64_t V = g.ssm_value_dim;    const uint64_t parts[] = {        q8k_bytes(g.n_embd), (uint64_t) (g.n_embd / 32) * 34, (uint64_t) g.n_embd * 2, (uint64_t) C * 4,        (uint64_t) C * 4, (uint64_t) C * 4,        (uint64_t) g.ssm_v_heads * 4, (uint64_t) g.ssm_v_heads * 4, (uint64_t) g.ssm_v_heads * 4,        (uint64_t) g.ssm_v_heads * g.ssm_state_size * 4, (uint64_t) V * 4, (uint64_t) V * 4, q8k_bytes(V),        (uint64_t) (V / 32) * 34,        (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size * 4, (uint64_t) C * (g.ssm_d_conv - 1) * 4,    };    uint8_t* p = (uint8_t*) base;    void* ptr[16];    uint64_t total = 0;    for (int i = 0; i < 16; ++i) {        ptr[i] = p;        const uint64_t al = (parts[i] + 15) & ~(uint64_t) 15;        p += al;        total += al;    }    b.x_q8k = (uint8_t*) ptr[0];    b.x_q8_0 = (uint8_t*) ptr[1];    b.x_bf16 = (uint16_t*) ptr[2];    b.qkv = (float*) ptr[3];    b.conv_out = (float*) ptr[4];    b.h = (float*) ptr[5];    b.alpha = (float*) ptr[6];    b.beta = (float*) ptr[7];    b.gate = (float*) ptr[8];    b.o = (float*) ptr[9];    b.z = (float*) ptr[10];    b.y = (float*) ptr[11];    b.y_q8k = (uint8_t*) ptr[12];    b.y_q8_0 = (uint8_t*) ptr[13];    b.state = (float*) ptr[14];    b.conv_state = (float*) ptr[15];    return total;}
void gdn_buffers_zero_state(const GdnBuffers& b, const ModelGeometry& g, void* stream) {    const uint64_t st = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size * sizeof(float);    const uint64_t cs = (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * sizeof(float);    cudaMemsetAsync(b.state, 0, st, (cudaStream_t) stream);    cudaMemsetAsync(b.conv_state, 0, cs, (cudaStream_t) stream);}
// Forward-declared because `gdn_layer` and `qsa_layer` are both defined above the timer's own definition, and
// the sub-stage marks live inside them.
static void st_begin(int64_t layer, int slot, void* stream);
static void st_end(int64_t layer, int slot, void* stream);
bool gdn_layer(const WeightTable& tables, const ModelGeometry& g, int64_t layer, const GdnBuffers& b,               const float* mixed, float* out, void* stream, std::string& err) {    const LayerView v(tables, layer);    const int64_t C = g.ssm_conv_channels;
// ---- resolve.  Every one of these was shape-checked by `check_layer` at load time, so a miss here is a
//      NAME problem and not a geometry one, and the message says which name.
const WeightRef* w_qkv = v.get("attn_qkv.weight");    const WeightRef* w_gate = v.get("attn_gate.weight");    const WeightRef* w_out = v.get("ssm_out.weight");    const WeightRef* w_alpha = v.get("ssm_alpha.weight");    const WeightRef* w_beta = v.get("ssm_beta.weight");    const WeightRef* w_conv = v.get("ssm_conv1d.weight");    const WeightRef* w_norm = v.get("ssm_norm.weight");    const WeightRef* w_dt = v.get("ssm_dt.bias");    const WeightRef* w_a = v.get("ssm_a");    const char* missing = !w_qkv ? "attn_qkv.weight" : !w_gate ? "attn_gate.weight"                          : !w_out ? "ssm_out.weight" : !w_alpha ? "ssm_alpha.weight"                          : !w_beta ? "ssm_beta.weight" : !w_conv ? "ssm_conv1d.weight"                          : !w_norm ? "ssm_norm.weight" : !w_dt ? "ssm_dt.bias"                          : !w_a ? "ssm_a" : nullptr;    if (missing) { err = v.name(missing) + " is missing"; return false; }
// The conv kernel and the two norms are F32 source types, and the loader copied them verbatim - so their
// device bytes ARE f32 and can be handed straight to the kernels that want `const float*`.
const float* conv_kernel = (const float*) w_conv->data;    const float* ssm_norm = (const float*) w_norm->data;    const float* dt = (const float*) w_dt->data;    const float* ssm_a = (const float*) w_a->data;    strata::kernels::SForm f_qkv, f_gate, f_out;    if (!sform_of(*w_qkv, f_qkv, v.name("attn_qkv.weight"), err)) return false;    if (!sform_of(*w_gate, f_gate, v.name("attn_gate.weight"), err)) return false;    if (!sform_of(*w_out, f_out, v.name("ssm_out.weight"), err)) return false;    Planes p_qkv, p_gate, p_out;    if (!plane_ptrs(*w_qkv, v.name("attn_qkv.weight"), p_qkv, err)) return false;    if (!plane_ptrs(*w_gate, v.name("attn_gate.weight"), p_gate, err)) return false;    if (!plane_ptrs(*w_out, v.name("ssm_out.weight"), p_out, err)) return false;    using namespace strata::kernels;    cudaStream_t st = (cudaStream_t) stream;
// ---- 1. the activation, in EVERY format a weight on this layer might ask for.  BOTH quantized images are
//         produced, because which one is wanted is a property of the TENSOR and the pack mixes them by
//         layer - producing only the "right" one is how the per-role assumption gets baked back in.
st_begin(layer, 8, stream);
    // Plan v0.3 P3: the canonical activation images only when a canonical projection reads them.
    if (!w_qkv->native_data || !w_gate->native_data) {
#if defined(STRATA_EXPERIMENTAL_SM60)
    if (strata::kernels::fuse_decode()) {
        quantize_act_images(mixed, g.n_embd, b.x_q8_0, b.x_q8k, nullptr, stream);   // ONE pass, was two kernels
    } else
#endif
    {
        quantize_q8_K(mixed, b.x_q8k, g.n_embd, stream);
        quantize_q8_0(mixed, b.x_q8_0, g.n_embd, stream);
    }
    }
    st_end(layer, 8, stream);
// ---- 2. qkv = wqkv @ x, with the activation THIS layer's `attn_qkv` asks for
st_begin(layer, 9, stream);    if (!gemv_quantized(*w_qkv, p_qkv, f_qkv, b.x_q8_0, b.x_q8k, b.qkv, g.n_embd, C,                        v.name("attn_qkv.weight"), stream, err, mixed)) return false;    st_end(layer, 9, stream);
// ---- 3. conv, then SiLU on the WHOLE conv output - before the q|k|v split, which is what
//         `ref/gdn.py` L262 says and what makes the split a reshape rather than a copy.
st_begin(layer, 10, stream);
// Plan v0.3 P3: under the native contract, conv + SiLU + both L2 norms are one kernel, and alpha/beta with their
// epilogues another (`fused_gdn.hpp`).
const bool fused_pre = g_fused_gdn && native_gdn_enabled() && native_bf16_projections && g.ssm_d_conv == 4 &&
                       g.ssm_state_size == 128;
if (fused_pre) {
    fused_gdn_conv_l2(b.conv_state, b.qkv, conv_kernel, b.h, (int) C, (int) (2 * g.ssm_k_heads), RMS_EPS, stream);
} else
try {
    if (native_gdn_enabled()) native_gdn_conv_silu(b.conv_state, b.qkv, conv_kernel, b.conv_out, b.h, C, g.ssm_d_conv, stream);
    else {
        gdn_conv_step(b.conv_state, b.qkv, conv_kernel, b.conv_out, C, g.ssm_d_conv, stream);
        cudaMemcpyAsync(b.h, b.conv_out, (size_t) C * 4, cudaMemcpyDeviceToDevice, st);
        silu_inplace(b.h, C, stream);
    }
} catch (const std::exception& error) { err = v.name("gdn_conv_silu") + ": " + error.what(); return false; }
st_end(layer, 10, stream);
// ---- 4. q,k = l2_norm; v is NOT normalised.  The three views are contiguous runs of the same buffer:
//         q at [0, 2048), k at [2048, 4096), v at [4096, 10240), each 128-wide heads.    const
st_begin(layer, 11, stream);    int64_t qk = g.ssm_state_size * g.ssm_k_heads;
if (!fused_pre) try {
    if (native_gdn_enabled()) {
        native_gdn_l2_norm(b.h, g.ssm_k_heads, g.ssm_state_size, RMS_EPS, stream);
        native_gdn_l2_norm(b.h + qk, g.ssm_k_heads, g.ssm_state_size, RMS_EPS, stream);
    } else {
        gdn_l2_norm(b.h, g.ssm_k_heads, g.ssm_state_size, RMS_EPS, stream);
        gdn_l2_norm(b.h + qk, g.ssm_k_heads, g.ssm_state_size, RMS_EPS, stream);
    }
} catch (const std::exception& error) { err = v.name("gdn_l2_norm") + ": " + error.what(); return false; }
// The recurrence needs another 1/sqrt(S) beyond Q/K normalization. The legacy
// recurrence consumes prescaled Q; pinned CUDA applies it after the readout dot.
// Keep the scale at that backend-specific boundary to preserve rounding order.
if (!native_gdn_enabled()) scale_inplace(b.h, qk, 1.0f / std::sqrt((float) g.ssm_state_size), stream);
st_end(layer, 11, stream);
// ---- 5. beta and the gate. The weights are BF16. Native single-token CUDA MMVF
//         consumes F32 activations; the legacy path rounds them to BF16 first.
//
//         THE ROW-SPLIT FORM, and here the case is stronger than the router's: `ssm_v_heads` is **48**, so
//         the plain one-thread-per-row form puts 48 threads on a 48-SM GPU and each walks 2,560 elements
//         alone.  Measured 0.1072 ms each, against 0.0277 for the split - and there are two of them, in all
//         36 GDN layers.
st_begin(layer, 12, stream);
    if (fused_pre) {
        fused_gdn_ab(mixed, (const uint16_t*) w_alpha->data, (const uint16_t*) w_beta->data, dt, ssm_a, b.gate, b.beta,
                     (int) g.n_embd, (int) g.ssm_v_heads, stream);
    } else {
    if (!native_bf16_projections) f32_to_bf16_bulk(mixed, b.x_bf16, g.n_embd, stream);
#if defined(STRATA_EXPERIMENTAL_SM60)
    if (strata::kernels::fuse_decode()) {
    project_bf16_pair(mixed, b.x_bf16, (const uint16_t*) w_alpha->data, b.alpha, g.ssm_v_heads,
                      (const uint16_t*) w_beta->data, b.beta, g.ssm_v_heads, g.n_embd, true, stream);
    } else
#endif
    {
    project_bf16(mixed, b.x_bf16, (const uint16_t*) w_alpha->data, b.alpha, g.n_embd, g.ssm_v_heads, true, stream);
    project_bf16(mixed, b.x_bf16, (const uint16_t*) w_beta->data, b.beta, g.n_embd, g.ssm_v_heads, true, stream);
    }
    }
    if (!fused_pre) try {
        if (native_gdn_enabled()) {
            native_gdn_beta_gate(b.beta, g.ssm_v_heads, stream);
            native_gdn_gate(b.alpha, dt, ssm_a, b.gate, g.ssm_v_heads, stream);
        } else {
            gdn_beta_gate(b.beta, g.ssm_v_heads, stream);
            gdn_gate(b.alpha, dt, ssm_a, b.gate, 1, g.ssm_v_heads, stream);
        }
    } catch (const std::exception& error) { err = v.name("gdn_gate") + ": " + error.what(); return false; }
    st_end(layer, 12, stream); st_begin(layer, 13, stream);
    GdnShapes gs{g.ssm_state_size, g.ssm_k_heads, g.ssm_v_heads};
    // Plan v0.3 P3: with the native contract the step and the output norm are one kernel (after the z GEMV).
    const bool fused_gdn = g_fused_gdn && native_gdn_enabled() && g.ssm_state_size == 128;
    if (!fused_gdn) try {
        if (native_gdn_enabled()) native_gdn_step(b.state, b.h, b.h + qk, b.h + 2 * qk, b.gate, b.beta, b.o, gs, stream);
        else gdn_step(b.state, b.h, b.h + qk, b.h + 2 * qk, b.gate, b.beta, b.o, gs, stream);
    } catch (const std::exception& error) {
        err = v.name("gdn_step") + ": " + error.what();
        return false;
    }
    st_end(layer, 13, stream); st_begin(layer, 14, stream);
// ---- 6. the recurrence.  q and k are the (16, 128) views; v is the (48, 128) view.  The stage marks for
// it live at the end of the alpha/beta/gate line above, because the gate and the step are one sequence.
// ---- 7. z, then y = rms_norm(o) * ssm_norm * sigmoid(z).  SIGMOID, not SiLU.
// qkv's native quantize of `mixed` is still in the shared q8_1 scratch: nothing native ran in between.
if (!gemv_quantized(*w_gate, p_gate, f_gate, b.x_q8_0, b.x_q8k, b.z, g.n_embd, g.ssm_value_dim, v.name("attn_gate.weight"), stream, err, mixed,
                    w_qkv->native_data != nullptr && w_gate->native_data != nullptr)) return false;
try {
    if (fused_gdn) fused_gdn_step_norm(b.state, b.h, b.h + qk, b.h + 2 * qk, b.gate, b.beta, b.z, ssm_norm, RMS_EPS, b.y,
                                       (int) g.ssm_k_heads, (int) g.ssm_v_heads, stream);
    else if (native_gdn_enabled()) native_gdn_out_norm(b.o, b.z, ssm_norm, b.y, g.ssm_v_heads, g.ssm_state_size, RMS_EPS, stream);
    else gdn_out_norm(b.o, b.z, ssm_norm, b.y, g.ssm_v_heads, g.ssm_state_size, RMS_EPS, stream);
} catch (const std::exception& error) { err = v.name("gdn_out_norm") + ": " + error.what(); return false; }
st_end(layer, 14, stream);
// ---- 8. out = ssm_out @ y, whose activation is whatever THIS layer's `ssm_out` asks for
st_begin(layer, 15, stream);
    if (!w_out->native_data) {
#if defined(STRATA_EXPERIMENTAL_SM60)
    if (strata::kernels::fuse_decode()) {
        quantize_act_images(b.y, g.ssm_value_dim, b.y_q8_0, b.y_q8k, nullptr, stream);   // ONE pass, was two
    } else
#endif
    {
        quantize_q8_K(b.y, b.y_q8k, g.ssm_value_dim, stream);
        quantize_q8_0(b.y, b.y_q8_0, g.ssm_value_dim, stream);
    }
    }    if (!gemv_quantized(*w_out, p_out, f_out, b.y_q8_0, b.y_q8k, out, g.ssm_value_dim, g.n_embd,                        v.name("ssm_out.weight"), stream, err, b.y)) return false;    st_end(layer, 15, stream);    return true;}
// ================================ the MoE block ================================
uint64_t moe_buffers_bytes(const ModelGeometry& g, int64_t k) {    const uint64_t parts[] = {        (uint64_t) g.n_embd * 2,
// x_bf16
(uint64_t) g.n_embd * 2,
// x_f16
(uint64_t) g.n_expert * 4,
// logits
(uint64_t) k * 4,
// ids (int32)
(uint64_t) k * 4,
// weights
(uint64_t) g.n_embd * 4,
// shared
strata::kernels::shared_expert_scratch_bytes(g.n_ff),
// the shared expert's own scratch
(uint64_t) (g.n_embd / 32) * 34,
// x_q8_0
q8k_bytes(g.n_embd),
// x_q8k
};    uint64_t total = 0;    for (uint64_t v : parts) total += (v + 15) & ~(uint64_t) 15;    return total;}
uint64_t moe_buffers_init(const ModelGeometry& g, int64_t k, void* base, MoEBuffers& b) {    const uint64_t parts[] = {        (uint64_t) g.n_embd * 2, (uint64_t) g.n_embd * 2, (uint64_t) g.n_expert * 4,        (uint64_t) k * 4, (uint64_t) k * 4, (uint64_t) g.n_embd * 4,        strata::kernels::shared_expert_scratch_bytes(g.n_ff),        (uint64_t) (g.n_embd / 32) * 34, q8k_bytes(g.n_embd),    };    uint8_t* p = (uint8_t*) base;    void* ptr[9];    uint64_t total = 0;    for (int i = 0; i < 9; ++i) {        ptr[i] = p;        const uint64_t al = (parts[i] + 15) & ~(uint64_t) 15;        p += al;        total += al;    }    b.x_bf16 = (uint16_t*) ptr[0];    b.x_f16 = (uint16_t*) ptr[1];    b.logits = (float*) ptr[2];    b.ids = (int*) ptr[3];    b.weights = (float*) ptr[4];    b.shared = (float*) ptr[5];    b.sh_scratch = (float*) ptr[6];    b.x_q8_0 = (uint8_t*) ptr[7];    b.x_q8k = (uint8_t*) ptr[8];    return total;}
bool moe_route(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t k, const MoEBuffers& b,               const float* x, void* stream, std::string& err, const Doorbell* db) {    using namespace strata::kernels;    const LayerView v(tables, layer);    const WeightRef* w_router = v.get("ffn_gate_inp.weight");    if (w_router == nullptr) { err = v.name("ffn_gate_inp.weight") + " is missing"; return false; }    if (k < 1 || k > 64) { err = "moe_route: k must be 1..64"; return false; }
// ---- the router's activation.  The router's weight is BF16 and that is the one `ref/moe.py` singles out.
if (!native_bf16_projections) f32_to_bf16_bulk(x, b.x_bf16, g.n_embd, stream);
// ---- logits = ffn_gate_inp @ bf16(x).  The weight is BF16, so this is `bf16_gemv` and not `s_gemv` -
//      and using the fp16 activation here instead is the 8.100e-03 error that flips a selection.
//
//      **THE ROW-SPLIT VARIANT, BECAUSE THE PLAIN ONE'S PARALLELISM IS THE OUTPUT WIDTH.**  `bf16_gemv` is
//      one thread per output row: for the router that is 512 threads over 48 SMs, each walking 2,560
//      elements sequentially, and it measured **0.2711 ms for a 2.6 MB weight** - about 65x its
//      memory-bound floor and the largest single item left inside the MoE after the top-10 was fixed.
//      `bf16_gemv_split` exists for precisely this case; its own header says so.  Measured here:
//      LEDGER L41 -> L42.
project_bf16(x, b.x_bf16, (const uint16_t*) w_router->data, b.logits, g.n_embd, g.n_expert, true, stream);
// ---- routing: softmax over ALL experts, stable descending argsort with ties by index, gather, renormalise
// the native fused router is canonical-512x10 only; anything else takes the generic top-k kernel
if (native_router_enabled() && g.n_expert == 512 && k == 10) {
    try { native_router_top10(b.logits, b.ids, b.weights, stream); }
    catch (const std::exception& error) { err = v.name("router") + ": " + error.what(); return false; }
} else router_top10(b.logits, 1, (int) g.n_expert, (int) k, b.ids, b.weights, stream);
// ---- THE DOORBELL, AND IT IS THE WHOLE POINT OF THE PROTOCOL.  The routed experts' INPUT (`x`) and the
//      miss list (`ids`, `weights`) are published to the host HERE, before anything that depends on them,
//      so a CPU pool can start on them while the GPU keeps working.  Both the copy and the ring are
//      captured into `G[l]` like anything else - a kernel writing mapped pinned memory, which round 209
//      measured the host seeing 0.050 ms into a 39.8 ms graph.
if (db != nullptr && g_publish_kernel) {
    strata::kernels::doorbell_publish(x, b.ids, b.weights, g.n_embd, k, db->d_x_f, db->d_ids, db->d_weights, db->d_seq,
                                      stream);
} else if (db != nullptr) {        if (cudaMemcpyAsync(db->d_x_f, x, (size_t) g.n_embd * 4, cudaMemcpyDeviceToDevice,                            (cudaStream_t) stream) != cudaSuccess ||            cudaMemcpyAsync(db->d_ids, b.ids, (size_t) k * 4, cudaMemcpyDeviceToDevice,                            (cudaStream_t) stream) != cudaSuccess ||            cudaMemcpyAsync(db->d_weights, b.weights, (size_t) k * 4, cudaMemcpyDeviceToDevice,                            (cudaStream_t) stream) != cudaSuccess) {            err = "moe_route: the doorbell handoff copy failed";            return false;        }
// THE RING IS LAST, so a host that sees it knows every byte above is in place.  Ordering within a
// stream is what makes that true; it is not a timing assumption.
// THE RING LIVES IN A .cu: this file is compiled by the HOST compiler, where `__global__` and `<<<>>>` do
// not exist.  The sequence value is read from the HOST copy and incremented, which is what makes the
// write idempotent across replays of the same graph - a captured literal would ring the same number
// forever and the host would never see a change.
strata::kernels::doorbell_ring(db->d_seq, stream);    }    return true;}
// The verify window's routing: n tokens' logits in one BF16 projection and their top-k in one launch, each token
// bitwise what `moe_route` gives it (the projection is the same kernel per column, the top-k kernel is per token).
// Anything else (a native router, the canonical BF16 GEMVs, a doorbell) takes the per-token `moe_route`.
bool moe_route_window(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t k, const MoEBuffers& b,
                      const float* x, float* logits, int32_t* ids, float* weights, int n, void* stream, std::string& err) {
    using namespace strata::kernels;
    const bool batched = native_bf16_projections && !(native_router_enabled() && g.n_expert == 512 && k == 10) &&
                         !std::getenv("STRATA_ROUTE_PER_TOKEN");
    if (!batched || n == 1) {
        for (int t = 0; t < n; ++t) {
            MoEBuffers mb = b;
            mb.logits = logits + t * g.n_expert; mb.ids = ids + t * k; mb.weights = weights + t * k;
            if (!moe_route(tables, g, layer, k, mb, x + t * g.n_embd, stream, err, nullptr)) return false;
        }
        return true;
    }
    const LayerView v(tables, layer);
    const WeightRef* w_router = v.get("ffn_gate_inp.weight");
    if (w_router == nullptr) { err = v.name("ffn_gate_inp.weight") + " is missing"; return false; }
    if (std::getenv("STRATA_ROUTE_PROJ_PER_TOKEN"))
        for (int t = 0; t < n; ++t)
            bf16_gemv_fp32_mmvf(x + t * g.n_embd, (const uint16_t*) w_router->data, logits + t * g.n_expert, g.n_embd,
                                g.n_expert, stream);
    else
        bf16_gemv_fp32_mmvf_cols(x, (const uint16_t*) w_router->data, logits, g.n_embd, g.n_expert, n, stream);
    router_top10(logits, n, (int) g.n_expert, (int) k, ids, weights, stream);
    return true;
}
bool moe_shared(const WeightTable& tables, const ModelGeometry& g, int64_t layer, const MoEBuffers& b,                const float* x, void* stream, std::string& err) {    using namespace strata::kernels;    const LayerView v(tables, layer);    const WeightRef* w_ginp = v.get("ffn_gate_inp_shexp.weight");    const WeightRef* w_sgate = v.get("ffn_gate_shexp.weight");    const WeightRef* w_sup = v.get("ffn_up_shexp.weight");    const WeightRef* w_sdown = v.get("ffn_down_shexp.weight");    const char* missing = !w_ginp ? "ffn_gate_inp_shexp.weight" : !w_sgate ? "ffn_gate_shexp.weight"                          : !w_sup ? "ffn_up_shexp.weight" : !w_sdown ? "ffn_down_shexp.weight" : nullptr;    if (missing) { err = v.name(missing) + " is missing"; return false; }
// ---- the shared expert.  Its three weights are quantized, so they need their planes.
SForm f_gate, f_up, f_down;    if (!sform_of(*w_sgate, f_gate, v.name("ffn_gate_shexp.weight"), err)) return false;    if (!sform_of(*w_sup, f_up, v.name("ffn_up_shexp.weight"), err)) return false;    if (!sform_of(*w_sdown, f_down, v.name("ffn_down_shexp.weight"), err)) return false;    Planes p_gate, p_up, p_down;    if (!plane_ptrs(*w_sgate, v.name("ffn_gate_shexp.weight"), p_gate, err)) return false;    if (!plane_ptrs(*w_sup, v.name("ffn_up_shexp.weight"), p_up, err)) return false;    if (!plane_ptrs(*w_sdown, v.name("ffn_down_shexp.weight"), p_down, err)) return false;
// `ffn_gate_inp_shexp` is a 1-D BF16 vector, so THE ARENA HOLDS 2 BYTES PER ELEMENT, not 4 - the loader
// re-rounded it to the engine form like any other BF16 tensor.  Reading it as f32 is 5120 bytes past the
// end of a 5120-byte tensor inside a 4.5 GiB arena, where nothing faults.  Refuse here rather than trust
// the layout check to have run: this function is reachable without it.
if (w_ginp->kind != WeightKind::Bf16InF32) {        err = v.name("ffn_gate_inp_shexp.weight") + " is engine form " +              std::to_string((int) w_ginp->kind) + " (" + std::to_string(w_ginp->bytes) +              " B); the scalar gate reads it as bf16, so it must be form 1 (" +              std::to_string((uint64_t) g.n_embd * 2) + " B)";        return false;    }
// THE SHARED EXPERT'S WEIGHTS ARE NOT ONE FAMILY: `ffn_gate_shexp` is Q2_0 on 21 layers and a K-quant on
// the rest, `ffn_up_shexp` Q2_0 on 13, and `ffn_down_shexp` is LEGACY in every layer (IQ4_NL/Q4_0/Q5_0/
// Q8_0/Q2_0, `n_in` 640 so Q8_K is impossible).  So BOTH quantized images of the activation are produced
// and each projection takes the one its own form asks for.
// ONE pass over `x` produces the bf16 image and both quantized images (was three kernels): which quantized
// one is wanted is a property of the TENSOR and the pack mixes them by layer, so both are produced.
    if (!w_sgate->native_data || !w_sup->native_data) {
#if defined(STRATA_EXPERIMENTAL_SM60)
    if (strata::kernels::fuse_decode()) {
        quantize_act_images(x, g.n_embd, b.x_q8_0, b.x_q8k, b.x_bf16, stream);   // ONE pass, was three kernels
    } else
#endif
    {
        f32_to_bf16_bulk(x, b.x_bf16, g.n_embd, stream);
        quantize_q8_K(x, b.x_q8k, g.n_embd, stream);
        quantize_q8_0(x, b.x_q8_0, g.n_embd, stream);
    }
    } else {
        f32_to_bf16_bulk(x, b.x_bf16, g.n_embd, stream);
    }
    NativeSharedWeights native;
    native.gate_type = w_sgate->native_type; native.gate_data = w_sgate->native_data;
    native.up_type = w_sup->native_type; native.up_data = w_sup->native_data;
    native.down_type = w_sdown->native_type; native.down_data = w_sdown->native_data;
    // The NativeDense owner publishes one shared workspace for this ordered session stream.
    native.q8_1 = w_sgate->native_q8_1 ? w_sgate->native_q8_1 :
                  w_sup->native_q8_1 ? w_sup->native_q8_1 : w_sdown->native_q8_1;
    try {
        shared_expert(b.x_q8_0, b.x_q8k, b.x_bf16, f_gate, p_gate.codes, p_gate.scales, p_gate.offset,
                      f_up, p_up.codes, p_up.scales, p_up.offset, f_down, p_down.codes, p_down.scales,
                      p_down.offset, (const uint16_t*) w_ginp->data, b.sh_scratch, b.shared,
                      g.n_embd, g.n_ff, /*tpr=*/32, stream, x, &native);
    } catch (const std::exception& error) {
        err = v.name("shared_expert") + ": " + error.what();
        return false;
    }
    return true;
}
bool moe_combine_parts(const ModelGeometry& g, int64_t layer, int64_t k, const MoEBuffers& b, const float* parts,
                       float* out, void* stream, std::string& err) {
    using namespace strata::kernels;
    if (k < 1 || k > 64) { err = "moe_finish: k must be 1..64"; return false; }
    // ---- the combination: sum of w[i] * parts[i], plus the shared output ADDED PLAIN.
//
// **`parts` IS COMBINED WITH `b.weights`, WHICH `moe_route` WROTE FOR *THIS* LAYER.**  That is the whole
// reason this is a separate function: a caller that passes the previous layer's expert vectors gets
// `sum_j w_l[j] * expert_{ids_{l-1},j}(x_{l-1})`, which is finite, fluent and not the model.
try {
    if (native_moe_combine_enabled()) native_moe_combine(parts, b.weights, b.shared, out, g.n_embd, k, stream);
    else moe_combine(parts, b.weights, b.shared, out, g.n_embd, k, stream);
} catch (const std::exception& error) {
    err = "blk." + std::to_string(layer) + ".moe_combine" + ": " + error.what();
    return false;
}
return true;}
bool moe_finish(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t k, const MoEBuffers& b,
                const float* x, const float* parts, float* out, void* stream, std::string& err) {
    if (k < 1 || k > 64) { err = "moe_finish: k must be 1..64"; return false; }
    if (!moe_shared(tables, g, layer, b, x, stream, err)) return false;
    return moe_combine_parts(g, layer, k, b, parts, out, stream, err);
}
bool layer_verify_compatible(std::string& why) {
    // Plan v0.3 P6: the verify window reproduces exactly this configuration's per-token arithmetic.
    if (!native_bf16_projections) why = "the native BF16 projections are off";
    else if (!g_fused_gr) why = "the fused hyper-connection read is off";
    else if (!g_fused_gdn || !strata::kernels::native_gdn_enabled()) why = "the fused native GDN kernels are off";
    else if (!g_fast_attn || native_flash_attn_short) why = "the split-K decode attention is off";
    else if (!g_fast_select) why = "the block top-k selection is off";
    else if (!strata::kernels::native_qsa_indexer_enabled()) why = "the native QSA indexer is off";
    else return true;
    return false;
}
void layer_set_publish_kernel(bool enabled) { g_publish_kernel = enabled; }
void layer_set_fused_gdn(bool enabled) { g_fused_gdn = enabled; }
void layer_set_fast_select(bool enabled) { g_fast_select = enabled; }
void layer_set_fast_attn(bool enabled) { g_fast_attn = enabled; }
void layer_set_fused_gr(bool enabled) { g_fused_gr = enabled; }
bool layer_fused_gr() { return g_fused_gr; }
void layer_set_shared_early(bool enabled) { g_shared_early = enabled; }
bool layer_shared_early() { return g_shared_early; }
bool moe_layer(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t k, const MoEBuffers& b,               const float* x, const float* parts, float* out, void* stream, std::string& err,               const Doorbell* db) {
// The two halves in sequence.  `moe_route` publishes `ids`/`weights` for THIS layer and `moe_finish`
// combines `parts` with them, so a caller of this wrapper must have `parts` ready BEFORE the call - which
// is what `session_token` (the uncaptured path) does and what the captured loop CANNOT, which is why the
// loop uses the halves separately.
if (!moe_route(tables, g, layer, k, b, x, stream, err, db)) return false;    return moe_finish(tables, g, layer, k, b, x, parts, out, stream, err);}
// ================================ the QSA mixer ================================
namespace {using strata::kernels::QsaIndexerBuffers;using strata::kernels::QsaShapes;
// `qsa_shapes(g)` (layout.hpp): the geometry the QSA kernels want, every field from the model's geometry
uint64_t align_up16(uint64_t n) { return (n + 15) & ~15ull; }
/// One cursor over an arena, so every region is 16-byte aligned without a list of hand-added offsets.
struct Cursor {    uint8_t* p;    uint64_t used = 0;    template <typename T>    T* take(uint64_t count) {        T* r = (T*) (p + used);        used = align_up16(used + count * sizeof(T));        return r;    }    uint8_t* take_bytes(uint64_t n) {        uint8_t* r = p + used;        used = align_up16(used + n);        return r;    }};
// `TPR` and `gemv_quantized` live in the FIRST anonymous namespace, above, because `gdn_layer` needs the
// dispatch too and the two must not be able to drift.
}
// namespace
namespace {
// KV streaming (docs/kv-streaming-design.md): 0 keeps every cell in VRAM.
int64_t g_kv_resident = 0;
uint64_t g_kv_host_bytes = 0;

// The KV host copy: pinned and device-mapped. cudaHostAlloc is the default, but under WSL2 the driver caps what
// cudaHostAlloc(Mapped) may pin at about 1 GiB in all, while cudaHostRegister of ordinary memory is not capped there:
// the expert arena registers tens of GiB the same way (pinned.cu). So under WSL2, detected as generate.cpp's
// under_wddm() does (/dev/dxg, the device its GPU goes through to the Windows driver), the copy is anonymous memory
// registered Portable|Mapped before any page is touched, which also leaves WSL's small cudaHostAlloc budget to the
// staging buffers. Elsewhere cudaHostAlloc stays first. Either way the other method is the fallback when the first
// is refused. STRATA_KV_HOST_REGISTER=1/0 forces registering first or cudaHostAlloc first. Like the cudaHostAlloc
// copy it lives as long as the process.
uint8_t* kv_host_alloc(uint64_t bytes) {
    auto host_alloc = [](uint64_t n) -> uint8_t* {
        uint8_t* h = nullptr;
        if (strata::core::host_alloc_guarded((void**) &h, n, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess) return h;
        (void) cudaGetLastError();
        return nullptr;
    };
#ifdef _WIN32
    return host_alloc(bytes);
#else
    static const bool register_first = [] {
        const char* e = std::getenv("STRATA_KV_HOST_REGISTER");
        if (e != nullptr && (e[0] == '0' || e[0] == '1')) return e[0] == '1';
        return std::filesystem::exists("/dev/dxg");
    }();
    auto register_anon = [](uint64_t n) -> uint8_t* {
        void* m = mmap(nullptr, (size_t) n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) return nullptr;
        if (cudaHostRegister(m, (size_t) n, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
            (void) cudaGetLastError();
            munmap(m, (size_t) n);
            return nullptr;
        }
        static bool said = false;
        if (!said) {
            std::fprintf(stderr, "strata: KV streaming: the host copy is pinned with cudaHostRegister\n");
            said = true;
        }
        return (uint8_t*) m;
    };
    if (register_first) {
        if (uint8_t* m = register_anon(bytes)) return m;
        return host_alloc(bytes);
    }
    if (uint8_t* h = host_alloc(bytes)) return h;
    return register_anon(bytes);
#endif
}

// ---- the elastic K/V (--kv-grow): one VMM range per state, each pool array at a chunk boundary in it
bool g_kv_elastic = false;
int64_t g_kv_elastic_init = 16384;
struct ElasticPool {
    strata::core::VmmRange range;
    std::vector<uint64_t> off, per_slot;   // each array's offset in the range (whole chunks) and bytes per page slot
    int64_t n_slots = 0, page_size = 0;
};
std::vector<std::unique_ptr<ElasticPool>> g_pools;

int64_t pool_chunks(const ElasticPool& p, size_t a, int64_t slots) {
    const uint64_t G = strata::core::vmm_granularity();
    return (int64_t) (((uint64_t) slots * p.per_slot[a] + G - 1) / G);
}
int64_t pool_slots(const ElasticPool& p, int64_t cells) {
    return std::min<int64_t>(p.n_slots, std::max<int64_t>((cells + p.page_size - 1) / p.page_size, 1));
}
/// Whole page slots every array of the pool has mapped.
int64_t pool_slots_mapped(const ElasticPool& p) {
    const uint64_t G = strata::core::vmm_granularity();
    int64_t slots = p.n_slots;
    for (size_t a = 0; a < p.off.size(); ++a) {
        const int64_t c0 = (int64_t) (p.off[a] / G);
        const int64_t end = a + 1 < p.off.size() ? (int64_t) (p.off[a + 1] / G) : p.range.chunks();
        int64_t c = 0;
        while (c0 + c < end && p.range.mapped(c0 + c)) ++c;
        slots = std::min<int64_t>(slots, (int64_t) ((uint64_t) c * G / p.per_slot[a]));
    }
    return slots;
}
bool pool_grow(ElasticPool& p, int64_t cells, const std::function<strata::core::VmmChunk()>& take) {
    const uint64_t G = strata::core::vmm_granularity();
    const int64_t slots = pool_slots(p, cells);
    for (size_t a = 0; a < p.off.size(); ++a) {
        const int64_t c0 = (int64_t) (p.off[a] / G), c1 = c0 + pool_chunks(p, a, slots);
        std::vector<int64_t> fresh;
        for (int64_t c = c0; c < c1; ++c)
            if (!p.range.mapped(c)) fresh.push_back(c);
        if (fresh.empty()) continue;
        if (!p.range.map_range(c0, c1, take)) {
            std::fprintf(stderr, "strata: the elastic K/V could not map %zu chunks (the driver has no VRAM left)\n",
                         fresh.size());
            return false;
        }
        // what the chunk held before (another pool's cells, an expert) is not K/V: zeroed, as a new session is
        for (const int64_t c : fresh)
            if (cudaMemset(p.range.base() + (uint64_t) c * G, 0, (size_t) G) != cudaSuccess) {
                std::fprintf(stderr, "strata: the elastic K/V could not clear a chunk: %s\n",
                             cudaGetErrorString(cudaGetLastError()));
                return false;
            }
    }
    return true;
}
}  // namespace

void qsa_set_kv_elastic(bool enabled, int64_t init_cells) {
    g_kv_elastic = enabled && strata::core::vmm_available();
    if (init_cells > 0) g_kv_elastic_init = init_cells;
}
bool qsa_kv_elastic() { return g_kv_elastic; }
int64_t qsa_kv_elastic_cells() {
    int64_t cells = std::numeric_limits<int64_t>::max();
    for (const auto& p : g_pools) cells = std::min<int64_t>(cells, pool_slots_mapped(*p) * p->page_size);
    return cells;
}
int64_t qsa_kv_elastic_need(int64_t cells) {
    const uint64_t G = strata::core::vmm_granularity();
    int64_t n = 0;
    for (const auto& p : g_pools) {
        const int64_t slots = pool_slots(*p, cells);
        for (size_t a = 0; a < p->off.size(); ++a) {
            const int64_t c0 = (int64_t) (p->off[a] / G), c1 = c0 + pool_chunks(*p, a, slots);
            for (int64_t c = c0; c < c1; ++c) n += !p->range.mapped(c);
        }
    }
    return n;
}
bool qsa_kv_elastic_grow(int64_t cells, const std::function<strata::core::VmmChunk()>& take) {
    for (auto& p : g_pools)
        if (!pool_grow(*p, cells, take)) return false;
    return cudaDeviceSynchronize() == cudaSuccess;
}
int64_t qsa_kv_elastic_shrink(int64_t cells, const std::function<void(strata::core::VmmChunk)>& give) {
    const uint64_t G = strata::core::vmm_granularity();
    int64_t n = 0;
    for (auto& p : g_pools) {
        const int64_t slots = pool_slots(*p, cells);
        for (size_t a = 0; a < p->off.size(); ++a) {
            const int64_t c0 = (int64_t) (p->off[a] / G), c1 = c0 + pool_chunks(*p, a, slots);
            const int64_t end = a + 1 < p->off.size() ? (int64_t) (p->off[a + 1] / G) : p->range.chunks();
            for (int64_t c = c1; c < end; ++c)
                if (const strata::core::VmmChunk h = p->range.unmap(c)) { give(h); ++n; }
        }
    }
    return n;
}
uint64_t qsa_kv_elastic_mapped_bytes() {
    uint64_t n = 0;
    for (const auto& p : g_pools) n += (uint64_t) p->range.mapped_count() * strata::core::vmm_granularity();
    return n;
}
uint64_t qsa_kv_elastic_full_bytes() {
    uint64_t n = 0;
    for (const auto& p : g_pools) n += (uint64_t) p->range.chunks() * strata::core::vmm_granularity();
    return n;
}
namespace {

/// How one state holds its K/V: `mode` as in QsaState::kv_mode, `slots` VRAM pages of `pages` logical ones.
struct KvPlan {
    int mode = 0;
    int64_t pages = 0, slots = 0, pooled_rows = 0;
};
KvPlan kv_plan(const QsaShapes& s, int64_t max_cells, int64_t ring_cells) {
    KvPlan p;
    p.pages = (max_cells + s.page_size - 1) / s.page_size;
    p.slots = p.pages;
    p.pooled_rows = max_cells / s.idx_block + 2;
    if (g_kv_resident <= 0 || ring_cells < 0) return p;   // ring_cells < 0: always fully resident
    // A/B only: STRATA_KV_RING_OFF keeps the drafter fully resident, STRATA_KV_MAIN_OFF the main layers
    static const bool ring_off = std::getenv("STRATA_KV_RING_OFF") != nullptr;
    static const bool main_off = std::getenv("STRATA_KV_MAIN_OFF") != nullptr;
    if (ring_cells > 0 ? ring_off : main_off) return p;
    if (ring_cells > 0) {
        const int64_t r = (ring_cells + s.page_size - 1) / s.page_size;
        if (r < p.pages) { p.mode = 2; p.slots = r; p.pooled_rows = 2; }   // the drafter has no indexer
    } else {
        const int64_t r = (std::max(g_kv_resident, qsa_kv_resident_min()) + s.page_size - 1) / s.page_size;
        if (r < p.pages) { p.mode = 1; p.slots = r; }
    }
    return p;
}
uint64_t kv_pool_bytes(const QsaShapes& s, int64_t pages, bool hybrid, bool int8) {
    if (hybrid) {   // K8V4: the INT8 K half (codes + scales) plus the Q4_0 V half (kv_q4.hpp's rotation)
        const uint64_t rows = (uint64_t) pages * s.page_size * s.n_head_kv;
        return rows * (uint64_t) s.head_dim + rows * (uint64_t) (s.head_dim / strata::kernels::KV_Q8_GROUP) * 2 +
               rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim) + 64;
    }
    if (g_kv_q4) return (uint64_t) pages * s.page_size * strata::kernels::kv_q4_bytes_per_cell(s) + 64;
    return int8 ? (uint64_t) pages * s.page_size * strata::kernels::kv_q8_bytes_per_cell(s) + 64
                : (uint64_t) pages * s.n_head_kv * s.page_size * s.head_dim * 2 * 2;
}
}  // namespace

void qsa_set_kv_resident(int64_t cells) { g_kv_resident = cells > 0 ? cells : 0; }
int64_t qsa_kv_resident() { return g_kv_resident; }
int64_t qsa_kv_resident_min() { return 20480; }
uint64_t qsa_kv_host_bytes() { return g_kv_host_bytes; }

uint64_t qsa_state_bytes(const ModelGeometry& g, int64_t max_cells, bool with_rope, int64_t ring_cells) {
    const QsaShapes s = qsa_shapes(g);
    const KvPlan p = kv_plan(s, max_cells, ring_cells);
    uint64_t n = 0;
    if (!(g_kv_elastic && p.mode == 0))   // the elastic K/V's pools are in their own VMM range
        n += kv_pool_bytes(s, p.slots, g_kv_hybrid && ring_cells <= 0,
                           g_kv_int8 || g_kv_hybrid) + 4 * 16;   // K/V pools (the VRAM slots)
    n += (uint64_t) p.pages * 4;                                               // page_table
    if (p.mode == 1) n += strata::kernels::kv_stream_map_bytes(p.slots) + 6 * 16;   // the residency map
    n += (uint64_t) (s.idx_block - 1) * s.idx_dim * 4;                         // tail
    n += (uint64_t) s.idx_dim * 4;                                             // dead
    n += (uint64_t) p.pooled_rows * s.idx_dim * 4;                             // pooled
    n += 16;                                                                   // block_pos
    if (with_rope) n += (uint64_t) max_cells * (s.n_rot / 2) * 4 * 2;          // cos + sin tables
    n += strata::kernels::qsa_step_bytes() + 16;                               // counts and aligned attention status
    n += (uint64_t) s.n_head * 4;                                              // pos_dev
    return align_up16(n) + 256;
}

uint64_t qsa_state_init(const ModelGeometry& g, int64_t max_cells, void* base, QsaState& st,
                        const QsaState* share_rope, int64_t ring_cells) {
    const QsaShapes s = qsa_shapes(g);
    const KvPlan p = kv_plan(s, max_cells, ring_cells);
    const int64_t pages = p.pages;
    Cursor c{(uint8_t*) base};
    st.kv_int8 = g_kv_int8 && !g_kv_q4;
    st.kv_q4 = g_kv_q4;
    // Hybrid K8V4, main layers only (the drafter's state is created with the globals toggled to INT8 -
    // mtp.cpp). A streamed one keeps its host copy in the same three runs (kv_stream.cu, kKvHybrid).
    if (g_kv_hybrid && ring_cells <= 0) {
        st.kv_hybrid = true;
        st.kv_int8 = false;
        st.kv_q4 = false;
    }
    st.kv_rot = st.kv_q4 || (st.kv_int8 && g_kv_int8_rot);   // K8V4 rotates only V (below)
    st.kv_mode = p.mode;
    st.n_slots = p.slots;
    const uint64_t rows = (uint64_t) p.slots * s.n_head_kv * s.page_size;   // VRAM rows: the slots
    const uint64_t q4_row = strata::kernels::kv_q4_bytes_per_head((int) s.head_dim);
    st.kv_elastic = -1;
    if (g_kv_elastic && p.mode == 0) {
        // the elastic K/V: each array at a chunk boundary of the state's own range, the first cells mapped
        const uint64_t slot_rows = (uint64_t) s.n_head_kv * s.page_size;
        const uint64_t scale_row = (uint64_t) (s.head_dim / strata::kernels::KV_Q8_GROUP) * 2;
        std::vector<uint64_t> per;   // bytes per page slot of each array, in the order they are assigned below
        if (st.kv_hybrid) per = {slot_rows * s.head_dim, slot_rows * scale_row, slot_rows * q4_row};
        else if (st.kv_q4) per = {slot_rows * q4_row, slot_rows * q4_row};
        else if (st.kv_int8) per = {slot_rows * s.head_dim, slot_rows * s.head_dim, slot_rows * scale_row, slot_rows * scale_row};
        else per = {slot_rows * s.head_dim * 2, slot_rows * s.head_dim * 2};
        auto pool = std::make_unique<ElasticPool>();
        const uint64_t G = strata::core::vmm_granularity();
        uint64_t at = 0;
        for (const uint64_t b : per) {
            pool->off.push_back(at);
            at += ((uint64_t) p.slots * b + G - 1) / G * G;
        }
        pool->per_slot = per;
        pool->n_slots = p.slots;
        pool->page_size = s.page_size;
        if (!pool->range.reserve(at) || !pool_grow(*pool, g_kv_elastic_init, [] { return (strata::core::VmmChunk) 0; })) {
            std::fprintf(stderr, "strata: the elastic K/V could not reserve or map its pools\n");
            return 0;
        }
        uint8_t* b = pool->range.base();
        const uint64_t* o = pool->off.data();
        if (st.kv_hybrid) { st.k_q = (int8_t*) (b + o[0]); st.k_scale = (uint16_t*) (b + o[1]); st.v_q4 = b + o[2]; }
        else if (st.kv_q4) { st.k_q4 = b + o[0]; st.v_q4 = b + o[1]; }
        else if (st.kv_int8) {
            st.k_q = (int8_t*) (b + o[0]);
            st.v_q = (int8_t*) (b + o[1]);
            st.k_scale = (uint16_t*) (b + o[2]);
            st.v_scale = (uint16_t*) (b + o[3]);
        } else { st.k_pool = (uint16_t*) (b + o[0]); st.v_pool = (uint16_t*) (b + o[1]); }
        st.kv_elastic = (int32_t) g_pools.size();
        g_pools.push_back(std::move(pool));
    } else if (st.kv_hybrid) {
        st.k_q = c.take<int8_t>(rows * s.head_dim);
        st.k_scale = c.take<uint16_t>(rows * (s.head_dim / strata::kernels::KV_Q8_GROUP));
        st.v_q4 = c.take<uint8_t>(rows * q4_row);
    } else if (st.kv_q4) {
        st.k_q4 = c.take<uint8_t>(rows * q4_row);
        st.v_q4 = c.take<uint8_t>(rows * q4_row);
    } else if (st.kv_int8) {   // st, not the global: the drafter's ring is INT8 under --kv k8v4 too
        st.k_q = c.take<int8_t>(rows * s.head_dim);
        st.v_q = c.take<int8_t>(rows * s.head_dim);
        st.k_scale = c.take<uint16_t>(rows * (s.head_dim / strata::kernels::KV_Q8_GROUP));
        st.v_scale = c.take<uint16_t>(rows * (s.head_dim / strata::kernels::KV_Q8_GROUP));
    } else {
        st.k_pool = c.take<uint16_t>(rows * s.head_dim);
        st.v_pool = c.take<uint16_t>(rows * s.head_dim);
    }
    st.page_table = c.take<int32_t>((uint64_t) pages);
    st.n_pages = pages;
    st.max_cells = max_cells;
    st.map = strata::kernels::KvStreamMap{};
    st.map.page_table = st.page_table;
    st.map.n_blocks = pages;
    st.map.n_slots = p.slots;
    if (p.mode == 1) {
        st.map.slot_block = c.take<int32_t>((uint64_t) p.slots);
        st.map.slot_stamp = c.take<int32_t>((uint64_t) p.slots);
        st.map.slot_ref = c.take<int32_t>((uint64_t) p.slots);
        st.map.miss_block = c.take<int32_t>((uint64_t) p.slots);
        st.map.miss_slot = c.take<int32_t>((uint64_t) p.slots);
        st.map.ctl = c.take<int32_t>((uint64_t) strata::kernels::kKvCtlInts);
    }
    st.idx_tail = c.take<float>((uint64_t) (s.idx_block - 1) * s.idx_dim);
    st.idx_dead = c.take<float>((uint64_t) s.idx_dim);
    st.idx_pooled_rows = p.pooled_rows;
    st.idx_pooled = c.take<float>((uint64_t) p.pooled_rows * s.idx_dim);
    st.idx_block_pos = c.take<int32_t>(1);
    if (share_rope != nullptr) {
        st.cos_tab = share_rope->cos_tab;
        st.sin_tab = share_rope->sin_tab;
    } else {
        st.cos_tab = c.take<float>((uint64_t) max_cells * (s.n_rot / 2));
        st.sin_tab = c.take<float>((uint64_t) max_cells * (s.n_rot / 2));
    }
    st.step = c.take<int32_t>((uint64_t) strata::kernels::kStepCount);
    st.attention_status = c.take<int32_t>(1);
    st.pos_dev = c.take<int32_t>((uint64_t) s.n_head);
    // the pinned staging the uploads copy FROM - see the note on `host_step` in the header
    if (strata::core::host_alloc_guarded((void**) &st.host_step, strata::kernels::qsa_step_bytes() + sizeof(int32_t),
                      cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        strata::core::host_alloc_guarded((void**) &st.host_pos, (size_t) s.n_head * 4, cudaHostAllocMapped | cudaHostAllocPortable) !=
            cudaSuccess) {
        return 0;   // the caller sees a zero byte count; a half-built state is worse than none
    }
    st.host_step[strata::kernels::kStepCount] = 0;
    // KV streaming: the authoritative K/V of every cell, pinned and device-mapped, in the identity layout
    st.host = strata::kernels::KvHostPools{};
    if (p.mode != 0) {
        const uint64_t hrows = (uint64_t) pages * s.n_head_kv * s.page_size;
        const uint64_t bytes = (uint64_t) pages * strata::kernels::kv_block_bytes(s, qsa_kv_format(st)) + 4 * 256;
        uint8_t* h = nullptr;
        uint8_t* d = nullptr;
        if ((h = kv_host_alloc(bytes)) == nullptr ||
            cudaHostGetDevicePointer((void**) &d, h, 0) != cudaSuccess) {
            // both cudaHostRegister and cudaHostAlloc were refused (kv_host_alloc): usually not enough free RAM
            if (p.mode == 1) std::fprintf(stderr, "strata: KV streaming: cannot pin %.2f GiB of RAM for a layer's KV copy "
                                 "(%.2f GiB pinned so far): RAM or the driver refused to pin it - lower the context, "
                                 "or run without --kv-resident\n", (double) bytes / 1073741824.0,
                                 (double) g_kv_host_bytes / 1073741824.0);
            return 0;
        }
        g_kv_host_bytes += bytes;
        Cursor hc{d};
        if (st.kv_hybrid) {
            st.host.k_q = hc.take<int8_t>(hrows * s.head_dim);
            st.host.k_scale = hc.take<uint16_t>(hrows * (s.head_dim / strata::kernels::KV_Q8_GROUP));
            st.host.v_q4 = hc.take<uint8_t>(hrows * q4_row);
        } else if (st.kv_q4) {
            st.host.k_q4 = hc.take<uint8_t>(hrows * q4_row);
            st.host.v_q4 = hc.take<uint8_t>(hrows * q4_row);
        } else if (st.kv_int8) {
            st.host.k_q = hc.take<int8_t>(hrows * s.head_dim);
            st.host.v_q = hc.take<int8_t>(hrows * s.head_dim);
            st.host.k_scale = hc.take<uint16_t>(hrows * (s.head_dim / strata::kernels::KV_Q8_GROUP));
            st.host.v_scale = hc.take<uint16_t>(hrows * (s.head_dim / strata::kernels::KV_Q8_GROUP));
        } else {
            st.host.k_pool = hc.take<uint16_t>(hrows * s.head_dim);
            st.host.v_pool = hc.take<uint16_t>(hrows * s.head_dim);
        }
    }
    // THE ROPE TABLE IS BUILT ON THE HOST IN FLOAT64 and uploaded once, because the reference computes its
    // frequencies in float64 and reproducing that on device means double-precision `pow`/`cos` that need not
    // agree with the host's libm.  A table shorter than the sequence would have the rotation read past it.
    // The process's rope scaling (none by default) is INSIDE the table - the rotation kernels cannot tell.
    if (share_rope == nullptr) {
        std::vector<float> hc((size_t) max_cells * (s.n_rot / 2)), hs((size_t) max_cells * (s.n_rot / 2));
        strata::kernels::build_rope_table((int) s.n_rot, strata::kernels::rope_scaling(), (int) max_cells,
                                          hc.data(), hs.data());
        cudaMemcpy(st.cos_tab, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(st.sin_tab, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice);
        st.owns_rope = true;
        if (s.n_rot == 64)   // the native and prompt-path rope kernels read it (mrope.hpp, STRATA_ROPE_TABLE=1)
            strata::kernels::rope_table_set(st.cos_tab, st.sin_tab, (int) max_cells, strata::kernels::rope_scaling());
    }
    // the page table starts as the IDENTITY, which is the simplest legal mapping and what a caller that does
    // not page at all wants; a streamed state starts with nothing resident, a ring at `block % n_slots`.
    if (p.mode == 0) {
        std::vector<int32_t> tab((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) tab[(size_t) i] = (int32_t) i;
        cudaMemcpy(st.page_table, tab.data(), tab.size() * 4, cudaMemcpyHostToDevice);
    } else if (p.mode == 1) {
        strata::kernels::kv_stream_reset(st.map, nullptr);
    } else {
        strata::kernels::kv_ring_table(st.page_table, pages, p.slots, nullptr);
    }
    cudaDeviceSynchronize();
    return c.used;
}

void qsa_state_zero(const QsaState& st, const ModelGeometry& g, void* stream) {
    const QsaShapes s = qsa_shapes(g);
    cudaStream_t cs = (cudaStream_t) stream;
    // an elastic state zeroes what is mapped (the rest is zeroed when it is mapped)
    const int64_t live_slots = st.kv_elastic >= 0 ? pool_slots_mapped(*g_pools[(size_t) st.kv_elastic]) : st.n_slots;
    const size_t rows = (size_t) live_slots * s.n_head_kv * s.page_size;
    if (st.kv_hybrid) {
        cudaMemsetAsync(st.k_q, 0, rows * s.head_dim, cs);
        cudaMemsetAsync(st.k_scale, 0, rows * (s.head_dim / strata::kernels::KV_Q8_GROUP) * 2, cs);
        cudaMemsetAsync(st.v_q4, 0, rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), cs);
    } else if (st.kv_q4) {
        cudaMemsetAsync(st.k_q4, 0, rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), cs);
        cudaMemsetAsync(st.v_q4, 0, rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), cs);
    } else if (st.kv_int8) {
        cudaMemsetAsync(st.k_q, 0, rows * s.head_dim, cs);
        cudaMemsetAsync(st.v_q, 0, rows * s.head_dim, cs);
        cudaMemsetAsync(st.k_scale, 0, rows * (s.head_dim / strata::kernels::KV_Q8_GROUP) * 2, cs);
        cudaMemsetAsync(st.v_scale, 0, rows * (s.head_dim / strata::kernels::KV_Q8_GROUP) * 2, cs);
    } else {
        cudaMemsetAsync(st.k_pool, 0, rows * s.head_dim * 2, cs);
        cudaMemsetAsync(st.v_pool, 0, rows * s.head_dim * 2, cs);
    }
    // A streamed state starts over with nothing resident. Its host copy is not cleared (GBs over PCIe per new
    // conversation): no reader names a cell before this sequence has written it, and a block copied in whole
    // carries the unwritten cells past the end, which nothing reads.
    if (st.kv_mode == 1) strata::kernels::kv_stream_reset(st.map, stream);
    cudaMemsetAsync(st.idx_tail, 0, (size_t) (s.idx_block - 1) * s.idx_dim * 4, cs);
    cudaMemsetAsync(st.idx_dead, 0, (size_t) s.idx_dim * 4, cs);
    cudaMemsetAsync(st.idx_pooled, 0, (size_t) st.idx_pooled_rows * s.idx_dim * 4, cs);
}

strata::kernels::QsaAttnPools qsa_attn_pools(const QsaState& st) {
    strata::kernels::QsaAttnPools pools;
    pools.page_table = st.page_table;
    if (st.kv_hybrid) { pools.k_q = st.k_q; pools.k_scale = st.k_scale; pools.v_q4 = st.v_q4; }
    else if (st.kv_q4) { pools.k_q4 = st.k_q4; pools.v_q4 = st.v_q4; }
    else if (st.kv_int8) { pools.k_q = st.k_q; pools.v_q = st.v_q; pools.k_scale = st.k_scale; pools.v_scale = st.v_scale; }
    else { pools.k_pool = st.k_pool; pools.v_pool = st.v_pool; }
    return pools;
}

void qsa_kv_resolve(const QsaState& st, const ModelGeometry& g, const int32_t* ids, const int32_t* steps, int64_t n_q,
                    int64_t cap, void* stream) {
    if (st.kv_mode != 1) return;
    strata::kernels::kv_stream_resolve(st.map, qsa_attn_pools(st), st.host, qsa_kv_format(st), ids, steps, n_q, cap,
                                       qsa_shapes(g), stream);
}

uint64_t qsa_buffers_bytes(const ModelGeometry& g, int64_t max_cells) {    const QsaShapes s = qsa_shapes(g);    const int64_t cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);    uint64_t n = 0;    n += (uint64_t) q8k_bytes(g.n_embd);    n += (uint64_t) (g.n_embd / 32) * 34;
// block_q8_0
n += (uint64_t) g.n_embd * 2;    n += (uint64_t) g.n_head * 2 * g.head_dim * 4;    n += (uint64_t) g.n_head * g.head_dim * 4;    n += (uint64_t) g.n_head_kv * g.head_dim * 4 * 2;    n += (uint64_t) g.idx_key_dim * 4;    n += (uint64_t) g.idx_q_heads * g.idx_key_dim * 4;    n += (uint64_t) max_cells * 4;    n += (uint64_t) cap * 4;    n += (uint64_t) cap * g.n_head_kv * g.head_dim * 2 * 2;    n += (uint64_t) g.n_head * g.head_dim * 4;
// attn
n += (uint64_t) g.n_head * g.head_dim * 2;
// attn16
n += (uint64_t) g.n_head * g.head_dim * 4;
// attn32
n += (uint64_t) q8k_bytes(g.n_head * g.head_dim);
// attn_q8k
n += strata::kernels::qsa_decode_attn_scratch_floats(cap, s) * 4 + 16;
// attn_scratch
return align_up16(n) + 256;}
uint64_t qsa_buffers_init(const ModelGeometry& g, int64_t max_cells, void* base, QsaBuffers& b) {    const QsaShapes s = qsa_shapes(g);    const int64_t cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);    Cursor c{(uint8_t*) base};    b.x_q8k = c.take_bytes(q8k_bytes(g.n_embd));    b.x_q8_0 = c.take_bytes((uint64_t) (g.n_embd / 32) * 34);    b.x_bf16 = c.take<uint16_t>((uint64_t) g.n_embd);    b.q_full = c.take<float>((uint64_t) g.n_head * 2 * g.head_dim);    b.qcur = c.take<float>((uint64_t) g.n_head * g.head_dim);    b.kcur = c.take<float>((uint64_t) g.n_head_kv * g.head_dim);    b.vcur = c.take<float>((uint64_t) g.n_head_kv * g.head_dim);    b.idx_raw = c.take<float>((uint64_t) g.idx_key_dim);    b.q_idx = c.take<float>((uint64_t) g.idx_q_heads * g.idx_key_dim);    b.cell_scores = c.take<float>((uint64_t) max_cells);    b.ids = c.take<int32_t>((uint64_t) cap);    b.k_scratch = c.take<uint16_t>((uint64_t) cap * g.n_head_kv * g.head_dim);    b.v_scratch = c.take<uint16_t>((uint64_t) cap * g.n_head_kv * g.head_dim);    b.attn = c.take<float>((uint64_t) g.n_head * g.head_dim);    b.attn16 = c.take<uint16_t>((uint64_t) g.n_head * g.head_dim);    b.attn32 = c.take<float>((uint64_t) g.n_head * g.head_dim);    b.attn_q8k = c.take_bytes(q8k_bytes(g.n_head * g.head_dim));    b.attn_scratch = c.take<float>(strata::kernels::qsa_decode_attn_scratch_floats(cap, s));    return c.used;}
// ================================ PER-STAGE TIMING, DEBUG ONLY ================================
//
// **THE ENGINE SPENDS 1.047 ms PER LAYER WITH THE EXPERTS OFF, AND EVERY COST MODEL IN `bench/` PREDICTS LESS
// THAN HALF OF THAT.**  `dense_pass.exe` puts all 302 dense GEMVs at 11.64 ms/token and the 1.377 GiB of
// BF16/F32 weights they exclude at another ~5.9, so ~17.5 ms of the 50.25 ms captured no-pool floor is
// accounted for and ~33 ms is not.  Arithmetic has already discounted the obvious candidates - the GDN
// recurrence moves 6.2 MB of state per token, the QSA attention at 1,024 tokens is 75M MACs over twelve
// layers, and `gr_norm_kernel`'s 4-block launch moves 40 KB - so the attribution has to come from the engine.
//
// Event pairs are recorded per (layer, stage) and read once at the end, so **NOTHING IS SYNCHRONISED DURING
// THE RUN** and the measurement does not perturb what it measures.  That matters here more than usual: a
// per-stage `cudaEventSynchronize` would serialise the pipeline and report the serialised time.
//
// **IT CANNOT BE USED WITH CAPTURED GRAPHS.**  An event record inside a stream capture is silently dropped
// (round 199: 2 nodes for two kernels plus an event record, and the event never completed across a 41 ms
// graph), so this is gated on the `--no-capture` path where the layers are ordinary launches.
constexpr int STAGE_SLOTS = 20;

struct StageTimer {
    bool on = false;
    cudaEvent_t a[64][STAGE_SLOTS] = {};
    cudaEvent_t b[64][STAGE_SLOTS] = {};
    const char* names[STAGE_SLOTS] = {};
    int nslots = 0;
};

static StageTimer g_st;

bool stage_timing_enable() {
    if (g_st.on) return true;
    for (int l = 0; l < 64; ++l) {
        for (int s = 0; s < STAGE_SLOTS; ++s) {
            if (cudaEventCreate(&g_st.a[l][s]) != cudaSuccess) return false;
            if (cudaEventCreate(&g_st.b[l][s]) != cudaSuccess) return false;
        }
    }
    g_st.on = true;
    return true;
}

void stage_timing_name(int slot, const char* name) {
    if (slot >= 0 && slot < STAGE_SLOTS) {
        g_st.names[slot] = name;
        if (slot + 1 > g_st.nslots) g_st.nslots = slot + 1;
    }
}

void stage_mark_begin(int64_t layer, int slot, void* stream) { st_begin(layer, slot, stream); }
void stage_mark_end(int64_t layer, int slot, void* stream) { st_end(layer, slot, stream); }

static void st_begin(int64_t layer, int slot, void* stream) {
    if (g_st.on && layer < 64) cudaEventRecord(g_st.a[layer][slot], (cudaStream_t) stream);
}

static void st_end(int64_t layer, int slot, void* stream) {
    if (g_st.on && layer < 64) cudaEventRecord(g_st.b[layer][slot], (cudaStream_t) stream);
}

void stage_timing_report(int64_t n_layers) {
    if (!g_st.on) return;
    cudaDeviceSynchronize();
    double tot[STAGE_SLOTS] = {};
    for (int s = 0; s < g_st.nslots; ++s) {
        for (int64_t l = 0; l < n_layers && l < 64; ++l) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, g_st.a[l][s], g_st.b[l][s]) == cudaSuccess) tot[s] += (double) ms;
        }
    }
    double sum = 0;
    for (int s = 0; s < g_st.nslots; ++s) sum += tot[s];
    std::fprintf(stderr, "\nper-stage GPU time, summed over %lld layers (one token):\n", (long long) n_layers);
    for (int s = 0; s < g_st.nslots; ++s) {
        std::fprintf(stderr, "  %-22s %8.3f ms/token   %6.3f ms/layer   %5.1f%%\n",
                     g_st.names[s] ? g_st.names[s] : "(unnamed)", tot[s], tot[s] / (double) n_layers,
                     sum > 0 ? 100.0 * tot[s] / sum : 0.0);
    }
    std::fprintf(stderr, "  %-22s %8.3f ms/token\n", "sum of stages", sum);
    // ---- THE PER-LAYER SPLIT, because the attention block is 55% of the token and the engine mixes two
    // entirely different attention implementations: GDN on 36 layers and QSA on the twelve where
    // `full_attention_interval` divides the index.  A single total cannot say which one is expensive, and the
    // two have nothing in common to optimise.
    std::fprintf(stderr, "\nper-layer, the attention block (ms).  A `*` marks a QSA layer:\n");
    double gdn = 0, qsa = 0;
    int ng = 0, nq = 0;
    for (int64_t l = 0; l < n_layers && l < 64; ++l) {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, g_st.a[l][1], g_st.b[l][1]) != cudaSuccess) continue;
        const bool is_qsa = (l % 4) == 3;
        if (is_qsa) { qsa += ms; ++nq; } else { gdn += ms; ++ng; }
        std::fprintf(stderr, "  %s%2lld %7.3f%s", is_qsa ? "*" : " ", (long long) l, (double) ms,
                     ((l + 1) % 8 == 0) ? "\n" : "");
    }
    std::fprintf(stderr, "\n  GDN: %d layers, %.3f ms total, %.4f ms/layer\n", ng, gdn, ng ? gdn / ng : 0.0);
    std::fprintf(stderr, "  QSA: %d layers, %.3f ms total, %.4f ms/layer\n", nq, qsa, nq ? qsa / nq : 0.0);
}

// Declared here because `qsa_layer` is defined above `block_layer_pre`, where the body lives.  The dump buffer
// is a plain host pointer and the copy is a graph node of whichever layer's capture is running, so the helper
// needs nothing from the layer it is called for beyond its index.
static void dump_slot(float* dump, const ModelGeometry& g, int64_t layer, const float* src, uint64_t off,
                      uint64_t n, void* stream);
bool qsa_layer(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t pos, int32_t pos_base,               const QsaState& st, const QsaBuffers& b, const float* x, float* out, void* stream,               std::string& err, float* dump) {    using namespace strata::kernels;    const QsaShapes s = qsa_shapes(g);    const LayerView v(tables, layer);    const int64_t n_kv = pos + 1;    /* P7 audit: RoPE reads cos/sin row pos_base + pos, and the table holds max_cells rows. */    if ((int64_t) pos_base + pos >= st.max_cells || pos_base < 0) {        err = "qsa_layer: position " + std::to_string((long long) pos_base + pos) + " is outside the RoPE table (" + std::to_string((long long) st.max_cells) + " rows)";        return false;    }    const int64_t cap = qsa_selection_width(kTopkMaxCells, s);
const auto normalize_rotate = [&](float* data, const WeightRef* norm, int rows, int cols) {
    try {
        if (native_qsa_enabled()) native_qsa_rms_norm_weighted(data, (const float*) norm->data, data, cols, rows, RMS_EPS, stream);
        else rms_norm_weighted(data, (const float*) norm->data, rows, cols, RMS_EPS, stream);
        if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), st.pos_dev, stream);
        else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st.cos_tab, st.sin_tab, st.pos_dev, stream);
        return true;
    } catch (const std::exception& error) {
        err = v.name("qsa_norm_rope") + ": " + error.what();
        return false;
    }
};
// ---- resolve every tensor by name, and REFUSE rather than reading a null.  `check_layer` has already
// asserted the shapes at load time; a missing name here is a wiring mistake.
struct Req { const char* suf; };    const WeightRef* w_idxk = v.get("indexer.k_proj.weight");    const WeightRef* w_attnq = v.get("attn_q.weight");    const WeightRef* w_attnk = v.get("attn_k.weight");    const WeightRef* w_attnv = v.get("attn_v.weight");    const WeightRef* w_attno = v.get("attn_output.weight");    const WeightRef* w_idxq = v.get("indexer.q_proj.weight");    const WeightRef* w_qn = v.get("attn_q_norm.weight");    const WeightRef* w_kn = v.get("attn_k_norm.weight");    const WeightRef* w_iqn = v.get("indexer.q_norm.weight");    const WeightRef* w_ikn = v.get("indexer.k_norm.weight");    const char* missing = !w_idxk ? "indexer.k_proj.weight" : !w_attnq ? "attn_q.weight"                          : !w_attnk ? "attn_k.weight" : !w_attnv ? "attn_v.weight"                          : !w_attno ? "attn_output.weight" : !w_idxq ? "indexer.q_proj.weight"                          : !w_qn ? "attn_q_norm.weight" : !w_kn ? "attn_k_norm.weight"                          : !w_iqn ? "indexer.q_norm.weight" : !w_ikn ? "indexer.k_norm.weight" : nullptr;    if (missing) { err = v.name(missing) + " is missing"; return false; }    if (pos < 0 || pos >= st.max_cells) {        err = "qsa_layer: pos " + std::to_string(pos) + " is outside the state's 0.." +              std::to_string(st.max_cells - 1);        return false;    }
// THE TWO BF16 PROJECTIONS: the arena holds them re-rounded to 2 B/elem, which is what `bf16_gemv` wants.
// Reading one as f32 would walk 2x its length inside the arena without faulting.
if (w_idxk->kind != WeightKind::Bf16InF32 || w_idxq->kind != WeightKind::Bf16InF32) {        err = v.name("indexer.*_proj.weight") + " must be engine form 1 (bf16); they are " +              std::to_string((int) w_idxk->kind) + " and " + std::to_string((int) w_idxq->kind);        return false;    }
// ---- 1. the three activation formats, once each
bool images_fused = false;
#if defined(STRATA_EXPERIMENTAL_SM60)
if (strata::kernels::fuse_decode() && (!w_attnk->native_data || !w_attnv->native_data || !w_attnq->native_data)) {
    quantize_act_images(x, g.n_embd, b.x_q8_0, b.x_q8k,
                        native_bf16_projections ? nullptr : b.x_bf16, stream);   // ONE pass, was two or three
    images_fused = true;
}
#endif
if (!images_fused) {
    if (!w_attnk->native_data || !w_attnv->native_data || !w_attnq->native_data) {
        quantize_q8_K(x, b.x_q8k, g.n_embd, stream);
        quantize_q8_0(x, b.x_q8_0, g.n_embd, stream);
    }
    if (!native_bf16_projections) f32_to_bf16_bulk(x, b.x_bf16, g.n_embd, stream);
}
// ---- 2. the step state: ONE H2D carries every per-token count the kernels need.  Nothing below passes a
// per-token scalar as an argument, which is what keeps this layer capturable.
//
// THE SOURCES ARE `st.host_step`/`st.host_pos`, WHICH LIVE AS LONG AS THE STATE DOES.  A stack array or a
// local vector here would be captured as a dangling POINTER and replayed as garbage - see the note in
// `layer.hpp`.  That is a silent failure, not a fault: the copy succeeds, the counts are wrong, and the
// token that comes out is plausible.
{        qsa_step_fill(st.host_step, pos, s);        for (int64_t h = 0; h < g.n_head; ++h) st.host_pos[h] = (int32_t) (pos_base + pos);
        int32_t* m_step = nullptr;
        int32_t* m_pos = nullptr;
        if (g_publish_kernel && cudaHostGetDevicePointer((void**) &m_step, st.host_step, 0) == cudaSuccess &&
            cudaHostGetDevicePointer((void**) &m_pos, st.host_pos, 0) == cudaSuccess) {
            strata::kernels::copy_i32_from_mapped(st.step, m_step, strata::kernels::kStepCount, stream);
            strata::kernels::copy_i32_from_mapped(st.pos_dev, m_pos, g.n_head, stream);
        } else
        if (cudaMemcpyAsync(st.step, st.host_step, qsa_step_bytes(), cudaMemcpyHostToDevice,                            (cudaStream_t) stream) != cudaSuccess ||            cudaMemcpyAsync(st.pos_dev, st.host_pos, (size_t) g.n_head * 4, cudaMemcpyHostToDevice,                            (cudaStream_t) stream) != cudaSuccess) {            err = "qsa_layer: the step-state upload failed";            return false;        }    }
// ---- 3. the indexer's RAW key (and, with the V100 switch, its QUERY in the same kernel; was two launches)
bool idx_pair = false;
#if defined(STRATA_EXPERIMENTAL_SM60)
if (strata::kernels::fuse_decode()) {
    // two BF16 projections of the same x in ONE kernel: the query is needed only at step 7 and nothing in between
    // touches it, so computing it here changes no order that matters; `idx_raw` is still appended at step 5.
    project_bf16_pair(x, b.x_bf16, (const uint16_t*) w_idxk->data, b.idx_raw, g.idx_key_dim,
                      (const uint16_t*) w_idxq->data, b.q_idx, g.idx_q_heads * g.idx_key_dim, g.n_embd,
                      false, stream);
    idx_pair = true;
}
#endif
if (!idx_pair)
    project_bf16(x, b.x_bf16, (const uint16_t*) w_idxk->data, b.idx_raw, g.n_embd, g.idx_key_dim, false, stream);
// ---- 4. K and V, in Q8_K, then norm and rotate K only
SForm f_k, f_v, f_o, f_q;    if (!sform_of(*w_attnk, f_k, v.name("attn_k.weight"), err)) return false;    if (!sform_of(*w_attnv, f_v, v.name("attn_v.weight"), err)) return false;    if (!sform_of(*w_attno, f_o, v.name("attn_output.weight"), err)) return false;    if (!sform_of(*w_attnq, f_q, v.name("attn_q.weight"), err)) return false;    Planes p_k, p_v, p_o, p_q;    if (!plane_ptrs(*w_attnk, v.name("attn_k.weight"), p_k, err)) return false;    if (!plane_ptrs(*w_attnv, v.name("attn_v.weight"), p_v, err)) return false;    if (!plane_ptrs(*w_attno, v.name("attn_output.weight"), p_o, err)) return false;    if (!plane_ptrs(*w_attnq, v.name("attn_q.weight"), p_q, err)) return false;
// k and v, with the activation THIS layer's tensors ask for.  Both are K-quants in every QSA layer of this
// artifact, but the dispatch is here for the same reason it is in `gdn_layer`: the pack decides per tensor,
// and "it happens to be uniform here" is the assumption that was wrong for `attn_q`.
if (!gemv_quantized(*w_attnk, p_k, f_k, b.x_q8_0, b.x_q8k, b.kcur, g.n_embd, g.n_head_kv * g.head_dim,                        v.name("attn_k.weight"), stream, err, x)) return false;
    if (!gemv_quantized(*w_attnv, p_v, f_v, b.x_q8_0, b.x_q8k, b.vcur, g.n_embd, g.n_head_kv * g.head_dim,                        v.name("attn_v.weight"), stream, err, x, w_attnk->native_data && w_attnv->native_data)) return false;    dump_slot(dump, g, layer, b.vcur,                            (uint64_t) 2 * g.n_embd + 2 * g.hc + (uint64_t) g.n_head * g.head_dim,                            (uint64_t) g.n_head_kv * g.head_dim, stream);
    if (!normalize_rotate(b.kcur, w_kn, (int) g.n_head_kv, (int) g.head_dim)) return false;
// ---- 5. into the cache, and the indexer's append (which pools AND rotates on a block completion).
// EVERY ENTRY POINT FROM HERE ON IS THE CAPTURABLE ONE: the per-token counts come from `st.step` and every
// launch is sized from a capacity in `st`/`b`, so this sequence can be captured and replayed.  The
// host-scalar wrappers would be correct here today and silently wrong in a graph.
if (st.kv_hybrid) {
    // K8V4: only V is rotated (kv_q4.hpp's H); the scores pair unrotated q with unrotated INT8 K, and the
    // output - a mix of rotated values - is rotated back after attention. Each append/gather call folds the
    // unused half's lanes onto the used pool (a bit-identical duplicate write), so no kernel variants exist.
    // Streamed (--kv-resident), each half also writes its part of the host copy (kv_hybrid_*_half).
    strata::kernels::fwht256_inplace_cuda(b.vcur, g.n_head_kv, stream);
    const strata::kernels::KvHostPools hk = strata::kernels::kv_hybrid_k_half(st.host),
                                       hv = strata::kernels::kv_hybrid_v_half(st.host);
    const bool mirror = st.host.present();
    kv_append_q8_step(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, st.step, b.kcur, b.kcur, s, stream,
                      mirror ? &hk : nullptr);
    strata::kernels::kv_append_q4_step(st.v_q4, st.v_q4, st.page_table, st.step, b.vcur, b.vcur, s, stream,
                                       mirror ? &hv : nullptr);
} else {
if (st.kv_rot) {   // rotated K and V (kv_q4.hpp): Q4_0, and INT8 with STRATA_KV_ROT=1
    strata::kernels::fwht256_inplace_cuda(b.kcur, g.n_head_kv, stream);
    strata::kernels::fwht256_inplace_cuda(b.vcur, g.n_head_kv, stream);
}
if (st.kv_q4) {
    strata::kernels::kv_append_q4_step(st.k_q4, st.v_q4, st.page_table, st.step, b.kcur, b.vcur, s, stream, &st.host);
} else if (st.kv_int8) kv_append_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, st.step, b.kcur, b.vcur, s, stream, &st.host);    else kv_append_step(st.k_pool, st.v_pool, st.page_table, st.step, b.kcur, b.vcur, s, stream, &st.host); } /* not K8V4 */    {        const uint64_t nvk = (uint64_t) g.n_head_kv * g.head_dim;        const uint64_t base = (uint64_t) 2 * g.n_embd + 2 * g.hc + (uint64_t) g.n_head * g.head_dim + 2 * nvk + 8;        if (!st.kv_int8 && !st.kv_hybrid) dump_slot(dump, g, layer, (const float*) st.k_pool, base, nvk / 2, stream);        if (!st.kv_int8 && !st.kv_hybrid) dump_slot(dump, g, layer, (const float*) st.v_pool, base + nvk / 2, nvk / 2, stream);        dump_slot(dump, g, layer, b.vcur, base + nvk, nvk, stream);        dump_slot(dump, g, layer, b.kcur, base + 2 * nvk, nvk, stream);    }    {        const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};        if (native_qsa_indexer_enabled()) {
    try {
        native_qsa_indexer_append(b.idx_raw, st.step + kStepPos, pos_base,
            (const float*) w_ikn->data, RMS_EPS, ib, s, st.max_cells, rope_scaling(), stream);
    } catch (const std::exception& error) { err = v.name("native_indexer") + ": " + error.what(); return false; }
} else indexer_key_append(b.idx_raw, st.pos_dev, pos_base, (const float*) w_ikn->data, RMS_EPS, ib, s,
                          st.cos_tab, st.sin_tab, stream);    }
// ---- 6. the query projection, and the HALF-SPLIT into q and gate.
//
// **`attn_q` IS *NOT* ALWAYS Q2_0.**  It is Q2_0 on TWO of the twelve QSA layers, Q3_K on six and IQ4_XS on
// four, and this line used to call `s2_gemv_q8` unconditionally - so TEN of the twelve layers decoded an
// S4 code plane with a Q8_0 activation.  Both are "a quantized activation", both produce a plausible
// vector, and the difference is 0.6-1.4%.
if (!gemv_quantized(*w_attnq, p_q, f_q, b.x_q8_0, b.x_q8k, b.q_full, g.n_embd, g.n_head * 2 * g.head_dim,                        v.name("attn_q.weight"), stream, err, x, w_attnv->native_data && w_attnq->native_data)) return false;
// `per_head[:, :head_dim]` - the FIRST half of each head's 2*head_dim block, copied out contiguously so
// the norm and the rotation see whole rows.  A 2-D copy is a memcpy node, which captures (`pinned_capture`
// case A) and needs no kernel.
if (cudaMemcpy2DAsync(b.qcur, (size_t) g.head_dim * 4, b.q_full, (size_t) g.head_dim * 2 * 4,                          (size_t) g.head_dim * 4, (size_t) g.n_head, cudaMemcpyDeviceToDevice,                          (cudaStream_t) stream) != cudaSuccess) {        err = "qsa_layer: the q/gate split failed";        return false;    }    if (!normalize_rotate(b.qcur, w_qn, (int) g.n_head, (int) g.head_dim)) return false;
// ---- 7. the indexer's query: with the V100 switch it was projected at step 3 (paired with the raw key);
// the trunk projects it here.  Either way, norm and rotate it now.
if (!idx_pair) project_bf16(x, b.x_bf16, (const uint16_t*) w_idxq->data, b.q_idx, g.n_embd, g.idx_q_heads * g.idx_key_dim,
             false, stream);
if (!normalize_rotate(b.q_idx, w_iqn, (int) g.idx_q_heads, (int) g.idx_key_dim)) return false;
// ---- 8. score, select, gather, attend.  `max_blocks` and `cap` are CAPACITIES from the state, not this
// token's counts: a grid or a shared-memory size that follows the sequence length is baked into a captured
// graph, and the kernels guard for the surplus.    const
int64_t max_blocks = (st.max_cells / s.idx_block) + 2;
    if (g_fast_select) {
        // Plan v0.3 P7: block-level FP32 scores and a radix selection over blocks (qsa_select.hpp).
        qsa_block_scores(st.idx_pooled, st.idx_dead, b.q_idx, st.step, 1, max_blocks, s, b.cell_scores, stream);
        qsa_block_topk(b.cell_scores, st.step, 1, max_blocks, cap, s, b.ids, stream);
    } else {
    qsa_index_step(st.idx_pooled, b.q_idx, nullptr, s, st.step, max_blocks, b.cell_scores, stream);    topk_512_step(b.cell_scores, s, cap, st.step, b.ids, stream);
    }
    // KV streaming: every block the selection names is made resident before anything reads it
    qsa_kv_resolve(st, g, b.ids, st.step, 1, cap, stream);
    if (st.kv_rot) strata::kernels::fwht256_inplace_cuda(b.qcur, g.n_head, stream);   // <Hq, Hk> = <q, k>
    if (g_fast_attn && !native_flash_attn_short && dump == nullptr) {
        const strata::kernels::QsaAttnPools pools = qsa_attn_pools(st);
        strata::kernels::qsa_decode_attn_step(b.qcur, pools, b.ids, st.step, cap, s, b.attn_scratch, b.attn, stream);
    } else {
    if (st.kv_hybrid) {
        kv_gather_q8_step(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, b.ids, st.step, cap, s,
                          b.k_scratch, b.k_scratch, stream);
        strata::kernels::kv_gather_q4_step(st.v_q4, st.v_q4, st.page_table, b.ids, st.step, cap, s,
                                           b.v_scratch, b.v_scratch, stream);
    }
    else if (st.kv_q4) strata::kernels::kv_gather_q4_step(st.k_q4, st.v_q4, st.page_table, b.ids, st.step, cap, s, b.k_scratch, b.v_scratch, stream);
    else if (st.kv_int8) kv_gather_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, b.ids, st.step, cap, s,                                 b.k_scratch, b.v_scratch, stream);
    else kv_gather_step(st.k_pool, st.v_pool, st.page_table, b.ids, st.step, cap, s, b.k_scratch, b.v_scratch,                   stream);
    if (native_flash_attn_short) {
    if (st.max_cells < 1 || st.max_cells > 256 || !st.attention_status || !st.host_step) {
        err = v.name("native_flash_attn") + ": short adapter requires context <=256 and persistent status storage";
        return false;
    }
    try {
        native_flash_attn_short_step(b.qcur, b.k_scratch, b.v_scratch, st.step, cap,
            (int) st.max_cells, s, b.attn, st.attention_status, nullptr, stream);
    } catch (const std::exception& error) { err = v.name("native_flash_attn") + ": " + error.what(); return false; }
    if (cudaMemcpyAsync(st.host_step + kStepCount, st.attention_status, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, (cudaStream_t) stream) != cudaSuccess) {
        err = v.name("native_flash_attn") + ": status readback failed"; return false;
    }
} else qsa_attend_step(b.qcur, b.k_scratch, b.v_scratch, st.step, cap, s, b.attn, nullptr, stream);
    }
    if (st.kv_rot || st.kv_hybrid) strata::kernels::fwht256_inplace_cuda(b.attn, g.n_head, stream);   // the output back: H is self-inverse
    dump_slot(dump, g, layer, b.attn, (uint64_t) 2 * g.n_embd + 2 * g.hc,                            (uint64_t) g.n_head * g.head_dim, stream);    {        const uint64_t vs = (uint64_t) 2 * g.n_embd + 2 * g.hc + (uint64_t) g.n_head * g.head_dim +                            (uint64_t) 2 * g.n_head_kv * g.head_dim;        dump_slot(dump, g, layer, (const float*) b.v_scratch, vs,                            (uint64_t) g.n_head_kv * g.head_dim / 2, stream);        dump_slot(dump, g, layer, (const float*) b.ids, vs + (uint64_t) g.n_head_kv * g.head_dim / 2, 4, stream);        dump_slot(dump, g, layer, (const float*) st.step,                            vs + (uint64_t) g.n_head_kv * g.head_dim / 2 + 4, 4, stream);    }
// ---- 9. Gate the attention output before its projection. Native CUDA keeps F32
// sigmoid/multiply arithmetic and uses Q8_1 for the native quantized projection.
// The canonical fallback retains its prior FP64 gate and Q8_K projection.
try {
    if (native_qsa_enabled()) native_qsa_gate_apply(b.attn, b.q_full, b.attn32, (int) g.n_head, (int) g.head_dim, stream);
    else qsa_gate_apply_f32(b.attn, b.q_full, s, b.attn32, stream);
} catch (const std::exception& error) { err = v.name("qsa_gate") + ": " + error.what(); return false; }
    if (!w_attno->native_data) quantize_q8_K(b.attn32, b.attn_q8k, g.n_head * g.head_dim, stream);
    if (w_attno->native_data) {
    if (!gemv_quantized(*w_attno, p_o, f_o, nullptr, b.attn_q8k, out,
        g.n_head * g.head_dim, g.n_embd, v.name("attn_output.weight"), stream, err, b.attn32)) return false;
} else {
    s_gemv_q8k_split(b.attn_q8k, p_o.codes, p_o.scales, p_o.offset, out,
                    g.n_head * g.head_dim, g.n_embd, f_o, stream);
}    return true;}
// ================================ THE DOORBELL ================================
namespace {}
// namespace
uint64_t doorbell_init(const ModelGeometry& g, int64_t k, Doorbell& db) {    db.n_embd = g.n_embd;    db.k = k;    uint64_t bytes = 0;
// ONE region per field, each MAPPED PINNED, so the device and the host have different pointers to the same
// bytes and no copy is needed to publish them.
auto alloc = [&](size_t n, void** h, void** d, const char* what) {        if (cudaHostAlloc(h, n, cudaHostAllocMapped) != cudaSuccess) {            std::fprintf(stderr, "doorbell_init: cudaHostAlloc(%s) failed\n", what);            return false;        }        if (cudaHostGetDevicePointer(d, *h, 0) != cudaSuccess) {            std::fprintf(stderr, "doorbell_init: cudaHostGetDevicePointer(%s) failed\n", what);            return false;        }        std::memset(*h, 0, n);        bytes += n;        return true;    };    if (!alloc((size_t) g.n_embd * 4, (void**) &db.h_x_f, (void**) &db.d_x_f, "x_f")) return 0;    if (!alloc((size_t) k * 4, (void**) &db.h_ids, (void**) &db.d_ids, "ids")) return 0;    if (!alloc((size_t) k * 4, (void**) &db.h_weights, (void**) &db.d_weights, "weights")) return 0;    if (!alloc(4, (void**) &db.h_seq, (void**) &db.d_seq, "seq")) return 0;    if (!alloc(4, (void**) &db.h_flag, (void**) &db.d_flag, "flag")) return 0;    return bytes;}
void doorbell_free(Doorbell& db) {    if (db.h_x_f) (void) cudaFreeHost(db.h_x_f);    if (db.h_ids) (void) cudaFreeHost(db.h_ids);    if (db.h_weights) (void) cudaFreeHost(db.h_weights);    if (db.h_seq) (void) cudaFreeHost(db.h_seq);    if (db.h_flag) (void) cudaFreeHost(db.h_flag);    db = Doorbell{};}
void doorbell_reset(const Doorbell& db) {
    if (db.h_seq) *db.h_seq = 0;
    if (db.h_flag) *(volatile uint32_t*) db.h_flag = 0;
}
// ================================ THE TWO ENDS OF A TOKEN ================================
bool embed_row(const WeightTable& tables, const ModelGeometry& g, int64_t token, float* out_dev,
               void* stream, std::string& err) {
    if (const NativeEmbed* ne = native_embed()) {   // plan v0.3 P6: the GGUF-form table (IQ model files)
        if (token < 0 || out_dev == nullptr) { err = "embed_row: invalid token or output"; return false; }
        ne->gather_one(token, out_dev, stream);
        return true;
    }
    const WeightRef* w = tables.find(EMBEDDING_NAME);
    if (w == nullptr) { err = "token_embd.weight is missing"; return false; }
    if (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8) {
        err = "embed_row: token_embd.weight is not an S2/S4/S8 tensor";
        return false;
    }
    if (w->codebook_iq4nl) {
        err = "embed_row: an IQ4NL embedding is not supported";
        return false;
    }
    if (w->data == nullptr || out_dev == nullptr || w->ne0 <= 0 || w->ne1 <= 0 ||
        w->ne0 != g.n_embd || w->group_elems <= 0 ||
        w->ne0 % w->group_elems != 0 || w->ne0 % (8 / w->code_bits) != 0) {
        err = "embed_row: invalid embedding pointers, dimensions or group size";
        return false;
    }
    if (token < 0 || token >= w->ne1) {
        err = "embed_row: token " + std::to_string(token) + " is outside 0.." + std::to_string(w->ne1 - 1);
        return false;
    }

    const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
    const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
    const uint64_t rows = (uint64_t) w->ne1;
    // Division-based checks avoid overflow in rows * row_bytes and reject partial
    // code/group rows before any device pointer arithmetic.
    if (w->codes_bytes > w->bytes || w->scales_bytes > w->bytes - w->codes_bytes ||
        w->offset_bytes != w->bytes - w->codes_bytes - w->scales_bytes ||
        w->codes_bytes % row_codes != 0 || w->codes_bytes / row_codes != rows ||
        w->codes_bytes % alignof(float) != 0 || w->scales_bytes % sizeof(float) != 0 ||
        (w->scales_bytes / sizeof(float)) % row_groups != 0 ||
        (w->scales_bytes / sizeof(float)) / row_groups != rows ||
        w->has_offset != (w->offset_bytes != 0) ||
        (w->has_offset && w->offset_bytes != w->scales_bytes)) {
        err = "embed_row: embedding planes do not match the loaded tensor shape";
        return false;
    }

    const auto* codes = static_cast<const uint8_t*>(w->data);
    const auto* scales = reinterpret_cast<const float*>(codes + w->codes_bytes);
    const auto* offsets = w->has_offset ? reinterpret_cast<const float*>(codes + w->codes_bytes + w->scales_bytes)
                                       : nullptr;
    const uint64_t group_offset = (uint64_t) token * row_groups;
    strata::kernels::embedding_gather(codes + (uint64_t) token * row_codes, scales + group_offset,
                                       offsets ? offsets + group_offset : nullptr, w->ne0,
                                       w->code_bits, w->code_bias, w->group_elems, out_dev, stream);
    return true;
}
bool lm_head_mix(const WeightTable& tables, const ModelGeometry& g, const BlockBuffers& bb,
                 void* stream, std::string& err) {
    const WeightRef* wn = tables.find("output_hc_norm.weight");
    const WeightRef* wd = tables.find("output_hc_down.weight");
    const WeightRef* wu = tables.find("output_hc_up.weight");
    if (!wn || !wd || !wu) {
        err = "lm_head: an output_hc_* weight is missing";
        return false;
    }
    if (wn->kind != WeightKind::F32 || wd->kind != WeightKind::Bf16InF32 ||
        wu->kind != WeightKind::Bf16InF32) {
        err = "lm_head: the output_hc_* weights have the wrong engine forms";
        return false;
    }
    const strata::kernels::GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    strata::kernels::gr_read(bb.R, (const float*) wn->data, (const uint16_t*) wd->data,
                            (const uint16_t*) wu->data, nullptr, RMS_EPS, gs, bb.gr,
                            bb.mixed, bb.inject, stream);
    return true;
}

bool lm_head(const WeightTable& tables, const ModelGeometry& g, const BlockBuffers& bb,
             float* logits, void* stream, std::string& err) {
    const WeightRef* wo = tables.find("output.weight");
    if (!wo) { err = "output.weight is missing"; return false; }
    strata::kernels::SForm form;
    Planes planes;
    if (!sform_of(*wo, form, "output.weight", err) ||
        !plane_ptrs(*wo, "output.weight", planes, err)) return false;
    if (g.n_embd % Q8K_ELEMS_PER_BLOCK != 0) {
        err = "lm_head: n_embd is not a multiple of 256";
        return false;
    }
    if (!lm_head_mix(tables, g, bb, stream, err)) return false;
    strata::kernels::quantize_q8_K(bb.mixed, bb.head_q8k, g.n_embd, stream);
    return gemv_quantized(*wo, planes, form, bb.head_q8k, bb.head_q8k, logits,
                          g.n_embd, wo->ne1, "output.weight", stream, err);
}
// ================================ ONE WHOLE BLOCK ================================
uint64_t block_buffers_bytes(const ModelGeometry& g) {    const strata::kernels::GrShapes s{g.n_embd, g.hc, g.hc_lr};    uint64_t n = 0;    n += (uint64_t) g.hc * g.n_embd * 4;
// R
n += (uint64_t) g.n_embd * 4;
// mixed
n += (uint64_t) g.n_embd * 4;
// block_out
n += (uint64_t) g.hc * 4;
// inject
n += (uint64_t) g.hc * 4 * 2 + 32;
// inject2, gr_rs
n += q8k_bytes(g.n_embd);
// head_q8k
n += strata::kernels::gr_workspace_bytes(s);    return align_up16(n) + 256;}
uint64_t block_buffers_init(const ModelGeometry& g, void* base, BlockBuffers& b) {    const strata::kernels::GrShapes s{g.n_embd, g.hc, g.hc_lr};    Cursor c{(uint8_t*) base};    b.R = c.take<float>((uint64_t) g.hc * g.n_embd);    b.mixed = c.take<float>((uint64_t) g.n_embd);    b.block_out = c.take<float>((uint64_t) g.n_embd);    b.inject = c.take<float>((uint64_t) g.hc);    b.inject2 = c.take<float>((uint64_t) g.hc);    b.gr_rs = c.take<float>((uint64_t) g.hc);    b.head_q8k = c.take_bytes(q8k_bytes(g.n_embd));    uint8_t* grw = c.take_bytes(strata::kernels::gr_workspace_bytes(s));    strata::kernels::gr_workspace_init(s, grw, b.gr);    return c.used;}
// ---- THE HALF-LEVEL C1 ORACLE.  `dump` is a HOST buffer of `2 * n_embd + 2 * hc + n_head * head_dim` floats
// per layer; this enqueues one slice of it as a device-to-host copy.  **IT RUNS INSIDE THE CAPTURE**, which is
// the point: the copy becomes a node of that layer's graph and replays with it, so the layer functions stay one
// function each and no per-layer plumbing reaches `generate.cpp`.  A null `dump` is the default and skips the
// enqueue entirely, so the fast path records no extra node.
uint64_t dump_stride_floats(const ModelGeometry& g) {
    // Layout per layer, all in floats:
    //   [0, 2*n_embd)                     the two halves' block_out
    //   [2*n_embd, +2*hc)                 the two halves' inject
    //   [+NH*HD)                          the QSA pre-gate attention
    //   [+2*NKV*HD)                       the QSA RAW vcur  (as floats)
    //   [+NKV*HD/2)                       the gathered v_scratch, as fp16 pairs
    //   [+4) [+4)                         ids[0..3], then step[0..3]
    //   [+NKV*HD/2)                       the v_pool prefix, as fp16 pairs
    //   [+NKV*HD)                         the post-norm POST-ROPE kcur
    // The last four are what separate a wrong projection from a wrong pool round trip.
    const uint64_t nvk = (uint64_t) g.n_head_kv * g.head_dim;
    return (uint64_t) 2 * g.n_embd + (uint64_t) 2 * g.hc + (uint64_t) g.n_head * g.head_dim + 5 * nvk + 8;
}
static void dump_slot(float* dump, const ModelGeometry& g, int64_t layer, const float* src, uint64_t off,
                      uint64_t n, void* stream) {
    if (dump == nullptr || src == nullptr || n == 0) return;
    cudaMemcpyAsync(dump + (size_t) layer * dump_stride_floats(g) + off, src, n * sizeof(float),
                    cudaMemcpyDeviceToHost, (cudaStream_t) stream);
}
static void dump_half(const BlockBuffers& bb, const ModelGeometry& g, int64_t layer, const float* src,
                      uint64_t off, uint64_t n, void* stream) {
    dump_slot(bb.dump, g, layer, src, off, n, stream);
}
bool block_layer_pre(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t pos,                     int32_t pos_base, const GdnBuffers& gb, const QsaState& qst, const QsaBuffers& qb,                     const MoEBuffers& mb, int64_t k, const BlockBuffers& bb, void* stream, std::string& err,                     const Doorbell* db, const PleRun* ple, int half, int stage_prefix) {
    // ================================ THE PLE, AT LAYER 1 ONLY ================================
    //
    // **THIS IS THE CALL WHOSE ABSENCE FAILED GATE C1.**  Layer 1 carries the six `blk.1.ple_*` tensors and no
    // other layer does; every part of the module was built and parity-tested and none of it was reachable from
    // here.  The block's output is `hidden + gated + silu(conv)` on the RESIDUAL STACK ITSELF, so it is a
    // residual update applied BEFORE layer 1's own `gr_read` - which is what the architecture means by the
    // gathered rows arriving "before layer 2".
    //
    // **THE HASH AND THE GATHER ARE *NOT* HERE, AND THAT IS THE WHOLE REASON `ple_stage_token` EXISTS.**  Both
    // are HOST operations - the hash is 64-bit multiply/xor that `qwen4exp.cpp` says is host-side because "ggml
    // has no int64 and no xor", and the table is a host mapping.  This function is CAPTURED, so anything host
    // side in it runs ONCE, at capture time, and the graph then replays that result for every token: the engine
    // would gather the capture-time token's rows forever and produce a perfectly finite, perfectly stable,
    // completely wrong answer.  It is the same bug as the doorbell's cloned literal, and it is silent.
    //
    // So the driver calls `ple_stage_token` once per token, BEFORE the graphs, and what is captured here is only
    // the device half: `ple_block` reading `emb_dev`, and the history shift.
    const bool fused = g_fused_gr && stage_prefix == 0 && half == 0 &&
                       strata::kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr);
    // layer-1's FFN write has not been applied to R yet - unless a control vector follows it, which needs R
    bool pending_ffn = fused && layer > 0 && !strata::kernels::cvec().covers(layer - 1);
    if (ple != nullptr && ple->ready() && layer == 1 && (half == 0 || half == 1)) {
        if (pending_ffn) {
            const strata::kernels::GrShapes gs0{g.n_embd, g.hc, g.hc_lr};
            gr_write(bb.R, bb.block_out, bb.inject2, gs0, bb.R, stream);
            pending_ffn = false;
        }
        const int64_t hcd = strata::kernels::NG_HC_DIM;
        strata::kernels::PleOut po;
        // Exports must not alias the block's internal workspace. The previous diagnostic views used a
        // different layout inside that workspace: exporting gated values overwrote normalized values
        // before the latter were copied to history. Allocate only the one export the recurrence needs.
        po.normalized = (float*) ((uint8_t*) ple->scratch + strata::kernels::ple_block_scratch_bytes());
        // `result` IS THE RESIDUAL, in place: the block computes `hidden[i] + gated[i] + conv[i]` elementwise,
        // so reading and writing `R` at the same index is well-defined.
        po.result = bb.R;
        try {
            strata::kernels::ple_block(ple->emb_dev, bb.R, ple->hist, ple->w, po, ple->scratch, stream);
        } catch (const std::exception& error) {
            err = std::string("block_layer_pre PLE: ") + error.what();
            return false;
        }

        // ---- AND THE HISTORY ADVANCES, ONCE PER TOKEN, WHICH IS WHAT MAKES IT A CONV HISTORY.
        //
        // `ple_block` deliberately does NOT write back to the history - "the caller keeps ownership of its
        // state and this stays a pure function of its inputs, which is what makes the chunked-versus-single-shot
        // property testable" - so the shift is here.  The layout is ROW-FASTEST (`hist[row + NG_HIST*channel]`,
        // from `ggml_reshape_3d(state, d_conv-1, conv_channels, n_seqs)`), so a row shift is a STRIDED copy of
        // NG_HIST-1 elements per channel and an append is a strided copy of one - not a flat memmove, which
        // would be the natural reading and would scramble the channels.
        strata::kernels::ple_history_advance(ple->hist, po.normalized, stream);
        if (cudaPeekAtLastError() != cudaSuccess) {
            err = "block_layer_pre: the PLE history shift failed";
            return false;
        }
    }
    const bool qsa = is_qsa_layer(g, layer);    const LayerView v(tables, layer);    const strata::kernels::GrShapes gs{g.n_embd, g.hc, g.hc_lr};
// ---- the two halves' GR tensors.  Both halves have the same four names with a different prefix, and the
// prefix is the ONLY thing that distinguishes them - so it is built rather than written twice.
const char* pre[2] = {"hc_attn_", "hc_ffn_"};    const WeightRef* w_norm[2];    const WeightRef* w_down[2];    const WeightRef* w_up[2];    const WeightRef* w_inject[2];    for (int h = 0; h < 2; ++h) {        const std::string a = std::string(pre[h]) + "norm.weight";        const std::string d = std::string(pre[h]) + "down.weight";        const std::string u = std::string(pre[h]) + "up.weight";        const std::string i = std::string(pre[h]) + "inject.weight";        w_norm[h] = v.get(a.c_str());        w_down[h] = v.get(d.c_str());        w_up[h] = v.get(u.c_str());        w_inject[h] = v.get(i.c_str());        if (!w_norm[h] || !w_down[h] || !w_up[h] || !w_inject[h]) {            err = v.name((std::string(pre[h]) + "{norm,down,up,inject}.weight").c_str()) + " is missing";            return false;        }
// The GR weights are BF16 and the arena holds them re-rounded to 2 B/elem.  `gr_read` wants exactly
// that; handing it f32 bytes would walk 2x the tensor inside the arena without faulting.
if (w_down[h]->kind != WeightKind::Bf16InF32 || w_up[h]->kind != WeightKind::Bf16InF32 ||            w_inject[h]->kind != WeightKind::Bf16InF32 || w_norm[h]->kind != WeightKind::F32) {            err = v.name(pre[h]) + "has the wrong engine forms (norm must be F32, the other three bf16)";            return false;        }    }
// ---- half 1: the mixer
float* R = bb.R;
    // R0.11: WHICH STAGES TO RUN.  `stage_prefix == 0` means "as `half` says", so every caller that predates
    // this parameter is bit-for-bit unaffected; `stage_prefix == k` runs stages 0..k-1 and ignores `half`.
    const bool run0 = stage_prefix > 0 ? (stage_prefix >= 1) : (half == 0 || half == 1);
    const bool run1 = stage_prefix > 0 ? (stage_prefix >= 2) : (half == 0 || half == 1);
    const bool run2 = stage_prefix > 0 ? (stage_prefix >= 3) : (half == 0 || half == 1);
    const bool run3 = stage_prefix > 0 ? (stage_prefix >= 4) : (half == 0 || half == 2);
    const bool run4 = stage_prefix > 0 ? (stage_prefix >= 5) : (half == 0 || half == 2);        // R0.9: this is the MIXER half, and `half == 2` skips it entirely.  A `half` of 1 or 0 runs it,
    // and 0 is what every caller before this used.
    if (run0) {
st_begin(layer, 0, stream);
    if (fused) {
        strata::kernels::FusedGrArgs fa;
        fa.R = R; fa.R_out = R; fa.apply = pending_ffn; fa.bo_prev = bb.block_out; fa.inj_prev = bb.inject2;
        fa.w_norm = (const float*) w_norm[0]->data; fa.w_down = NativeDense::hc_bf16_view(*w_down[0], stream, 0, 1);
        fa.w_up = NativeDense::hc_bf16_view(*w_up[0], stream, 1, 1); fa.s_down = w_down[0]->q8_scale; fa.s_up = w_up[0]->q8_scale; fa.w_inject = (const uint16_t*) w_inject[0]->data;
        fa.eps = RMS_EPS; fa.lo = bb.gr.lo; fa.rs = bb.gr_rs; fa.inject_out = bb.inject; fa.mixed = bb.mixed;
        strata::kernels::fused_gr_read(fa, stream);
    } else {
    if (w_down[0]->q8_scale || w_up[0]->q8_scale) { err = "STRATA_HC_REQ8 needs the fused hyper-connection read"; return false; }    gr_read(R, (const float*) w_norm[0]->data, NativeDense::hc_bf16_view(*w_down[0], stream, 0, 1),            NativeDense::hc_bf16_view(*w_up[0], stream, 1, 1), (const uint16_t*) w_inject[0]->data, RMS_EPS, gs, bb.gr, bb.mixed,            bb.inject, stream);
    }
    st_end(layer, 0, stream);    dump_half(bb, g, layer, bb.inject, 2 * g.n_embd, g.hc, stream);        }
    if (run1) {
    st_begin(layer, 1, stream);    if (qsa) {        if (!qsa_layer(tables, g, layer, pos, pos_base, qst, qb, bb.mixed, bb.block_out, stream, err,                            bb.dump))            return false;    } else {        if (!gdn_layer(tables, g, layer, gb, bb.mixed, bb.block_out, stream, err)) return false;    }    st_end(layer, 1, stream);    dump_half(bb, g, layer, bb.block_out, 0, g.n_embd, stream);        }
    if (run2) {
    st_begin(layer, 2, stream);    if (!fused) gr_write(R, bb.block_out, bb.inject, gs, R, stream);    st_end(layer, 2, stream);
// ---- half 2, UP TO AND INCLUDING THE ROUTER.  `gr_read` leaves the normed activation in `bb.mixed` and
// the per-stream injection in `bb.inject`, and BOTH must survive until `block_layer_post` runs - which is
// the contract the two functions have with each other and with the host loop's ordering.
    }
    // R0.9: the FFN front and the ROUTER.  It reads `bb.mixed`/`bb.inject`, which the mixer left -
    // the same contract `block_layer_post` has, and the reason the two are only meaningful in order.
    if (run3) {
st_begin(layer, 3, stream);
    if (fused) {
        strata::kernels::FusedGrArgs fa;
        fa.R = R; fa.R_out = R; fa.apply = true; fa.bo_prev = bb.block_out; fa.inj_prev = bb.inject;
        fa.w_norm = (const float*) w_norm[1]->data; fa.w_down = NativeDense::hc_bf16_view(*w_down[1], stream, 0, 1);
        fa.w_up = NativeDense::hc_bf16_view(*w_up[1], stream, 1, 1); fa.s_down = w_down[1]->q8_scale; fa.s_up = w_up[1]->q8_scale; fa.w_inject = (const uint16_t*) w_inject[1]->data;
        fa.eps = RMS_EPS; fa.lo = bb.gr.lo; fa.rs = bb.gr_rs; fa.inject_out = bb.inject2; fa.mixed = bb.mixed;
        strata::kernels::fused_gr_read(fa, stream);
    } else {
    if (w_down[1]->q8_scale || w_up[1]->q8_scale) { err = "STRATA_HC_REQ8 needs the fused hyper-connection read"; return false; }    gr_read(R, (const float*) w_norm[1]->data, NativeDense::hc_bf16_view(*w_down[1], stream, 0, 1),            NativeDense::hc_bf16_view(*w_up[1], stream, 1, 1), (const uint16_t*) w_inject[1]->data, RMS_EPS, gs, bb.gr, bb.mixed,            bb.inject, stream);
    }
    st_end(layer, 3, stream);        }
    if (run4) {
    st_begin(layer, 4, stream);    {        const bool ok = moe_route(tables, g, layer, k, mb, bb.mixed, stream, err, db);        st_end(layer, 4, stream);        if (!ok) return false;    }
    // Plan v0.3 P3: after the ring, so the GPU computes the shared expert while the host runs the pool.
    if (g_shared_early && !moe_shared(tables, g, layer, mb, bb.mixed, stream, err)) return false;
    }
    // Every stage this call was asked to run has run.  There is no other way out: the old `return ok;`
    // lived inside the stage-4 block and restructuring that left the function with no return at all.
    return true;
}
bool ple_issue_token(const PleRun& p, std::string& err) {
    if (!p.ready()) { err = "ple_issue_token: the PLE run is not ready"; return false; }
    uint32_t rows[strata::kernels::PLE_N_HEADS];
    strata::kernels::ngram_rows(p.token, p.prev, 1, p.consts, rows);
    if (!p.table->issue(rows)) { err = "ple_issue_token: the previous token's rows were never collected"; return false; }
    return true;
}

bool ple_finish_token(const PleRun& p, void* stream, std::string& err) {
    if (!p.ready()) { err = "ple_finish_token: the PLE run is not ready"; return false; }
    if (!p.table->collect(p.emb_host, err)) { err = "ple_finish_token: " + err; return false; }
    if (cudaMemcpyAsync(p.emb_dev, p.emb_host, (size_t) strata::kernels::NG_N_EMBD * sizeof(float),
                        cudaMemcpyHostToDevice, (cudaStream_t) stream) != cudaSuccess) {
        err = "ple_finish_token: the row upload failed";
        return false;
    }
    return true;
}

bool ple_stage_token(const PleRun& p, void* stream, std::string& err) {
    if (!ple_issue_token(p, err)) return false;
    return ple_finish_token(p, stream, err);
}

uint64_t ple_run_scratch_bytes() {
    // Internal workspace followed by a disjoint normalized export used for the next history row.
    return strata::kernels::ple_block_scratch_bytes() + strata::kernels::NG_HC_DIM * sizeof(float);
}

bool block_layer_post(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t k,                      const MoEBuffers& mb, const BlockBuffers& bb, const float* parts, void* stream,                      std::string& err) {    const strata::kernels::GrShapes gs{g.n_embd, g.hc, g.hc_lr};
// `bb.mixed` and `bb.inject` are what `block_layer_pre` left, and NOTHING between the two calls may touch
// them - that is what makes `post[l]` safe to launch after the host has run the pool.
st_begin(layer, 5, stream);
    if (g_shared_early ? !moe_combine_parts(g, layer, k, mb, parts, bb.block_out, stream, err)
                       : !moe_finish(tables, g, layer, k, mb, bb.mixed, parts, bb.block_out, stream, err)) return false;
    st_end(layer, 5, stream);    dump_half(bb, g, layer, bb.block_out, (uint64_t) g.n_embd, g.n_embd, stream);    dump_half(bb, g, layer, bb.inject, (uint64_t) 2 * g.n_embd + g.hc, g.hc, stream);    st_begin(layer, 6, stream);
    const bool steer = strata::kernels::cvec().covers(layer);   // --control-vector-scaled: after this write
    try {
        if (!(g_fused_gr && strata::kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr))) {
            gr_write(bb.R, bb.block_out, bb.inject, gs, bb.R, stream);
            if (steer) strata::kernels::cvec_apply(bb.R, layer, 1, g.hc * g.n_embd, nullptr, 0, nullptr, 0, false, stream);
        } else if (layer == g.n_layers - 1) {
            gr_write(bb.R, bb.block_out, bb.inject2, gs, bb.R, stream);   // materialise R for the head
            if (steer) strata::kernels::cvec_apply(bb.R, layer, 1, g.hc * g.n_embd, nullptr, 0, nullptr, 0, false, stream);
        } else if (steer) {
            // the write the next layer's fused read would have folded, then the vector
            strata::kernels::cvec_apply(bb.R, layer, 1, g.hc * g.n_embd, bb.block_out, g.n_embd, bb.inject2, g.hc,
                                        true, stream);
        }
    } catch (const std::exception& e) {
        err = std::string("block_layer_post: ") + e.what();
        return false;
    }
    st_end(layer, 6, stream);    return true;}
bool block_layer(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t pos, int32_t pos_base,                 const GdnBuffers& gb, const QsaState& qst, const QsaBuffers& qb, const MoEBuffers& mb,                 int64_t k, const BlockBuffers& bb, const float* parts, void* stream, std::string& err,                 const Doorbell* db, const PleRun* ple) {
// The two halves back to back.  `parts` must be THIS layer's experts and must be ready before the call -
// which holds for `session_token`, where the caller supplies them, and does NOT hold for a host loop that
// has only just seen the router.  That is why the captured path calls the halves separately.
if (!block_layer_pre(tables, g, layer, pos, pos_base, gb, qst, qb, mb, k, bb, stream, err, db, ple))        return false;    return block_layer_post(tables, g, layer, k, mb, bb, parts, stream, err);}}
// namespace strata::core
