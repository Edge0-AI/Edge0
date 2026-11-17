// include/strata/prefill/kernels.hpp - plan v0.3 P5: the element-wise and stateful kernels of a prompt chunk.
//
// T tokens at a time, all row-major with the token as the slow index.  The arithmetic mirrors the per-token native
// kernels (the decode path), batched; where a step is a recurrence in time (the GDN conv and state, the PLE conv)
// the kernel walks the chunk's tokens in order inside one launch.
#pragma once

#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <cstdint>

namespace strata::prefill {

// ---- hyper-connection (n_embd 2560, hc 4, hc_lr 320)
/// xn[t, c*2560 + d] = R[t,c,d] * rsqrt(mean_d R[t,c,:]^2 + eps) * w_norm[c*2560 + d]; also its BF16 image.
/// `xn16_lo` (null: none) takes bf16(x - xn16): W.xn16 + W.xn16_lo is the product with ~16 mantissa bits of x.
void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream,
             uint16_t* xn16_lo = nullptr, bool f16img = false);
/// F-1: gr_norm without its FP32 output: the row scales rs[t*4 + c] and the BF16 image; gr_mix_r then reads R.
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream,
                uint16_t* xn16_lo = nullptr, int64_t ldx = 0, bool f16img = false);   // ldx: xn16's token stride (0 = 10240)
/// gr_mix with xn recomputed from R, rs and w_norm exactly as gr_norm computes it (the same bits).
void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
              int64_t T, void* stream, uint16_t* mixed_h = nullptr, uint16_t* mixed16_lo = nullptr);
/// F-2: gr_write, then gr_norm_rs of the next half (its norm weights) over the rows just written - the same bits as
/// the two calls, without reading R back.
/// S23 (opt-in STRATA_HC_UPMIX=1): the up projection (lo16 x w_up^T, BF16, FP32 accumulate) with gr_mix_r as its
/// epilogue - `gated` is never written.  gfx11 and CUDA sm_80+; false (nothing launched) elsewhere.
bool gr_upmix(const uint16_t* lo16, const uint16_t* w_up, const float* R, const float* rs, const float* w_norm,
              float* mixed, uint16_t* mixed16, uint16_t* mixed_h, int64_t T, void* stream);
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo = nullptr,
                      int64_t ldx = 0, bool f16img = false);
/// S23 (STRATA_CVEC_FUSE=1): gr_write, then the control vector (v_l = layer l's direction row, *s_l its scale,
/// *on the request flag, mode 0 project / 1 add; cvec_kernel's arithmetic), then gr_norm_rs with the next half's
/// norm - one pass over R, the same bits as the three kernels.
void gr_write_cvec_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* v_l,
                           const float* s_l, const int* on, int mode, const float* w_norm_next, float eps, float* rs,
                           uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo = nullptr, int64_t ldx = 0, bool f16img = false);
/// lo16[t, k] = bf16(silu(lo[t, k] / hc))
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo = nullptr);
/// The BF16-weight GEMMs' activation image (gr_* kernels, to_bf16) in FP16 instead of BF16, on the current device.
void set_act_f16(bool on);
/// STRATA_F16_RANGE: count what the activation images convert to FP16 on this device (hf_sat): the largest |value| and
/// how many were beyond 65504 or NaN.  f16_range_read returns them (and resets when asked); false if the device
/// refused the copy.
void set_f16_range(bool on);
bool f16_range_read(float& max_abs, unsigned long long& over, bool reset);
/// mixed[t, d] = mean_c xn[t, c, d] * sigmoid(gated[t, c, d]); FP32, BF16 and FP16 (either image may be null).
void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h = nullptr, uint16_t* mixed16_lo = nullptr);
/// R[t, c, d] += bo[t, d] * 2 sigmoid(inj[t, c] / hc)   (inj has row stride inj_ld)
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream);
/// R[t, c, :] = e[t, :] for all four streams (the embedding broadcast).
void gr_broadcast(const float* e, float* R, int64_t T, void* stream);

// ---- GDN (state 128, 16 k heads, 48 v heads, 10240 conv channels, 4 taps)
/// gate[t,h] = softplus(ab[t,h] + dt[h]) * ssm_a[h];  beta[t,h] = sigmoid(ab[t, 48 + h])  (ab: [T, 96])
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream);
/// The 4-tap causal conv + SiLU over the chunk (history [C][3] in, updated to the chunk's last three inputs), then
/// the L2 norm of the q and k heads of every token.  h: [T, C].
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream);
/// The recurrence over the chunk, state in registers; y16[t] = rmsnorm(o) * gamma * sigmoid(z) in FP16 (what the
/// out projection reads); y is FP32 scratch.
void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream,
                    int64_t ld16 = 0);   // ld16: y16's row stride (0 = 6144; S23 STRATA_PF_PAD pads it)
/// The kernels behind gdn_recurrence, for the parity test: 0 = the column-split kernels + the norm kernel (or the
/// one-block-per-head kernel under STRATA_GDN_REC_HEADS), 1 = four lanes per column (no barrier per token) + the grid-stride norm, 2 = the
/// one-block-per-head kernel (its fused norm rounds differently).  0 and 1 give the same bits.  On HIP, 0 and 1 run the opt-in
/// chunked WMMA recurrence instead (gfx11, STRATA_GDN_CHUNKED=1, 64 tokens and more: other bits, see docs/AMD_HIP.md).
/// SYCL only: 6 = the Xe sub-group recurrence (16-lane sub-groups, one barrier per token) + the norm kernel, the same
/// bits as 0 on an Arc A770; gdn_recurrence takes it by default on Alchemist (STRATA_GDN_XE=0 off, =1 any Intel GPU).
void gdn_recurrence_variant(int variant, float* state, const float* h, const float* gate, const float* beta,
                            const float* z, const float* gamma, float eps, float* y, uint16_t* y16, int64_t T,
                            void* stream, int64_t ld16 = 0);
/// HIP gfx12 only, STRATA_GDN_CHUNKED=1 (STRATA_GDN_FUSE=0 off): gdn_conv + gdn_recurrence on the chunked matrix-core path with
/// the conv output kept as FP16 in `hbuf` (the FP32 h buffer of gdn_conv, reused).  Returns false, having launched nothing, when
/// it does not apply (not gfx12, short prompt, unaligned stride, off): the caller then calls the two.
bool gdn_conv_recurrence_fused(float* history, const float* qkv, const float* conv_w, float* hbuf, float* state, const float* gate,
                               const float* beta, const float* z, const float* gamma, float eps, float l2eps, float* y,
                               uint16_t* y16, int64_t T, void* stream, int64_t ld16 = 0);
/// KVA (CUDA/HIP only): the same recurrence kernels as gdn_recurrence, bit for bit, without the output
/// normalization: state, h, gate, beta and the scratch `y` only.  The output projection and the gated residual
/// write have no consumer when only the state advances, and their input (z, the gate projection) is not needed
/// either.  Declared here for the CUDA build; the SYCL tree keeps its own recurrence and does not reference it.
void gdn_recurrence_state_only(float* state, const float* h, const float* gate, const float* beta, float* y,
                               int64_t T, void* stream);

// ---- MoE
/// softmax over 512, top-10 (ties to the lower id), weights renormalised over the ten (the native router).
void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream);
/// Expert blob (Strata pack layout, Q2_0) -> BF16 matrices: gate/up interleaved [1280, 2560], down [2560, 640].
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream);
/// h16[n, r] = fp16(silu(gu[n, 2r]) * gu[n, 2r + 1])   (the interleaved expert gate/up)
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream);
/// the Volta prompt experts: SwiGLU of gate/up rows (gate first, or interleaved) to FP16, saturated like swiglu_interleaved
void swiglu_split_f16(const float* gu, uint16_t* h16, int64_t rows, int n_ff, bool interleaved, void* stream);
/// h16[n, r] = fp16(silu(g[n, r]) * u[n, r])   (the shared expert, gate and up separate, width 640)
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream);
/// dst[i] = src[i] for n int32s, as a kernel: either side may be mapped host memory, and the copy never waits
/// behind the copy engine's queue (the prompt path's grouping tables, while the expert stream fills it).
void copy_i32(int32_t* dst, const int32_t* src, int64_t n, void* stream);
/// dst[0, n) = src[0, n) in 16-byte loads (both 16-byte aligned; src may be mapped host memory: read over PCIe by
/// the kernel, not queued on a copy engine).
void copy_f32_wide(float* dst, const float* src, int64_t n, void* stream);
/// Gather rows: dst16[i, :] = x16[src[i], :] (n rows of `width` BF16).
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream);
/// bo[t, :] = shared[t, :] * sigmoid(sg[t]) + sum_k w[t, k] * D[slot[t, k], :]
void moe_combine(const float* D, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream, const float* row_sd = nullptr);
/// Layer-major prefill (STRATA_PREFILL_LAYER_MAJOR=1): acc[t, :] = shared[t, :] * sigmoid(sg[t]) - the shared expert's
/// part of moe_combine, written before the routed experts are known.
void moe_shared_init(const float* shared, const float* sg, float* acc, int64_t T, void* stream);
/// Layer-major prefill: one expert group's rows added into the super-chunk's accumulator, per token in a fixed order
/// (deterministic, no atomics): for each u < n_u, acc[u_tok[u], :] += sum_{j in [u_off[u], u_off[u+1])} u_w[j] *
/// Dm[u_rows[j], :].  The sum runs in the group's row order, not moe_combine's k order (FP32 rounding only).
void moe_scatter_add(const float* Dm, const int32_t* u_tok, const int32_t* u_off, const int32_t* u_rows,
                     const float* u_w, int64_t n_u, float* acc, void* stream);
/// --peer-device's prompt share as sums: bo[t, :] = (sum over the k with slot[t, k] < rows_local of w[t, k] *
/// D[slot[t, k], :], in k order) + peer[t, :] + shared[t, :] * sigmoid(sg[t]).  `peer` (the peer's per-token sums)
/// may be mapped host memory.
void moe_combine_peer(const float* D, const int32_t* slot, const float* w, const float* shared, const float* sg,
                      const float* peer, int64_t rows_local, float* bo, int64_t T, void* stream);
/// sum[t, :] += wk[p] * rows[r, :] with p = pair[r], t = p / 10 (the routed pair's token), for n rows of ONE expert
/// (no token twice: launches in a fixed expert order make the sums repeatable).  After eddoursul/Strata's
/// moe_scatter_add (f8de703).
void peer_scatter_add(float* sum, const float* rows, const float* wk, const int32_t* pair, int64_t n, void* stream);
/// The same for a group of experts in one launch (a token may recur): token tok[b] adds its rows list[start[b] ..
/// start[b + 1]) in that order (the experts' order), row r at rows[r - r0]: bitwise the per-expert launches'.
/// eddoursul/Strata's moe_gather_add, with the weight looked up through the row's pair.
void peer_gather_add(float* sum, const float* rows, int64_t r0, const float* wk, const int32_t* pair, const int32_t* tok,
                     const int32_t* start, const int32_t* list, int64_t n_tok, void* stream);
/// moe_combine_peer with the peer's sums in FP16 (`peer16`, may be mapped host memory)
void moe_combine_peer16(const float* D, const int32_t* slot, const float* w, const float* shared, const float* sg,
                        const uint16_t* peer16, int64_t rows_local, float* bo, int64_t T, void* stream);
/// y = FP16 bits of x, saturated at +-65504 (the peer's sums on their way back; eddoursul/Strata 2acfac4)
void sums_to_f16(const float* x, uint16_t* y, int64_t n, void* stream);
/// dst[0, n) = FP32 of the FP16 src[0, n) (n a multiple of 8, 16-byte aligned; src may be mapped host memory)
void f16_to_f32_wide(float* dst, const uint16_t* src, int64_t n, void* stream);

// ---- QSA helpers
/// In place: x[r, :] = x[r, :] * rsqrt(mean x^2 + eps) * w  over rows of `cols` (row stride `ld`).
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream);
/// NEOX rotary on the first 64 dims of each head: x [T, heads, dim] at positions pos0 + t.  The rope
/// scaling (rope_scaling.hpp) rides in as the process config - none is today's arithmetic exactly.
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
          const strata::kernels::RopeScaling& scaling, void* stream);
/// q_full [T, 24, 512] (q | gate per head) -> q [T, 24, 256]
void split_q(const float* q_full, float* q, int64_t T, void* stream);
/// attn[t, h, d] *= sigmoid(q_full[t, h, 256 + d]) -> out16 (fp16 bits: the o-projection is quantized)
void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream, int64_t ld16 = 0);

/// K and V of T consecutive cells (positions pos0..pos0+T-1; K normed and rotated) into the paged pools: FP16
/// (`k_pool`/`v_pool`) or INT8 codes + FP16 scale per 64 (`k_q`...), the decode append's arithmetic.
/// K, V: [T, 2, 256].
void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, const strata::kernels::KvHostPools* host = nullptr,
               const strata::kernels::KvHostPools* stage = nullptr);

/// fp32 -> fp16 bits and fp32 -> bf16, n elements (the two activation images of the prompt GEMMs).
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream);
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo = nullptr);
/// y = fp32(fp16(x)): what an FP16 store of x would read back (the FP16 indexer-key experiment)
void round_f16(const float* x, float* y, int64_t n, void* stream);
/// Expert blob -> FP16 (Q2_0 values are exact in FP16).
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream);

}  // namespace strata::prefill
