// include/strata/prefill/moe_fused.hpp - #136 (prompt speed): the routed experts of a Q2_0 pack in the prompt path on
// Strata's own int8 tensor-core kernels, opt-in with STRATA_PF_FUSED=1 (NVIDIA sm_80 and newer; sm_75 and HIP: MMQ).
//
// The MMQ path (moe_mmq.hpp) per MoE layer: a host sync to group the (token, k) pairs by expert, a gather of each
// expert into a 16-expert group buffer, then per group the q8_1 activations of every routed slot, gate/up, SwiGLU,
// the q8_1 of H and down - five passes over per-slot buffers.  Here:
//   - the activations are rounded to int8 once per TOKEN (not per slot), with a scale per 32 values;
//   - the grouping (counts, offsets, the rows of each expert, a list of 64-row tiles) runs on the GPU from the
//     router's ids, so the host never waits for the routing;
//   - one launch per batch of up to kMaxBatch experts reads each expert where it already is (a resident cache slot
//     or a slot of the streamed ring: a pointer per expert, no gather) - gate/up with SwiGLU and the int8 rounding
//     of H in its epilogue, then down into the per-slot rows the existing combine reads.
// The 2-bit codes are unpacked in registers and multiplied with mma.sync m16n8k32 (s8 x s8 -> s32); Q2_0's offset
// (w = d * (q - 1)) is folded into the activation blocks' sums.  The numbers differ from MMQ's (another int8
// rounding of the activations and of H) by about as much as MMQ differs from FP32 (tests/cuda/prefill_fused_moe_test).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::fused {

/// Routed rows (token, k pairs of one expert) per tile.
constexpr int kTileRows = 64;
/// Experts per launch: their blob pointers travel in the launch's parameters (1 KB).
constexpr int kMaxBatch = 128;
/// The room of WideBatch::blob (below): a whole layer of 512 experts, which STRATA_PF_BATCH=512 (opt-in) launches at
/// once on the native kernels' NT = 256 instances when all of them are resident (4 KB of kernel parameters: CUDA 12.1+,
/// Volta and newer).  HIP, gfx906 and the Pascal / Volta build (STRATA_EXPERIMENTAL_SM60): kMaxBatch.
#if defined(__HIPCC__) || defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_EXPERIMENTAL_SM60)
constexpr int kBatchRoom = kMaxBatch;
#else
constexpr int kBatchRoom = 512;
#endif

/// This build has the kernels (a CUDA build; HIP and builds without MMQ do not).
bool built();
/// built(), and the current device is sm_80 or newer (mma.sync with s8 operands at m16n8k32).
bool available();
/// available(), unless STRATA_PF_FUSED=0 (read once): the Q2_0 pack's fused experts, on by default since 0.1.36.
bool enabled();
/// STRATA_PF_FUSED=1 given explicitly, and available(): the native IQ packs' fused kernels (opt-in).  On gfx12 (RDNA4)
/// only the native kernels exist: available() / enabled() / requested() stay false there and
/// native_supported() (moe_fused_iq.hpp) reads the same opt-in itself.
bool requested();

/// Bytes of `rows` activation rows of `cols` values (a multiple of 64) in the kernels' int8 form: per 64 values 64
/// codes and {d, c} of each half (c = -(1.5 * 2^23 + the half's code sum), what the epilogue adds back).
size_t act_bytes(int64_t rows, int64_t cols);
/// Bytes of the grouping tables for `n` routed rows over `n_expert` experts.
size_t group_bytes(int64_t n, int n_expert);

/// x [rows][cols] FP32 -> `xa` (act_bytes(rows, cols)).
void quantize_act(const float* x, int64_t rows, int64_t cols, void* xa, void* stream);

/// The routing `ids` [n = tokens * k] -> per expert its rows (counts, offsets, the 64-row tile list, in `scratch`,
/// group_bytes(n, n_expert)), `slot[i]` = the row of pair i (the row of the per-slot outputs the combine reads) and
/// `src[row]` = its token (i / k).  The order of the rows within an expert is not fixed; no output depends on it.
void group(const int32_t* ids, int64_t n, int k, int n_expert, void* scratch, int32_t* slot, int32_t* src,
           void* stream);

/// Experts [e0, e1) of one layer (e1 - e0 <= kMaxBatch); blob[e - e0]: the Strata Q2_0 blob of expert e on the
/// device (16-byte aligned), read only for an expert with rows.
struct Batch {
    int e0 = 0, e1 = 0;
    const uint8_t* blob[kMaxBatch] = {};
};
/// Batch with room for kBatchRoom experts (e1 - e0 <= kBatchRoom): only the opt-in launches of STRATA_PF_BATCH above
/// kMaxBatch on the native NT = 256 or warp-specialized kernels (moe_fused_iq.hpp, native_wide()) take it; every other
/// launch takes Batch.
struct WideBatch {
    int e0 = 0, e1 = 0;
    const uint8_t* blob[kBatchRoom] = {};
};
/// The batch's products: gate/up from `xa` (quantize_act of the layer's input, [tokens][2560]) through the rows
/// `src` of `group`, SwiGLU, H as int8 into `ha` (act_bytes(n, 640)), down into `dm` [n][2560] FP32 at the rows of
/// `group`.  `n`: the layer's routed rows (tokens * k), which bounds the launch grid.
void experts(const Batch& b, int n_expert, int64_t n, const void* scratch, const void* xa, const int32_t* src, void* ha,
             float* dm, void* stream);

/// The latent (edge0) A-stage blob's geometry: the native NVFP4 layout native_expert_layout(40, 40, ell, iff)
/// computes.  `ell` = the E width (gate/up K, down outputs), `iff` = the features (gate/up rows per projection,
/// down K); `gu_row` / `d_row` the row bytes, `up_off` / `down_off` the blob's section offsets, `tail_off` the
/// NVFP4 tail {s_gate, s_up, s_down, 0} (0: no tail, the scales 1).
struct LatentGeom {
    int64_t ell = 0, iff = 0;
    size_t gu_row = 0, d_row = 0, up_off = 0, down_off = 0, tail_off = 0;
};

/// This device runs the latent fused kernels (sm_80 and newer; read once per device).
bool latent_fused_available();
/// Bytes of the grouping tables experts_latent reads (the layout of group_bytes).
size_t latent_group_bytes(int64_t n, int n_expert);
/// Where the H buffer starts inside the latent scratch: after the tables, 256-byte aligned.
size_t latent_h_off(int64_t n, int n_expert);
/// The tables plus the H rows: what the latent fused A stage needs of the grouped kernel's scratch.
size_t latent_scratch_bytes(int64_t n, int n_expert, int64_t iff);
/// The layer's whole grouping (the host's per-expert entry counts and offsets, off with n_expert + 1 entries)
/// into `host` (latent_group_bytes(n, n_expert) bytes): the counts and offsets as they are, the 64-row tile list
/// in expert id order.  `fill` stays zero: the latent entries are already grouped, no placement runs.
void group_latent(const int32_t* cnt, const int32_t* off, int n_expert, void* host, size_t bytes);
/// The latent A stage on the fused batch kernels: gate/up from `xa` (the xl rows' q8_1 blocks, [rows][ell] of
/// 36-byte blocks) through the rows `src` of the tables in `scratch` (src[row] = that entry's xl row), SwiGLU, H
/// as q8_1 into `hbuf` (at latent_h_off inside the scratch, n rows), down into `dm` [n][ell] FP32 at the tables'
/// rows; the blobs are the native NVFP4 layout `g` describes (blob[e - e0] of a WideBatch: the batch's experts in
/// id order, an expert without a blob in the span is skipped).  `n` bounds the launch grid.
void experts_latent(const WideBatch& b, const LatentGeom& g, int n_expert, int64_t n, const void* scratch,
                    const void* xa, const int32_t* src, void* hbuf, float* dm, void* stream);
/// The same walk for the q3k pack (Q3_K gate/up, IQ4_NL down): one kernel, its weight decode a policy.
void experts_latent_q3k(const WideBatch& b, const LatentGeom& g, int n_expert, int64_t n, const void* scratch,
                        const void* xa, const int32_t* src, void* hbuf, float* dm, void* stream);

/// The latent D projection on the fused dense kernel: the layer input's q8_1 rows [M = T, K = N] (`xq`,
/// quantize_q8_1_rows of the mixed rows) against the layer's G*ell Q8_0 weight rows (`w`, the dense pack's D
/// section), the products quantized to the xl rows' own q8_1 blocks in `outq` (row t*G + g) - the FP32 xl round
/// trip of iq_mmvq + quantize_q8_1_rows never runs.
void latent_dense_d(const void* w, const void* xq, int64_t M, int K, int G, int64_t ell, void* outq, void* stream,
                     int fmt = 8);
/// The latent U projection: each entry's ylq row (`xq`, [M = T*K, K = ELL]) against its label's H Q8_0 weight
/// rows (`w`, the dense pack's U section, n_groups groups of H rows).  The entries are bucketed by their label
/// on the host: `goff` [n_groups + 1] the per-group run in `rows` [M] (the entry index list), both in device
/// memory; `nwork` the tile-item count that bounds the launch grid.  FP32 into `out` [M, H].
void latent_dense_u(const void* w, const void* yq, const int32_t* rows, const int32_t* goff, int n_groups,
                    int64_t nwork, int64_t M, int K, int H, float* out, void* stream, int fmt = 8);
/// The U tables on the device: goff[0..n_groups] (the per-group run ends), rows[0..n) (the entries bucketed by
/// their expert's label), cursors (n_groups + 1 scratch) - all from ids[] and labels[] already resident.  The
/// caller bounds the launch grid without the counts: ceil(n / 64) + n_groups tiles cover every split.
void latent_dense_u_tables(const int32_t* ids, const int32_t* labels, int64_t n, int n_groups, int32_t* goff,
                           int32_t* cursors, int32_t* rows, void* stream);

}  // namespace strata::prefill::fused
