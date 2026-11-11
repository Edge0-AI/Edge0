// include/strata/kernels/gr.hpp - the gated residual (hyper-connection), P2.S2.
//
// `ref/gr.py::gr_read` / `gr_write`, which are transcribed from
// `llama_model_qwen4exp::graph::build_hc_mix` / `build_hc_combine` (qwen4exp.cpp L267-350 at 3cf03257).
//
// The residual is a STACK of `hc` streams, each `n_embd` wide - not one vector with a gate.  `gr_read`
// collapses the stack to the one vector the mixer consumes and produces a per-stream injection; `gr_write`
// puts the mixer's output back into every stream, weighted per stream.
//
// WEIGHTS ARE BF16, stored as the high 16 bits of an f32 in a `uint16_t`.  That is not a shortcut: every one
// of these tensors has source type BF16 in the pack (288 of the shard's 483 BF16 tensors are `hc_*`), so the
// pack's 32-bit promotion is lossless padding and re-rounding to 16 bits costs nothing.  It is also worth
// 1.374 GiB of the 5.305 GiB dense file, which is the difference between the dense weights fitting in VRAM
// beside the expert cache and not fitting at all.
//
// The default activation contract is the historical BF16-rounded reference.  The opt-in FP32 contract
// matches the pinned llama.cpp CUDA single-token BF16 MMVF input precision: BF16 weights multiply FP32
// activations.  The CPU oracle and some batched CUDA paths instead round activations to BF16.  Therefore
// activation precision must be pinned with the oracle backend and batch geometry, not inferred from weights.
#pragma once

#include <cstdint>
#include <cstddef>

namespace strata::kernels {

/// Select the activation precision for subsequent gr_read launches (default: false/BF16).
/// Configure before creating/capturing a session, and do not mutate concurrently with gr_read.
/// A captured graph retains the kernel variants chosen at capture; changing this flag cannot alter it.
/// This changes activation precision only; the existing FP32 reduction trees remain unchanged.
void gr_set_fp32_activations(bool enabled);

/// Select pinned single-token BF16/F32 MMVF projections for subsequent gr_read launches (default false).
/// This implies FP32 activations regardless of gr_set_fp32_activations. Configure before session capture;
/// a captured graph retains its selected kernels. Requires even hc*n_embd and hc_lr, as MMVF reads pairs.
/// This also selects pinned weighted F32 RMSNorm and nonlinear/combine arithmetic.
/// Layer reads use fused HC pre; the final mixer (null w_inject) retains its unfused graph order.
/// Subsequent gr_write calls select pinned sigmoid/scatter arithmetic under the same setting.
void gr_set_native_mmvf(bool enabled);
/// Whether gr_write currently uses the native write (native_gr_post).
bool gr_native_mmvf_enabled();

/// Geometry of the hyper-connection.  The real model: n_embd = 2560, hc = 4, hc_lr = 320.
struct GrShapes {
    int64_t n_embd = 0;   ///< width of one residual stream
    int64_t hc = 0;       ///< number of streams in the stack
    int64_t hc_lr = 0;    ///< rank of the down/up bottleneck
};

/// Scratch for `gr_read`.  EXPLICIT and caller-owned, because P2.T10 requires ZERO token-path allocations:
/// a hidden `cudaMalloc` on first use would satisfy every test here and fail the memory test in P2.S10.
///
/// **IT MUST BE DEVICE MEMORY.**  `gr_workspace_init` takes a raw pointer and does no checking, so handing it
/// a host buffer produces device kernels writing to a truncated host address - an illegal access whose
/// reported fault address is the low 32 bits of a host pointer, which reads as a wild pointer rather than as
/// the mistake it is.  That cost one round in `gr_parity`;
///
///   xn    hc*n_embd floats   the unrounded activated streams (the gated mean needs the unrounded value)
///   xq    hc*n_embd uint16   bf16(xn), used by the default activation contract
///   lq    hc_lr     uint16   bf16(lo), used by the default activation contract
///   gated hc*n_embd floats   xn * sigmoid(gate), before the mean over streams
///   lo    hc_lr     floats   SiLU bottleneck, used by the FP32 activation contract
///
/// `cudaMalloc(gr_workspace_bytes(s))`, then `gr_workspace_init(s, ptr, ws)` to carve it.
struct GrWorkspace {
    float* xn = nullptr;
    uint16_t* xq = nullptr;
    uint16_t* lq = nullptr;
    float* gated = nullptr;
    float* lo = nullptr;
    /// Set by `gr_workspace_init`.  `gr_read` CHECKS it, so an under-sized buffer is a loud failure at the
    /// first call instead of silent corruption - which is what a mis-ordered region table used to produce.
    size_t bytes = 0;
};

/// Carves `base` (which must be DEVICE memory, or null to size only) into `out` and returns the bytes needed.
/// 16-byte aligned between regions.
size_t gr_workspace_init(const GrShapes& s, void* base, GrWorkspace& out);
inline size_t gr_workspace_bytes(const GrShapes& s) { GrWorkspace w; return gr_workspace_init(s, nullptr, w); }

/// `build_hc_mix`.  ONE token: `R` is (hc, n_embd) with n_embd fastest, i.e. stream c starts at c*n_embd.
///
///   w_norm   (hc*n_embd)          per-stream RMSNorm gamma, STORED AS (1 + w) by the converter
///   w_down   (hc_lr, hc*n_embd)   MANIFEST LAYOUT - row k is contiguous, `w_down[k*hc*n_embd + i]`
///   w_up     (hc*n_embd, hc_lr)   MANIFEST LAYOUT - row i is contiguous, `w_up[i*hc_lr + k]`
///   w_inject (hc, hc*n_embd)      null for the FINAL mixer, which has no write-back to gate
///
///   mixed    (n_embd)   = mean over streams of  xn * sigmoid(bf16(lo) @ w_up.T)
///   inject   (hc)       = bf16(xn) @ w_inject.T, or untouched when w_inject is null
///
/// `lo = silu((bf16(xn) @ w_down.T) / hc)` - the division by hc is INSIDE the silu, which is easy to read
/// past and changes the result materially.
/// With gr_set_fp32_activations(true), replace bf16(xn) and bf16(lo) by xn and lo respectively.
///
/// **NO PERMUTATION: both matrices are taken exactly as the manifest stores them**, so the loader does nothing
/// to them.  Both are coalesced because each projection assigns ONE WARP PER OUTPUT ROW and strides the
/// reduction axis across its 32 lanes: for a fixed output, consecutive lanes touch consecutive addresses in
/// either orientation.  An earlier version assigned one THREAD per row, which is coalesced with itself and 32
/// transactions away from its warp neighbours - see the history below.
///
/// **HISTORY, because this kernel was the engine's worst and the reason is instructive.**  The first version
/// was launched `<<<1, 256>>>` with `xn`/`xq` in shared memory, and at the real geometry it cost **2.67 ms per
/// call, 96 calls per token = 262 ms/token, about 3.8 tok/s, 134x off the 643 GB/s memory floor.**  The cause
/// was isolated by elimination: NOT memory latency (four independent accumulator chains made it 17% SLOWER),
/// NOT the SM clock (`nvidia-smi` reported 2917 MHz and 99% utilisation throughout), and only partly
/// coalescing (transposing one of the two matrices gave 31% and transposing the other gave nothing).  It was
/// ISSUE THROUGHPUT - ~8.4M iterations of 6-8 instructions issued by ONE of 48 SMs.
///
/// This version splits the work the way the shapes want: `hc` blocks for the per-stream norm, one warp per
/// row for each of the two projections, and one block-tile for the mean.  The transposes are gone because the
/// warp-per-row mapping makes them unnecessary.
void gr_read(const float* R, const float* w_norm, const uint16_t* w_down, const uint16_t* w_up,
             const uint16_t* w_inject, float eps, const GrShapes& s, const GrWorkspace& ws, float* mixed,
             float* inject, void* stream);

void gr_read_multi(const float* R, const float* w_norm, const uint16_t* w_down, const uint16_t* w_up,
                   const uint16_t* w_inject, float eps, const GrShapes& s, const GrWorkspace& ws,
                   float* xn_multi, float* lo_multi, float* gated_multi, float* mixed, float* inject,
                   int n_tok, void* stream);

/// `build_hc_combine`.  `R_out[i] = R[i] + block_out[d] * w[c]` with `w[c] = 2*sigmoid(inject[c]/hc)`.
///
/// The `2*sigmoid` is what CENTRES the gate on 1, so a zero injection is a plain residual add - the source
/// comment says so outright and `gr_write`'s test asserts it.  Note the block output is added to EVERY stream
/// identically; only the weight is per-stream.
void gr_write(const float* R, const float* block_out, const float* inject, const GrShapes& s, float* R_out,
              void* stream);

void gr_write_multi(const float* R, const float* block_out, const float* inject, const GrShapes& s, float* R_out,
                    int n_tok, void* stream);

}  // namespace strata::kernels
