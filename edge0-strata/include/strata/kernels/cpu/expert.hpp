// include/strata/kernels/cpu/expert.hpp - P2.S3: the CPU expert path.
//
// This is the Q2_0 expert kernel promoted out of `bench/micro/cpu_s2.cpp`, which is the instrument that
// measured `c` (Memory/LEDGER.md L9/L10: 15.03 ms for one token's full routed set on 6 cores, DRAM-bound at
// 44.14 GB/s) and whose P0.T2 parity passes at 1.461e-06.
//
// IT IS THE SAME CODE, NOT A COPY.  Keeping a second implementation next to a validated one is how a project
// ends up with two answers and no way to say which is right, so `cpu_s2.cpp` is expected to become a thin
// benchmark over these functions.  (Until it is, the two exist side by side and this header says so.)
//
// WHAT MAKES IT FAST, and each of the three was paid for once:
//
//   * `vpdpbusd` (AVX512-VNNI) does four int8 MACs per instruction.  The scalar loop it replaces ran at
//     9.89 GB/s on one core; VNNI is the reason the 6-core figure is 42.55 GB/s.
//   * `vpmultishiftqb` (AVX512-VBMI) unpacks the 2-bit codes eight at a time.  Its ARGUMENT ORDER is
//     (control, data) and was determined empirically - the Intel guide states the opposite, and following the
//     guide produced 43 of 64 wrong codes.
//   * The row accumulator stays a FLOAT VECTOR and is reduced ONCE at the end of the row.  Reducing per
//     64-weight block cost more than the dot products themselves: it made the kernel compute-bound at
//     21 GB/s instead of DRAM-bound at 32.
//
// THE CONTRACT THAT MATTERS, because collapsing it is invisible and cannot match ggml's logits: one Q2_0
// WEIGHT block is QK=64 weights with one fp16 scale, while one ACTIVATION chunk is QKA=32 elements with
// its OWN scale.  So a weight block spans TWO activation chunks with two different scales.  Using one scale
// per weight block is the natural-looking mistake.
#pragma once

#include "strata/core/geometry.hpp"

#include <cstdint>
#include <cstddef>

namespace strata::kernels::cpu {

// ---- geometry, all of it fixed by the artifact (tools/verify_q2_0_geometry.py: 202/202 tensors at QK=64) ----
inline constexpr int H = 2560;      // n_embd
inline constexpr int FF = 640;      // expert intermediate width
inline constexpr int NE = 512;      // routed experts per layer
static_assert(H == strata::core::kCompiledGeometry.n_embd && FF == strata::core::kCompiledGeometry.n_ff,
              "the CPU expert path is compiled for core/geometry.hpp");
inline constexpr int QK = 64;       // QK2_0: weights per fp16 scale
inline constexpr int QKA = 32;      // QK8_1/QK8_0: ACTIVATION elements per scale
inline constexpr int BB = 18;       // bytes per 64-weight Q2_0 block: 2 fp16-scale + 16 codes
inline constexpr int MAXC = H / QKA;             // 80 activation chunks across the widest reduction
inline constexpr int ROW_GU = H * 2 / 8;         // 640 B of codes per gate/up row
inline constexpr int ROW_D = FF * 2 / 8;         // 160 B of codes per down row
inline constexpr int SC_GU = H / QK;             // 40 weight blocks per gate/up row
inline constexpr int SC_D = FF / QK;             // 10 weight blocks per down row
inline constexpr size_t BLOB = 3ull * (H * FF * BB / QK);   // 1,382,400

// ---- the blob's internal layout (one expert) ----
inline constexpr size_t O_GU_CODES = 0;
inline constexpr size_t O_D_CODES = 2ull * FF * ROW_GU;
inline constexpr size_t O_GU_SCALES = O_D_CODES + 1ull * H * ROW_D;
inline constexpr size_t O_D_SCALES = O_GU_SCALES + 2ull * FF * SC_GU * 2;

/// One activation in Strata's planar storage: `QKA` elements per chunk.
/// The default legacy quantizer uses FP32 scales and half-away rounding. It is
/// not the pinned ggml CPU Q8_0 contract; expert_set_oracle_q8_0 selects that
/// experimental contract (FP16-rounded scales and x86 nearest-even codes).
///
/// The historical function name `act_quant_q8_1` does not describe a native
/// ggml block layout. One 64-weight block always spans two 32-element chunks.
///
/// `hx[k] = scale[k] * sum[k]` is the weight-independent correction the Q2_0 identity needs
/// (`sum (c-1) d_w xhat d_x = d_w (d_x*sum(c*xhat) - d_x*sum(xhat))`), precomputed once per layer rather than
/// once per row: 640 rows would otherwise recompute it 640 times.
struct ActQ {
    alignas(64) int8_t q[H];
    float scale[MAXC];
    int32_t sum[MAXC];
    float hx[MAXC];
    int nchunks;
    /// The AVX-2 "bit-plane" image of `q` (src/kernels/cpu/q2_avx2.cpp explains it), filled only by
    /// `act_quant_q8_1_avx2` with STRATA_Q2_BITPLANE=1 and read only by `q2_0_gguf_rows_multi_avx2`.  Per pair p of 64-value blocks:
    /// `qp[128p + 32k + 16h + i] = q[128p + 64h + 4i + k]`, `psum[8p + 4h + m]` = the sum of q[128p + 64h + 16m ..
    /// + 16), `pscale[8p + 2c + e] = scale[4p + c]`.  `bp_pairs` is 0 when the image is absent (n % 128 != 0).
    alignas(32) int8_t qp[H];
    alignas(32) int32_t psum[H / 16];
    alignas(32) float pscale[H / 16];
    int bp_pairs = 0;
};

/// Per-worker scratch.  Owned by the caller and passed in, so the token path performs NO allocations
/// (P2.T10) - the kernel itself allocates nothing.
struct ExpertScratch {
    ActQ a1;
    ActQ a2;
    alignas(64) float ff[FF];
};

/// Runtime CPU feature check.  The kernel uses AVX512-VNNI + AVX512-VBMI + AVX512VL, and code compiled with
/// `/arch:AVX512` can emit AVX-512 anywhere in its translation unit, so a machine without them must be
/// REFUSED rather than silently run.  `s2_expert_scalar` is the fallback and exists for tests.
struct CpuFeatures {
    bool avx512f = false;
    bool avx512bw = false;
    bool avx512vl = false;
    bool avx512_vnni = false;
    bool avx512_vbmi = false;
    bool usable() const { return avx512f && avx512bw && avx512vl && avx512_vnni && avx512_vbmi; }
    /// A one-line description of what is missing, or "ok".
    const char* reason() const;
};
CpuFeatures cpu_features();

/// Exits with a clear message if the CPU cannot run `s2_expert_vnni`.  Called once at startup, so a user on an
/// older CPU learns why at second zero instead of seeing an illegal instruction at token 4000.
void cpu_require_expert_support();

/// Select the pinned x86 Q8_0 activation contract at BOTH expert quantization
/// boundaries and the pinned generic CPU Q2_0 dot reduction in all projections.
/// Default false preserves the legacy quantizer and eight-lane FP32 reduction.
/// Configure before starting worker threads and do not change during a request.
/// GPU expert-cache parity is not established for this mode; the driver must
/// reject combining it with that cache. No allocation occurs in either mode.
void expert_set_oracle_q8_0(bool enabled);

/// `x` -> `a` in the configured contract. The function name is historical;
/// `n` must be a multiple of QKA and at most H. No allocation occurs.
void act_quant_q8_1(const float* x, int n, ActQ& a);

/// One expert: `out(H) = down( swiglu( gate(x), up(x) ) )`, all three projections Q2_0,
/// with the configured activation and dot contracts at both quantization boundaries.
///
/// `blob` is one 1,382,400-byte expert in the pack's layout; `x` is the H-wide input; `out` is H-wide.
void s2_expert_vnni(const uint8_t* blob, const float* x, float* out, ExpertScratch& ws);

/// The same, with the activation ALREADY quantized in the configured contract.
/// Workers may share this read-only activation; each must own its scratch/output.
///
/// This split exists for the pool.  A layer runs TEN experts against ONE activation, so quantizing inside
/// `s2_expert_vnni` would do the same 2560-element conversion ten times per layer per token - 480 redundant
/// conversions per token.  P2.S3's work item is `(blob, x-hat quantized, block sums, weight, out)`, and this
/// is the entry point that matches it.
void s2_expert_vnni_q(const uint8_t* blob, const ActQ& a1, float* out, ExpertScratch& ws);

/// Plan v0.3 P6: ONE expert applied to up to MAXT tokens that were all routed to it (a speculative verify
/// window routes ~1.5 tokens to each distinct expert). The 1.38 MB blob is read and its 2-bit codes unpacked
/// ONCE per block for all tokens; only the VNNI dot and the accumulation run per token. Every token's output
/// is bitwise equal to `s2_expert_vnni_q` on that token (same operation order per token). Legacy activation
/// contract only; with `expert_set_oracle_q8_0(true)` it falls back to one `s2_expert_vnni_q` per token.
inline constexpr int MAXT = 8;
/// Plan v0.3 P4: the expert in row ranges, so several threads can share one expert.  `s2_expert_gu_rows` writes
/// ff[r] = silu(gate_r . x) * (up_r . x) for r in [r0, r1) (of FF); `s2_expert_down_rows` writes out[r] for r in
/// [r0, r1) (of H) from the intermediate's quantized image.  Together with `act_quant_q8_1(ff, FF, a2)` in
/// between they are exactly `s2_expert_vnni_q`, row for row (same `row_dot`), so the result is bitwise the same.
bool expert_oracle_q8_0_enabled();
void s2_expert_gu_rows(const uint8_t* blob, const ActQ& a1, float* ff, int r0, int r1);
void s2_expert_down_rows(const uint8_t* blob, const ActQ& a2, float* out, int r0, int r1);

/// Plan v0.3 P6: the row-range kernels for up to MAXT tokens routed to the same expert (legacy contract only).
/// Each token's rows are bitwise the single-token `s2_expert_gu_rows` / `s2_expert_down_rows`.
void s2_expert_gu_rows_multi(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* ff, int r0,
                             int r1);
void s2_expert_down_rows_multi(const uint8_t* blob, const ActQ* const* a2, int n_tokens, float* const* out, int r0,
                               int r1);

struct ExpertScratchMulti {
    ActQ a2[MAXT];
    alignas(64) float ff[MAXT][FF];
    ExpertScratch single;                 // the oracle-contract fallback
};
void s2_expert_vnni_multi(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* out,
                          ExpertScratchMulti& ws);

/// The scalar transcription of ggml's formula - NOT a second opinion, but the ORACLE the VNNI path is checked
/// against, and the only path available on a CPU without AVX-512.
///
/// `quant_acts` makes the oracle consume the SAME INT8 activation values the VNNI path uses, **at BOTH
/// stages**.  With it OFF the oracle is exact-FP32-activation and the gap between them IS the activation
/// contract; with it ON the two differ only by FP32 evaluation order, so a small gap there proves the kernel
/// itself correct.  That separation is the whole point of P0.T2 and it is why this parameter exists rather
/// than two functions.
///
/// CORRECTION TO THE ORIGINAL: `bench/micro/cpu_s2.cpp`'s comment says "at both stages" but its code
/// quantizes only the intermediate, leaving the gate/up projections on raw f32 `x`.  That is invisible when
/// the caller passes an already-quantized input - which the engine does - and shows up as a ~1.2% gap when it
/// does not, which reads as a kernel bug and is not one.  This version quantizes both.
/// Plan v0.3 P6: rows [r0, r1) of a Q2_0 matrix in the GGUF block layout (18 bytes per 64 weights, `row_bytes`
/// per row, `nblocks` blocks each) against `nt` activations: out[t][r].
void q2_0_gguf_rows_multi(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                          float* const* out, int r0, int r1);
/// The same two for CPUs without AVX-512 (src/kernels/cpu/q2_avx2.cpp, compiled for AVX2 only).
void q2_0_gguf_rows_multi_avx2(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1);
/// STRATA_Q2_BITPLANE=1 (read once): the AVX-2 Q2_0 kernel reads the bit-plane image instead of the legacy one.
bool q2_bitplane_enabled();
/// The pre-bit-plane AVX-2 kernel (the default), kept for A/B and parity.
void q2_0_gguf_rows_multi_avx2_legacy(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                      float* const* out, int r0, int r1);
/// The same rows with AVX-VNNI on or off, not as cpu_avxvnni_ok() says (tests and benches; on only where it holds).
void q2_0_gguf_rows_multi_avx2_v(bool vnni, const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a,
                                 int nt, float* const* out, int r0, int r1);
void act_quant_q8_1_avx2(const float* x, int n, ActQ& a);
/// The same for CPUs with AVX but no AVX2 (src/kernels/cpu/q2_avx1.cpp, the third rung of q2_rows_any /
/// act_quant_any, issue #1699): same arithmetic, 128-bit integer lanes, mul+add instead of FMA (last bits),
/// software fp16 conversion.  Called only behind cpu_avx1_ok() && !cpu_avx2_ok().
void q2_0_gguf_rows_multi_avx1(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1);
void q2_0_gguf_rows_multi_avx1_legacy(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                      float* const* out, int r0, int r1);
void act_quant_q8_1_avx1(const float* x, int n, ActQ& a);

void s2_expert_scalar(const uint8_t* blob, const float* x, float* out, bool quant_acts);

}  // namespace strata::kernels::cpu
