// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_gguf_cpu.hpp - a GGUF's MXFP4 experts on the CPU lane of maya-ds4 (the format: ds4_gguf.cuh).  The
// weights are read where they lie (the GGUF file's mapping), each block decoded once for all the activations of a
// call (mx_rows: int8 dots, the CPU's best vector tier at run time; a scalar loop otherwise).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::ds4::gguf {

constexpr int kMxCpuMaxN = 8;   // activations a call

/// y[i][r - r0] = (row r of W) . x[i] for r0 <= r < r1 and i < n (n <= kMxCpuMaxN): W's rows hold k values
/// (k % 32 == 0) as MXFP4 blocks, rb bytes apart.  (F32 activations: the tests' and the benchmark's old path.)
void mx_cpu_rows(const uint8_t* w, size_t rb, int k, int r0, int r1, const float* const* x, int n, float* const* y);

/// A GGUF's low-bit experts (ds4_iq.cuh's types) on the CPU lane - ggml-cpu's arithmetic through the GLM CPU lane's
/// native experts (strata::dsv41k::cpu::native_*: its AVX-512 / AVX2 multi-token rows where they are ggml's bit for bit,
/// ggml's dots otherwise; ds4_iq_cpu.cpp): an activation quantized once to the weight type's vec_dot_type (Q8_K: an F32
/// scale per 256 values - the device takes one per 32, so the two sides agree to that rounding), each row one dot.
struct IqCpu {
    int tg = 0, td = 0, n_embd = 0, n_ff = 0;
    int gu_act = 0, d_act = 0;                 // the activations' ggml types
    size_t gu_row = 0, d_row = 0;              // bytes a weight row
    size_t act_bytes = 0, h_bytes = 0;         // bytes of x (n_embd) and of h (n_ff) as those
};
/// The layer's geometry; false (err) when this build or ggml-cpu cannot take it.
bool iq_cpu_fmt(int tg, int td, int n_embd, int n_ff, IqCpu& f, std::string& err);
/// x (k values) as the activation type `act` (f.gu_act / f.d_act) into dst.
void iq_cpu_quant(int act, const float* x, int k, void* dst);
/// ff[t][r] = clamped swiglu(gate_r . a[t], up_r . a[t]) for rows [r0, r1) (limit 0: no clamp), nt tokens.
void iq_cpu_gate_up(const IqCpu& f, const uint8_t* gate, const uint8_t* up, const void* const* act, int nt,
                    float* const* ff, int r0, int r1, float limit);
/// out[t][r] = down_r . hq[t] for rows [r0, r1).
void iq_cpu_down(const IqCpu& f, const uint8_t* down, const void* const* hq, int nt, float* const* out, int r0, int r1);
/// The trellis types (ds4_iq.cuh's is_tq; ds4_tq_cpu.cpp, Strata's tq_cpu kernels): the same four calls, reached from the
/// iq_cpu_* ones above for those types (their activations: F32 scale a 32 values, then the int8 values).
bool tq_cpu_type(int type);
int tq_cpu_act();
bool tq_cpu_fmt(int tg, int td, int n_embd, int n_ff, IqCpu& f, std::string& err);
void tq_cpu_quant(const float* x, int k, void* dst);
void tq_cpu_gate_up(const IqCpu& f, const uint8_t* gate, const uint8_t* up, const void* const* act, int nt, float* const* ff,
                    int r0, int r1, float limit);
void tq_cpu_down(const IqCpu& f, const uint8_t* down, const void* const* hq, int nt, float* const* out, int r0, int r1);

/// ggml block types on the CPU lane through ggml-cpu's row dots (ds4_gg_cpu.cpp), for the IQ / K-quant experts.
bool gg_cpu_type(int type);
/// One row of a GGUF block type (Q4_K / Q5_K / Q6_K / Q8_0 / any type ggml has a to_float for) -> k values as BF16
/// (round to nearest even), the host embedding's format.  False: ggml has no decode for this type.
bool gg_row_to_bf16(int type, const uint8_t* row, int k, uint16_t* out);
bool gg_cpu_fmt(int tg, int td, int n_embd, int n_ff, IqCpu& f, std::string& err);
void gg_cpu_quant(int act, const float* x, int k, void* dst);
void gg_cpu_gate_up(const IqCpu& f, const uint8_t* gate, const uint8_t* up, const void* const* act, int nt, float* const* ff,
                    int r0, int r1, float limit);
/// -1: from the environment (default: Strata's AVX2 i-quant rows, bit-identical to ggml-cpu's dot, from 2 tokens - 1 on IQ3_S, 3 on IQ1_M;
/// DS4_GG_IQ256=0 never, =1 always), 0 ggml-cpu, 1 always (benches), 2 the default rule.
void gg_cpu_set_iq256(int mode);
void gg_cpu_down(const IqCpu& f, const uint8_t* down, const void* const* hq, int nt, float* const* out, int r0, int r1);

/// The CPU lane's MXFP4 rows in the device's arithmetic (ds4_gguf.cu's mx_dot): the activations rounded to int8 per 32
/// values as the device rounds them (q8_round's q and scale), each block's products summed exactly in integers, then
/// times (weight scale x activation scale) into F32 sums - the same numbers as the device up to the order of the
/// blocks' sums.  The vector tiers (ds4_mx_kern.inc): AVX2 vpmaddubsw, AVX-VNNI and AVX-512 (VNNI) vpdpbusd; the CPU's
/// fastest is taken at run time (DS4_MX_TIER=<name> picks another one it runs).  Every tier, the scalar one included,
/// sums in the same order: they give the same bits, so the choice never changes an answer.
enum MxTier { kMxScalar = 0, kMxAvx2, kMxAvxVnni, kMxAvx512, kMxAvx512Vnni, kMxTiers };
/// Bytes of an activation of k values (k % 32 == 0) for mx_rows: the int8 values in quads of blocks, then a scale and
/// an offset correction a block in groups of 8 (ds4_mx_kern.inc) - 1.25 k for k % 256 == 0.
inline size_t mx_act_bytes(int k) {
    const size_t nb = (size_t) k / 32;
    return 128 * ((nb + 3) / 4) + 64 * ((nb + 7) / 8);
}
/// (tests) value j (0..31) of block b of an activation, and the block's scale
int mx_act_value(const void* act, int b, int j);
float mx_act_scale(const void* act, int k, int b);
/// Blocks [b0, b1) of a k-value activation (x at block b0's first value: x[32 (b - b0) + j]) into act.
void mx_quant_blocks(const float* x, int b0, int b1, void* act, int k);
void mx_quant_blocks_scalar(const float* x, int b0, int b1, void* act, int k);   // (the same bytes, no vectors)
inline void mx_quant(const float* x, int k, void* act) { mx_quant_blocks(x, 0, k / 32, act, k); }
/// y[i][r - r0] = (row r of W) . act[i] for r0 <= r < r1, i < n (any n; act[i] mx_quant's of k values).
void mx_rows(const uint8_t* w, size_t rb, int k, int r0, int r1, const void* const* act, int n, float* const* y);
/// The same with tier t (the scalar one when the CPU does not run t, or k % 256 != 0); pf: bytes prefetched ahead.
void mx_rows_tier(int t, const uint8_t* w, size_t rb, int k, int r0, int r1, const void* const* act, int n,
                  float* const* y, int pf = 0);
int mx_tier();                   // the tier mx_rows takes (measured on first use: the fastest a core, ds4_gguf_cpu.cpp)
const char* mx_tier_report();    // the measurement behind it (each candidate's time), or how it was picked
bool mx_tier_ok(int t);          // this CPU runs tier t
const char* mx_tier_name(int t);
/// Whether the rows of `type` are ds4's own here (IQ3_XXS / IQ3_S on AVX2 / AVX-512 VBMI: ggml-cpu's numbers bit for
/// bit, checked against its dot on first use; DS4_IQ_OWN=0 turns them off) rather than ggml's / the GLM lane's.
bool iq_cpu_own_rows(int type);

/// x rounded as the device's MXFP4 kernels round their activations (q8_quant: per 32 values sc = max|x| / 127, q =
/// round-half-even(x / sc), the same F32 operations, so the same q), into out as q sc - an expert gives the same sums
/// on the CPU lane as in VRAM (up to their order), whichever side the expert cache puts it on.
void q8_round(const float* x, int k, float* out);

}  // namespace strata::ds4::gguf
