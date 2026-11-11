// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_iq.cuh - a GGUF's routed experts in ggml's low-bit block types on the device for maya-ds4: the i-quants
// IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS and the K-quants Q2_K, Q3_K, Q4_K (256 values a block, ggml's layouts:
// ggml-common.h, llama.cpp MIT).  ds4_gguf.cu sends an expert layer here when its experts are not MXFP4
// (x3::ExpertLayout::fmt kFmtMXFP4 with K[r] = role r's ggml type; K[r] == kTypeMXFP4: ds4_gguf.cu's own kernels).
//
// Every weight is decoded exactly (ggml's dequantize_row_*: a block scale times an integer scale times a grid or code
// value) and every product takes the activations rounded to int8 per 32 values with an F32 scale (ds4_gguf.cu's
// q8_quant): a 32-value sub-block's sum is an exact integer dot (dp4a) times its scales in F32 - no integer division as
// llama.cpp's vecdotq takes (its sums truncate), so these are the double reference's sums to F32 rounding.  A window's
// entry gets the token path's bits (the same kernel, the same order).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace strata::ds4::gguf::iq {

// ggml type ids
constexpr int kQ5_K = 13, kQ6_K = 14;   // (the dense matrices' K-quants, GEMV only)
constexpr int kQ2_K = 10, kQ3_K = 11, kQ4_K = 12, kIQ2_XXS = 16, kIQ2_XS = 17, kIQ3_XXS = 18, kIQ1_S = 19,
              kIQ4_NL = 20, kIQ3_S = 21, kIQ2_S = 22, kIQ4_XS = 23, kIQ1_M = 29;

/// The trellis types (Strata's, ds4_tq.cu): ggml's TQK6 / TQK7 (145 / 146, the table codebook) and the experimental
/// TQK5X / TQK4X (224 / 225, computed codebook) and TQK5T / TQK4T (226 / 227, table): a block of 128 values is
/// {fp16 d; 32 steps of K bits}, 2 + 4 K bytes (tools/trellis/tqenc.hpp).  They go through the same entry points below.
constexpr int kTQK6 = 145, kTQK7 = 146, kTQK5X = 224, kTQK4X = 225, kTQK5T = 226, kTQK4T = 227;
constexpr int tq_kbits(int type) {
    return type == kTQK6 ? 6 : type == kTQK7 ? 7 : (type == kTQK5X || type == kTQK5T) ? 5 : (type == kTQK4X || type == kTQK4T) ? 4 : 0;
}
constexpr bool is_tq(int type) { return tq_kbits(type) > 0; }

/// Whether the device kernels take this ggml type.
bool supported(int type);
/// Whether the type's block layout is known (its row bytes can be computed).  It covered more than supported() while
/// IQ1_S, IQ1_M and IQ4_NL had no device kernel; they have one now, so both take the same types.
bool known(int type);
/// Values a block of the type holds (256; the trellis types 128; 0: not one of these).
int block_values(int type);
/// Bytes of a block of the type (0: not one of these).
int block_bytes(int type);
/// Bytes of a row of k values (k % block_values == 0).
inline size_t row_bytes(int type, int64_t k) { return (size_t) (k / block_values(type)) * (size_t) block_bytes(type); }
/// The type's name ("?" when unknown).
const char* name(int type);

/// The token / window experts (ds4_gguf.cu's MxArgs, with the roles' types): group g = an expert (grp_ptr null: expert g
/// of the plan, plan_ptr[g] its blob or 0, its entry g, input row 0; else blob grp_ptr[g], entries grp_ent[grp_start[g]
/// ..], entry e's input row e >> 3).  x / h as int8 rows with an F32 scale per 32 (xq, xd: rows of n_embd; hq, hd: row e
/// = entry e, rows of n_ff).
struct Args {
    const unsigned long long* plan_ptr = nullptr;
    const unsigned long long* grp_ptr = nullptr;
    const int *grp_start = nullptr, *grp_ent = nullptr, *n_groups = nullptr;
    size_t tr[3] = {0, 0, 0};   // the roles' offsets in a blob
    int n_embd = 0, n_ff = 0;
    float limit = 0.0f;
    const int8_t* xq = nullptr;
    const float* xd = nullptr;
    const int8_t* hq = nullptr;
    const float* hd = nullptr;
    float* h = nullptr;      // [entries][n_ff]: the clamped swiglu
    float* part = nullptr;   // [entries][n_embd]: each entry's down output
    int tg = 0, td = 0;      // gate / up's type, down's
};
/// grid (n_ff / 8, groups): h of every entry (ne_max: a group's most entries).  false: a type not covered.
bool gate_up(const Args& a, int groups, int ne_max, cudaStream_t s);
/// grid (n_embd / 8, groups): part of every entry.
bool down(const Args& a, int groups, int ne_max, cudaStream_t s);

/// The prompt path's expert sets (ds4_gguf.cu's SetK): expert i's blob at wbase + i stride owns the sorted rows
/// [bounds[i], bounds[i + 1]) (local row = sorted row - row0); gate / up: h16[local] = F16 swiglu over x[idx[local]]
/// (F32 rows, ldx); down: y[local] = down(h16[local]) (F16 rows in, ldx; F32 out, ldo).  A set's experts of at most 16
/// rows take the token kernels' form (each weight read once, the rows' inputs rounded to int8 per 32 as the token path
/// rounds them; STRATA_DS4_NO_LIGHT=1: never), the others GEMM tiles (tensor cores on CUDA, F16 operands).
struct SetArgs {
    const uint8_t* wbase = nullptr;
    size_t stride = 0;
    size_t tr[3] = {0, 0, 0};
    const int* bounds = nullptr;
    int row0 = 0, max_rows = 0, n_exp = 0;
    int n_embd = 0, n_ff = 0;
    float limit = 0.0f;
    const float* xf = nullptr;
    const int* idx = nullptr;
    const uint16_t* xh = nullptr;
    // gate / up from the caller's F16 copy of the activations (the same rounding the tiles make themselves: the same
    // bits), so the F32 rows are not read 18+ times a prompt (MarLin's "keep the small operand in its cheap format")
    const uint16_t* xf16 = nullptr;
    int ldx = 0;
    uint16_t* h = nullptr;
    float* y = nullptr;
    int ldo = 0;
    int tg = 0, td = 0;
    int skip_le = 0;   // (the tiles) the experts of at most this many rows are left out: the light kernels took them
    int skip_ge = 0;   // (the tiles) > 0: the experts of at least this many rows are left out - the caller's GEMMs
#if defined(STRATA_METAL)   // (only on Metal: the CUDA and HIP kernels take this struct by value, so its size is part of their ABI)
    const int* hbounds = nullptr;   // the bounds on the host: [n_exp + 1], absolute rows
    int total_rows = 0;             // the set's rows in all
#endif
};
bool set_gate_up(const SetArgs& a, cudaStream_t s);
bool set_down(const SetArgs& a, cudaStream_t s);
/// Test hooks for the matrix-core (HIP gfx11 / gfx12) prompt tiles: hipk_force(-1 default rules, 0 scalar tiles, 1 matrix-core tiles) and the
/// count of matrix-core launches so far (both do nothing / return -1 where there are no such tiles; hipk_force returns whether this device has them)
bool hipk_force(int mode);
long long hipk_launches();
/// The busy experts' path (a prompt's experts of many rows): one expert's matrix - n rows of k values, row_bytes(k)
/// apart - as FP16 rows (ldd apart), the values the tensor-core tiles multiply (their loader's, rounded to FP16 the same
/// way), for the caller's GEMM: the tiles decode an expert's weights again for every 64 of its rows.
bool f16_rows(int type, const uint8_t* w, int k, int n, uint16_t* dst, int ldd, cudaStream_t s);
/// Whether f16_rows decodes this type (the IQ and K-quant types of the tiles; not the trellis types, not Q5_K / Q6_K).
bool f16_rows_ok(int type);
/// dst[r] = x[idx[r]] for r < n (FP16 rows of k values; x ldx apart, dst k apart)
void gather_f16(const uint16_t* x, int ldx, const int* idx, int n, int k, uint16_t* dst, cudaStream_t s);
/// h[r][c] = FP16 of the tiles' clamped SwiGLU of gu[r][c] (gate) and gu[r][ff + c] (up), F32 rows ld apart
void swiglu_f16(const float* gu, int ld, int n, int ff, float limit, uint16_t* h, int ldh, cudaStream_t s);
/// The light form alone (the set's experts of at most 16 rows; the others untouched), for any of these types and MXFP4
/// (tg / td 39: ds4_gguf.cu's experts): false (nothing launched) when it is switched off - the caller's tiles take every
/// expert then; true: they take those of more rows (skip_le 16).
bool light_gate_up(const SetArgs& a, cudaStream_t s);
bool light_down(const SetArgs& a, cudaStream_t s);

/// GEMV jobs on such matrices (an expert's views on the host-routed path): y[c][r] = alpha (row r . x_c) + bias[r] for
/// ncol activations, x as int8 rows (xq + c k, scales xd + c k / 32).
struct MvJob {
    const uint8_t* w = nullptr;
    size_t ld = 0;   // bytes a row
    int type = 0, k = 0, n = 0, ncol = 1;
    const int8_t* xq = nullptr;
    const float* xd = nullptr;
    float* y = nullptr;
    size_t ycs = 0;
    const float* bias = nullptr;
    float alpha = 1.0f;
};
bool mv(const MvJob* jobs, int n, cudaStream_t s);

/// n rows of k values of a matrix of the type (ld bytes a row) as BF16 (bf16) or F16 bits, dst[n][k]: the chunked prompt path's
/// dense GEMM operand.  false: a type not covered.
bool dequant_rows(int type, const void* raw, size_t ld, int k, int n, uint16_t* dst, bool bf16, cudaStream_t s);

/// The trellis types' kernels (ds4_tq.cu), reached from the entry points above for those types: arithmetic as the CPU lane's
/// (ds4_tq_cpu.cpp: int8 activations with an F32 scale a 32, exact weights), on the tensor cores of gfx12 (WMMA) or in FMAs.
namespace tq {
bool gate_up(const Args& a, int groups, int ne_max, cudaStream_t s);
bool down(const Args& a, int groups, int ne_max, cudaStream_t s);
bool set_gate_up(const SetArgs& a, cudaStream_t s);
bool set_down(const SetArgs& a, cudaStream_t s);
bool mv(const MvJob* jobs, int n, cudaStream_t s);
}  // namespace tq

}  // namespace strata::ds4::gguf::iq
