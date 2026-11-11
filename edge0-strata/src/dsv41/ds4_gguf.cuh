// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_gguf.cuh - a GGUF's weights on the device for maya-ds4 (unsloth's DeepSeek-V4.1-Flash-1017 GGUFs,
// tools/ds4/pack_gguf.py): the dense matrices in BF16 and the routed experts in MXFP4.  They stand in for
// strata::dsv41k::exl3's matrices and experts (an x3::Mat with fmt kFmtBF16, an x3::ExpertLayout with fmt kFmtMXFP4)
// with the same arguments and the same results' layout, so the engine's paths (decode, verify windows, chunked prompts,
// tensor parallel) take either format.
//
// MXFP4 is ggml's block_mxfp4: 32 values in 17 bytes, an E8M0 scale e then 16 bytes of E2M1 codes - value j of the
// block in byte j's low nibble (j < 16) or byte j - 16's high nibble, worth kvalues[code] * 2^(e - 128) with kvalues
// {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12} (the E2M1 values doubled).  A matrix is [out][in / 32] blocks
// (row r = output r), as the GGUF stores it, so an expert's blob is its gate, up and down matrices one after another.
// Every product is an F32 sum of exactly dequantised weights times F32 activations, and a window's entry gets the bits
// the token path gives it (the same kernel, the same order).
#pragma once

#include "strata/dsv41/exl3.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace strata::ds4::gguf {

namespace x3 = strata::dsv41k::exl3;

constexpr int kFmtBF16 = 1;    // x3::Mat::fmt
constexpr int kFmtMXFP4 = 2;   // x3::ExpertLayout::fmt (and an expert's matrix as an x3::Mat)
constexpr int kFmtQ8_0 = 3;    // x3::Mat::fmt: ggml's Q8_0, an FP16 scale and 32 int8 in 34 bytes (UD-Q4_K_XL's dense)
constexpr int kFmtGG = 4;      // x3::Mat::fmt: ggml's low-bit block type m.K (ds4_iq.cuh's: IQ2_XXS .. Q4_K), [n][k / 256]
constexpr int kMxBlock = 32, kMxBytes = 17, kQ8Bytes = 34;
constexpr int kTypeMXFP4 = 39;   // ggml's type id

/// Bytes of an MXFP4 row of k values (k % 32 == 0), and of a matrix of n such rows.
__host__ __device__ inline size_t mx_row_bytes(int64_t k) { return (size_t) (k / kMxBlock) * kMxBytes; }
__host__ __device__ inline size_t mx_bytes(int64_t n, int64_t k) { return (size_t) n * mx_row_bytes(k); }
__host__ __device__ inline size_t q8_row_bytes(int64_t k) { return (size_t) (k / 32) * kQ8Bytes; }

/// A GGUF expert layer (x3::ExpertLayout::fmt kFmtMXFP4: block-quantized experts read in place) keeps each role's ggml
/// type in lay.K[r] (r: gate, up, down; 0 = MXFP4): MXFP4 takes these kernels, the low-bit types ds4_iq.cuh's (gate
/// and up of one type).  Roles of one layer may differ (down of another type); MXFP4 and a low-bit type do not mix.
inline int expert_type(const x3::ExpertLayout& l, int r) { return l.K[r] == 0 ? kTypeMXFP4 : l.K[r]; }
inline bool mx_experts(const x3::ExpertLayout& l) {
    return expert_type(l, 0) == kTypeMXFP4 && expert_type(l, 1) == kTypeMXFP4 && expert_type(l, 2) == kTypeMXFP4;
}
/// Bytes of a row of k values of an expert type (0: not one the device kernels take, or k not whole blocks).
size_t type_row_bytes(int type, int64_t k);
/// Values a block of an expert type (32: MXFP4, 256: the low-bit ones; 0: neither).
int type_block(int type);
/// The type's name (ggml's), "?" when unknown.
const char* type_name(int type);
/// Whether the device kernels take this type (MXFP4 is its own path; the i-quants and K-quants ds4_iq.cu instantiates).
/// A false here with a layout known to type_row_bytes() means the CPU lane reads it and it must not enter the VRAM tier.
bool type_on_device(int type);
/// Whether the *chunk* path's expert tile GEMM (ds4_mm.metal on Metal) takes this type.  IQ1_M has a matvec kernel on
/// Metal but no tile kernel there, so a prompt in chunks would stream it to the GPU and stop; false here means the
/// prompts must run in windows instead (ds4_model.cu switches, and says so).
bool type_on_chunk(int type);

/// The per-stream scratch of the MXFP4 experts on the current device (once per stream, as x3::prepare).
bool prepare(cudaStream_t s);

/// x3::mv's contract for GGUF matrices (F32 activations): each job's m->fmt kFmtBF16 (raw = BF16 [n][ld], ld values a
/// row), kFmtMXFP4 (raw = MXFP4 [n][k / 32] blocks, ld bytes a row; 0: mx_row_bytes(k)) - an expert's matrix in VRAM -
/// or kFmtQ8_0 (raw = Q8_0 [n][k / 32] blocks, ld bytes a row; 0: q8_row_bytes(k)).
bool mv(const x3::MvJob* jobs, int n, cudaStream_t s);
/// The columns [c0, c0 + nc) of such a matrix: its rows c0 .. c0 + nc - 1, a contiguous range.
x3::Mat view(const x3::Mat& m, int c0, int nc);
/// x3::reconstruct_rows' layout from a BF16 or Q8_0 matrix: dst[r][c] = W[c][r] (n rows of k values), F16 or BF16
/// bits (a Q8_0 weight rounded to them).
void rows(const x3::Mat& m, uint16_t* dst, bool bf16, cudaStream_t s);

/// x3::moe_gate_up / moe_down / win_experts / set_gate_up / set_down for a GGUF's experts (lay.fmt == kFmtMXFP4; the
/// roles' types in lay.K, see expert_type).
void moe_gate_up(const x3::MoeArgs& a, const float* x, float* h, cudaStream_t s);
void moe_down(const x3::MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s);
bool win_experts(const x3::WinArgs& a, cudaStream_t s);
void set_gate_up(const x3::SetArgs& a, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh,
                 cudaStream_t s);
void set_down(const x3::SetArgs& a, const uint16_t* h16, int ldh, float* y, int ldy, cudaStream_t s);

/// Launch failures of these kernels since the process started (each is also printed).
int launch_errors();

}  // namespace strata::ds4::gguf
