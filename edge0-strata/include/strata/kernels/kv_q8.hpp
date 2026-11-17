// include/strata/kernels/kv_q8.hpp - plan v0.3 P7: 8-bit KV storage for the QSA layers.
//
// The FP16 KV pools cost 2 x 12,288 bytes per token over the 12 QSA layers; with the FP32 pooled indexer keys the
// engine allocates 29 KB per token, 7.65 GB at 262K (bench/results/2026-09-23-p7-int-audit), which cannot fit
// 12 GB. This stores K and V as int8 codes with one FP16 scale per 64 values (per cell, per KV head):
//     scale = fp16(max|x| / 127),  code = clamp(rint(x / scale), -127, 127),  x' = fp16(code * scale)
// Per layer and token: 2 KV heads x 256 values x (K, V) = 1,024 values -> 1,024 B of codes + 32 B of scales,
// instead of 2,048 B in FP16. Over the 12 QSA layers: 12,672 B/token instead of 24,576 (3.3 GB at 262K, not 6.4).
//
// Only STORAGE changes: `kv_gather_q8_step` dequantizes the selected cells into the same FP16 scratch the
// attention kernels already read, so `qsa_attend` and the native attention adapter are untouched. Layout mirrors
// the FP16 pools: codes `[page][kv_head][page_size][head_dim]`, scales `[page][kv_head][page_size][head_dim/64]`.
#pragma once

#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

inline constexpr int KV_Q8_GROUP = 64;

/// Bytes per cell (one token, one layer): K and V codes plus their scales.
inline uint64_t kv_q8_bytes_per_cell(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * s.head_dim * 2 + (uint64_t) s.n_head_kv * (s.head_dim / KV_Q8_GROUP) * 2 * 2;
}

/// Append the cell at step[kStepPos] (graph-capturable: position and page come from device memory). With a host
/// copy (KV streaming, `kv_stream.hpp`) the cell is written there too, and to VRAM only if its block is resident.
/// S26 STRATA_LFUSE: n_tok kv_append_q8_step calls (token j: step + j * step_stride, kcur / vcur + j * cur_stride)
/// in one launch, bitwise the same stores (the tokens' cells are distinct).
void kv_append_q8_steps(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                        const int32_t* step, int step_stride, const float* kcur, const float* vcur, int cur_stride,
                        int n_tok, const QsaShapes& s, void* stream, const KvHostPools* host);
void kv_append_q8_step(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                       const int32_t* step, const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host = nullptr);

/// Gather step[kStepWidth] cells named by `ids` into FP16 scratch `[id][kv_head][head_dim]`; the grid is sized by
/// `max_ids` (capacity), the kernel reads the real count from `step`.
void kv_gather_q8_step(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                       const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                       const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream);

}  // namespace strata::kernels
