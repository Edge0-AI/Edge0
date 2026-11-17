// src/prefill/moe_ws.cuh - the weight-stationary expert kernels' shared pieces (see moe_ws.hpp): included after
// ggml-cuda's common.cuh and mmq.cuh by moe_ws.cu and the moe_ws_*.cu that hold the kernels (a few formats per file,
// so the instances compile in parallel).
#pragma once

#include "strata/prefill/moe_ws.hpp"

#include <cuda_runtime.h>

namespace strata::prefill::ws::detail {

// The two ways every kernel multiplies: int8 values sign-extended to int16 pairs (one PRMT per pair: bytes 0,2 and 1,3)
// and a pair of products added to a 32-bit sum by two XMAD (mad.wide.s16) - MMQ's dp4a on sm_60, one product at a time.
// A 32-value block's integer sum is exact (|sum| < 2^22 for every format here), so the order of its MACs is free.
__device__ __forceinline__ void split2(const int v, int& lo02, int& hi13) {
    asm("prmt.b32 %0, %1, 0, 0xA280;" : "=r"(lo02) : "r"(v));
    asm("prmt.b32 %0, %1, 0, 0xB391;" : "=r"(hi13) : "r"(v));
}
__device__ __forceinline__ int mad2(const int w, const int a, int c) {
    asm("{\n\t.reg .s16 al, ah, bl, bh;\n\t"
        "mov.b32 {al, ah}, %1;\n\tmov.b32 {bl, bh}, %2;\n\t"
        "mad.wide.s16 %0, al, bl, %0;\n\tmad.wide.s16 %0, ah, bh, %0;\n\t}"
        : "+r"(c) : "r"(w), "r"(a));
    return c;
}
// float(sum) for |sum| < 2^22 without the quarter-rate I2F: a chain seeded with 1.5 * 2^23 holds the float's bits
constexpr int kMagicI = 0x4B400000;
constexpr float kMagicF = 12582912.0f;

// CTA b's chunk and first weight row: chunk b / pieces, piece b % pieces (`pieces` = weight rows / rows per CTA) - the
// chunks' order, each chunk's pieces next to each other
template <int PIECES, int ROWS>
__device__ __forceinline__ Chunk cta_work(const Chunk* __restrict__ chunks, int& r0) {
    r0 = (int) (blockIdx.x % PIECES) * ROWS;
    const int4 c = __ldg((const int4*) chunks + blockIdx.x / PIECES);
    return Chunk{c.x, c.y, c.z, c.w};
}

// one launch of a layer's product, the variant picked by u (rows per routed expert); grid = nchunks x pieces
void gu_iq2_xxs(const Layer& L, double u, cudaStream_t s);
void gu_iq2_xs(const Layer& L, double u, cudaStream_t s);
void gu_iq2_s(const Layer& L, double u, cudaStream_t s);
void gu_iq3_xxs(const Layer& L, double u, cudaStream_t s);
void gu_iq3_s(const Layer& L, double u, cudaStream_t s);
void dn_q2_0(const Layer& L, double u, cudaStream_t s);
// `fast`: ws::fast() (STRATA_WS_FAST=1) - the 32-token tier's fast kernel, the same Dm rows
void dn_iq4_nl(const Layer& L, double u, cudaStream_t s, bool fast);

}  // namespace strata::prefill::ws::detail
