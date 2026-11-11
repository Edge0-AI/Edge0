// A770 port: grouped FP16 XMX GEMM for the prompt path's experts (STRATA_PF_XMX=1).  See sycl/src/prefill/xmx_moe.dp.cpp.
#pragma once
#include <cstdint>

namespace strata::prefill::xmx {
constexpr int kMaxGroup = 16;

/// STRATA_PF_XMX=1: the FP16 expert path dequantizes a group of up to kMaxGroup experts, then runs each product
/// over the group in one launch.
bool enabled();
/// 0 off, 1 dequantize to FP16 then grouped XMX GEMM, 2 grouped XMX with the IQ decode fused (streamed walk only)
int mode();
/// gate/up on XMX only while the group's mean rows per expert stay below this (STRATA_PF_XMX_GU_MAX, default 250)
int gu_max_mean();

/// For each expert i < n: Y_i = X_i . W_i^T, X_i = cnt[i] FP16 rows of K, packed back to back from X (Y likewise,
/// N floats per row); W_i row-major [N][K] FP16.  N a multiple of 128, K of 32.  Enqueued on `stream`, no sync.
void grouped_f16(const uint16_t* X, const uint16_t* const* W, const int32_t* cnt, int n, float* Y, int N, int K,
                 void* stream);
}  // namespace strata::prefill::xmx
