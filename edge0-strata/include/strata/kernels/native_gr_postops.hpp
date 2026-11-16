#pragma once

namespace strata::kernels {

// Pinned llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d contiguous F32
// single-token GR post-operations. Explicit stream, no allocation or wait; null
// selects the default stream. Positive dimensions with n*hc <= INT_MAX, finite
// values/intermediates, and four-byte alignment are required.
// Buffers must not overlap except the explicitly in-place arguments below.
// Implementation compiles separately with --use_fast_math.

// In-place pinned SCALE(1/hc), then SiLU, applied to the raw down projection.
void native_gr_down_silu(float* lo, int hc_lr, int hc, void* stream);

// xn and raw gate contain [n_embd,hc]. gate is overwritten with xn*sigmoid(gate).
// Layer path: pinned DSV4_HC_PRE_GATED ordered FMA accumulation, then scale.
// Head path (fused_layer=false): individually rounded products, ordered adds
// starting from stream zero, then SCALE. This retains the oracle's il=-1 graph.
void native_gr_pre_gated(const float* xn, float* gate, float* mixed,
                         int n_embd, int hc, bool fused_layer, void* stream);

void native_gr_pre_gated_multi(const float* xn, float* gate, float* mixed,
                               int n_embd, int hc, int n_tok, bool fused_layer, void* stream);

// inject[hc] -> SCALE(1/hc), sigmoid, SCALE(2), identity DSV4_HC_POST.
// residual/output contain [n_embd,hc], block_out has n_embd elements.
// output may equal residual exactly; all other overlaps are invalid.
void native_gr_post(const float* residual, const float* block_out, const float* inject,
                    float* output, int n_embd, int hc, void* stream);
/// native_gr_post for n_tok tokens in one launch, token t's operands at t times the given strides (floats): the same
/// expression per element.  output may equal residual exactly.
void native_gr_post_multi(const float* residual, const float* block_out, const float* inject, float* output,
                          int n_embd, int hc, int n_tok, long long r_stride, long long b_stride, long long i_stride,
                          void* stream);

void native_gr_post_multi(const float* residual, const float* block_out, const float* inject,
                          float* output, int n_embd, int hc, int n_tok, void* stream);

} // namespace strata::kernels
