// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{dsv4-hc.cu,scale.cu,unary.cu,unary.cuh}.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/native_gr_postops.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
constexpr int THREADS = 256;
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
// ggml SCALE uses scale*x+bias, including its +0 bias. Retain this operation
// explicitly so compile-time zero does not change signed-zero behavior.
__device__ __forceinline__ float scale_zero_bias(float x, float scale) {
    return __fmaf_rn(scale, x, 0.0f);
}
__global__ void down_silu(float* lo, int count, float scale) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(count)) return;
    const float x = scale_zero_bias(lo[i], scale);
    lo[i] = x / (1.0f + expf(-x));
}
template<bool Fused>
__global__ void pre_gated(const float* __restrict__ xn, float* __restrict__ gate,
                          float* __restrict__ mixed, int n_embd, int hc, float scale) {
    const std::size_t d = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (d >= std::size_t(n_embd)) return;
    const std::size_t t = blockIdx.y;
    xn += t * std::size_t(n_embd) * hc;
    gate += t * std::size_t(n_embd) * hc;
    mixed += t * std::size_t(n_embd);
    float sum = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const std::size_t i = std::size_t(c) * n_embd + d;
        const float x = xn[i], w = sigmoid(gate[i]);
        const float product = __fmul_rn(x, w);
        gate[i] = product;
        if constexpr (Fused) sum = __fmaf_rn(x, w, sum);
        else sum = c == 0 ? product : __fadd_rn(sum, product);
    }
    if constexpr (Fused) mixed[d] = scale * sum;
    else mixed[d] = scale_zero_bias(sum, scale);
}
__global__ void post(const float* residual, const float* __restrict__ block_out,
                     const float* __restrict__ inject, float* output,
                     int n_embd, int hc, float scale) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(n_embd) * hc) return;
    const std::size_t t = blockIdx.y;
    residual += t * std::size_t(n_embd) * hc;
    block_out += t * std::size_t(n_embd);
    inject += t * std::size_t(hc);
    output += t * std::size_t(n_embd) * hc;
    const int c = int(i / n_embd), d = int(i % n_embd);
    const float weight = scale_zero_bias(sigmoid(scale_zero_bias(inject[c], scale)), 2.0f);
    // Exact residual/output alias is supported; no other thread reads residual[i].
    output[i] = __fmaf_rn(block_out[d], weight, residual[i]);
}
// `post` for n tokens (blockIdx.y = token): the same expression per element
__global__ void post_multi(const float* residual, const float* __restrict__ block_out,
                           const float* __restrict__ inject, float* output, int n_embd, int hc, float scale,
                           long long r_stride, long long b_stride, long long i_stride) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(n_embd) * hc) return;
    const long long t = blockIdx.y;
    residual += t * r_stride; output += t * r_stride; block_out += t * b_stride; inject += t * i_stride;
    const int c = int(i / n_embd), d = int(i % n_embd);
    const float weight = scale_zero_bias(sigmoid(scale_zero_bias(inject[c], scale)), 2.0f);
    // Exact residual/output alias is supported; no other thread reads residual[i].
    output[i] = __fmaf_rn(block_out[d], weight, residual[i]);
}
void check_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float))
        throw std::invalid_argument("native GR postops require non-null four-byte aligned pointers");
}
void check_shape(int n, int hc) {
    if (n <= 0 || hc <= 0 || std::uint64_t(n) * hc > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native GR postops require positive bounded dimensions");
}
unsigned blocks(std::size_t n) { return unsigned((n + THREADS - 1) / THREADS); }
void check_launch() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("native GR postops launch: ") + cudaGetErrorString(error));
}
} // namespace

void native_gr_down_silu(float* lo, int hc_lr, int hc, void* stream) {
    check_shape(hc_lr, hc);
    check_pointer(lo);
    down_silu<<<blocks(hc_lr), THREADS, 0, static_cast<cudaStream_t>(stream)>>>(lo, hc_lr, 1.0f / float(hc));
    check_launch();
}
void native_gr_pre_gated(const float* xn, float* gate, float* mixed,
                         int n_embd, int hc, bool fused_layer, void* stream) {
    native_gr_pre_gated_multi(xn, gate, mixed, n_embd, hc, 1, fused_layer, stream);
}
void native_gr_pre_gated_multi(const float* xn, float* gate, float* mixed,
                               int n_embd, int hc, int n_tok, bool fused_layer, void* stream) {
    if (n_tok <= 0)
        throw std::invalid_argument("native GR postops require positive n_tok");
    check_shape(n_embd, hc);
    check_pointer(xn); check_pointer(gate); check_pointer(mixed);
    const dim3 grid{blocks(n_embd), unsigned(n_tok), 1u};
    if (fused_layer)
        pre_gated<true><<<grid, THREADS, 0, static_cast<cudaStream_t>(stream)>>>(xn, gate, mixed, n_embd, hc, 1.0f / float(hc));
    else
        pre_gated<false><<<grid, THREADS, 0, static_cast<cudaStream_t>(stream)>>>(xn, gate, mixed, n_embd, hc, 1.0f / float(hc));
    check_launch();
}
void native_gr_post(const float* residual, const float* block_out, const float* inject,
                    float* output, int n_embd, int hc, void* stream) {
    native_gr_post_multi(residual, block_out, inject, output, n_embd, hc, 1, stream);
}
void native_gr_post_multi(const float* residual, const float* block_out, const float* inject,
                          float* output, int n_embd, int hc, int n_tok, void* stream) {
    if (n_tok <= 0)
        throw std::invalid_argument("native GR postops require positive n_tok");
    check_shape(n_embd, hc);
    check_pointer(residual); check_pointer(block_out); check_pointer(inject); check_pointer(output);
    const dim3 grid{blocks(std::size_t(n_embd) * hc), unsigned(n_tok), 1u};
    post<<<grid, THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        residual, block_out, inject, output, n_embd, hc, 1.0f / float(hc));
    check_launch();
}
void native_gr_post_multi(const float* residual, const float* block_out, const float* inject, float* output,
                          int n_embd, int hc, int n_tok, long long r_stride, long long b_stride, long long i_stride,
                          void* stream) {
    check_shape(n_embd, hc);
    check_pointer(residual); check_pointer(block_out); check_pointer(inject); check_pointer(output);
    if (n_tok < 1) return;
    post_multi<<<dim3(blocks(std::size_t(n_embd) * hc), unsigned(n_tok)), THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        residual, block_out, inject, output, n_embd, hc, 1.0f / float(hc), r_stride, b_stride, i_stride);
    check_launch();
}
} // namespace strata::kernels
