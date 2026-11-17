// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh,unary.cu,unary.cuh,ssm-conv.cu,scale.cu}.
// Compile with --use_fast_math, as the pinned backend does.
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
#include "strata/kernels/native_gdn_preprocess.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace strata::kernels {
namespace {
constexpr int S = 128;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}
__device__ __forceinline__ float norm_sum(float value, float* sums) {
    const int lane = threadIdx.x % 32;
    value = warp_sum(value);
    if (lane == 0) sums[threadIdx.x / 32] = value;
    __syncthreads();
    value = lane < 8 ? sums[lane] : 0.0f;
    return warp_sum(value);
}
__device__ __forceinline__ float sigmoid(float value) { return 1.0f / (1.0f + expf(-value)); }

__global__ void conv_silu(float* __restrict__ history, const float* __restrict__ input,
                           const float* __restrict__ weights, float* __restrict__ raw_output,
                           float* __restrict__ silu_output, int channels) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float values[4] = {history[c * 3], history[c * 3 + 1], history[c * 3 + 2], input[c]};
    float sum = 0.0f;
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) sum += values[tap] * weights[c * 4 + tap];
    // The native SSM kernel adds its zero bias even when there is no bias input.
    sum = __fadd_rn(sum, 0.0f);
    raw_output[c] = sum;
    silu_output[c] = sum / (1.0f + expf(-sum));
#pragma unroll
    for (int tap = 0; tap < 3; ++tap) history[c * 3 + tap] = values[tap + 1];
}

__global__ void l2_norm(float* input, float epsilon, float scale_after) {
    const int col = threadIdx.x;
    input += size_t(blockIdx.x) * S;
    const float value = col < S ? input[col] : 0.0f;
    float partial = 0.0f;
    if (col < S) partial += value * value;
    __shared__ float sums[32];
    partial = norm_sum(partial, sums);
    const float scale = rsqrtf(partial / S + epsilon);
    if (col < S) {
        // Preserve the FP32 store boundary between RMSNorm and ggml_scale.
        const float normalized = __fmul_rn(scale, value);
        input[col] = __fmaf_rn(normalized, scale_after, 0.0f);
    }
}

__global__ void beta_sigmoid(float* beta, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) beta[i] = sigmoid(beta[i]);
}
__global__ void gate_softplus(const float* __restrict__ alpha, const float* __restrict__ dt,
                               const float* __restrict__ ssm_a, float* __restrict__ gate, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float value = __fadd_rn(alpha[i], dt[i]);
    const float softplus = value > 20.0f ? value : log1pf(expf(value));   // 1 + e^v loses e^v below ~1e-7
    gate[i] = softplus * ssm_a[i];
}

__global__ void out_norm(const float* __restrict__ input, const float* __restrict__ z,
                          const float* __restrict__ gamma, float* __restrict__ output, float epsilon) {
    const int col = threadIdx.x;
    const size_t offset = size_t(blockIdx.x) * S;
    const float value = col < S ? input[offset + col] : 0.0f;
    float partial = 0.0f;
    if (col < S) partial += value * value;
    __shared__ float sums[32];
    partial = norm_sum(partial, sums);
    const float scale = rsqrtf(partial / S + epsilon);
    if (col < S) {
        // RMSNorm+gamma is one pinned fused operator, followed by sigmoid*mul.
        const float weighted = __fmul_rn(__fmul_rn(scale, value), gamma[col]);
        output[offset + col] = weighted * sigmoid(z[offset + col]);
    }
}

struct Span { const void* pointer; size_t bytes; };
void valid(Span span) {
    const auto address = reinterpret_cast<uintptr_t>(span.pointer);
    if (!span.pointer || address % sizeof(float) || span.bytes > UINTPTR_MAX - address)
        throw std::invalid_argument("native GDN preprocessing requires aligned nonnull valid spans");
}
void disjoint(Span a, Span b) {
    const auto ap = reinterpret_cast<uintptr_t>(a.pointer), bp = reinterpret_cast<uintptr_t>(b.pointer);
    if (ap < bp + b.bytes && bp < ap + a.bytes)
        throw std::invalid_argument("native GDN preprocessing requires disjoint writable spans");
}
void count_and_stream(int64_t count, void* stream) {
    if (!stream || count <= 0 || count > 65535)
        throw std::invalid_argument("native GDN preprocessing requires a stream and count in [1,65535]");
}
void norm_geometry(int64_t rows, int64_t cols, float epsilon, void* stream) {
    count_and_stream(rows, stream);
    if (cols != S || !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native GDN norm requires width 128 and finite nonnegative epsilon");
}
void check_launch() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}

void native_gdn_conv_silu(float* history, const float* input, const float* weights,
                          float* raw_output, float* silu_output, int64_t channels,
                          int64_t d_conv, void* stream) {
    count_and_stream(channels, stream);
    if (d_conv != 4) throw std::invalid_argument("native GDN convolution requires four taps");
    const size_t bytes = size_t(channels) * sizeof(float);
    const Span writable[] = {{history, 3 * bytes}, {raw_output, bytes}, {silu_output, bytes}};
    const Span inputs[] = {{input, bytes}, {weights, 4 * bytes}};
    for (auto span : writable) valid(span);
    for (auto span : inputs) valid(span);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < i; ++j) disjoint(writable[i], writable[j]);
        for (auto span : inputs) disjoint(writable[i], span);
    }
    conv_silu<<<unsigned((channels + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        history, input, weights, raw_output, silu_output, int(channels));
    check_launch();
}
void native_gdn_l2_norm(float* input, int64_t rows, int64_t cols, float epsilon, void* stream) {
    norm_geometry(rows, cols, epsilon, stream);
    valid({input, size_t(rows) * S * sizeof(float)});
    l2_norm<<<unsigned(rows), 256, 0, static_cast<cudaStream_t>(stream)>>>(input, epsilon / S, 1.0f / sqrtf(float(S)));
    check_launch();
}
void native_gdn_beta_gate(float* beta, int64_t heads, void* stream) {
    count_and_stream(heads, stream);
    valid({beta, size_t(heads) * sizeof(float)});
    beta_sigmoid<<<unsigned((heads + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(beta, int(heads));
    check_launch();
}
void native_gdn_gate(const float* alpha, const float* dt, const float* ssm_a,
                     float* gate, int64_t heads, void* stream) {
    count_and_stream(heads, stream);
    const size_t bytes = size_t(heads) * sizeof(float);
    const Span output{gate, bytes};
    valid(output);
    for (auto input : {Span{alpha, bytes}, Span{dt, bytes}, Span{ssm_a, bytes}}) {
        valid(input);
        disjoint(output, input);
    }
    gate_softplus<<<unsigned((heads + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(alpha, dt, ssm_a, gate, int(heads));
    check_launch();
}
void native_gdn_out_norm(const float* output, const float* z, const float* gamma,
                         float* destination, int64_t heads, int64_t cols,
                         float epsilon, void* stream) {
    norm_geometry(heads, cols, epsilon, stream);
    const size_t bytes = size_t(heads) * S * sizeof(float);
    const Span writable{destination, bytes};
    valid(writable);
    for (auto input : {Span{output, bytes}, Span{z, bytes}, Span{gamma, S * sizeof(float)}}) {
        valid(input);
        disjoint(writable, input);
    }
    out_norm<<<unsigned(heads), 256, 0, static_cast<cudaStream_t>(stream)>>>(output, z, gamma, destination, epsilon);
    check_launch();
}
} // namespace strata::kernels
