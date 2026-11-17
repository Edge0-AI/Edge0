// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh}. Scope: contiguous weighted F32 RMSNorm.
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

#include "strata/kernels/native_gr_norm.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

__device__ __forceinline__ float norm_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    }
    return value;
}

template<int BlockSize>
__global__ void weighted_rms_norm(const float* __restrict__ input,
                                   const float* __restrict__ gamma,
                                   float* __restrict__ output, int n_cols, float epsilon) {
    const int tid = threadIdx.x;
    const std::size_t row = std::size_t(blockIdx.y) * gridDim.x + blockIdx.x;
    const std::size_t row_offset = row * n_cols;
    input += row_offset;
    gamma += std::size_t(blockIdx.x) * n_cols;
    output += row_offset;
    float partial = 0.0f;
    for (int col = tid; col < n_cols; col += BlockSize) {
        const float value = input[col];
        partial += value * value;
    }

    // Pinned block_reduce<SUM,BlockSize>: every warp repeats the final XOR
    // reduction. There is no warp-0-only broadcast or downward-shuffle tree.
    __shared__ float sums[32];
    partial = norm_warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    __syncthreads();
    partial = 0.0f;
    if (lane < BlockSize / 32) partial = sums[lane];
    partial = norm_warp_sum(partial);

    const float mean = partial / n_cols;
    const float scale = rsqrtf(mean + epsilon);
    for (int col = tid; col < n_cols; col += BlockSize) {
        output[col] = scale * input[col] * gamma[col];
    }
}

void check_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float) != 0)
        throw std::invalid_argument("native GR RMSNorm requires non-null four-byte aligned pointers");
}

} // namespace

void native_gr_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                 int n_cols, int n_rows, float epsilon, void* stream) {
    native_gr_rms_norm_weighted_multi(input, gamma, output, n_cols, n_rows, 1, epsilon, stream);
}

void native_gr_rms_norm_weighted_multi(const float* input, const float* gamma, float* output,
                                       int n_cols, int n_rows, int n_tok, float epsilon, void* stream) {
    if (n_cols <= 0 || n_rows <= 0 || n_tok <= 0 || !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native GR RMSNorm requires positive dimensions and finite nonnegative epsilon");
    check_pointer(input);
    check_pointer(gamma);
    check_pointer(output);
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    const dim3 grid{unsigned(n_rows), unsigned(n_tok), 1u};
    if (n_cols < 1024)
        weighted_rms_norm<256><<<grid, 256, 0, cuda_stream>>>(input, gamma, output, n_cols, epsilon);
    else
        weighted_rms_norm<1024><<<grid, 1024, 0, cuda_stream>>>(input, gamma, output, n_cols, epsilon);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("native GR RMSNorm launch: ") + cudaGetErrorString(error));
}

} // namespace strata::kernels
