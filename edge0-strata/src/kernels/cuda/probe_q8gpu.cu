// src/kernels/cuda/probe_q8gpu.cu - DESIGN-4: the GPU-side Q8_0 quantizer for the latent CPU
// share's act rows (the D-stage output quantized on the device, written straight into mapped host
// memory - the host's 7.2 ms/window serial act quantization becomes a pointer).
//
// Bit-exact against ggml's quantize_row_q8_0_ref by construction (probe_std.cu, 2M blocks x2,
// zero diff): d = amax/127 (f16), q = roundf(x * (1/d)) - the MULTIPLICATION, not the division
// (x/d vs x*(1/d) differs at half-points: 10 bytes per 2M blocks, all +-1).
#include "strata/kernels/q8_1_finite.hpp"

namespace strata::kernels {

struct block_q8_0_g { __half d; int8_t qs[32]; };

__global__ void quantize_q8_0_rows_kernel(const float* __restrict__ x, block_q8_0_g* __restrict__ y,
                                          long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xi = x[i];
    float amax = fabsf(xi);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = q8_1_finite(amax / 127.0f);
    const long long ib = i / 32, iqs = i % 32;
    float q = 0.0f;
    if (amax != 0.0f) {
        const float id = 1.0f / d;   // the CPU ref multiplies - replicate exactly (half-point parity)
        q = roundf(xi * id);
    }
    q = q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q;
    y[ib].qs[iqs] = (int8_t) q;
    if (iqs == 0) y[ib].d = __float2half(d);
}

void quantize_q8_0_rows(const float* x, long long n_rows, long long n_cols, void* y, void* stream) {
    const long long n = n_rows * n_cols;
    if (n <= 0) return;
    quantize_q8_0_rows_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(
        x, (block_q8_0_g*) y, n);
}

}  // namespace strata::kernels
