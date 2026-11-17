// src/kernels/cuda/quantize_act.cu - P2.S2: quantize an activation the way ggml converts src1.
//
// WHY THIS EXISTS AT ALL.  `ggml_mul_mat` converts the ACTIVATION (src1) to the weight's `vec_dot_type`
// before the dot product - the rule established over rounds 119-136 and written into `ref/quant.py` as
// `apply_src1`.  The CPU expert path already honours it (`bench/micro/cpu_s2.cpp` quantizes to int8 and uses
// `vpdpbusd`), and its P0.T2 parity passes at 1.461e-06.  The GPU kernels added in rounds 166-177 take FP16
// activations instead.  **Left that way, the two paths would compute different numbers for different experts
// of the SAME token** - not a tolerance question, an inconsistency inside one forward pass.
//
// Q2_0 IS THE CASE THAT MATTERS: the routed experts are all Q2_0, `Q2_0 -> Q8_0`, and that is 31.64 GiB of the
// 38 GiB pack.  Q8_K (for the K-quants and IQ4_XS) is the other half of the table and is not here yet.
//
// THE THREE SUBTLETIES, each of which was paid for once already and is transcribed rather than recalled:
//
//   1. `d32 = amax / 127.0f` is computed in FP32, and the QUANTIZED INTEGERS divide by **d32**, not by the
//      fp16-rounded value that gets stored.  Conflating the two shifts every quant near a rounding boundary -
//      measured against ggml's own bytes, 18 of 80 blocks differed, and because Q2_0 experts convert to Q8_0
//      the error landed in all 48 MoE blocks and made the end-to-end KL WORSE.
//   2. The reference divides in FLOAT64 (`blk.astype(np.float64) / float(np.float32(d32))`) and rounds with
//      `rint` - half to even.  This kernel therefore divides in double too: an FP32 division followed by
//      `rintf` differs from the reference wherever the f32 quotient rounds across a .5 boundary.
//   3. The DEQUANTIZED value is `q * d16`, the fp16 scale, not `q * d32`.  The block stores fp16 and that is
//      what a reader multiplies by.
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/quantize_act_dev.cuh"
#include "strata/kernels/swiglu.cuh"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK8_0 = 32;

// THE fp16 CONVERSION LIVES IN `strata/kernels/f16_bits.hpp`, and this file used to carry its own copy.
//
// Round 198 found the private copy wrong in a way no fixture here could see: it tested `if (exp >= 31)` to
// detect an out-of-range exponent, which conflates an f32 INF/NAN (raw exponent 255) with a FINITE value too
// large for fp16.  Every finite overflow - 1e30, 65536, 1e45 - came back as a NaN instead of saturating to
// inf, and this file's fixture is O(1) throughout, so it passed.  A numpy-generated oracle caught it on its
// first run.  Three copies of a converter whose failure mode is silent wrong bits was two copies too many.


__global__ void quantize_q8_0_kernel(const float* __restrict__ x, uint8_t* __restrict__ blocks,
                                     long long n_blocks) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    quantize_q8_0_block(x + b * QK8_0, blocks + b * 34);
}

/// **THE HIT PATH'S QUANTIZER, AND IT EXISTS TO REPRODUCE `act_quant_q8_1` EXACTLY (R4.2h, round 331).**
///
/// The engine computes every routed expert twice.  MISSES go through `cpu/expert.cpp:138 act_quant_q8_1`
/// into the VNNI kernel; HITS went through `quantize_q8_0` above.  **Those are not the same quantization**,
/// and round 330 measured the difference with both real implementations linked
/// (`bench/micro/act_quant_parity.cu`):
///
///     int8 activations differing : 0 of 2560                       <- this rule's ties are measure-zero
///     chunks whose SCALE differs : 80 of 80, max rel 4.761e-04     <- 2^-11: fp32 vs the block's fp16
///
/// The scale is the difference that matters: `ActQ::scale` is `float` and the CPU kernel multiplies by it
/// (`expert.cpp:92`), while `row_dot_s2_q8` read `f16_at(xb)` out of the `block_q8_0`.  So this variant
/// writes the SAME 34-byte blocks (the kernel's indexing is unchanged) **and** a parallel fp32 scale array
/// carrying `d32` unrounded.
///
/// It also adopts the CPU's ROUNDING RULE - multiply by the reciprocal, round half AWAY FROM ZERO - rather
/// than this file's `rint`.  That is deliberate and it is the opposite of the direction the comment above
/// argues for.  `quantize_q8_0` was tuned to match **ggml's Q8_0 bytes**, which is right for the pack and for
/// `moe_hit_parity`.  This function's job is different: it must match **this engine's own CPU reference**,
/// because a hit and a miss for the same expert on the same layer have to produce the same number.  The CPU
/// path is the reference - C1 passes on it - so the hit path is brought to it, not the reverse.
#if defined(__HIPCC__)   // AMD keeps the thread-a-block kernel
__global__ void quantize_q8_0_scaled_kernel(const float* __restrict__ x, uint8_t* __restrict__ blocks,
                                            float* __restrict__ scales, long long n_blocks) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    quantize_q8_0_scaled_block(x + b * QK8_0, blocks + b * 34, scales + b);
}

#else
// A warp a block, lane i on value i: the block's |max| is a max over the same values (0 for none above it, as the
// serial scan from 0 through fmaxf), so every byte is the per-thread loop's.
__global__ void quantize_q8_0_scaled_kernel(const float* __restrict__ x, uint8_t* __restrict__ blocks,
                                            float* __restrict__ scales, long long n_blocks) {
    const long long b = ((long long) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int i = (int) (threadIdx.x & 31);
    if (b >= n_blocks) return;
    const float xi = x[b * QK8_0 + i];
    uint8_t* out = blocks + b * 34;

    float amax = fmaxf(0.0f, fabsf(xi));
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    // VERBATIM from `cpu/expert.cpp:144-145`, including the `amax > 0` guard, so the fp32 value written here
    // is bit-identical to the `s` the CPU path used.
    const float s = amax > 0.f ? amax / 127.f : 0.f;
    const float inv = s > 0.f ? 1.f / s : 0.f;
    if (i == 0) {
        scales[b] = s;
        const uint16_t d16bits = f16_from_f32(s);
        out[0] = (uint8_t) (d16bits & 0xFF);
        out[1] = (uint8_t) (d16bits >> 8);
    }
    // VERBATIM from `cpu/expert.cpp:159-162`: reciprocal multiply, then `t + copysign(0.5, t)` truncated
    // toward zero, which is `lround`'s rule - round half away from zero.
    const float t = xi * inv;
    const float r = t + (t >= 0.f ? 0.5f : -0.5f);
    int v = (int) r;
    v = v < -127 ? -127 : (v > 127 ? 127 : v);
    out[2 + i] = (uint8_t) (int8_t) v;
}

#endif

__global__ void dequant_q8_0_kernel(const uint8_t* __restrict__ blocks, float* __restrict__ x,
                                    long long n_blocks) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const uint8_t* blk = blocks + b * 34;
    const uint16_t dbits = (uint16_t) (blk[0] | (blk[1] << 8));
    const float d = f32_from_f16(dbits);
    float* out = x + b * QK8_0;
    for (int i = 0; i < QK8_0; ++i) out[i] = (float) (int8_t) blk[2 + i] * d;
}

// ===================== Q8_K =====================
//
// `block_q8_K` = { float d ; int8_t qs[256] ; int16_t bsums[16] } = 292 bytes, no padding
// (`static_assert` in ggml-common.h).  QK_K = 256.

constexpr int QK_K = 256;
constexpr int Q8K_BYTES = 292;

// `nearest_int_dev` and the per-block quantizers live in `quantize_act_dev.cuh` (shared with the fused
// kernels); the comment that used to justify this copy is there.

__global__ void quantize_q8_K_kernel(const float* __restrict__ x, uint8_t* __restrict__ blocks,
                                     long long n_blocks) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    quantize_q8_K_block(x + b * QK_K, blocks + b * Q8K_BYTES);
}

__global__ void dequant_q8_K_kernel(const uint8_t* __restrict__ blocks, float* __restrict__ x,
                                    long long n_blocks) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const uint8_t* blk = blocks + b * Q8K_BYTES;
    float d;
    memcpy(&d, blk, 4);
    const int8_t* qs = (const int8_t*) (blk + 4);
    float* out = x + b * QK_K;
    for (int i = 0; i < QK_K; ++i) out[i] = (float) qs[i] * d;
}

// ===================== fused silu(gate) * up + quantize =====================
//
// ONE kernel where two ran: `swiglu_kernel` wrote the products and the quantizer read them back.  The
// products land in `gate_out` exactly as the standalone kernels stored them and the blocks are the same
// bytes the standalone quantizers produced - the quantizer bodies (`quantize_act_dev.cuh`) run verbatim over
// the shared products.  The swilu is PARALLEL (one lane per element): the first version ran 32 exp calls
// serially per thread and measured SLOWER than the two kernels it replaced (M=8: +22.9%).
//
// `SwiluKind` selects the expression because the engine really has three (see `swiglu.cuh`); each is the
// verbatim expression of the standalone kernel that this replaces.

template <int KIND>
__device__ __forceinline__ float swilu_apply(float g, float u) {
    if constexpr (KIND == 0) return swilu_legacy(g, u);
    else if constexpr (KIND == 1) return swilu_fast(g, u);
    else return swilu_native(g, u);
}

// One WARP per 32-element block of pairs: lane j computes pair j (parallel exp), the products stage through
// shared memory, then the quantizer body runs on them exactly as `quantize_q8_0_kernel` did.
template <int KIND, bool SCALED>
__global__ void swilu_quantize_q8_0_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                           float* __restrict__ gate_out, uint8_t* __restrict__ blocks,
                                           float* __restrict__ scales, long long n_blocks) {
    __shared__ float p[4][Q8_0_BLOCK];                     // one row per warp (4 warps per block)
    const int warps_per_block = (int) (blockDim.x >> 5);
    const long long b = (long long) blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (b >= n_blocks) return;
    const long long i0 = b * Q8_0_BLOCK;
    const float v = swilu_apply<KIND>(gate[i0 + lane], up[i0 + lane]);
    gate_out[i0 + lane] = v;
    p[warp][lane] = v;
    __syncwarp();
    if (lane == 0) {
        if constexpr (SCALED) quantize_q8_0_scaled_block(p[warp], blocks + b * 34, scales + b);
        else quantize_q8_0_block(p[warp], blocks + b * 34);
    }
}

// The same for the Q8_K contract (the shared expert's down projection): one warp per 256-element block,
// lane j taking pairs j, j+32, ... so the 256 exp calls spread over the warp.
template <int KIND>
__global__ void swilu_quantize_q8_K_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                           float* __restrict__ gate_out, uint8_t* __restrict__ blocks,
                                           long long n_blocks) {
    __shared__ float p[2][QK_K];                           // one row per warp (2 warps per block)
    const int warps_per_block = (int) (blockDim.x >> 5);
    const long long b = (long long) blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (b >= n_blocks) return;
    const long long i0 = b * QK_K;
    for (int j = lane; j < QK_K; j += 32) {
        const float v = swilu_apply<KIND>(gate[i0 + j], up[i0 + j]);
        gate_out[i0 + j] = v;
        p[warp][j] = v;
    }
    __syncwarp();
    if (lane == 0) quantize_q8_K_block(p[warp], blocks + b * Q8K_BYTES);
}

template <int KIND>
void swilu_quantize_q8_0_launch(const float* gate, const float* up, float* gate_out, int64_t n, uint8_t* blocks,
                                float* scales, void* stream) {
    const long long nb = n / QK8_0;
    const int threads = 128;                                 // 4 warps = 4 blocks of 32 pairs
    const unsigned grid = (unsigned) ((nb + 3) / 4);
    if (scales != nullptr)
        swilu_quantize_q8_0_kernel<KIND, true><<<grid, threads, 0, (cudaStream_t) stream>>>(
            gate, up, gate_out, blocks, scales, nb);
    else
        swilu_quantize_q8_0_kernel<KIND, false><<<grid, threads, 0, (cudaStream_t) stream>>>(
            gate, up, gate_out, blocks, nullptr, nb);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "swilu_quantize_q8_0 launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

// ===================== every activation image of x in ONE pass =====================
//
// The layer start quantized `x` with two or three kernels (Q8_0, Q8_K, bf16) - three reads of the same
// buffer and three launches, per layer.  One kernel here produces whichever images the caller asks for;
// every image's bytes are what its standalone kernel produced (the per-block math is shared, and each block
// reads the same `x` values).  Null pointers skip that image.
__global__ void quantize_images_kernel(const float* __restrict__ x, uint8_t* __restrict__ b80,
                                       uint8_t* __restrict__ bK, uint16_t* __restrict__ b16, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const long long nb80 = n / Q8_0_BLOCK;                       // 32-element blocks
    if (i < nb80) {
        if (b80 != nullptr) quantize_q8_0_block(x + i * Q8_0_BLOCK, b80 + i * Q8_0_BLOCK_BYTES);
        if (b16 != nullptr)
            for (int j = 0; j < Q8_0_BLOCK; ++j) b16[i * Q8_0_BLOCK + j] = bf16_from_f32(x[i * Q8_0_BLOCK + j]);
    }
    // q8_K blocks are 256 wide: the thread whose 32-block starts a super-block computes it (256 values in
    // one thread, exactly as `quantize_q8_K_kernel` did).
    if (bK != nullptr && (i % (Q8_K_BLOCK / Q8_0_BLOCK)) == 0) {
        const long long k = i / (Q8_K_BLOCK / Q8_0_BLOCK);
        if (i < nb80 && k < n / Q8_K_BLOCK) quantize_q8_K_block(x + k * Q8_K_BLOCK, bK + k * Q8_K_BLOCK_BYTES);
    }
}

}  // namespace

void swilu_quantize_q8_0(const float* gate, const float* up, float* gate_out, int64_t n, int swilu_kind,
                         uint8_t* blocks, void* stream) {
    if (n <= 0) return;
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "swilu_quantize_q8_0: n %lld is not a multiple of %d\n", (long long) n, QK8_0);
        std::exit(1);
    }
    switch (swilu_kind) {
        case 0: swilu_quantize_q8_0_launch<0>(gate, up, gate_out, n, blocks, nullptr, stream); break;
        case 1: swilu_quantize_q8_0_launch<1>(gate, up, gate_out, n, blocks, nullptr, stream); break;
        default: swilu_quantize_q8_0_launch<2>(gate, up, gate_out, n, blocks, nullptr, stream); break;
    }
}

void swilu_quantize_q8_0_scaled(const float* gate, const float* up, float* gate_out, int64_t n, int swilu_kind,
                                uint8_t* blocks, float* scales, void* stream) {
    if (n <= 0) return;
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "swilu_quantize_q8_0_scaled: n %lld is not a multiple of %d\n", (long long) n, QK8_0);
        std::exit(1);
    }
    if (scales == nullptr) {
        std::fprintf(stderr, "swilu_quantize_q8_0_scaled: scales is null\n");
        std::exit(1);
    }
    switch (swilu_kind) {
        case 0: swilu_quantize_q8_0_launch<0>(gate, up, gate_out, n, blocks, scales, stream); break;
        case 1: swilu_quantize_q8_0_launch<1>(gate, up, gate_out, n, blocks, scales, stream); break;
        default: swilu_quantize_q8_0_launch<2>(gate, up, gate_out, n, blocks, scales, stream); break;
    }
}

void swilu_quantize_q8_K(const float* gate, const float* up, float* gate_out, int64_t n, int swilu_kind,
                         uint8_t* blocks, void* stream) {
    if (n <= 0) return;
    if (n % QK_K != 0) {
        std::fprintf(stderr, "swilu_quantize_q8_K: n %lld is not a multiple of %d\n", (long long) n, QK_K);
        std::exit(1);
    }
    const long long nb = n / QK_K;
    const int threads = 64;                                  // 2 warps = 2 blocks of 256 pairs
    const unsigned grid = (unsigned) ((nb + 1) / 2);
    switch (swilu_kind) {
        case 0: swilu_quantize_q8_K_kernel<0><<<grid, threads, 0, (cudaStream_t) stream>>>(gate, up, gate_out, blocks, nb); break;
        case 1: swilu_quantize_q8_K_kernel<1><<<grid, threads, 0, (cudaStream_t) stream>>>(gate, up, gate_out, blocks, nb); break;
        default: swilu_quantize_q8_K_kernel<2><<<grid, threads, 0, (cudaStream_t) stream>>>(gate, up, gate_out, blocks, nb); break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "swilu_quantize_q8_K launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

void quantize_q8_0(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "quantize_q8_0: n %lld is not a multiple of %d\n", (long long) n, QK8_0);
        std::exit(1);
    }
    const long long nb = n / QK8_0;
    const int threads = 128;
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    quantize_q8_0_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(x, blocks, nb);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "quantize_q8_0 launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

/// See `quantize_q8_0_scaled_kernel`.  Writes the same 34-byte `block_q8_0` layout as `quantize_q8_0`, plus
/// `scales[n/32]` carrying the fp32 `s` the CPU path used - so a hit can be computed with the CPU's
/// multiplier instead of the block's fp16 `d`.  `scales` must not be null.
void quantize_q8_0_scaled(const float* x, uint8_t* blocks, float* scales, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "quantize_q8_0_scaled: n %lld is not a multiple of %d\n", (long long) n, QK8_0);
        std::exit(1);
    }
    if (scales == nullptr) {
        std::fprintf(stderr, "quantize_q8_0_scaled: scales is null\n");
        std::exit(1);
    }
    const long long nb = n / QK8_0;
    const int threads = 128;
#if defined(__HIPCC__)
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
#else
    const unsigned grid = (unsigned) ((nb + threads / 32 - 1) / (threads / 32));   // a warp a block
#endif
    quantize_q8_0_scaled_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(x, blocks, scales, nb);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "quantize_q8_0_scaled launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

void dequant_q8_0(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    const long long nb = n / QK8_0;
    const int threads = 128;
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    dequant_q8_0_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(blocks, x, nb);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dequant_q8_0 launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

void quantize_q8_K(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK_K != 0) {
        std::fprintf(stderr, "quantize_q8_K: n %lld is not a multiple of %d\n", (long long) n, QK_K);
        std::exit(1);
    }
    const long long nb = n / QK_K;
    const int threads = 64;                       // one block per thread, and a block is 256 elements
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    quantize_q8_K_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(x, blocks, nb);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "quantize_q8_K launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

void dequant_q8_K(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    const long long nb = n / QK_K;
    const int threads = 64;
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    dequant_q8_K_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(blocks, x, nb);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dequant_q8_K launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

void quantize_act_images(const float* x, int64_t n, uint8_t* q8_0_blocks, uint8_t* q8_K_blocks,
                         uint16_t* bf16, void* stream) {
    if (n <= 0) return;
    if (q8_0_blocks == nullptr && q8_K_blocks == nullptr && bf16 == nullptr) return;
    if (n % Q8_0_BLOCK != 0 || (q8_K_blocks != nullptr && n % QK_K != 0)) {
        std::fprintf(stderr, "quantize_act_images: n %lld must be a multiple of %d (and %d for Q8_K)\n",
                     (long long) n, Q8_0_BLOCK, QK_K);
        std::exit(1);
    }
    const long long nb = n / Q8_0_BLOCK;
    const int threads = 128;
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    quantize_images_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(x, q8_0_blocks, q8_K_blocks, bf16, n);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "quantize_act_images launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

}  // namespace strata::kernels
