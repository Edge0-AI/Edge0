// src/kernels/cuda/elementwise.cu - P2.S5's glue kernels.  See the header for why each exists.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/dp4a.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;

__global__ void embedding_gather_kernel(const uint8_t* __restrict__ codes,
                                       const float* __restrict__ scales,
                                       const float* __restrict__ offsets, int64_t n,
                                       int code_bits, int code_bias, int group_elems,
                                       float* __restrict__ out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (codes[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    const float product = __fmul_rn((float) (code + code_bias), scales[group]);
    out[i] = __fadd_rn(product, offsets ? offsets[group] : 0.0f);
}

/// `ggml_compute_softplus_f32`: `log1p(exp(x))`, with the large-x branch that avoids overflow.
///
/// The branch is not cosmetic. `exp(89)` overflows f32 and `exp(20)` is already 4.85e8 where `log1p` loses
/// relative precision; above 20 the function is `x` to within f32 anyway.
__device__ __forceinline__ float softplus_dev(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

__global__ void gdn_gate_kernel(const float* __restrict__ alpha, const float* __restrict__ dt,
                                const float* __restrict__ ssm_a, float* __restrict__ gate, int64_t h_v) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h_v) return;
    const int64_t t = i / h_v;      // `n_tokens` is the leading dim; the real call has one token
    const int64_t h = i % h_v;
    gate[i] = softplus_dev(alpha[i] + dt[h]) * ssm_a[h];
    (void) t;
}

__global__ void scale_kernel(float* __restrict__ x, int64_t n, float s) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// `dst[i] += src[i]`.  See the header: this is what lets R4's GPU half and CPU half run at the same
// time and still add up to one `parts` buffer.
__global__ void add_kernel(float* __restrict__ dst, const float* __restrict__ src, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

__global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f16_from_f32(x[i]);
}

__global__ void to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = bf16_from_f32(x[i]);
}

__global__ void silu_kernel(float* __restrict__ x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // DOUBLE then cast, matching `ref/gdn.py`'s numpy: its arrays are f32 but `np.exp` on an f32 array is
    // computed to f32 precision by a different algorithm than `expf`, and the reference is the oracle.  The
    // difference is in the last bits and this is one line.
    const double v = (double) x[i];
    x[i] = (float) (v / (1.0 + exp(-v)));
}

/// A bounded launch: a zero-length grid is illegal, and `n == 0` is a real call (an empty selection).
inline unsigned grid_for(int64_t n) { return (unsigned) ((n + THREADS - 1) / THREADS); }

/// One WARP per row, reduced through shuffles.  The rows here are short and few - (24, 256), (2, 256),
/// (4, 128) - so a block-per-row tree would spend its time in `__syncthreads` for 8 warps of 32, and the whole
/// call is 30 rows.  A warp reduction with NO shared memory and NO barrier is the shape that fits.
__global__ void rms_norm_weighted_kernel(float* __restrict__ x, const float* __restrict__ w, int64_t rows,
                                         int64_t cols, float eps) {

    const int lane = threadIdx.x & 31;
    const int64_t row = ((int64_t) blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    // **THE ROW GUARD IS NOT DECORATION, AND ITS ABSENCE WAS THE QSA BUG.**  The launcher rounds the grid up to
    // whole 4-warp blocks, so a call with `rows` = 2 - the QSA k-norm, the only non-multiple-of-4 row count in
    // the engine - runs EIGHT warps: rows 0 and 1 are the data, and rows 2 and 3 write 2*cols floats PAST the
    // end of `b.kcur`, which in the arena is exactly where `b.vcur` begins.  `b.vcur` was therefore silently
    // replaced by two rows of `rms_norm(...) * attn_k_norm`: a normalized vector with l2 ~20.6 against V's
    // ~5.7, which the attention then attended to.  `rope_neox_kernel` has had this guard all along.
    if (row >= rows) return;
    float* r = x + row * cols;
    float acc = 0.0f;
    for (int64_t c = lane; c < cols; c += 32) acc += r[c] * r[c];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    // The MEAN, not the sum: `ref/qsa.py::rms_norm` divides by `np.mean(np.square(x))`.  Broadcasting the
    // reciprocal from lane 0 keeps all 32 lanes on the same value - computing `rsqrt` per lane would be the
    // same number but a needless 32-way divergence in the last bit.
    float inv = 0.0f;
    if (lane == 0) inv = rsqrtf(acc / (float) cols + eps);
    inv = __shfl_sync(0xFFFFFFFFu, inv, 0);
    for (int64_t c = lane; c < cols; c += 32) r[c] = (w ? r[c] * w[c] : r[c]) * inv;
}

void sync_if_needed(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

bool check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

}  // namespace

void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets,
                      int64_t n, int code_bits, int code_bias, int group_elems,
                      float* out, void* stream) {
    if (n <= 0) return;
    embedding_gather_kernel<<<grid_for(n), THREADS, 0, (cudaStream_t) stream>>>(
        codes, scales, offsets, n, code_bits, code_bias, group_elems, out);
    check_launch("embedding_gather");
}

void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens,
              int64_t h_v, void* stream) {
    if (n_tokens <= 0 || h_v <= 0) return;
    const int64_t n = n_tokens * h_v;
    gdn_gate_kernel<<<grid_for(n), THREADS, 0, (cudaStream_t) stream>>>(alpha, dt, ssm_a, gate, h_v);
    check_launch("gdn_gate");
    sync_if_needed(stream, "gdn_gate");
}

void scale_inplace(float* x, int64_t n, float s, void* stream) {
    if (n <= 0) return;
    scale_kernel<<<grid_for(n), THREADS, 0, (cudaStream_t) stream>>>(x, n, s);
    check_launch("scale_inplace");
    sync_if_needed(stream, "scale_inplace");
}

void add_inplace(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    add_kernel<<<grid_for(n), THREADS, 0, (cudaStream_t) stream>>>(dst, src, n);
    check_launch("add_inplace");
    sync_if_needed(stream, "add_inplace");
}

void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_f16_kernel<<<grid_for(n), THREADS, 0, (cudaStream_t) stream>>>(x, y, n);
    check_launch("f32_to_f16_bulk");
    sync_if_needed(stream, "f32_to_f16_bulk");
}

void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_bf16_kernel<<<grid_for(n), THREADS, 0, (cudaStream_t) stream>>>(x, y, n);
    check_launch("f32_to_bf16_bulk");
    sync_if_needed(stream, "f32_to_bf16_bulk");
}

void silu_inplace(float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    silu_kernel<<<grid_for(n), THREADS, 0, (cudaStream_t) stream>>>(x, n);
    check_launch("silu_inplace");
    sync_if_needed(stream, "silu_inplace");
}

/// THE DOORBELL.  One thread, one INCREMENT - the cost is the launch, and inside a graph that is paid once.
///
/// **IT INCREMENTS THE MEMORY, AND IT DOES NOT TAKE THE VALUE AS AN ARGUMENT.**  The first version did -
/// doorbell_ring(d_seq, *(h_seq) + 1u) - and that is a CLONED LITERAL: the expression is evaluated on the HOST
/// at CAPTURE time, so the graph rings 1 on every replay and a host waiting for a change waits forever.  It
/// would have looked like a working doorbell on the first token, which is the worst way for it to be wrong.
/// **THE FENCE IS NOT DECORATION, AND ITS ABSENCE WAS COSTING ~10 ms PER TOKEN ON THE HOST SIDE.**
///
/// The ring PUBLISHES three buffers the host is about to read - `h_x_f`, `h_ids`, `h_weights` - so the increment
/// must be ordered after their writes, or the host can observe the ring and then read a payload that has not
/// landed.  `__threadfence_system()` is what orders them, and it covers the HOST as well as the device, which is
/// the whole point: the reader is the CPU.
///
/// Without it the host loop was compensating with a driver call.  `session_loop` polled with `cudaEventQuery`
/// on EVERY spin iteration, because round 195 had measured that a memory-only spin never saw the datum flip -
/// and that measurement was right about the symptom and wrong about the cause.  The cause is this missing fence:
/// the write was not ordered into host-visible memory, so no amount of reading it would show it, and the driver
/// call was flushing the whole pipeline enough to make it appear.  A 10-22 us driver call per iteration is a
/// very expensive substitute for one fence instruction.
///
/// **THE STORE IS VOLATILE**, like `doorbell_publish_kernel`'s.  On RDNA4 (gfx1201) a plain store to mapped pinned
/// memory stays in the GPU's L2 until the stream is synchronized - the host never saw the ring (tests/hip/handoff:
/// 0 of 100 rings seen without a sync; volatile, a system-scope atomic store or a fence after the store: 100 of 100).
__global__ void doorbell_ring_kernel(uint32_t* seq) {
    __threadfence_system();
    *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
#if defined(__HIPCC__)  // #697: HIP only; on RDNA4 the volatile store alone can sit in L2 until the stream syncs
    __threadfence_system();
#endif
}

__global__ void doorbell_wait_kernel(const volatile uint32_t* flag, const volatile uint32_t* seq) {
    const uint32_t want = *seq;
    while (*flag != want) strata_spin_pause();
    __threadfence_system();
}

void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream) {
    if (d_flag == nullptr || d_seq == nullptr) return;
    doorbell_wait_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(d_flag, d_seq);
    check_launch("doorbell_wait");
}

__global__ void copy_from_mapped_kernel(float4* __restrict__ dst, const volatile float4* src, int64_t n4) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (int64_t) gridDim.x * blockDim.x) {
        const float4 v = const_cast<const float4*>(src)[i];
        dst[i] = v;
    }
}

// the CPU rows of a verify window, skipping the rows the GPU plan computes itself (the
// pool writes +0.0 into those, so this writes +0.0 too): block = row, the plan's hit rows `dst[0, *count)`.
__global__ void copy_rows_from_mapped_kernel(float4* __restrict__ dst, const volatile float4* src, int64_t row4,
                                             const int32_t* __restrict__ hit_rows, const int32_t* __restrict__ count,
                                             const volatile uint32_t* wait_flag = nullptr, uint32_t wait_value = 0) {
    const int row = blockIdx.x;
    __shared__ int hit;
    if (threadIdx.x == 0) {
        if (wait_flag != nullptr) {   // 143-amdk: the wait_flag_ge launch folded in
            while (*wait_flag < wait_value) strata_spin_pause();
            __threadfence_system();
        }
        int h = 0;
        const int c = *count;
        for (int i = 0; i < c; ++i) h |= hit_rows[i] == row;
        hit = h;
    }
    __syncthreads();
    float4* d = dst + (int64_t) row * row4;
    if (hit) {
        for (int64_t i = threadIdx.x; i < row4; i += blockDim.x) d[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        const volatile float4* sr = src + (int64_t) row * row4;
        for (int64_t i = threadIdx.x; i < row4; i += blockDim.x) d[i] = const_cast<const float4*>(sr)[i];
    }
}
namespace {
__global__ void scatter_rows_kernel(const float4* __restrict__ src, float4* dst, const int32_t* __restrict__ rows,
                                    int64_t w4) {
    const int64_t r = blockIdx.x;
    const float4* s = src + r * w4;
    float4* d = dst + (int64_t) rows[r] * w4;
    for (int64_t i = threadIdx.x; i < w4; i += blockDim.x) d[i] = s[i];
}
}  // namespace
void scatter_rows_f32(const float* src, float* dst, const int32_t* rows, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "scatter_rows_f32: width must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    scatter_rows_kernel<<<(unsigned) n, 128, 0, (cudaStream_t) stream>>>((const float4*) src, (float4*) dst, rows, width / 4);
}
namespace {
// knife-0: a plain byte copy of mapped host memory into device memory, as a KERNEL.  The host-issued
// cudaMemcpyAsync inside a verify window (the GPU spinning on the flag it raises) is the path issue #31 shows
// stuck on a driver lock (generate.cpp:10329); a kernel moves the same bytes with no host CUDA call in the window.
__global__ void copy_bytes16_from_mapped_kernel(uint4* __restrict__ dst, const volatile uint4* src, int64_t n4) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        const uint4 v = const_cast<const uint4*>(src)[i];
        dst[i] = v;
    }
}
__global__ void copy_bytes_from_mapped_kernel(uint8_t* __restrict__ dst, const volatile uint8_t* src, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = const_cast<const uint8_t*>(src)[i];
}
}  // namespace
void copy_bytes_from_mapped(void* dst, const void* src, int64_t bytes, void* stream) {
    if (bytes <= 0) return;
    const cudaStream_t s = (cudaStream_t) stream;
    if ((bytes & 15) == 0 && ((uintptr_t) dst & 15) == 0 && ((uintptr_t) src & 15) == 0) {
        const int64_t n4 = bytes / 16;
        copy_bytes16_from_mapped_kernel<<<(unsigned) ((n4 + 255) / 256), 256, 0, s>>>(
            (uint4*) dst, (const volatile uint4*) src, n4);
    } else {
        copy_bytes_from_mapped_kernel<<<(unsigned) ((bytes + 255) / 256), 256, 0, s>>>(
            (uint8_t*) dst, (const volatile uint8_t*) src, bytes);
    }
    check_launch("copy_bytes_from_mapped");
}
void copy_rows_from_mapped(float* dst, const float* src, int64_t rows, int64_t width, const int32_t* hit_rows,
                           const int32_t* count, void* stream) {
    if (rows <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_rows_from_mapped: width must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    copy_rows_from_mapped_kernel<<<(unsigned) rows, 128, 0, (cudaStream_t) stream>>>((float4*) dst, (const volatile float4*) src,
                                                                                  width / 4, hit_rows, count);
}
void copy_rows_from_mapped_after(const uint32_t* flag, uint32_t value, float* dst, const float* src, int64_t rows,
                                 int64_t width, const int32_t* hit_rows, const int32_t* count, void* stream) {
    if (rows <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_rows_from_mapped_after: width must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    copy_rows_from_mapped_kernel<<<(unsigned) rows, 128, 0, (cudaStream_t) stream>>>(
        (float4*) dst, (const volatile float4*) src, width / 4, hit_rows, count, (const volatile uint32_t*) flag, value);
}
void copy_from_mapped(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    if ((n & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_from_mapped: n must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    const int64_t n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    copy_from_mapped_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>((float4*) dst, (const volatile float4*) src, n4);
    check_launch("copy_from_mapped");
}

__global__ void doorbell_publish_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids,
                                        const float* __restrict__ w, int n, int k, float* x_out, int32_t* ids_out,
                                        float* w_out, uint32_t* seq) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) x_out[i] = x[i];
    if ((int) threadIdx.x < k) {
        ids_out[threadIdx.x] = ids[threadIdx.x];
        if (w_out != nullptr) w_out[threadIdx.x] = w[threadIdx.x];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
#if defined(__HIPCC__)  // #697: HIP only; on RDNA4 the volatile store alone can sit in L2 until the stream syncs
        __threadfence_system();
#endif
    }
}

__global__ void doorbell_publish_res_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids,
                                            const int32_t* __restrict__ d_res, int n_expert, int n, int k,
                                            float* x_out, int32_t* ids_out, uint32_t* seq) {
    int any_miss = 0;
    if ((int) threadIdx.x < k) {
        const int32_t id = ids[threadIdx.x];
        ids_out[threadIdx.x] = id;
        if (d_res == nullptr || id < 0 || id >= n_expert || d_res[id] < 0) any_miss = 1;
    }
    if (__syncthreads_or(any_miss)) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) x_out[i] = x[i];
        __threadfence_system();
    } else if ((int) threadIdx.x < k) {
        __threadfence_system();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
#if defined(__HIPCC__)  // #697: HIP only; on RDNA4 the volatile store alone can sit in L2 until the stream syncs
        __threadfence_system();
#endif
    }
}

// #649 (HIP, STRATA_DOORBELL_STORE=1): the same publish, but the ring is STORED (the step's own number, known at
// capture) instead of read-modify-written over PCIe - one store, no read of host memory from the GPU.
__global__ void doorbell_publish_value_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids,
                                              const float* __restrict__ w, int n, int k, float* x_out, int32_t* ids_out,
                                              float* w_out, uint32_t* seq, uint32_t value) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) x_out[i] = x[i];
    if ((int) threadIdx.x < k) { ids_out[threadIdx.x] = ids[threadIdx.x]; w_out[threadIdx.x] = w[threadIdx.x]; }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        *(volatile uint32_t*) seq = value;
        __threadfence_system();
    }
}

void doorbell_publish_value(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k,
                            float* x_out, int32_t* ids_out, float* weights_out, uint32_t* d_seq, uint32_t value,
                            void* stream) {
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish: k too large\n"); std::exit(1); }
    doorbell_publish_value_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(x, ids, weights, (int) n, (int) k, x_out,
                                                                          ids_out, weights_out, d_seq, value);
}

void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out,
                      int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream) {
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish: k too large\n"); std::exit(1); }
    doorbell_publish_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(x, ids, weights, (int) n, (int) k, x_out, ids_out,
                                                                    weights_out, d_seq);
    check_launch("doorbell_publish");
}

void doorbell_publish_res(const float* x, const int32_t* ids, const int32_t* d_res, int n_expert, int64_t n, int64_t k,
                          float* x_out, int32_t* ids_out, uint32_t* d_seq, void* stream) {
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish_res: k too large\n"); std::exit(1); }
    doorbell_publish_res_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(x, ids, d_res, n_expert, (int) n, (int) k,
                                                                        x_out, ids_out, d_seq);
    check_launch("doorbell_publish_res");
}

__global__ void copy_i32_from_mapped_kernel(int32_t* __restrict__ dst, const volatile int32_t* src, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

#if !defined(__HIPCC__)
struct MappedCopies {
    MappedCopy c[8];
};

__global__ void copy_from_mapped_multi_kernel(MappedCopies a) {
    const MappedCopy c = a.c[blockIdx.y];
    const int64_t i0 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x, st = (int64_t) gridDim.x * blockDim.x;
    if ((c.words & 3) == 0 && ((uintptr_t) c.dst & 15) == 0 && ((uintptr_t) c.src & 15) == 0) {
        const volatile float4* s = (const volatile float4*) c.src;
        for (int64_t i = i0; i < c.words / 4; i += st) ((float4*) c.dst)[i] = const_cast<const float4*>(s)[i];
    } else {
        const volatile int32_t* s = (const volatile int32_t*) c.src;
        for (int64_t i = i0; i < c.words; i += st) ((int32_t*) c.dst)[i] = s[i];
    }
}

void copy_from_mapped_multi(const MappedCopy* copies, int n, void* stream) {
    if (n <= 0) return;
    if (n > 8) { std::fprintf(stderr, "copy_from_mapped_multi: at most 8 copies\n"); std::exit(1); }
    MappedCopies a;
    int64_t most = 0;
    for (int i = 0; i < n; ++i) {
        a.c[i] = copies[i];
        most = copies[i].words > most ? copies[i].words : most;
    }
    const int64_t units = (most + 3) / 4;
    const unsigned bx = (unsigned) ((units + 255) / 256 < 32 ? (units + 255) / 256 : 32);
    copy_from_mapped_multi_kernel<<<dim3(bx < 1 ? 1 : bx, (unsigned) n), 256, 0, (cudaStream_t) stream>>>(a);
    check_launch("copy_from_mapped_multi");
}
#endif

void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    copy_i32_from_mapped_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(dst, (const volatile int32_t*) src, (int) n);
    check_launch("copy_i32_from_mapped");
}

#if defined(__HIPCC__)   // AMD keeps one copy a launch
void copy_from_mapped_multi(const MappedCopy* copies, int n, void* stream) {
    for (int i = 0; i < n; ++i)
        copy_i32_from_mapped((int32_t*) copies[i].dst, (const int32_t*) copies[i].src, copies[i].words, stream);
}
#endif

void doorbell_ring(uint32_t* d_seq, void* stream) {
    if (d_seq == nullptr) return;
    doorbell_ring_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(d_seq);
    check_launch("doorbell_ring");
    sync_if_needed(stream, "doorbell_ring");
}

void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    // 4 warps per block, so a row count that is not a multiple of 4 wastes at most 3 warps rather than
    // launching a block per row for a 2-row call.
    const unsigned warps_per_block = 4;
    const unsigned grid = (unsigned) ((rows + warps_per_block - 1) / warps_per_block);
    rms_norm_weighted_kernel<<<grid, warps_per_block * 32, 0, (cudaStream_t) stream>>>(x, w, rows, cols, eps);
    check_launch("rms_norm_weighted");
    sync_if_needed(stream, "rms_norm_weighted");
}

}  // namespace strata::kernels
