// src/kernels/cuda/gr.cu - P2.S2: the gated residual / hyper-connection, `gr_read` and `gr_write`.
//
// See include/strata/kernels/gr.hpp for the semantics, for why the weights are bf16, and for the history of
// this kernel's 134x-off-the-floor first version.  The short form: it was launched `<<<1, 256>>>` so that
// `xn`/`xq` fit in shared memory, which used ONE of 48 SMs and cost 262 ms/token.
//
// THE SHAPE OF THE FIX is that each projection is an ordinary GEMV and is now mapped the way the shapes want:
//
//     norm    hc blocks, one per residual stream            (each needs its own reduction)
//     down    ONE WARP PER OUTPUT ROW, lanes stride the 10240 reduction axis
//     gate    ONE WARP PER OUTPUT ROW, lanes stride the 320 reduction axis
//     mean    a flat elementwise pass over n_embd
//
// WARP-PER-ROW RATHER THAN THREAD-PER-ROW, and that is the whole difference.  A thread walking one row is
// coalesced with ITSELF and 32 transactions away from its warp neighbours: `w_down[k*hc_dim + i]` for fixed k
// is contiguous, but two threads on rows k and k+1 are hc_dim*2 bytes apart.  With the warp on one row, lanes
// i, i+1, i+2 touch consecutive addresses in EITHER orientation - which is why this version needs no
// transposed copies and takes both matrices exactly as the manifest stores them.
//
// Layouts are the MANIFEST's: `hc_attn_down.weight` is [10240, 320] with ne0 = hc*n_embd fast, so it holds
// `w_down[k][i]` with i contiguous - `ref/gr.py`'s (hc_lr, hc*n_embd) row-major.  `hc_attn_up.weight` is
// [320, 10240] so it holds `w_up[i][k]` with k contiguous - `ref/gr.py`'s (hc*n_embd, hc_lr) row-major.  Both
// index with no permutation at all.
#include "strata/kernels/gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_gr_postops.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
bool fp32_activations = false;
bool native_mmvf = false;

/// The bf16 conversions live in `strata/kernels/bf16_bits.hpp`, and this file used to carry its own copies.
/// Two files with a private copy of a conversion whose failure mode is silent wrong bits is one too many -
/// and the reason the shared header exists is that `bf16` and `fp16` are NOT two spellings of one idea, so a
/// routine written for one and reused for the other is wrong in a way that still produces numbers.
///
/// The local names are kept as thin wrappers so the body below is untouched by the swap.
__device__ __forceinline__ uint16_t f32_to_bf16_bits(float f) { return bf16_from_f32(f); }
__device__ __forceinline__ float bf16_bits_to_f32(uint16_t h) { return f32_from_bf16(h); }
__device__ __forceinline__ float activation_f32(float x) { return x; }
__device__ __forceinline__ float activation_f32(uint16_t x) { return bf16_bits_to_f32(x); }
__device__ __forceinline__ void store_activation(float* dst, int i, float x) { dst[i] = x; }
__device__ __forceinline__ void store_activation(uint16_t* dst, int i, float x) {
    dst[i] = f32_to_bf16_bits(x);
}

/// silu and sigmoid exactly as `ref/gr.py` writes them - in FLOAT, not double.
///
/// This is the opposite of `shared_expert.cu`, which uses double, and the difference is not stylistic:
/// `ref/gr.py`'s arrays are float32, so `x / (1.0 + np.exp(-x))` evaluates in float32 under NEP 50, while
/// `ref/moe.py` works in float64.  Matching the reference's PRECISION is part of transcribing it.
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ double warp_sum(double v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    return __shfl_sync(0xFFFFFFFFu, v, 0);
}

__device__ __forceinline__ float warp_sumf(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    return __shfl_sync(0xFFFFFFFFu, v, 0);
}

/// The FP32 block-wide sum, for the reason the review's G5 states: this is a GeForce part and FP64 runs at a
/// small fraction of FP32, so a `double` reduction turns a bandwidth kernel into a latency kernel.
///
/// The scratch is taken as `double*` and reinterpreted as `float*` rather than changing the callers' shared
/// arrays, because `double scratch[8]` is 64 bytes and so is at least as large as the `float[8]` this needs -
/// the cast cannot overrun, and it keeps the change to one function instead of one per caller.
__device__ float block_sumf(float v, double* scratch_raw) {
    float* scratch = (float*) scratch_raw;
    // The leading barrier is not decoration - see the note on `block_sum` above: the result is read straight out
    // of `scratch[0]` by every thread and a later call reuses the array, so without it a fast thread can
    // overwrite `scratch[0]` before a slow one has read the previous result.
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sumf(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    const int nw = ((int) blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? scratch[threadIdx.x] : 0.0f;
    if (warp == 0) v = warp_sumf(v);
    if (threadIdx.x == 0) scratch[0] = v;
    __syncthreads();
    return scratch[0];
}

/// Block-wide sum in DOUBLE, broadcast to every thread.
///
/// Double because `ref/gr.py` reduces in float64 for the norm and the bottleneck, and because a block-wide
/// f32 sum of 2560 squares is a different number from the reference's; the shuffle tree here also reorders
/// the additions, so the only defence is enough precision that the order stops mattering.
///
/// The leading `__syncthreads()` is not decoration: the result is read straight out of `scratch[0]` by every
/// thread, and a later call reuses the same array.  Without a barrier on entry a fast thread can overwrite
/// `scratch[0]` before a slow one has read the previous result - a race that is invisible in most runs.
__device__ double block_sum(double v, double* scratch) {
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    const int nw = ((int) blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? scratch[threadIdx.x] : 0.0;
    if (warp == 0) v = warp_sum(v);
    if (threadIdx.x == 0) scratch[0] = v;
    __syncthreads();
    return scratch[0];
}

/// `hc` blocks, one per residual stream: per-stream RMSNorm, then the activated stream in both forms.
///
/// ggml_rms_norm reduces over ne[0], and ne[0] is n_embd, so the reduction is PER STREAM.  The `[n_embd, hc]`
/// gamma then scales each stream individually.
///
/// A per-thread accumulator is only a sum when the block is narrower than the row, which is why this is a
/// block reduction: with `n_embd` equal to the thread count every thread would hold exactly one element and
/// normalise by its own square.  That mistake is silent - it produces a plausible vector of the right shape -
/// and it is what `gr_parity`'s `mixed vs reference` caught at 2.0e+01 in round 196.
template <bool FP32_ACT>
__global__ void gr_norm_kernel(const float* __restrict__ R, const float* __restrict__ w_norm, float eps,
                               int n_embd, float* __restrict__ xn, uint16_t* __restrict__ xq) {
    __shared__ double scratch[8];
    const int c = blockIdx.x;
    const float* Rc = R + (size_t) c * n_embd;
    float* xnc = xn + (size_t) c * n_embd;
    uint16_t* xqc = xq + (size_t) c * n_embd;

    // FP32, for the same reason as `gr_down_kernel` and `gr_inject_kernel`: this is a GeForce part, FP64 runs
    // at a small fraction of FP32, and the sum of squares is a pure reduction.  It was the last `double` on the
    // `gr_read` path.  `gr_norm` is only 0.80 ms/token, so this is hygiene rather than a headline - but it is
    // 12 of the FP64 sites the review's G5 counted in this file, and the rule it states ("no FP64 on any decode
    // or prefill path") is worth being able to check mechanically rather than case by case.
    float ss = 0.0f;
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) {
        const float v = Rc[d];
        ss += v * v;
    }
    const float ms = block_sumf(ss, scratch) / (float) n_embd;
    const float rs = rsqrtf(ms + eps);
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) {
        const float x = Rc[d] * rs * w_norm[(size_t) c * n_embd + d];
        xnc[d] = x;
        if constexpr (!FP32_ACT) xqc[d] = f32_to_bf16_bits(x);
    }
}

/// `lo = silu((bf16(xn) @ w_down.T) / hc)`, ONE BLOCK PER OUTPUT ROW with its warps splitting the reduction.
template <typename Activation>
__global__ void gr_down_kernel(const Activation* __restrict__ xq, const uint16_t* __restrict__ w_down, int hc_dim,
                               int hc_lr, int hc, Activation* __restrict__ lq,
                               const uint16_t* __restrict__ w_inject, float* __restrict__ inject, bool do_inject) {
    const int k = blockIdx.x;
    if (do_inject && k == hc_lr) {
        // Track F: the inject projection is the down grid's last block.  One warp per stream, the same
        // lane stride and warp_sumf as `gr_inject_kernel`, so the four values are bitwise the old kernel's.
        const int c = threadIdx.x >> 5;
        if (c >= hc) return;
        const int lane = threadIdx.x & 31;
        const uint16_t* row = w_inject + (size_t) c * hc_dim;
        float acc = 0.0f;
        for (int i = lane; i < hc_dim; i += 32)
            acc += activation_f32(xq[i]) * bf16_bits_to_f32(row[i]);
        acc = warp_sumf(acc);
        if (lane == 0) inject[c] = acc;
        return;
    }
    if (k >= hc_lr) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nw = (int) (blockDim.x >> 5);          // warps per row == warps per block
    const uint16_t* row = w_down + (size_t) k * hc_dim;
    // **THE OCCUPANCY HERE IS 10% AND IT IS REAL, BUT FIXING IT DOES NOT PAY.**  `hc_lr` is 320, one warp per
    // row, 8 warps per block, so this launches 40 blocks = 320 warps: **6.7 warps per SM out of the 64 a
    // Blackwell SM holds**, each lane walking 10,240 elements in 320 dependent steps.
    //
    // Four independent accumulators were tried to break the dependency chain and put four loads in flight per
    // lane.  **Measured: 39.86 ms/token of pure GPU against 39.80, i.e. no change**, with `gr_parity` green
    // either way.  Removed, on the rule this file already states about `#pragma unroll 4`: a hint that does not
    // change the timing is a claim that is not true.
    //
    // The reason is that `gr_read`'s cost is not here.  Its four kernels are the norm (4 blocks), this (40),
    // `gr_gate_kernel` (1,280) and the mean (10), and the 1,280-block gate kernel reads 6.55 MB of weights with
    // good occupancy - so `gr_read`'s ~93 us is mostly the gate kernel at its own memory rate, and the ~165 GB/s
    // whole-call figure is an average over kernels that do not share a rate.
    // **FP32, AND THE FP64 THAT WAS HERE COST MORE THAN EVERYTHING ELSE IN `gr_read`.**
    //
    // This accumulated in `double`.  GeForce Blackwell executes FP64 at a small fraction of FP32 - 1/64 on
    // recent parts - so a 320-step dependent FP64 chain per lane turned a memory-bound reduction into a
    // latency-bound one.  The `nsys` per-kernel table (round 2) measured it at **5.07 ms/token, 15.3% of all
    // GPU time**, against `gr_gate_kernel`'s 1.23 ms - which REFUTES the comment that used to sit here, that
    // "gr_read's ~93 us is mostly the gate kernel at its own memory rate".  It was not: down + inject were
    // 8.77 of gr_read's 10.8 ms, and the gate was 1.23.
    //
    // The justification for `double` was matching `ref/gr.py`'s float64 transcription.  **The ground truth is
    // llama.cpp, and llama.cpp accumulates in FP32** - so this does not move away from the oracle, it moves
    // toward it, and the review's G1 lists it first.
    //
    // Four independent accumulators were tried here BEFORE, against the FP64 version, and measured no change
    // (39.86 vs 39.80 ms/token) - because FP64 throughput was the limit, not the dependency chain.  With FP32
    // the kernel becomes memory-bound: it reads ~13 MB per call across xq and w_down, which at the measured
    // 641 GB/s is ~20 us against the 52 us this took.
    // **THE OCCUPANCY CONCLUSION ABOVE IS STALE, AND THIS IS THE RETEST IT ASKED FOR.**
    //
    // "Fixing it does not pay" was measured against the FP64 version, where FP64 THROUGHPUT was the limit and
    // more parallelism could not help.  The kernel is FP32 now (round 282: 5.07 -> 2.38 ms/token) and the limit
    // moved to memory - which is the regime where occupancy does pay, because at 6.7 warps/SM there are not
    // enough bytes in flight to cover DRAM latency.  The old result is not wrong; it is about a different kernel.
    //
    // One BLOCK per row, its 8 warps splitting the 10,240-element reduction: 320 blocks x 8 = **2,560 warps =
    // 53 warps/SM instead of 6.7**, with no second kernel and no partial buffer - the eight partials meet in
    // shared memory.  Lanes within a warp still read consecutive addresses, because warp `w` starts at
    // `w * 32` and strides by `nw * 32`.
    float acc = 0.0f;
    for (int i = warp * 32 + lane; i < hc_dim; i += nw * 32)
        acc += activation_f32(xq[i]) * bf16_bits_to_f32(row[i]);
    acc = warp_sumf(acc);

    __shared__ float part[WARPS];
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < nw) ? part[lane] : 0.0f;
        t = warp_sumf(t);
        // NOTE: the reduction order differs from the one-warp version, so this is a NUMERICS CHANGE.  C1 is the
        // gate that judges it - not this comment, and not `gr_parity`, which compares against the reference on
        // a fixture that a changed summation order can still pass.
        if (lane == 0) store_activation(lq, k, silu_f(t / (float) hc));
    }
}

/// `gated[i] = xn[i] * sigmoid(bf16(lo) @ w_up.T)`, one warp per output row, lanes striding hc_lr.
template <typename Activation>
__global__ void gr_gate_kernel(const Activation* __restrict__ lq, const uint16_t* __restrict__ w_up,
                               const float* __restrict__ xn, int hc_dim, int hc_lr, float* __restrict__ gated) {
    const int i = blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (i >= hc_dim) return;
    const int lane = threadIdx.x & 31;
    const uint16_t* row = w_up + (size_t) i * hc_lr;
    float acc = 0.0f;
    for (int k = lane; k < hc_lr; k += 32)
        acc += activation_f32(lq[k]) * bf16_bits_to_f32(row[k]);
    acc = warp_sumf(acc);
    // Every lane computes the sigmoid (the shuffle broadcasts acc), so there is no second sync point and no
    // branch; lane 0 stores.
    if (lane == 0) gated[i] = xn[i] * sigmoid_f(acc);
}

/// `mixed[d] = mean over c of gated[c][d]`.  Flat elementwise: the mean is over the streams, which are
/// strided by n_embd, so no cross-thread reduction is needed at all.
__global__ void gr_mean_kernel(const float* __restrict__ gated, int n_embd, int hc, float* __restrict__ mixed) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    float m = 0.0f;
    for (int c = 0; c < hc; ++c) m += gated[(size_t) c * n_embd + d];
    mixed[d] = m / (float) hc;
}

/// Track F: `gr_gate_kernel` and `gr_mean_kernel` in one launch.  One block owns WARPS columns for every
/// stream; a warp computes one column's dot per stream (the same lane stride and warp_sumf as the old gate),
/// the per-stream values meet in shared, and the mean adds them in the same c order as the old kernel.
template <typename Activation>
__global__ void gr_gate_mean_kernel(const Activation* __restrict__ lq, const uint16_t* __restrict__ w_up,
                                    const float* __restrict__ xn, int hc_lr, int n_embd, int hc,
                                    float* __restrict__ mixed) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int d = blockIdx.x * WARPS + warp;
    __shared__ float g[4][WARPS];
    if (d < n_embd) {
        for (int c = 0; c < hc; ++c) {
            const int i = c * n_embd + d;
            const uint16_t* row = w_up + (size_t) i * hc_lr;
            float acc = 0.0f;
            for (int k = lane; k < hc_lr; k += 32)
                acc += activation_f32(lq[k]) * bf16_bits_to_f32(row[k]);
            acc = warp_sumf(acc);
            if (lane == 0) g[c][warp] = xn[i] * sigmoid_f(acc);
        }
    }
    __syncthreads();
    if (threadIdx.x < WARPS) {
        const int dd = blockIdx.x * WARPS + threadIdx.x;
        if (dd < n_embd) {
            float m = 0.0f;
            for (int c = 0; c < hc; ++c) m += g[c][threadIdx.x];
            mixed[dd] = m / (float) hc;
        }
    }
}

/// `inject[c] = bf16(xn) @ w_inject[c]`, one value per stream: ONE WARP PER STREAM, lanes striding hc_dim.
///
/// This mirrors `gr_down_kernel` rather than using a block-wide reduction.  `hc` is 4 at the real geometry,
/// so a block per stream would put 256 threads on a 10240-element dot and leave the reduction to do all the
/// work; a warp per stream is the same shape as the down projection and reuses its pattern exactly.  The
/// whole kernel is one block of `32*hc` threads, which is small - and it is a 4-value output, so it is.
template <typename Activation>
__global__ void gr_inject_kernel(const Activation* __restrict__ xq, const uint16_t* __restrict__ w_inject,
                                 int hc_dim, int hc, float* __restrict__ inject) {
    const int c = threadIdx.x >> 5;
    if (c >= hc) return;
    const int lane = threadIdx.x & 31;
    const uint16_t* row = w_inject + (size_t) c * hc_dim;
    // FP32, for the same reason as `gr_down_kernel` - and this one is worse.  A ONE-BLOCK kernel doing four
    // 10,240-element FP64 reductions measured **3.70 ms/token, 11.1% of all GPU time, from a single SM out of
    // 48.**  It is the fourth largest consumer of GPU time in the engine and it cannot use more than 1/48th of
    // the machine.  In FP32 the same reduction is a few microseconds and the block size stops mattering.
    float acc = 0.0f;
    for (int i = lane; i < hc_dim; i += 32)
        acc += activation_f32(xq[i]) * bf16_bits_to_f32(row[i]);
    acc = warp_sumf(acc);
    if (lane == 0) inject[c] = acc;
}

__global__ void gr_write_kernel(const float* __restrict__ R, const float* __restrict__ block_out,
                                const float* __restrict__ inject, int n_embd, int hc, float* __restrict__ out) {
    // w = 2*sigmoid(inject/hc), computed once per stream in shared rather than per element.
    extern __shared__ unsigned char smem_raw[];
    float* w = (float*) smem_raw;
    if ((int) threadIdx.x < hc) w[threadIdx.x] = 2.0f * sigmoid_f(inject[threadIdx.x] / (float) hc);
    __syncthreads();
    const long long n = (long long) hc * n_embd;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x) {
        const int c = (int) (i / n_embd), d = (int) (i % n_embd);
        // every stream adds the SAME block output; only the weight differs per stream
        out[i] = R[i] + block_out[d] * w[c];
    }
}

}  // namespace

void gr_set_fp32_activations(bool enabled) { fp32_activations = enabled; }
void gr_set_native_mmvf(bool enabled) { native_mmvf = enabled; }
bool gr_native_mmvf_enabled() { return native_mmvf; }

size_t gr_workspace_init(const GrShapes& s, void* base, GrWorkspace& out) {
    const size_t hc_dim = (size_t) s.hc * (size_t) s.n_embd;
    // ONE table, indexed by the same `k` that assigns the pointers below.  The first version of this function
    // sized three locals named `xn`, `xq`, `lq` in a DIFFERENT order from the pointer assignments - `xq` got
    // hc_lr*2 while `lq` got hc_dim*4 - so for the real geometry xq was given 640 bytes where it needed
    // 20,480 and every region overlapped its neighbour.
    //
    // What that looked like is worth recording: `inject` came out ~20% wrong, and rewriting its reduction
    // from block-wide to warp-wide produced BYTE-IDENTICAL wrong answers.  Two different reductions agreeing
    // exactly is not a coincidence to investigate - it means the reduction is not where the bug is, and the
    // inputs are.  Chasing the reduction instead cost most of a round.
    const size_t sz[5] = {
        hc_dim * sizeof(float),                // 0: xn
        hc_dim * sizeof(uint16_t),             // 1: xq
        (size_t) s.hc_lr * sizeof(uint16_t),   // 2: lq
        hc_dim * sizeof(float),                // 3: gated
        (size_t) s.hc_lr * sizeof(float),        // 4: lo (FP32 activation experiment)
    };
    size_t al[5], bytes = 0;
    for (int k = 0; k < 5; ++k) {
        al[k] = (sz[k] + 15) & ~(size_t) 15;
        bytes += al[k];
    }
    out.bytes = bytes;
    if (base != nullptr) {
        unsigned char* p = (unsigned char*) base;
        void* ptr[5];
        for (int k = 0; k < 5; ++k) {
            ptr[k] = p;
            p += al[k];
        }
        out.xn = (float*) ptr[0];
        out.xq = (uint16_t*) ptr[1];
        out.lq = (uint16_t*) ptr[2];
        out.gated = (float*) ptr[3];
        out.lo = (float*) ptr[4];
    }
    return bytes;
}

void gr_read(const float* R, const float* w_norm, const uint16_t* w_down, const uint16_t* w_up,
             const uint16_t* w_inject, float eps, const GrShapes& s, const GrWorkspace& ws, float* mixed,
             float* inject, void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0 || s.hc_lr <= 0) return;
    if (ws.xn == nullptr || ws.xq == nullptr || ws.lq == nullptr || ws.gated == nullptr || ws.lo == nullptr) {
        std::fprintf(stderr, "gr_read: GrWorkspace is not initialised (see gr_workspace_init)\n");
        std::exit(1);
    }
    if (ws.bytes < gr_workspace_bytes(s)) {
        std::fprintf(stderr, "gr_read: GrWorkspace is %zu bytes but this geometry needs %zu\n",
                     ws.bytes, gr_workspace_bytes(s));
        std::exit(1);
    }
    const int n_embd = (int) s.n_embd, hc = (int) s.hc, hc_lr = (int) s.hc_lr;
    const int hc_dim = (int) (s.hc * s.n_embd);
    cudaStream_t st = (cudaStream_t) stream;

    // One setting selects every projection in this call. Captured graphs retain these kernel variants.
    const bool use_native = native_mmvf;
    const bool use_fp32 = fp32_activations || use_native;
    const bool fused_gm = hc <= 4;   // gr_gate_mean_kernel's shared array holds hc <= 4 streams
    if (use_native) {
        if ((hc_dim & 1) != 0 || (hc_lr & 1) != 0)
            throw std::invalid_argument("gr_read native MMVF requires even hc*n_embd and hc_lr");
        native_gr_rms_norm_weighted(R, w_norm, ws.xn, n_embd, hc, eps, stream);
        bf16_gemv_fp32_mmvf(ws.xn, w_down, ws.lo, hc_dim, hc_lr, stream);
        native_gr_down_silu(ws.lo, hc_lr, hc, stream);
        bf16_gemv_fp32_mmvf(ws.lo, w_up, ws.gated, hc_lr, hc_dim, stream);
        // The existing null-injection contract identifies the final mixer.
        // Pinned qwen4exp uses fused HC pre for layers and the unfused graph for il=-1.
        native_gr_pre_gated(ws.xn, ws.gated, mixed, n_embd, hc, w_inject != nullptr, stream);
    } else if (use_fp32) {
        gr_norm_kernel<true><<<hc, THREADS, 0, st>>>(R, w_norm, eps, n_embd, ws.xn, ws.xq);
        gr_down_kernel<float><<<hc_lr + (w_inject != nullptr ? 1 : 0), THREADS, 0, st>>>(
            ws.xn, w_down, hc_dim, hc_lr, hc, ws.lo, w_inject, inject, w_inject != nullptr);
        if (fused_gm)
            gr_gate_mean_kernel<float><<<(n_embd + WARPS - 1) / WARPS, THREADS, 0, st>>>(
                ws.lo, w_up, ws.xn, hc_lr, n_embd, hc, mixed);
        else
            gr_gate_kernel<float><<<(hc_dim + WARPS - 1) / WARPS, THREADS, 0, st>>>(
                ws.lo, w_up, ws.xn, hc_dim, hc_lr, ws.gated);
    } else {
        gr_norm_kernel<false><<<hc, THREADS, 0, st>>>(R, w_norm, eps, n_embd, ws.xn, ws.xq);
        gr_down_kernel<uint16_t><<<hc_lr + (w_inject != nullptr ? 1 : 0), THREADS, 0, st>>>(
            ws.xq, w_down, hc_dim, hc_lr, hc, ws.lq, w_inject, inject, w_inject != nullptr);
        if (fused_gm)
            gr_gate_mean_kernel<uint16_t><<<(n_embd + WARPS - 1) / WARPS, THREADS, 0, st>>>(
                ws.lq, w_up, ws.xn, hc_lr, n_embd, hc, mixed);
        else
            gr_gate_kernel<uint16_t><<<(hc_dim + WARPS - 1) / WARPS, THREADS, 0, st>>>(
                ws.lq, w_up, ws.xn, hc_dim, hc_lr, ws.gated);
    }
    if (!use_native && !fused_gm)
        gr_mean_kernel<<<(n_embd + THREADS - 1) / THREADS, THREADS, 0, st>>>(ws.gated, n_embd, hc, mixed);
    // Absent for the final mixer, and then nothing is written - computing an injection nobody reads would be
    // a claim, not a convenience.  The non-native calls compute it in the down grid's last block (Track F).
    if (w_inject != nullptr && use_native)
        bf16_gemv_fp32_mmvf(ws.xn, w_inject, inject, hc_dim, hc, stream);

    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "gr_read launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) {
        const cudaError_t se = cudaDeviceSynchronize();
        if (se != cudaSuccess) {
            std::fprintf(stderr, "gr_read: %s\n", cudaGetErrorString(se));
            std::exit(1);
        }
    }
}

static bool no_multi_gr() {
    static const bool off = [] {
        const char* v = std::getenv("STRATA_NO_MULTI_GR");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return off;
}

void gr_read_multi(const float* R, const float* w_norm, const uint16_t* w_down, const uint16_t* w_up,
                   const uint16_t* w_inject, float eps, const GrShapes& s, const GrWorkspace& ws,
                   float* xn_multi, float* lo_multi, float* gated_multi, float* mixed, float* inject,
                   int n_tok, void* stream) {
    if (n_tok <= 0 || s.n_embd <= 0 || s.hc <= 0 || s.hc_lr <= 0) return;
    if (!no_multi_gr() && native_mmvf && n_tok > 1 && xn_multi != nullptr && lo_multi != nullptr && gated_multi != nullptr) {
        const int n_embd = (int) s.n_embd, hc = (int) s.hc, hc_lr = (int) s.hc_lr;
        const int hc_dim = (int) (s.hc * s.n_embd);
        if ((hc_dim & 1) != 0 || (hc_lr & 1) != 0)
            throw std::invalid_argument("gr_read native MMVF requires even hc*n_embd and hc_lr");
        native_gr_rms_norm_weighted_multi(R, w_norm, xn_multi, n_embd, hc, n_tok, eps, stream);
        bf16_gemv_fp32_mmvf_multi(xn_multi, hc_dim, w_down, lo_multi, hc_lr, hc_dim, hc_lr, n_tok, stream);
        native_gr_down_silu(lo_multi, hc_lr * n_tok, hc, stream);
        bf16_gemv_fp32_mmvf_multi(lo_multi, hc_lr, w_up, gated_multi, hc_dim, hc_lr, hc_dim, n_tok, stream);
        native_gr_pre_gated_multi(xn_multi, gated_multi, mixed, n_embd, hc, n_tok, w_inject != nullptr, stream);
        if (w_inject != nullptr)
            bf16_gemv_fp32_mmvf_multi(xn_multi, hc_dim, w_inject, inject, hc, hc_dim, hc, n_tok, stream);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "gr_read_multi launch: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
        if (stream == nullptr) {
            const cudaError_t se = cudaDeviceSynchronize();
            if (se != cudaSuccess) {
                std::fprintf(stderr, "gr_read_multi: %s\n", cudaGetErrorString(se));
                std::exit(1);
            }
        }
        return;
    }
    const size_t hc_dim = (size_t) s.hc * (size_t) s.n_embd;
    for (int t = 0; t < n_tok; ++t) {
        gr_read(R + (size_t) t * hc_dim, w_norm, w_down, w_up, w_inject, eps, s, ws,
                mixed + (size_t) t * s.n_embd, inject ? inject + (size_t) t * s.hc : nullptr, stream);
    }
}

void gr_write(const float* R, const float* block_out, const float* inject, const GrShapes& s, float* R_out,
              void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0) return;
    const long long n = (long long) s.hc * s.n_embd;
    const int blocks = (int) ((n + THREADS - 1) / THREADS);
    if (native_mmvf)
        native_gr_post(R, block_out, inject, R_out, (int) s.n_embd, (int) s.hc, stream);
    else
        gr_write_kernel<<<blocks, THREADS, (size_t) s.hc * sizeof(float), (cudaStream_t) stream>>>(
            R, block_out, inject, (int) s.n_embd, (int) s.hc, R_out);
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "gr_write: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

void gr_write_multi(const float* R, const float* block_out, const float* inject, const GrShapes& s, float* R_out,
                    int n_tok, void* stream) {
    if (n_tok <= 0 || s.n_embd <= 0 || s.hc <= 0) return;
    if (!no_multi_gr() && native_mmvf && n_tok > 1) {
        native_gr_post_multi(R, block_out, inject, R_out, (int) s.n_embd, (int) s.hc, n_tok, stream);
        if (stream == nullptr) {
            const cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) {
                std::fprintf(stderr, "gr_write_multi: %s\n", cudaGetErrorString(e));
                std::exit(1);
            }
        }
        return;
    }
    const size_t hc_dim = (size_t) s.hc * (size_t) s.n_embd;
    for (int t = 0; t < n_tok; ++t) {
        gr_write(R + (size_t) t * hc_dim, block_out + (size_t) t * s.n_embd, inject + (size_t) t * s.hc, s,
                 R_out + (size_t) t * hc_dim, stream);
    }
}

}  // namespace strata::kernels
