// src/kernels/cuda/ple.cu - P2.S4: the PLE block's GPU half.
//
// See include/strata/kernels/ple.hpp for the structure and for the `normalized`-is-the-conv-input finding.
//
// The legacy arithmetic below follows the captured CPU ggml graph
// (`bench/micro/ple_in.bin` / `ple_out.bin`, produced by `ple_layer_xcheck.cpp`):
//
//   * `rms_norm` accumulates `(double)(x*x)` with the PRODUCT rounded in f32 first - that is literally what
//     `ggml_compute_forward_rms_norm_f32` does (`sum += (ggml_float)(x[i00]*x[i00])` with `ggml_float` =
//     double), and it is not the same as accumulating in f32 or as widening before the multiply.
//   * `silu` is `x / (1 + expf(-x))` in f32, which is `ggml_silu_f32`.
//   * the gate's `sqrt` and `sigmoid` are f32.
//
// The opt-in native BF16 path replaces only ple_value with the pinned CUDA single-token BF16/F32 MMVF.
// A separate opt-in native postops path follows the pinned CUDA arithmetic after both projections.
#include "strata/kernels/ple.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_ple_postops.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
bool native_bf16 = false;
bool native_postops = false;

bool overlap(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
    if (a == nullptr || b == nullptr || a_bytes == 0 || b_bytes == 0) return false;
    const uintptr_t aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
    return aa < bb ? bb - aa < a_bytes : aa - bb < b_bytes;
}

__global__ void history_advance_kernel(float* __restrict__ history, const float* __restrict__ normalized) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= NG_HC_DIM) return;
    float* column = history + (size_t) channel * NG_HIST;
    for (int row = 0; row + 1 < NG_HIST; ++row) column[row] = column[row + 1];
    column[NG_HIST - 1] = normalized[channel];
}

__device__ __forceinline__ uint16_t bf16_bits(float f) {
    uint32_t i;
    memcpy(&i, &f, 4);
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return (uint16_t) (i >> 16);
}

__device__ __forceinline__ float bf16_float(uint16_t h) {
    const uint32_t i = (uint32_t) h << 16;
    float f;
    memcpy(&f, &i, 4);
    return f;
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

/// Block-wide sum in double, broadcast to all threads.  The leading `__syncthreads` matters: the result is
/// read out of `scratch[0]` and a later call reuses the array, so without a barrier a fast thread can
/// overwrite it before a slow one has read it - invisible in most runs and a slightly different norm when it
/// fires.
__device__ double block_sum(double v, double* scratch) {
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    const int nw = ((int) blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? scratch[threadIdx.x] : 0.0;
    if (warp == 0)
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    if (threadIdx.x == 0) scratch[0] = v;
    __syncthreads();
    return scratch[0];
}

/// `grouped_norm`, ONE BLOCK PER STREAM.  The ggml form is `reshape_3d(x, n_embd, hc, T)` then
/// `rms_norm(..., eps) * w`, so ne0 = n_embd is the reduction axis and each hc stream is normalised on its
/// own.  In the flat ggml layout (ne0 fastest) stream c occupies [c*n_embd, (c+1)*n_embd).
__global__ void gnorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                             float* __restrict__ y, int n_embd, float eps) {
    __shared__ double scratch[8];
    const int c = blockIdx.x;
    const float* xc = x + (size_t) c * n_embd;
    const float* wc = w + (size_t) c * n_embd;
    float* yc = y + (size_t) c * n_embd;

    double acc = 0.0;
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) {
        // The PRODUCT is rounded in f32 before it is widened - `(ggml_float)(x[i00]*x[i00])`.
        const float sq = xc[d] * xc[d];
        acc += (double) sq;
    }
    const float mean = (float) (block_sum(acc, scratch) / (double) n_embd);
    const float scale = 1.0f / sqrtf(mean + eps);
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) yc[d] = xc[d] * scale * wc[d];
}

/// `s[c] = sum_d key[c][d]*query[c][d] / sqrt(n_embd)`, then the signed square root and the sigmoid.
///
/// `mag = sqrt(clamp(|s|, 1e-6, inf))` and `gate = sigmoid(sgn(s) * mag)`.  The clamp is a floor on |s|, so
/// it only bites near zero; the SIGN is carried separately, which is what keeps the gate symmetric about 0.5.
__global__ void gate_kernel(const float* __restrict__ key, const float* __restrict__ query,
                            float* __restrict__ gate, int n_embd, float inv_sqrt_n) {
    __shared__ double scratch[8];
    const int c = blockIdx.x;
    const float* kc = key + (size_t) c * n_embd;
    const float* qc = query + (size_t) c * n_embd;
    double acc = 0.0;
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) acc += (double) (kc[d] * qc[d]);
    const float s = (float) (block_sum(acc, scratch) / (double) 1.0) * inv_sqrt_n;
    const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
    const float sgn = (s > 0.0f) ? 1.0f : ((s < 0.0f) ? -1.0f : 0.0f);
    if (threadIdx.x == 0) gate[c] = 1.0f / (1.0f + expf(-(sgn * mag)));
}

/// `gated[c][d] = value[d] * gate[c]` - the value broadcast across the hc streams.
__global__ void bcast_kernel(const float* __restrict__ value, const float* __restrict__ gate,
                             float* __restrict__ gated, int n_embd, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd * hc) return;
    gated[i] = value[i % n_embd] * gate[i / n_embd];
}

/// The depthwise causal dilated conv, then SiLU.  One thread per channel.
///
/// `out[c] = sum_k kW[k][c] * terms[t - (kern-1-k)*dil][c]` with `t = hist` (the first new position), so tap k
/// reads row `hist - (kern-1-k)*dil` of `[history | new]`.  For the real geometry (kern 4, dil 3, hist 9) the
/// four taps read rows 0, 3, 6 and 9 - and row 9 is the NEW row, which is the only one beyond the history.
/// That is why no `terms` buffer is built: three taps come from the caller's history and one from `norm`.
///
/// `kW` IS GGML-NATIVE: `kW[k + kern*c]`.  The manifest's shape is [4, 10240] with ne0 = 4 fast, and
/// `ref/ngram.py`'s (4, 10240) numpy `kernel[k][c]` is the TRANSPOSE of that.  The source's own view
/// (`ggml_view_2d(model.layers[il].ple_conv1d, 1, hc_dim, nb[1], k*nb[0])`) settles which one is meant.
__global__ void conv_kernel(const float* __restrict__ hist, const float* __restrict__ norm,
                            const uint16_t* __restrict__ kW, float* __restrict__ out, int hc_dim, int kern,
                            int dil, int nhist) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hc_dim) return;
    float acc = 0.0f;
    for (int k = 0; k < kern; ++k) {
        const int row = nhist - (kern - 1 - k) * dil;      // tap 0 reads the FURTHEST back
        // ROW-FASTEST: `hist[row + nhist*c]`.  The conv state is `ggml_reshape_3d(state, d_conv-1,
        // conv_channels, n_seqs)`, so ne0 = hist varies fastest and the flat index is row + hist*channel.
        // Channel-slowest (`row*hc_dim + c`) is the natural thing to write and would read a transposed state -
        // `ple_layer_xcheck.cpp` L79-83 records getting this wrong once already.
        const float v = (row == nhist) ? norm[c] : hist[(size_t) row + (size_t) nhist * c];
        acc += f32_from_f16(kW[k + kern * c]) * v;
    }
    out[c] = silu_f(acc);
}

/// `result = hidden + gated + conv`, elementwise over hc_dim.
__global__ void add3_kernel(const float* __restrict__ hidden, const float* __restrict__ gated,
                            const float* __restrict__ conv, float* __restrict__ result, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) result[i] = hidden[i] + gated[i] + conv[i];
}

/// `y[o] = sum_i bf16(x[i]) * bf16(w[o*n_in + i])`.  The weight is BF16 and so is the ACTIVATION, which is
/// the contract for a BF16 tensor (`docs/activation-contract.md`); with both sides bf16 the products are
/// exact in f32 and only the summation order differs from ggml's.
__global__ void bf16_gemv_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                 float* __restrict__ y, int n_in, int n_out) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    const uint16_t* row = w + (size_t) o * n_in;
    double acc = 0.0;
    for (int i = 0; i < n_in; ++i) acc += (double) bf16_float(x[i]) * (double) bf16_float(row[i]);
    y[o] = (float) acc;
}

__global__ void to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = bf16_bits(x[i]);
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ple_block: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

void ple_set_native_bf16(bool enabled) { native_bf16 = enabled; }
bool ple_native_bf16_enabled() { return native_bf16; }
void ple_set_native_postops(bool enabled) { native_postops = enabled; }
bool ple_native_postops_enabled() { return native_postops; }

void ple_history_advance(float* hist, const float* normalized, void* stream) {
    if (hist == nullptr || normalized == nullptr)
        throw std::invalid_argument("ple_history_advance: null history or normalized input");
    if (overlap(hist, (size_t) NG_HIST * NG_HC_DIM * sizeof(float),
                normalized, (size_t) NG_HC_DIM * sizeof(float)))
        throw std::invalid_argument("ple_history_advance: history and normalized input overlap");
    history_advance_kernel<<<(NG_HC_DIM + THREADS - 1) / THREADS, THREADS, 0, (cudaStream_t) stream>>>(hist, normalized);
    ck(cudaGetLastError(), "history advance launch");
}

bool ple_block_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

uint64_t ple_block_scratch_bytes() {
    // five hc_dim floats + n_embd + hc, then the Q8 activation image, then the BF16 embedding copy, each
    // 16-byte aligned because `d_scratch` is cast to `float*` and `d_act` to `uint8_t*` at those offsets.
    const size_t f = (size_t) (5 * NG_HC_DIM + NG_N_EMBD + NG_HC) * sizeof(float);
    const size_t q = (size_t) (NG_N_EMBD / 32) * 34;
    const size_t e = (size_t) NG_N_EMBD * sizeof(uint16_t);
    return ((f + 15) & ~(size_t) 15) + ((q + 15) & ~(size_t) 15) + e + 256;
}

void ple_block(const float* emb, const float* hidden, const float* hist_rows, const PleWeights& w,
               PleOut& out, void* scratch, void* stream) {
    const bool native_key = w.key_native_data != nullptr && w.key_bf16 == nullptr;
    if (native_key && (!emb || !hidden || !hist_rows || !out.result || !scratch || !stream ||
                       !w.key_native_q8_1 ||
                       (w.key_native_type != 42 && w.key_native_type != 18 && w.key_native_type != 23 &&
                        w.key_native_type != 8)))
        throw std::invalid_argument("ple_block: native key requires Q2_0, IQ3_XXS, IQ4_XS or Q8_0 weights, input/output, private scratch and explicit stream");
    if (emb == nullptr || hidden == nullptr || hist_rows == nullptr || out.result == nullptr) return;
    const int n_embd = NG_N_EMBD, hc = NG_HC, hc_dim = NG_HC_DIM;
    static_assert(NG_N_EMBD == 2560 && NG_HC_DIM == 10240, "native PLE key geometry changed");
    const size_t float_bytes = (size_t) (5 * hc_dim + n_embd + hc) * sizeof(float);
    cudaStream_t st = (cudaStream_t) stream;

    // One allocation for every intermediate.
    //
    // THE ENGINE WILL NOT DO THIS.  P2.T10 requires zero token-path allocations, and the fix is the same one
    // `gr_read` already uses: a caller-owned workspace carved once at startup.  It is done this way here
    // because this function's job in P2.S4 is to be CORRECT and comparable against the oracle, and matching
    // `shared_expert.cu`'s existing shape keeps the diff small.  Recorded rather than hidden.
    //
    // FIVE SEPARATE hc_dim BUFFERS, deliberately.  `key` and `normalized` are both needed at the end for the
    // oracle comparison, so aliasing one onto the other - which the first version did - silently returns the
    // WRONG `key`.  Two buffers of the same size look like an obvious saving and the only thing it saves is
    // 40 KB.
    // **THE WORKSPACE IS CARVED FROM THE CALLER'S, NOT ALLOCATED.**  Three `cudaMalloc`s and a
    // `cudaStreamSynchronize` used to live here, and inside a stream capture both are errors: the engine
    // reported `ple_block: scratch: operation not permitted when stream is capturing`, and a token-path
    // allocation is a P2.T10 violation besides.  The comment this replaces already said the right shape was
    // "a caller-owned workspace carved once at startup"; this is that.
    const size_t q8_bytes = (size_t) (n_embd / 32) * 34;
    if (scratch == nullptr) {
        std::fprintf(stderr, "ple_block: scratch is null; the caller owns it (see ple_block_scratch_bytes)\n");
        std::exit(1);
    }
    const struct Export { const float* pointer; size_t count; const char* name; } exports[] = {
        {out.key, (size_t) hc_dim, "key"}, {out.value, (size_t) n_embd, "value"},
        {out.gate, (size_t) hc, "gate"}, {out.gated, (size_t) hc_dim, "gated"},
        {out.normalized, (size_t) hc_dim, "normalized"}, {out.conv, (size_t) hc_dim, "conv"},
        {out.result, (size_t) hc_dim, "result"}
    };
    for (const auto& item : exports) {
        if (overlap(item.pointer, item.count * sizeof(float), scratch, (size_t) ple_block_scratch_bytes()))
            throw std::invalid_argument(std::string("ple_block: output ") + item.name + " overlaps scratch");
    }
    if (native_key) {
        const size_t native_bytes = native_q8_1_bytes(n_embd);
        const size_t weight_bytes = native_mmvq_weight_bytes(w.key_native_type, n_embd, hc_dim);
        if ((reinterpret_cast<uintptr_t>(w.key_native_data) & 3u) ||
            (reinterpret_cast<uintptr_t>(w.key_native_q8_1) & 3u))
            throw std::invalid_argument("ple_block: native key buffers require four-byte alignment");
        const struct Region { const void* pointer; size_t bytes; } regions[] = {
            {scratch, (size_t) ple_block_scratch_bytes()}, {emb, (size_t) n_embd * 4},
            {hidden, (size_t) hc_dim * 4}, {hist_rows, (size_t) NG_HIST * hc_dim * 4},
            {w.key_native_data, weight_bytes}, {w.value_bf16, (size_t) n_embd * n_embd * 2},
            {w.norm_key, (size_t) hc_dim * 4}, {w.norm_query, (size_t) hc_dim * 4},
            {w.norm_conv, (size_t) hc_dim * 4}, {w.conv1d_f16, (size_t) PLE_CONV_KERNEL * hc_dim * 2}
        };
        for (const auto& region : regions)
            if (overlap(w.key_native_q8_1, native_bytes, region.pointer, region.bytes))
                throw std::invalid_argument("ple_block: native key scratch overlaps workspace, input or weight");
        for (const auto& item : exports)
            if (overlap(w.key_native_q8_1, native_bytes, item.pointer, item.count * sizeof(float)))
                throw std::invalid_argument(std::string("ple_block: native key scratch overlaps output ") + item.name);
    }
    uint8_t* base = (uint8_t*) scratch;
    float* d_scratch = (float*) base;
    uint8_t* d_act = base + ((float_bytes + 15) & ~(size_t) 15);
    uint16_t* d_emb16 = (uint16_t*) (d_act + ((q8_bytes + 15) & ~(size_t) 15));
    float* d_key = d_scratch;
    float* d_query = d_key + hc_dim;
    float* d_norm = d_query + hc_dim;
    float* d_gated = d_norm + hc_dim;
    float* d_conv = d_gated + hc_dim;
    float* d_value = d_conv + hc_dim;
    float* d_gate = d_value + n_embd;

    // ---- key = grouped_norm(ple_key @ emb). The optional native projection
    // follows pinned CUDA Q8_1 MMVQ; the default retains its canonical Q8_0 path.
    if (w.pre_key != nullptr) {
        cudaMemcpyAsync(d_key, w.pre_key, (size_t) hc_dim * sizeof(float), cudaMemcpyDeviceToDevice, st);
    } else if (w.key_bf16 != nullptr) {
        bf16_gemv_fp32_mmvf(emb, w.key_bf16, d_key, n_embd, hc_dim, stream);
    } else if (native_key) {
        native_quantize_q8_1(emb, w.key_native_q8_1, n_embd, 1, stream);
        native_mmvq(w.key_native_type, w.key_native_data, w.key_native_q8_1,
                    d_key, n_embd, hc_dim, 1, stream);
    } else {
        quantize_q8_0(emb, d_act, n_embd, st);
        s2_gemv_q8(d_act, w.key_codes, w.key_scales, d_key, n_embd, hc_dim, 8, st);
    }
    if (!native_postops) {
        gnorm_kernel<<<hc, THREADS, 0, st>>>(d_key, w.norm_key, d_key, n_embd, NG_RMS_EPS);
        gnorm_kernel<<<hc, THREADS, 0, st>>>(hidden, w.norm_query, d_query, n_embd, NG_RMS_EPS);
    }

    // The value projection's independent option leaves the nonlinear PLE operations unchanged.
    if (w.pre_value != nullptr) {
        cudaMemcpyAsync(d_value, w.pre_value, (size_t) n_embd * sizeof(float), cudaMemcpyDeviceToDevice, st);
    } else if (native_bf16) {
        bf16_gemv_fp32_mmvf(emb, w.value_bf16, d_value, n_embd, n_embd, stream);
    } else {
        to_bf16_kernel<<<(n_embd + THREADS - 1) / THREADS, THREADS, 0, st>>>(emb, d_emb16, n_embd);
        bf16_gemv_kernel<<<(n_embd + THREADS - 1) / THREADS, THREADS, 0, st>>>(d_emb16, w.value_bf16, d_value,
                                                                              n_embd, n_embd);
    }

    const float* normalized_key = d_key;
    if (native_postops) {
        // The key norm has a distinct destination; temporary query storage can
        // be reused for normalized gated values after the gate consumes it.
        NativePlePostopsBuffers buffers{d_query,d_norm,d_gate,d_gated,d_norm,d_conv,out.result};
        native_ple_postops(d_key,hidden,d_value,hist_rows,w,buffers,stream);
        normalized_key = d_query;
    } else {
        gate_kernel<<<hc, THREADS, 0, st>>>(d_key, d_query, d_gate, n_embd, 1.0f / sqrtf((float) n_embd));
        bcast_kernel<<<(hc_dim + THREADS - 1) / THREADS, THREADS, 0, st>>>(d_value, d_gate, d_gated, n_embd, hc);
        gnorm_kernel<<<hc, THREADS, 0, st>>>(d_gated, w.norm_conv, d_norm, n_embd, NG_RMS_EPS);
        conv_kernel<<<(hc_dim + THREADS - 1) / THREADS, THREADS, 0, st>>>(hist_rows, d_norm, w.conv1d_f16, d_conv,
                                                                         hc_dim, PLE_CONV_KERNEL, NGRAM_SIZE,
                                                                         NG_HIST);
        add3_kernel<<<(hc_dim + THREADS - 1) / THREADS, THREADS, 0, st>>>(hidden, d_gated, d_conv, out.result,
                                                                         hc_dim);
    }

    // ---- the intermediates the oracle comparison needs.  `key` is the NORMALISED key, because the source's
    //      `cb(key, ...)` capture is after `gnorm`; `value` is the projection before the gate.  Each stage the
    //      oracle records is reproduced here so a mismatch can be attributed instead of guessed at.
    if (out.key) ck(cudaMemcpyAsync(out.key, normalized_key, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice, st), "key");
    if (out.value)
        ck(cudaMemcpyAsync(out.value, d_value, n_embd * sizeof(float), cudaMemcpyDeviceToDevice, st), "value");
    if (out.gate) ck(cudaMemcpyAsync(out.gate, d_gate, hc * sizeof(float), cudaMemcpyDeviceToDevice, st), "gate");
    if (out.gated)
        ck(cudaMemcpyAsync(out.gated, d_gated, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice, st), "gated");
    if (out.normalized)
        ck(cudaMemcpyAsync(out.normalized, d_norm, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice, st), "norm");
    if (out.conv)
        ck(cudaMemcpyAsync(out.conv, d_conv, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice, st), "conv");

    ck(cudaGetLastError(), "launch");
    // **NO `cudaStreamSynchronize` HERE.**  It was there to make the function self-contained for the parity
    // test, and inside a capture it is an error - a caller that wants the result immediately synchronises
    // itself, and the engine's caller does not want that at all.
}

void ple_block_projected(const float* projected_key, const float* projected_value, const float* hidden,
                         const float* hist_rows, const PleWeights& w, PleOut& out, void* scratch, void* stream) {
    if (!projected_key || !projected_value || !hidden || !hist_rows || !out.result || !scratch) return;
    const int n_embd = NG_N_EMBD, hc = NG_HC, hc_dim = NG_HC_DIM;
    float* d_scratch = (float*) scratch;
    float* d_key = d_scratch;
    float* d_query = d_key + hc_dim;
    float* d_norm = d_query + hc_dim;
    float* d_gated = d_norm + hc_dim;
    float* d_conv = d_gated + hc_dim;
    float* d_value = d_conv + hc_dim;
    float* d_gate = d_value + n_embd;
    float* norm_dst = out.normalized ? out.normalized : d_norm;
    NativePlePostopsBuffers buffers{d_query, norm_dst, d_gate, d_gated, norm_dst, d_conv, out.result};
    native_ple_postops(projected_key, hidden, projected_value, hist_rows, w, buffers, stream);
}

namespace {
__constant__ int8_t kPleIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
// one thread a value: row r, block b (18 bytes: fp16 d, 16 bytes of codes), element j of 32 (j < 16: low nibble of
// qs[j], else the high nibble of qs[j - 16] - the split halves of iq4nl_dequant_row)
__global__ void ple_iq4nl_dequant_kernel(const uint8_t* __restrict__ raw, int64_t n, float* __restrict__ out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / PLE_HEAD_DIM;
    const int e = (int) (i - r * PLE_HEAD_DIM), b = e >> 5, j = e & 31;
    const uint8_t* blk = raw + r * PLE_ROW_BYTES + b * 18;
    const float d = __half2float(__ushort_as_half((unsigned short) (blk[0] | (blk[1] << 8))));
    const uint8_t q = blk[2 + (j & 15)];
    out[i] = d * (float) kPleIq4Nl[j < 16 ? (q & 15) : (q >> 4)];
}
}  // namespace

void ple_iq4nl_dequant(const uint8_t* raw, int64_t rows, float* out, void* stream) {
    const int64_t n = rows * PLE_HEAD_DIM;
    if (n <= 0) return;
    ple_iq4nl_dequant_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(raw, n, out);
}

}  // namespace strata::kernels
