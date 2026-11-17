// src/prefill/moe_mmq.cu - see include/strata/prefill/moe_mmq.hpp.  llama.cpp's MMQ (ggml-cuda, MIT) is compiled
// from the pinned llama.cpp checkout the build already takes ggml from; src/prefill/ggml_cuda_host.cu supplies the
// few host symbols of ggml-cuda.cu it references.
#include "strata/prefill/moe_mmq.hpp"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#if defined(STRATA_MMQ_TAB) && !defined(__HIPCC__)
#include "moe_mmq_tab.hpp"   // declarations only: the tab kernels live in the per-type mmq-tab-*.cu TUs
#endif

#include "strata/prefill/mmq_kq_abi.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::prefill::mmq {
namespace {
// The Gyro trellis types (ids 144-146) are outside the type table of an upstream ggml (indexed by the id): their block
// geometry is known here (128 weights; 34 / 26 / 30 bytes), everything else asks ggml.
inline bool is_tq(int t) { return t >= 144 && t <= 149; }   // 144-146 TQ/TQK, 147-149 the pair trellis (Gyro-L)
inline int64_t type_blck(int t) { return is_tq(t) ? 128 : (int64_t) ggml_blck_size((ggml_type) t); }
inline size_t type_bytes(int t) {
    return t == 144 ? 34 : t == 145 ? 26 : t == 146 ? 30 : t == 147 ? 42 : t == 148 ? 50 : t == 149 ? 58
                                                                                     : ggml_type_size((ggml_type) t);
}
inline const char* type_label(int t) {
    return t == 144 ? "tq2_t" : t == 145 ? "tqk6" : t == 146 ? "tqk7"
         : t == 147 ? "tqp5" : t == 148 ? "tqp6" : t == 149 ? "tqp7" : ggml_type_name((ggml_type) t);
}
}  // namespace
#if defined(STRATA_KQ_DSO)
namespace kq {
int abi_args_size() { return (int) sizeof(mmq_args); }
}  // namespace kq
#endif
namespace {
#if defined(STRATA_MMQ_KQUANTS)
// MODEL-UPDATE Phase 1: the K-quant products through MMQ are opt-in at run time (STRATA_MMQ_KQ=1); on CUDA their
// kernels are the strata_kq library (mmq_kq_loader.cpp), opened only then
bool kq_ok() {
#if defined(STRATA_KQ_DSO)
    return kq::available();
#else
    static const bool on = [] {
        const char* v = std::getenv("STRATA_MMQ_KQ");
        return v != nullptr && v[0] == '1';
    }();
    return on;
#endif
}
#endif

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill mmq: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

// ... plus, in the same launch, the expert's 16-byte NVFP4 tail (tail -> tail_dst) and the zeroed MMQ tails after
// the group's last slot (nz uint4 after ab_dst's and c_dst's copies): one launch per expert instead of a kernel,
// a 16-byte copy and, per group, two memsets.
__global__ void copy16_kernel(const uint4* __restrict__ a, int64_t na, const uint4* __restrict__ b, int64_t nb,
                              uint4* __restrict__ ab_dst, const uint4* __restrict__ c, int64_t nc, uint4* __restrict__ c_dst,
                              const uint4* __restrict__ tail, uint4* __restrict__ tail_dst, int64_t nz) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = na + nb + nc;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < n) c_dst[i - na - nb] = c[i - na - nb];
    else if (i < n + nz) ab_dst[na + nb + (i - n)] = make_uint4(0, 0, 0, 0);
    else if (i < n + 2 * nz) c_dst[nc + (i - n - nz)] = make_uint4(0, 0, 0, 0);
    else if (i == n + 2 * nz && tail != nullptr) *tail_dst = *tail;
}
// copy16_kernel for an MMQ group: blockIdx.y is the expert (first + y); the last one (n - 1) also zeroes nz uint4
// after its gu and d slots, and each copies its NVFP4 tail (tail_off, in uint4; tail_dst null = none)
struct GroupArgs {
    const uint8_t* blob[kGatherGroupMax];
    int64_t up_off, down_off, gu_stride, d_stride, tail_off;   // in uint4
};
__global__ void copy16_group_kernel(GroupArgs ga, int first, int n, int64_t na, int64_t nc, int64_t nz,
                                    uint4* __restrict__ gu_dst, uint4* __restrict__ d_dst, uint4* __restrict__ tail_dst) {
    const int q = first + (int) blockIdx.y;
    const uint4* src = (const uint4*) ga.blob[q];
    uint4* ab = gu_dst + (int64_t) q * ga.gu_stride;
    uint4* cd = d_dst + (int64_t) q * ga.d_stride;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t m = 2 * na + nc;
    if (i < na) ab[i] = src[i];
    else if (i < 2 * na) ab[i] = src[ga.up_off + (i - na)];
    else if (i < m) cd[i - 2 * na] = src[ga.down_off + (i - 2 * na)];
    else if (i < m + nz) { if (q == n - 1) ab[ga.gu_stride + (i - m)] = make_uint4(0, 0, 0, 0); }
    else if (i < m + 2 * nz) { if (q == n - 1) cd[ga.d_stride + (i - m - nz)] = make_uint4(0, 0, 0, 0); }
    else if (i == m + 2 * nz && tail_dst != nullptr) tail_dst[q] = src[ga.tail_off];
}
__global__ void copy1_kernel(const uint8_t* __restrict__ a, int64_t na, const uint8_t* __restrict__ b, int64_t nb,
                             uint8_t* __restrict__ ab_dst, const uint8_t* __restrict__ c, int64_t nc, uint8_t* __restrict__ c_dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}

// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales
// [2560][10] f16 (the layout of prefill/kernels.cu's blob_dequant_kernel).  A GGUF Q2_0 block is {f16 d; 16 code
// bytes} with the same 2-bit codes in the same order, so a block is a scale and a 16-byte run of codes.
__global__ void strata_q2_kernel(const uint8_t* __restrict__ blob, uint16_t* __restrict__ gu, uint16_t* __restrict__ dn) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one GGUF block
    const int64_t n_gu = 1280LL * 40, n_d = 2560LL * 10;
    const uint8_t* codes;
    const uint16_t* scale;
    uint16_t* out;
    if (i < n_gu) {
        const int64_t row = i / 40, b = i % 40;
        codes = blob + row * 640 + b * 16;
        scale = (const uint16_t*) (blob + O_GU_SC) + row * 40 + b;
        out = gu + i * 9;
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 10, b = j % 10;
        codes = blob + O_D_CODES + row * 160 + b * 16;
        scale = (const uint16_t*) (blob + O_D_SC) + row * 10 + b;
        out = dn + j * 9;
    } else {
        return;
    }
    const uint4 q = *(const uint4*) codes;
    const uint16_t* qh = (const uint16_t*) &q;
    out[0] = *scale;
#pragma unroll
    for (int k = 0; k < 8; ++k) out[1 + k] = qh[k];
}

__global__ void swiglu_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int64_t n_ff,
                              bool interleaved) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff;
    const float* row = gu + r * 2 * n_ff;
    const float g = interleaved ? row[2 * k] : row[k], u = interleaved ? row[2 * k + 1] : row[n_ff + k];
    h[i] = g / (1.0f + __expf(-g)) * u;
}

#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
// swiglu_kernel (n_ff 640), then quantize_mmq_q8_1<D4>'s rows of its output, in one pass: a block per row (128
// threads, 4 values each, blockIdx.y the row's 512-value half), H never stored.  Each value is swiglu_kernel's
// expression and each 32-value block quantize_mmq_q8_1's (the same unit and flags: the same bits).
template <bool IL>
__global__ void __launch_bounds__(128) swiglu_quant_kernel(const float* __restrict__ gu, block_q8_1_mmq* __restrict__ y,
                                                           int64_t rows) {
    constexpr int64_t n_ff = 640, ne0 = 1024;   // the padded row: pad512(640)
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) return;
    const int64_t r = blockIdx.x;
    float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i0 < n_ff) {
        const float* row = gu + r * 2 * n_ff;
        float4 g4, u4;
        if constexpr (IL) {   // gate 2k, up 2k + 1
            const float4 a = *reinterpret_cast<const float4*>(row + 2 * i0), b = *reinterpret_cast<const float4*>(row + 2 * i0 + 4);
            g4 = make_float4(a.x, a.z, b.x, b.z);
            u4 = make_float4(a.y, a.w, b.y, b.w);
        } else {
            g4 = *reinterpret_cast<const float4*>(row + i0);
            u4 = *reinterpret_cast<const float4*>(row + n_ff + i0);
        }
        xi.x = g4.x / (1.0f + __expf(-g4.x)) * u4.x;
        xi.y = g4.y / (1.0f + __expf(-g4.y)) * u4.y;
        xi.z = g4.z / (1.0f + __expf(-g4.z)) * u4.z;
        xi.w = g4.w / (1.0f + __expf(-g4.w)) * u4.w;
    }
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = 32 / 8; offset > 0; offset >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    const float d_inv = 127.0f / amax;
    char4 c;
    c.x = roundf(xi.x * d_inv);
    c.y = roundf(xi.y * d_inv);
    c.z = roundf(xi.z * d_inv);
    c.w = roundf(xi.w * d_inv);
    const float d = 1.0f / d_inv;
    const int64_t k_block = i0 / QK8_1_MMQ, iqs = i0 % QK8_1_MMQ;
    block_q8_1_mmq* yb = y + k_block * rows + r;
    reinterpret_cast<char4*>(yb->qs)[iqs / 4] = c;
    if (iqs % 32 == 0) yb->d4[iqs / 32] = d;
}
#endif

// swiglu_kernel on the MMQ gate/up output with each expert's gate and up scales applied as it is read: the products
// scale_gu_rows stored, without the pass that stored them
__global__ void swiglu_scaled_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int64_t n_ff,
                                     const int32_t* __restrict__ bounds, int n, const float* __restrict__ tails,
                                     int64_t row0) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff, ra = row0 + r;
    int q = 0;
    while (q + 1 < n && ra >= bounds[q + 1]) ++q;
    const float* row = gu + r * 2 * n_ff;
    const float g = row[k] * tails[4 * q], u = row[n_ff + k] * tails[4 * q + 1];
    h[i] = g / (1.0f + __expf(-g)) * u;
}
__global__ void down_row_scales_kernel(float* __restrict__ sd, const int32_t* __restrict__ bounds, int n,
                                       const float* __restrict__ tails, int64_t nrows) {
    const int64_t r = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    int q = 0;
    while (q + 1 < n && r >= bounds[q + 1]) ++q;
    sd[r] = tails[4 * q + 2];
}

__global__ void iota_kernel(int32_t* dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (int32_t) i;
}

#if defined(__HIPCC__)   // #820: only the HIP dense-MMQ path (Gemm::native_mmq) uses these two
__global__ void f16_to_f32_kernel(const uint16_t* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(__ushort_as_half(x[i]));
}

__global__ void set_bounds_kernel(int32_t* d, int32_t rows) {
    d[0] = 0;
    d[1] = rows;
}

#endif

unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

#if defined(STRATA_MMQ_TAB) && !defined(__HIPCC__)
// STRATA_MMQ_DIRECT: a group's guards (y < g.n, byte stores) and its two zero tails (y == g.n: gate/up, y == g.n + 1:
// down; uint4 stores when `tail16`, which the host checked: both pointers and the size 16-byte aligned).  Only in a
// STRATA_MMQ_TAB build: the engine loads every kernel at start (EAGER), so a build without it keeps today's image.
__global__ void prep_group_kernel(GuardTab g, int64_t guard, uint8_t* tail_gu, uint8_t* tail_d, int64_t tail, bool tail16) {
    const int y = (int) blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (y < g.n) {
        if (i < guard) g.p[y][i] = 0;
        return;
    }
    uint8_t* t = y == g.n ? tail_gu : tail_d;
    if (t == nullptr) return;
    if (tail16) {
        if (i < tail / 16) ((uint4*) t)[i] = make_uint4(0, 0, 0, 0);
    } else if (i < tail) {
        t[i] = 0;
    }
}

// the pointer-table MMQ per type: the native packs' nine types (CMake's _strata_mmq_tab_types, all of them checked
// bit for bit by prefill_mmq_tab_test).  false: no instance - the optional Q5_0 and K-quant experts keep the gather
bool run_tab(ggml_backend_cuda_context* ctx, const mmq_args* a, const mmq_x_tab* tab, cudaStream_t s) {
    const ggml_type t = a ? a->type_x : GGML_TYPE_COUNT;
#define STRATA_TAB_CASE(T) case T: return ctx == nullptr || mul_mat_q_tab_case<T>(*ctx, *a, *tab, s)
    switch (t) {
        STRATA_TAB_CASE(GGML_TYPE_Q2_0);
        STRATA_TAB_CASE(GGML_TYPE_IQ2_XXS);
        STRATA_TAB_CASE(GGML_TYPE_IQ2_XS);
        STRATA_TAB_CASE(GGML_TYPE_IQ2_S);
        STRATA_TAB_CASE(GGML_TYPE_IQ3_XXS);
        STRATA_TAB_CASE(GGML_TYPE_IQ3_S);
        STRATA_TAB_CASE(GGML_TYPE_IQ4_NL);
        STRATA_TAB_CASE(GGML_TYPE_IQ4_XS);
        STRATA_TAB_CASE(GGML_TYPE_Q8_0);
        default: return false;
    }
#undef STRATA_TAB_CASE
}
// whether run_tab has an instance for type t (no launch)
bool tab_has_type(int t) {
    mmq_args a{};
    a.type_x = (ggml_type) t;
    return run_tab(nullptr, &a, nullptr, nullptr);
}
#else
bool run_tab(ggml_backend_cuda_context*, const mmq_args*, const void*, cudaStream_t) {
    std::fprintf(stderr, "prefill mmq: this build has no pointer-table MMQ (STRATA_MMQ_TAB)\n");
    std::exit(1);
}
bool tab_has_type(int) { return false; }
#endif

}  // namespace

// mmq_nvfp4_w4a8.cu: mmq.cuh's int8 NVFP4 path, compiled for this GPU with Blackwell's FP4 MMA hidden
void run_nvfp4_w4a8(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s);
// mmq_nvfp4_w4a4.cu (the one unit built for 12xa): FP4 x FP4 and its activation quantizer, if this card runs it
bool w4a4_available();
void run_nvfp4_w4a4(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s);
void quantize_nvfp4_w4a4(const float* x, const int32_t* ids, void* xq, float* yscale, bool aligned, int64_t cols,
                         int64_t ld, int64_t rows, int64_t padded, cudaStream_t s);

bool built() { return true; }

Nvfp4Mode nvfp4_mode() {
    static const Nvfp4Mode m = [] {
        const char* e = std::getenv("STRATA_PREFILL_NVFP4");
        if (e == nullptr || std::strcmp(e, "w4a8") == 0) return Nvfp4Mode::W4A8;
        if (std::strcmp(e, "w4a4") == 0) return Nvfp4Mode::W4A4;
        if (std::strcmp(e, "fp16") == 0) return Nvfp4Mode::FP16;
        std::fprintf(stderr, "STRATA_PREFILL_NVFP4=%s: expected w4a8, w4a4 or fp16\n", e);
        std::exit(1);
    }();
    return m;
}

bool supported(int t) {
    switch ((ggml_type) t) {
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q3_K:   // the v7l1280 q3k latent pack's gate/up
#ifdef STRATA_ORCA_Q4KS_MMQ
        case GGML_TYPE_Q5_0:   // #296 (Q4_K and Q5_1: STRATA_MMQ_KQUANTS)
#endif
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q8_0:   // the draft layer's dense matrices (E-9)
#ifdef STRATA_MMQ_TQ
        case GGML_TYPE_TQ2_T: case GGML_TYPE_TQK6: case GGML_TYPE_TQK7:   // APR experts (agention fork's MMQ)
#endif
#ifdef STRATA_MMQ_TQP   // a ggml with the pair-trellis types (the overlay, or an agention tree that has their instances)
        case GGML_TYPE_TQP5: case GGML_TYPE_TQP6: case GGML_TYPE_TQP7:     // Gyro-L pair trellis (types 147-149)
#endif
            return true;
#ifdef STRATA_MMQ_KQUANTS
        // Unsloth's UD-Q4/Q5/Q6_K_XL experts; on HIP also the dense GGUF projections (STRATA_DENSE_MMQ)
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q5_1: case GGML_TYPE_Q6_K:
            return kq_ok();
#endif
#if !defined(GGML_USE_HIP)
        case GGML_TYPE_NVFP4:
            return nvfp4_mode() != Nvfp4Mode::FP16;
#endif
        default:
            return false;
    }
}

bool fits(int t, int64_t w_rows) {
    if (!supported(t)) return false;
    // mul_mat_q_case's choice: the "fallback" configs when the rows are not a multiple of 128; then
    // mul_mat_q_switch_J's loop - a tile size whose config exists for this card and fits its shared memory
    const bool fallback = w_rows % 128 != 0;
    const ggml_cuda_device_info& info = ggml_cuda_info();
    for (int id = 0; id < info.device_count; ++id) {
        const int cc = info.devices[id].cc;
        const size_t smpbo = info.devices[id].smpbo;
        bool any = false;
        for (int J = 8; J <= 128 && !any; J += 8) {
            const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) t, J, fallback, cc);
            any = c.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(c, cc) <= smpbo;
        }
        if (!any) {
            static bool said[GGML_TYPE_COUNT] = {};
            if (t >= 0 && t < GGML_TYPE_COUNT && !said[t]) {
                said[t] = true;
                std::fprintf(stderr, "strata: prompt kernels: llama.cpp's MMQ has no tile for %s (%lld rows) on GPU %d "
                                     "(cc %d, %zu bytes of shared memory per block): that product takes the non-MMQ path "
                                     "(#420)\n", type_label((int) t), (long long) w_rows, id, cc, smpbo);
            }
            return false;
        }
    }
    return true;
}

size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / type_blck((int) t)) * type_bytes((int) t);
}

size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) pad512(cols) * sizeof(block_q8_1_mmq) / (4 * QK8_1) + 128 * sizeof(block_q8_1_mmq);
}

namespace {
__global__ void scale_gu_rows_kernel(float* __restrict__ gu, int64_t ld, int64_t n_ff, const int32_t* __restrict__ bounds,
                                     int n, const float* __restrict__ tails, int64_t row0) {
    const int64_t r = row0 + blockIdx.x;
    int q = 0;
    while (q + 1 < n && r >= bounds[q + 1]) ++q;
    const float sg = tails[4 * q], su = tails[4 * q + 1];
    float* row = gu + r * ld;
    for (int64_t k = threadIdx.x; k < 2 * n_ff; k += blockDim.x) row[k] *= k < n_ff ? sg : su;
}
__global__ void scale_down_rows_kernel(float* __restrict__ d, int64_t ld, const int32_t* __restrict__ bounds, int n,
                                       const float* __restrict__ tails) {
    const int64_t r = blockIdx.x;
    int q = 0;
    while (q + 1 < n && r >= bounds[q + 1]) ++q;
    const float sd = tails[4 * q + 2];
    float* row = d + r * ld;
    for (int64_t k = threadIdx.x; k < ld; k += blockDim.x) row[k] *= sd;
}
}  // namespace

bool fp4_activations(int t) {
#if defined(GGML_USE_HIP)   // no NVFP4 MMQ on HIP (and no W4A4 unit)
    (void) t;
    return false;
#else
    if (t != GGML_TYPE_NVFP4 || nvfp4_mode() != Nvfp4Mode::W4A4) return false;
    static const bool on = [] {
        const bool ok = w4a4_available();
        if (!ok) std::fprintf(stderr, "prefill mmq: STRATA_PREFILL_NVFP4=w4a4 needs an sm_12x card in a 12x build; W4A8\n");
        return ok;
    }();
    return on;
#endif
}

void scale_gu_rows(float* gu, int64_t ld, int64_t n_ff, const int32_t* bounds, int n, const float* tails,
                   int64_t row0, int64_t nrows, void* stream) {
    if (nrows <= 0 || n <= 0) return;
    scale_gu_rows_kernel<<<(unsigned) nrows, 256, 0, (cudaStream_t) stream>>>(gu, ld, n_ff, bounds, n, tails, row0);
    ck(cudaGetLastError(), "scale_gu_rows");
}

void scale_down_rows(float* d, int64_t ld, const int32_t* bounds, int n, const float* tails, int64_t nrows, void* stream) {
    if (nrows <= 0 || n <= 0) return;
    scale_down_rows_kernel<<<(unsigned) nrows, 256, 0, (cudaStream_t) stream>>>(d, ld, bounds, n, tails);
    ck(cudaGetLastError(), "scale_down_rows");
}

void quantize(const float* x, const int32_t* ids, void* xq, int t, int64_t cols, int64_t ld, int64_t rows, void* stream,
              float* yscale) {
    if (rows <= 0) return;
#if !defined(GGML_USE_HIP)
    if (fp4_activations(t)) {
        if (yscale == nullptr) { std::fprintf(stderr, "prefill mmq: NVFP4 activations need a scale buffer\n"); std::exit(1); }
        const bool aligned = ((uintptr_t) x % 32 == 0) && ((size_t) ld * sizeof(float)) % 32 == 0;
        quantize_nvfp4_w4a4(x, ids, xq, yscale, aligned, cols, ld, rows, pad512(cols), (cudaStream_t) stream);
        ck(cudaGetLastError(), "quantize");
        return;
    }
#endif
    (void) yscale;
    quantize_mmq_q8_1_cuda(x, ids, xq, (ggml_type) t, cols, ld, rows * ld, rows * ld, pad512(cols), rows, 1, 1,
                           (cudaStream_t) stream);
    ck(cudaGetLastError(), "quantize");
}

void quantize_scatter(const float* x, const int32_t* slot, const int32_t* src, void* xq, int t, int64_t cols, int64_t ld,
                      int64_t tokens, int k_used, void* stream) {
    if (tokens <= 0 || k_used <= 0) return;
    static const bool gather = [] { const char* e = std::getenv("STRATA_QUANT_GATHER"); return e != nullptr && std::atoi(e) != 0; }();
    if (gather) {
        quantize(x, src, xq, t, cols, ld, tokens * k_used, stream);
        return;
    }
    quantize_scatter_mmq_q8_1_cuda(x, slot, xq, (ggml_type) t, cols, ld, pad512(cols), tokens, tokens * k_used, k_used,
                                   (cudaStream_t) stream);
    ck(cudaGetLastError(), "quantize_scatter");
}

Context::Context() {
    int dev = 0;
    cudaGetDevice(&dev);
    ctx_ = new ggml_backend_cuda_context(dev);
}
Context::~Context() { delete (ggml_backend_cuda_context*) ctx_; }

void Context::run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    const ggml_type t = (ggml_type) p.type;
    const int64_t qk = type_blck((int) t), bpr = p.w_cols / qk;
    const mmq_args a = {(const char*) p.w, t, (const int*) p.xq, p.ids, p.bounds, p.dst, p.y_scale,
                        p.w_cols, p.w_rows, p.total_rows, bpr, p.total_rows, p.ld_dst,
                        p.n, p.n, p.tab ? 0 : (int64_t) (p.expert_bytes / type_bytes((int) t)), 0, 0,
                        1, 1, 0, 0, 0,
                        p.max_rows, p.max_rows};
    {
        static const bool dump = std::getenv("STRATA_MMQ_ARGS") != nullptr;
        if (dump) {
            std::fprintf(stderr, "mmqargs t=%d x=%p y=%p ids=%p b=%p dst=%p ys=%p | cx=%lld rx=%lld cd=%lld srx=%lld cy=%lld "
                         "rd=%lld | nchx=%lld nchy=%lld scx=%lld scy=%lld scd=%lld | nsx=%lld nsy=%lld ssx=%lld ssy=%lld "
                         "ssd=%lld | nmax=%lld nopt=%lld\n",
                         (int) a.type_x, (const void*) a.x, (const void*) a.y, (const void*) a.ids_dst,
                         (const void*) a.expert_bounds, (void*) a.dst, (const void*) a.y_scale, (long long) a.ncols_x,
                         (long long) a.nrows_x, (long long) a.ncols_dst, (long long) a.stride_row_x,
                         (long long) a.ncols_y, (long long) a.nrows_dst, (long long) a.nchannels_x,
                         (long long) a.nchannels_y, (long long) a.stride_channel_x, (long long) a.stride_channel_y,
                         (long long) a.stride_channel_dst, (long long) a.nsamples_x, (long long) a.nsamples_y,
                         (long long) a.stride_sample_x, (long long) a.stride_sample_y,
                         (long long) a.stride_sample_dst, (long long) a.ncols_max, (long long) a.ncols_opt);
        }
    }
    auto& ctx = *(ggml_backend_cuda_context*) ctx_;
    const cudaStream_t s = (cudaStream_t) stream;
    if (p.tab) {
        if (p.n > kGatherGroupMax) {
            std::fprintf(stderr, "prefill mmq: a pointer table holds at most %d experts (got %d)\n", kGatherGroupMax, p.n);
            std::exit(1);
        }
#if defined(STRATA_MMQ_TAB) && !defined(__HIPCC__)
        mmq_x_tab tab{};
        for (int z = 0; z < p.n; ++z) tab.p[z] = (const char*) p.tab->p[z];
        if (!run_tab(&ctx, &a, &tab, s)) {
            std::fprintf(stderr, "prefill mmq: no pointer-table MMQ for %s with %lld rows\n", type_label((int) t),
                         (long long) p.w_rows);
            std::exit(1);
        }
#else
        run_tab(&ctx, &a, nullptr, s);
#endif
        ck(cudaGetLastError(), "mul_mat_q (pointer table)");
        return;
    }
    switch (t) {
#ifdef STRATA_ORCA_Q4KS_MMQ
        case GGML_TYPE_Q5_0: mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, a, s); break;
#endif
        case GGML_TYPE_Q2_0: mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, a, s); break;
        case GGML_TYPE_Q3_K: mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XXS: mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XS: mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_S: mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, a, s); break;
        case GGML_TYPE_IQ3_XXS: mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ3_S: mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, a, s); break;
        case GGML_TYPE_IQ4_NL: mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, a, s); break;
        case GGML_TYPE_IQ4_XS: mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, a, s); break;
        case GGML_TYPE_Q8_0: mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, a, s); break;
#ifdef STRATA_MMQ_TQ
        case GGML_TYPE_TQ2_T: mul_mat_q_case<GGML_TYPE_TQ2_T>(ctx, a, s); break;
        case GGML_TYPE_TQK6: mul_mat_q_case<GGML_TYPE_TQK6>(ctx, a, s); break;
        case GGML_TYPE_TQK7: mul_mat_q_case<GGML_TYPE_TQK7>(ctx, a, s); break;
#endif
#ifdef STRATA_MMQ_TQP
        case GGML_TYPE_TQP5: mul_mat_q_case<GGML_TYPE_TQP5>(ctx, a, s); break;
        case GGML_TYPE_TQP6: mul_mat_q_case<GGML_TYPE_TQP6>(ctx, a, s); break;
        case GGML_TYPE_TQP7: mul_mat_q_case<GGML_TYPE_TQP7>(ctx, a, s); break;
#endif
#if defined(STRATA_MMQ_KQUANTS) && defined(STRATA_KQ_DSO)
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_Q5_1: {
            const int e = kq::run((int) t, &a, s);
            if (e != 0) {
                std::fprintf(stderr, "prefill mmq: the K-quant library failed on type %d (error %d)\n", (int) t, e);
                std::exit(1);
            }
            break;
        }
#elif defined(STRATA_MMQ_KQUANTS)
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_K: mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, a, s); break;
        case GGML_TYPE_Q6_K: mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_1: mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, a, s); break;
#endif
#if !defined(GGML_USE_HIP)   // NVFP4: CUDA only (no HIP instance, no W4A8 unit)
        case GGML_TYPE_NVFP4:
            if (fp4_activations(t)) run_nvfp4_w4a4(ctx, a, s);
            else run_nvfp4_w4a8(ctx, a, s);
            break;
#endif
        default:
            std::fprintf(stderr, "prefill mmq: type %d is not covered\n", (int) t);
            std::exit(1);
    }
    ck(cudaGetLastError(), "mul_mat_q");
}

// ---- STRATA_MMQ_DIRECT ----
bool tab_built() {
#if defined(STRATA_MMQ_TAB) && !defined(__HIPCC__)
    return true;
#else
    return false;
#endif
}

bool tab_fits(int t, int64_t w_rows) {
    return tab_built() && w_rows % 128 == 0 && tab_has_type(t) && fits(t, w_rows);
}

size_t overread_bytes(int t, int64_t cols) {
    if (!supported(t)) return 0;
    // the K tile's over-read for each (fallback=false) config on any visible GPU, the largest of them: the over-read
    // is not monotonic in the tile length (20 blocks: 12 more with a 16-block tile, 4 with a 24-block one), so each
    // config's own; MMQ_ITER_K's is the floor (it is the tile when no config is known)
    const int64_t qk = type_blck((int) t), bpn = cols / qk;
    auto over = [&](int64_t k_vram) -> int64_t {
        const int64_t bpi = k_vram / qk;
        return bpi > 0 ? (bpi - bpn % bpi) % bpi : 0;
    };
    int64_t blocks = over(MMQ_ITER_K);
    const ggml_cuda_device_info& info = ggml_cuda_info();
    for (int id = 0; id < info.device_count; ++id)
        for (int J = 8; J <= 128; J += 8) {
            const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) t, J, false, info.devices[id].cc);
            if (c.type != GGML_TYPE_COUNT) blocks = std::max<int64_t>(blocks, over(c.K_vram));
        }
    return (size_t) blocks * type_bytes((int) t);
}

bool whole_tiles(int t, int64_t w_rows, int n_experts) {
    if (!supported(t) || w_rows % 128 != 0 || n_experts <= 0) return false;
    // launch_mul_mat_q's grid: without stream-k one block per tile; with it one block per tile when the tiles fill
    // their waves to >= 90 % (NVIDIA), else nsm blocks that split tiles along K at boundaries set by the tile count.
    // The tile count is nty * n_experts * ntx, ntx = any positive count of J-wide column tiles (max_rows decides),
    // and ntiles >= 9 * nsm always fills its waves to >= 90 %: the ntx below that are checked one by one.
    const ggml_cuda_device_info& info = ggml_cuda_info();
    for (int id = 0; id < info.device_count; ++id) {
        const int cc = info.devices[id].cc, nsm = info.devices[id].nsm;
        const size_t smpbo = info.devices[id].smpbo;
        bool any = false, stream_k = false;
        ggml_cuda_mmq_config c0 = ggml_cuda_mmq_get_config((ggml_type) t, 8, false, cc);
        for (int J = 8; J <= 128; J += 8) {
            const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) t, J, false, cc);
            if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, cc) > smpbo) continue;
            if (!any) c0 = c;
            any = true;
            // every tile width accumulates each output over the same K tiles in the same order
            if (c.I != c0.I || c.K_vram != c0.K_vram || c.sram_layout != c0.sram_layout) return false;
            stream_k = stream_k || c.stream_k;
        }
        if (!any) return false;
        if (!stream_k) continue;
        if (!GGML_CUDA_CC_IS_NVIDIA(cc) || nsm <= 0) return false;
        const int64_t per_ntx = (w_rows / c0.I) * n_experts;
        for (int64_t ntx = 1; per_ntx * ntx < 9 * (int64_t) nsm + per_ntx; ++ntx) {
            const int64_t tiles = per_ntx * ntx, waves = (tiles + nsm - 1) / nsm;
            if (100 * tiles / (nsm * waves) < 90) return false;
        }
    }
    return true;
}

int direct_mode() {
    static const int mode = [] {
        const char* v = std::getenv("STRATA_MMQ_DIRECT");
        const int m = v ? std::atoi(v) : 0;
        if (m != 1 && m != 2) return 0;
        if (!tab_built()) {
            std::fprintf(stderr, "strata: STRATA_MMQ_DIRECT=%d is ignored: this build has no pointer-table MMQ "
                                 "(CUDA with STRATA_MMQ_TAB only)\n", m);
            return 0;
        }
        return m;
    }();
    return mode;
}

size_t direct_slot_guard() { return direct_mode() != 0 ? kSlotGuard : 0; }

void prep_group(const GuardTab& g, size_t guard_bytes, void* tail_gu, void* tail_d, size_t tail_bytes, void* stream) {
#if !defined(STRATA_MMQ_TAB) || defined(__HIPCC__)
    // never reached: only a direct group calls it, and direct_mode() is 0 without the pointer-table MMQ
    (void) g; (void) guard_bytes; (void) tail_gu; (void) tail_d; (void) tail_bytes; (void) stream;
    std::fprintf(stderr, "prefill mmq: this build has no pointer-table MMQ (STRATA_MMQ_TAB)\n");
    std::exit(1);
#else
    const int n = g.n > 0 ? std::min(g.n, kGatherGroupMax) : 0;
    const size_t guard = n > 0 ? guard_bytes : 0;
    const size_t tail = tail_gu || tail_d ? tail_bytes : 0;
    if (guard == 0 && tail == 0) return;
    GuardTab gt = g;
    gt.n = guard > 0 ? n : 0;
    const bool tail16 = ((uintptr_t) tail_gu | (uintptr_t) tail_d | tail_bytes) % 16 == 0;
    const size_t span = std::max(guard, tail);
    prep_group_kernel<<<dim3(blocks((int64_t) span), (unsigned) (gt.n + 2)), 256, 0, (cudaStream_t) stream>>>(
        gt, (int64_t) guard, (uint8_t*) tail_gu, (uint8_t*) tail_d, (int64_t) tail, tail16);
    ck(cudaGetLastError(), "prep_group");
#endif
}

void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream, const void* tail, void* tail_dst, size_t zero_bytes) {
    const cudaStream_t s = (cudaStream_t) stream;
    const bool a16 = ((uintptr_t) gate | (uintptr_t) up | (uintptr_t) down | (uintptr_t) gu_dst | (uintptr_t) d_dst |
                      (uintptr_t) tail | (uintptr_t) tail_dst | gu_half_bytes | d_bytes | zero_bytes) % 16 == 0;
    if (a16) {
        const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16, nz = (int64_t) zero_bytes / 16;
        copy16_kernel<<<blocks(2 * na + nc + 2 * nz + 1), 256, 0, s>>>(
            (const uint4*) gate, na, (const uint4*) up, na, (uint4*) gu_dst, (const uint4*) down, nc, (uint4*) d_dst,
            (const uint4*) tail, (uint4*) tail_dst, nz);
    } else {
        if (tail) cudaMemcpyAsync(tail_dst, tail, 16, cudaMemcpyDeviceToDevice, s);
        if (zero_bytes) {
            cudaMemsetAsync((uint8_t*) gu_dst + 2 * gu_half_bytes, 0, zero_bytes, s);
            cudaMemsetAsync((uint8_t*) d_dst + d_bytes, 0, zero_bytes, s);
        }
        const int64_t na = (int64_t) gu_half_bytes, nc = (int64_t) d_bytes;
        copy1_kernel<<<blocks(2 * na + nc), 256, 0, s>>>((const uint8_t*) gate, na, (const uint8_t*) up, na,
                                                         (uint8_t*) gu_dst, (const uint8_t*) down, nc, (uint8_t*) d_dst);
    }
    ck(cudaGetLastError(), "gather_native");
}

bool gather_native_group(const GatherGroup& g, size_t up_off, size_t gu_half_bytes, size_t down_off, size_t d_bytes,
                         void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream, size_t tail_off,
                         void* tail_dst, size_t zero_bytes) {
    if (g.first < 0 || g.n <= g.first || g.n > kGatherGroupMax) return false;
    if (tail_off == 0) tail_dst = nullptr;
    uintptr_t a = (uintptr_t) gu_dst | (uintptr_t) d_dst | up_off | gu_half_bytes | down_off | d_bytes | gu_stride | d_stride |
                  tail_off | (uintptr_t) tail_dst | zero_bytes;
    for (int q = g.first; q < g.n; ++q) a |= (uintptr_t) g.blob[q];
    if (a % 16 != 0) return false;
    GroupArgs ga{};
    for (int q = g.first; q < g.n; ++q) ga.blob[q] = g.blob[q];
    ga.up_off = (int64_t) up_off / 16;
    ga.down_off = (int64_t) down_off / 16;
    ga.gu_stride = (int64_t) gu_stride / 16;
    ga.d_stride = (int64_t) d_stride / 16;
    ga.tail_off = (int64_t) tail_off / 16;
    const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16, nz = (int64_t) zero_bytes / 16;
    const int64_t work = 2 * na + nc + 2 * nz + (tail_dst != nullptr ? 1 : 0);
    copy16_group_kernel<<<dim3(blocks(work), (unsigned) (g.n - g.first)), 256, 0, (cudaStream_t) stream>>>(
        ga, g.first, g.n, na, nc, nz, (uint4*) gu_dst, (uint4*) d_dst, (uint4*) tail_dst);
    ck(cudaGetLastError(), "gather_native_group");
    return true;
}

void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream) {
    strata_q2_kernel<<<blocks(1280LL * 40 + 2560LL * 10), 256, 0, (cudaStream_t) stream>>>(blob, (uint16_t*) gu_dst,
                                                                                         (uint16_t*) d_dst);
    ck(cudaGetLastError(), "gather_strata_q2");
}

void swiglu_scaled(const float* gu, float* h, int64_t rows, int64_t n_ff, const int32_t* bounds, int n,
                   const float* tails, int64_t row0, void* stream) {
    if (rows <= 0) return;
    swiglu_scaled_kernel<<<blocks(rows * n_ff), 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, bounds, n, tails, row0);
    ck(cudaGetLastError(), "swiglu_scaled");
}
void down_row_scales(float* sd, const int32_t* bounds, int n, const float* tails, int64_t nrows, void* stream) {
    if (nrows <= 0 || n <= 0) return;
    down_row_scales_kernel<<<blocks(nrows), 256, 0, (cudaStream_t) stream>>>(sd, bounds, n, tails, nrows);
    ck(cudaGetLastError(), "down_row_scales");
}

void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    swiglu_kernel<<<blocks(rows * n_ff), 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, interleaved);
    ck(cudaGetLastError(), "swiglu");
}

bool swiglu_quant_ok(int t) {
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
    // the q8_1 D4 rows quantize() writes for a down product's H (K 640)
    return supported(t) && !fp4_activations(t) && mmq_get_q8_1_ds_layout((ggml_type) t) == MMQ_Q8_1_DS_LAYOUT_D4;
#else
    (void) t;
    return false;   // HIP: not checked against the HIP build's quantizer; outside the SM60 build: not built
#endif
}

void swiglu_quant(const float* gu, void* hq, int64_t rows, bool interleaved, void* stream) {
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
    if (rows <= 0) return;
    const dim3 grid((unsigned) rows, 2);
    if (interleaved) swiglu_quant_kernel<true><<<grid, 128, 0, (cudaStream_t) stream>>>(gu, (block_q8_1_mmq*) hq, rows);
    else swiglu_quant_kernel<false><<<grid, 128, 0, (cudaStream_t) stream>>>(gu, (block_q8_1_mmq*) hq, rows);
    ck(cudaGetLastError(), "swiglu_quant");
#else
    (void) gu; (void) hq; (void) rows; (void) interleaved; (void) stream;
#endif
}

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n <= 0) return;
    iota_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(dst, n);
    ck(cudaGetLastError(), "iota");
}

#if defined(__HIPCC__)
void f16_to_f32(const uint16_t* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    f16_to_f32_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    ck(cudaGetLastError(), "f16_to_f32");
}

void set_bounds(int32_t* dst, int32_t rows, void* stream) {
    set_bounds_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(dst, rows);
    ck(cudaGetLastError(), "set_bounds");
}
#endif

}  // namespace strata::prefill::mmq
