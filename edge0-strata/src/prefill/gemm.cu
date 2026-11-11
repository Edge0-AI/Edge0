// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
#include "strata/prefill/moe_mmq.hpp"
#endif

#ifdef STRATA_USE_HIP
#include "wmma_gemm.h"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
// The HIP compatibility shim maps CUDA shuffle spellings to Strata helpers.
// hipBLASLt's public headers declare native HIP shuffle functions, so keep
// those declarations from being macro-expanded in this translation unit.
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#endif

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
#include "hipblaslt_tuning.hpp"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>
#include <set>
#include <tuple>
#endif
#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
#include "rocblas_tuning.hpp"
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <filesystem>
#include <map>
#include <string>
#include <system_error>
#include <tuple>
#endif

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#endif

namespace strata::prefill {
namespace {
#if defined(__HIPCC__)
// STRATA_F16_RANGE (gemm_set_f16_range): the largest |value| and the count beyond 65504 / non-finite, for the BF16
// weights converted to FP16 (slot 0, bf16_to_f16_rows) and the FP16 GEMM outputs widened (slot 1, widen_rows_f16 -
// an output rocBLAS wrote as FP16 is Inf there, not saturated).  As the activation counter in kernels.cu.
__device__ int g_f16_range_on_g = 0;
__device__ unsigned g_f16_range_max_g[2] = {0, 0};
__device__ unsigned long long g_f16_range_over_g[2] = {0, 0};
__device__ __forceinline__ void f16_range_note_g(int k, float f) {
    const float a = fabsf(f);
    const unsigned bits = __float_as_uint(a);
    if (bits > g_f16_range_max_g[k]) atomicMax(&g_f16_range_max_g[k], bits);
    if (a > 65504.0f || isnan(f)) atomicAdd(&g_f16_range_over_g[k], 1ull);
}
// Y's rows, written by an FP16-out GEMM as FP16 at the start of each FP32 row (ldc = 2 ldy halves), widened in place.
// Float c overwrites halves 2c and 2c+1, so a row is walked from its end in blocks: a block's halves are read into
// registers, the block syncs, then writes its floats - which only cover halves of blocks already read.
// STRATA_DBG_NAN: the FP16 outputs that are not finite (an FP32-accumulated sum past 65504 becomes inf in FP16).
__device__ unsigned long long g_f16_nonfinite = 0;
__global__ void widen_rows_f16(float* __restrict__ Y, int64_t n, int64_t ldy, int count_nonfinite) {
    float* y = Y + (int64_t) blockIdx.x * ldy;
    const __half* h = reinterpret_cast<const __half*>(y);
    const int64_t nb = (n + blockDim.x - 1) / blockDim.x;
    for (int64_t b = nb - 1; b >= 0; --b) {
        const int64_t c = b * blockDim.x + threadIdx.x;
        const float v = c < n ? __half2float(h[c]) : 0.0f;
        if (count_nonfinite && c < n && !isfinite(v)) atomicAdd(&g_f16_nonfinite, 1ull);
        if (c < n && g_f16_range_on_g) f16_range_note_g(1, v);
        __syncthreads();
        if (c < n) y[c] = v;
        __syncthreads();
    }
}
// BF16 weight rows -> FP16, saturated (bench/results/2026-10-04-rdna2-fp16-prompt: none leaves FP16's range)
__global__ void bf16_to_f16_rows(const uint16_t* __restrict__ s, __half* __restrict__ d, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float f = __uint_as_float((uint32_t) s[i] << 16);
        if (g_f16_range_on_g) f16_range_note_g(0, f);
        d[i] = __float2half(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f));   // as hf_sat: a NaN stays NaN
    }
}
#endif
}  // namespace

void gemm_set_f16_range(bool on) {
#if defined(__HIPCC__)
    const int v = on ? 1 : 0;
    if (cudaMemcpyToSymbol(g_f16_range_on_g, &v, sizeof v) != cudaSuccess) {
        std::fprintf(stderr, "prefill gemm: setting the FP16 range check failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
    }
#else
    (void) on;
#endif
}
bool gemm_f16_range_read(float max_abs[2], unsigned long long over[2], bool reset) {
#if defined(__HIPCC__)
    unsigned bits[2] = {0, 0}; unsigned long long o[2] = {0, 0};
    if (cudaMemcpyFromSymbol(bits, g_f16_range_max_g, sizeof bits) != cudaSuccess ||
        cudaMemcpyFromSymbol(o, g_f16_range_over_g, sizeof o) != cudaSuccess) { cudaGetLastError(); return false; }
    for (int k = 0; k < 2; ++k) { union { unsigned u; float f; } c; c.u = bits[k]; max_abs[k] = c.f; over[k] = o[k]; }
    if (reset) {
        const unsigned z[2] = {0, 0}; const unsigned long long zz[2] = {0, 0};
        cudaMemcpyToSymbol(g_f16_range_max_g, z, sizeof z); cudaMemcpyToSymbol(g_f16_range_over_g, zz, sizeof zz);
    }
    return true;
#else
    (void) max_abs; (void) over; (void) reset;
    return false;
#endif
}

namespace {

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

// #247/#325: on Windows (seen on gfx1201), hipBLAS can return success with the correct BF16/FP16 product for some
// shapes (hc up once T >= 96, the router) and still leave hipErrorInvalidValue set, which the next kernel's error
// check turns into an exit. The multiply has finished, so that one stale error is cleared after a GEMM that succeeded;
// any other error still stops the engine. Windows only: on Linux a stale hipErrorInvalidValue is a real error from
// an earlier call and keeps being reported. A no-op everywhere else (CUDA compiles none of it).
#if defined(__HIPCC__) && defined(_WIN32)
void absorb_hipblas_sticky(const char* what) {
    const hipError_t sticky = hipGetLastError();
    if (sticky == hipSuccess || sticky == hipErrorInvalidValue) return;
    std::fprintf(stderr, "prefill gemm: %s left %s\n", what, hipGetErrorString(sticky));
    std::exit(1);
}
#define STRATA_ABSORB_HIPBLAS_STICKY(what) absorb_hipblas_sticky(what)
#else
#define STRATA_ABSORB_HIPBLAS_STICKY(what) ((void) 0)
#endif

// A setup call whose failure the engine survives (the handle keeps its defaults), as before #240 - but said.
void note(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d (continuing)\n", what, (int) s);
}

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
struct HipLtCallKey {
    strata::prefill::hipblaslt::InputType type;
    int t;
    int n;
    int k;
    int ldy;
    uint32_t beta_bits;

    bool operator<(const HipLtCallKey& other) const {
        return std::tie(type, n, k, ldy, t, beta_bits) <
               std::tie(other.type, other.n, other.k, other.ldy, other.t, other.beta_bits);
    }
};

struct HipLtCachedAlgo {
    bool supported = false;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
};

struct HipLtState {
    hipblasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    strata::prefill::hipblaslt::TuningTable table;
    std::map<HipLtCallKey, HipLtCachedAlgo> cache;
    uint64_t lt_launches = 0;
    uint64_t fallbacks = 0;
    std::set<std::tuple<strata::prefill::hipblaslt::InputType, int, int, int, int>> fallback_shapes;

    ~HipLtState() {
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: hipBLASLt summary launches=%llu fallbacks=%llu unique_fallback_shapes=%zu\n",
                         (unsigned long long) lt_launches, (unsigned long long) fallbacks, fallback_shapes.size());
            for (const auto& shape : fallback_shapes) {
                const auto type = std::get<0>(shape);
                std::fprintf(stderr, "prefill gemm: fallback shape dtype=%s T=%d N=%d K=%d ldy=%d\n",
                             type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16",
                             std::get<1>(shape), std::get<2>(shape), std::get<3>(shape), std::get<4>(shape));
            }
        }
        if (handle) hipblasLtDestroy(handle);
    }
};

struct HipLtDescriptors {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr;
    hipblasLtMatrixLayout_t b = nullptr;
    hipblasLtMatrixLayout_t c = nullptr;

    ~HipLtDescriptors() {
        if (op) hipblasLtMatmulDescDestroy(op);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
    }

    bool init(hipDataType type, int t, int n, int k, int ldy, int ldb = 0) {
        const hipblasOperation_t trans_a = HIPBLAS_OP_T;
        const hipblasOperation_t trans_b = HIPBLAS_OP_N;
        if (hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&a, type, k, n, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&b, type, k, t, ldb > k ? ldb : k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, n, t, ldy) != HIPBLAS_STATUS_SUCCESS) {
            return false;
        }
        return true;
    }
};

// The engine package's own tuning folder (<engine folder>/tuning), for a start that did not come through the launcher (a hand-made
// config, a CLI run): without a table the prompt GEMMs fall back to plain hipBLAS and a Strix Halo reads prompts at half speed
// (340 against 690 tok/s, IQ3_S 4K).  The launcher sets STRATA_HIPBLASLT_TUNING itself; this is the same table found by the engine.
std::string own_tuning_dir() {
#if defined(_WIN32)
    char p[4096] = {};
    const DWORD n = GetModuleFileNameA(nullptr, p, (DWORD) sizeof(p));
    std::string s(p, n);
    const size_t k = s.find_last_of("\\/");
    return k == std::string::npos ? std::string() : s.substr(0, k + 1) + "tuning/";
#else
    char p[4096] = {};
    const ssize_t n = readlink("/proc/self/exe", p, sizeof(p) - 1);
    std::string s(p, n > 0 ? (size_t) n : 0);
    const size_t k = s.find_last_of('/');
    return k == std::string::npos ? std::string() : s.substr(0, k + 1) + "tuning/";
#endif
}

std::unique_ptr<HipLtState> create_hipblaslt_state(void* workspace, size_t workspace_bytes) {
    std::string path_own;
    const char* path = std::getenv("STRATA_HIPBLASLT_TUNING");
    bool from_package = false;
    if (!path || !*path) {
        const char* sw = std::getenv("STRATA_HIPBLASLT_TABLE");     // the launcher's switch: off / 0 / false = never the packaged table
        if (sw != nullptr && (std::strcmp(sw, "off") == 0 || std::strcmp(sw, "0") == 0 || std::strcmp(sw, "false") == 0)) return nullptr;
        from_package = true;                                          // the file is found below, once the card and the version are known
    }

    auto state = std::make_unique<HipLtState>();
    state->workspace = workspace;
    state->workspace_bytes = workspace_bytes;
    if (hipblasLtCreate(&state->handle) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt handle creation failed; using hipBLASEx\n");
        return nullptr;
    }

    int version = 0;
    if (hipblasLtGetVersion(state->handle, &version) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt version query failed; using hipBLASEx\n");
        return nullptr;
    }
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess) {
        std::fprintf(stderr, "prefill gemm: HIP device query failed; using hipBLASEx\n");
        return nullptr;
    }
    std::string arch(properties.gcnArchName);
    const auto suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);

    if (from_package) {
        const std::string dir = own_tuning_dir();
        if (dir.empty()) return nullptr;
        path_own = dir + arch + "-hipblaslt-" + std::to_string(version) + ".txt";
        std::FILE* f = std::fopen(path_own.c_str(), "rb");
        if (f == nullptr) return nullptr;       // no table for this card and hipBLASLt version beside the engine: plain hipBLAS as before
        std::fclose(f);
        path = path_own.c_str();
    }
    std::string error;
    if (!state->table.load(path, arch, version, error)) {
        std::fprintf(stderr, "prefill gemm: %s; using hipBLASEx\n", error.c_str());
        return nullptr;
    }
    std::fprintf(stderr, "prefill gemm: hipBLASLt tuning enabled (%zu rows, %s, version %d)\n",
                 state->table.rows().size(), arch.c_str(), version);
    return state;
}

HipLtCachedAlgo resolve_hipblaslt_algo(HipLtState& state, strata::prefill::hipblaslt::InputType type, int t,
                                       int n, int k, int ldy, float beta) {
    uint32_t beta_bits = 0;
    static_assert(sizeof(beta_bits) == sizeof(beta));
    std::memcpy(&beta_bits, &beta, sizeof(beta));
    const HipLtCallKey key{type, t, n, k, ldy, beta_bits};
    const auto cached = state.cache.find(key);
    if (cached != state.cache.end()) return cached->second;

    HipLtCachedAlgo resolved;
    const bool verbose = std::getenv("STRATA_HIPBLASLT_VERBOSE") != nullptr;
    const auto* row = state.table.closest(type, n, k, ldy, t);
    if (!row) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; no calibration for dtype=%s T=%d N=%d K=%d ldy=%d\n",
                         type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, t, n, k, ldy)) {
        return state.cache.emplace(key, resolved).first->second;
    }

    std::vector<int> solution_ids{row->solution_id};
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
    if (hipblaslt_ext::getAlgosFromIndex(state.handle, solution_ids, candidates) != HIPBLAS_STATUS_SUCCESS ||
        candidates.empty() || candidates.front().state != HIPBLAS_STATUS_SUCCESS ||
        hipblaslt_ext::getIndexFromAlgo(candidates.front().algo) != row->solution_id) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d unavailable for T=%d N=%d K=%d ldy=%d\n",
                         row->solution_id, t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    const float alpha = 1.0f;
    size_t required_workspace = 0;
    auto algo = candidates.front().algo;
    if (hipblaslt_ext::matmulIsAlgoSupported(state.handle, desc.op, &alpha, desc.a, desc.b, &beta, desc.c, desc.c,
                                             algo, required_workspace) != HIPBLAS_STATUS_SUCCESS) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d rejects actual T=%d N=%d K=%d ldy=%d beta=%.9g\n",
                         row->solution_id, t, n, k, ldy, beta);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    resolved.supported = true;
    resolved.algo = algo;
    resolved.workspace_bytes = required_workspace;
    if (verbose) {
        std::fprintf(stderr,
                     "prefill gemm: Lt solution=%d dtype=%s T=%d N=%d K=%d ldy=%d beta=%.9g workspace=%zu\n",
                     row->solution_id, type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n,
                     k, ldy, beta, required_workspace);
    }
    return state.cache.emplace(key, resolved).first->second;
}

// the hipBLASLt solution index the tuning table gives this call (-1: none / unsupported)
int hipblaslt_solution_for(void* opaque_state, strata::prefill::hipblaslt::InputType type, int64_t t, int64_t n, int64_t k,
                           int64_t ldy, float beta) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX || ldy < n) return -1;
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported || resolved.workspace_bytes > state->workspace_bytes) return -1;
    auto algo = resolved.algo;
    return hipblaslt_ext::getIndexFromAlgo(algo);
}

bool try_hipblaslt(void* opaque_state, strata::prefill::hipblaslt::InputType type, const uint16_t* x,
                   const uint16_t* w, float* y, int64_t t, int64_t n, int64_t k, int64_t ldy, float beta,
                   void* stream, int64_t ldx = 0) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX ||
        ldy < n) {
        return false;
    }
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        return false;
    }
    if (resolved.workspace_bytes > state->workspace_bytes) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution needs %zu workspace bytes, have %zu\n",
                         resolved.workspace_bytes, state->workspace_bytes);
        }
        return false;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, (int) t, (int) n, (int) k, (int) ldy, (int) ldx)) return false;
    const float alpha = 1.0f;
    const hipblasStatus_t status = hipblasLtMatmul(state->handle, desc.op, &alpha, w, desc.a, x, desc.b, &beta, y,
                                                   desc.c, y, desc.c, &resolved.algo, state->workspace,
                                                   state->workspace_bytes, (hipStream_t) stream);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        ++state->lt_launches;
        return true;
    }

    std::fprintf(stderr, "prefill gemm: hipBLASLt launch failed with status %d\n", (int) status);
    if (beta != 0.0f) {
        std::fprintf(stderr, "prefill gemm: refusing a fallback after hipBLASLt failed with nonzero beta\n");
        std::exit(1);
    }
    auto* mutable_state = static_cast<HipLtState*>(opaque_state);
    uint32_t beta_bits = 0;
    std::memcpy(&beta_bits, &beta, sizeof(beta_bits));
    auto cached = mutable_state->cache.find(HipLtCallKey{type, (int) t, (int) n, (int) k, (int) ldy, beta_bits});
    if (cached != mutable_state->cache.end()) cached->second.supported = false;
    ++mutable_state->fallbacks;
    mutable_state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
    return false;
}
#endif

#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
// The rocBLAS solution table of f16_inplace (STRATA_ROCBLAS_TUNING: a table file, or a directory holding
// <arch>-rocblas-<version>.txt for this card and rocBLAS build; src/prefill/rocblas_tuning.hpp).  rocBLAS's own
// pick for some shapes is up to 6x slower than the best kernel it holds for them (gfx1030: tools/hip/tune_rocblas.cpp).
struct RocBlasState {
    rocblas_handle handle = nullptr;
    strata::prefill::rocblas_tuning::Table table;
    struct Pick { bool tuned = false; int solution = 0; int refusals = 0; };
    std::map<std::tuple<int, int, int, int>, Pick> cache;   // (T, N, K, ldy) -> the solution to run, or the default
    uint64_t tuned_launches = 0;
    uint64_t fallbacks = 0;

    ~RocBlasState() {
        if (std::getenv("STRATA_ROCBLAS_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: rocBLAS tuning summary tuned_launches=%llu fallbacks=%llu\n",
                         (unsigned long long) tuned_launches, (unsigned long long) fallbacks);
        }
        if (handle) rocblas_destroy_handle(handle);
    }
};

std::unique_ptr<RocBlasState> create_rocblas_state(void* stream) {
    const char* path = std::getenv("STRATA_ROCBLAS_TUNING");
    if (!path || !*path) return nullptr;
    auto state = std::make_unique<RocBlasState>();
    if (rocblas_create_handle(&state->handle) != rocblas_status_success) {
        std::fprintf(stderr, "prefill gemm: rocBLAS handle creation failed; its default kernels run\n");
        return nullptr;
    }
    rocblas_set_stream(state->handle, (hipStream_t) stream);
    char version[128] = {0};
    if (rocblas_get_version_string(version, sizeof version) != rocblas_status_success) {
        std::fprintf(stderr, "prefill gemm: rocBLAS version query failed; its default kernels run\n");
        return nullptr;
    }
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess) {
        std::fprintf(stderr, "prefill gemm: HIP device query failed; rocBLAS's default kernels run\n");
        return nullptr;
    }
    std::string arch(properties.gcnArchName);
    if (const auto suffix = arch.find(':'); suffix != std::string::npos) arch.resize(suffix);
    std::filesystem::path file(path);
    std::error_code ec;
    if (std::filesystem::is_directory(file, ec)) {
        // the table of this card and this rocBLAS build (its version string, trailing dots dropped)
        std::string tag(version);
        while (!tag.empty() && tag.back() == '.') tag.pop_back();
        file /= arch + "-rocblas-" + tag + ".txt";
    }
    std::string error;
    if (!state->table.load(file.string(), arch, version, error)) {
        std::fprintf(stderr, "prefill gemm: %s (%s); rocBLAS's default kernels run\n", error.c_str(),
                     file.string().c_str());
        return nullptr;
    }
    // rocBLAS loads its kernel library on the first product through a handle; a solution-index call before that
    // fails (rocblas_status_internal_error) although the index is valid.  One tiny default product opens it.
    {
        uint16_t* tiny = nullptr;
        if (hipMalloc(&tiny, 4 * 64 * 64 * 2) == hipSuccess) {
            const float one = 1.0f, zero = 0.0f;
            (void) rocblas_gemm_ex(state->handle, rocblas_operation_transpose, rocblas_operation_none, 64, 64, 64, &one,
                                   tiny, rocblas_datatype_f16_r, 64, tiny + 64 * 64, rocblas_datatype_f16_r, 64, &zero,
                                   tiny + 2 * 64 * 64, rocblas_datatype_f16_r, 128, tiny + 2 * 64 * 64,
                                   rocblas_datatype_f16_r, 128, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
            (void) hipStreamSynchronize((hipStream_t) stream);
            (void) hipFree(tiny);
        }
    }
    size_t tuned_rows = 0;
    for (const auto& row : state->table.rows()) tuned_rows += row.tuned ? 1 : 0;
    std::fprintf(stderr, "prefill gemm: rocBLAS tuning enabled (%zu rows, %zu with a solution, %s, rocBLAS %s)\n",
                 state->table.rows().size(), tuned_rows, arch.c_str(), version);
    return state;
}

// Gemm::f16_inplace's product through the table's solution for its shape.  False when the shape has no row, or the
// launch was refused (then the default kernel runs, and for that shape from now on).
bool try_rocblas_f16o(void* opaque_state, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N,
                      int64_t K, int64_t ldy) {
    auto* state = static_cast<RocBlasState*>(opaque_state);
    if (!state || T <= 0 || N <= 0 || K <= 0 || T > INT_MAX || N > INT_MAX || K > INT_MAX || ldy > INT_MAX / 2) return false;
    const auto key = std::make_tuple((int) T, (int) N, (int) K, (int) ldy);
    auto it = state->cache.find(key);
    if (it == state->cache.end()) {
        RocBlasState::Pick pick;
        if (const auto* row = state->table.closest((int) N, (int) K, (int) ldy, (int) T); row && row->tuned) {
            pick.tuned = true;
            pick.solution = row->solution;
        }
        if (std::getenv("STRATA_ROCBLAS_VERBOSE")) {
            if (pick.tuned)
                std::fprintf(stderr, "prefill gemm: rocBLAS solution %d for T=%lld N=%lld K=%lld ldy=%lld\n",
                             pick.solution, (long long) T, (long long) N, (long long) K, (long long) ldy);
            else
                std::fprintf(stderr, "prefill gemm: rocBLAS default kernel; no row for T=%lld N=%lld K=%lld ldy=%lld\n",
                             (long long) T, (long long) N, (long long) K, (long long) ldy);
        }
        it = state->cache.emplace(key, pick).first;
    }
    if (!it->second.tuned) return false;
    const float one = 1.0f, zero = 0.0f;
    const rocblas_status status = rocblas_gemm_ex(
        state->handle, rocblas_operation_transpose, rocblas_operation_none, (int) N, (int) T, (int) K, &one, W,
        rocblas_datatype_f16_r, (int) K, X, rocblas_datatype_f16_r, (int) K, &zero, Y, rocblas_datatype_f16_r,
        (int) (2 * ldy), Y, rocblas_datatype_f16_r, (int) (2 * ldy), rocblas_datatype_f32_r,
        rocblas_gemm_algo_solution_index, it->second.solution, 0);
    if (status == rocblas_status_success) {
        ++state->tuned_launches;
        return true;
    }
    // A refusal writes nothing (the default kernel runs now).  The first one can be rocBLAS still setting up for the
    // kernel (its device workspace: seen as invalid_value once, success after); a solution refused three times is
    // dropped for this shape.
    ++state->fallbacks;
    if (++it->second.refusals >= 3) {
        std::fprintf(stderr, "prefill gemm: rocBLAS solution %d refused (status %d) for T=%lld N=%lld K=%lld ldy=%lld "
                             "three times; the default kernel runs for this shape\n", it->second.solution, (int) status,
                     (long long) T, (long long) N, (long long) K, (long long) ldy);
        it->second.tuned = false;
    } else if (std::getenv("STRATA_ROCBLAS_VERBOSE")) {
        std::fprintf(stderr, "prefill gemm: rocBLAS solution %d refused (status %d) for T=%lld N=%lld K=%lld ldy=%lld; "
                             "the default kernel this time\n", it->second.solution, (int) status, (long long) T,
                     (long long) N, (long long) K, (long long) ldy);
    }
    return false;
}
#endif

}  // namespace

bool prompt_f16() {
#if defined(__HIPCC__)
    constexpr int kMaxDev = 64;
    static std::atomic<int8_t> cached[kMaxDev] = {};   // 0 unknown, 1 off, 2 on
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= kMaxDev) { (void) hipGetLastError(); return false; }
    if (const int8_t c = cached[dev].load(std::memory_order_relaxed)) return c == 2;
    // Opt-in (#835): only STRATA_HIP_PROMPT_F16=1 turns it on.  It changes the prompt path's numbers (an FP16 rounding of
    // each 16-bit GEMM's output), so no card gets it unasked.  On gfx103x it is ~2x faster, so we say so once.
    const char* e = std::getenv("STRATA_HIP_PROMPT_F16");
    const bool on = e != nullptr && e[0] == '1';
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, dev) != hipSuccess) { (void) hipGetLastError(); p.gcnArchName[0] = 0; }
    if (on) std::fprintf(stderr, "strata prefill: HIP device %d%s%s - STRATA_HIP_PROMPT_F16=1: the prompt's 16-bit GEMMs run FP16 in and out "
                                 "(rocBLAS is tuned only for that on gfx103x; not bit-identical to the default)%s",
                         dev, p.gcnArchName[0] ? " " : "", p.gcnArchName, "\n");
    else if (e == nullptr && std::strncmp(p.gcnArchName, "gfx103", 6) == 0)
        std::fprintf(stderr, "strata prefill: HIP device %d %s - tip: STRATA_HIP_PROMPT_F16=1 reads long prompts about 2x faster here "
                             "(FP16 prompt GEMMs, rocBLAS is tuned only for FP16 on gfx103x; the numbers differ slightly, see docs/AMD_HIP.md)%s",
                     dev, p.gcnArchName, "\n");
    cached[dev].store(on ? 2 : 1, std::memory_order_relaxed);
    return on;
#else
    return false;
#endif
}

// A padded X (ldx > K) no kernel took: its rows copied to a contiguous FP16 buffer of stride K.  Grown, never shrunk
// (one buffer per Gemm; a padded X only arrives from STRATA_PF_PAD's producers).  One 2-D copy: the same values, and
// the ordinary paths below then read them exactly as if the producer had written them unpadded.
const uint16_t* Gemm::unpad_x(const uint16_t* X, int64_t T, int64_t K, int64_t ldx) {
    if (X == nullptr || T <= 0 || K <= 0 || ldx <= K) return nullptr;
    const int64_t need = T * K;
    if (need > unpadded_x_elems_) {
        if (unpadded_x_ != nullptr) { cudaFree(unpadded_x_); unpadded_x_ = nullptr; unpadded_x_elems_ = 0; }
        void* p = nullptr;
        if (cudaMalloc(&p, (size_t) need * sizeof(uint16_t)) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        unpadded_x_ = (uint16_t*) p;
        unpadded_x_elems_ = need;
    }
    if (cudaMemcpy2DAsync(unpadded_x_, (size_t) K * sizeof(uint16_t), X, (size_t) ldx * sizeof(uint16_t),
                          (size_t) K * sizeof(uint16_t), (size_t) T, cudaMemcpyDeviceToDevice,
                          (cudaStream_t) stream_) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return unpadded_x_;
}

bool gemm_dequant_side() {
#if defined(__HIPCC__)
    return false;
#else
    static const bool on = [] { const char* v = std::getenv("STRATA_DEQUANT_SIDE"); return v && v[0] == '1'; }();
    return on;
#endif
}

Gemm::~Gemm() {
    for (W16Entry& e : w16_) if (e.d) (void) cudaFree(e.d);
    for (void* e : side_ready_)
        if (e) (void) cudaEventDestroy((cudaEvent_t) e);
    for (void* e : side_free_)
        if (e) (void) cudaEventDestroy((cudaEvent_t) e);
    if (side_stream_) (void) cudaStreamDestroy((cudaStream_t) side_stream_);
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
    delete static_cast<strata::prefill::mmq::Context*>(mmq_ctx_);
    if (mmq_buf_) cudaFree(mmq_buf_);
#endif
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    delete static_cast<HipLtState*>(hipblaslt_state_);
#endif
#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
    delete static_cast<RocBlasState*>(rocblas_state_);
#endif
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (tc_w_) cudaFree(tc_w_);
    if (tc_x_) cudaFree(tc_x_);
    if (unpadded_x_) cudaFree(unpadded_x_);
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    workspace_ = workspace;
    note(cublasSetWorkspace(h, workspace_, ws_bytes), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws_bytes).release();
#endif
#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
    rocblas_state_ = create_rocblas_state(stream).release();
#endif
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    cublasSetWorkspace((cublasHandle_t) handle_, workspace_, ws_bytes);
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (hipblaslt_state_) {
        auto* state = static_cast<HipLtState*>(hipblaslt_state_);
        state->workspace = workspace_;
        state->workspace_bytes = ws_bytes;
    }
#endif
}

bool Gemm::set_dequant_side(uint16_t* scratch2, std::string& err) {
    side_scratch_ = scratch2;
    side_cur_ = 0;
    if (scratch2 == nullptr) return true;
    if (side_stream_ == nullptr) {   // once: the stream and the four events outlive every rebind
        cudaStream_t s = nullptr;
        if (const cudaError_t e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking); e != cudaSuccess) {
            err = std::string("prefill gemm: dequant side stream: ") + cudaGetErrorString(e);
            side_scratch_ = nullptr;
            return false;
        }
        side_stream_ = s;
        for (int i = 0; i < 2; ++i) {
            cudaEvent_t r = nullptr, f = nullptr;
            const bool made = cudaEventCreateWithFlags(&r, cudaEventDisableTiming) == cudaSuccess &&
                              cudaEventCreateWithFlags(&f, cudaEventDisableTiming) == cudaSuccess;
            side_ready_[i] = r;   // kept even on a failure: the destructor frees what was made
            side_free_[i] = f;
            if (!made) {
                err = "prefill gemm: dequant side events";
                side_scratch_ = nullptr;
                return false;
            }
        }
    }
    // both scratches are free once what the compute stream holds now is done (a rebind follows a stream sync, so
    // this is belt and braces; a wait on an event never recorded would pass anyway)
    for (int i = 0; i < 2; ++i)
        (void) cudaEventRecord((cudaEvent_t) side_free_[i], (cudaStream_t) stream_);
    return true;
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    // #240: every failure names the call and the real status, so "no VRAM" can be told from a broken install
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (const cudaError_t e = cudaMalloc(&workspace_, ws); e != cudaSuccess) {
        err = std::string("prefill gemm: workspace of 32 MiB: ") + cudaGetErrorString(e);
        return false;
    }
    note(cublasSetWorkspace(h, workspace_, ws), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws).release();
#endif
#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
    rocblas_state_ = create_rocblas_state(stream).release();
#endif
    if (scratch_elems > 0) {
        if (const cudaError_t e = cudaMalloc((void**) &scratch_, (size_t) scratch_elems * 2); e != cudaSuccess) {
            err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB: " +
                  cudaGetErrorString(e);
            return false;
        }
    }
    scratch_elems_ = scratch_elems;
    return true;
}

#if !defined(__HIPCC__)
// ---- below sm_80: BF16 GEMMs without BF16 tensor cores (PR #655, #540, #395) ----------------------------------------
// Turing and Volta have no BF16 tensor cores: cublasGemmEx on CUDA_R_16BF inputs falls back to a SIMT fp32 kernel
// (magma_sgemmEx), a fifth of a 4K prompt on an RTX 2080 Ti (#655) and ~10% of a V100's prompt (#540).  BF16 -> FP16
// is exact for every value inside FP16's normal range (the 7-bit mantissa fits in 10 bits); weights and normalized
// activations sit there.  With the FP16 path, the weight is converted once per call and the activations in row
// slices, into the instance's own buffers, and the product runs as the FP16 GEMM (fp32 accumulate) on the tensor
// cores.  Finite values beyond FP16's range are clamped to +-65504 (#540) instead of becoming Inf.  A beta = 1
// product (STRATA_PREFILL_BF16X2's remainder, ~2^-9 of the original, deep in FP16's subnormal band) keeps cuBLAS.
//   Default: on for compute capability 7.0 - 7.4 (Volta: only the experimental STRATA_EXPERIMENTAL_SM60 build runs
//   there), OFF for 7.5 (RTX 20 in the ready-made engine: the sums round differently, so it is opt-in until it has
//   been gated).  STRATA_BF16_TC=1 / =0 turns it on / off on any 7.x card (=2, a test mode: on any card).
// Pascal (6.x) has no tensor cores and cuBLAS has no BF16 GEMM for it (#395, measured NOT_SUPPORTED on a P40): both
// operands are widened to fp32 by an exact shift and the product is cublasSgemm with the same fp32 accumulator.
namespace {
__global__ void bf16_to_f16_kernel(const uint16_t* __restrict__ in, __half* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float f = __uint_as_float((uint32_t) in[i] << 16);
        if (f > 65504.0f && !isinf(f)) f = 65504.0f;
        else if (f < -65504.0f && !isinf(f)) f = -65504.0f;
        out[i] = __float2half_rn(f);
    }
}
void bf16_to_f16(const uint16_t* in, uint16_t* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    bf16_to_f16_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, reinterpret_cast<__half*>(out), n);
}
#if defined(STRATA_EXPERIMENTAL_SM60)   // Pascal runs only the experimental build
__global__ void bf16_to_f32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float((uint32_t) in[i] << 16);
}
void bf16_to_f32(const uint16_t* in, float* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    bf16_to_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, out, n);
}
#endif
// The current device's compute capability as 10 * major + minor, per device (a layer split can mix cards); 0: not
// known, read as "not an old card" so a failed query keeps the cuBLAS BF16 call.
int current_cc() {
    static std::atomic<int> cc[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    int v = cc[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        v = 10 * major + minor;
        cc[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}
// 0: cuBLAS's BF16 GEMM, 1: through FP16 (tensor cores), 2: through FP32 (Pascal)
int bf16_path() {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_BF16_TC");
        return v != nullptr && v[0] != '\0' ? std::atoi(v) : -1;
    }();
    const int cc = current_cc();
    if (forced == 2 && cc > 0) return 1;   // a test mode: the FP16 path on any card (gemm_bf16_parity on sm_80+)
    if (cc <= 0 || cc >= 80) return 0;
    if (cc < 70) return 2;
    if (forced >= 0) return forced != 0 ? 1 : 0;
    return cc < 75 ? 1 : 0;
}
bool grow(uint16_t*& p, int64_t& have, int64_t want) {   // `have`, `want`: 2-byte elements
    if (have >= want) return true;
    if (p) cudaFree(p);
    p = nullptr;
    have = 0;
    if (cudaMalloc((void**) &p, (size_t) want * 2) != cudaSuccess) { cudaGetLastError(); p = nullptr; return false; }
    have = want;
    return true;
}
constexpr int64_t kXSliceElems = 16ll << 20;   // 32 MiB of FP16 activations per slice (64 MiB as fp32)

#if defined(STRATA_EXPERIMENTAL_SM60)
// ---- Pascal, opt-in (STRATA_DENSE_F32=1): the dense projections' FP16-operand GEMMs as cublasSgemm ----------------
// Measured on a Tesla P40 (sm_61) with cuBLAS 12.4.5.8 (bench/results/2026-10-12-p40-sm61-kernels).  For FP16
// operands with FP32 sums cuBLAS has two Maxwell kernels there, maxwell_sgemm_fp16_64x64 and _32x128, and picks the
// slow one for most chunk sizes: the 10240 x 2560 qkv projection runs at 7.6 TFLOPS for T 7030-7040 and at 4.3 from
// T 7041 (49 -> 85 ms), and at 4.3-4.6 for two thirds of the T between 256 and 8192; the 12288-row query projection
// likewise.  A 7K-token prompt reads in 16.3 or 19.5 s by its token count.  cublasSgemm on FP32 copies
// (maxwell_sgemm_128x64) runs at 8.4-9.8 TFLOPS for every T from 512 (6.7+ from 256).  The copies are exact
// (FP16 -> FP32), and for the shapes and sizes taken here the two kernels add every output's products in the same
// order: the result is bitwise the cuBLAS FP16 call's (gemm_f32_parity: 5,201 chunk sizes on a P40 with cuBLAS 12.4, a
// measurement of this library, not a guarantee of cuBLAS's; below the sizes taken here - T under
// ~144 at 6144+ rows, under ~408 at 2560 rows - cuBLAS switches one of them to another kernel family and the bits
// differ, so those keep the FP16 call).  The weight goes through the BF16 path's conversion buffer (126 MiB for the
// largest, 21 MiB more than that path's own), the activations through its slices.
// Opt-in, and on Pascal (compute capability 6.x) only: the equality belongs to the cuBLAS build, so run
// gemm_f32_parity with yours before turning it on; newer cards have tensor cores for the FP16 call.
__global__ void f16_to_f32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(__ushort_as_half(in[i]));
}
void f16_to_f32(const uint16_t* in, float* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    f16_to_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, out, n);
}
int g_dense_f32 = -1;   // gemm_set_dense_f32 (the parity test: 0 / 1 on any card); -1 = STRATA_DENSE_F32
bool dense_f32_on() {
    static const bool env = [] {
        const char* v = std::getenv("STRATA_DENSE_F32");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (g_dense_f32 >= 0) return g_dense_f32 != 0;
    if (!env) return false;
    const int cc = current_cc();
    return cc >= 60 && cc < 70;
}
// The smallest T taken for a weight of N rows and K columns (0: never): the measured shapes only - the 2560-column
// qkv (10240 rows), query (12288) and gate (6144) projections and the 2560-row output projections (6144 and 4096
// columns), each from a T where the FP32 route (its conversions included) is the faster one or within ~7% of the FP16
// call (the gate projection below ~1300 tokens, the 4096-column output projection near 4096) and well above the T
// where the two stop agreeing bitwise.  Every other shape (a row slice of a weight too) keeps the FP16 call.
int64_t dense_f32_min_t(int64_t N, int64_t K) {
    if (K == 2560 && (N == 10240 || N == 12288)) return 256;
    if (K == 2560 && N == 6144) return 512;
    if (N == 2560 && (K == 6144 || K == 4096)) return 4096;
    return 0;
}
#endif
}  // namespace
#endif
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
void gemm_set_dense_f32(int mode) { g_dense_f32 = mode; }
#else
void gemm_set_dense_f32(int) {}
#endif
bool Gemm::bf16_hcd_exact(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K) {
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    static std::atomic<bool> told{false};
    if (!hipblaslt_state_ || N != 320 || K != 10240 || T < 1) {
        if (N == 320 && K == 10240 && !told.exchange(true))
            std::fprintf(stderr, "strata: STRATA_HCD_EXACT: no hipBLASLt tuning table is loaded, so the exact kernel has nothing to match: "
                                 "the hyper-connection down projection stays on hipBLAS\n");
        return false;
    }
    // The kernel copies hipBLASLt solution 1176 / 1177's summation order, which is only true of the library build the
    // table was calibrated with: take it only when the table makes hipBLASLt pick one of those two for this T.
    const int id = hipblaslt_solution_for(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, T, N, K, N, 0.0f);
    if (id != 1176 && id != 1177) {
        if (T >= 4096 && !told.exchange(true))   // (a short chunk may legitimately get another solution: not worth a line)
            std::fprintf(stderr, "strata: STRATA_HCD_EXACT: the tuning table gives the hyper-connection down projection (N 320, K 10240) "
                                 "solution %d at T %lld, not 1176 / 1177: hipBLASLt runs it (a changed table or library; "
                                 "tools/hip/gfx1151-hipblaslt-100401.txt is the one measured)\n",
                         id, (long long) T);
        return false;
    }
    return strata_pf_hcdown_exact_bf16(X, ldx, W, Y, T, N, K, stream_);
#else
    (void) X; (void) ldx; (void) W; (void) Y; (void) T; (void) N; (void) K;
    return false;
#endif
}

#if defined(__HIPCC__)
// ---- RDNA2 (gfx103x): the prompt path's GEMMs as FP32 SGEMMs ------------------------------------------------------
// The GEMMs here take 16-bit inputs (FP16 or BF16) and write FP32 (rocBLAS's "HS" / "BS" types).  The rocBLAS the engine
// ships has tuned gfx1030 kernels for FP16 -> FP16 ("HH"), int8 and FP32 -> FP32 ("SS") only: every 16-bit-input,
// FP32-output GEMM runs a "fallback" kernel at ~5 TFLOPS.  Measured on an RX 6800 (ROCm 10.2.0a20260930) with the
// engine's exact calls (opA = T, opB = N, fp32 compute): at the Coder's prompt shapes the FP16 GEMMs of one fresh
// 12K-token prompt take 14.1 s, the same GEMMs as SGEMM on widened copies 4.7 s including the widening (e.g.
// N 10240 x T 8192 x K 2560: 86.7 -> 27.7 ms); the BF16 hyper-connection GEMMs 10.9 -> 7.5 ms (N 320, K 10240).
// Widening FP16 / BF16 to FP32 is exact and SGEMM accumulates in FP32 like the native call, so the products are the
// same up to summation order.  The weight is widened once per call, the activations in slices of up to 128 MiB of
// FP32 (an 8,192-token chunk at K = 2560 is one slice).  Small N (< 64: routers, gates) keeps the native call:
// there the widening costs more than the GEMM gains.  If the FP32 buffers cannot be allocated, the native call runs.
//   OPT-IN here: STRATA_RDNA2_SGEMM=1 turns it on for gfx103x devices (other cards keep the native call, also in a
//   mixed layer split); =all turns it on for every AMD card (A/B arms, tests); unset or 0 is the native path.  The
//   author ran it as the gfx103x default (RX 6800 343 -> 534 tok/s); it is not bit-identical to the native call.
namespace {
__global__ void rdna2_bf16_to_f32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float((uint32_t) in[i] << 16);
}
__global__ void rdna2_f16_to_f32_kernel(const __half* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}
void rdna2_widen(const uint16_t* in, float* out, int64_t n, bool bf16, cudaStream_t st) {
    if (n <= 0) return;
    const unsigned blocks = (unsigned) ((n + 255) / 256);
    if (bf16) rdna2_bf16_to_f32_kernel<<<blocks, 256, 0, st>>>(in, out, n);
    else rdna2_f16_to_f32_kernel<<<blocks, 256, 0, st>>>(reinterpret_cast<const __half*>(in), out, n);
}
// whether this GEMM runs as an SGEMM (see above)
bool rdna2_sgemm_on(int64_t N) {
    static const int mode = [] {   // 0 off (unset), 1 gfx103x devices, 2 every AMD device
        const char* v = std::getenv("STRATA_RDNA2_SGEMM");
        if (v == nullptr || v[0] == '\0') return 0;
        if (std::strcmp(v, "all") == 0) return 2;
        return std::atoi(v) > 0 ? 1 : 0;
    }();
    if (mode == 0 || N < 64) return false;
    if (mode == 2) return true;
    static std::atomic<int> arch[64] = {};   // per device: 0 not known, 1 gfx103x, 2 another architecture
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) { (void) hipGetLastError(); return false; }
    int v = arch[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) { (void) hipGetLastError(); return false; }
        v = std::strncmp(prop.gcnArchName, "gfx103", 6) == 0 ? 1 : 2;
        arch[dev].store(v, std::memory_order_relaxed);
    }
    return v == 1;
}
// `have`, `want`: FP32 elements (the buffers are the instance's 2-byte tc_ buffers, sized in 2-byte elements)
bool rdna2_grow(uint16_t*& p, int64_t& have_u16, int64_t want_f32) {
    const int64_t want = 2 * want_f32;
    if (have_u16 >= want) return true;
    if (p) cudaFree(p);
    p = nullptr;
    have_u16 = 0;
    if (cudaMalloc((void**) &p, (size_t) want * 2) != cudaSuccess) { (void) cudaGetLastError(); p = nullptr; return false; }
    have_u16 = want;
    return true;
}
constexpr int64_t kRdna2SliceF32 = 32ll << 20;   // 128 MiB of FP32 activations per slice
}  // namespace

// Y^T[N, T] (+)= W[N, K] . X^T[K, T] as SGEMMs on FP32 copies; false: not this card / shape, or no buffers (the caller
// runs its native GEMM, nothing was written)
bool Gemm::rdna2_sgemm(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                       float beta, bool bf16) {
    if (K <= 0 || !rdna2_sgemm_on(N)) return false;
    const int64_t rows = std::max<int64_t>(1, std::min<int64_t>(T, kRdna2SliceF32 / K));
    // growing frees a buffer an earlier GEMM on this stream may still read (as the sm_60 FP32 route does)
    if (tc_w_elems_ < 2 * N * K || tc_x_elems_ < 2 * rows * K) (void) hipStreamSynchronize((hipStream_t) stream_);
    if (!rdna2_grow(tc_w_, tc_w_elems_, N * K) || !rdna2_grow(tc_x_, tc_x_elems_, rows * K)) return false;
    float* const wf = reinterpret_cast<float*>(tc_w_);
    float* const xf = reinterpret_cast<float*>(tc_x_);
    const float alpha = 1.0f;
    rdna2_widen(W, wf, N * K, bf16, (cudaStream_t) stream_);
    for (int64_t t0 = 0; t0 < T; t0 += rows) {
        const int64_t n = std::min<int64_t>(rows, T - t0);
        rdna2_widen(X + t0 * K, xf, n * K, bf16, (cudaStream_t) stream_);
        // every slice is a disjoint block of Y, so each gets the caller's beta
        ck(hipblasSgemm((hipblasHandle_t) handle_, HIPBLAS_OP_T, HIPBLAS_OP_N, (int) N, (int) n, (int) K, &alpha, wf,
                        (int) K, xf, (int) K, &beta, Y + t0 * ldy, (int) ldy),
           "hipblasSgemm (RDNA2)");
    }
    return true;
}
#endif

bool Gemm::bf16_converts_x() const {
#if defined(__HIPCC__) || defined(STRATA_EXPERIMENTAL_SM60)
    return false;
#else
    return bf16_path() == 1;
#endif
}
__global__ void f16_to_bf16_kernel(const __half* __restrict__ in, uint16_t* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (uint16_t) (__float_as_uint(__half2float(in[i])) >> 16);   // exact: the value was a BF16
}
void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta, int64_t ldx, bool x_f16) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (ldx <= K) ldx = 0;
#ifdef STRATA_USE_HIP
    // #313 (opt-in STRATA_WMMA_GEMM=1; STRATA_WMMA_BF16=0 excludes bf16): RDNA3 / RDNA3.5 WMMA dense GEMM, before
    // hipBLASLt; it returns false for shapes it does not take (and off gfx11) and this falls through
    static const bool wmma_on = [] { const char* v = std::getenv("STRATA_WMMA_GEMM"); return v && v[0] != 0 && v[0] != '0'; }();
    static const bool bf16_on = [] { const char* v = std::getenv("STRATA_WMMA_BF16"); return !v || (v[0] != 0 && v[0] != '0'); }();
    if (wmma_on && bf16_on && ldx == 0 && T >= 16 && (beta == 0.0f || beta == 1.0f) &&
        strata_wmma_gemm_bf16(X, W, Y, T, N, K, ldy, beta, stream_)) {
        return;
    }
#endif
#if defined(__HIPCC__)
    if (f16_io_) {   // X is the FP16 image (set_act_f16); W goes through the dequantization scratch as FP16
        // the only accumulating BF16 product is the BF16X2 low part, which bf16x2_mode() turns off with this path
        if (beta != 0.0f) { std::fprintf(stderr, "prefill gemm: an accumulating BF16 GEMM on the FP16 prompt path\n"); std::exit(1); }
        const int64_t rows = std::min<int64_t>(N, scratch_elems_ / K);
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = std::min(rows, N - r0);
            bf16_to_f16_rows<<<1024, 256, 0, (cudaStream_t) stream_>>>(W + r0 * K, (__half*) scratch_, n * K);
            f16_inplace(X, scratch_, Y + r0, T, n, K, ldy);
        }
        return;
    }
#endif
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, X, W, Y, T, N, K, ldy,
                      beta, stream_, ldx)) {
        STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt bf16");
        return;
    }
#endif
#if defined(__HIPCC__)
    // a padded X (STRATA_PF_PAD, ldx > K) keeps the native call: the widening copies contiguous rows
    if (ldx == 0 && rdna2_sgemm(X, W, Y, T, N, K, ldy, beta, true)) return;
#endif
#if !defined(__HIPCC__)
#if defined(STRATA_EXPERIMENTAL_SM60)
    constexpr bool kPascalPath = true;
#else
    constexpr bool kPascalPath = false;
#endif
    if (const int path = K > 0 ? bf16_path() : 0; (path == 1 && N > 1 && beta == 0.0f) || (path == 2 && kPascalPath)) {
        // FP16 (path 1) or FP32 (path 2, Pascal) copies of the weight and of X, converted in tiles.  Issue #1757: when
        // the buffers did not fit in the VRAM that was left (a stage of a layer split keeps ~100 MiB), this fell through
        // to the raw BF16 cuBLAS call, which a Pascal card answers with CUBLAS_STATUS_INTERNAL_ERROR (14).  Now the
        // weight is converted in tiles of N rows (the whole weight when it fits, as before) and X in slices; a card with
        // no room even for a tile says so.  A single output row stays cuBLAS's GEMV for path 1 (faster than the
        // conversions).  Every tile is a disjoint block of Y, so each gets the caller's beta.
        const int mult = path == 2 ? 2 : 1;   // 2-byte elements per converted value
        int64_t w_rows = N;
        int64_t x_rows = std::max<int64_t>(1, std::min<int64_t>(T, kXSliceElems / K));
        bool fit = false;
        for (;;) {
            if (grow(tc_w_, tc_w_elems_, mult * w_rows * K) && grow(tc_x_, tc_x_elems_, mult * x_rows * K)) { fit = true; break; }
            if (w_rows <= 64 && x_rows <= 64) break;
            w_rows = std::max<int64_t>(std::min<int64_t>(w_rows, 64), w_rows / 2);
            x_rows = std::max<int64_t>(std::min<int64_t>(x_rows, 64), x_rows / 2);
        }
        if (!fit && path == 2) {   // Pascal has no BF16 cuBLAS call to fall back to; Volta keeps its raw one below
            std::fprintf(stderr, "prefill gemm: no GPU memory left for the %s conversion buffers of a %lld x %lld BF16 weight "
                                 "(Pascal cards run the BF16 products through them); free VRAM on this card: raise "
                                 "--vram-reserve-mib, lower --max-context or --kv-resident, or give it fewer layers\n",
                         "FP32", (long long) N, (long long) K);
            std::exit(1);
        }
        if (fit) {
            for (int64_t n0 = 0; n0 < N; n0 += w_rows) {
                const int64_t nn = std::min<int64_t>(w_rows, N - n0);
                if (path == 2) {
#if defined(STRATA_EXPERIMENTAL_SM60)
                    float* const wf = reinterpret_cast<float*>(tc_w_);
                    float* const xf = reinterpret_cast<float*>(tc_x_);
                    bf16_to_f32(W + n0 * K, wf, nn * K, (cudaStream_t) stream_);
                    for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                        const int64_t n = std::min<int64_t>(x_rows, T - t0);
                        bf16_to_f32(X + t0 * K, xf, n * K, (cudaStream_t) stream_);
                        ck(cublasSgemm((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) nn, (int) n, (int) K, &alpha,
                                       wf, (int) K, xf, (int) K, &beta, Y + t0 * ldy + n0, (int) ldy),
                           "cublasSgemm (bf16 on Pascal)");
                    }
#endif
                } else {
                    bf16_to_f16(W + n0 * K, tc_w_, nn * K, (cudaStream_t) stream_);
                    for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                        const int64_t n = std::min<int64_t>(x_rows, T - t0);
                        if (x_f16) {   // already the converted values (the same slices: the same GEMMs)
                            f16(X + t0 * K, tc_w_, Y + t0 * ldy + n0, n, nn, K, ldy, beta);
                        } else {
                            bf16_to_f16(X + t0 * K, tc_x_, n * K, (cudaStream_t) stream_);
                            f16(tc_x_, tc_w_, Y + t0 * ldy + n0, n, nn, K, ldy, beta);
                        }
                    }
                }
            }
            return;
        }
    }
#endif
#if !defined(__HIPCC__)
    if (x_f16) {   // not the tensor-core path after all (one output row, or no room): X back to BF16, a slice at a time
        const int64_t x_rows = std::max<int64_t>(1, std::min<int64_t>(T, kXSliceElems / K));
        if (!grow(tc_x_, tc_x_elems_, x_rows * K)) {
            std::fprintf(stderr, "prefill gemm: no GPU memory for converting an FP16 image back to BF16\n");
            std::exit(1);
        }
        for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
            const int64_t n = std::min<int64_t>(x_rows, T - t0);
            f16_to_bf16_kernel<<<(unsigned) ((n * K + 255) / 256), 256, 0, (cudaStream_t) stream_>>>(
                reinterpret_cast<const __half*>(X + t0 * K), tc_x_, n * K);
            bf16(tc_x_, W, Y + t0 * ldy, n, N, K, ldy, beta, 0, false);
        }
        return;
    }
#endif
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) (ldx ? ldx : K), &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx");
    STRATA_ABSORB_HIPBLAS_STICKY("cublasGemmEx");
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
#ifdef STRATA_USE_HIP
    // #313 (opt-in STRATA_WMMA_GEMM=1): RDNA3 / RDNA3.5 WMMA dense GEMM, before hipBLASLt (falls through when false)
    static const bool wmma_on = [] { const char* v = std::getenv("STRATA_WMMA_GEMM"); return v && v[0] != 0 && v[0] != '0'; }();
    static const bool pf_on = [] { const char* v = std::getenv("STRATA_PF_GEMM"); return v && v[0] == '1'; }();
    static const int64_t pf_min_t = [] { const char* v = std::getenv("STRATA_PF_SWITCH_MIN_T"); return v ? (int64_t) std::atoll(v) : (int64_t) 0; }();
    if (pf_on && T >= pf_min_t && strata_pf_gemm_f16(X, W, Y, T, N, K, ldy, beta, stream_)) return;   // S23: opt-in
    if (wmma_on && T >= 16 && (beta == 0.0f || beta == 1.0f) &&
        strata_wmma_gemm_f16(X, W, Y, T, N, K, ldy, beta, stream_)) {
        return;
    }
#endif
#if defined(__HIPCC__)
    if (f16_io_ && beta == 0.0f) { f16_inplace(X, W, Y, T, N, K, ldy); return; }
#endif
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::f16, X, W, Y, T, N, K, ldy,
                      beta, stream_)) {
        STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt f16");
        return;
    }
#endif
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
    // Pascal, STRATA_DENSE_F32=1: cublasSgemm on FP32 copies (dense_f32_on above: where it is bitwise the FP16 call)
    if (const int64_t min_t = beta == 0.0f ? dense_f32_min_t(N, K) : 0; min_t > 0 && T >= min_t && dense_f32_on()) {
        const int64_t n_sl = (T * K + kXSliceElems - 1) / kXSliceElems;   // even slices of at most 64 MiB as fp32
        const int64_t x_rows = (T + n_sl - 1) / n_sl;
        // grow() frees a buffer an earlier GEMM on this stream may still read
        if (tc_w_elems_ < 2 * N * K || tc_x_elems_ < 2 * x_rows * K) cudaStreamSynchronize((cudaStream_t) stream_);
        if (grow(tc_w_, tc_w_elems_, 2 * N * K) && grow(tc_x_, tc_x_elems_, 2 * x_rows * K)) {
            static bool said = false;
            if (!said) {
                said = true;
                std::fprintf(stderr, "prefill gemm: the dense projections' FP16 GEMMs run as cublasSgemm on FP32 copies "
                                     "(STRATA_DENSE_F32=1; gemm_f32_parity checks the bits against this cuBLAS)\n");
            }
            float* const wf = reinterpret_cast<float*>(tc_w_);
            float* const xf = reinterpret_cast<float*>(tc_x_);
            f16_to_f32(W, wf, N * K, (cudaStream_t) stream_);
            for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                const int64_t n = std::min<int64_t>(x_rows, T - t0);
                f16_to_f32(X + t0 * K, xf, n * K, (cudaStream_t) stream_);
                ck(cublasSgemm((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) n, (int) K, &alpha, wf, (int) K,
                               xf, (int) K, &beta, Y + t0 * ldy, (int) ldy),
                   "cublasSgemm (FP16 operands, STRATA_DENSE_F32)");
            }
            return;
        }
        cudaGetLastError();   // no room for the copies: the FP16 call below
    }
#endif
#if defined(__HIPCC__)
    if (rdna2_sgemm(X, W, Y, T, N, K, ldy, beta, false)) return;
#endif
#if !defined(__HIPCC__)
    const auto f16_gemm = [&](cublasGemmAlgo_t algo) {
        return cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha,
                            W, CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                            CUBLAS_COMPUTE_32F, algo);
    };
    // Issue #1650: with a second CUDA device initialized in the process, cuBLAS 12 answers the default algorithm of a
    // few shapes - narrow ones, and the set moves with K - with CUBLAS_STATUS_INTERNAL_ERROR; a fixed algorithm keeps
    // a kernel that state does not break.  Only the shapes that fail are moved (the failing call itself is harmless);
    // the wide shapes, for which the fixed algorithms are markedly slower, stay on the default.
    const int64_t shape = ((int64_t) K << 32) | (uint32_t) N;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    for (int i = 0; i < f16_algo_n_; ++i)
        if (f16_algo_shape_[i] == shape) { algo = CUBLAS_GEMM_ALGO2; break; }
    cublasStatus_t st = f16_gemm(algo);
    if (st == CUBLAS_STATUS_INTERNAL_ERROR && algo == CUBLAS_GEMM_DEFAULT) {
        if (f16_algo_n_ < kF16AlgoSlots) f16_algo_shape_[f16_algo_n_++] = shape;
        st = f16_gemm(CUBLAS_GEMM_ALGO2);
    }
    ck(st, "cublasGemmEx f16");
#else
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16");
#endif
    STRATA_ABSORB_HIPBLAS_STICKY("cublasGemmEx f16");
}

#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
bool Gemm::native_mmq(const uint16_t* X, int type, const void* W, float* Y, int64_t T, int64_t N, int64_t K,
                      int64_t ldy) {
    namespace mmq = strata::prefill::mmq;
    constexpr int64_t kRows = 1024, kMaxK = 8192;
    static const bool enabled = [] {
        const char* e = std::getenv("STRATA_DENSE_MMQ");
        return e && e[0] == '1';
    }();
    if (!enabled || mmq_failed_ || !mmq::built() || !mmq::fits(type, N)) return false;
    // K must be a multiple of 256: llama.cpp's MMQ loads the weights in 256-value K chunks, and the
    // chunk past a partial row reads past the row (the next row's bytes - or, for the last weight
    // row, bytes past the tensor, which no caller guarantees to be zeros). The MoE path is safe
    // because its gather buffers carry a zeroed tail; a dense tensor is passed as-is, so a K that
    // is not a full chunk multiple (e.g. the 640-value shared-expert down: 20 blocks of 32) stays on
    // the dequantize + cuBLAS path.
    if (K > kMaxK || K % 256 != 0) return false;
    int64_t R = scratch_elems_ / (2 * K);   // FP32 activations for a chunk fit in the FP16 dequant scratch
    if (R > kRows) R = kRows;
    if (R > T) R = T;
    if (R < 1) return false;
    if (ldy <= 0) ldy = N;

    const auto up = [](size_t v) { return (v + 255) & ~(size_t) 255; };
    const size_t ident_off = up(mmq::q8_bytes(kRows, kMaxK));
    const size_t bounds_off = ident_off + up((size_t) kRows * 4);
    if (!mmq_buf_) {
        if (cudaMalloc(&mmq_buf_, bounds_off + 256) != cudaSuccess) {
            cudaGetLastError();
            mmq_failed_ = true;
            return false;
        }
        mmq_ctx_ = new mmq::Context();
        mmq::iota((int32_t*) ((uint8_t*) mmq_buf_ + ident_off), kRows, stream_);
        std::fprintf(stderr, "prefill gemm: dense MMQ on (STRATA_DENSE_MMQ=1)\n");
    }
    void* xq = mmq_buf_;
    const int32_t* ident = (const int32_t*) ((uint8_t*) mmq_buf_ + ident_off);
    int32_t* bounds = (int32_t*) ((uint8_t*) mmq_buf_ + bounds_off);
    float* xf = (float*) scratch_;
    auto* ctx = static_cast<mmq::Context*>(mmq_ctx_);

    for (int64_t r0 = 0; r0 < T; r0 += R) {
        const int64_t rows = (T - r0 < R) ? T - r0 : R;
        mmq::f16_to_f32(X + r0 * K, xf, rows * K, stream_);
        mmq::quantize(xf, nullptr, xq, type, K, K, rows, stream_);
        mmq::set_bounds(bounds, (int32_t) rows, stream_);
        mmq::Product p;
        p.w = W;
        p.type = type;
        p.w_rows = N;
        p.w_cols = K;
        p.expert_bytes = mmq::matrix_bytes(type, N, K);
        p.n = 1;
        p.xq = xq;
        p.bounds = bounds;
        p.ids = ident;
        p.total_rows = rows;
        p.max_rows = rows;
        p.dst = Y + r0 * ldy;
        p.ld_dst = ldy;
        ctx->run(p, stream_);
    }
    return true;
}
#endif
void Gemm::f16_inplace(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy) {
#if defined(__HIPCC__)
    const float one = 1.0f, zero = 0.0f;
    bool done = false;
#if defined(STRATA_ROCBLAS_AVAILABLE)
    done = try_rocblas_f16o(rocblas_state_, X, W, Y, T, N, K, ldy);   // the table's kernel for this shape, if any
#endif
    if (!done) {
        ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, W,
                        CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &zero, Y, CUDA_R_16F, (int) (2 * ldy),
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
           "cublasGemmEx f16 out");
    }
    static const bool dbg_nan = std::getenv("STRATA_DBG_NAN") != nullptr;
    widen_rows_f16<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(Y, N, ldy, dbg_nan ? 1 : 0);
    if (dbg_nan) {   // debug only: a sync per GEMM
        unsigned long long bad = 0, zero_count = 0;
        (void) hipStreamSynchronize((hipStream_t) stream_);
        if (hipMemcpyFromSymbol(&bad, HIP_SYMBOL(g_f16_nonfinite), sizeof bad) == hipSuccess && bad != 0) {
            std::fprintf(stderr, "strata prefill: STRATA_DBG_NAN: %llu non-finite FP16 GEMM outputs (T=%lld N=%lld K=%lld; FP16 ends at 65504)%s",
                         bad, (long long) T, (long long) N, (long long) K, "\n");
            (void) hipMemcpyToSymbol(HIP_SYMBOL(g_f16_nonfinite), &zero_count, sizeof zero_count);
        }
    }
#else
    (void) X; (void) W; (void) Y; (void) T; (void) N; (void) K; (void) ldy;
#endif
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta, int64_t ldx) {
#ifdef STRATA_USE_HIP
    // S23 (opt-in STRATA_PF_PAD=1, with STRATA_PF_GEMM=1 on chunks of STRATA_PF_SWITCH_MIN_T+ tokens): the weight
    // dequantized into the scratch with row stride K + 64 halves, so no 4 KB-multiple stride camps on the memory
    // channels (gemm_probe8: K 6144 19 -> 29 TFLOPS from W alone, 35 with X padded too).  Bitwise the same products.
    static const bool pad_on = [] { const char* v = std::getenv("STRATA_PF_PAD"); return v && v[0] == '1'; }();
    static const bool pf_on = [] { const char* v = std::getenv("STRATA_PF_GEMM"); return v && v[0] == '1'; }();
    static const int64_t pf_min_t = [] { const char* v = std::getenv("STRATA_PF_SWITCH_MIN_T"); return v ? (int64_t) std::atoll(v) : (int64_t) 0; }();
    // (strata_pf_padded_gemm_available(): the CURRENT device runs the padded gfx11 product.  On a gfx1151 + gfx1201 split the process-wide switch
    // is on for both stages; the gfx12 stage would dequantize the weight padded, find no product, and do it again unpadded below.)
    if (pad_on && pf_on && T >= pf_min_t && T >= 64 && N >= 512 && K % 32 == 0 && N * (K + 64) <= scratch_elems_ &&
        strata_pf_padded_gemm_available()) {
        const int64_t ldw = K + 64;
        if (strata::kernels::dequant_f16_ld(ggml_type, W_blocks, 0, N, K, ldw, scratch_, stream_) &&
            strata_pf_gemm_f16_ld(X, ldx > 0 ? ldx : K, scratch_, ldw, Y, T, N, K, ldy, beta, stream_)) {
            return;
        }
    }
#endif
    if (ldx > 0 && ldx != K) {
#ifdef STRATA_USE_HIP
        // X arrived padded but the weight cannot be (an i-quant without a strided dequant, or the scratch is too
        // small for N x (K + 64)): the weight unpadded, X read at its stride - the same products, the same bits
        const int64_t rows = scratch_elems_ / K;
        bool ok = rows > 0 && strata_pf_padded_gemm_available();
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; ok && r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            ok = strata_pf_gemm_f16_ld(X, ldx, scratch_, K, Y + r0, T, n, K, ldy, beta, stream_);
        }
        if (ok) return;
#endif
        // Nothing took a padded X: the padded product is the gfx11 matrix cores' (STRATA_PF_PAD asks
        // strata_pf_padded_gemm_available() before padding now, so this is a stale producer or another card's
        // activation).  Copy the rows unpadded and run the ordinary paths - the same products, the same bits - rather
        // than ending the engine.
        const uint16_t* Xu = unpad_x(X, T, K, ldx);
        if (Xu != nullptr) {
            native(Xu, ggml_type, W_blocks, Y, T, N, K, ldy, beta, 0);
            return;
        }
        std::fprintf(stderr, "prefill gemm: a padded X (ldx %lld, K %lld) has no path on this card and could not be "
                             "copied unpadded (no device memory)\n", (long long) ldx, (long long) K);
        std::exit(1);
    }
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
    if (beta == 0.0f && (ldx == 0 || ldx == K) && native_mmq(X, ggml_type, W_blocks, Y, T, N, K, ldy)) return;
#endif
#ifndef STRATA_USE_HIP
    // STRATA_PF_W16_CACHE_MIB=N: the FP16 weight kept on the device after its first dequant (same values, same product)
    if (w16_budget_ < 0) {
        const char* v = std::getenv("STRATA_PF_W16_CACHE_MIB");
        w16_budget_ = v ? (int64_t) std::atoll(v) * (1ll << 20) : 0;
    }
    if (w16_budget_ > 0 && !side_scratch_) {
        const int64_t bytes = N * K * 2;
        uint16_t* cw = nullptr;
        for (const W16Entry& e : w16_)
            if (e.p == W_blocks && e.type == ggml_type && e.n == N && e.k == K) { cw = e.d; break; }
        if (cw == nullptr && w16_used_ + bytes <= w16_budget_) {
            size_t fb = 0, tb = 0;
            if (cudaMemGetInfo(&fb, &tb) == cudaSuccess && fb > (size_t) bytes + (2ull << 30) &&
                cudaMalloc((void**) &cw, (size_t) bytes) == cudaSuccess) {
                strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, cw, stream_);
                w16_.push_back({W_blocks, ggml_type, N, K, cw});
                w16_used_ += bytes;
            } else {
                (void) cudaGetLastError();
                cw = nullptr;
                w16_budget_ = w16_used_ > 0 ? w16_used_ : 0;   // out of room: stop trying
            }
        }
        if (cw != nullptr) {
            f16(X, cw, Y, T, N, K, ldy, beta);
            return;
        }
    }
#endif
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        // the side stream's next dequant into scratch_ waits for these slices' reads and writes of it
        if (side_scratch_) (void) cudaEventRecord((cudaEvent_t) side_free_[0], (cudaStream_t) stream_);
        return;
    }
    if (side_scratch_) {
        // STRATA_DEQUANT_SIDE: the dequant on the side stream, into the scratch the compute stream read last but one.
        // free[cur] -> dequant -> ready[cur] -> GEMM -> free[cur] orders the dequant after the GEMM that last read
        // this scratch and the GEMM after the dequant that filled it, so every read and write of a scratch is where
        // the serial form had it - the same kernels, the same bits - and the dequant runs while the kernels queued
        // ahead of this GEMM do.
        uint16_t* const s = side_cur_ == 0 ? scratch_ : side_scratch_;
        (void) cudaStreamWaitEvent((cudaStream_t) side_stream_, (cudaEvent_t) side_free_[side_cur_], 0);
        strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, s, side_stream_);
        (void) cudaEventRecord((cudaEvent_t) side_ready_[side_cur_], (cudaStream_t) side_stream_);
        (void) cudaStreamWaitEvent((cudaStream_t) stream_, (cudaEvent_t) side_ready_[side_cur_], 0);
        f16(X, s, Y, T, N, K, ldy, beta);
        (void) cudaEventRecord((cudaEvent_t) side_free_[side_cur_], (cudaStream_t) stream_);
        side_cur_ ^= 1;
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill
