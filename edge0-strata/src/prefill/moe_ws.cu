// src/prefill/moe_ws.cu - see include/strata/prefill/moe_ws.hpp: which layers and products take the weight-stationary
// kernels (moe_ws_gu_iq2.cu, moe_ws_gu_iq3.cu, moe_ws_down.cu), their work tables and launches.
#include "common.cuh"
#include "mmq.cuh"

#include "moe_ws.cuh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::prefill::ws {
namespace {

// ws / per-group MMQ (one 512-expert top-10 layer, lognormal counts sigma 0.5 and 1.0, the worse of the two; isolated
// sustained medians measured on a GP100 die, the variant moe_ws_*.cu picks) at 2.5, 10, 35 and 160 rows per routed
// expert
struct Measured { int type; float r[4]; };
constexpr double kU[4] = {2.5, 10.0, 35.0, 160.0};
constexpr Measured kMeasured[] = {
    {GGML_TYPE_IQ2_XXS, {0.276f, 0.494f, 0.728f, 0.754f}},
    {GGML_TYPE_IQ2_XS, {0.299f, 0.513f, 0.750f, 0.824f}},
    {GGML_TYPE_IQ2_S, {0.317f, 0.520f, 0.754f, 0.816f}},
    {GGML_TYPE_IQ3_XXS, {0.329f, 0.514f, 0.755f, 0.786f}},
    {GGML_TYPE_IQ3_S, {0.347f, 0.521f, 0.715f, 0.793f}},
    {GGML_TYPE_Q2_0, {0.246f, 0.389f, 0.555f, 0.614f}},
    {GGML_TYPE_IQ4_NL, {0.292f, 0.492f, 0.682f, 0.742f}},
};

const Measured* measured(int t) {
    for (const Measured& m : kMeasured)
        if (m.type == t) return &m;
    return nullptr;
}
bool gu_type(int t) {
    return t == GGML_TYPE_IQ2_XXS || t == GGML_TYPE_IQ2_XS || t == GGML_TYPE_IQ2_S || t == GGML_TYPE_IQ3_XXS ||
           t == GGML_TYPE_IQ3_S;
}

}  // namespace

namespace {
std::atomic<int> g_mode_override{-1};   // set_mode (tests); -1: the environment
}  // namespace

void set_mode(int v) { g_mode_override.store(v < 0 ? -1 : (v != 0 ? 1 : 0), std::memory_order_relaxed); }

int mode() {
#if defined(__HIPCC__)
    return 0;
#else
    const int o = g_mode_override.load(std::memory_order_relaxed);
    if (o >= 0) return o;
    static const int m = [] {
        const char* e = std::getenv("STRATA_MMQ_WS");   // opt-in
        if (e == nullptr || std::atoi(e) != 1) return 0;
        // GP100 (cc 6.0) only: tuned and checked there (any visible GPU that is not: off for all)
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) { cudaGetLastError(); return 0; }
        for (int d = 0; d < n; ++d) {
            cudaDeviceProp p{};
            if (cudaGetDeviceProperties(&p, d) != cudaSuccess || p.major != 6 || p.minor != 0) { cudaGetLastError(); return 0; }
        }
        return 1;
    }();
    return m;
#endif
}

bool supported(int gt, int dt) { return mode() != 0 && gu_type(gt) && (dt == GGML_TYPE_Q2_0 || dt == GGML_TYPE_IQ4_NL); }

namespace {
std::atomic<int> g_fast_override{-1};   // set_fast (benches, tests); -1: the environment
}  // namespace

int fast() {
    const int o = g_fast_override.load(std::memory_order_relaxed);
    if (o >= 0) return o;
    static const int f = [] {
        const char* e = std::getenv("STRATA_WS_FAST");   // opt-in
        return e != nullptr && std::atoi(e) == 1 ? 1 : 0;
    }();
    return f;
}

void set_fast(int v) { g_fast_override.store(v < 0 ? -1 : (v != 0 ? 1 : 0), std::memory_order_relaxed); }

bool faster(int t, double u) {
    const Measured* m = measured(t);
    if (m == nullptr) return false;
    // linear in log u between the measured points, the ends held
    if (u <= kU[0]) return m->r[0] < 1.0f;
    for (int i = 1; i < 4; ++i)
        if (u <= kU[i]) {
            const double f = std::log(u / kU[i - 1]) / std::log(kU[i] / kU[i - 1]);
            return m->r[i - 1] + f * (m->r[i] - m->r[i - 1]) < 1.0;
        }
    return m->r[3] < 1.0f;
}

size_t chunks(const int32_t* order, size_t n, const int32_t* cnt, const int32_t* bounds, Chunk* out) {
    // largest expert first: MMQ's order is ascending by count where the resident sort ran (walked backwards), else
    // by id (sorted here)
    bool asc = true;
    for (size_t j = 1; j < n && asc; ++j) asc = cnt[order[j - 1]] <= cnt[order[j]];
    static thread_local std::vector<int32_t> by;
    by.resize(n);
    for (size_t j = 0; j < n; ++j) by[j] = (int32_t) (n - 1 - j);
    if (!asc)
        std::stable_sort(by.begin(), by.end(), [&](int32_t a, int32_t b) { return cnt[order[a]] > cnt[order[b]]; });
    size_t k = 0;
    for (const int32_t j : by) {
        const int32_t c = cnt[order[j]], b = bounds[j];
        for (int32_t tk = 0; tk < c; tk += kChunkRows) out[k++] = Chunk{j, b + tk, std::min(kChunkRows, c - tk), 0};
    }
    return k;
}

bool gate_up(const Layer& L, void* stream) {
    if (L.n <= 0 || L.nchunks <= 0) return true;
    const double u = (double) L.rows / L.n;
    if (mode() != 1 || !faster(L.gu_type, u)) return false;
    const cudaStream_t s = (cudaStream_t) stream;
    switch (L.gu_type) {
        case GGML_TYPE_IQ2_XXS: detail::gu_iq2_xxs(L, u, s); break;
        case GGML_TYPE_IQ2_XS: detail::gu_iq2_xs(L, u, s); break;
        case GGML_TYPE_IQ2_S: detail::gu_iq2_s(L, u, s); break;
        case GGML_TYPE_IQ3_XXS: detail::gu_iq3_xxs(L, u, s); break;
        case GGML_TYPE_IQ3_S: detail::gu_iq3_s(L, u, s); break;
        default:
            std::fprintf(stderr, "prefill ws: gate/up type %d has no kernel\n", L.gu_type);
            std::exit(1);
    }
    return true;
}

void down(const Layer& L, void* stream) {
    if (L.n <= 0 || L.nchunks <= 0) return;
    const double u = (double) L.rows / L.n;
    const cudaStream_t s = (cudaStream_t) stream;
    switch (L.d_type) {
        case GGML_TYPE_Q2_0: detail::dn_q2_0(L, u, s); break;
        case GGML_TYPE_IQ4_NL: {
            const bool f = fast() == 1;
            if (f) {   // once per process: the switch is on and the WS path runs
                static const bool said = [] {
                    std::fprintf(stderr, "strata prefill: STRATA_WS_FAST=1: the weight-stationary IQ4_NL down's 32-token "
                                         "tier with switched token pairs and transposed dy (the same Dm bits)\n");
                    return true;
                }();
                (void) said;
            }
            detail::dn_iq4_nl(L, u, s, f);
            break;
        }
        default:
            std::fprintf(stderr, "prefill ws: down type %d has no kernel\n", L.d_type);
            std::exit(1);
    }
}

}  // namespace strata::prefill::ws
