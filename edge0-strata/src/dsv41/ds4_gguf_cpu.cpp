// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_gguf_cpu.cpp - a GGUF's MXFP4 experts on the CPU lane (see ds4_gguf_cpu.hpp).
#include "ds4_gguf_cpu.hpp"

#include "strata/dsv41/exl3_cpu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#define MX_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

// a function's instruction set (GCC / Clang, clang-cl too; MSVC compiles any intrinsic as it is): the file is built for
// the baseline x86-64, and each tier's code runs only where detect() says the CPU and the OS take it
#if defined(MX_X86) && (defined(__GNUC__) || defined(__clang__))
#define MX_TARGET(s) __attribute__((target(s)))
#else
#define MX_TARGET(s)
#endif
#define MX_AVX2 MX_TARGET("avx2,fma")
// AVX-VNNI's intrinsics: GCC 11, Clang 12, MSVC 2022
#if defined(MX_X86) && (defined(_MSC_VER) || (defined(__clang__) && __clang_major__ >= 12) || \
                        (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 11))
#define MX_AVXVNNI 1
#endif

namespace strata::ds4::gguf {
namespace {

constexpr int kBlock = 32, kBytes = 17;
constexpr float kValues[16] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
                               0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f};
constexpr int kCodes[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

float e8m0_half(uint32_t e) {   // 2^(e - 128), as the device's
    const uint32_t bits = e < 2 ? 0x00200000u << e : (e - 1) << 23;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
struct E8m0Table {
    float v[256];
    E8m0Table() {
        for (int e = 0; e < 256; ++e) v[e] = e8m0_half((uint32_t) e);
    }
};
const E8m0Table& e8() {
    static const E8m0Table t;
    return t;
}

void rows_scalar(const uint8_t* w, size_t rb, int k, int r0, int r1, const float* const* x, int n, float* const* y) {
    for (int r = r0; r < r1; ++r) {
        const uint8_t* p = w + (size_t) r * rb;
        float acc[kMxCpuMaxN] = {};
        for (int b = 0; b < k / kBlock; ++b, p += kBytes) {
            const float d = e8m0_half(p[0]);
            float wv[kBlock];
            for (int j = 0; j < 16; ++j) {
                wv[j] = kValues[p[1 + j] & 15] * d;
                wv[j + 16] = kValues[p[1 + j] >> 4] * d;
            }
            for (int i = 0; i < n; ++i) {
                const float* xb = x[i] + b * kBlock;
                float s = 0.0f;
                for (int j = 0; j < kBlock; ++j) s += wv[j] * xb[j];
                acc[i] += s;
            }
        }
        for (int i = 0; i < n; ++i) y[i][r - r0] = acc[i];
    }
}

#if defined(MX_X86)
MX_AVX2 float hsum(__m256 v) {
    const __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 b = _mm_add_ps(a, _mm_movehl_ps(a, a));
    return _mm_cvtss_f32(_mm_add_ss(b, _mm_shuffle_ps(b, b, 1)));
}

// a block: the 16 code bytes -> 32 int8 values through a shuffle table (low nibbles: values 0..15, high: 16..31),
// widened to F32 in 4 x 8, times the block's scale (exact), FMAs against each activation
MX_AVX2 void rows_avx2(const uint8_t* w, size_t rb, int k, int r0, int r1, const float* const* x, int n,
                       float* const* y) {
    const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* p = w + (size_t) r * rb;
        __m256 acc[kMxCpuMaxN];
        for (int i = 0; i < n; ++i) acc[i] = _mm256_setzero_ps();
        for (int b = 0; b < k / kBlock; ++b, p += kBytes) {
            const __m256 d = _mm256_set1_ps(e8m0_half(p[0]));
            const __m128i q = _mm_loadu_si128((const __m128i*) (p + 1));
            const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q, m4));
            const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
            const __m256 w0 = _mm256_mul_ps(d, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)));
            const __m256 w1 = _mm256_mul_ps(d, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))));
            const __m256 w2 = _mm256_mul_ps(d, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)));
            const __m256 w3 = _mm256_mul_ps(d, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))));
            for (int i = 0; i < n; ++i) {
                const float* xb = x[i] + b * kBlock;
                __m256 a = acc[i];
                a = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xb), a);
                a = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xb + 8), a);
                a = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xb + 16), a);
                a = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xb + 24), a);
                acc[i] = a;
            }
        }
        for (int i = 0; i < n; ++i) y[i][r - r0] = hsum(acc[i]);
    }
}

// ---- the int8 path's vector tiers (ds4_mx_kern.inc)
#define MXK_NS mx_avx2
#define MXK_ATTR MX_TARGET("avx2,fma")
#define MXK_256 1
#include "ds4_mx_kern.inc"
#if defined(MX_AVXVNNI)
#define MXK_NS mx_avxvnni
#define MXK_ATTR MX_TARGET("avx2,fma,avxvnni")
#define MXK_256 1
#define MXK_VNNI 1
#include "ds4_mx_kern.inc"
#endif
#define MXK_NS mx_avx512
#define MXK_ATTR MX_TARGET("avx512f,avx512bw,fma")
#define MXK_512 1
#include "ds4_mx_kern.inc"
#define MXK_NS mx_avx512vnni
#define MXK_ATTR MX_TARGET("avx512f,avx512bw,avx512vnni,fma")
#define MXK_512 1
#define MXK_VNNI 1
#include "ds4_mx_kern.inc"
#endif

// ---- the int8 path's scalar tier: ds4_mx_kern.inc's order (8 accumulators, block j of a group of 8 into L(j)), so it
// gives the vector tiers' bits - also the tests' check on them.  The activation's layout (ds4_mx_kern.inc), for any k
// % 32 == 0: the int8 values in quads (a partial last quad padded), then per group of 8 blocks the scales and the
// offset corrections in lane order (a partial last group padded)
inline int act_lane(int b) { return 8 * (b >> 3) + ((b & 7) & 1) * 4 + ((b & 7) >> 1); }
inline size_t act_qbytes(int k) { return (size_t) 128 * (size_t) ((k / kBlock + 3) / 4); }
inline size_t act_groups(int k) { return (size_t) ((k / kBlock + 7) / 8); }
inline int act_q(const uint8_t* act, int b, int j) {   // block b's value j (0..31)
    return ((const int8_t*) act)[(b >> 2) * 128 + (j >= 16 ? 64 : 0) + 16 * (b & 3) + (j & 15)];
}
inline float act_d(const uint8_t* act, int k, int b) { return ((const float*) (act + act_qbytes(k)))[act_lane(b)]; }

void quant_scalar(const float* x, int b0, int b1, uint8_t* act, int k) {
    int8_t* q = (int8_t*) act;
    float* dx = (float*) (act + act_qbytes(k));
    int32_t* corr = (int32_t*) (act + act_qbytes(k) + 32 * act_groups(k));
    for (int b = b0; b < b1; ++b, x += kBlock) {
        float am = 0.0f;
        for (int j = 0; j < kBlock; ++j) am = std::max(am, std::fabs(x[j]));
        const float sc = am / 127.0f;
        int8_t v[kBlock];
        int s = 0;
        for (int j = 0; j < kBlock; ++j) {
            v[j] = am > 0.0f ? (int8_t) (int) std::nearbyint(x[j] / sc) : (int8_t) 0;
            s += v[j];
        }
        int8_t* lo = q + (b >> 2) * 128 + 16 * (b & 3);
        for (int j = 0; j < 16; ++j) {
            lo[j] = v[j];
            lo[64 + j] = v[16 + j];
        }
        dx[act_lane(b)] = sc;
        corr[act_lane(b)] = -12 * s;
    }
}

void rows_q_scalar(const uint8_t* w, size_t rb, int k, int r0, int r1, const void* const* act, int n, float* const* y) {
    const float* t = e8().v;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* p = w + (size_t) r * rb;
        float acc[kMxCpuMaxN][8] = {};
        for (int b = 0; b < k / kBlock; ++b, p += kBytes) {
            int wv[kBlock];
            for (int j = 0; j < 16; ++j) {
                wv[j] = kCodes[p[1 + j] & 15];
                wv[j + 16] = kCodes[p[1 + j] >> 4];
            }
            const int l = act_lane(b) & 7;
            for (int i = 0; i < n; ++i) {
                const uint8_t* a = (const uint8_t*) act[i];
                int s = 0;
                for (int j = 0; j < kBlock; ++j) s += wv[j] * act_q(a, b, j);
                acc[i][l] = std::fma((float) s, t[p[0]] * act_d(a, k, b), acc[i][l]);
            }
        }
        for (int i = 0; i < n; ++i) {
            const float* A = acc[i];
            y[i][r - r0] = ((A[0] + A[4]) + (A[2] + A[6])) + ((A[1] + A[5]) + (A[3] + A[7]));
        }
    }
}

// ---- which tiers this CPU runs
struct Cpu {
    bool avx2 = false, vnni256 = false, avx512 = false, vnni512 = false, vbmi = false, bmi2 = false;
};
Cpu detect() {
    Cpu c;
#if defined(MX_X86)
    unsigned r[4] = {0, 0, 0, 0};
    auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
        int x[4];
        __cpuidex(x, (int) leaf, (int) sub);
        for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
        __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
    };
    cpuid(0, 0);
    const unsigned max_leaf = r[0];
    if (max_leaf < 7) return c;
    cpuid(1, 0);
    const bool fma = (r[2] >> 12) & 1u, osxsave = (r[2] >> 27) & 1u, avx = (r[2] >> 28) & 1u;
    if (!fma || !osxsave || !avx) return c;
#if defined(_MSC_VER)
    const unsigned long long xcr0 = _xgetbv(0);
#else
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
    if ((xcr0 & 0x6) != 0x6) return c;   // the OS saves the YMM state
    cpuid(7, 0);
    const unsigned b7 = r[1], c7 = r[2];
    c.avx2 = (b7 >> 5) & 1u;
    const bool zmm_os = (xcr0 & 0xE6) == 0xE6;
    c.avx512 = c.avx2 && zmm_os && ((b7 >> 16) & 1u) && ((b7 >> 30) & 1u);   // F, BW
    c.vnni512 = c.avx512 && ((c7 >> 11) & 1u);
    c.vbmi = c.avx512 && ((c7 >> 1) & 1u);   // AVX-512 VBMI
    c.bmi2 = (b7 >> 8) & 1u;
    cpuid(7, 1);
    c.vnni256 = c.avx2 && ((r[0] >> 4) & 1u);   // AVX-VNNI
#endif
    return c;
}
const Cpu& cpu() {
    static const Cpu c = detect();
    return c;
}

// The tier for this CPU, measured: every vector tier it runs times the same small warm job (64 rows of a 5120-wide
// matrix, in L2, against 1, 2 and 4 activations - the compute a row takes, which is where the tiers differ; streaming
// from RAM they all reach the memory's speed) in interleaved rounds, the best round of each counting.  The choice never
// changes an answer (the tiers give the same bits), only the speed.  DS4_MX_TIER=<name> picks one by hand.
std::string g_tier_report;
int measure_tier(const std::vector<int>& cand);
int pick_tier() {
    if (const char* f = std::getenv("DS4_MX_TIER"))
        for (int t = 0; t < kMxTiers; ++t)
            if (std::string(f) == mx_tier_name(t) && mx_tier_ok(t)) {
                g_tier_report = std::string("DS4_MX_TIER=") + f;
                return t;
            }
    std::vector<int> cand;
    for (int t = 1; t < kMxTiers; ++t)
        if (mx_tier_ok(t)) cand.push_back(t);
    if (cand.empty()) return kMxScalar;
    if (cand.size() == 1) return cand[0];
    return measure_tier(cand);
}
int measure_tier(const std::vector<int>& cand) {
    constexpr int K = 5120, R = 64, NT = 4;
    const size_t rb = (size_t) K / kBlock * kBytes;
    std::vector<uint8_t> w(rb * R), act(mx_act_bytes(K) * NT);
    uint32_t s = 0x9E3779B9u;
    for (size_t i = 0; i < w.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        w[i] = (uint8_t) (s >> 24);
    }
    for (size_t b = 0; b < w.size() / kBytes; ++b) w[b * kBytes] = (uint8_t) (118 + b % 6);
    std::vector<float> x(K);
    for (int i = 0; i < NT; ++i) {
        for (int j = 0; j < K; ++j) {
            s = s * 1664525u + 1013904223u;
            x[(size_t) j] = (float) ((int) (s >> 16) - 32768) / 32768.0f;
        }
        quant_scalar(x.data(), 0, K / kBlock, act.data() + (size_t) i * mx_act_bytes(K), K);
    }
    const void* ap[NT];
    for (int i = 0; i < NT; ++i) ap[i] = act.data() + (size_t) i * mx_act_bytes(K);
    float y[NT][R];
    float* yp[NT] = {y[0], y[1], y[2], y[3]};
    std::vector<double> best(cand.size(), 1e30);
    for (int round = 0; round < 6; ++round)
        for (size_t c = 0; c < cand.size(); ++c) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int rep = 0; rep < 8; ++rep)
                for (int n : {1, 2, 4}) mx_rows_tier(cand[c], w.data(), rb, K, 0, R, ap, n, yp, 0);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            if (round > 0) best[c] = std::min(best[c], us);   // (round 0: the clock and the caches warm)
        }
    size_t bi = 0;
    for (size_t c = 1; c < cand.size(); ++c)
        if (best[c] < best[bi]) bi = c;
    char buf[64];
    for (size_t c = 0; c < cand.size(); ++c) {
        std::snprintf(buf, sizeof buf, "%s%s %.0f us", c ? ", " : "", mx_tier_name(cand[c]), best[c]);
        g_tier_report += buf;
    }
    return cand[bi];
}
int prefetch_bytes() {
    const char* f = std::getenv("DS4_MX_PF");
    return f != nullptr ? std::max(0, std::atoi(f)) : 0;
}

}  // namespace

void q8_round(const float* x, int k, float* out) {
    for (int b = 0; b < k / kBlock; ++b) {
        const float* xb = x + b * kBlock;
        float am = 0.0f;
        for (int j = 0; j < kBlock; ++j) am = std::max(am, std::fabs(xb[j]));
        const float sc = am / 127.0f;
        for (int j = 0; j < kBlock; ++j)
            out[b * kBlock + j] = am > 0.0f ? std::nearbyint(xb[j] / sc) * sc : 0.0f;
    }
}

void mx_cpu_rows(const uint8_t* w, size_t rb, int k, int r0, int r1, const float* const* x, int n, float* const* y) {
#if defined(MX_X86)
    static const bool avx2 = strata::dsv41k::cpu::exl3_cpu_tier() >= 1;
    if (avx2) return rows_avx2(w, rb, k, r0, r1, x, n, y);
#endif
    rows_scalar(w, rb, k, r0, r1, x, n, y);
}

// ---- the int8 path
bool mx_tier_ok(int t) {
    switch (t) {
        case kMxScalar: return true;
        case kMxAvx2: return cpu().avx2;
#if defined(MX_AVXVNNI)
        case kMxAvxVnni: return cpu().vnni256;
#endif
        case kMxAvx512: return cpu().avx512;
        case kMxAvx512Vnni: return cpu().vnni512;
        default: return false;
    }
}
const char* mx_tier_name(int t) {
    switch (t) {
        case kMxScalar: return "scalar";
        case kMxAvx2: return "avx2";
        case kMxAvxVnni: return "avxvnni";
        case kMxAvx512: return "avx512";
        case kMxAvx512Vnni: return "avx512vnni";
        default: return "?";
    }
}
int mx_tier() {
    static const int t = pick_tier();
    return t;
}
const char* mx_tier_report() {
    (void) mx_tier();
    return g_tier_report.c_str();
}

void mx_quant_blocks(const float* x, int b0, int b1, void* act, int k) {
#if defined(MX_X86)
    if (cpu().avx2 && k % 256 == 0) return mx_avx2::quant(x, b0, b1, (uint8_t*) act, k);
#endif
    quant_scalar(x, b0, b1, (uint8_t*) act, k);
}
void mx_quant_blocks_scalar(const float* x, int b0, int b1, void* act, int k) {
    quant_scalar(x, b0, b1, (uint8_t*) act, k);
}
int mx_act_value(const void* act, int b, int j) { return act_q((const uint8_t*) act, b, j); }
float mx_act_scale(const void* act, int k, int b) { return act_d((const uint8_t*) act, k, b); }

void mx_rows_tier(int t, const uint8_t* w, size_t rb, int k, int r0, int r1, const void* const* act, int n,
                  float* const* y, int pf) {
    if (k % 256 != 0 || !mx_tier_ok(t)) t = kMxScalar;   // (the vector tiers take a group of 8 blocks a step)
    const float* tb = e8().v;
#if defined(MX_X86)
    switch (t) {
        case kMxAvx2: return mx_avx2::rows(w, rb, k, r0, r1, act, n, y, tb, pf);
#if defined(MX_AVXVNNI)
        case kMxAvxVnni: return mx_avxvnni::rows(w, rb, k, r0, r1, act, n, y, tb, pf);
#endif
        case kMxAvx512: return mx_avx512::rows(w, rb, k, r0, r1, act, n, y, tb, pf);
        case kMxAvx512Vnni: return mx_avx512vnni::rows(w, rb, k, r0, r1, act, n, y, tb, pf);
        default: break;
    }
#endif
    (void) tb, (void) pf;
    for (int i0 = 0; i0 < n; i0 += kMxCpuMaxN)
        rows_q_scalar(w, rb, k, r0, r1, act + i0, std::min(kMxCpuMaxN, n - i0), y + i0);
}

void mx_rows(const uint8_t* w, size_t rb, int k, int r0, int r1, const void* const* act, int n, float* const* y) {
    static const int pf = prefetch_bytes();
    mx_rows_tier(mx_tier(), w, rb, k, r0, r1, act, n, y, pf);
}

}  // namespace strata::ds4::gguf
