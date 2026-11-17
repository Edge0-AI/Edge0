// tools/trellis/tqenc.cpp - see tqenc.hpp.
#include "tqenc.hpp"

#include "ggml-common-tq.h"
#include "tq_computed.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define TQENC_AVX2 1
#endif

namespace tqenc {
namespace {

constexpr float kInf = 1e30f;

// ---------------------------------------------------------------------------------------------------------------
// fp16

}  // namespace

float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, bits;
    if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (man << 13);
    } else if (exp != 0) {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    } else if (man == 0) {
        bits = sign;
    } else {
        const float m = (float) man * (1.0f / 16777216.0f);
        return sign ? -m : m;
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) return (uint16_t) (sign | 0x7c00u | (ax > 0x7f800000u ? 0x200u : 0u));   // inf / nan
    if (ax >= 0x477ff000u) return (uint16_t) (sign | 0x7c00u);   // >= 65520 rounds to inf
    if (ax < 0x33000001u) return (uint16_t) sign;                // <= 2^-25: rounds to zero (ties to even)
    int e = (int) (ax >> 23) - 127;
    uint32_t man = (ax & 0x7fffffu) | 0x800000u;
    if (e < -14) {   // subnormal half
        const int shift = -14 - e + 13;   // total right shift of the 24-bit mantissa
        const uint32_t half = 1u << (shift - 1), rest = man & ((1u << shift) - 1);
        uint32_t q = man >> shift;
        if (rest > half || (rest == half && (q & 1))) ++q;
        return (uint16_t) (sign | q);
    }
    uint32_t q = ((uint32_t) (e + 15) << 10) | ((man >> 13) & 0x3ffu);
    const uint32_t rest = man & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (q & 1))) ++q;   // a carry into the exponent is correct
    return (uint16_t) (sign | q);
}

int type_kbits(int type) {
    switch (type) {
        case TQ2_T: return 8;
        case TQK6: return 6;
        case TQK7: return 7;
        case TQK5X: case TQK5T: return 5;
        case TQK4X: case TQK4T: return 4;
        default: return 0;
    }
}

bool type_computed(int type) { return type == TQK5X || type == TQK4X; }

double codebook_rms(int type);

namespace {

// ---------------------------------------------------------------------------------------------------------------
// codebook: the decoder's int16 table (round half away from zero of fp16 * 8192, as tq_cpu.cpp fill_table), SoA floats

struct Codebook {
    int16_t tab[kStates * 4];
    alignas(64) float soa[4][kStates];
    double rms = 1.072;   // of the weights: the first scale guess is rms(block) / this
    explicit Codebook(bool computed) {
        if (computed) {   // tq_computed.hpp
            double ss = 0;
            for (uint32_t s = 0; s < (uint32_t) kStates; ++s) {
                strata::kernels::cpu::tq_detail::comp_state4(s, tab + 4 * s);
                for (int j = 0; j < 4; ++j) {
                    soa[j][s] = (float) tab[4 * s + j] * (1.0f / 8192.0f);
                    ss += (double) soa[j][s] * soa[j][s];
                }
            }
            rms = std::sqrt(ss / (4.0 * kStates));
            return;
        }
        double ss = 0;
        for (uint32_t s = 0; s < (uint32_t) kStates; ++s) {
            const uint32_t x = s * 0x9e3779b1u;
            const uint32_t pts[2] = {x >> 21, (x >> 10) & 2047u};
            for (int h = 0; h < 2; ++h)
                for (int j = 0; j < 2; ++j) {
                    const float v = half_to_float(tq_lut_f16[2 * pts[h] + j]) * 8192.0f;
                    const int16_t q = (int16_t) (v < 0.0f ? v - 0.5f : v + 0.5f);
                    tab[4 * s + 2 * h + j] = q;
                    soa[2 * h + j][s] = (float) q * (1.0f / 8192.0f);
                    ss += (double) soa[2 * h + j][s] * soa[2 * h + j][s];
                }
        }
        rms = std::sqrt(ss / (4.0 * kStates));
    }
};

const Codebook& cb(bool computed) {
    static const Codebook table(false), comp(true);
    return computed ? comp : table;
}

struct Geo {
    const Codebook* cbk = nullptr;
    int K = 0;
    bool t2 = false;   // TQ2_T (byte stream, the other shift direction)
    int G = 0;         // predecessor groups
};

Geo geo_for(int type) {
    Geo g;
    g.cbk = &cb(type_computed(type));
    g.K = type_kbits(type);
    g.t2 = type == TQ2_T;
    g.G = g.t2 ? 128 : (1 << (15 - g.K));
    return g;
}

// group of a state as a successor / as a predecessor
inline int gsucc(const Geo& g, uint32_t s) { return g.t2 ? (int) (s >> 8) : (int) (s & (uint32_t) (g.G - 1)); }
inline int gpred(const Geo& g, uint32_t s) { return g.t2 ? (int) (s & 0x7f) : (int) (s >> g.K); }

// weight index (0..127) of component j of processing position p
inline int widx(const Geo& g, int p, int j) { return g.t2 ? 4 * p + j : 4 * (31 - p) + j; }

struct Scratch {
    alignas(64) float D[2][kStates];        // the two live columns of the DP
    alignas(64) uint16_t bp[32][2048];      // per step and predecessor group: the best predecessor state
    alignas(64) float m[2048];
    alignas(64) int32_t am[2048];
    alignas(64) float zero[kStates];
};

Scratch& scratch() {
    thread_local std::unique_ptr<Scratch> s;
    if (!s) {
        s.reset(new Scratch);
        std::memset(s->zero, 0, sizeof s->zero);
        std::memset(s->m, 0, sizeof s->m);
    }
    return *s;
}

inline float min_range(const float* p, int n) {
#if TQENC_AVX2
    if (n >= 8) {
        __m256 v = _mm256_loadu_ps(p);
        for (int i = 8; i + 8 <= n; i += 8) v = _mm256_min_ps(v, _mm256_loadu_ps(p + i));
        __m128 a = _mm_min_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        a = _mm_min_ps(a, _mm_movehl_ps(a, a));
        a = _mm_min_ss(a, _mm_shuffle_ps(a, a, 1));
        return _mm_cvtss_f32(a);
    }
#endif
    float m = p[0];
    for (int i = 1; i < n; ++i) m = std::min(m, p[i]);
    return m;
}

// out[i] = add[i] + sum_j h_j (t_j - cb_j[s0 + i])^2   for i in [0, n)
inline void step_cost(const Codebook& C, float* out, const float* add, int s0, int n, const float* t, const float* h) {
    const float *c0 = C.soa[0] + s0, *c1 = C.soa[1] + s0, *c2 = C.soa[2] + s0, *c3 = C.soa[3] + s0;
    int i = 0;
#if TQENC_AVX2
    const __m256 t0 = _mm256_set1_ps(t[0]), t1 = _mm256_set1_ps(t[1]), t2 = _mm256_set1_ps(t[2]), t3 = _mm256_set1_ps(t[3]);
    const __m256 h0 = _mm256_set1_ps(h[0]), h1 = _mm256_set1_ps(h[1]), h2 = _mm256_set1_ps(h[2]), h3 = _mm256_set1_ps(h[3]);
    for (; i + 8 <= n; i += 8) {
        const __m256 e0 = _mm256_sub_ps(t0, _mm256_loadu_ps(c0 + i)), e1 = _mm256_sub_ps(t1, _mm256_loadu_ps(c1 + i));
        const __m256 e2 = _mm256_sub_ps(t2, _mm256_loadu_ps(c2 + i)), e3 = _mm256_sub_ps(t3, _mm256_loadu_ps(c3 + i));
        __m256 a = _mm256_fmadd_ps(_mm256_mul_ps(h0, e0), e0, _mm256_loadu_ps(add + i));
        a = _mm256_fmadd_ps(_mm256_mul_ps(h1, e1), e1, a);
        a = _mm256_fmadd_ps(_mm256_mul_ps(h2, e2), e2, a);
        a = _mm256_fmadd_ps(_mm256_mul_ps(h3, e3), e3, a);
        _mm256_storeu_ps(out + i, a);
    }
#endif
    for (; i < n; ++i) {
        const float e0 = t[0] - c0[i], e1 = t[1] - c1[i], e2 = t[2] - c2[i], e3 = t[3] - c3[i];
        out[i] = add[i] + h[0] * e0 * e0 + h[1] * e1 * e1 + h[2] * e2 * e2 + h[3] * e3 * e3;
    }
}

// the broadcast variant: out[i] = a + cost(s0 + i)
inline void step_cost_bc(const Codebook& C, float* out, float a, int s0, int n, const float* t, const float* h) {
    alignas(32) float tmp[256];
    for (int i = 0; i < n; ++i) tmp[i] = a;
    step_cost(C, out, tmp, s0, n, t, h);
}

// min and first index of the minimum over p[0..n)
inline float min_arg_range(const float* p, int n, int* arg) {
    const float m = min_range(p, n);
    int i = 0;
#if TQENC_AVX2
    const __m256 vm = _mm256_set1_ps(m);
    for (; i + 8 <= n; i += 8) {
        const int mask = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(p + i), vm, _CMP_EQ_OQ));
        if (mask) { *arg = i + __builtin_ctz((unsigned) mask); return m; }
    }
#endif
    for (; i < n; ++i)
        if (p[i] == m) { *arg = i; return m; }
    *arg = 0;
    return m;
}

// T[p][4]: targets of processing position p (weights / d), H[p][4]: importance.  rot: the sequence is processed from
// position rot (cyclically); fixed_c >= 0: tail-biting with the successor group of the first state and the predecessor
// group of the last state both == fixed_c.  Writes the states by position into st[32]; returns the path cost.
float viterbi(const Geo& g, const float (*T)[4], const float (*H)[4], int rot, int fixed_c, uint16_t* st) {
    Scratch& sc = scratch();
    const int K = g.K, G = g.G;
    float* cur = sc.D[0];
    // q = 0
    {
        const int p = rot & 31;
        step_cost(*g.cbk, cur, sc.zero, 0, kStates, T[p], H[p]);
        if (fixed_c >= 0) {
            if (g.t2) {   // gsucc(s) = s >> 8 == c: the 256 states [c<<8, c<<8 + 256) allowed
                for (int s = 0; s < kStates; ++s)
                    if ((s >> 8) != fixed_c) cur[s] = kInf;
            } else {      // s & (G-1) == c
                for (int s = 0; s < kStates; ++s)
                    if ((s & (G - 1)) != fixed_c) cur[s] = kInf;
            }
        }
    }
    for (int q = 1; q < 32; ++q) {
        const float* prev = cur;
        cur = sc.D[q & 1];
        const int p = (rot + q) & 31;
        float* m = sc.m;
        uint16_t* bp = sc.bp[q];
        if (g.t2) {
            // m[a] = min over h of prev[a + 128 h], argmin h
            for (int a = 0; a < 128; ++a) { m[a] = prev[a]; sc.am[a] = 0; }
            for (int h = 1; h < 256; ++h) {
                const float* row = prev + 128 * h;
                int a = 0;
#if TQENC_AVX2
                const __m256i vh = _mm256_set1_epi32(h);
                for (; a + 8 <= 128; a += 8) {
                    const __m256 r = _mm256_loadu_ps(row + a), mm = _mm256_loadu_ps(m + a);
                    const __m256 lt = _mm256_cmp_ps(r, mm, _CMP_LT_OQ);   // strictly smaller: the first minimum stays
                    _mm256_storeu_ps(m + a, _mm256_blendv_ps(mm, r, lt));
                    const __m256i old = _mm256_loadu_si256((const __m256i*) (sc.am + a));
                    _mm256_storeu_si256((__m256i*) (sc.am + a), _mm256_castps_si256(_mm256_blendv_ps(
                        _mm256_castsi256_ps(old), _mm256_castsi256_ps(vh), lt)));
                }
#endif
                for (; a < 128; ++a)
                    if (row[a] < m[a]) { m[a] = row[a]; sc.am[a] = h; }
            }
            for (int a = 0; a < 128; ++a) bp[a] = (uint16_t) (a + 128 * sc.am[a]);
            for (int b = 0; b < 128; ++b) step_cost_bc(*g.cbk, cur + 256 * b, m[b], 256 * b, 256, T[p], H[p]);
        } else {
            const int span = 1 << K;
            for (int a = 0; a < G; ++a) {
                int arg;
                m[a] = min_arg_range(prev + (size_t) a * span, span, &arg);
                bp[a] = (uint16_t) ((a << K) + arg);
            }
            for (int base = 0; base < kStates; base += G) step_cost(*g.cbk, cur + base, m, base, G, T[p], H[p]);
        }
    }
    // final state
    const float* last = cur;
    int best = -1;
    float bc = 3e38f;
    if (fixed_c < 0) {
        for (int s = 0; s < kStates; ++s)
            if (last[s] < bc) { bc = last[s]; best = s; }
    } else if (g.t2) {
        for (int h = 0; h < 256; ++h) {
            const int s = fixed_c + 128 * h;
            if (last[s] < bc) { bc = last[s]; best = s; }
        }
    } else {
        for (int i = 0; i < (1 << K); ++i) {
            const int s = (fixed_c << K) + i;
            if (last[s] < bc) { bc = last[s]; best = s; }
        }
    }
    if (best < 0 || bc >= kInf * 0.5f) return kInf;
    uint32_t s = (uint32_t) best;
    st[(rot + 31) & 31] = (uint16_t) s;
    for (int q = 31; q >= 1; --q) {
        s = sc.bp[q][gsucc(g, s)];
        st[(rot + q - 1) & 31] = (uint16_t) s;
    }
    return bc;
}

// junction group of a free pass: gpred of the state at orig position 31 (== gsucc of the state at position 0)
inline int junction_group(const Geo& g, const uint16_t* st) { return gpred(g, st[31]); }

struct Search {
    const Geo* g;
    const Params* p;
    float T[32][4];
    float H[32][4];
    uint16_t st[32];
    int passes = 0;

    // one tail-biting search at scale d (weights already divided: set_targets); returns the path cost in scaled units
    float run() {
        static const int rots[8] = {16, 8, 24, 12, 20, 4, 28, 0};
        float best = kInf;
        uint16_t tmp[32], free_st[32];
        if (p->exact) {
            for (int c = 0; c < g->G; ++c) {
                const float v = viterbi(*g, T, H, 0, c, tmp);
                ++passes;
                if (v < best) { best = v; std::memcpy(st, tmp, sizeof tmp); }
            }
            return best;
        }
        int seen[8], ns = 0;
        const int nc = std::max(1, std::min(p->ncand, 8));
        for (int r = 0; r < nc; ++r) {
            viterbi(*g, T, H, rots[r], -1, free_st);
            ++passes;
            const int c = junction_group(*g, free_st);
            bool dup = false;
            for (int i = 0; i < ns; ++i) dup |= seen[i] == c;
            if (dup) continue;
            seen[ns++] = c;
            const float v = viterbi(*g, T, H, 0, c, tmp);
            ++passes;
            if (v < best) { best = v; std::memcpy(st, tmp, sizeof tmp); }
        }
        return best;
    }
};

void pack_stream(const Geo& g, const uint16_t* st, uint8_t* qs) {
    const int K = g.K;
    std::memset(qs, 0, (size_t) 4 * K);
    if (g.t2) {
        for (int t = 0; t < 32; ++t) qs[t] = (uint8_t) (st[t] & 0xff);
        return;
    }
    for (int u = 0; u < 32; ++u) {
        const uint32_t x = st[u] & ((1u << K) - 1u);
        for (int b = 0; b < K; ++b)
            if ((x >> b) & 1u) qs[(u * K + b) >> 3] |= (uint8_t) (1u << ((u * K + b) & 7));
    }
}

}  // namespace

float debug_viterbi(int type, const float (*T)[4], const float (*H)[4], int rot, int fixed_c, uint16_t* st) {
    return viterbi(geo_for(type), T, H, rot, fixed_c, st);
}
int debug_groups(int type) { return geo_for(type).G; }
int debug_widx(int type, int p, int j) { return widx(geo_for(type), p, j); }
void debug_pack(int type, const uint16_t* st, uint8_t* qs) { pack_stream(geo_for(type), st, qs); }

const float* codebook_state(int type, uint32_t state) {
    static thread_local float v[4];
    for (int j = 0; j < 4; ++j) v[j] = cb(type_computed(type)).soa[j][state & 0x7fff];
    return v;
}

const int16_t* codebook_table(int type) { return cb(type_computed(type)).tab; }
double codebook_rms(int type) { return cb(type_computed(type)).rms; }

const float* point_table() {
    static float pts[4096];
    static std::once_flag once;
    std::call_once(once, [] {
        for (int i = 0; i < 4096; ++i) {
            const float x = half_to_float(tq_lut_f16[i]) * 8192.0f;
            const int16_t q = (int16_t) (x < 0.0f ? x - 0.5f : x + 0.5f);
            pts[i] = (float) q * (1.0f / 8192.0f);
        }
    });
    return pts;
}

uint32_t block_state(int type, const uint8_t* qs, int t) {
    const int K = type_kbits(type);
    if (type == TQ2_T) return (((uint32_t) qs[(t + 31) & 31] << 8) | qs[t]) & 0x7fffu;
    const int total = 32 * K;
    uint32_t s = 0;
    for (int i = 0; i < 15; ++i) {
        const int b = ((31 - t) * K + i) % total;
        s |= (uint32_t) ((qs[b >> 3] >> (b & 7)) & 1u) << i;
    }
    return s;
}

void decode_blocks(int type, const uint8_t* src, int64_t nblocks, float* out) {
    const int K = type_kbits(type), bb = 2 + 4 * K;
    for (int64_t b = 0; b < nblocks; ++b) {
        const uint8_t* blk = src + (size_t) b * bb;
        const uint16_t dh = (uint16_t) (blk[0] | (blk[1] << 8));
        const float d = half_to_float(dh);
        for (int t = 0; t < 32; ++t) {
            const uint32_t s = block_state(type, blk + 2, t);
            for (int j = 0; j < 4; ++j) out[b * 128 + 4 * t + j] = d * cb(type_computed(type)).soa[j][s];
        }
    }
}

BlockResult encode_block(int type, const float* w, const float* h, uint8_t* dst, const Params& prm) {
    BlockResult r;
    const Geo g = geo_for(type);
    const int K = g.K, bb = 2 + 4 * K;
    std::memset(dst, 0, (size_t) bb);
    double sumw2 = 0, sumhw2 = 0, amax = 0;
    for (int i = 0; i < 128; ++i) {
        if (!std::isfinite(w[i]) || (h && !std::isfinite(h[i]))) { r.status = 2; r.err = NAN; return r; }
        const double hi = h ? h[i] : 1.0;
        sumw2 += (double) w[i] * w[i];
        sumhw2 += hi * (double) w[i] * w[i];
        amax = std::max(amax, (double) std::fabs(w[i]));
    }
    if (amax == 0.0) { r.status = 1; return r; }
    const double rms = std::sqrt(sumw2 / 128.0);
    // initial scale: the codebook has unit-ish rms
    const double kCbRms = g.cbk->rms;
    static const double mult[7] = {1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3};

    Search se;
    se.g = &g;
    se.p = &prm;
    for (int p = 0; p < 32; ++p)
        for (int j = 0; j < 4; ++j) se.H[p][j] = h ? h[widx(g, p, j)] : 1.0f;

    double best_err = 1e300;
    uint16_t best_st[32] = {};
    uint16_t best_d = 0;
    bool have = false;

    auto fit = [&](const uint16_t* st, uint16_t dh, double& err) {
        // err of (st, dh): sum h (w - d cv)^2
        const float d = half_to_float(dh);
        double e = 0;
        for (int p = 0; p < 32; ++p)
            for (int j = 0; j < 4; ++j) {
                const int wi = widx(g, p, j);
                const float rec = d * g.cbk->soa[j][st[p]];
                const double df = (double) w[wi] - rec;
                e += (h ? h[wi] : 1.0) * df * df;
            }
        err = e;
    };

    const int tries = prm.fixed_scale ? 1 : std::max(1, std::min(prm.scale_tries, 7));
    for (int tr = 0; tr < tries; ++tr) {
        double dd = rms / kCbRms * mult[tr];
        uint16_t dh = float_to_half((float) std::min(dd, 65504.0));
        if (dh == 0) dh = 1;   // the smallest subnormal: weights below ~3e-8 are noise at this scale
        if (prm.fixed_scale) dh = prm.fixed_scale;
        for (int it = 0; it <= (prm.fixed_scale ? 0 : prm.iters); ++it) {
            const float d = half_to_float(dh);
            for (int p = 0; p < 32; ++p)
                for (int j = 0; j < 4; ++j) se.T[p][j] = w[widx(g, p, j)] / d;
            if (se.run() >= kInf * 0.5f) break;
            double e;
            fit(se.st, dh, e);
            if (e < best_err) {
                best_err = e;
                best_d = dh;
                std::memcpy(best_st, se.st, sizeof best_st);
                have = true;
            }
            if (e == 0.0 || prm.fixed_scale) break;
            // least-squares scale for this path
            double num = 0, den = 0;
            for (int p = 0; p < 32; ++p)
                for (int j = 0; j < 4; ++j) {
                    const int wi = widx(g, p, j);
                    const double hi = h ? h[wi] : 1.0, c = g.cbk->soa[j][se.st[p]];
                    num += hi * w[wi] * c;
                    den += hi * c * c;
                }
            if (den <= 0 || num <= 0) break;
            uint16_t nh = float_to_half((float) std::min(num / den, 65504.0));
            if (nh == 0 || nh == dh) break;
            double en;
            fit(se.st, nh, en);
            if (en < best_err) {
                best_err = en;
                best_d = nh;
                std::memcpy(best_st, se.st, sizeof best_st);
            }
            if (en >= e) break;   // the scale step did not help this path: stop refining this start
            dh = nh;
        }
        r.searches = se.passes;
    }
    if (!have || best_err >= sumhw2) {   // nothing beats a zero block (weights below the fp16 scale resolution)
        std::memset(dst, 0, (size_t) bb);
        r.status = 1;
        r.err = sumhw2;
        return r;
    }
    dst[0] = (uint8_t) (best_d & 0xff);
    dst[1] = (uint8_t) (best_d >> 8);
    pack_stream(g, best_st, dst + 2);
    r.err = best_err;
    r.scale = half_to_float(best_d);
    r.searches = se.passes;
    return r;
}

double encode_blocks(int type, const float* w, const float* h, int64_t nblocks, uint8_t* dst, const Params& p,
                     int nthreads, double* err_out, int64_t* nonfinite, int64_t* zero_blocks) {
    const int bb = block_bytes(type);
    if (nthreads <= 0) nthreads = (int) std::max(1u, std::thread::hardware_concurrency());
    nthreads = (int) std::min<int64_t>(nthreads, std::max<int64_t>(1, nblocks));
    std::atomic<int64_t> next{0}, nf{0}, zb{0};
    std::vector<double> sums((size_t) nthreads, 0.0);
    auto work = [&](int ti) {
        double s = 0;
        for (;;) {
            const int64_t b0 = next.fetch_add(4);
            if (b0 >= nblocks) break;
            const int64_t b1 = std::min(nblocks, b0 + 4);
            for (int64_t b = b0; b < b1; ++b) {
                const BlockResult r = encode_block(type, w + b * 128, h ? h + b * 128 : nullptr, dst + (size_t) b * bb, p);
                if (r.status == 2) { nf.fetch_add(1); }
                else s += r.err;
                if (r.status == 1) zb.fetch_add(1);
                if (err_out) err_out[b] = r.err;
            }
        }
        sums[ti] = s;
    };
    std::vector<std::thread> th;
    for (int i = 1; i < nthreads; ++i) th.emplace_back(work, i);
    work(0);
    for (auto& t : th) t.join();
    if (nonfinite) *nonfinite = nf.load();
    if (zero_blocks) *zero_blocks = zb.load();
    double tot = 0;
    for (double s : sums) tot += s;
    return tot;
}

}  // namespace tqenc

// ---------------------------------------------------------------------------------------------------------------
// C API for the Python wrapper (tools/tqenc.py, ctypes)
extern "C" {
#if defined(_WIN32)
#define TQE_EXPORT __declspec(dllexport)
#else
#define TQE_EXPORT __attribute__((visibility("default")))
#endif
TQE_EXPORT int tqe_block_bytes(int type) { return tqenc::block_bytes(type); }
TQE_EXPORT int tqe_kbits(int type) { return tqenc::type_kbits(type); }
TQE_EXPORT void tqe_decode(int type, const uint8_t* src, int64_t nblocks, float* out) {
    tqenc::decode_blocks(type, src, nblocks, out);
}
// returns the summed block error; stats[0] = nonfinite blocks, stats[1] = zero blocks
TQE_EXPORT double tqe_encode(int type, const float* w, const float* h, int64_t nblocks, uint8_t* dst, int iters,
                             int ncand, int exact, int scale_tries, int nthreads, double* err_out, int64_t* stats) {
    tqenc::Params p;
    p.iters = iters;
    p.ncand = ncand;
    p.exact = exact;
    p.scale_tries = scale_tries;
    int64_t nf = 0, zb = 0;
    const double e = tqenc::encode_blocks(type, w, h, nblocks, dst, p, nthreads, err_out, &nf, &zb);
    if (stats) { stats[0] = nf; stats[1] = zb; }
    return e;
}
TQE_EXPORT const int16_t* tqe_table(int type) { return tqenc::codebook_table(type); }
}
