// tests/gdn/gdn_fixture.hpp - backend-independent fixtures and an FP64 oracle for the Flash-Next GDN prompt path.
//
// Plain C++17, no GPU headers: the CUDA harness (src/prefill/gdn_harness.cu) uses it today; the HIP and Metal/SYCL
// harnesses are meant to include the same file, so every backend is judged against the same inputs and the same
// reference.  Contract: docs/GDN_CONTRACT.md (shapes, layouts, what a candidate may change).
//
//   Fixture       raw inputs of one layer for T tokens: qkv [T][C], ab [T][2 HV], dt, ssm_a, conv_w [C][4], conv
//                 history [C][3], the recurrent state [S][HV][S] (row, head, column), z [T][HV][S], gamma [S]
//   RefResult     FP64 oracle for some value heads over the whole sequence: normalised output (what the out
//                 projection reads, before the FP16 rounding), the final state of those heads, the conv history
//   ErrStat       max_abs, normalized_max (max_abs / max|ref|), relative_L2 (gdn-research 06)
// The oracle starts from the RAW inputs (conv, SiLU, L2 norm, gates, recurrence, output norm), so a bug shared by a
// kernel and a test copy of it cannot hide.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace gdnfx {

constexpr int S = 128, HK = 16, HV = 48, VPK = HV / HK, C = 2 * HK * S + HV * S;   // C = 10240
constexpr float EPS = 1e-6f;                                                      // the engine's L2 / RMS epsilon

struct Rng {   // splitmix64: identical on every platform and compiler
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ULL + 0x1234567ULL) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    double uni() { return (double) (next() >> 11) * (1.0 / 9007199254740992.0); }   // [0, 1)
    double gauss() {
        double u1 = uni(), u2 = uni();
        if (u1 < 1e-300) u1 = 1e-300;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

struct FixtureSpec {
    int64_t T = 0;
    uint64_t seed = 1;
    // state0: "random" (as after an earlier prompt chunk), "zero" (a fresh conversation), "big", "small"
    const char* state = "random";
    float v_scale = 1.0f;       // values x this (50: range stress)
    float gate_scale = 1.0f;    // ssm_a x this (0.005: near-zero decay, 3: strong decay)
    bool coherent_keys = false; // every token's k is a head constant plus a small variation (T = (I+A)^-1 large)
    bool alt_beta = false;      // beta alternates 0.02 / 0.98 per token (endpoint regime)
    bool tiny_k = false;        // every 7th token's k/q channels are ~1e-4 before the L2 norm (eps matters)
};

struct Fixture {
    FixtureSpec spec;
    int64_t T = 0;
    std::vector<float> qkv, ab, dt, ssm_a, conv_w, hist, state0, z, gamma;
};

inline Fixture make_fixture(const FixtureSpec& sp) {
    Fixture f;
    f.spec = sp;
    f.T = sp.T;
    const int64_t T = sp.T;
    Rng r(sp.seed);
    f.qkv.resize((size_t) T * C);
    f.ab.resize((size_t) T * 2 * HV);
    f.dt.resize(HV);
    f.ssm_a.resize(HV);
    f.conv_w.resize((size_t) C * 4);
    f.hist.resize((size_t) C * 3);
    f.state0.resize((size_t) S * HV * S);
    f.z.resize((size_t) T * HV * S);
    f.gamma.resize(S);
    for (auto& x : f.qkv) x = (float) r.gauss();
    if (sp.coherent_keys) {
        std::vector<float> base((size_t) HK * S);
        for (auto& x : base) x = (float) r.gauss();
        for (int64_t t = 0; t < T; ++t)
            for (int i = 0; i < HK * S; ++i) {
                float& x = f.qkv[(size_t) t * C + HK * S + i];   // k channels
                x = base[i] + 0.05f * x;
            }
    }
    if (sp.tiny_k)
        for (int64_t t = 0; t < T; t += 7)
            for (int i = 0; i < 2 * HK * S; ++i) f.qkv[(size_t) t * C + i] *= 1e-4f;
    if (sp.v_scale != 1.0f)
        for (int64_t t = 0; t < T; ++t)
            for (int i = 0; i < HV * S; ++i) f.qkv[(size_t) t * C + 2 * HK * S + i] *= sp.v_scale;
    for (auto& x : f.ab) x = (float) r.gauss();
    if (sp.alt_beta)
        for (int64_t t = 0; t < T; ++t)
            for (int h = 0; h < HV; ++h) f.ab[(size_t) t * 2 * HV + HV + h] = (t & 1) ? 3.9f : -3.9f;
    for (int h = 0; h < HV; ++h) {
        f.dt[h] = (float) (0.5 * r.gauss() - 0.5);
        const double u = r.uni();
        f.ssm_a[h] = (float) (-(0.05 + 1.5 * u * u) * sp.gate_scale);
    }
    for (auto& x : f.conv_w) x = (float) (0.35 * r.gauss());
    for (auto& x : f.hist) x = (float) r.gauss();
    for (auto& x : f.z) x = (float) r.gauss();
    for (auto& x : f.gamma) x = (float) (1.0 + 0.1 * r.gauss());
    float ss = 0.05f;
    if (!std::strcmp(sp.state, "zero")) ss = 0.0f;
    else if (!std::strcmp(sp.state, "big")) ss = 2.5f;
    else if (!std::strcmp(sp.state, "small")) ss = 5e-4f;
    for (auto& x : f.state0) x = ss == 0.0f ? 0.0f : (float) (ss * r.gauss());
    return f;
}

// ------------------------------------------------------------------------------------------------ FP64 oracle
inline double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }

struct RefResult {
    std::vector<int> heads;                 // value heads computed
    std::vector<std::vector<double>> y;     // per head: [T][S] normalised output (before FP16)
    std::vector<std::vector<double>> state; // per head: final [S][S] (row, column)
    std::vector<float> hist;                // conv history after the whole sequence (exact: copies of inputs)
};

// One value head over tokens [0, T).  state_io: that head's [S][S] (row, column), updated in place.
inline void ref_head(const Fixture& f, int vh, std::vector<double>& st, std::vector<double>& y) {
    const int64_t T = f.T;
    const int qh = vh % HK;
    y.assign((size_t) T * S, 0.0);
    std::vector<double> q(S), k(S), v(S), kv(S), delta(S), o(S);
    auto in = [&](int64_t t, int c) -> double {
        return t >= 0 ? (double) f.qkv[(size_t) t * C + c] : (double) f.hist[(size_t) c * 3 + (int) (t + 3)];
    };
    auto conv = [&](int64_t t, int c) -> double {
        const double s = (double) f.conv_w[(size_t) c * 4] * in(t - 3, c) + (double) f.conv_w[(size_t) c * 4 + 1] * in(t - 2, c) +
                         (double) f.conv_w[(size_t) c * 4 + 2] * in(t - 1, c) + (double) f.conv_w[(size_t) c * 4 + 3] * in(t, c);
        return s / (1.0 + std::exp(-s));
    };
    for (int64_t t = 0; t < T; ++t) {
        double qq = 0, kk = 0;
        for (int i = 0; i < S; ++i) {
            q[i] = conv(t, qh * S + i);
            k[i] = conv(t, HK * S + qh * S + i);
            v[i] = conv(t, 2 * HK * S + vh * S + i);
            qq += q[i] * q[i];
            kk += k[i] * k[i];
        }
        const double qs = 1.0 / std::sqrt(qq + (double) EPS), ks = 1.0 / std::sqrt(kk + (double) EPS);
        for (int i = 0; i < S; ++i) { q[i] *= qs; k[i] *= ks; }
        const double a = (double) f.ab[(size_t) t * 2 * HV + vh] + (double) f.dt[vh];
        const double sp = a > 20.0 ? a : std::log1p(std::exp(a));
        const double g = std::exp(sp * (double) f.ssm_a[vh]);
        const double beta = sigm((double) f.ab[(size_t) t * 2 * HV + HV + vh]);
        for (int c = 0; c < S; ++c) kv[c] = 0;
        for (int i = 0; i < S; ++i) {
            const double* row = &st[(size_t) i * S];
            for (int c = 0; c < S; ++c) kv[c] += row[c] * k[i];
        }
        for (int c = 0; c < S; ++c) delta[c] = (v[c] - g * kv[c]) * beta;
        for (int c = 0; c < S; ++c) o[c] = 0;
        for (int i = 0; i < S; ++i) {
            double* row = &st[(size_t) i * S];
            for (int c = 0; c < S; ++c) {
                row[c] = g * row[c] + k[i] * delta[c];
                o[c] += row[c] * q[i];
            }
        }
        double ss = 0;
        for (int c = 0; c < S; ++c) { o[c] /= std::sqrt((double) S); ss += o[c] * o[c]; }
        const double rn = 1.0 / std::sqrt(ss / S + (double) EPS);
        for (int c = 0; c < S; ++c)
            y[(size_t) t * S + c] = o[c] * rn * (double) f.gamma[c] * sigm((double) f.z[((size_t) t * HV + vh) * S + c]);
    }
}

// The oracle for `heads`, one thread each (state taken from the fixture's state0, layout [row][head][col]).
inline RefResult ref_run(const Fixture& f, const std::vector<int>& heads) {
    RefResult r;
    r.heads = heads;
    r.y.resize(heads.size());
    r.state.resize(heads.size());
    std::vector<std::thread> th;
    for (size_t n = 0; n < heads.size(); ++n)
        th.emplace_back([&, n] {
            const int vh = heads[n];
            std::vector<double> st((size_t) S * S);
            for (int i = 0; i < S; ++i)
                for (int c = 0; c < S; ++c) st[(size_t) i * S + c] = (double) f.state0[((size_t) i * HV + vh) * S + c];
            ref_head(f, vh, st, r.y[n]);
            r.state[n] = std::move(st);
        });
    for (auto& t : th) t.join();
    r.hist.resize((size_t) C * 3);
    for (int c = 0; c < C; ++c)
        for (int k = 0; k < 3; ++k) {
            const int64_t t = f.T - 3 + k;
            r.hist[(size_t) c * 3 + k] = t >= 0 ? f.qkv[(size_t) t * C + c] : f.hist[(size_t) c * 3 + (int) (t + 3)];
        }
    return r;
}

// ------------------------------------------------------------------------------------------------ metrics
struct ErrStat {
    double max_abs = 0, ref_max = 0, sq_diff = 0, sq_ref = 0;
    long nonfinite = 0, n = 0;
    void add(double got, double ref) {
        ++n;
        if (!std::isfinite(got)) { ++nonfinite; return; }
        const double d = std::fabs(got - ref);
        if (d > max_abs) max_abs = d;
        const double a = std::fabs(ref);
        if (a > ref_max) ref_max = a;
        sq_diff += d * d;
        sq_ref += ref * ref;
    }
    double normalized_max() const { return max_abs / std::max(ref_max, 1e-30); }
    double rel_l2() const { return std::sqrt(sq_diff) / std::max(std::sqrt(sq_ref), 1e-30); }
};

inline uint64_t fnv1a(const void* p, size_t bytes) {
    const unsigned char* b = (const unsigned char*) p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

inline float half_to_float(uint16_t h) {   // IEEE binary16 -> float, host side
    const uint32_t sign = (uint32_t) (h >> 15) << 31;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff, out;
    if (e == 0) {
        if (m == 0) out = sign;
        else {
            e = 127 - 15 + 1;
            while (!(m & 0x400)) { m <<= 1; --e; }
            out = sign | (e << 23) | ((m & 0x3ff) << 13);
        }
    } else if (e == 31) out = sign | 0x7f800000u | (m << 13);
    else out = sign | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

}  // namespace gdnfx
