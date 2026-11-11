// src/kernels/sampler_parity.cpp - P2.S2's test for the sampler chain.
//
// THE CHECK THAT MATTERS IS THAT THE ORDER IS OBSERVABLE.  A sampler in the wrong order still returns a valid
// token, so "it produced a token" proves nothing; and because temperature is MONOTONIC it does not change which
// tokens `top_k` keeps, so a top-k-only fixture cannot see the order either.  What it changes is `top_p`'s CUT:
// at T < 1 the distribution sharpens, the cumulative mass reaches p sooner, and fewer tokens survive.
//
// So this builds a fixture where that happens, computes the greedy pick under BOTH orders, and requires them to
// DIFFER - then requires the kernel to agree with the specified one.  Without the first half, the test would
// pass against either order.
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// The host reference for the specified order: top_k -> top_p -> temperature -> argmax.
// `temp_first` swaps the first and last stages, which is the intuitive-but-wrong order.
int reference_pick(const std::vector<float>& l, const strata::kernels::SamplerParams& p, bool temp_first) {
    const int nv = (int) l.size();
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    auto val = [&](int v) { return temp_first ? l[(size_t) v] * inv_t : l[(size_t) v]; };

    std::vector<int> ids;
    const int k = p.top_k > 0 ? p.top_k : nv;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1;
        float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            if (best < 0 || val(v) > bv) { best = v; bv = val(v); }
        }
        taken[(size_t) best] = 1;
        ids.push_back(best);
    }
    if (p.top_p < 1.0f) {
        float mx = val(ids[0]);
        for (int v : ids) mx = std::fmax(mx, val(v));
        double sum = 0;
        for (int v : ids) sum += std::exp((double) val(v) - (double) mx);
        double cum = 0;
        int cut = (int) ids.size();
        for (size_t i = 0; i < ids.size(); ++i) {
            cum += std::exp((double) val(ids[i]) - (double) mx) / sum;
            if (cum >= (double) p.top_p) { cut = (int) i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < (int) ids.size() ? p.min_keep : (int) ids.size();
        ids.resize((size_t) cut);
    }
    return ids[0];      // greedy: the largest SURVIVING logit, and `ids` is in descending order
}

// The number of survivors AFTER top_p, in the given order - the quantity the order actually changes.
int reference_cut(const std::vector<float>& l, const strata::kernels::SamplerParams& p, bool temp_first) {
    const int nv = (int) l.size();
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    auto val = [&](int v) { return temp_first ? l[(size_t) v] * inv_t : l[(size_t) v]; };
    const int k = p.top_k > 0 ? p.top_k : nv;
    std::vector<int> ids;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1; float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            if (best < 0 || val(v) > bv) { best = v; bv = val(v); }
        }
        taken[(size_t) best] = 1; ids.push_back(best);
    }
    if (p.top_p < 1.0f) {
        float mx = val(ids[0]);
        for (int v : ids) mx = std::fmax(mx, val(v));
        double sum = 0;
        for (int v : ids) sum += std::exp((double) val(v) - (double) mx);
        double cum = 0;
        int cut = (int) ids.size();
        for (size_t i = 0; i < ids.size(); ++i) {
            cum += std::exp((double) val(ids[i]) - (double) mx) / sum;
            if (cum >= (double) p.top_p) { cut = (int) i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < (int) ids.size() ? p.min_keep : (int) ids.size();
        return cut;
    }
    return (int) ids.size();
}

int run(const char* name, const std::vector<float>& logits, int n_tokens, const strata::kernels::SamplerParams& p,
        const std::vector<int>& want, const std::vector<int>& hist = {}, int hist_len = 0) {
    float* d_l = nullptr;
    int* d_o = nullptr;
    check(cudaMalloc(&d_l, logits.size() * sizeof(float)), "malloc logits");
    check(cudaMalloc(&d_o, (size_t) n_tokens * sizeof(int)), "malloc out");
    check(cudaMemcpy(d_l, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");
    // -1 in every output slot first: a row the kernel leaves unwritten can never match (a verify window reads
    // every row, so "no output" is a wrong answer, not a skipped one)
    check(cudaMemset(d_o, 0xFF, (size_t) n_tokens * sizeof(int)), "fill out");
    int* d_h = nullptr;
    if (hist_len > 0) {
        check(cudaMalloc(&d_h, hist.size() * sizeof(int)), "malloc hist");
        check(cudaMemcpy(d_h, hist.data(), hist.size() * sizeof(int), cudaMemcpyHostToDevice), "copy hist");
    }
    strata::kernels::sample_tokens(d_l, n_tokens, (int) (logits.size() / n_tokens), d_h, hist_len, p, d_o,
                                   nullptr);
    std::vector<int> got((size_t) n_tokens);
    check(cudaMemcpy(got.data(), d_o, got.size() * sizeof(int), cudaMemcpyDeviceToHost), "back");
    int bad = 0;
    for (int t = 0; t < n_tokens; ++t) if (got[(size_t) t] != want[(size_t) t]) ++bad;
    std::printf("  %-34s %s (%d of %d differ)", name, bad ? "*** WRONG ***" : "matches", bad, n_tokens);
    if (bad) std::printf("   first: want %d got %d", want[0], got[0]);
    std::printf("\n");
    cudaFree(d_l);
    cudaFree(d_o);
    if (d_h) cudaFree(d_h);
    return bad;
}

// The Philox draw, host side - a transcription of the kernel's `philox_uniform` so the SAMPLED pick (not
// just the greedy argmax) can be pinned against a reference.  `__umulhi(a, b)` is the high half of a 32x32
// multiply, spelled `(uint32_t)(((uint64_t) a * b) >> 32)` here.
struct PhiloxRound {
    uint32_t& c0; uint32_t& c1; uint32_t& c2; uint32_t& c3;
    void step(uint32_t k0, uint32_t k1) const {
        const uint32_t hi0 = (uint32_t) (((uint64_t) 0x9E3779B9u * c0) >> 32);
        const uint32_t hi1 = (uint32_t) (((uint64_t) 0xBB67AE85u * c2) >> 32);
        const uint32_t lo0 = 0x9E3779B9u * c0;
        const uint32_t lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ k0;
        const uint32_t n1 = lo1;
        const uint32_t n2 = hi0 ^ c3 ^ k1;
        const uint32_t n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
};

float host_philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    PhiloxRound r{c0, c1, c2, c3};
    for (int i = 0; i < 10; ++i) r.step((uint32_t) i, 0u);
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

// The Gumbel-max noise, host side - a transcription of the kernel's `gumbel_exp` (STRATA_SPEC_GUMBEL): Exp(1) from
// a splitmix64 hash of (seed, counter, token id).  The pick is argmax_i p_i / E_i, an exact sample of p.
double host_gumbel_exp(uint64_t seed, uint64_t counter, uint32_t token) {
    uint64_t z = seed ^ (counter * 0x9E3779B97F4A7C15ull) ^ ((uint64_t) token * 0xD1B54A32D192ED03ull);
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const double u = ((double) (z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    return -std::log(u);
}

// The full SAMPLED chain, host side - the kernel's `sampler_kernel` in serial form, in llama.cpp's order:
// penalties on the raw logits during the top_k selection (ties to the lowest index), top_p's cut in double over
// the top_k list, the min_p prefix cut on its survivors, the temperature, and one Philox draw at
// (seed, counter + row).  One penalties stage (issue #53: this reference used to repeat the kernel's second one).
int sampled_reference(const std::vector<float>& l, const std::vector<int>& hist,
                      const strata::kernels::SamplerParams& p, int row) {
    auto penal = [&](float logit, int count) {
        if (count <= 0) return logit;
        if (logit <= 0.0f) logit *= p.penalty_repeat; else logit /= p.penalty_repeat;
        logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
        return logit;
    };
    auto count = [&](int v) { int c = 0; for (int h : hist) if (h == v) ++c; return c; };
    const int nv = (int) l.size();
    const int KMAX = 64;                       // 1..64 as given; 0 (off) and wider keep the widest list, 64
    const int k = std::min(nv, (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX);
    std::vector<int> sel_ids;
    std::vector<float> sel_logit;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1; float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            const float s = penal(l[(size_t) v], count(v));
            if (best < 0 || s > bv) { best = v; bv = s; }
        }
        taken[(size_t) best] = 1;
        sel_ids.push_back(best); sel_logit.push_back(bv);
    }
    // llama.cpp's order (issue #53): top_p over the whole top_k list, then min_p on its survivors
    const int n_sel = (int) sel_ids.size();
    int n_keep = n_sel;
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < n_sel; ++i) sum += std::exp((double) sel_logit[(size_t) i] - (double) sel_logit[0]);
        double cum = 0.0;
        int cut = n_sel;
        for (int i = 0; i < n_sel; ++i) {
            cum += std::exp((double) sel_logit[(size_t) i] - (double) sel_logit[0]) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < n_sel ? p.min_keep : n_sel;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + std::log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[(size_t) i] < thresh) { n_keep = i; break; }
    }
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    auto scaled = [&](int i) { return sel_logit[(size_t) i] * inv_t; };   // one penalties stage, before (#53)
    float smx = scaled(0);
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, scaled(i));
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp((double) scaled(i) - (double) smx);
    int pick = sel_ids[(size_t) (n_keep - 1)];
    if (p.gumbel) {
        double best = -1.0;
        for (int i = 0; i < n_keep; ++i) {
            const double r = std::exp((double) scaled(i) - (double) smx) / sum /
                             host_gumbel_exp(p.seed, p.counter + (uint64_t) row, (uint32_t) sel_ids[(size_t) i]);
            if (r > best) { best = r; pick = sel_ids[(size_t) i]; }
        }
        return pick;
    }
    const float u = host_philox_uniform(p.seed, p.counter + (uint64_t) row);
    double cum = 0.0;
    for (int i = 0; i < n_keep; ++i) {
        cum += std::exp((double) scaled(i) - (double) smx) / sum;
        if ((double) u < cum) { pick = sel_ids[(size_t) i]; break; }
    }
    return pick;
}

// Survivors after the min_p + top_p cuts, in the sampled chain - the quantity an order or a threshold
// actually changes, used to assert a fixture can SEE the feature before asserting the kernel matches.
int sampled_cut(const std::vector<float>& l, const std::vector<int>& hist, const strata::kernels::SamplerParams& p) {
    auto penal = [&](float logit, int count) {
        if (count <= 0) return logit;
        if (logit <= 0.0f) logit *= p.penalty_repeat; else logit /= p.penalty_repeat;
        logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
        return logit;
    };
    auto count = [&](int v) { int c = 0; for (int h : hist) if (h == v) ++c; return c; };
    const int nv = (int) l.size();
    const int KMAX = 64;
    const int k = std::min(nv, (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX);
    std::vector<float> sel;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1; float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            const float s = penal(l[(size_t) v], count(v));
            if (best < 0 || s > bv) { best = v; bv = s; }
        }
        taken[(size_t) best] = 1; sel.push_back(bv);
    }
    int n_minp = (int) sel.size();
    if (p.min_p > 0.0f) {
        const float thresh = sel[0] + std::log(p.min_p);
        for (int i = 0; i < (int) sel.size(); ++i)
            if (sel[(size_t) i] < thresh) { n_minp = i; break; }
    }
    if (p.top_p >= 1.0f) return n_minp;
    double sum = 0.0;
    for (int i = 0; i < n_minp; ++i) sum += std::exp((double) sel[(size_t) i] - (double) sel[0]);
    double cum = 0.0;
    int cut = n_minp;
    for (int i = 0; i < n_minp; ++i) {
        cum += std::exp((double) sel[(size_t) i] - (double) sel[0]) / sum;
        if (cum >= (double) p.top_p) { cut = i + 1; break; }
    }
    if (cut < p.min_keep) cut = p.min_keep < n_minp ? p.min_keep : n_minp;
    return cut;
}

// ---- the kernel's own semantics, for the fixtures ----
//
// `sampled_reference` picks the first unpicked logit even when it is -inf or NaN; the kernels never pick either, and
// a round that finds nothing stores id 0 with a -inf logit (and later rounds treat id 0 as taken, as `sampler_kernel`
// does).  The mirror below follows the kernels, so rows with -inf, NaN, +inf and more requested than finite logits
// can be pinned exactly.  On rows without those it is `sampled_reference`.
struct SelList {
    std::vector<int> ids;
    std::vector<float> logit;
};

// The top_k list of `sampler_kernel` for one row: `window` is the counted history (the row's last penalty_last_n).
SelList mirror_select(const float* l, int nv, const std::vector<int>& window, const strata::kernels::SamplerParams& p,
                      int k) {
    std::vector<float> s((size_t) nv);
    for (int v = 0; v < nv; ++v) {
        int c = 0;
        for (int h : window) c += h == v;
        float x = l[v];
        if (c > 0) {
            if (x <= 0.0f) x *= p.penalty_repeat; else x /= p.penalty_repeat;
            x -= (float) c * p.penalty_freq + (c > 0 ? 1.0f : 0.0f) * p.penalty_present;
        }
        s[(size_t) v] = x;
    }
    SelList out;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = nv;
        float bv = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < nv; ++v)
            if (!taken[(size_t) v] && s[(size_t) v] > bv) { bv = s[(size_t) v]; best = v; }
        const int id = best < nv ? best : 0;
        taken[(size_t) id] = 1;
        out.ids.push_back(id);
        out.logit.push_back(bv);
    }
    return out;
}

// The tail of `sampler_kernel` over a list (its first `k` entries): top_p, min_p, temperature, the Philox draw.
int mirror_pick(const SelList& sel, int k, const strata::kernels::SamplerParams& p, int row) {
    int n_keep = k;
    float mx = sel.logit[0];
    for (int i = 1; i < k; ++i) mx = std::fmax(mx, sel.logit[(size_t) i]);
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += std::exp((double) sel.logit[(size_t) i] - (double) mx);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += std::exp((double) sel.logit[(size_t) i] - (double) mx) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = sel.logit[0] + std::log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel.logit[(size_t) i] < thresh) { n_keep = i; break; }
    }
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    auto scaled = [&](int i) { return sel.logit[(size_t) i] * inv_t; };
    float smx = scaled(0);
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, scaled(i));
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp((double) scaled(i) - (double) smx);
    int pick = sel.ids[(size_t) (n_keep > 0 ? n_keep - 1 : 0)];
    if (p.gumbel) {
        double best = -1.0;
        for (int i = 0; i < n_keep; ++i) {
            const double r = std::exp((double) scaled(i) - (double) smx) / sum /
                             host_gumbel_exp(p.seed, p.counter + (uint64_t) row, (uint32_t) sel.ids[(size_t) i]);
            if (r > best) { best = r; pick = sel.ids[(size_t) i]; }
        }
        return pick;
    }
    const float u = host_philox_uniform(p.seed, p.counter + (uint64_t) row);
    double cum = 0.0;
    for (int i = 0; i < n_keep; ++i) {
        cum += std::exp((double) scaled(i) - (double) smx) / sum;
        if ((double) u < cum) { pick = sel.ids[(size_t) i]; break; }
    }
    return pick;
}

int sampled_k(int top_k, int nv) { return std::min(nv, (top_k > 0 && top_k < 64) ? top_k : 64); }

// Rows uploaded once and sampled under many parameter sets: `sample` returns the picks of one launch on `stream`
// (nullptr: the legacy stream), -1 prefilled as in `run`.
struct DeviceRows {
    float* l = nullptr;
    int* h = nullptr;
    int* o = nullptr;
    int n_tokens = 0, nv = 0, hist_len = 0;
    DeviceRows(const std::vector<float>& logits, int n_tokens_, const std::vector<int>& hist, int hist_len_)
        : n_tokens(n_tokens_), nv((int) (logits.size() / (size_t) n_tokens_)), hist_len(hist_len_) {
        check(cudaMalloc(&l, logits.size() * sizeof(float)), "malloc logits");
        check(cudaMalloc(&o, (size_t) n_tokens * sizeof(int)), "malloc out");
        check(cudaMemcpy(l, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");
        if (hist_len > 0) {
            check(cudaMalloc(&h, hist.size() * sizeof(int)), "malloc hist");
            check(cudaMemcpy(h, hist.data(), hist.size() * sizeof(int), cudaMemcpyHostToDevice), "copy hist");
        }
    }
    DeviceRows(const DeviceRows&) = delete;
    DeviceRows& operator=(const DeviceRows&) = delete;
    ~DeviceRows() {
        cudaFree(l);
        cudaFree(o);
        if (h) cudaFree(h);
    }
    std::vector<int> sample(const strata::kernels::SamplerParams& p, cudaStream_t stream) {
        check(cudaMemset(o, 0xFF, (size_t) n_tokens * sizeof(int)), "fill out");
        strata::kernels::sample_tokens(l, n_tokens, nv, h, hist_len, p, o, stream);
        if (stream != nullptr) check(cudaStreamSynchronize(stream), "stream sync");
        std::vector<int> got((size_t) n_tokens);
        check(cudaMemcpy(got.data(), o, got.size() * sizeof(int), cudaMemcpyDeviceToHost), "back");
        return got;
    }
};

// The counted window of row t: the last min(penalty_last_n, hist_len) entries (none without penalties).
std::vector<int> window_of(const std::vector<int>& hist, int hist_len, int t, int last_n) {
    if (hist_len <= 0 || last_n <= 0) return {};
    const int h = std::min(last_n, hist_len);
    const int* row = hist.data() + (size_t) t * hist_len;
    return std::vector<int>(row + (hist_len - h), row + hist_len);
}

// `--bench`: the sampled path alone at the engine's vocabulary, per call, on the path the environment selects.
void bench_sampled() {
    const int NV = 248320;
    cudaStream_t s = nullptr;
    check(cudaStreamCreate(&s), "stream");
    std::mt19937 rng(20);
    std::normal_distribution<float> g(0.0f, 3.0f);
    for (int T : {1, 4, 8}) {
        std::vector<float> l((size_t) NV * T);
        for (auto& v : l) v = g(rng);
        float* d_l = nullptr;
        int* d_o = nullptr;
        check(cudaMalloc(&d_l, l.size() * sizeof(float)), "bench logits");
        check(cudaMalloc(&d_o, (size_t) T * sizeof(int)), "bench out");
        check(cudaMemcpy(d_l, l.data(), l.size() * sizeof(float), cudaMemcpyHostToDevice), "bench copy");
        for (int k : {20, 64}) {
            strata::kernels::SamplerParams p;
            p.top_k = k; p.top_p = 0.95f; p.temperature = 0.7f; p.seed = 1;
            for (int w = 0; w < 3; ++w) strata::kernels::sample_tokens(d_l, T, NV, nullptr, 0, p, d_o, s);
            check(cudaStreamSynchronize(s), "bench warmup");
            cudaEvent_t e0, e1;
            check(cudaEventCreate(&e0), "event");
            check(cudaEventCreate(&e1), "event");
            const int iters = 50;
            check(cudaEventRecord(e0, s), "record");
            for (int it = 0; it < iters; ++it) {
                p.counter = (uint64_t) it;
                strata::kernels::sample_tokens(d_l, T, NV, nullptr, 0, p, d_o, s);
            }
            check(cudaEventRecord(e1, s), "record");
            check(cudaEventSynchronize(e1), "bench sync");
            float ms = 0.0f;
            check(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
            std::printf("  bench: n_vocab %d, rows %d, top_k %2d, top_p 0.95: %8.1f us per call\n", NV, T, k,
                        1000.0 * (double) ms / iters);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
        }
        cudaFree(d_l);
        cudaFree(d_o);
    }
    cudaStreamDestroy(s);
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false, bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else if (std::string(argv[i]) == "--bench") bench = true;
        else { std::fprintf(stderr, "usage: sampler_parity [--selftest] [--bench]\n"); return 2; }
    }
    {
        // the sampled path under test; ctest runs this binary once per path
        auto on = [](const char* n) { const char* e = std::getenv(n); return e && *e && std::strcmp(e, "0") != 0; };
        std::printf("  sampled path: %s\n", on("STRATA_OLD_SAMPLER")         ? "sampler_kernel (STRATA_OLD_SAMPLER)"
                                            : on("STRATA_SAMPLER_ONE_BLOCK") ? "one block (STRATA_SAMPLER_ONE_BLOCK)"
                                                                             : "split top_k (default)");
    }
    if (bench) {
        bench_sampled();
        return 0;
    }
    int bad = 0;
    const int NV = 512, NT = 4;

    // Request bias affects every target row, including greedy; omission leaves logits and the default pick intact.
    // Compare against an independently pre-biased input through each ctest sampler implementation.
    {
        std::vector<float> l(NV * NT), bias(NV, 0.0f), adjusted;
        for (int t = 0; t < NT; ++t)
            for (int v = 0; v < NV; ++v) l[t * NV + v] = float(v % 17) / 4;
        l[0] = 50;
        bias[0] = -std::numeric_limits<float>::infinity();
        bias[7] = 20;
        bias[16] = -9;
        adjusted = l;
        for (int t = 0; t < NT; ++t)
            for (int v = 0; v < NV; ++v) adjusted[t * NV + v] += bias[v];
        float* db = nullptr;
        check(cudaMalloc(&db, NV * sizeof(float)), "bias malloc");
        check(cudaMemcpy(db, bias.data(), NV * sizeof(float), cudaMemcpyHostToDevice), "bias upload");
        std::vector<int> hist(NT * 4, 7);
        DeviceRows raw(l, NT, hist, 4), expected(adjusted, NT, hist, 4);
        for (bool greedy : {true, false}) for (bool penalty : {false, true}) {
            strata::kernels::SamplerParams p;
            p.greedy = greedy; p.temperature = greedy ? 0 : 0.8f; p.seed = 71; p.top_k = 16;
            if (penalty) { p.penalty_last_n = 4; p.penalty_freq = 3; p.penalty_repeat = 1.1f; }
            const auto want = expected.sample(p, nullptr);
            p.logit_bias = db;
            const auto got = raw.sample(p, nullptr);
            if (got != want || std::find(got.begin(), got.end(), 0) != got.end()) ++bad;
            p.logit_bias = nullptr;
            DeviceRows plain(l, NT, hist, 4);
            if (raw.sample(p, nullptr) != plain.sample(p, nullptr)) ++bad;
        }
        std::vector<float> unchanged(l.size());
        check(cudaMemcpy(unchanged.data(), raw.l, l.size() * sizeof(float), cudaMemcpyDeviceToHost), "raw logits");
        if (unchanged != l) ++bad;
        // The reporter's vocabulary scale, with only one permitted token: masked entries cannot win top-k tails.
        const int vocab = 248320;
        std::vector<float> large(vocab, 100), bans(vocab, -std::numeric_limits<float>::infinity());
        bans[109266] = 0; large[109266] = -50;
        cudaFree(db);
        check(cudaMalloc(&db, vocab * sizeof(float)), "large bias malloc");
        check(cudaMemcpy(db, bans.data(), bans.size() * sizeof(float), cudaMemcpyHostToDevice), "large bias upload");
        DeviceRows many(large, 1, {}, 0);
        for (bool greedy : {true, false}) {
            strata::kernels::SamplerParams p; p.greedy = greedy; p.logit_bias = db;
            if (many.sample(p, nullptr) != std::vector<int>{109266}) ++bad;
        }
        cudaFree(db);
    }

    // ---- fixture 1: plain greedy.  top_k = 0 (disabled), top_p = 1 (disabled), T = 1 -> argmax.
    {
        strata::kernels::SamplerParams p; p.top_k = 0; p.top_p = 1.0f; p.temperature = 1.0f; p.greedy = true;
        std::mt19937 rng(3); std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> l((size_t) NV * NT);
        for (auto& v : l) v = g(rng);
        std::vector<int> want((size_t) NT);
        for (int t = 0; t < NT; ++t) want[(size_t) t] = reference_pick({l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV}, p, false);
        bad += run("greedy argmax", l, NT, p, want);
    }

    // ---- fixture 2: THE ORDER FIXTURE.  T = 0.5 sharpens the distribution enough that top_p = 0.5 cuts
    // differently before and after the scaling, and the two orders then pick DIFFERENT tokens.
    {
        strata::kernels::SamplerParams p; p.top_k = 0; p.top_p = 0.5f; p.temperature = 0.5f;
        p.min_keep = 1; p.greedy = true;
        std::vector<float> l((size_t) NV * NT, -1000.0f);
        for (int t = 0; t < NT; ++t) {
            // a flat-ish head so the cumulative mass crosses 0.5 inside it, and one clear leader
            l[(size_t) t * NV + 0] = 3.0f;
            for (int v = 1; v < 8; ++v) l[(size_t) t * NV + v] = 2.6f - 0.05f * (float) v;
        }
        std::vector<int> want((size_t) NT), other((size_t) NT);
        for (int t = 0; t < NT; ++t) {
            const std::vector<float> row(l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV);
            want[(size_t) t] = reference_pick(row, p, false);      // the SPECIFIED order
            other[(size_t) t] = reference_pick(row, p, true);      // temperature first
        }
        // If the two orders agree on this fixture the test cannot see the order, and saying "the kernel
        // matches the spec" would be vacuous.
        // GREEDY CANNOT SEE THE ORDER, and saying otherwise would be a vacuous check: no filter removes the
        // global argmax, and temperature is monotonic, so the greedy pick is order-independent by
        // construction.  What the order changes is the top_p CUT, so the fixture is asserted to be
        // order-SENSITIVE at the cut, which is a property of the fixture rather than of the kernel.
        int cut_spec = 0, cut_alt = 0;
        for (int t = 0; t < NT; ++t) {
            const std::vector<float> row(l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV);
            cut_spec += reference_cut(row, p, false);
            cut_alt += reference_cut(row, p, true);
        }
        const bool distinguishable = (cut_spec != cut_alt);
        std::printf("  %-34s %s (survivors: spec %d, temp-first %d)\n", "order is observable on this fixture",
                    distinguishable ? "yes" : "*** NO - THE FIXTURE CANNOT SEE THE ORDER ***", cut_spec,
                    cut_alt);
        if (!distinguishable) ++bad;
        // greedy is still checked here, but as an ARGMAX check, not an order check
        bad += run("greedy over this fixture", l, NT, p, want);
    }

    // ---- fixture 3: greedy consumes NO random number.  Two runs with different seeds must agree, or the
    // seeded streams diverge between greedy and sampled runs - which docs/sampling.md §3 calls out.
    {
        strata::kernels::SamplerParams a; a.top_k = 20; a.top_p = 0.95f; a.temperature = 1.0f; a.greedy = true; a.seed = 1;
        strata::kernels::SamplerParams b = a; b.seed = 999999;
        std::mt19937 rng(5); std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> l((size_t) NV * NT);
        for (auto& v : l) v = g(rng);
        std::vector<int> wa((size_t) NT);
        for (int t = 0; t < NT; ++t) wa[(size_t) t] = reference_pick({l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV}, a, false);
        float* d_l = nullptr; int *d_a = nullptr, *d_b = nullptr;
        check(cudaMalloc(&d_l, l.size() * sizeof(float)), "m1");
        check(cudaMalloc(&d_a, (size_t) NT * sizeof(int)), "m2");
        check(cudaMalloc(&d_b, (size_t) NT * sizeof(int)), "m3");
        check(cudaMemcpy(d_l, l.data(), l.size() * sizeof(float), cudaMemcpyHostToDevice), "c1");
        strata::kernels::sample_tokens(d_l, NT, NV, nullptr, 0, a, d_a, nullptr);
        strata::kernels::sample_tokens(d_l, NT, NV, nullptr, 0, b, d_b, nullptr);
        std::vector<int> ga((size_t) NT), gb((size_t) NT);
        check(cudaMemcpy(ga.data(), d_a, ga.size() * sizeof(int), cudaMemcpyDeviceToHost), "g1");
        check(cudaMemcpy(gb.data(), d_b, gb.size() * sizeof(int), cudaMemcpyDeviceToHost), "g2");
        int mismatch = 0, wrong = 0;
        for (int t = 0; t < NT; ++t) {
            if (ga[(size_t) t] != gb[(size_t) t]) ++mismatch;
            if (ga[(size_t) t] != wa[(size_t) t]) ++wrong;
        }
        std::printf("  %-34s %s (seed-independent: %d differ; vs reference: %d wrong)\n",
                    "greedy ignores the seed", (!mismatch && !wrong) ? "matches" : "*** WRONG ***", mismatch,
                    wrong);
        bad += mismatch + wrong;
        cudaFree(d_l); cudaFree(d_a); cudaFree(d_b);
    }


    // ---- fixture 4: PENALTIES.  Two sub-cases, each built so the rule it tests decides the answer.
    {
        // host reference for the penalty stage, transcribed from llama_sampler_penalties_apply
        auto penal = [](float logit, int count, const strata::kernels::SamplerParams& p) {
            if (count <= 0) return logit;
            if (logit <= 0.0f) logit *= p.penalty_repeat; else logit /= p.penalty_repeat;
            logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
            return logit;
        };
        auto pick = [&](const std::vector<float>& l, const std::vector<int>& hist,
                        const strata::kernels::SamplerParams& p, bool divide_unconditionally) {
            int best = 0; float bv = 0; bool first = true;
            for (int v = 0; v < (int) l.size(); ++v) {
                int c = 0; for (int h : hist) if (h == v) ++c;
                float s;
                if (divide_unconditionally && c > 0) {
                    s = l[(size_t) v] / p.penalty_repeat
                        - (float) c * p.penalty_freq - (c > 0 ? 1.0f : 0.0f) * p.penalty_present;
                } else {
                    s = penal(l[(size_t) v], c, p);
                }
                if (first || s > bv) { bv = s; best = v; first = false; }
            }
            return best;
        };

        const int NV2 = 8, NT2 = 2;
        strata::kernels::SamplerParams p; p.top_k = 0; p.top_p = 1.0f; p.temperature = 1.0f;
        p.greedy = true; p.penalty_last_n = 4; p.penalty_repeat = 2.0f;

        // A: ALL logits negative, so the multiply-or-divide rule decides the argmax
        std::vector<float> la((size_t) NV2 * NT2, -8.0f);
        for (int t = 0; t < NT2; ++t) {
            la[(size_t) t * NV2 + 0] = -1.0f;      // in the history -> penalised
            la[(size_t) t * NV2 + 1] = -1.2f;      // not penalised -> should win
        }
        std::vector<int> hist_a((size_t) NT2 * 4, -1);
        for (int t = 0; t < NT2; ++t) hist_a[(size_t) t * 4 + 0] = 0;
        std::vector<int> want_a((size_t) NT2), alt_a((size_t) NT2);
        for (int t = 0; t < NT2; ++t) {
            const std::vector<float> row(la.begin() + (size_t) t * NV2, la.begin() + (size_t) (t + 1) * NV2);
            const std::vector<int> h(hist_a.begin() + (size_t) t * 4, hist_a.begin() + (size_t) (t + 1) * 4);
            want_a[(size_t) t] = pick(row, h, p, false);
            alt_a[(size_t) t] = pick(row, h, p, true);      // divide unconditionally
        }
        const bool A_visible = want_a[0] != alt_a[0];
        std::printf("  %-34s %s (multiply-rule %d, divide-always %d)\n",
                    "multiply-or-divide is observable", A_visible ? "yes" : "*** NO ***", want_a[0],
                    alt_a[0]);
        if (!A_visible) ++bad;
        else bad += run("penalties: repeat on negatives", la, NT2, p, want_a, hist_a, 4);

        // B: the PRESENCE penalty is a boolean, so two occurrences cost the same as one.  The runner's margin
        // is inside the difference between one and two applications.
        std::vector<float> lb((size_t) NV2 * NT2, -8.0f);
        for (int t = 0; t < NT2; ++t) {
            lb[(size_t) t * NV2 + 0] = 5.0f;       // seen twice -> penalised ONCE (presence) + freq*2
            lb[(size_t) t * NV2 + 1] = 3.4f;       // unseen
        }
        std::vector<int> hist_b((size_t) NT2 * 4, -1);
        for (int t = 0; t < NT2; ++t) {
            hist_b[(size_t) t * 4 + 0] = 0;
            hist_b[(size_t) t * 4 + 1] = 0;        // twice
        }
        strata::kernels::SamplerParams q = p; q.penalty_present = 1.5f; q.penalty_freq = 0.0f;
        std::vector<int> want_b((size_t) NT2);
        for (int t = 0; t < NT2; ++t) {
            const std::vector<float> row(lb.begin() + (size_t) t * NV2, lb.begin() + (size_t) (t + 1) * NV2);
            const std::vector<int> h(hist_b.begin() + (size_t) t * 4, hist_b.begin() + (size_t) (t + 1) * 4);
            want_b[(size_t) t] = pick(row, h, q, false);
        }
        std::printf("  %-34s want token %d (with present=1.5, token 0 goes 5.0/2 - 1.5 = 1.0 vs token 1 at "
                    "3.4)\n", "presence penalty is a boolean", want_b[0]);
        bad += run("penalties: presence is boolean", lb, NT2, q, want_b, hist_b, 4);
    }

    // ---- fixture 5: TEMPERATURE 0 MUST STILL RETURN THE ARGMAX.  Regression test for a real bug.
    //
    // The greedy branch used to read `apply_penalties(l[v] * inv_t, ...)`.  `inv_t` is 0.0f whenever
    // temperature <= 0, so at temperature 0 EVERY logit became 0.0f and the argmax returned index 0 - the
    // sampler emitted token 0 forever, whatever the model predicted.  OpenAI clients send `temperature: 0`
    // for greedy decoding, so this was reachable from any ordinary client.
    //
    // It survived because EVERY other greedy fixture in this file sets temperature = 1.0f, where inv_t = 1.0
    // and the extra multiply is harmless.  The bug needs temperature <= 0 to appear, and no fixture used it.
    // The fixture below makes token 0 the WORST token in every row, so returning 0 is unambiguously wrong.
    {
        const int NV3 = 512, NT3 = 4;
        std::vector<float> l((size_t) NV3 * NT3, -5.0f);
        std::vector<int> want((size_t) NT3);
        for (int t = 0; t < NT3; ++t) {
            const int best = 100 + t;                    // the argmax is never token 0
            l[(size_t) t * NV3 + best] = 3.0f;
            l[(size_t) t * NV3 + 0] = -9.0f;             // token 0 is the worst in the row
            want[(size_t) t] = best;
        }
        strata::kernels::SamplerParams p0;
        p0.top_k = 0; p0.top_p = 1.0f; p0.temperature = 0.0f; p0.greedy = false;
        bad += run("T=0 greedy=false is the argmax", l, NT3, p0, want);

        strata::kernels::SamplerParams p1 = p0; p1.greedy = true;
        bad += run("T=0 greedy=true  is the argmax", l, NT3, p1, want);

        strata::kernels::SamplerParams p2 = p0; p2.greedy = true; p2.temperature = 1.0f;
        bad += run("T=1 greedy=true  is the argmax", l, NT3, p2, want);
    }

    // ---- fixture 6: PENALTIES IN THE SAMPLED CHAIN.  Fixture 4 pins the greedy (argmax) path; the sampled
    // chain gets its own reference (the full chain with the host Philox) and its own observability check: with
    // the penalties on, the history row's favourite must LOSE a pick it would win penalty-free.
    {
        const int NV2 = 8, NT2 = 2;
        strata::kernels::SamplerParams p;
        p.top_k = 5; p.top_p = 0.9f; p.temperature = 0.8f; p.seed = 9; p.counter = 0;
        p.penalty_last_n = 4; p.penalty_repeat = 3.0f; p.penalty_freq = 0.2f; p.penalty_present = 0.6f;

        std::vector<float> l((size_t) NV2 * NT2, -8.0f);
        std::vector<int> hist((size_t) NT2 * 4, -1);
        for (int t = 0; t < NT2; ++t) {
            float* row = l.data() + (size_t) t * NV2;
            row[0] = 9.0f; row[1] = 4.5f; row[2] = 4.4f; row[3] = 4.3f;   // token 0 leads clean (9 vs 4.5)
            hist[(size_t) t * 4 + 0] = 0;                                  // and falls to 2.2/2.0 penalised
            hist[(size_t) t * 4 + 1] = t == 1 ? 0 : -1;                    // (repeat 3, freq, presence)
        }
        std::vector<int> want((size_t) NT2), clean((size_t) NT2);
        strata::kernels::SamplerParams clean_p = p;
        clean_p.penalty_last_n = 0; clean_p.penalty_repeat = 1.0f;
        clean_p.penalty_freq = 0.0f; clean_p.penalty_present = 0.0f;
        for (int t = 0; t < NT2; ++t) {
            const std::vector<float> row(l.begin() + (size_t) t * NV2, l.begin() + (size_t) (t + 1) * NV2);
            const std::vector<int> h(hist.begin() + (size_t) t * 4, hist.begin() + (size_t) (t + 1) * 4);
            want[(size_t) t] = sampled_reference(row, h, p, t);
            clean[(size_t) t] = sampled_reference(row, h, clean_p, t);
        }
        const bool visible = want[0] != clean[0] || want[1] != clean[1];
        std::printf("  %-34s %s (penalised picks %d/%d, clean %d/%d)\n",
                    "sampled penalties are observable", visible ? "yes" : "*** NO ***", want[0], want[1],
                    clean[0], clean[1]);
        if (!visible) ++bad;
        else bad += run("sampled chain: penalties + top_k/p", l, NT2, p, want, hist, 4);
    }

    // ---- fixture 7: MIN_P.  The cut is a PREFIX of the descending top_k list (logit >= max + log(min_p)),
    // so the fixture asserts the survivor count moves with the threshold (the observability half) and that
    // the kernel's pick equals the reference's through the full sampled chain (the correctness half).
    {
        const int NV3 = 8, NT3 = 2;
        std::vector<float> l((size_t) NV3 * NT3, -8.0f);
        for (int t = 0; t < NT3; ++t) {
            float* row = l.data() + (size_t) t * NV3;
            row[0] = 4.0f; row[1] = 3.5f; row[2] = 3.2f; row[3] = 3.1f;   // gaps keep the cut off the
            row[4] = 2.0f;                                                // logf/rounding knife edge
        }
        strata::kernels::SamplerParams base;
        base.top_k = 6; base.top_p = 1.0f; base.temperature = 0.9f; base.seed = 77;

        int c0 = 0, c05 = 0, c09 = 0;
        for (int t = 0; t < NT3; ++t) {
            const std::vector<float> row(l.begin() + (size_t) t * NV3, l.begin() + (size_t) (t + 1) * NV3);
            strata::kernels::SamplerParams q = base; q.min_p = 0.0f;
            c0 += sampled_cut(row, {}, q);
            q.min_p = 0.5f; c05 += sampled_cut(row, {}, q);
            q.min_p = 0.9f; c09 += sampled_cut(row, {}, q);
        }
        const bool visible = c0 > c05 && c05 > c09 && c09 >= NT3;
        std::printf("  %-34s %s (survivors: min_p 0 -> %d, 0.5 -> %d, 0.9 -> %d)\n",
                    "min_p cut is observable", visible ? "yes" : "*** NO ***", c0, c05, c09);
        if (!visible) ++bad;

        for (float mp : {0.0f, 0.5f, 0.9f}) {
            strata::kernels::SamplerParams q = base; q.min_p = mp;
            std::vector<int> want((size_t) NT3);
            for (int t = 0; t < NT3; ++t) {
                const std::vector<float> row(l.begin() + (size_t) t * NV3, l.begin() + (size_t) (t + 1) * NV3);
                want[(size_t) t] = sampled_reference(row, {}, q, t);
            }
            char name[64];
            std::snprintf(name, sizeof name, "sampled chain: min_p=%.1f", (double) mp);
            bad += run(name, l, NT3, q, want);
        }
    }

    // ---- fixture 8: THE PENALTY WINDOW IS THE TAIL.  With an 8-entry history and penalty_last_n = 4, only
    // the LAST four entries count: a token punished in the old half must come back to full strength, and one
    // punished in the tail half stays down.  The reference counts the same tail; the observability check runs
    // the reference once more WITHOUT the clamp (counting all 8) and requires the picks to differ.
    {
        const int NV4 = 8;
        strata::kernels::SamplerParams p;
        p.top_k = 0; p.top_p = 1.0f; p.temperature = 1.0f; p.greedy = true;
        p.penalty_last_n = 4; p.penalty_repeat = 3.0f; p.penalty_freq = 0.3f; p.penalty_present = 0.5f;

        std::vector<float> l((size_t) NV4, -8.0f);
        l[0] = 6.0f; l[3] = 6.5f;                       // token 0 leads clean; token 3 is the tail offender
        std::vector<int> hist = {0, 0, 0, 0, 3, 3, 3, 3};   // token 0 old (out), token 3 in the tail

        auto pick_clamped = [&](bool clamp) {
            int best = 0; float bv = 0; bool first = true;
            for (int v = 0; v < NV4; ++v) {
                int c = 0;
                for (int i = 0; i < (clamp ? 4 : 8); ++i) if (hist[(size_t) (8 - (clamp ? 4 : 8) + i)] == v) ++c;
                float logit = l[(size_t) v];
                if (c > 0) { logit = logit <= 0.0f ? logit * p.penalty_repeat : logit / p.penalty_repeat;
                             logit -= (float) c * p.penalty_freq + p.penalty_present; }
                if (first || logit > bv) { bv = logit; best = v; first = false; }
            }
            return best;
        };
        const int want = pick_clamped(true), unclamped = pick_clamped(false);
        const bool visible = want != unclamped;
        std::printf("  %-34s %s (clamped pick %d, full-history pick %d)\n",
                    "penalty window clamp is observable", visible ? "yes" : "*** NO ***", want, unclamped);
        if (!visible) ++bad;
        else bad += run("penalty window: tail only", {l.begin(), l.end()}, 1, p, {want}, hist, 8);
    }

    // ---- fixture 9: ONE PENALTIES STAGE (issue #53), against an independently computed distribution, not the
    // reference above (which had copied the kernel's mistake).  Two tokens with equal logits, token 0 in the
    // history, presence penalty 1.5, temperature 0.7: llama.cpp's chain gives token 0 the logit (0 - 1.5) / 0.7,
    // P = 1 / (1 + exp(1.5 / 0.7)) = 0.1050; a second penalty after the temperature made it 0.0255.  Over 8,000
    // draws the kernel's share of token 0 must be near 0.105 (4 sigma = 0.014), and its picks must equal the
    // reference's draw for draw.
    {
        const int NT9 = 8000;
        strata::kernels::SamplerParams p;
        p.top_k = 2; p.top_p = 1.0f; p.min_p = 0.0f; p.temperature = 0.7f; p.seed = 53; p.counter = 0;
        p.penalty_last_n = 1; p.penalty_repeat = 1.0f; p.penalty_freq = 0.0f; p.penalty_present = 1.5f;
        std::vector<float> l((size_t) 2 * NT9, 0.0f);
        std::vector<int> hist((size_t) NT9, 0);
        std::vector<int> want((size_t) NT9);
        int zeros = 0;
        for (int t = 0; t < NT9; ++t) {
            want[(size_t) t] = sampled_reference({0.0f, 0.0f}, {0}, p, t);
            zeros += want[(size_t) t] == 0;
        }
        const double expect = 1.0 / (1.0 + std::exp(1.5 / 0.7)), share = (double) zeros / NT9;
        const bool near = std::fabs(share - expect) < 0.014;
        std::printf("  %-34s %s (token 0 drawn %.4f of %d, expected %.4f; twice-penalised would be 0.0255)\n",
                    "one penalties stage (#53)", near ? "yes" : "*** NO ***", share, NT9, expect);
        if (!near) ++bad;
        bad += run("sampled chain: #53's example", l, NT9, p, want, hist, 1);
    }

    // ---- fixture 10: TOP_P BEFORE MIN_P (llama.cpp's order).  Probabilities 0.4 / 0.3 / 0.2 / 0.1, top_p 0.75,
    // min_p 0.3 (keeps p >= 0.12): top_p over all four keeps three (0.4 + 0.3 < 0.75 <= 0.9) and min_p keeps them;
    // min_p first would drop 0.1, renormalise, and top_p would then stop at two (0.444 + 0.333 >= 0.75).  So token
    // 2 must be drawn sometimes - never in the old order - and every pick must equal the reference's.
    {
        const int NT10 = 256;
        strata::kernels::SamplerParams p;
        p.top_k = 4; p.top_p = 0.75f; p.min_p = 0.3f; p.temperature = 1.0f; p.seed = 10; p.counter = 0;
        const float lp[4] = {std::log(0.4f), std::log(0.3f), std::log(0.2f), std::log(0.1f)};
        std::vector<float> row = {lp[0], lp[1], lp[2], lp[3], -30.0f, -30.0f, -30.0f, -30.0f};
        std::vector<float> l;
        for (int t = 0; t < NT10; ++t) l.insert(l.end(), row.begin(), row.end());
        std::vector<int> want((size_t) NT10);
        int twos = 0;
        for (int t = 0; t < NT10; ++t) {
            want[(size_t) t] = sampled_reference(row, {}, p, t);
            twos += want[(size_t) t] == 2;
        }
        std::printf("  %-34s %s (token 2 drawn %d of %d times)\n", "top_p before min_p is observable",
                    twos > 0 ? "yes" : "*** NO ***", twos, NT10);
        if (twos == 0) ++bad;
        bad += run("sampled chain: top_p then min_p", l, NT10, p, want);
    }

    // ---- fixture 11: A STALE HISTORY WITH last_n = 0 IS INERT (PR #59).  A caller can hand over a history buffer
    // from a previous penalised request while this request disables the penalties - the run must equal the
    // no-history run in both kernels, and the bitmap the launch did not size must stay untouched.
    {
        std::mt19937 rng(11); std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> l((size_t) NV * NT);
        for (auto& v : l) v = g(rng);
        std::vector<int> hist((size_t) NT * 8, -1);
        for (int t = 0; t < NT; ++t) hist[(size_t) t * 8] = 3;

        strata::kernels::SamplerParams p;
        p.top_k = 20; p.top_p = 0.95f; p.temperature = 0.8f; p.seed = 5;
        std::vector<int> want((size_t) NT);
        for (int t = 0; t < NT; ++t)
            want[(size_t) t] = sampled_reference({l.begin() + (size_t) t * NV,
                                                  l.begin() + (size_t) (t + 1) * NV}, {}, p, t);
        bad += run("stale history, last_n=0 (sampled)", l, NT, p, want, hist, 8);

        strata::kernels::SamplerParams gp = p; gp.greedy = true; gp.top_k = 0; gp.top_p = 1.0f;
        std::vector<int> gwant((size_t) NT);
        for (int t = 0; t < NT; ++t)
            gwant[(size_t) t] = reference_pick({l.begin() + (size_t) t * NV,
                                                l.begin() + (size_t) (t + 1) * NV}, gp, false);
        bad += run("stale history, last_n=0 (greedy)", l, NT, gp, gwant, hist, 8);
    }

    // ---- fixture 12: ONE HISTORY PER ROW (engine 0.1.19).  A verify window samples T rows, and row t's pick
    // follows the drafts 1..t: its penalties must count them.  The engine used to stage row 0 alone, so rows
    // 1..T-1 read slots nobody wrote.  Here every row gets `penalty_rows`' history and is pinned against a
    // per-row scalar reference; the fixture first asserts it can SEE the difference - with row 0's history
    // copied to every row (the nearest well-defined stand-in for the old staging) some row must pick differently.
    // Row t's own newest token (window[t]) is its favourite by a margin the presence penalty overturns.
    {
        auto greedy_pen = [](const std::vector<float>& l, const std::vector<int>& hist,
                             const strata::kernels::SamplerParams& p) {
            int best = 0; float bv = 0; bool first = true;
            for (int v = 0; v < (int) l.size(); ++v) {
                int c = 0; for (int h : hist) if (h == v) ++c;
                float s = l[(size_t) v];
                if (c > 0) {
                    if (s <= 0.0f) s *= p.penalty_repeat; else s /= p.penalty_repeat;
                    s -= (float) c * p.penalty_freq + p.penalty_present;
                }
                if (first || s > bv) { bv = s; best = v; first = false; }
            }
            return best;
        };
        std::mt19937 rng(12); std::normal_distribution<float> g(0.0f, 1.0f);
        const int TAIL = 5000;                           // longer than the widest window below
        std::vector<int32_t> tail((size_t) TAIL);
        for (auto& v : tail) v = (int32_t) (rng() % NV);
        int observable = 0, rows_checked = 0;
        for (int T : {1, 2, 4, 8}) {
            for (int H : {1, 64, 1024, 4096}) {
                std::vector<int32_t> window((size_t) T);
                for (int t = 0; t < T; ++t) window[(size_t) t] = (int32_t) (100 + 37 * t);   // distinct, in range
                std::vector<int32_t> rows((size_t) T * H);
                strata::kernels::penalty_rows(tail.data(), TAIL, window.data(), T, H, rows.data());
                std::vector<float> l((size_t) T * NV);
                for (auto& v : l) v = g(rng);
                for (int t = 0; t < T; ++t) l[(size_t) t * NV + window[(size_t) t]] = 5.0f;
                strata::kernels::SamplerParams gp;
                gp.greedy = true; gp.temperature = 0.0f; gp.top_k = 0; gp.top_p = 1.0f;
                gp.penalty_last_n = H; gp.penalty_present = 4.0f;
                strata::kernels::SamplerParams sp2;
                sp2.top_k = 20; sp2.top_p = 0.9f; sp2.temperature = 0.7f; sp2.seed = 1000 + (uint64_t) H;
                sp2.counter = 77; sp2.penalty_last_n = H; sp2.penalty_present = 4.0f; sp2.penalty_repeat = 1.1f;
                std::vector<int> gwant((size_t) T), swant((size_t) T), rows_int(rows.begin(), rows.end());
                for (int t = 0; t < T; ++t) {
                    const std::vector<float> lr(l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV);
                    const std::vector<int> own(rows.begin() + (size_t) t * H, rows.begin() + (size_t) (t + 1) * H);
                    const std::vector<int> row0(rows.begin(), rows.begin() + H);
                    gwant[(size_t) t] = greedy_pen(lr, own, gp);
                    swant[(size_t) t] = sampled_reference(lr, own, sp2, t);
                    if (t > 0 && greedy_pen(lr, row0, gp) != gwant[(size_t) t]) ++observable;
                    ++rows_checked;
                }
                char name[64];
                std::snprintf(name, sizeof name, "per-row history T=%d H=%d greedy", T, H);
                bad += run(name, l, T, gp, gwant, rows_int, H);
                std::snprintf(name, sizeof name, "per-row history T=%d H=%d sampled", T, H);
                bad += run(name, l, T, sp2, swant, rows_int, H);
            }
        }
        std::printf("  %-34s %s (%d drafted rows pick differently with row 0's history, of %d rows)\n",
                    "per-row histories are observable", observable > 0 ? "yes" : "*** NO ***", observable,
                    rows_checked);
        if (observable == 0) ++bad;
    }

    // ---- fixture 13: HISTORY IDS OUTSIDE THE VOCABULARY ARE IGNORED.  The bitmap is sized for n_vocab bits;
    // an id >= n_vocab used to set a bit past its end (a shared-memory write out of bounds - compute-sanitizer
    // memcheck reports it).  They can never be a candidate, so the result equals the reference without them.
    {
        std::mt19937 rng(13); std::normal_distribution<float> g(0.0f, 1.0f);
        const int H = 16;
        std::vector<float> l((size_t) NV * NT);
        for (auto& v : l) v = g(rng);
        std::vector<int> hist((size_t) NT * H, -1), valid_only((size_t) NT * H, -1);
        const int junk[] = {NV, NV + 1000, 0x7fffffff, -5, 1 << 20};
        for (int t = 0; t < NT; ++t) {
            for (int j = 0; j < H; ++j) {
                const bool bogus = j % 3 == 0;
                const int v = bogus ? junk[(size_t) (j / 3) % 5] : (int) (rng() % NV);
                hist[(size_t) t * H + j] = v;
                if (!bogus) valid_only[(size_t) t * H + j] = v;
            }
            l[(size_t) t * NV + hist[(size_t) t * H + 1]] = 4.0f;     // a penalised favourite, so penalties matter
        }
        strata::kernels::SamplerParams gp;
        gp.greedy = true; gp.temperature = 0.0f; gp.top_k = 0; gp.top_p = 1.0f;
        gp.penalty_last_n = H; gp.penalty_present = 3.0f;
        strata::kernels::SamplerParams sp2 = gp;
        sp2.greedy = false; sp2.temperature = 0.8f; sp2.top_k = 20; sp2.top_p = 0.95f; sp2.seed = 13;
        std::vector<int> gwant((size_t) NT), swant((size_t) NT);
        for (int t = 0; t < NT; ++t) {
            const std::vector<float> lr(l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV);
            const std::vector<int> ok(valid_only.begin() + (size_t) t * H, valid_only.begin() + (size_t) (t + 1) * H);
            {   // the greedy reference with the penalty (reference_pick has none)
                int best = 0; float bv = 0; bool first = true;
                for (int v = 0; v < NV; ++v) {
                    int c = 0; for (int h : ok) if (h == v) ++c;
                    float s = lr[(size_t) v];
                    if (c > 0) { if (s <= 0.0f) s *= gp.penalty_repeat; else s /= gp.penalty_repeat; s -= gp.penalty_present; }
                    if (first || s > bv) { bv = s; best = v; first = false; }
                }
                gwant[(size_t) t] = best;
            }
            swant[(size_t) t] = sampled_reference(lr, ok, sp2, t);
        }
        bad += run("out-of-vocab history ids (greedy)", l, NT, gp, gwant, hist, H);
        bad += run("out-of-vocab history ids (sampled)", l, NT, sp2, swant, hist, H);
    }

    // ---- fixture 14: THE top_k CONTRACT.  1..64 as given; 0 ("off") and anything wider use the widest list the
    // kernel keeps, 64 - and every row is written (the sampled kernel used to print an error for 0 and leave the
    // row unwritten, which a verify window then read as a token; `run` pre-fills -1 so that fails here).
    {
        std::mt19937 rng(14); std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> l((size_t) NV * NT);
        for (auto& v : l) v = g(rng) * 0.3f;             // flat: the 64-wide list matters to the draw
        strata::kernels::SamplerParams p64;
        p64.top_k = 64; p64.top_p = 1.0f; p64.temperature = 1.5f; p64.seed = 14;
        std::vector<int> want((size_t) NT);
        for (int t = 0; t < NT; ++t)
            want[(size_t) t] = sampled_reference({l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV},
                                                 {}, p64, t);
        bad += run("sampled top_k=64", l, NT, p64, want);
        strata::kernels::SamplerParams p0 = p64; p0.top_k = 0;
        bad += run("sampled top_k=0 means 64", l, NT, p0, want);
        strata::kernels::SamplerParams p100 = p64; p100.top_k = 100;
        bad += run("sampled top_k=100 means 64", l, NT, p100, want);
        strata::kernels::SamplerParams pneg = p64; pneg.top_k = -3;
        bad += run("sampled top_k=-3 means 64", l, NT, pneg, want);
    }

    // ---- fixture 15: `penalty_rows`, host only, against the plain definition: row t = the last h tokens of
    // tail + window[0..t], -1 padded in front.  Covers a tail shorter than, equal to and longer than h, and row 0
    // equal to the single row the engine staged before 0.1.19.
    {
        int wrong = 0, cases = 0;
        for (int n_tail : {0, 1, 5, 63, 64, 65, 300}) {
            for (int T : {1, 3, 8}) {
                for (int h : {1, 4, 64, 100}) {
                    std::vector<int32_t> tail((size_t) n_tail), window((size_t) T);
                    for (int i = 0; i < n_tail; ++i) tail[(size_t) i] = 1000 + i;
                    for (int i = 0; i < T; ++i) window[(size_t) i] = 5000 + i;
                    std::vector<int32_t> rows((size_t) T * h, 12345);
                    strata::kernels::penalty_rows(tail.data(), n_tail, window.data(), T, h, rows.data());
                    for (int t = 0; t < T; ++t) {
                        std::vector<int32_t> seq(tail);
                        seq.insert(seq.end(), window.begin(), window.begin() + t + 1);
                        std::vector<int32_t> expect((size_t) h, -1);
                        const int take = (int) std::min<size_t>((size_t) h, seq.size());
                        for (int j = 0; j < take; ++j) expect[(size_t) (h - take + j)] = seq[seq.size() - take + j];
                        if (!std::equal(expect.begin(), expect.end(), rows.begin() + (size_t) t * h)) ++wrong;
                        ++cases;
                    }
                    // row 0 = the old single-row staging: consumed tail, then the fed-back head last
                    std::vector<int32_t> old((size_t) h, -1);
                    const int take0 = (int) std::min<int64_t>(h, (int64_t) n_tail + 1);
                    for (int j = 0; j < take0 - 1; ++j) old[(size_t) (h - take0 + j)] = tail[(size_t) (n_tail - (take0 - 1) + j)];
                    old[(size_t) (h - 1)] = window[0];
                    if (!std::equal(old.begin(), old.end(), rows.begin())) ++wrong;
                    ++cases;
                }
            }
        }
        std::printf("  %-34s %s (%d of %d rows differ)\n", "penalty_rows layout", wrong ? "*** WRONG ***" : "matches",
                    wrong, cases);
        bad += wrong;
    }

    // A continuous stream and individual decode calls consume the same draw counters.
    {
        constexpr int count = 32, vocab = 16;
        std::vector<float> uniform(count * vocab, 0.0f);
        float* input = nullptr;
        int* output = nullptr;
        check(cudaMalloc(&input, uniform.size() * sizeof(float)), "counter logits");
        check(cudaMalloc(&output, count * sizeof(int)), "counter output");
        check(cudaMemcpy(input, uniform.data(), uniform.size() * sizeof(float), cudaMemcpyHostToDevice), "counter upload");
        strata::kernels::SamplerParams p;
        p.top_k = vocab; p.top_p = 1.0f; p.seed = 123; p.counter = (uint64_t(1) << 32) + 7;
        strata::kernels::sample_tokens(input, count, vocab, nullptr, 0, p, output, nullptr);
        std::vector<int> batch(count), singles(count), repeated(count);
        check(cudaMemcpy(batch.data(), output, count * sizeof(int), cudaMemcpyDeviceToHost), "counter batch");
        for (int i = 0; i < count; ++i) {
            auto one = p; one.counter += i;
            strata::kernels::sample_tokens(input, 1, vocab, nullptr, 0, one, output + i, nullptr);
        }
        check(cudaMemcpy(singles.data(), output, count * sizeof(int), cudaMemcpyDeviceToHost), "counter singles");
        strata::kernels::sample_tokens(input, count, vocab, nullptr, 0, p, output, nullptr);
        check(cudaMemcpy(repeated.data(), output, count * sizeof(int), cudaMemcpyDeviceToHost), "counter repeated");
        bool varies = false;
        for (int i = 1; i < count; ++i) varies |= batch[i] != batch[0];
        const bool valid = batch == singles && batch == repeated && varies;
        std::printf("  sampler draw counter segmentation/repeat: %s\n", valid ? "PASS" : "FAIL");
        bad += !valid;
        cudaFree(input); cudaFree(output);
    }

    // ---- fixture 16: THE WHOLE top_k LIST, POSITION BY POSITION, UNDER TIES.  A pick shows the list
    // through one draw; this reads the list itself.  Every row holds a +inf logit, so the tail's arithmetic is NaN
    // (inf - inf), no cut fires and no draw lands: the chain returns its LAST kept entry, sel_ids[k - 1].  Launching
    // top_k = 1..64 then reads the list one position at a time - its set and its order.  The rows make the order
    // rest on the tie rule: hundreds of logits share the top finite values, spread over every split block, warp and
    // lane; -0 and +0 tie; -inf and NaN are mixed in; a row has fewer finite logits than 64 (the sentinel id 0 must
    // come out); a penalised row lands its penalised tokens exactly on other tokens' values.  Vocabularies: the
    // engine's 248,320 (a partial last split block), 100,003, 262,144 (the widest split), 262,145 (one more: the
    // one-block fallback) and 1,000.
    {
        const float inf = std::numeric_limits<float>::infinity();
        const float qnan = std::numeric_limits<float>::quiet_NaN();
        int wrong = 0, probes = 0, sentinels = 0;
        for (int nv : {248320, 100003, 262144, 262145, 1000}) {
            const int T = 4, H = 64;
            std::mt19937 rng((unsigned) (1600 + nv));
            std::normal_distribution<float> g(0.0f, 1.0f);
            std::vector<float> l((size_t) nv * T);
            for (auto& v : l) v = std::floor(g(rng) * 4.0f) / 4.0f;          // quarter steps: ties everywhere
            auto spread = [&](int j) { return (int) (((int64_t) j * 7919 + 13) % nv); };
            // row 0: 20 logits at 6.25 and 300 at 6.0 over the whole row, two +inf
            float* r0 = l.data();
            for (int j = 0; j < 320; ++j) r0[spread(j)] = j < 20 ? 6.25f : 6.0f;
            r0[nv / 2] = inf;
            r0[nv - 1] = inf;
            // row 1: nothing above 0, every zero signed by its id's parity (-0 at even ids), one +inf
            float* r1 = l.data() + (size_t) nv;
            for (int v = 0; v < nv; ++v) {
                r1[v] = -std::fabs(r1[v]);
                if (r1[v] == 0.0f) r1[v] = (v & 1) ? 0.0f : -0.0f;
            }
            r1[3] = inf;
            // row 2: -inf everywhere but ten finite logits (two values), two NaN and a +inf: 11 candidates in all
            float* r2 = l.data() + (size_t) 2 * nv;
            for (int v = 0; v < nv; ++v) r2[v] = -inf;
            for (int j = 0; j < 10; ++j) r2[spread(j + 400)] = j < 5 ? 1.0f : 0.5f;
            r2[spread(500)] = qnan;
            r2[spread(501)] = qnan;
            r2[spread(502)] = inf;
            // row 3: penalties.  Window ids on block and warp edges, repeats, and ids outside the vocabulary; the
            // penalised tokens sit at 9.0, which repeat 2 / freq 0.25 / present 0.5 turns into 4.0 - 0.25 x count
            // (3.75, 3.5, ...), values the quarter-step logits share.
            float* r3 = l.data() + (size_t) 3 * nv;
            std::vector<int> hist((size_t) T * H, -1);
            int* h3 = hist.data() + (size_t) 3 * H;
            const int edges[] = {0, 1023, 1024, 4095, 4096, 8191, 8192, 12345, nv / 2 + 1, nv - 2};
            int hn = 0;
            for (int e : edges)
                if (e < nv && e != nv / 2) {
                    h3[hn++] = e;
                    if (hn % 3 == 0) h3[hn++] = e;                              // counted twice
                    r3[e] = 9.0f;
                }
            h3[hn++] = nv;                                                      // ignored: outside
            h3[hn++] = nv + 77;
            h3[hn++] = -5;
            for (int j = 0; j < 30; ++j) r3[spread(j + 600)] = j < 15 ? 3.75f : 3.5f;
            r3[nv / 2] = inf;                                                   // not in the window

            strata::kernels::SamplerParams base;
            base.top_p = 1.0f; base.min_p = 0.0f; base.temperature = 0.8f; base.seed = 16;
            base.penalty_last_n = H; base.penalty_repeat = 2.0f; base.penalty_freq = 0.25f;
            base.penalty_present = 0.5f;
            const int kmax = sampled_k(64, nv);
            std::vector<SelList> lists;
            for (int t = 0; t < T; ++t)
                lists.push_back(mirror_select(l.data() + (size_t) t * nv, nv, window_of(hist, H, t, H), base, kmax));
            DeviceRows rows(l, T, hist, H);
            for (int top_k = 0; top_k <= 66; ++top_k) {
                strata::kernels::SamplerParams p = base;
                p.top_k = top_k == 65 ? 100 : top_k == 66 ? -3 : top_k;       // 0, 100 and -3 mean 64
                p.top_p = (top_k & 1) ? 0.5f : 1.0f;                            // both tail branches (NaN: no cut)
                p.counter = (uint64_t) top_k;
                const int k = sampled_k(p.top_k, nv);
                const std::vector<int> got = rows.sample(p, nullptr);
                for (int t = 0; t < T; ++t) {
                    const int want = lists[(size_t) t].ids[(size_t) (k - 1)];
                    sentinels += (t == 2 && k > 11);
                    ++probes;
                    if (got[(size_t) t] != want) {
                        if (wrong < 8)
                            std::printf("    n_vocab %d row %d top_k %d: position %d want id %d got %d\n", nv, t,
                                        p.top_k, k - 1, want, got[(size_t) t]);
                        ++wrong;
                    }
                }
            }
        }
        // the fixture must reach the sentinel (row 2 has 11 candidates) or it cannot see the "nothing left" rule
        std::printf("  %-34s %s (%d of %d positions differ; %d sentinel positions)\n", "top_k list under ties",
                    wrong || !sentinels ? "*** WRONG ***" : "matches", wrong, probes, sentinels);
        bad += wrong + (sentinels == 0);
    }

    // ---- fixture 17: SAMPLED DRAWS UNDER TIES.  The realistic chain - finite logits on half steps,
    // so dozens of tokens share each value near the top - through every stage: top_k 1 / 20 / 64, top_p 0.9 / 1,
    // min_p 0 / 0.05, a hot temperature that spreads the draws over the whole list, penalties off and on (half of
    // each window on the row's head, so they reorder it).  17 rows (the split's scratch is first sized for 16: this
    // regrows it), on the legacy stream and on a created one.  Observability: some picks must be tokens that tie
    // with another kept token, or the tie rule would go untested.
    {
        const int T = 17, H = 64;
        cudaStream_t cs = nullptr;
        check(cudaStreamCreate(&cs), "fixture 17 stream");
        int wrong = 0, draws = 0, tied = 0;
        for (int nv : {248320, 512}) {
            std::mt19937 rng((unsigned) (1700 + nv));
            std::normal_distribution<float> g(0.0f, 1.5f);
            std::vector<float> l((size_t) nv * T);
            for (auto& v : l) v = std::floor(g(rng) * 2.0f) / 2.0f;
            if (nv == 512)
                for (auto& v : l) v = std::floor(v / 2.0f);                    // whole steps: a few big tie groups
            std::vector<int> hist((size_t) T * H);
            for (int t = 0; t < T; ++t) {
                const float* row = l.data() + (size_t) t * nv;
                const float mx = *std::max_element(row, row + nv);
                std::vector<int> head;
                for (int v = 0; v < nv; ++v)
                    if (row[v] >= mx - 1.0f) head.push_back(v);
                for (int j = 0; j < H; ++j)
                    hist[(size_t) t * H + j] = (j & 1) ? (int) (rng() % (unsigned) nv)
                                                       : head[(size_t) (rng() % (unsigned) head.size())];
            }
            DeviceRows rows(l, T, hist, H);
            int config = 0;
            for (int pen = 0; pen < 2; ++pen) {
                strata::kernels::SamplerParams base;
                base.temperature = 2.5f; base.seed = 17;
                base.penalty_last_n = pen ? H : 0;
                base.penalty_repeat = 1.25f; base.penalty_freq = 0.25f; base.penalty_present = 0.5f;
                const int kmax = sampled_k(64, nv);
                std::vector<SelList> lists;
                for (int t = 0; t < T; ++t)
                    lists.push_back(mirror_select(l.data() + (size_t) t * nv, nv, window_of(hist, H, t, base.penalty_last_n),
                                                  base, kmax));
                for (int top_k : {1, 20, 64})
                    for (float top_p : {0.9f, 1.0f})
                        for (float min_p : {0.0f, 0.05f})
                            for (cudaStream_t st : {(cudaStream_t) nullptr, cs}) {
                                strata::kernels::SamplerParams p = base;
                                p.top_k = top_k; p.top_p = top_p; p.min_p = min_p;
                                p.counter = (uint64_t) (1000 * ++config);
                                const int k = sampled_k(top_k, nv);
                                const std::vector<int> got = rows.sample(p, st);
                                for (int t = 0; t < T; ++t) {
                                    const SelList& sl = lists[(size_t) t];
                                    const int want = mirror_pick(sl, k, p, t);
                                    ++draws;
                                    int same = 0;
                                    for (int i = 0; i < k; ++i) {
                                        if (sl.ids[(size_t) i] != want) continue;
                                        for (int j = 0; j < k; ++j) same += sl.logit[(size_t) j] == sl.logit[(size_t) i];
                                        break;
                                    }
                                    tied += same > 1;
                                    if (got[(size_t) t] != want) {
                                        if (wrong < 8)
                                            std::printf("    n_vocab %d row %d top_k %d top_p %.2f min_p %.2f pen %d: "
                                                        "want %d got %d\n", nv, t, top_k, (double) top_p,
                                                        (double) min_p, pen, want, got[(size_t) t]);
                                        ++wrong;
                                    }
                                }
                            }
            }
        }
        cudaStreamDestroy(cs);
        std::printf("  %-34s %s (%d of %d draws differ; %d picks tie with another kept token)\n",
                    "sampled draws under ties", wrong || !tied ? "*** WRONG ***" : "matches", wrong, draws, tied);
        bad += wrong + (tied == 0);
    }

    // ---- fixture 18: THE DEFAULT PATH'S AUTOMATIC FALLBACKS.  (a) A stream under graph capture
    // (ThreadLocal mode) gets the one-block kernel, penalty bitmap in dynamic shared memory, inside the graph: the
    // capture must succeed and the replayed graph (twice) must pick the mirror's tokens; the same stream uncaptured
    // then takes the split path (its scratch is first allocated after the capture) and picks the same.  (b) More
    // rows than a split launch takes (64) fall back to one block; exactly 64 stay split.
    {
        int wrong = 0, draws = 0;
        const int H = 64;
        auto compare = [&](const char* what, const std::vector<int>& got, const std::vector<SelList>& lists, int k,
                           const strata::kernels::SamplerParams& p) {
            for (size_t t = 0; t < got.size(); ++t) {
                const int want = mirror_pick(lists[t], k, p, (int) t);
                ++draws;
                if (got[t] != want) {
                    if (wrong < 8) std::printf("    %s row %zu: want %d got %d\n", what, t, want, got[t]);
                    ++wrong;
                }
            }
        };
        auto make = [&](int nv, int T, unsigned seed, std::vector<float>& l, std::vector<int>& hist) {
            std::mt19937 rng(seed);
            std::normal_distribution<float> g(0.0f, 1.5f);
            l.assign((size_t) nv * T, 0.0f);
            for (auto& v : l) v = std::floor(g(rng) * 2.0f) / 2.0f;
            hist.assign((size_t) T * H, 0);
            for (auto& h : hist) h = (int) (rng() % (unsigned) nv);
        };
        {
            const int nv = 248320, T = 3;
            std::vector<float> l;
            std::vector<int> hist;
            make(nv, T, 1800u, l, hist);
            strata::kernels::SamplerParams p;
            p.top_k = 20; p.top_p = 0.9f; p.temperature = 2.5f; p.seed = 18; p.counter = 77;
            p.penalty_last_n = H; p.penalty_repeat = 1.25f; p.penalty_freq = 0.25f; p.penalty_present = 0.5f;
            const int k = sampled_k(p.top_k, nv);
            std::vector<SelList> lists;
            for (int t = 0; t < T; ++t)
                lists.push_back(mirror_select(l.data() + (size_t) t * nv, nv, window_of(hist, H, t, H), p, k));
            DeviceRows rows(l, T, hist, H);
            cudaStream_t cs = nullptr;
            check(cudaStreamCreate(&cs), "fixture 18 stream");
            check(cudaMemset(rows.o, 0xFF, (size_t) T * sizeof(int)), "fixture 18 fill");
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            check(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "begin capture");
            strata::kernels::sample_tokens(rows.l, T, nv, rows.h, H, p, rows.o, cs);
            check(cudaStreamEndCapture(cs, &graph), "end capture");
            check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
            for (int replay = 0; replay < 2; ++replay) {
                check(cudaMemset(rows.o, 0xFF, (size_t) T * sizeof(int)), "fixture 18 refill");
                check(cudaGraphLaunch(exec, cs), "graph launch");
                check(cudaStreamSynchronize(cs), "graph sync");
                std::vector<int> got((size_t) T);
                check(cudaMemcpy(got.data(), rows.o, got.size() * sizeof(int), cudaMemcpyDeviceToHost), "back");
                compare("captured graph", got, lists, k, p);
            }
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph);
            compare("same stream, uncaptured", rows.sample(p, cs), lists, k, p);
            cudaStreamDestroy(cs);
        }
        for (int T : {64, 70}) {
            const int nv = 512;
            std::vector<float> l;
            std::vector<int> hist;
            make(nv, T, 1810u + (unsigned) T, l, hist);
            strata::kernels::SamplerParams p;
            p.top_k = 64; p.top_p = 1.0f; p.temperature = 2.5f; p.seed = 18; p.counter = (uint64_t) T;
            const int k = sampled_k(p.top_k, nv);
            std::vector<SelList> lists;
            for (int t = 0; t < T; ++t) lists.push_back(mirror_select(l.data() + (size_t) t * nv, nv, {}, p, k));
            DeviceRows rows(l, T, hist, 0);
            compare(T > 64 ? "70 rows (one block)" : "64 rows (split)", rows.sample(p, nullptr), lists, k, p);
        }
        std::printf("  %-34s %s (%d of %d draws differ)\n", "fallbacks: graph capture, row cap",
                    wrong ? "*** WRONG ***" : "matches", wrong, draws);
        bad += wrong;
    }

    // ---- GUMBEL-MAX (STRATA_SPEC_GUMBEL, p.gumbel).  (a) The kernel's pick equals the host reference's, draw for draw,
    // under the chains the server uses (top_k / top_p / min_p / temperature).  (b) It is an exact sample: over many
    // counters the picks' frequencies match the softmax of the surviving logits.  (c) It is a different stream from
    // the inverse-CDF draw (the flag is really read), and greedy rows ignore it.
    {
        int wrong = 0, draws = 0;
        const int nv = 512, T = 16;
        std::vector<float> l((size_t) nv * T);
        std::vector<int> hist;
        {
            std::mt19937 rng(4242u);
            std::normal_distribution<float> gd(0.0f, 2.0f);
            for (auto& v : l) v = gd(rng);
        }
        struct Chain { int top_k; float top_p, min_p, temp; };
        for (const Chain c : {Chain{20, 0.95f, 0.0f, 1.0f}, Chain{0, 1.0f, 0.0f, 0.7f}, Chain{40, 0.9f, 0.05f, 1.3f}}) {
            strata::kernels::SamplerParams p;
            p.top_k = c.top_k; p.top_p = c.top_p; p.min_p = c.min_p; p.temperature = c.temp;
            p.seed = 77; p.counter = 1000; p.gumbel = true;
            std::vector<int> want((size_t) T);
            for (int t = 0; t < T; ++t)
                want[(size_t) t] = sampled_reference({l.begin() + (size_t) t * nv, l.begin() + (size_t) (t + 1) * nv}, {}, p, t);
            DeviceRows rows(l, T, hist, 0);
            const std::vector<int> got = rows.sample(p, nullptr);
            for (int t = 0; t < T; ++t) wrong += got[(size_t) t] != want[(size_t) t];
            draws += T;
        }
        std::printf("  %-34s %s (%d of %d draws differ)\n", "gumbel: kernel vs reference", wrong ? "*** WRONG ***" : "matches",
                    wrong, draws);
        bad += wrong;

        // (b) frequencies: one 12-logit row, every row of a launch the same logits at its own counter
        const int nv2 = 12, R = 64, launches = 160;          // 10,240 draws
        std::vector<float> row = {2.0f, 1.5f, 1.2f, 1.0f, 0.5f, 0.3f, 0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f};
        std::vector<float> lr;
        for (int r = 0; r < R; ++r) lr.insert(lr.end(), row.begin(), row.end());
        strata::kernels::SamplerParams g;
        g.top_k = 0; g.top_p = 1.0f; g.temperature = 1.0f; g.seed = 5; g.gumbel = true;
        std::vector<double> freq((size_t) nv2, 0.0);
        DeviceRows rr(lr, R, {}, 0);
        for (int k2 = 0; k2 < launches; ++k2) {
            g.counter = (uint64_t) k2 * R;
            for (int id : rr.sample(g, nullptr)) freq[(size_t) id] += 1.0;
        }
        double z = 0.0, tv = 0.0;
        for (float v : row) z += std::exp((double) v);
        for (int i = 0; i < nv2; ++i) tv += std::fabs(freq[(size_t) i] / (R * launches) - std::exp((double) row[(size_t) i]) / z);
        tv *= 0.5;
        const bool freq_bad = tv > 0.02;                     // 10,240 draws: total variation ~0.01 by chance
        std::printf("  %-34s %s (total variation %.4f over %d draws)\n", "gumbel: frequencies = softmax",
                    freq_bad ? "*** WRONG ***" : "matches", tv, R * launches);
        bad += freq_bad;

        // (c) the flag is read (a different stream than the inverse-CDF draw), and greedy ignores it
        strata::kernels::SamplerParams a; a.top_k = 20; a.top_p = 0.95f; a.temperature = 1.0f; a.seed = 9; a.counter = 3;
        strata::kernels::SamplerParams b = a; b.gumbel = true;
        DeviceRows rows(l, T, hist, 0);
        const std::vector<int> ga = rows.sample(a, nullptr), gb = rows.sample(b, nullptr);
        int same = 0;
        for (int t = 0; t < T; ++t) same += ga[(size_t) t] == gb[(size_t) t];
        strata::kernels::SamplerParams ga2; ga2.greedy = true; ga2.temperature = 0.0f;
        strata::kernels::SamplerParams gb2 = ga2; gb2.gumbel = true;
        const bool greedy_same = rows.sample(ga2, nullptr) == rows.sample(gb2, nullptr);
        const bool flag_bad = same == T || !greedy_same;
        std::printf("  %-34s %s (%d of %d sampled picks equal the inverse-CDF ones; greedy %s)\n", "gumbel: flag read, greedy unchanged",
                    flag_bad ? "*** WRONG ***" : "matches", same, T, greedy_same ? "unchanged" : "CHANGED");
        bad += flag_bad;
    }

    std::printf("\nsampler: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("sampler_parity OK\n");
    return 0;
}
