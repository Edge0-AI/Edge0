// src/kernels/spec_verify_parity.cpp - the GPU verifier of probabilistic draft acceptance (sample_tokens_spec) against
// the host reference (include/strata/core/spec_prob.hpp), token for token.
//
// Random logits at three vocabulary sizes (one block, a few blocks, the engine's 248,320), windows of 2..8 rows,
// chains with top_k / top_p / min_p / temperature and repetition penalties over a per-row history, drafts taken from
// the row's own top tokens, outside them, and from q; q lists cut by a DIFFERENT chain than p's; n_q from 0 (every
// row a point mass) to all rows.  The host computes p the sampler's way (penalties on the raw logits, top_k with
// ties to the lowest index, top_p's cut in double, min_p in logit space, temperature), the Philox draws of the kernel
// (u_pick = pos, u_accept = pos | 1<<62, u_resid = pos | 1<<63), and judges every row with spec_verify_row /
// spec_sample.  Every out[t] must match, accepted or not.  Needs a GPU.
#include "strata/core/spec_prob.hpp"
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using strata::core::SpecP;
using strata::core::SpecQ;
using strata::kernels::SamplerParams;

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

float host_philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        const uint32_t hi0 = (uint32_t) (((uint64_t) 0x9E3779B9u * c0) >> 32);
        const uint32_t hi1 = (uint32_t) (((uint64_t) 0xBB67AE85u * c2) >> 32);
        const uint32_t lo0 = 0x9E3779B9u * c0;
        const uint32_t lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ (uint32_t) i;
        const uint32_t n2 = hi0 ^ c3 ^ 0u;
        c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
    }
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

struct PList {
    std::vector<int32_t> id;
    std::vector<double> p;
    std::vector<float> pf;
};

// the sampler's chain over one row (penalties over `hist`), as sampled_reference in sampler_parity.cpp
PList chain_list(const std::vector<float>& l, const std::vector<int>& hist, const SamplerParams& p) {
    auto penal = [&](float logit, int count) {
        if (count <= 0) return logit;
        if (logit <= 0.0f) logit *= p.penalty_repeat; else logit /= p.penalty_repeat;
        logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
        return logit;
    };
    const int nv = (int) l.size();
    std::vector<float> s((size_t) nv);
    for (int v = 0; v < nv; ++v) s[(size_t) v] = l[(size_t) v];
    if (p.penalty_last_n > 0) {
        std::vector<int> cnt((size_t) nv, 0);
        const int h = (int) hist.size();
        for (int i = std::max(0, h - p.penalty_last_n); i < h; ++i)
            if (hist[(size_t) i] >= 0 && hist[(size_t) i] < nv) ++cnt[(size_t) hist[(size_t) i]];
        for (int v = 0; v < nv; ++v) s[(size_t) v] = penal(s[(size_t) v], cnt[(size_t) v]);
    }
    const int k = std::min(nv, (p.top_k > 0 && p.top_k < 64) ? p.top_k : 64);
    std::vector<int> idx((size_t) nv);
    for (int i = 0; i < nv; ++i) idx[(size_t) i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
        return s[(size_t) a] > s[(size_t) b] || (s[(size_t) a] == s[(size_t) b] && a < b);
    });
    std::vector<float> sl((size_t) k);
    for (int i = 0; i < k; ++i) sl[(size_t) i] = s[(size_t) idx[(size_t) i]];
    int n_keep = k;
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += std::exp((double) sl[(size_t) i] - (double) sl[0]);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += std::exp((double) sl[(size_t) i] - (double) sl[0]) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = sl[0] + std::log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sl[(size_t) i] < thresh) { n_keep = i; break; }
    }
    const float inv_t = 1.0f / p.temperature;
    float smx = sl[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, sl[(size_t) i] * inv_t);
    PList L;
    double sum = 0.0;
    std::vector<double> e((size_t) n_keep);
    for (int i = 0; i < n_keep; ++i) {
        e[(size_t) i] = std::exp((double) (sl[(size_t) i] * inv_t) - (double) smx);
        sum += e[(size_t) i];
    }
    for (int i = 0; i < n_keep; ++i) {
        L.id.push_back(idx[(size_t) i]);
        L.p.push_back(e[(size_t) i] / sum);
        L.pf.push_back((float) (e[(size_t) i] / sum));
    }
    return L;
}

}  // namespace

int main(int argc, char** argv) {
    (void) argc; (void) argv;
    std::mt19937_64 g(777);
    long long rows = 0, accepted = 0, rejected = 0, plain = 0, bad = 0, qrows_total = 0;
    const int vocabs[] = {1000, 50000, 248320};
    const int Ts[] = {2, 3, 5, 8};
    const int kMaxT = 8;
    float* d_logits = nullptr;
    int* d_out = nullptr;
    int32_t* d_hist = nullptr;
    int32_t* d_dtok = nullptr;
    int32_t* d_q = nullptr;
    check(cudaMalloc((void**) &d_logits, (size_t) kMaxT * 248320 * sizeof(float)), "malloc logits");
    check(cudaMalloc((void**) &d_out, kMaxT * sizeof(int)), "malloc out");
    check(cudaMalloc((void**) &d_hist, (size_t) kMaxT * 64 * sizeof(int32_t)), "malloc hist");
    check(cudaMalloc((void**) &d_dtok, kMaxT * sizeof(int32_t)), "malloc dtok");
    check(cudaMalloc((void**) &d_q, (size_t) kMaxT * strata::core::kSpecQStride * sizeof(int32_t)), "malloc q");
    for (int nv : vocabs) {
        for (int T : Ts) {
            for (int rep = 0; rep < 24; ++rep) {
                SamplerParams p;
                const int tk[] = {1, 5, 20, 64, 0};
                const float tp[] = {1.0f, 0.95f, 0.5f};
                const float mp[] = {0.0f, 0.05f, 0.1f};
                const float tt[] = {0.3f, 0.7f, 1.0f, 1.5f};
                p.top_k = tk[g() % 5];
                p.top_p = tp[g() % 3];
                p.min_p = mp[g() % 3];
                p.temperature = tt[g() % 4];
                p.seed = g();
                p.counter = g() % 100000;
                const bool pen = rep % 3 == 2;
                if (pen) { p.penalty_last_n = 32; p.penalty_repeat = 1.2f; p.penalty_freq = 0.1f; p.penalty_present = 0.2f; }
                std::normal_distribution<float> nd(0.0f, 2.5f);
                std::vector<std::vector<float>> logits((size_t) T, std::vector<float>((size_t) nv));
                std::vector<float> flat((size_t) T * nv);
                for (int t = 0; t < T; ++t) {
                    for (int v = 0; v < nv; ++v) logits[(size_t) t][(size_t) v] = nd(g);
                    // a few clustered peaks, so top_p / min_p cut at interesting places
                    for (int j = 0; j < 6; ++j) logits[(size_t) t][g() % nv] += 4.0f + (float) (g() % 100) / 25.0f;
                    std::copy(logits[(size_t) t].begin(), logits[(size_t) t].end(), flat.begin() + (size_t) t * nv);
                }
                std::vector<int32_t> hist((size_t) T * 64, -1);
                std::vector<std::vector<int>> rowhist((size_t) T);
                if (pen)
                    for (int t = 0; t < T; ++t) {
                        for (int i = 0; i < 40; ++i) hist[(size_t) t * 64 + 24 + (size_t) i] = -1;
                        rowhist[(size_t) t].clear();
                        for (int i = 0; i < 32; ++i) {
                            const int v = (int) (g() % 200);
                            hist[(size_t) t * 64 + 32 + (size_t) i] = v;   // the last 32 slots
                            rowhist[(size_t) t].push_back(v);
                        }
                    }
                // host: p lists, q lists, drafts
                std::vector<PList> P((size_t) T), Q((size_t) T);
                const int n_q = (int) (g() % (T));   // 0 .. T-1 rows with a list
                std::vector<int32_t> dtok((size_t) (T - 1), 0);
                std::vector<int32_t> qbuf((size_t) T * strata::core::kSpecQStride, -1);
                for (int t = 0; t < T; ++t) P[(size_t) t] = chain_list(logits[(size_t) t], rowhist[(size_t) t], p);
                for (int t = 0; t < T - 1; ++t) {
                    if (t < n_q) {
                        // the drafter's own chain: a perturbed copy, its own top_k / top_p / temperature, no penalties
                        std::vector<float> lq = logits[(size_t) t];
                        for (float& x : lq) x += 0.7f * nd(g) / 2.5f;
                        SamplerParams q;
                        q.top_k = tk[g() % 5]; q.top_p = tp[g() % 3]; q.min_p = 0.0f; q.temperature = tt[g() % 4];
                        Q[(size_t) t] = chain_list(lq, {}, q);
                        int32_t* row = qbuf.data() + (size_t) t * strata::core::kSpecQStride;
                        float* prow = (float*) (row + strata::core::kSpecQEntries);
                        for (size_t i = 0; i < Q[(size_t) t].id.size(); ++i) { row[i] = Q[(size_t) t].id[i]; prow[i] = Q[(size_t) t].pf[i]; }
                        // the draft: drawn from q
                        const float u = (float) (g() >> 40) * (1.0f / 16777216.0f);
                        double cum = 0.0;
                        int32_t d = Q[(size_t) t].id.back();
                        for (size_t i = 0; i < Q[(size_t) t].id.size(); ++i) { cum += Q[(size_t) t].p[i]; if ((double) u < cum) { d = Q[(size_t) t].id[i]; break; } }
                        dtok[(size_t) t] = d;
                    } else {
                        const int r = (int) (g() % 10);
                        const PList& L = P[(size_t) t];
                        dtok[(size_t) t] = r < 6 ? L.id[g() % std::min<size_t>(L.id.size(), 5)] : (int32_t) (g() % nv);
                    }
                }
                check(cudaMemcpy(d_logits, flat.data(), flat.size() * sizeof(float), cudaMemcpyHostToDevice), "up logits");
                check(cudaMemcpy(d_hist, hist.data(), hist.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "up hist");
                check(cudaMemcpy(d_dtok, dtok.data(), dtok.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "up dtok");
                check(cudaMemcpy(d_q, qbuf.data(), qbuf.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "up q");
                const bool ran = strata::kernels::sample_tokens_spec(d_logits, T, nv, pen ? d_hist : nullptr, pen ? 64 : 0, p,
                                                                     d_dtok, d_q, n_q, d_out, nullptr);
                if (!ran) { std::printf("sample_tokens_spec did not run (nv %d T %d)\n", nv, T); return 1; }
                check(cudaDeviceSynchronize(), "sync");
                int out[kMaxT];
                check(cudaMemcpy(out, d_out, (size_t) T * sizeof(int), cudaMemcpyDeviceToHost), "down");
                for (int t = 0; t < T; ++t) {
                    const uint64_t pos = p.counter + (uint64_t) t;
                    const SpecP sp{P[(size_t) t].id.data(), P[(size_t) t].p.data(), (int) P[(size_t) t].id.size()};
                    int32_t want;
                    if (t == T - 1) {
                        want = strata::core::spec_sample(sp, host_philox_uniform(p.seed, pos));
                        ++plain;
                    } else {
                        SpecQ sq{nullptr, nullptr, -1};
                        if (t < n_q) sq = SpecQ{Q[(size_t) t].id.data(), Q[(size_t) t].pf.data(), (int) Q[(size_t) t].id.size()};
                        const auto r = strata::core::spec_verify_row(sp, dtok[(size_t) t], sq,
                                                                     host_philox_uniform(p.seed, pos | strata::core::kSpecCtrAccept),
                                                                     host_philox_uniform(p.seed, pos | strata::core::kSpecCtrResid));
                        want = r.token;
                        (r.accepted ? accepted : rejected) += 1;
                        qrows_total += t < n_q;
                    }
                    ++rows;
                    if (out[t] != want) {
                        if (++bad <= 10)
                            std::printf("MISMATCH nv %d T %d row %d: kernel %d host %d (draft %d, n_q %d, top_k %d top_p %.2f min_p %.2f temp %.1f%s)\n",
                                        nv, T, t, out[t], want, t < T - 1 ? dtok[(size_t) t] : -1, n_q, p.top_k, p.top_p, p.min_p,
                                        p.temperature, pen ? " penalties" : "");
                    }
                }
            }
        }
    }
    std::printf("spec_verify_parity: %lld rows (%lld drafts kept, %lld rejected, %lld plain last rows, %lld with a q list): %lld mismatches\n",
                rows, accepted, rejected, plain, qrows_total, bad);
    cudaFree(d_logits); cudaFree(d_out); cudaFree(d_hist); cudaFree(d_dtok); cudaFree(d_q);
    return bad == 0 ? 0 : 1;
}
