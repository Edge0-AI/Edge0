// GDN prompt stage bench (production-linked): one layer's gates + conv + recurrence/out through the public entry points,
// timed per piece, at the prompt lengths of the 4K / 16K / 32K tables.  Unlike gdn_chunked --bench it sets NO environment:
// the STRATA_GDN_* knobs (and the arch defaults a real run would apply, passed by the caller) are read as they are, so one
// process measures one knob set.  It prints a hash of the output rows and the final state, so an "exact" knob can be
// checked bit for bit against the default run, and a split-call hash (the prompt cut at an unaligned token, state carried)
// for the continuation gate.
//
//   gdn_stage_bench [T ...]        default T = 4096 16384 32768
//   env GDN_BENCH_REPS=N           timed repeats (median), default 7
//   env GDN_BENCH_SPLIT=0          skip the split-call hash
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {
constexpr int S = 128, HK = 16, HV = 48, C = 10240, ROW = HV * S;
void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(e)); std::exit(2); }
}
template<class X> X* up(const std::vector<X>& v) {
    X* p = nullptr;
    ck(cudaMalloc(&p, v.size() * sizeof(X)), "malloc");
    ck(cudaMemcpy(p, v.data(), v.size() * sizeof(X), cudaMemcpyHostToDevice), "upload");
    return p;
}
template<class X> std::vector<X> down(const X* p, size_t n) {
    std::vector<X> v(n);
    ck(cudaMemcpy(v.data(), p, n * sizeof(X), cudaMemcpyDeviceToHost), "download");
    return v;
}
uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
struct Rng {
    std::mt19937 g{7};
    std::normal_distribution<float> n{0, 1};
    std::uniform_real_distribution<float> u{0, 1};
};
}

int main(int argc, char** argv) {
    std::vector<int64_t> lens;
    for (int i = 1; i < argc; ++i) lens.push_back(std::atoll(argv[i]));
    if (lens.empty()) lens = {4096, 16384, 32768};
    const char* rp = std::getenv("GDN_BENCH_REPS");
    const int reps = rp ? std::max(1, std::atoi(rp)) : 7;
    const bool do_split = !(std::getenv("GDN_BENCH_SPLIT") && std::atoi(std::getenv("GDN_BENCH_SPLIT")) == 0);
    int dev = 0;
    cudaDeviceProp props{};
    ck(cudaGetDevice(&dev), "device"); ck(cudaGetDeviceProperties(&props, dev), "props");
    std::printf("device=%s CUs=%d\n", props.gcnArchName, props.multiProcessorCount);
    for (const char* k : {"STRATA_GDN_CHUNKED", "STRATA_GDN_HEAD", "STRATA_GDN_PP", "STRATA_GDN_CONVL2", "STRATA_GDN_NOY",
                          "STRATA_GDN_CONV_SERIAL", "STRATA_GDN_REC_HEADS"})
        std::printf("env %s=%s\n", k, std::getenv(k) ? std::getenv(k) : "(unset)");
    const int64_t Tmax = *std::max_element(lens.begin(), lens.end());
    Rng r;
    std::vector<float> qkv(Tmax * C), ab(Tmax * 2 * HV), dt(HV), ssm_a(HV), cw(C * 4), hist0(C * 3), z(Tmax * ROW), gamma(S),
        state0(S * ROW);
    for (auto& x : qkv) x = r.n(r.g);
    for (int64_t t = 0; t < Tmax; ++t)
        for (int j = 0; j < HV; ++j) { ab[t * 2 * HV + j] = r.n(r.g) - 1; ab[t * 2 * HV + HV + j] = r.n(r.g); }
    for (auto& x : dt) x = r.n(r.g);
    for (auto& x : ssm_a) x = -(0.5f + r.u(r.g));
    for (auto& x : cw) x = 0.4f * r.n(r.g);
    for (auto& x : hist0) x = r.n(r.g);
    for (auto& x : z) x = r.n(r.g);
    for (auto& x : gamma) x = 0.5f + r.u(r.g);
    for (auto& x : state0) x = 0.05f * r.n(r.g);
    auto dqkv = up(qkv), dab = up(ab), ddt = up(dt), dssm = up(ssm_a), dcw = up(cw), dz = up(z), dgamma = up(gamma);
    std::vector<float>().swap(qkv); std::vector<float>().swap(z);
    float *dgate = nullptr, *dbeta = nullptr, *dh = nullptr, *dhist = nullptr, *dstate = nullptr, *dy = nullptr;
    uint16_t* dy16 = nullptr;
    const int64_t ld = ROW + 8;
    ck(cudaMalloc(&dgate, Tmax * HV * 4), "m"); ck(cudaMalloc(&dbeta, Tmax * HV * 4), "m");
    ck(cudaMalloc(&dh, Tmax * C * 4), "m"); ck(cudaMalloc(&dhist, C * 3 * 4), "m");
    ck(cudaMalloc(&dstate, S * ROW * 4), "m"); ck(cudaMalloc(&dy, Tmax * ROW * 4), "m");
    ck(cudaMalloc(&dy16, (Tmax * ld + 8) * 2), "m");
    cudaEvent_t e0, e1, e2;
    ck(cudaEventCreate(&e0), "ev"); ck(cudaEventCreate(&e1), "ev"); ck(cudaEventCreate(&e2), "ev");
    auto reset = [&] {
        ck(cudaMemcpy(dhist, hist0.data(), hist0.size() * 4, cudaMemcpyHostToDevice), "reset");
        ck(cudaMemcpy(dstate, state0.data(), state0.size() * 4, cudaMemcpyHostToDevice), "reset");
    };
    auto layer = [&](int64_t t0, int64_t n) {   // the prompt-path order: gates, conv, recurrence (+ norm)
        strata::prefill::gdn_gates(dab + t0 * 2 * HV, ddt, dssm, dgate + t0 * HV, dbeta + t0 * HV, n, nullptr);
        if (strata::prefill::gdn_conv_recurrence_fused(dhist, dqkv + t0 * C, dcw, dh, dstate, dgate + t0 * HV, dbeta + t0 * HV,
                                                       dz + t0 * ROW, dgamma, 1e-6f, 1e-6f, dy, dy16 + t0 * ld, n, nullptr, ld))
            return;
        strata::prefill::gdn_conv(dhist, dqkv + t0 * C, dcw, dh + t0 * C, n, 1e-6f, nullptr);
        strata::prefill::gdn_recurrence(dstate, dh + t0 * C, dgate + t0 * HV, dbeta + t0 * HV, dz + t0 * ROW, dgamma, 1e-6f,
                                        dy, dy16 + t0 * ld, n, nullptr, ld);
    };
    if (std::getenv("GDN_BENCH_CHECK")) {   // fused / unfused chunked path against the FP32 sequential kernels, and a split-call check
        for (int64_t T : lens) {
            auto half = [](uint16_t b) { __half_raw x; x.x = b; return __half2float(__half(x)); };
            struct R { std::vector<uint16_t> y; std::vector<float> st, hist; };
            auto grab = [&] { return R{down(dy16, T * ld), down(dstate, S * ROW), down(dhist, C * 3)}; };
            auto prep = [&] { reset(); strata::prefill::gdn_gates(dab, ddt, dssm, dgate, dbeta, T, nullptr); };
            prep();
            strata::prefill::gdn_conv(dhist, dqkv, dcw, dh, T, 1e-6f, nullptr);
            strata::prefill::gdn_recurrence_variant(5, dstate, dh, dgate, dbeta, dz, dgamma, 1e-6f, dy, dy16, T, nullptr, ld);
            ck(cudaDeviceSynchronize(), "sync");
            const R ref = grab();
            prep();
            strata::prefill::gdn_conv(dhist, dqkv, dcw, dh, T, 1e-6f, nullptr);
            strata::prefill::gdn_recurrence(dstate, dh, dgate, dbeta, dz, dgamma, 1e-6f, dy, dy16, T, nullptr, ld);
            ck(cudaDeviceSynchronize(), "sync");
            const R unf = grab();
            prep();
            const bool f = strata::prefill::gdn_conv_recurrence_fused(dhist, dqkv, dcw, dh, dstate, dgate, dbeta, dz, dgamma, 1e-6f,
                                                                      1e-6f, dy, dy16, T, nullptr, ld);
            ck(cudaDeviceSynchronize(), "sync");
            const R fus = grab();
            auto err = [&](const R& a, const R& b, const char* what) {
                double sq = 0, rf = 0, mx = 0, mr = 0, ssq = 0, srf = 0;
                for (int64_t t = 0; t < T; ++t)
                    for (int d = 0; d < ROW; ++d) {
                        const double x = half(a.y[t * ld + d]), y = half(b.y[t * ld + d]);
                        sq += (x - y) * (x - y); rf += x * x; mx = std::max(mx, std::fabs(x - y)); mr = std::max(mr, std::fabs(x));
                    }
                for (size_t i = 0; i < a.st.size(); ++i) { const double x = a.st[i], y = b.st[i]; ssq += (x - y) * (x - y); srf += x * x; }
                double hm = 0;
                for (size_t i = 0; i < a.hist.size(); ++i) hm = std::max(hm, (double) std::fabs(a.hist[i] - b.hist[i]));
                std::printf("CHECK T=%lld %s: y_rms=%.3e y_max=%.3e (ref max %.3e) state_rms=%.3e hist_maxdiff=%.1e\n", (long long) T, what,
                            std::sqrt(sq / std::max(rf, 1e-30)), mx, mr, std::sqrt(ssq / std::max(srf, 1e-30)), hm);
            };
            err(ref, unf, "unfused-vs-fp32");
            if (f) err(ref, fus, "fused-vs-fp32"); else std::printf("CHECK T=%lld fused path not taken\n", (long long) T);
            if (f) err(unf, fus, "fused-vs-unfused");
            if (f) {   // continuation: the fused path in two calls (odd cut) against one call
                reset();
                const int64_t cut = (T / 2) | 1;
                auto call = [&](int64_t t0, int64_t n) {
                    strata::prefill::gdn_gates(dab + t0 * 2 * HV, ddt, dssm, dgate + t0 * HV, dbeta + t0 * HV, n, nullptr);
                    strata::prefill::gdn_conv_recurrence_fused(dhist, dqkv + t0 * C, dcw, dh, dstate, dgate + t0 * HV, dbeta + t0 * HV,
                                                               dz + t0 * ROW, dgamma, 1e-6f, 1e-6f, dy, dy16 + t0 * ld, n, nullptr, ld);
                };
                call(0, cut); call(cut, T - cut);
                ck(cudaDeviceSynchronize(), "sync");
                err(fus, grab(), "fused split-vs-whole");
            }
        }
        return 0;
    }
    for (int64_t T : lens) {
        std::vector<float> conv_ms, rec_ms;
        for (int round = -2; round < reps; ++round) {
            reset();
            ck(cudaEventRecord(e0), "ev");
            strata::prefill::gdn_gates(dab, ddt, dssm, dgate, dbeta, T, nullptr);
            const bool fused = strata::prefill::gdn_conv_recurrence_fused(dhist, dqkv, dcw, dh, dstate, dgate, dbeta, dz, dgamma,
                                                                          1e-6f, 1e-6f, dy, dy16, T, nullptr, ld);
            if (!fused) strata::prefill::gdn_conv(dhist, dqkv, dcw, dh, T, 1e-6f, nullptr);
            ck(cudaEventRecord(e1), "ev");   // fused: conv and recurrence are both inside the FIRST piece
            if (!fused) strata::prefill::gdn_recurrence(dstate, dh, dgate, dbeta, dz, dgamma, 1e-6f, dy, dy16, T, nullptr, ld);
            ck(cudaEventRecord(e2), "ev");
            ck(cudaEventSynchronize(e2), "sync");
            float a = 0, b = 0;
            ck(cudaEventElapsedTime(&a, e0, e1), "t"); ck(cudaEventElapsedTime(&b, e1, e2), "t");
            if (round >= 0) { conv_ms.push_back(a); rec_ms.push_back(b); }
        }
        std::sort(conv_ms.begin(), conv_ms.end()); std::sort(rec_ms.begin(), rec_ms.end());
        const double cm = conv_ms[conv_ms.size() / 2], rm = rec_ms[rec_ms.size() / 2];
        // hashes of one clean whole-prompt run (the state is the one-call result)
        reset(); layer(0, T); ck(cudaDeviceSynchronize(), "sync");
        const auto y16 = down(dy16, T * ld); const auto st = down(dstate, S * ROW);
        uint64_t hy = 1469598103934665603ull;
        for (int64_t t = 0; t < T; ++t) hy = fnv(y16.data() + t * ld, ROW * 2, hy);   // padding columns excluded
        const uint64_t hs = fnv(st.data(), st.size() * 4);
        bool same = true;
        if (do_split) {
            const int64_t cut = (T / 2) | 1;   // unaligned (odd) split point
            reset(); layer(0, cut); layer(cut, T - cut); ck(cudaDeviceSynchronize(), "sync");
            const auto y2 = down(dy16, T * ld); const auto s2 = down(dstate, S * ROW);
            uint64_t hy2 = 1469598103934665603ull;
            for (int64_t t = 0; t < T; ++t) hy2 = fnv(y2.data() + t * ld, ROW * 2, hy2);
            same = hy == hy2 && hs == fnv(s2.data(), s2.size() * 4);
        }
        std::printf("STAGE T=%lld gates_conv_ms=%.4f rec_out_ms=%.4f sum_ms=%.4f y_hash=%016llx state_hash=%016llx split_same=%d\n",
                    (long long) T, cm, rm, cm + rm, (unsigned long long) hy, (unsigned long long) hs, same);
    }
    return 0;
}
