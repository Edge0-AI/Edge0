// src/kernels/cpu/kq_cpu_bench.cpp - MODEL-UPDATE: how close the CPU pool's expert rows come to the RAM bandwidth, per
// expert format.  No GPU, no model: --mb MB of expert blobs (ggml-quantized random weights, far more than any L3) are
// computed by T threads at once through the engine's own entry points (native_gu_rows + native_down_rows, the pool's
// calls), each thread walking its own experts, nt tokens per expert - the decode case, where every CPU expert is a
// fresh read from DRAM.  Prints the weight bytes per second against the --ram-gbs given (a separate read test).
//
//     kq_cpu_bench [--pairs q4_K/q5_1,q5_K/q8_0,q6_K/q8_0,q8_0/q8_0,iq3_s/iq4_nl] [--threads 1,8,16,32] [--nt 1,2,4]
//                  [--mb 4096] [--reps 3] [--ram-gbs N]
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

constexpr int64_t kH = 2560, kFF = 640;
constexpr int kMaxT = 8;

int type_of(const std::string& s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name((ggml_type) t);
        if (n != nullptr && s == n) return t;
    }
    return -1;
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v;
    size_t from = 0;
    for (;;) {
        const size_t c = s.find(',', from);
        v.push_back(s.substr(from, c == std::string::npos ? std::string::npos : c - from));
        if (c == std::string::npos) break;
        from = c + 1;
    }
    return v;
}

// one expert blob in the GGUF form: gate rows, up rows (n_ff x n_embd each), down rows (n_embd x n_ff), from random
// weights quantized by ggml (valid scales, as a real file has).  ggml_quantize_chunk, not the CPU traits' from_float:
// ggml-cpu has no from_float for the i-quants (IQ3_S called a null pointer: an access violation on Windows)
void make_blob(const cpu::NativeFmt& f, uint8_t* blob, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> row((size_t) std::max(kH, kFF));
    const std::vector<float> imatrix((size_t) std::max(kH, kFF), 1.0f);
    auto quant = [&](int type, uint8_t* dst, int64_t n) {
        ggml_quantize_chunk((ggml_type) type, row.data(), dst, 0, 1, n,
                            ggml_quantize_requires_imatrix((ggml_type) type) ? imatrix.data() : nullptr);
    };
    for (int64_t r = 0; r < 2 * kFF; ++r) {
        for (int64_t i = 0; i < kH; ++i) row[(size_t) i] = nd(rng);
        quant(f.gu_type, blob + (size_t) r * f.gu_row, kH);
    }
    for (int64_t r = 0; r < kH; ++r) {
        for (int64_t i = 0; i < kFF; ++i) row[(size_t) i] = nd(rng);
        quant(f.d_type, blob + f.down_off + (size_t) r * f.d_row, kFF);
    }
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> pairs = split("q4_K/q5_1,q5_K/q8_0,q6_K/q8_0,q8_0/q8_0,iq3_s/iq4_nl");
    std::vector<int> threads = {1, 8, 16, 32}, nts = {1, 2, 4};
    int mb = 4096, reps = 3;
    double ram = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        auto ints = [](const std::string& s) { std::vector<int> v; for (auto& x : split(s)) v.push_back(std::atoi(x.c_str())); return v; };
        if (a == "--pairs" && more) pairs = split(argv[++i]);
        else if (a == "--threads" && more) threads = ints(argv[++i]);
        else if (a == "--nt" && more) nts = ints(argv[++i]);
        else if (a == "--mb" && more) mb = std::atoi(argv[++i]);
        else if (a == "--reps" && more) reps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--ram-gbs" && more) ram = std::atof(argv[++i]);
        else {
            std::fprintf(stderr, "usage: kq_cpu_bench [--pairs a/b,...] [--threads 1,8] [--nt 1,2,4] [--mb 4096] [--reps 3] [--ram-gbs N]\n");
            return 2;
        }
    }
    ggml_cpu_init();
    std::printf("kq_cpu_bench: %s (avx2 %d, avx512 %d), %d MB of blobs per format\n", cpu::cpu_name().c_str(),
                (int) cpu::cpu_avx2_ok(), (int) cpu::cpu_avx512_ok(), mb);
    for (const std::string& pair : pairs) {
        const size_t slash = pair.find('/');
        const int gt = slash == std::string::npos ? -1 : type_of(pair.substr(0, slash));
        const int dt = slash == std::string::npos ? -1 : type_of(pair.substr(slash + 1));
        cpu::NativeFmt f;
        std::string err;
        if (gt < 0 || dt < 0 || !cpu::native_fmt(gt, dt, kH, kFF, f, err)) {
            std::printf("%s: %s\n", pair.c_str(), err.empty() ? "unknown types" : err.c_str());
            continue;
        }
        ggml_quantize_init((ggml_type) gt);   // the i-quants' quantizers need their grids first
        ggml_quantize_init((ggml_type) dt);
        const size_t E = std::max<size_t>(64, ((size_t) mb << 20) / f.bytes);
        std::unique_ptr<uint8_t[]> mem(new uint8_t[E * f.bytes + 4096]);
        uint8_t* blobs = (uint8_t*) (((uintptr_t) mem.get() + 4095) & ~(uintptr_t) 4095);
        {   // 16 distinct experts, copied over the whole buffer (the bytes in DRAM are what is measured)
            std::mt19937 rng(1234 + gt * 64 + dt);
            const size_t U = std::min<size_t>(16, E);
            for (size_t e = 0; e < U; ++e) make_blob(f, blobs + e * f.bytes, rng);
            for (size_t e = U; e < E; ++e) std::memcpy(blobs + e * f.bytes, blobs + (e % U) * f.bytes, f.bytes);
        }
        // activations: kMaxT tokens of x (n_embd) and h (n_ff), quantized the way the pool quantizes them
        std::mt19937 rng(77);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<std::vector<uint8_t>> xa(kMaxT, std::vector<uint8_t>(cpu::kNativeActBytes)),
            ha(kMaxT, std::vector<uint8_t>(cpu::kNativeHBytes));
        const void* act[kMaxT];
        const void* hq[kMaxT];
        // a Q2_0 down takes the pool's own path (pool.cpp, run_split_multi_native and drain mode 6): the activation
        // as an ActQ (act_quant_any) and Strata's Q2_0 rows (q2_rows_any), not ggml-cpu's Q2_0 dot
        const bool q2_down = cpu::q2_native_kernels(f.d_type);
        std::vector<cpu::ActQ> h2(q2_down ? kMaxT : 0);
        const cpu::ActQ* a2[kMaxT] = {};
        // a Q2_0 gate/up the same (pool.cpp mode 5): x as an ActQ, gate and up on q2_rows_any, then SwiGLU
        const bool q2_gu = cpu::q2_native_kernels(f.gu_type);
        std::vector<cpu::ActQ> x2(q2_gu ? kMaxT : 0);
        const cpu::ActQ* ax[kMaxT] = {};
        for (int t = 0; t < kMaxT; ++t) {
            std::vector<float> x(kH), h(kFF);
            for (auto& v : x) v = nd(rng);
            for (auto& v : h) v = 0.1f * nd(rng);
            if (q2_gu) {
                cpu::act_quant_any(x.data(), (int) kH, x2[(size_t) t]);
                ax[t] = &x2[(size_t) t];
            } else {
                cpu::native_quant_act(f, x.data(), xa[t].data());
            }
            if (q2_down) {
                cpu::act_quant_any(h.data(), (int) kFF, h2[(size_t) t]);
                a2[t] = &h2[(size_t) t];
            } else {
                cpu::native_quant_h(f, h.data(), ha[t].data());
            }
            act[t] = xa[t].data();
            hq[t] = ha[t].data();
        }
        std::printf("\n%s: %zu experts of %.3f MB (%.0f MB)%s%s\n", pair.c_str(), E, f.bytes / 1e6, E * (double) f.bytes / 1e6,
                    q2_gu ? ", Q2_0 gate/up" : "", q2_down ? ", Q2_0 down on Strata's rows (the pool's path)" : "");
        for (int nt : nts) {
            for (int T : threads) {
                if (T < 1) continue;
                std::vector<double> gbs;
                for (int rep = 0; rep < reps; ++rep) {
                    std::atomic<size_t> next{0};
                    auto work = [&] {
                        std::vector<float> ffb((size_t) kMaxT * kFF), outb((size_t) kMaxT * kH);
                        std::vector<float> gb(q2_gu ? (size_t) kMaxT * kFF : 0), ub(q2_gu ? (size_t) kMaxT * kFF : 0);
                        float* gp[kMaxT] = {};
                        float* up[kMaxT] = {};
                        if (q2_gu)
                            for (int t = 0; t < kMaxT; ++t) { gp[t] = &gb[(size_t) t * kFF]; up[t] = &ub[(size_t) t * kFF]; }
                        float* ff[kMaxT];
                        float* out[kMaxT];
                        for (int t = 0; t < kMaxT; ++t) { ff[t] = &ffb[(size_t) t * kFF]; out[t] = &outb[(size_t) t * kH]; }
                        std::vector<std::vector<uint8_t>> hb(kMaxT, std::vector<uint8_t>(cpu::kNativeHBytes));
                        for (;;) {
                            const size_t e = next.fetch_add(1);
                            if (e >= E) break;
                            const uint8_t* blob = blobs + e * f.bytes;
                            if (q2_gu) {
                                cpu::q2_rows_any(blob, f.gu_row, (int) (kH / 64), ax, nt, gp, 0, (int) kFF);
                                cpu::q2_rows_any(blob + f.up_off, f.gu_row, (int) (kH / 64), ax, nt, up, 0, (int) kFF);
                                for (int t = 0; t < nt; ++t)
                                    for (int64_t r = 0; r < kFF; ++r)
                                        ff[t][r] = (gp[t][r] / (1.f + std::exp(-gp[t][r]))) * up[t][r];
                            } else {
                                cpu::native_gu_rows(f, blob, act, nt, ff, 0, (int) kFF);
                            }
                            if (q2_down)
                                cpu::q2_rows_any(blob + f.down_off, f.d_row, (int) (f.n_ff / 64), a2, nt, out, 0, (int) kH);
                            else
                                cpu::native_down_rows(f, blob, hq, nt, out, 0, (int) kH);
                        }
                        volatile float sink = outb[0] + ffb[0];
                        (void) sink;
                    };
                    const auto t0 = std::chrono::steady_clock::now();
                    std::vector<std::thread> th;
                    for (int k = 0; k < T; ++k) th.emplace_back(work);
                    for (auto& x : th) x.join();
                    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    gbs.push_back(E * (double) f.bytes / s / 1e9);
                }
                std::sort(gbs.begin(), gbs.end());
                const double med = gbs[gbs.size() / 2];
                std::printf("  nt %d threads %2d: %6.1f GB/s of weights (%.3f ms per expert per thread)%s\n", nt, T, med,
                            f.bytes / (med * 1e9 / T) * 1e3,
                            ram > 0 ? (" = " + std::to_string((int) std::lround(100 * med / ram)) + "% of RAM").c_str() : "");
            }
        }
    }
    return 0;
}
