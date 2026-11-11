// src/kernels/cpu/tq_cpu_parity.cpp - the CPU trellis kernels (tq_cpu.hpp: TQ2_T / TQK6 / TQK7) against their scalar
// reference, against ggml-cpu's own vec_dot and to_float, and their speed.  CPU only (no GPU runtime needed).
//
//   tq_cpu_parity                                  every check below for tqk6/tqk7 and tq2_t/tq2_t, plain and Hadamard
//   tq_cpu_parity --synthetic GU/DOWN              one expert pair (types by ggml name, e.g. tqk6/tqk7)
//   tq_cpu_parity --synthetic-hadamard GU/DOWN     the same with the block-128 Hadamard rotation of the activations
//   tq_cpu_parity --bench [--quick] [--only K]     the micro-benchmark (run it pinned: taskset -c <cores>);
//                                                  --only: the kernels whose name contains K
//
// Checks, per matrix (random stream bytes and scales, so every state is reached, and ggml-quantized Gaussian rows):
//   (1) AVX2 and AVX-512 kernels == the scalar kernel, bit for bit, for groups of 1..8 tokens
//   (2) a token's row in a group of k == the same token alone, bit for bit (no group-size dependence)
//   (3) vs ggml-cpu's vec_dot on the same Q8_0 activations: the same integers summed in another float order, so
//       ~1e-7 relative
//   (4) vs ggml's to_float weights . the dequantized Q8_0 activations in double: the int16 state table's rounding
//       (exact for |codebook| >= 2^-3, within 2^-14 below) plus float rounding, ~1e-6 relative
// and per expert (native_gu_rows -> native_quant_h -> native_down_rows, the engine's entry points, which now take
// these kernels) against the same expert computed with ggml-cpu's vec_dot: float rounding plus the odd Q8_0
// rounding flip of the intermediate.
#include "strata/kernels/cpu/tq_cpu.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace strata::kernels::cpu;

namespace {

constexpr int H = 2560, FF = 640;

int type_by_name(const std::string& s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name((ggml_type) t);
        if (n != nullptr && s == n) return t;
    }
    return -1;
}

uint16_t f2h(float f) { return ggml_fp32_to_fp16(f); }

// rows x n weights of `type`: random stream bytes and scales (every state reachable, both scale signs), and the
// last rows (up to 32, ggml's trellis encoder is slow) ggml's own encoding of Gaussian rows
std::vector<uint8_t> make_matrix(int type, int rows, int n, uint32_t seed) {
    const size_t rb = ggml_row_size((ggml_type) type, n);
    std::vector<uint8_t> m(rb * rows);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.05f);
    const size_t bsz = rb / (size_t) (n / 128);
    const int half = rows - std::min(rows / 2, 32);
    for (int r = 0; r < half; ++r)
        for (int b = 0; b < n / 128; ++b) {
            uint8_t* blk = m.data() + (size_t) r * rb + (size_t) b * bsz;
            const uint16_t d = f2h((byte(rng) & 1 ? -1.f : 1.f) * scale(rng));
            std::memcpy(blk, &d, 2);
            for (size_t j = 2; j < bsz; ++j) blk[j] = (uint8_t) byte(rng);
        }
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> wf((size_t) (rows - half) * n);
    for (auto& v : wf) v = nd(rng);
    ggml_quantize_chunk((ggml_type) type, wf.data(), m.data() + (size_t) half * rb, 0, rows - half, n, nullptr);
    return m;
}

std::vector<float> make_x(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(n);
    for (auto& v : x) v = nd(rng);
    x[(size_t) (seed % (uint32_t) n)] = 40.f;   // a massive channel, as the residual stream has
    return x;
}

std::vector<uint8_t> q8_0(const std::vector<float>& x) {
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q8_0, (int64_t) x.size()));
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(x.data(), q.data(), (int64_t) x.size());
    return q;
}

bool same_bits(const float* a, const float* b, size_t n) { return std::memcmp(a, b, n * sizeof(float)) == 0; }

// one matrix: checks (1) - (4); returns failures
int check_matrix(int type, int rows, int n, uint32_t seed, bool hadamard) {
    int fails = 0;
    const size_t rb = ggml_row_size((ggml_type) type, n);
    const auto m = make_matrix(type, rows, n, seed);
    constexpr int NT = kTqMaxTokens;
    std::vector<std::vector<uint8_t>> act(NT);
    std::vector<std::vector<float>> xr(NT);
    std::vector<float> signs(n);
    for (int i = 0; i < n; ++i) signs[i] = ((i * 2654435761u) >> 7) & 1 ? -1.f : 1.f;
    const void* ap[NT];
    for (int t = 0; t < NT; ++t) {
        xr[t] = make_x(n, seed * 31 + t);
        if (hadamard) hadamard_rotate(xr[t].data(), n, 128, signs.data());
        act[t] = q8_0(xr[t]);
        ap[t] = act[t].data();
    }
    auto run = [&](TqIsa isa, int nt, const void* const* a, std::vector<float>& o) {
        o.assign((size_t) nt * rows, 0.f);
        float* op[NT];
        for (int t = 0; t < nt; ++t) op[t] = o.data() + (size_t) t * rows;
        // two row ranges, as the pool splits them
        tq_rows(isa, type, m.data(), rb, n, a, nt, op, 0, rows / 3);
        tq_rows(isa, type, m.data(), rb, n, a, nt, op, rows / 3, rows);
    };
    std::vector<float> ref, got;
    run(TqIsa::scalar, NT, ap, ref);
    // (1) every ISA this CPU has, every group size, == scalar bit for bit; (2) group-size independence
    for (TqIsa isa : {TqIsa::scalar, TqIsa::avx2, TqIsa::avx512}) {
        if (!tq_isa_ok(isa)) { std::printf("    %-6s not on this CPU: skipped\n", tq_isa_name(isa)); continue; }
        bool ok = true;
        for (int nt = 1; nt <= NT; ++nt)
            for (int t0 = 0; t0 + nt <= NT; t0 += nt) {
                run(isa, nt, ap + t0, got);
                ok = ok && same_bits(got.data(), ref.data() + (size_t) t0 * rows, (size_t) nt * rows);
            }
        std::printf("    %-6s groups of 1..%d tokens vs the scalar kernel's 8-token run: %s\n", tq_isa_name(isa), NT,
                    ok ? "bitwise equal" : "DIFFERENT");
        fails += !ok;
    }
    // (3) ggml-cpu's vec_dot, (4) to_float in double
    const auto* tc = ggml_get_type_traits_cpu((ggml_type) type);
    const auto* tt = ggml_get_type_traits((ggml_type) type);
    const auto* q8t = ggml_get_type_traits(GGML_TYPE_Q8_0);
    double num3 = 0, num4 = 0, den = 0, max3 = 0;
    std::vector<float> wf(n), af(n);
    for (int t = 0; t < 2; ++t) {
        q8t->to_float(act[t].data(), af.data(), n);
        for (int r = 0; r < rows; ++r) {
            float g = 0;
            tc->vec_dot(n, &g, 0, m.data() + (size_t) r * rb, 0, act[t].data(), 0, 1);
            tt->to_float(m.data() + (size_t) r * rb, wf.data(), n);
            double d = 0, mag = 0;
            for (int i = 0; i < n; ++i) { d += (double) wf[i] * af[i]; mag += std::fabs((double) wf[i] * af[i]); }
            const double ours = ref[(size_t) t * rows + r];
            num3 += (ours - g) * (ours - g);
            num4 += (ours - d) * (ours - d);
            den += d * d;
            max3 = std::max(max3, std::fabs(ours - g) / (mag + 1e-30));
        }
    }
    const double r3 = std::sqrt(num3 / den), r4 = std::sqrt(num4 / den);
    const bool ok3 = r3 < 1e-5 && max3 < 1e-5, ok4 = r4 < 1e-4;
    std::printf("    vs ggml-cpu vec_dot: rel %.2e (worst row %.2e of sum|w a|) %s | vs to_float in double: rel %.2e %s\n",
                r3, max3, ok3 ? "ok" : "FAILED", r4, ok4 ? "ok" : "FAILED");
    return fails + !ok3 + !ok4;
}

// a whole expert through the engine's entry points vs the same expert on ggml-cpu's vec_dot
int check_expert(int gu, int dn, bool hadamard) {
    NativeFmt f;
    std::string err;
    if (!native_fmt(gu, dn, H, FF, f, err)) { std::printf("native_fmt: %s\n", err.c_str()); return 1; }
    std::vector<float> sh(H), sf(FF);
    for (int i = 0; i < H; ++i) sh[i] = (i * 7 % 3) ? 1.f : -1.f;
    for (int i = 0; i < FF; ++i) sf[i] = (i * 5 % 3) ? -1.f : 1.f;
    if (hadamard) {
        f.had_block = 128;
        f.had_gu = f.had_d = true;
        f.gu_signs = sh.data();
        f.d_signs = sf.data();
    }
    std::vector<uint8_t> blob(f.bytes);
    const auto g = make_matrix(gu, FF, H, 101), u = make_matrix(gu, FF, H, 202), d = make_matrix(dn, H, FF, 303);
    std::memcpy(blob.data(), g.data(), g.size());
    std::memcpy(blob.data() + f.up_off, u.data(), u.size());
    std::memcpy(blob.data() + f.down_off, d.data(), d.size());
    constexpr int NT = 3;
    double worst = 0;
    bool bitwise_group = true;
    std::vector<std::vector<float>> outs(NT, std::vector<float>(H));
    std::vector<std::vector<uint8_t>> acts(NT, std::vector<uint8_t>(kNativeActBytes));
    std::vector<std::vector<float>> ffs(NT, std::vector<float>(FF));
    std::vector<std::vector<uint8_t>> hqs(NT, std::vector<uint8_t>(kNativeHBytes));
    const void* a[NT];
    float* ffp[NT];
    const void* hp[NT];
    float* op[NT];
    for (int t = 0; t < NT; ++t) {
        const auto x = make_x(H, 7 + t);
        native_quant_act(f, x.data(), acts[t].data());
        a[t] = acts[t].data();
        ffp[t] = ffs[t].data();
        hp[t] = hqs[t].data();
        op[t] = outs[t].data();
    }
    native_gu_rows(f, blob.data(), a, NT, ffp, 0, FF);
    for (int t = 0; t < NT; ++t) native_quant_h(f, ffs[t].data(), hqs[t].data());
    native_down_rows(f, blob.data(), hp, NT, op, 0, H);
    const auto* tg = ggml_get_type_traits_cpu((ggml_type) gu);
    const auto* td = ggml_get_type_traits_cpu((ggml_type) dn);
    for (int t = 0; t < NT; ++t) {
        std::vector<float> ff(FF), out(H);
        for (int r = 0; r < FF; ++r) {
            float gg = 0, uu = 0;
            tg->vec_dot(H, &gg, 0, blob.data() + (size_t) r * f.gu_row, 0, a[t], 0, 1);
            tg->vec_dot(H, &uu, 0, blob.data() + f.up_off + (size_t) r * f.gu_row, 0, a[t], 0, 1);
            ff[r] = (gg / (1.f + std::exp(-gg))) * uu;
        }
        std::vector<uint8_t> hq(kNativeHBytes);
        native_quant_h(f, ff.data(), hq.data());
        for (int r = 0; r < H; ++r)
            td->vec_dot(FF, &out[r], 0, blob.data() + f.down_off + (size_t) r * f.d_row, 0, hq.data(), 0, 1);
        double num = 0, den = 0;
        for (int r = 0; r < H; ++r) { num += std::pow((double) outs[t][r] - out[r], 2); den += (double) out[r] * out[r]; }
        worst = std::max(worst, std::sqrt(num / den));
        // the same token alone through the engine path: the same bits as in the group of three
        std::vector<float> ff1(FF), o1(H);
        float* f1[1] = {ff1.data()};
        float* o1p[1] = {o1.data()};
        const void* a1[1] = {a[t]};
        native_gu_rows(f, blob.data(), a1, 1, f1, 0, FF);
        std::vector<uint8_t> h1(kNativeHBytes);
        native_quant_h(f, ff1.data(), h1.data());
        const void* h1p[1] = {h1.data()};
        native_down_rows(f, blob.data(), h1p, 1, o1p, 0, H);
        bitwise_group = bitwise_group && same_bits(o1.data(), outs[t].data(), H);
    }
    const bool ok = worst < 2e-3 && bitwise_group;
    std::printf("  expert %s/%s%s through native_gu_rows/native_down_rows (%s, %s): vs ggml-cpu vec_dot rel %.2e, "
                "alone vs in a group of %d: %s  %s\n",
                ggml_type_name((ggml_type) gu), ggml_type_name((ggml_type) dn), hadamard ? " +hadamard" : "",
                tq_kernels_enabled() ? "tq kernels" : "ggml-cpu", tq_isa_name(tq_isa()), worst, NT,
                bitwise_group ? "bitwise equal" : "DIFFERENT", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}

int synthetic(int gu, int dn, bool hadamard) {
    int fails = 0;
    std::printf("%s/%s%s (tq kernels: %s, best ISA here %s)\n", ggml_type_name((ggml_type) gu),
                ggml_type_name((ggml_type) dn), hadamard ? " +hadamard" : "", tq_kernels_enabled() ? "on" : "off",
                tq_isa_name(tq_isa()));
    std::printf("  %s, 96 rows of %d (gate/up shape)\n", ggml_type_name((ggml_type) gu), H);
    fails += check_matrix(gu, 96, H, 1000 + gu, hadamard);
    std::printf("  %s, 256 rows of %d (down shape)\n", ggml_type_name((ggml_type) dn), FF);
    fails += check_matrix(dn, 256, FF, 2000 + dn, hadamard);
    fails += check_expert(gu, dn, hadamard);
    return fails;
}

// ------------------------------------------------------------------------------------------------ the benchmark

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct BenchExpert {
    std::vector<uint8_t> gu, dn;   // gate rows then up rows; down rows
    size_t gu_row, d_row;
};

// one variant: compute one expert (gate+up rows, then down rows) for `nt` tokens
using ExpertFn = std::function<void(const BenchExpert&, int nt)>;

// the core clock right now: a chain of dependent adds runs at one per cycle (GCC / Clang on x86-64; 0 elsewhere)
double core_ghz() {
#if defined(__GNUC__) && defined(__x86_64__)
    uint64_t x = 0;
    const long n = 1000000;
    const double t0 = now_us();
    for (long i = 0; i < n; ++i) __asm__ volatile("add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0" : "+r"(x));
    return 4.0 * n / ((now_us() - t0) * 1e3);
#else
    return 0.0;
#endif
}

constexpr double kBlocksPerExpert = 2.0 * FF * (H / 128) + (double) H * (FF / 128);   // 128-weight blocks

// one slice of `ms` milliseconds cycling over the experts: ns per expert call and the clock around it (negated when
// the clock moved by more than 5% during the slice: such samples count only when a variant has no other)
std::pair<double, double> time_slice(const std::vector<BenchExpert>& ex, const ExpertFn& fn, int nt, double ms,
                                     size_t& next) {
    const double g0 = core_ghz();
    int it = 0;
    const double t0 = now_us();
    double t1 = t0;
    do {
        fn(ex[next++ % ex.size()], nt);
        ++it;
        t1 = now_us();
    } while (t1 - t0 < ms * 1000.0 || it < 2);
    const double g1 = core_ghz();
    const double ghz = 0.5 * (g0 + g1);
    return {(t1 - t0) * 1e3 / it, ghz > 0 && std::fabs(g1 - g0) > 0.05 * g0 ? -ghz : ghz};
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// `threads` threads, each cycling over its own experts: aggregate weight GB/s
double time_parallel(const std::vector<BenchExpert>& ex, const ExpertFn& fn, int threads, double ms) {
    std::atomic<long> done{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> th;
    const double t0 = now_us();
    for (int k = 0; k < threads; ++k)
        th.emplace_back([&, k] {
            long c = 0;
            for (size_t i = (size_t) k; !stop.load(std::memory_order_relaxed); i += (size_t) threads, ++c)
                fn(ex[i % ex.size()], 1);
            done += c;
        });
    std::this_thread::sleep_for(std::chrono::milliseconds((int) ms));
    stop = true;
    for (auto& t : th) t.join();
    const double us = now_us() - t0;
    const double bytes = (double) (ex[0].gu_row * 2 * FF + ex[0].dn.size()) * (double) done.load();
    return bytes / (us * 1e3);
}

int bench(bool quick, const std::string& only) {
    const int ncold = quick ? 48 : 96;   // ~50-130 MB of weights: past the last-level cache
    std::printf("CPU: %s | AVX-512 kernels: %s, AVX2: %s | ggml-cpu built for this CPU (-march=native unless "
                "STRATA_PORTABLE)\n", cpu_name().c_str(), cpu_avx512_ok() ? "yes" : "no", cpu_avx2_ok() ? "yes" : "no");
    std::printf("expert = gate+up %d x %d and down %d x %d of one type, %d distinct experts cycled (their weights come "
                "from DRAM, as a cache miss's do)\n\n", FF, H, H, FF, ncold);
    struct Variant { std::string name; int type; ExpertFn fn; bool ok; };
    std::vector<Variant> vars;
    auto add_tq = [&](int type) {
        for (TqIsa isa : {TqIsa::scalar, TqIsa::avx2, TqIsa::avx512}) {
            vars.push_back({std::string("strata ") + tq_isa_name(isa), type,
                            [type, isa](const BenchExpert& e, int nt) {
                                thread_local std::vector<uint8_t> a, h;
                                thread_local std::vector<float> ff, out;
                                a.resize(ggml_row_size(GGML_TYPE_Q8_0, H) * kTqMaxTokens);
                                h.resize(ggml_row_size(GGML_TYPE_Q8_0, FF) * kTqMaxTokens);
                                ff.resize((size_t) FF * kTqMaxTokens);
                                out.resize((size_t) H * kTqMaxTokens);
                                const void* ap[kTqMaxTokens];
                                const void* hp[kTqMaxTokens];
                                float* fp[kTqMaxTokens];
                                float* op[kTqMaxTokens];
                                for (int t = 0; t < nt; ++t) {
                                    ap[t] = a.data() + t * ggml_row_size(GGML_TYPE_Q8_0, H);
                                    hp[t] = h.data() + t * ggml_row_size(GGML_TYPE_Q8_0, FF);
                                    fp[t] = ff.data() + (size_t) t * FF;
                                    op[t] = out.data() + (size_t) t * H;
                                }
                                tq_gu_rows(isa, type, e.gu.data(), e.gu_row, e.gu_row * FF, H, ap, nt, fp, 0, FF);
                                tq_rows(isa, type, e.dn.data(), e.d_row, FF, hp, nt, op, 0, H);
                            },
                            tq_isa_ok(isa)});
        }
        vars.push_back({"ggml-cpu vec_dot", type,
                        [type](const BenchExpert& e, int nt) {
                            const auto* tc = ggml_get_type_traits_cpu((ggml_type) type);
                            thread_local std::vector<uint8_t> a(ggml_row_size(GGML_TYPE_Q8_0, H)),
                                h(ggml_row_size(GGML_TYPE_Q8_0, FF));
                            thread_local std::vector<float> ff(FF), out(H);
                            for (int t = 0; t < nt; ++t) {
                                for (int r = 0; r < FF; ++r) {
                                    float g, u;
                                    tc->vec_dot(H, &g, 0, e.gu.data() + (size_t) r * e.gu_row, 0, a.data(), 0, 1);
                                    tc->vec_dot(H, &u, 0, e.gu.data() + (size_t) (FF + r) * e.gu_row, 0, a.data(), 0, 1);
                                    ff[r] = (g / (1.f + std::exp(-g))) * u;
                                }
                                for (int r = 0; r < H; ++r)
                                    tc->vec_dot(FF, &out[r], 0, e.dn.data() + (size_t) r * e.d_row, 0, h.data(), 0, 1);
                            }
                        },
                        true});
    };
    auto add_q2 = [&](bool avx512) {
        vars.push_back({avx512 ? "strata Q2_0 avx512" : "strata Q2_0 avx2", 42,
                        [avx512](const BenchExpert& e, int nt) {
                            thread_local ActQ a1[MAXT], a2[MAXT];
                            thread_local bool init = false;
                            if (!init) {   // real scales and codes (the kernel reads every field)
                                const auto x = make_x(H, 3), h = make_x(FF, 4);
                                for (int t = 0; t < MAXT; ++t) { act_quant_q8_1_avx2(x.data(), H, a1[t]); act_quant_q8_1_avx2(h.data(), FF, a2[t]); }
                                init = true;
                            }
                            thread_local float g[MAXT][FF], u[MAXT][FF], out[MAXT][H];
                            const ActQ* ap[MAXT];
                            const ActQ* hp[MAXT];
                            float* gp[MAXT];
                            float* up[MAXT];
                            float* op[MAXT];
                            for (int t = 0; t < nt; ++t) { ap[t] = &a1[t]; hp[t] = &a2[t]; gp[t] = g[t]; up[t] = u[t]; op[t] = out[t]; }
                            auto rows = avx512 ? q2_0_gguf_rows_multi : q2_0_gguf_rows_multi_avx2;
                            rows(e.gu.data(), e.gu_row, H / 64, ap, nt, gp, 0, FF);
                            rows(e.gu.data() + e.gu_row * FF, e.gu_row, H / 64, ap, nt, up, 0, FF);
                            for (int t = 0; t < nt; ++t)
                                for (int r = 0; r < FF; ++r) g[t][r] = (g[t][r] / (1.f + std::exp(-g[t][r]))) * u[t][r];
                            rows(e.dn.data(), e.d_row, FF / 64, hp, nt, op, 0, H);
                        },
                        avx512 ? cpu_avx512_ok() : cpu_avx2_ok()});
    };
    for (int type : {145, 146, 144}) add_tq(type);
    add_q2(false);
    add_q2(true);
    // the Q2_0 pack's own AVX-512 kernel (the repacked 1,382,400-byte blob that setup makes on AVX-512 CPUs)
    vars.push_back({"strata Q2_0 avx512 pack", 42,
                    [](const BenchExpert& e, int nt) {
                        thread_local ActQ a1, a2;
                        thread_local bool init = false;
                        if (!init) {
                            const auto x = make_x(H, 3), h = make_x(FF, 4);
                            act_quant_q8_1(x.data(), H, a1);
                            act_quant_q8_1(h.data(), FF, a2);
                            init = true;
                        }
                        thread_local float ff[FF], out[H];
                        for (int t = 0; t < nt; ++t) {   // its multi-token kernels are the same per token
                            s2_expert_gu_rows(e.gu.data(), a1, ff, 0, FF);
                            s2_expert_down_rows(e.gu.data(), a2, out, 0, H);
                        }
                    },
                    cpu_avx512_ok()});

    auto make_experts = [&](int type, int count) {
        std::vector<BenchExpert> v((size_t) count);
        for (int k = 0; k < count; ++k) {
            BenchExpert& e = v[(size_t) k];
            e.gu_row = ggml_row_size((ggml_type) type, H);
            e.d_row = ggml_row_size((ggml_type) type, FF);
            // (Q2_0: room for one repacked blob too, which the pack kernel reads from the same buffer)
            e.gu.resize(type == 42 ? std::max(e.gu_row * 2 * FF, (size_t) BLOB) : e.gu_row * 2 * FF);
            e.dn.resize(e.d_row * H);
            std::mt19937 rng(77 + k);
            for (auto& b : e.gu) b = (uint8_t) rng();
            for (auto& b : e.dn) b = (uint8_t) rng();
            // sane fp16 scales at every block start (Q2_0: 18-byte blocks, TQ: 26/30/34)
            const size_t bsz = type == 42 ? 18 : ggml_row_size((ggml_type) type, 128);
            const uint16_t d = f2h(0.01f);
            for (size_t o = 0; o + bsz <= e.gu.size(); o += bsz) std::memcpy(e.gu.data() + o, &d, 2);
            for (size_t o = 0; o + bsz <= e.dn.size(); o += bsz) std::memcpy(e.dn.data() + o, &d, 2);
            if (type == 42) {   // the same bytes also hold one repacked Q2_0 blob (codes, then fp16 scales)
                for (size_t o = O_GU_SCALES; o + 2 <= BLOB; o += 2) std::memcpy(e.gu.data() + o, &d, 2);
            }
        }
        return v;
    };
    // Single thread, the variants of a type interleaved in short slices over several rounds (this machine's clock
    // moves; a slice whose clock moved is dropped), medians reported: ns and core cycles per 128-weight block, us
    // per expert, weight GB/s; and per token in a group of three (a verify window's tokens routed to one expert).
    // The activations are zero-filled (the kernels do not branch on values) except Q2_0's, which are real.
    const int rounds = quick ? 5 : 15;
    const double slice_ms = quick ? 40 : 60;
    std::printf("%-6s %-24s %9s %8s %8s %8s | %10s %8s\n", "type", "kernel", "us/expert", "ns/blk", "cyc/blk", "GB/s",
                "x3: us/tok", "cyc/blk");
    for (int type : {145, 146, 144, 42}) {
        const auto cold = make_experts(type, ncold);
        const double eb = (double) (cold[0].gu_row * 2 * FF + cold[0].dn.size());   // weight bytes per expert
        std::vector<const Variant*> vs;
        for (const auto& v : vars)
            if (v.type == type && v.name.find(only) != std::string::npos) {
                if (v.ok) vs.push_back(&v);
                else std::printf("%-6s %-24s (not on this CPU)\n", ggml_type_name((ggml_type) type), v.name.c_str());
            }
        std::vector<std::vector<double>> ns1(vs.size()), cy1(vs.size()), ns3(vs.size()), cy3(vs.size());
        std::vector<std::vector<double>> ns1m(vs.size()), cy1m(vs.size()), ns3m(vs.size()), cy3m(vs.size());
        size_t next = 0;
        for (auto* v : vs) v->fn(cold[0], 3);   // warm-up: tables, buffers
        for (int r = 0; r < rounds; ++r)
            for (size_t k = 0; k < vs.size(); ++k) {
                const bool slow = vs[k]->name.find("scalar") != std::string::npos;
                const auto a = time_slice(cold, vs[k]->fn, 1, slow ? slice_ms / 2 : slice_ms, next);
                const auto b = time_slice(cold, vs[k]->fn, 3, slow ? slice_ms / 2 : slice_ms, next);
                if (a.second >= 0) { ns1[k].push_back(a.first); cy1[k].push_back(a.first * a.second); }
                else { ns1m[k].push_back(a.first); cy1m[k].push_back(-a.first * a.second); }
                if (b.second >= 0) { ns3[k].push_back(b.first); cy3[k].push_back(b.first * b.second); }
                else { ns3m[k].push_back(b.first); cy3m[k].push_back(-b.first * b.second); }
            }
        std::vector<size_t> steady(vs.size());
        for (size_t k = 0; k < vs.size(); ++k) {
            steady[k] = ns1[k].size();
            if (ns1[k].empty()) { ns1[k] = ns1m[k]; cy1[k] = cy1m[k]; }
            if (ns3[k].empty()) { ns3[k] = ns3m[k]; cy3[k] = cy3m[k]; }
        }
        for (size_t k = 0; k < vs.size(); ++k) {
            const double n1 = median(ns1[k]), c1 = median(cy1[k]), n3 = median(ns3[k]), c3 = median(cy3[k]);
            std::printf("%-6s %-24s %9.1f %8.1f %8.1f %8.2f | %10.1f %8.1f   (%zu/%d steady)\n",
                        ggml_type_name((ggml_type) type), vs[k]->name.c_str(), n1 / 1e3, n1 / kBlocksPerExpert,
                        c1 / kBlocksPerExpert, eb / n1, n3 / 3e3, c3 / kBlocksPerExpert, steady[k], rounds);
        }
        std::fflush(stdout);
    }
    // several workers at once (as the pool runs a layer's misses): aggregate GB/s on the cold set
    const int nthr = (int) std::max(1u, std::min(4u, std::thread::hardware_concurrency()));
    std::printf("\n%d threads at once, cold experts, one token: aggregate weight GB/s (us per expert per thread)\n", nthr);
    for (int type : {145, 146, 144, 42}) {
        const auto cold = make_experts(type, ncold);
        const double eb = (double) (cold[0].gu_row * 2 * FF + cold[0].dn.size());
        std::vector<const Variant*> vs;
        for (const auto& v : vars)
            if (v.type == type && v.ok && v.name.find("scalar") == std::string::npos && v.name.find(only) != std::string::npos)
                vs.push_back(&v);
        std::vector<std::vector<double>> g(vs.size());
        for (int r = 0; r < (quick ? 2 : 4); ++r)
            for (size_t k = 0; k < vs.size(); ++k) g[k].push_back(time_parallel(cold, vs[k]->fn, nthr, quick ? 150 : 250));
        for (size_t k = 0; k < vs.size(); ++k) {
            const double gbs = median(g[k]);
            std::printf("%-6s %-24s %8.2f GB/s (%7.1f us per expert per thread)\n", ggml_type_name((ggml_type) type),
                        vs[k]->name.c_str(), gbs, eb / (gbs * 1e3) * nthr);
        }
        std::fflush(stdout);
    }
    // the per-token Hadamard rotation of one layer's activation (once per token per layer, shared by its experts)
    {
        std::vector<float> x = make_x(H, 5), s(H, 1.f);
        const int it = 20000;
        const double t0 = now_us();
        for (int i = 0; i < it; ++i) hadamard_rotate(x.data(), H, 128, s.data());
        std::printf("\nHadamard rotation of one %d-wide activation (block 128): %.2f us\n", H, (now_us() - t0) / it);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    ggml_cpu_init();
    if (GGML_TYPE_COUNT <= 146) {
        std::fprintf(stderr, "this ggml has no trellis types (build with STRATA_GGML_DIR = the agention llama.cpp fork): skipped\n");
        return 77;   // ctest SKIP_RETURN_CODE: a plain ggml is not a failure (#3358)
    }
    if (argc >= 2 && std::strcmp(argv[1], "--bench") == 0) {   // --bench [--quick] [--only <kernel name part>]
        bool quick = false;
        std::string only;
        for (int i = 2; i < argc; ++i) {
            if (std::strcmp(argv[i], "--quick") == 0) quick = true;
            else if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) only = argv[++i];
        }
        return bench(quick, only);
    }
    if (argc >= 3 && (std::strcmp(argv[1], "--synthetic") == 0 || std::strcmp(argv[1], "--synthetic-hadamard") == 0)) {
        const std::string pair = argv[2];
        const size_t sl = pair.find('/');
        const int gu = type_by_name(pair.substr(0, sl)), dn = sl == std::string::npos ? gu : type_by_name(pair.substr(sl + 1));
        if (!tq_supported(gu) || !tq_supported(dn)) { std::fprintf(stderr, "not a trellis pair: %s\n", argv[2]); return 2; }
        const int f = synthetic(gu, dn, std::strcmp(argv[1], "--synthetic-hadamard") == 0);
        std::printf("%d failures\n", f);
        return f ? 1 : 0;
    }
    int f = 0;
    for (bool had : {false, true}) {
        f += synthetic(145, 146, had);   // Gyro-S
        f += synthetic(144, 144, had);   // Gyro-M
    }
    std::printf("%d failures\n", f);
    return f ? 1 : 0;
}
