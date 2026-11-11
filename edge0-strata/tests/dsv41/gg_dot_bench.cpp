// Strata. The DeepSeek-V4.1 CPU lane's ggml-type rows (src/dsv41/ds4_gg_cpu.cpp): ggml-cpu's row dot against Strata's AVX2
// multi-token i-quant rows, on random blocks of IQ2_XXS / IQ3_XXS.
//   ds4_gg_bench [--type 16|18] [--threads N] [--mb M] [--nt 1..8] [--reps R]
// Checks first (every row, every token): the two paths' results agree to a few float ulps (the integer sums are the same;
// only the order of the float additions differs) and, with --exact, bit for bit.  Then the time of a pass over M MB of
// weights (default 3072, far past L3) on N threads, as GB/s of weights read, beside a plain streaming read of the same
// bytes (the memory ceiling the rows are measured against).
#include "ds4_gguf_cpu.hpp"

#include "ggml-cpu.h"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace strata::ds4::gguf;
using clk = std::chrono::steady_clock;

static double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }

// ---- --expert: the engine's expert pipeline on random experts (gate+up+SwiGLU, quantize h, down) in jobs of 64 rows over N threads, for
// ggml-cpu's dots (mode 0), the default rule (2) and the AVX2 rows always (1); GB/s of all three matrices' bytes, outputs compared bit for bit.
static void fill_blocks(uint8_t* w, size_t bytes, int type, uint64_t seed) {
    uint64_t st = 0x9E3779B97F4A7C15ull ^ (seed * 0xD1B54A32D192ED03ull);
    auto next = [&st] { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
    for (size_t o = 0; o + 8 <= bytes; o += 8) { const uint64_t v = next(); std::memcpy(w + o, &v, 8); }
    const size_t bb = ggml_type_size((ggml_type) type);
    for (size_t o = 0; o + bb <= bytes; o += bb) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.002f + 0.001f * (float) (next() % 16));
        if (type == 29) {
            uint16_t sc[4];
            std::memcpy(sc, w + o + 48, 8);
            for (int k = 0; k < 4; ++k) sc[k] = (uint16_t) ((sc[k] & 0x0fff) | ((d >> (4 * k)) & 0xf) << 12);
            std::memcpy(w + o + 48, sc, 8);
        } else {
            std::memcpy(w + o, &d, sizeof d);
        }
    }
}

static int run_expert(int tg, int td, int threads, long mb, int nt, int reps) {
    ggml_cpu_init();
    const int E = 5120, FF = 2304;
    IqCpu f;
    std::string err;
    if (!gg_cpu_fmt(tg, td, E, FF, f, err)) { std::fprintf(stderr, "gg_cpu_fmt: %s\n", err.c_str()); return 2; }
    const size_t gbytes = f.gu_row * FF, dbytes = f.d_row * E, ebytes = 2 * gbytes + dbytes;
    const int ne = (int) std::max(2L, (long) ((size_t) mb * 1000000 / ebytes));
    std::vector<uint8_t> gw((size_t) ne * gbytes), uw((size_t) ne * gbytes), dw((size_t) ne * dbytes);
    for (int e = 0; e < ne; ++e) {
        fill_blocks(&gw[(size_t) e * gbytes], gbytes, tg, 3 * (uint64_t) e + 1);
        fill_blocks(&uw[(size_t) e * gbytes], gbytes, tg, 3 * (uint64_t) e + 2);
        fill_blocks(&dw[(size_t) e * dbytes], dbytes, td, 3 * (uint64_t) e + 3);
    }
    uint64_t st = 12345;
    auto next = [&st] { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
    std::vector<std::vector<uint8_t>> act((size_t) nt, std::vector<uint8_t>(f.act_bytes));
    for (int t = 0; t < nt; ++t) {
        std::vector<float> x((size_t) E);
        for (auto& v : x) v = ((float) (next() % 2001) - 1000.0f) * 1e-3f;
        gg_cpu_quant(f.gu_act, x.data(), E, act[(size_t) t].data());
    }
    std::vector<float> h((size_t) ne * nt * FF), y((size_t) ne * nt * E);
    std::vector<uint8_t> hq((size_t) ne * nt * f.h_bytes);
    const int RB = 64;
    auto run_jobs = [&](long njobs, const std::function<void(long)>& fn) {
        std::atomic<long> cur{0};
        std::vector<std::thread> th;
        for (int k = 0; k < threads; ++k)
            th.emplace_back([&] { for (;;) { const long j = cur.fetch_add(1); if (j >= njobs) break; fn(j); } });
        for (auto& t : th) t.join();
    };
    auto pass = [&](int mode) {
        gg_cpu_set_iq256(mode);
        double best = 1e30;
        for (int r = 0; r < reps + 1; ++r) {
            const auto t0 = clk::now();
            const long gb = (FF + RB - 1) / RB, db = (E + RB - 1) / RB;
            run_jobs((long) ne * gb, [&](long j) {
                const int e = (int) (j / gb), r0 = (int) (j % gb) * RB, r1 = std::min(FF, r0 + RB);
                const void* ap[64];
                float* hp[64];
                for (int t = 0; t < nt; ++t) { ap[t] = act[(size_t) t].data(); hp[t] = &h[((size_t) e * nt + t) * FF]; }
                gg_cpu_gate_up(f, &gw[(size_t) e * gbytes], &uw[(size_t) e * gbytes], ap, nt, hp, r0, r1, 10.0f);
            });
            run_jobs((long) ne * nt, [&](long j) { gg_cpu_quant(f.d_act, &h[(size_t) j * FF], FF, &hq[(size_t) j * f.h_bytes]); });
            run_jobs((long) ne * db, [&](long j) {
                const int e = (int) (j / db), r0 = (int) (j % db) * RB, r1 = std::min(E, r0 + RB);
                const void* hp[64];
                float* yp[64];
                for (int t = 0; t < nt; ++t) { hp[t] = &hq[((size_t) e * nt + t) * f.h_bytes]; yp[t] = &y[((size_t) e * nt + t) * E]; }
                gg_cpu_down(f, &dw[(size_t) e * dbytes], hp, nt, yp, r0, r1);
            });
            if (r > 0) best = std::min(best, secs(t0));
        }
        return best;
    };
    const double gb = (double) ne * ebytes / 1e9;
    std::vector<float> y0;
    double t0s = 0;
    int rc = 0;
    for (int mode : {0, 2, 1}) {
        const double t = pass(mode);
        if (mode == 0) { y0 = y; t0s = t; }
        const bool same = std::memcmp(y0.data(), y.data(), y.size() * 4) == 0;
        std::printf("expert gate/up %d down %d nt %d threads %d: %d experts, %.2f GB | mode %s %.3f s = %.1f GB/s (x%.2f) %s\n", tg, td, nt,
                    threads, ne, gb, mode == 0 ? "ggml " : mode == 2 ? "auto " : "own  ", t, gb / t, t0s / t, same ? "bit-identical" : "DIFFERENT");
        if (!same) rc = 1;
    }
    return rc;
}

int main(int argc, char** argv) {
    int type = 18, threads = 32, nt = 1, reps = 3;
    long mb = 3072;
    bool exact = false, expert = false, k512 = false;
    int dtype = -1;
    std::string dump;   // --dump FILE: ggml-cpu's results (nt x rows floats) written raw - compare two builds with cmp
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&] { return i + 1 < argc ? std::atol(argv[++i]) : 0L; };
        if (a == "--type") type = (int) val();
        else if (a == "--threads") threads = (int) val();
        else if (a == "--mb") mb = val();
        else if (a == "--nt") nt = (int) val();
        else if (a == "--reps") reps = (int) val();
        else if (a == "--exact") exact = true;
        else if (a == "--expert") expert = true;
        else if (a == "--k512") k512 = true;
        else if (a == "--dtype") dtype = (int) val();
        else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
    }
    if (expert) return run_expert(type, dtype < 0 ? type : dtype, threads, mb, nt, reps);
    ggml_cpu_init();
    const int n = 5120;   // a gate / up row
    IqCpu f;
    std::string err;
    if (!gg_cpu_fmt(type, type, n, 2304, f, err)) {
        std::fprintf(stderr, "gg_cpu_fmt: %s\n", err.c_str());
        return 2;
    }
    const size_t rb = f.gu_row;
    const long rows = (long) ((size_t) mb * 1000000 / rb) / 8 * 8;
    std::vector<uint8_t> w((size_t) rows * rb);
    uint64_t st = 0x9E3779B97F4A7C15ull;
    auto next = [&st] { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
    for (size_t o = 0; o + 8 <= w.size(); o += 8) { const uint64_t v = next(); std::memcpy(&w[o], &v, 8); }
    const size_t bb = ggml_type_size((ggml_type) type);
    for (size_t o = 0; o + bb <= w.size(); o += bb) {   // the blocks' F16 scale: a small normal value
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.002f + 0.001f * (float) (next() % 16));
        if (type == 29) {   // IQ1_M: no d field - its f16 is spread over the top nibbles of the four scale words (bytes 48..55)
            uint16_t sc[4];
            std::memcpy(sc, &w[o + 48], 8);
            for (int k = 0; k < 4; ++k) sc[k] = (uint16_t) ((sc[k] & 0x0fff) | ((d >> (4 * k)) & 0xf) << 12);
            std::memcpy(&w[o + 48], sc, 8);
        } else {
            std::memcpy(&w[o], &d, sizeof d);
        }
    }
    std::vector<std::vector<uint8_t>> a((size_t) nt, std::vector<uint8_t>(f.act_bytes));
    std::vector<const void*> ap((size_t) nt);
    for (int t = 0; t < nt; ++t) {
        std::vector<float> x((size_t) n);
        for (auto& v : x) v = ((float) (next() % 2001) - 1000.0f) * 1e-3f;
        gg_cpu_quant(f.gu_act, x.data(), n, a[(size_t) t].data());
        ap[(size_t) t] = a[(size_t) t].data();
    }
    if (k512) {   // --k512: Strata's AVX-512 rows (iq512x_rows) called directly against ggml-cpu's dot, bit for bit: no dispatcher in between
        const bool cpu_ok = strata::kernels::cpu::cpu_avx512_ok(), sup = strata::kernels::cpu::iq512x_supported(type);
        std::printf("k512: type %d nt %d: cpu_avx512_ok %d, iq512x_supported %d%s\n", type, nt, (int) cpu_ok, (int) sup,
                    cpu_ok && sup ? "" : "  -> the AVX-512 rows cannot run here");
        if (!cpu_ok || !sup) return 3;
        const ggml_type_traits_cpu* tc = ggml_get_type_traits_cpu((ggml_type) type);
        std::vector<std::vector<float>> r0v((size_t) nt, std::vector<float>((size_t) rows)), r1v = r0v;
        std::vector<float*> q1;
        for (int t = 0; t < nt; ++t) q1.push_back(r1v[(size_t) t].data());
        for (long r = 0; r < rows; ++r)
            for (int t = 0; t < nt; ++t) tc->vec_dot(n, &r0v[(size_t) t][(size_t) r], 0, w.data() + (size_t) r * rb, 0, ap[(size_t) t], 0, 1);
        strata::kernels::cpu::iq512x_rows(type, w.data(), rb, n, ap.data(), nt, q1.data(), 0, (int) rows);
        long dif = 0;
        for (int t = 0; t < nt; ++t)
            for (long r = 0; r < rows; ++r)
                if (std::memcmp(&r0v[(size_t) t][(size_t) r], &r1v[(size_t) t][(size_t) r], 4) != 0) ++dif;
        std::printf("k512: iq512x_rows against ggml-cpu's dot: %ld of %ld differ in bits\n%s\n", dif, (long) nt * rows, dif == 0 ? "OK" : "FAIL");
        return dif == 0 ? 0 : 1;
    }
    std::vector<std::vector<float>> o0((size_t) nt, std::vector<float>((size_t) rows)), o1 = o0;
    std::vector<float*> p0, p1;
    for (int t = 0; t < nt; ++t) { p0.push_back(o0[(size_t) t].data()); p1.push_back(o1[(size_t) t].data()); }

    // a pass over all rows on `threads` threads, mode 0 ggml-cpu, 1 Strata's rows; returns the best time of reps
    auto pass = [&](int mode, float* const* out) {
        gg_cpu_set_iq256(mode);
        double best = 1e30;
        for (int r = 0; r < reps + 1; ++r) {
            const auto t0 = clk::now();
            std::vector<std::thread> th;
            std::atomic<long> cursor{0};
            const long chunk = 64;
            for (int k = 0; k < threads; ++k)
                th.emplace_back([&] {
                    for (;;) {
                        const long r0 = cursor.fetch_add(chunk);
                        if (r0 >= rows) break;
                        // one "expert part" is 2304 rows; the engine hands a thread a few rows of it at a time
                        gg_cpu_down(f, w.data(), ap.data(), nt, out, (int) r0, (int) std::min(rows, r0 + chunk));
                    }
                });
            for (auto& t : th) t.join();
            if (r > 0) best = std::min(best, secs(t0));
        }
        return best;
    };
    // gg_cpu_down uses f.td / d_row; here td == tg and the activation has n_embd values: f.d_row would be for n_ff.
    // Use the gate rows through a second IqCpu whose down geometry is the gate's.
    f.td = f.tg; f.d_row = f.gu_row; f.n_ff = n; f.d_act = f.gu_act; f.h_bytes = f.act_bytes;

    const double t_gg = pass(0, p0.data());
    const double t_own = pass(1, p1.data());
    if (!dump.empty()) {
        if (std::FILE* df = std::fopen(dump.c_str(), "wb")) {
            for (int t = 0; t < nt; ++t) std::fwrite(o0[(size_t) t].data(), 4, (size_t) rows, df);
            std::fclose(df);
        }
    }
    long bad = 0, differ = 0;
    double worst = 0, worst_ref = 0, absmax = 0, absdiff = 0;
    for (int t = 0; t < nt; ++t)
        for (long r = 0; r < rows; ++r) {
            const float x = o0[(size_t) t][(size_t) r], y = o1[(size_t) t][(size_t) r];
            absmax = std::max(absmax, (double) std::fabs(x));
            absdiff = std::max(absdiff, (double) std::fabs((double) x - y));
            if (std::memcmp(&x, &y, 4) != 0) {
                ++differ;
                const double rel = std::fabs((double) x - y) / (std::fabs((double) x) + 1e-3);
                worst = std::max(worst, rel);
                if (rel > 1e-5) ++bad;
            }
        }
    // The pass criterion above divides by one value's magnitude, so a token whose dot cancels to near zero fails it while
    // agreeing to a few ulps.  The real bound is the L1 norm of the products (|x - ref| <= eps * sum|w x|): decode a
    // sample of rows with ggml's own to_float and the activation as its Q8_K blocks hold it, and compare in double.
    int sampled = 0;
    {
        const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) type);
        std::vector<float> wrow((size_t) n);
        // the activations (Q8_K: float d, int8 qs[256], int16 bsums[16]) as the rows see them
        std::vector<std::vector<float>> xv((size_t) nt, std::vector<float>((size_t) n));
        for (int t = 0; t < nt; ++t)
            for (int b = 0; b < n / 256; ++b) {
                const uint8_t* p = (const uint8_t*) ap[(size_t) t] + (size_t) b * (f.act_bytes / (size_t) (n / 256));
                float d;
                std::memcpy(&d, p, 4);
                const int8_t* q = (const int8_t*) (p + 4);
                for (int j = 0; j < 256; ++j) xv[(size_t) t][(size_t) b * 256 + j] = d * (float) q[j];
            }
        const long step = std::max(1L, rows / 64);
        for (long r = 0; r < rows && sampled < 64; r += step, ++sampled) {
            if (tt == nullptr || tt->to_float == nullptr) break;
            tt->to_float(w.data() + (size_t) r * rb, wrow.data(), n);
            for (int t = 0; t < nt; ++t) {
                double ref = 0, l1 = 0;
                for (int j = 0; j < n; ++j) {
                    const double pr = (double) wrow[(size_t) j] * (double) xv[(size_t) t][(size_t) j];
                    ref += pr;
                    l1 += std::fabs(pr);
                }
                const double g = o0[(size_t) t][(size_t) r], o = o1[(size_t) t][(size_t) r];
                worst_ref = std::max(worst_ref, std::fabs(g - ref) / (l1 + 1e-30));
                worst_ref = std::max(worst_ref, std::fabs(o - ref) / (l1 + 1e-30));
            }
        }
    }
    // memory ceiling: a plain read of the same bytes on the same threads
    double t_rd = 1e30;
    for (int r = 0; r < reps + 1; ++r) {
        const auto t0 = clk::now();
        std::vector<std::thread> th;
        std::vector<uint64_t> sink((size_t) threads);
        for (int k = 0; k < threads; ++k)
            th.emplace_back([&, k] {
                const size_t lo = w.size() / (size_t) threads * (size_t) k, hi = k + 1 == threads ? w.size() : w.size() / (size_t) threads * (size_t) (k + 1);
                uint64_t s = 0, v;
                for (size_t o = lo; o + 8 <= hi; o += 8) { std::memcpy(&v, &w[o], 8); s += v; }
                sink[(size_t) k] = s;
            });
        for (auto& t : th) t.join();
        if (r > 0) t_rd = std::min(t_rd, secs(t0));
    }
    const double gb = (double) w.size() / 1e9;
    std::printf("type %d nt %d threads %d: %.2f GB of weights (%ld rows)\n", type, nt, threads, gb, rows);
    std::printf("  ggml-cpu rows %.3f s = %.1f GB/s | Strata AVX2 rows %.3f s = %.1f GB/s (x%.2f) | plain read %.1f GB/s\n", t_gg, gb / t_gg,
                t_own, gb / t_own, t_gg / t_own, gb / t_rd);
    std::printf("  results: %ld of %ld differ in bits, worst |x-y|/|x| %.2e, %ld beyond 1e-5\n", differ, (long) nt * rows, worst, bad);
    std::printf("  vs a double reference on %d sampled rows: worst %.2e of the L1 norm (tol 1e-5); |x-y| max %.2e of |x| max %.2e\n",
                sampled, worst_ref, absdiff, absmax);
    if (worst_ref > 1e-5 || absdiff > 1e-6 * absmax + 1e-30 || (exact && differ != 0)) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
