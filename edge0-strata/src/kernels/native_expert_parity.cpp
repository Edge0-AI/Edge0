// src/kernels/native_expert_parity.cpp - plan v0.3 P6: one native expert three ways.
//
//     build/native_expert_parity <shard.gguf> [layer ...]     real rows (a split model's shards are found by name)
//     build/native_expert_parity --synthetic GU/DOWN ...      random weights quantized by ggml (e.g. q4_K/q5_1)
//     build/native_expert_parity --q5_1-min                   the Q5_1 min term on crafted activations
//     (the CPU-only checks of the trellis kernels, AVX2 / AVX-512 / scalar, are in tq_cpu_parity)
//
// (a) float reference: ggml's own dequantizer (`to_float`) and a float SwiGLU expert, (b) the CPU path
// (ggml-cpu vec_dot with its quantized activations), (c) the GPU path (`native_expert_grouped`, q8_1
// activations).  (b) and (c) each differ from (a) by their activation rounding (a few 1e-3 relative) and, for the
// trellis types on the GPU, by its int8 codebook (tq_lut_i8, see iq_kernels.cu "Numerics"); both must stay within
// relative L1 error 3e-2 of (a).  (c) runs twice, with the kernels that decode a weight part once for all entries and
// with the per-entry ones: those two are bitwise equal to each other.
// (d) the GPU dequantizers of the prompt path and the embedding against `to_float` (Q8_0: bit for bit).
// The synthetic mode and (d) follow eddoursul/Strata 8029fa9.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"
#include "ggml-cpu.h"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/expert_source.hpp"

#include <fstream>
#include <set>
#include <sstream>

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

static double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}


namespace {
constexpr int NT = 3, E = 7;
constexpr int64_t H = 2560, FF = 640;

// The three ways and the dequantizers for one expert blob; returns the number of failed checks.
int check_blob(const cpu::NativeFmt& f, const std::vector<uint8_t>& blob, int seed, const std::string& label,
               cudaStream_t s) {
    int failures = 0;
    // (a) the float reference
    const auto* tg = ggml_get_type_traits((ggml_type) f.gu_type);
    const auto* td = ggml_get_type_traits((ggml_type) f.d_type);
    std::vector<float> G((size_t) FF * H), U((size_t) FF * H), D((size_t) H * FF);
    for (int64_t r = 0; r < FF; ++r) {
        tg->to_float(blob.data() + r * f.gu_row, G.data() + r * H, H);
        tg->to_float(blob.data() + f.up_off + r * f.gu_row, U.data() + r * H, H);
    }
    for (int64_t r = 0; r < H; ++r) td->to_float(blob.data() + f.down_off + r * f.d_row, D.data() + r * FF, FF);
    if (f.tail_off) {   // the blob's scales, as the engine applies them (gate and up as read, down on its output)
        float tail[4];
        std::memcpy(tail, blob.data() + f.tail_off, sizeof tail);
        for (auto& v : G) v *= tail[0];
        for (auto& v : U) v *= tail[1];
        for (auto& v : D) v *= tail[2];
    }
    std::mt19937 rng(11 + seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) NT * H);
    for (auto& v : x) v = nd(rng);
    std::vector<float> ref((size_t) NT * H), got_c((size_t) NT * H), got_g((size_t) NT * H);
    // APR (--hadamard): the folded expert's reference takes H (s * x) and H (s * h), as llama.cpp's graph does
    std::vector<float> xr(x);
    if (f.had_gu) for (int k = 0; k < NT; ++k) cpu::hadamard_rotate(xr.data() + k * H, (int) H, f.had_block, f.gu_signs);
    for (int k = 0; k < NT; ++k) {
        std::vector<float> h(FF);
        for (int64_t r = 0; r < FF; ++r) {
            double g = 0, u = 0;
            for (int64_t i = 0; i < H; ++i) { g += (double) G[r * H + i] * xr[k * H + i]; u += (double) U[r * H + i] * xr[k * H + i]; }
            h[r] = (float) (g / (1.0 + std::exp(-g)) * u);
        }
        if (f.had_d) cpu::hadamard_rotate(h.data(), (int) FF, f.had_block, f.d_signs);
        for (int64_t r = 0; r < H; ++r) {
            double o = 0;
            for (int64_t i = 0; i < FF; ++i) o += (double) D[r * FF + i] * h[i];
            ref[k * H + r] = (float) o;
        }
    }
    // (b) the CPU
    {
        std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(cpu::kNativeActBytes));
        std::vector<std::vector<uint8_t>> hq(NT, std::vector<uint8_t>(cpu::kNativeHBytes));
        std::vector<std::vector<float>> ff(NT, std::vector<float>(FF));
        const void* a[NT];
        float* ffp[NT];
        const void* hp[NT];
        float* op[NT];
        for (int k = 0; k < NT; ++k) {
            cpu::native_quant_act(f, x.data() + k * H, act[k].data());
            a[k] = act[k].data();
            ffp[k] = ff[k].data();
        }
        if ((f.gu_type == 18 || f.gu_type == 21 || f.gu_type == 22) && cpu::cpu_avx2_ok() && !cpu::cpu_avx512_ok()) {
            const auto dot = ggml_get_type_traits_cpu((ggml_type) f.gu_type)->vec_dot;
            size_t differ = 0;
            std::vector<float> exact((size_t) FF);
            for (int k = 0; k < NT; ++k) {
                cpu::iq256_gu_rows_exact_one(f.gu_type, blob.data(), f.gu_row, f.up_off, (int) H, a[k],
                                              exact.data(), 0, (int) FF);
                for (int r = 0; r < FF; ++r) {
                    float g = 0.f, u = 0.f;
                    dot((int) H, &g, 0, blob.data() + (size_t) r * f.gu_row, 0, a[k], 0, 1);
                    dot((int) H, &u, 0, blob.data() + f.up_off + (size_t) r * f.gu_row, 0, a[k], 0, 1);
                    const float want = (g / (1.f + std::exp(-g))) * u;
                    differ += std::memcmp(&want, &exact[(size_t) r], sizeof(float)) != 0;
                }
            }
            std::printf("%s exact singleton: %zu of %lld real gate/up rows differ from ggml\n",
                        label.c_str(), differ, (long long) (NT * FF));
            if (differ) ++failures;
        }
        cpu::native_gu_rows(f, blob.data(), a, NT, ffp, 0, (int) FF);
        // either multi-token kernel: IQ4_XS has an AVX-2 one and no AVX-512 one, so the gate cannot be
        // iq512_supported alone - that would leave the format untested on every CPU.
        if (cpu::iq512_supported(f.gu_type) || cpu::iq256_supported(f.gu_type)) {
            // ggml's own vec_dot, same Q8_K activations: the reference for both multi-token kernels
            // (float-order differences only)
            const auto* tc = ggml_get_type_traits_cpu((ggml_type) f.gu_type);
            std::vector<float> gref((size_t) NT * FF);
            for (int k = 0; k < NT; ++k)
                for (int64_t r = 0; r < FF; ++r)
                    tc->vec_dot((int) H, &gref[k * FF + r], 0, blob.data() + r * f.gu_row, 0, a[k], 0, 1);
            double usg = 0.0;
            {   // ggml single-token throughput, the weights cache-resident (compute bound)
                const int it = 50;
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < it; ++i)
                    for (int64_t r = 0; r < FF; ++r) {
                        float gg, uu;
                        tc->vec_dot((int) H, &gg, 0, blob.data() + r * f.gu_row, 0, a[0], 0, 1);
                        tc->vec_dot((int) H, &uu, 0, blob.data() + f.up_off + r * f.gu_row, 0, a[0], 0, 1);
                    }
                auto t1 = std::chrono::steady_clock::now();
                usg = std::chrono::duration<double, std::micro>(t1 - t0).count() / it;
            }
            // one multi-token kernel: parity against ggml's vec_dot, then one-token and NT-token throughput
            auto check = [&](const char* tag, bool is512) {
                std::vector<float> g((size_t) NT * FF);
                float* gp[NT];
                for (int k = 0; k < NT; ++k) gp[k] = g.data() + k * FF;
                (is512 ? cpu::iq512_rows : cpu::iq256_rows)(f.gu_type, blob.data(), f.gu_row, (int) H, a, NT, gp, 0, (int) FF);
                const double rg = rel(g, gref);
                std::printf("          %s %s gate rows vs ggml vec_dot: rel %.2e\n",
                            ggml_type_name((ggml_type) f.gu_type), tag, rg);
                if (rg > 1e-5) {   // ggml's own vec_dot is the reference: above 1e-5 it is a real mismatch,
                    std::printf("          %s %s gate rows MISMATCH (rel %.2e > 1e-5)\n",
                                ggml_type_name((ggml_type) f.gu_type), tag, rg);   // not rounding (measured 3e-8)
                    ++failures;
                }
                float* f1[1] = {ffp[0]};
                const int it = 50;
                const auto gu = is512 ? cpu::iq512_gu_rows : cpu::iq256_gu_rows;
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < it; ++i) gu(f.gu_type, blob.data(), f.gu_row, f.up_off, (int) H, a, 1, f1, 0, (int) FF);
                auto t1 = std::chrono::steady_clock::now();
                for (int i = 0; i < it; ++i) gu(f.gu_type, blob.data(), f.gu_row, f.up_off, (int) H, a, NT, ffp, 0, (int) FF);
                auto t2 = std::chrono::steady_clock::now();
                const double us1 = std::chrono::duration<double, std::micro>(t1 - t0).count() / it;
                const double usn = std::chrono::duration<double, std::micro>(t2 - t1).count() / it;
                std::printf("          gate+up one token one thread: %s %.0f us (%.2f GB/s), ggml %.0f us (%.2f GB/s)\n",
                            tag, us1, 2.0 * f.up_off / us1 / 1e3, usg, 2.0 * f.up_off / usg / 1e3);
                std::printf("          gate+up %d tokens one thread: %s %.0f us vs ggml %d x %.0f us\n", NT, tag, usn, NT, usg);
            };
            // guarded: the binary runs on AVX-2 CPUs too, and IQ4_XS has no AVX-512 kernel (an empty switch)
            if (cpu::cpu_avx512_ok() && cpu::iq512_supported(f.gu_type)) check("avx512", true);
            if (cpu::cpu_avx2_ok() && cpu::iq256_supported(f.gu_type)) check("avx2", false);   // no AVX2: ggml-cpu only
        }
        for (int k = 0; k < NT; ++k) {
            cpu::native_quant_h(f, ff[k].data(), hq[k].data());
            hp[k] = hq[k].data();
            op[k] = got_c.data() + k * H;
        }
        cpu::native_down_rows(f, blob.data(), hp, NT, op, 0, (int) H);
        {   // #152: a token's rows must not depend on how many tokens share the call - each token alone (nt = 1)
            // against its row of the NT-token call above, bit for bit, gate/up and down
            size_t gu_diff = 0, dn_diff = 0;
            std::vector<float> one_ff(FF), one_out(H);
            // the NT-token gate/up rows again: the throughput loops above overwrote ff with other kernels' rows
            for (int k = 0; k < NT; ++k) ffp[k] = ff[k].data();
            cpu::native_gu_rows(f, blob.data(), a, NT, ffp, 0, (int) FF);
            for (int k = 0; k < NT; ++k) {
                const void* a1[1] = {a[k]};
                float* f1[1] = {one_ff.data()};
                cpu::native_gu_rows(f, blob.data(), a1, 1, f1, 0, (int) FF);
                for (int64_t r = 0; r < FF; ++r) gu_diff += std::memcmp(&one_ff[r], &ff[k][r], 4) != 0;
                const void* h1[1] = {hp[k]};
                float* o1[1] = {one_out.data()};
                cpu::native_down_rows(f, blob.data(), h1, 1, o1, 0, (int) H);
                for (int64_t r = 0; r < H; ++r) dn_diff += std::memcmp(&one_out[r], &got_c[k * H + r], 4) != 0;
            }
            std::printf("          width invariance (1 vs %d tokens): gate/up %zu, down %zu rows differ\n", NT,
                        gu_diff, dn_diff);
            if (gu_diff || dn_diff) ++failures;
        }
        // the trellis types (tq_cpu.hpp, which native_gu_rows / native_down_rows above used): every ISA this CPU has
        // against the scalar kernel BIT FOR BIT, and against ggml-cpu's vec_dot (float summation order only)
        for (int role = 0; role < 2; ++role) {
            const int type = role == 0 ? f.gu_type : f.d_type;
            if (!cpu::tq_supported(type)) continue;
            const int n = role == 0 ? (int) H : (int) FF, rows = role == 0 ? (int) FF : (int) H;
            const size_t rb = role == 0 ? f.gu_row : f.d_row;
            const uint8_t* w = blob.data() + (role == 0 ? 0 : f.down_off);
            const void* const* acts = role == 0 ? a : hp;
            std::vector<float> ref((size_t) NT * rows), got((size_t) NT * rows);
            float* rp[NT];
            float* gp[NT];
            for (int k = 0; k < NT; ++k) { rp[k] = ref.data() + (size_t) k * rows; gp[k] = got.data() + (size_t) k * rows; }
            cpu::tq_rows(cpu::TqIsa::scalar, type, w, rb, n, acts, NT, rp, 0, rows);
            const auto* tc = ggml_get_type_traits_cpu((ggml_type) type);
            double num = 0, den = 0;
            for (int k = 0; k < NT; ++k)
                for (int r = 0; r < rows; ++r) {
                    float v = 0;
                    tc->vec_dot(n, &v, 0, w + (size_t) r * rb, 0, acts[k], 0, 1);
                    num += std::pow((double) v - ref[(size_t) k * rows + r], 2);
                    den += (double) v * v;
                }
            const double rg = std::sqrt(num / (den + 1e-30));
            std::printf("          %s %s rows: scalar kernel vs ggml vec_dot rel %.2e", ggml_type_name((ggml_type) type),
                        role == 0 ? "gate" : "down", rg);
            if (rg > 1e-5) ++failures;
            for (cpu::TqIsa isa : {cpu::TqIsa::avx2, cpu::TqIsa::avx512}) {
                if (!cpu::tq_isa_ok(isa)) continue;
                cpu::tq_rows(isa, type, w, rb, n, acts, NT, gp, 0, rows);
                const bool same = std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) == 0;
                std::printf(" | %s %s", cpu::tq_isa_name(isa), same ? "bitwise equal" : "DIFFERENT");
                if (!same) ++failures;
            }
            {   // time: the scalar kernel against the best SIMD one this CPU has, same rows and tokens
                const int it = 30;
                const auto t0 = std::chrono::steady_clock::now();
                for (int k = 0; k < it; ++k) cpu::tq_rows(cpu::TqIsa::scalar, type, w, rb, n, acts, NT, rp, 0, rows);
                const auto t1 = std::chrono::steady_clock::now();
                const cpu::TqIsa best = cpu::tq_isa();
                for (int k = 0; k < it; ++k) cpu::tq_rows(best, type, w, rb, n, acts, NT, gp, 0, rows);
                const auto t2 = std::chrono::steady_clock::now();
                const double us0 = std::chrono::duration<double, std::micro>(t1 - t0).count() / it;
                const double us1 = std::chrono::duration<double, std::micro>(t2 - t1).count() / it;
                std::printf(" | %s %.0f us, scalar %.0f us (%.2fx)", cpu::tq_isa_name(best), us1, us0,
                            us1 > 0 ? us0 / us1 : 0.0);
            }
            std::printf("\n");
        }
        // UD-Q4_K_XL's formats: the multi-token AVX2 kernels (kq_avx2.cpp) against ggml-cpu's vec_dot per token,
        // BIT FOR BIT, for 1..8 tokens; then the time of 4 tokens (a verify window) both ways
        for (int role = 0; role < 2; ++role) {
            const int type = role == 0 ? f.gu_type : f.d_type;
            if (!cpu::cpu_avx2_ok() || !cpu::kq256_supported(type) || (role == 0 && type != 12)) continue;
            const int n = role == 0 ? (int) H : (int) FF, rows = role == 0 ? (int) FF : (int) H;
            const size_t rb = role == 0 ? f.gu_row : f.d_row;
            const uint8_t* w = blob.data() + (role == 0 ? 0 : f.down_off);
            const auto* tc = ggml_get_type_traits_cpu((ggml_type) type);
            const ggml_type at = tc->vec_dot_type;
            std::vector<std::vector<uint8_t>> acts(8, std::vector<uint8_t>(ggml_row_size(at, n)));
            std::mt19937 arng(77 + seed);
            std::normal_distribution<float> and_(0.f, 1.f);
            std::vector<float> xs((size_t) n);
            const void* ap[8];
            for (int t = 0; t < 8; ++t) {
                for (auto& v : xs) v = and_(arng);
                ggml_get_type_traits_cpu(at)->from_float(xs.data(), acts[t].data(), n);
                ap[t] = acts[t].data();
            }
            std::vector<float> ref((size_t) 8 * rows), got((size_t) 8 * rows);
            for (int t = 0; t < 8; ++t)
                for (int r = 0; r < rows; ++r) tc->vec_dot(n, &ref[(size_t) t * rows + r], 0, w + (size_t) r * rb, 0, ap[t], 0, 1);
            size_t differ = 0;
            for (int nt = 1; nt <= 8; ++nt) {
                float* op[8];
                for (int t = 0; t < nt; ++t) op[t] = got.data() + (size_t) t * rows;
                cpu::kq256_rows(type, w, rb, n, ap, nt, op, 0, rows);
                for (int t = 0; t < nt; ++t)
                    differ += std::memcmp(op[t], ref.data() + (size_t) t * rows, (size_t) rows * 4) != 0;
            }
            const int it = 30;
            float* op4[4] = {got.data(), got.data() + rows, got.data() + 2 * rows, got.data() + 3 * rows};
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i) cpu::kq256_rows(type, w, rb, n, ap, 4, op4, 0, rows);
            auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i)
                for (int r = 0; r < rows; ++r)
                    for (int t = 0; t < 4; ++t) tc->vec_dot(n, op4[t] + r, 0, w + (size_t) r * rb, 0, ap[t], 0, 1);
            auto t2 = std::chrono::steady_clock::now();
            const double us_k = std::chrono::duration<double, std::micro>(t1 - t0).count() / it;
            const double us_g = std::chrono::duration<double, std::micro>(t2 - t1).count() / it;
            std::printf("          %s %s rows, AVX2 multi-token vs ggml vec_dot: %zu of 36 token-sets differ in any bit; "
                        "4 tokens one thread %.0f us vs ggml %.0f us (%.2fx)\n", ggml_type_name((ggml_type) type),
                        role == 0 ? "gate" : "down", differ, us_k, us_g, us_g / us_k);
            if (differ) ++failures;
        }
        if (cpu::q2_native_kernels(f.d_type)) {
            // (b2) the GGUF-layout Q2_0 kernel the pool uses for Q2_0 down projections - the AVX-512 one
            // where the CPU has it, the AVX-2 one (q2_avx2.cpp) where it does not.  Calling the AVX-512
            // kernel unconditionally faults on a Zen 2/3 CPU.
            std::vector<cpu::ActQ> a2(NT);
            const cpu::ActQ* ap[NT];
            std::vector<float> alt((size_t) NT * H);
            float* altp[NT];
            for (int k = 0; k < NT; ++k) {
                cpu::act_quant_any(ff[k].data(), (int) FF, a2[k]);   // gated: AVX-512 or AVX-2 per CPU
                ap[k] = &a2[k];
                altp[k] = alt.data() + k * H;
            }
            cpu::q2_rows_any(blob.data() + f.down_off, f.d_row, (int) (FF / 64), ap, NT, altp, 0, (int) H);
            std::printf("          q2_0 %s down vs ggml down: rel %.2e\n",
                        cpu::cpu_avx512_ok() ? "AVX-512" : "AVX-2", rel(alt, got_c));
        }
        if (f.d_type == 20 && cpu::cpu_avx2_ok()) {
            // (b3) the IQ4_NL multi-token AVX-2 kernel the pool now uses for IQ4_NL down projections,
            // against ggml-cpu's single-token vec_dot on the SAME Q8_0 activations (h), plus timing.
            std::vector<float> alt((size_t) NT * H), refd((size_t) NT * H);
            float* altp[NT];
            for (int k = 0; k < NT; ++k) altp[k] = alt.data() + k * H;
            cpu::iq4nl256_down_rows(blob.data() + f.down_off, f.d_row, (int) FF, hp, NT, altp, 0, (int) H);
            const auto* tdc = ggml_get_type_traits_cpu((ggml_type) f.d_type);
            for (int k = 0; k < NT; ++k)
                for (int64_t r = 0; r < H; ++r)
                    tdc->vec_dot((int) FF, refd.data() + k * H + r, 0,
                                 blob.data() + f.down_off + (size_t) r * f.d_row, 0, hp[k], 0, 1);
            const double rdn = rel(alt, refd);
            std::printf("          iq4_nl AVX-2 multi-token down vs ggml down: rel %.2e\n", rdn);
            if (rdn > 1e-5) {   // measured 1e-7; same reasoning as the gate/up rows above
                std::printf("          iq4_nl down MISMATCH (rel %.2e > 1e-5)\n", rdn);
                ++failures;
            }
            const int it = 50;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i)
                cpu::iq4nl256_down_rows(blob.data() + f.down_off, f.d_row, (int) FF, hp, NT, altp, 0, (int) H);
            const auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i)
                for (int k = 0; k < NT; ++k)
                    for (int64_t r = 0; r < H; ++r)
                        tdc->vec_dot((int) FF, refd.data() + k * H + r, 0,
                                    blob.data() + f.down_off + (size_t) r * f.d_row, 0, hp[k], 0, 1);
            const auto t2 = std::chrono::steady_clock::now();
            const double usn = std::chrono::duration<double, std::micro>(t1 - t0).count() / it;
            const double usg = std::chrono::duration<double, std::micro>(t2 - t1).count() / it;
            std::printf("          down %d tokens one thread: avx2 %.0f us (%.2f GB/s) vs ggml %.0f us (%.2f GB/s)\n",
                        NT, usn, (double) (f.bytes - f.down_off) / usn / 1e3, usg,
                        (double) (f.bytes - f.down_off) / usg / 1e3);
        }
    }
    // (c) the GPU: one group holding the NT entries
    {
        auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, H, FF);
        if (f.had_d) { L.had_block = f.had_block; L.d_signs = strata::kernels::hadamard_device_signs(f.d_signs, (int) FF); }
        void *dblob, *dx, *dxq, *dscr;
        float* dout;
        unsigned long long* dptr;
        int32_t *dstart, *dn, *ddst, *dtok;
        cudaMalloc(&dblob, blob.size());
        cudaMalloc(&dx, x.size() * 4);
        cudaMalloc(&dxq, (size_t) NT * H / 32 * 36);
        cudaMalloc(&dscr, strata::kernels::native_expert_scratch_bytes(NT, FF));
        cudaMalloc((void**) &dout, (size_t) NT * H * 4);
        cudaMalloc((void**) &dptr, 8);
        cudaMalloc((void**) &dstart, 8);
        cudaMalloc((void**) &dn, 4);
        cudaMalloc((void**) &ddst, NT * 4);
        cudaMalloc((void**) &dtok, NT * 4);
        cudaMemcpy(dblob, blob.data(), blob.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        const unsigned long long p = (unsigned long long) dblob;
        const int32_t st[2] = {0, NT}, one = 1, idx[NT] = {0, 1, 2};
        cudaMemcpy(dptr, &p, 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dstart, st, 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dn, &one, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(ddst, idx, NT * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dtok, idx, NT * 4, cudaMemcpyHostToDevice);
        if (f.had_gu) {   // APR: the GPU rotation of x (in place here), then its floats against the CPU's
            strata::kernels::hadamard_rows((const float*) dx, (float*) dx, NT, H, f.had_block,
                                           strata::kernels::hadamard_device_signs(f.gu_signs, (int) H), s);
            std::vector<float> gx(x.size());
            cudaStreamSynchronize(s);
            cudaMemcpy(gx.data(), dx, gx.size() * 4, cudaMemcpyDeviceToHost);
            const bool same = std::memcmp(gx.data(), xr.data(), gx.size() * 4) == 0;
            std::printf("          gpu hadamard_rows vs cpu hadamard_rotate: %s (rel %.2e)\n",
                        same ? "bitwise equal" : "differ", rel(gx, xr));
            if (!(rel(gx, xr) < 1e-6)) ++failures;
            // the f16 form (prompt path) on the unrotated x: within fp16 rounding of the float rotation
            std::vector<uint16_t> xh(x.size()), xh_out(x.size());
            for (size_t i = 0; i < x.size(); ++i) xh[i] = ggml_fp32_to_fp16(x[i]);
            uint16_t* dxh = nullptr;
            cudaMalloc((void**) &dxh, xh.size() * 2);
            cudaMemcpy(dxh, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice);
            strata::kernels::hadamard_rows_f16(dxh, NT, H, f.had_block,
                                               strata::kernels::hadamard_device_signs(f.gu_signs, (int) H), s);
            cudaStreamSynchronize(s);
            cudaMemcpy(xh_out.data(), dxh, xh.size() * 2, cudaMemcpyDeviceToHost);
            cudaFree(dxh);
            std::vector<float> xf(x.size());
            for (size_t i = 0; i < xf.size(); ++i) xf[i] = ggml_fp16_to_fp32(xh_out[i]);
            std::printf("          gpu hadamard_rows_f16 vs cpu float rotation: rel %.2e\n", rel(xf, xr));
            if (!(rel(xf, xr) < 2e-3)) ++failures;
        }
        strata::kernels::quantize_q8_1_rows((const float*) dx, NT, H, dxq, s);
        strata::kernels::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, 1, NT, dxq, dscr, dout, s);
        cudaStreamSynchronize(s);
        cudaMemcpy(got_g.data(), dout, got_g.size() * 4, cudaMemcpyDeviceToHost);
        {   // the kernels that decode a weight part once for the NT entries, bitwise vs the per-entry ones (#242;
            // the formats without a decode-once split take the per-entry kernels either way)
            std::vector<float> old_g(got_g.size());
            const bool was = strata::kernels::iq_old_kernels();
            strata::kernels::iq_set_old_kernels(!was);
            strata::kernels::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, 1, NT, dxq, dscr, dout, s);
            strata::kernels::iq_set_old_kernels(was);
            cudaStreamSynchronize(s);
            cudaMemcpy(old_g.data(), dout, old_g.size() * 4, cudaMemcpyDeviceToHost);
            const bool same = std::memcmp(old_g.data(), got_g.data(), got_g.size() * 4) == 0;
            std::printf("          gpu decode-once vs per-entry kernels: %s\n", same ? "bitwise equal" : "MISMATCH");
            if (!same) ++failures;
        }
        {   // (d) the GPU dequantizers (prompt path: iq_dequant_f32/_f16; embedding: iq_embed_rows) against to_float
            float* dq = nullptr;
            uint16_t* dh = nullptr;
            int32_t* dtk = nullptr;
            cudaMalloc((void**) &dq, (size_t) FF * H * 4);
            cudaMalloc((void**) &dh, (size_t) FF * H * 2);
            cudaMalloc((void**) &dtk, (size_t) FF * 4);
            std::vector<float> got_dq((size_t) FF * H);
            std::vector<uint16_t> got_h((size_t) FF * H);
            std::vector<int32_t> tk((size_t) FF);
            for (int64_t r = 0; r < FF; ++r) tk[(size_t) r] = (int32_t) (FF - 1 - r);   // rows in reverse order
            cudaMemcpy(dtk, tk.data(), tk.size() * 4, cudaMemcpyHostToDevice);
            double dq_err = 0.0;
            size_t differ = 0, h_differ = 0;
            const struct { const uint8_t* src; int type; const std::vector<float>* want; } mats[3] = {
                {(const uint8_t*) dblob, f.gu_type, &G}, {(const uint8_t*) dblob + f.up_off, f.gu_type, &U},
                {(const uint8_t*) dblob + f.down_off, f.d_type, &D}};
            for (const auto& m : mats) {
                strata::kernels::iq_dequant_f32(m.type, m.src, FF * H, dq, s);
                strata::kernels::iq_dequant_f16(m.type, m.src, FF * H, dh, s);
                cudaStreamSynchronize(s);
                cudaMemcpy(got_dq.data(), dq, got_dq.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(got_h.data(), dh, got_h.size() * 2, cudaMemcpyDeviceToHost);
                dq_err = (std::max)(dq_err, rel(got_dq, *m.want));
                for (size_t i = 0; i < got_dq.size(); ++i) {
                    differ += std::memcmp(&got_dq[i], &(*m.want)[i], 4) != 0;
                    h_differ += got_h[i] != ggml_fp32_to_fp16(got_dq[i]);
                }
            }
            // the gate matrix as an embedding table of FF rows of H values
            strata::kernels::iq_embed_rows(f.gu_type, dblob, f.gu_row, dtk, FF, H, dq, s);
            cudaStreamSynchronize(s);
            cudaMemcpy(got_dq.data(), dq, got_dq.size() * 4, cudaMemcpyDeviceToHost);
            size_t emb_differ = 0;
            for (int64_t r = 0; r < FF; ++r)
                emb_differ += std::memcmp(got_dq.data() + r * H, G.data() + (FF - 1 - r) * H, (size_t) H * 4) != 0;
            cudaFree(dq); cudaFree(dh); cudaFree(dtk);
            std::printf("          GPU dequant vs ggml to_float: rel %.2e, %zu of %lld values differ in any bit; fp16 of "
                        "it: %zu differ; embedding rows: %zu of %lld differ\n", dq_err, differ, (long long) (3 * FF * H),
                        h_differ, emb_differ, (long long) FF);
            // exact where there is one product (Q8_0), a rounding at most elsewhere (fused multiply-adds)
            const bool exact = f.gu_type == 8 && f.d_type == 8;
            if (!(dq_err < 1e-6) || h_differ || (exact && (differ || emb_differ))) ++failures;
        }
        cudaFree(dblob); cudaFree(dx); cudaFree(dxq); cudaFree(dscr); cudaFree(dout); cudaFree(dptr);
        cudaFree(dstart); cudaFree(dn); cudaFree(ddst); cudaFree(dtok);
    }
    const double ec = rel(got_c, ref), eg = rel(got_g, ref), ecg = rel(got_c, got_g);
    const bool ok = ec < 3e-2 && eg < 3e-2 && std::isfinite(ec) && std::isfinite(eg);
    std::printf("%-9s %-8s/%-7s blob %8zu  cpu rel %.2e  gpu rel %.2e  cpu-gpu %.2e  %s\n", label.c_str(),
                ggml_type_name((ggml_type) f.gu_type), ggml_type_name((ggml_type) f.d_type), f.bytes, ec, eg, ecg,
                ok ? "ok" : "FAIL");
    return failures + (ok ? 0 : 1);
}

int type_of(std::string name) {
    for (auto& c : name) c = (char) std::tolower((unsigned char) c);
    // The trellis types sit above GGML_TYPE_COUNT on an upstream ggml (the overlay adds them to ggml-cuda, not to the
    // type table), so ggml_type_name cannot resolve them: name them here, as the engine's own type ids do.
    static const struct { const char* n; int t; } trellis[] = {{"tq2_t", 144}, {"tqk6", 145}, {"tqk7", 146},
                                                              {"tqp5", 147}, {"tqp6", 148}, {"tqp7", 149}};
    for (const auto& e : trellis) if (name == e.n) return e.t;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name((ggml_type) t);
        if (n == nullptr) continue;
        std::string m(n);
        for (auto& c : m) c = (char) std::tolower((unsigned char) c);
        if (m == name) return t;
    }
    return -1;
}

// Random weights (rows of different magnitudes) quantized by ggml into the blob layout.
std::vector<uint8_t> synthetic_blob(const cpu::NativeFmt& f, int seed) {
    std::vector<uint8_t> blob(f.bytes);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto quant = [&](int type, int64_t rows, int64_t cols, uint8_t* dst) {
        std::vector<float> w((size_t) (rows * cols));
        for (int64_t r = 0; r < rows; ++r) {
            const float scale = 0.02f * (0.5f + (float) (r % 7) / 7.0f);
            for (int64_t c = 0; c < cols; ++c) w[(size_t) (r * cols + c)] = scale * nd(rng);
        }
        const std::vector<float> imatrix((size_t) cols, 1.0f);
        ggml_quantize_chunk((ggml_type) type, w.data(), dst, 0, rows, cols,
                            ggml_quantize_requires_imatrix((ggml_type) type) ? imatrix.data() : nullptr);
    };
    quant(f.gu_type, FF, H, blob.data());
    quant(f.gu_type, FF, H, blob.data() + f.up_off);
    quant(f.d_type, H, FF, blob.data() + f.down_off);
    if (f.tail_off) {   // NVFP4's {s_gate, s_up, s_down, 0}: non-unit for an NVFP4 projection, 1.0 for another format
        const bool g4 = f.gu_type == 40, d4 = f.d_type == 40;
        const float tail[4] = {g4 ? 0.75f : 1.f, g4 ? 1.25f : 1.f, d4 ? 0.5f : 1.f, 0.f};
        std::memcpy(blob.data() + f.tail_off, tail, sizeof tail);
    }
    return blob;
}

// The Q5_1 min term.  Activations x = d * (q + 0.45) round to q, so each q8_1 block's sum of the ORIGINAL values
// (`ds.y`, what llama.cpp CUDA multiplies the min by) exceeds d * sum(q) by 0.45 * 32 * d.  The GPU dot must take
// the quantized sum, as ggml-cpu's vec_dot does: equal to ggml-cpu on the same blocks with s = d * sum(q), and far
// from the ds.y variant.  Weights with large mins make the difference visible.
int check_q5_1_min(cudaStream_t s) {
    constexpr int n = 640, rows = 64;
    ggml_cpu_init();   // the fp16 tables vec_dot uses (the other modes get this through native_fmt)
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> qd(-100, 100);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> w((size_t) rows * n), x(n);
    for (auto& v : w) v = 0.3f + 0.05f * nd(rng);        // positive weights: a large min per block
    const float d = 0.01f;
    for (auto& v : x) v = d * ((float) qd(rng) + 0.45f);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q5_1, n);
    std::vector<uint8_t> wq((size_t) rows * row_bytes);
    ggml_quantize_chunk(GGML_TYPE_Q5_1, w.data(), wq.data(), 0, rows, n, nullptr);
    // the GPU: its own q8_1 quantizer, then iq_mmvq on Q5_1
    void *dw, *dx, *dxq;
    float* dy;
    cudaMalloc(&dw, wq.size());
    cudaMalloc(&dx, n * 4);
    cudaMalloc(&dxq, (size_t) n / 32 * 36);
    cudaMalloc((void**) &dy, rows * 4);
    cudaMemcpy(dw, wq.data(), wq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice);
    strata::kernels::quantize_q8_1_rows((const float*) dx, 1, n, dxq, s);
    strata::kernels::iq_mmvq(GGML_TYPE_Q5_1, dw, dxq, dy, n, rows, 1, s);
    cudaStreamSynchronize(s);
    std::vector<float> gpu(rows);
    std::vector<uint8_t> xq((size_t) n / 32 * 36);
    cudaMemcpy(gpu.data(), dy, rows * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(xq.data(), dxq, xq.size(), cudaMemcpyDeviceToHost);
    cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dy);
    // ggml-cpu on the GPU's own blocks: once with s = d * sum(q) (the quantized sum), once with s = ds.y as stored
    struct Q81 { ggml_fp16_t d, s; int8_t qs[32]; };   // ggml's block_q8_1: {d, s} halves, 32 codes
    static_assert(sizeof(Q81) == 36, "block_q8_1 layout");
    auto* blocks = (Q81*) xq.data();
    std::vector<Q81> qsum(blocks, blocks + n / 32);
    double shift = 0.0;
    for (auto& b : qsum) {
        int sq = 0;
        for (int i = 0; i < 32; ++i) sq += b.qs[i];
        const float dd = ggml_fp16_to_fp32(b.d);
        shift = (std::max)(shift, (double) std::fabs(ggml_fp16_to_fp32(b.s) - dd * sq));
        b.s = ggml_fp32_to_fp16(dd * (float) sq);
    }
    const auto* tc = ggml_get_type_traits_cpu(GGML_TYPE_Q5_1);
    std::vector<float> want(rows), stored(rows);
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row = wq.data() + (size_t) r * row_bytes;
        tc->vec_dot(n, &want[r], 0, row, 0, qsum.data(), 0, 1);
        tc->vec_dot(n, &stored[r], 0, row, 0, blocks, 0, 1);
    }
    const double e_q = rel(gpu, want), e_s = rel(gpu, stored);
    // the quantized sum goes through fp16 in ggml-cpu's block (s is a half), so 1e-3 rather than bitwise
    const bool ok = e_q < 1e-3 && e_s > 10 * e_q && shift > 0.0;
    std::printf("q5_1 min   row 0: GPU %.6g, ggml-cpu (quantized sum) %.6g, (stored sum) %.6g\n", gpu[0], want[0],
                stored[0]);
    std::printf("q5_1 min   GPU vs ggml-cpu with s = d*sum(q): rel %.2e; vs s = sum(x) (llama.cpp CUDA's): rel %.2e "
                "(largest per-block sum shift %.3g)  %s\n", e_q, e_s, shift, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
// #290: the BF16 token embedding (--embd-gguf) - iq_embed_rows and iq_dequant_f32 on a random BF16 table against
// the exact widening (bits << 16), every bit; rows gathered out of order, with repeats
int check_bf16_embd(cudaStream_t s) {
    constexpr int kBf16 = 30;
    const int64_t H = 2560, V = 61, NT = 97;
    int failures = 0;
    if (!strata::kernels::embed_type_supported(kBf16) || strata::kernels::iq_supported(kBf16) ||
        strata::kernels::iq_row_bytes(kBf16, H) != (size_t) H * 2) {
        std::printf("bf16 embedding: type support / row bytes wrong\n");
        return 1;
    }
    std::mt19937 rng(290);
    std::vector<uint16_t> table((size_t) (V * H));
    for (auto& v : table) {   // any bit pattern but NaN (a NaN's payload is not what the test is about)
        do v = (uint16_t) (rng() & 0xffff); while ((v & 0x7f80) == 0x7f80 && (v & 0x7f));
    }
    std::vector<int32_t> tok((size_t) NT);
    for (auto& t : tok) t = (int32_t) (rng() % V);
    void* dt = nullptr;
    int32_t* dtok = nullptr;
    float* dout = nullptr;
    cudaMalloc(&dt, table.size() * 2);
    cudaMalloc((void**) &dtok, tok.size() * 4);
    const int64_t out_rows = NT > V ? NT : V;   // the gathered rows (NT) and the whole table (V) share the buffer
    cudaMalloc((void**) &dout, (size_t) (out_rows * H) * 4);
    cudaMemcpy(dt, table.data(), table.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dtok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice);
    auto widen = [](uint16_t b) { const uint32_t u = (uint32_t) b << 16; float f; std::memcpy(&f, &u, 4); return f; };
    std::vector<float> got((size_t) (out_rows * H));
    strata::kernels::iq_embed_rows(kBf16, dt, (size_t) H * 2, dtok, NT, H, dout, s);
    cudaStreamSynchronize(s);
    cudaMemcpy(got.data(), dout, (size_t) (NT * H) * 4, cudaMemcpyDeviceToHost);
    size_t rows_differ = 0;
    for (int64_t r = 0; r < NT; ++r)
        for (int64_t d = 0; d < H; ++d) {
            const float want = widen(table[(size_t) (tok[(size_t) r] * H + d)]);
            if (std::memcmp(&got[(size_t) (r * H + d)], &want, 4) != 0) { ++rows_differ; break; }
        }
    strata::kernels::iq_dequant_f32(kBf16, dt, V * H, dout, s);
    cudaStreamSynchronize(s);
    cudaMemcpy(got.data(), dout, table.size() * 4, cudaMemcpyDeviceToHost);
    size_t values_differ = 0;
    for (size_t i = 0; i < table.size(); ++i) {
        const float want = widen(table[i]);
        values_differ += std::memcmp(&got[i], &want, 4) != 0;
    }
    cudaFree(dt); cudaFree(dtok); cudaFree(dout);
    std::printf("bf16 embedding: %lld gathered rows, %zu differ; dequant of %lld values, %zu differ in any bit\n",
                (long long) NT, rows_differ, (long long) (V * H), values_differ);
    if (rows_differ || values_differ) ++failures;
    return failures;
}

// APR: a GGUF's token_embd as the engine gathers it (iq_embed_rows, any embed_type_supported type: Gyro-S's Q6_K),
// `n` random rows against ggml's to_float, bit for bit; then iq_dequant_f32 of one row
int check_gguf_embd(const char* path, int n, cudaStream_t s) {
    const strata::GgufModel model(strata::gguf_split_paths(path));
    size_t at = 0;
    const strata::TensorInfo* t = model.find("token_embd.weight", &at);
    if (!t || t->shape.size() != 2 || t->shape[0] != (uint64_t) H) { std::printf("%s: no token_embd [2560, V]\n", path); return 1; }
    const int type = (int) t->type;
    const int64_t V = (int64_t) t->shape[1];
    if (!strata::kernels::embed_type_supported(type)) { std::printf("token_embd type %d: no GPU dequantizer\n", type); return 1; }
    const size_t rb = strata::kernels::iq_row_bytes(type, H);
    if (rb != ggml_row_size((ggml_type) type, H)) { std::printf("row bytes %zu != ggml %zu\n", rb, ggml_row_size((ggml_type) type, H)); return 1; }
    const uint8_t* table = model.shard(at).tensor_data(*t);
    std::mt19937 rng(17);
    std::vector<int32_t> tok((size_t) n);
    for (auto& v : tok) v = (int32_t) (rng() % (uint64_t) V);
    tok[0] = 0; tok[(size_t) n - 1] = (int32_t) (V - 1);
    // the rows the test reads, packed contiguously as a small table (the engine maps the whole one)
    std::vector<uint8_t> small((size_t) n * rb);
    for (int i = 0; i < n; ++i) std::memcpy(small.data() + (size_t) i * rb, table + (size_t) tok[(size_t) i] * rb, rb);
    std::vector<int32_t> idx((size_t) n);
    for (int i = 0; i < n; ++i) idx[(size_t) i] = n - 1 - i;     // gathered in reverse
    void* dt = nullptr; int32_t* dtok = nullptr; float* dout = nullptr;
    cudaMalloc(&dt, small.size());
    cudaMalloc((void**) &dtok, (size_t) n * 4);
    cudaMalloc((void**) &dout, (size_t) n * H * 4);
    cudaMemcpy(dt, small.data(), small.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dtok, idx.data(), (size_t) n * 4, cudaMemcpyHostToDevice);
    strata::kernels::iq_embed_rows(type, dt, rb, dtok, n, H, dout, s);
    cudaStreamSynchronize(s);
    std::vector<float> got((size_t) n * H), want((size_t) H);
    cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
    const auto* tr = ggml_get_type_traits((ggml_type) type);
    size_t rows_differ = 0, values_differ = 0;
    for (int i = 0; i < n; ++i) {
        tr->to_float(small.data() + (size_t) idx[(size_t) i] * rb, want.data(), H);
        const size_t d = (size_t) std::count_if(want.begin(), want.end(), [&, k = (size_t) 0](const float& w) mutable {
            return std::memcmp(&w, &got[(size_t) i * H + k++], 4) != 0; });
        values_differ += d;
        rows_differ += d != 0;
    }
    strata::kernels::iq_dequant_f32(type, dt, H, dout, s);
    cudaStreamSynchronize(s);
    cudaMemcpy(got.data(), dout, (size_t) H * 4, cudaMemcpyDeviceToHost);
    tr->to_float(small.data(), want.data(), H);
    const bool one = std::memcmp(got.data(), want.data(), (size_t) H * 4) == 0;
    cudaFree(dt); cudaFree(dtok); cudaFree(dout);
    std::printf("%s token_embd (%s, %lld rows): %d random rows gathered on the GPU, %zu rows / %zu values differ from "
                "ggml to_float; iq_dequant_f32 row: %s\n", path, ggml_type_name((ggml_type) type), (long long) V, n,
                rows_differ, values_differ, one ? "bitwise equal" : "DIFFERS");
    return (rows_differ || !one) ? 1 : 0;
}

// APR: a GGUF's PLE table through PleTable (the engine's reader; mapped and direct SSD I/O), `n` random rows of 16
// heads against ggml's to_float of the same bytes, bit for bit (Gyro-M's Q8_0 table; IQ4_NL for the others)
int check_gguf_ple(const char* path, int n) {
    const strata::GgufModel model(strata::gguf_split_paths(path));
    size_t at = 0;
    const strata::TensorInfo* t = model.find("per_layer_token_embd.weight", &at);
    if (!t) { std::printf("%s: no per_layer_token_embd.weight\n", path); return 1; }
    const std::string shard = model.shard(at).path();
    const uint8_t* table = model.shard(at).tensor_data(*t);
    const int type = (int) t->type;
    const size_t rb = ggml_row_size((ggml_type) type, 160);
    const uint64_t rows = t->shape[1];
    const auto* tr = ggml_get_type_traits((ggml_type) type);
    std::mt19937 rng(23);
    std::vector<uint32_t> r16((size_t) n * 16);
    for (auto& v : r16) v = (uint32_t) (rng() % rows);
    r16[0] = 0; r16.back() = (uint32_t) (rows - 1);
    int failures = 0;
    for (const auto mode : {strata::kernels::PleIo::Mmap, strata::kernels::PleIo::Direct}) {
        strata::kernels::PleTable ple;
        strata::kernels::PleIoOptions io;
        io.mode = mode;
        std::string err;
        if (!ple.open(shard, err, io)) { std::printf("PleTable open (%s): %s\n", mode == strata::kernels::PleIo::Direct ? "direct" : "mmap", err.c_str()); ++failures; continue; }
        std::vector<float> got(r16.size() * 160), want(160);
        if (!ple.gather_batch(r16.data(), (size_t) n, got.data(), err)) { std::printf("gather_batch: %s\n", err.c_str()); ++failures; continue; }
        size_t rows_differ = 0;
        for (size_t i = 0; i < r16.size(); ++i) {
            tr->to_float(table + (size_t) r16[i] * rb, want.data(), 160);
            rows_differ += std::memcmp(want.data(), got.data() + i * 160, 160 * 4) != 0;
        }
        std::printf("%s PLE table (%s via PleTable as %s, %s I/O): %zu random rows, %zu differ from ggml to_float\n",
                    path, ggml_type_name((ggml_type) type), ple.format(),
                    mode == strata::kernels::PleIo::Direct ? "direct" : "mmap", r16.size(), rows_differ);
        if (rows_differ) ++failures;
    }
    return failures;
}

// APR: the engine's start-up checks for a pack made from a published GGUF, without a GPU allocation: the expert
// table (native_experts.txt v5 + hadamard.txt), GPU expert kernels for every layer's formats, the GGUF expert
// offsets (check_experts_gguf), every quantized tensor the index serves from the GGUF has a native kernel (NativeDense
// projections, the head) or dequantizer (the embedding), and the PLE table opens with the engine's reader.
int check_load(const char* pack, const char* gguf) {
    int failures = 0;
    std::string err;
    auto fail = [&](const std::string& m) { std::printf("  FAIL %s\n", m.c_str()); ++failures; };
    if (!cpu::expert_layout_load(pack, 48, 512, err)) { fail("expert_layout_load: " + err); return failures; }
    const auto& lay = cpu::expert_layout();
    const auto& hs = cpu::hadamard_spec();
    int folded = 0;
    for (int64_t l = 0; l < (int64_t) lay.fmt.size(); ++l) {
        const auto& f = lay.fmt[(size_t) l];
        folded += f.had_gu && f.had_d;
        if (!strata::kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff))
            fail("layer " + std::to_string(l) + ": no GPU expert kernels for " + std::to_string(f.gu_type) + "/" +
                 std::to_string(f.d_type));
    }
    std::printf("  experts: native_experts.txt v%d, %zu layers (%s/%s ...), %.2f GiB; hadamard block %d, %d layers folded\n",
                lay.version, lay.fmt.size(), ggml_type_name((ggml_type) lay.fmt[0].gu_type),
                ggml_type_name((ggml_type) lay.fmt[0].d_type), lay.total / 1073741824.0, hs.block, folded);
    if (!strata::core::check_experts_gguf(gguf, lay, err)) fail("check_experts_gguf: " + err);
    else std::printf("  check_experts_gguf: ok\n");
    const std::vector<std::string> shards = strata::gguf_split_paths(gguf);
    std::set<std::string> served;
    if (!strata::core::NativeDense::served_names(shards, true, served, err)) fail("served_names: " + err);
    const strata::GgufModel model(shards);
    std::ifstream idx(std::string(pack) + "/index.txt");
    std::string line;
    int from_gguf = 0, ok_rows = 0;
    while (std::getline(idx, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string name, file, kind;
        ss >> name >> file >> kind;
        if (kind != "0") continue;                   // a dense.bin row
        ++from_gguf;
        size_t at = 0;
        const strata::TensorInfo* t = model.find(name, &at);
        if (!t) { fail(name + ": not in the GGUF"); continue; }
        bool ok = served.count(name) > 0;
        if (name == "token_embd.weight") ok = strata::kernels::embed_type_supported((int) t->type);
        if (name == "output.weight") ok = strata::kernels::native_mmvq_supported((int) t->type);
        if (!ok) fail(name + " (" + t->type_name() + "): served from the GGUF but no kernel takes it");
        else ++ok_rows;
    }
    std::printf("  index: %d tensors served from the GGUF, %d with a kernel (incl. token_embd %s, output %s)\n",
                from_gguf, ok_rows, model.find("token_embd.weight")->type_name(), model.find("output.weight")->type_name());
    size_t pat = 0;
    if (model.find("per_layer_token_embd.weight", &pat)) {
        strata::kernels::PleTable ple;
        strata::kernels::PleIoOptions io;   // the engine's default: direct SSD reads
        if (!ple.open(model.shard(pat).path(), err, io)) fail("PleTable: " + err);
        else std::printf("  PLE table: %s, %llu rows, direct I/O: ok\n", ple.format(), (unsigned long long) ple.rows());
    }
    return failures;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // keep the trail on a crash
    if (argc < 2) {
        std::fprintf(stderr, "usage: native_expert_parity <shard.gguf> [layer ...]\n"
                             "       native_expert_parity --synthetic GU/DOWN ...   (ggml type names, e.g. q4_K/q5_1)\n"
                             "       native_expert_parity --synthetic-hadamard GU/DOWN ...   (the same, Hadamard-folded: APR)\n"
                             "       native_expert_parity --q5_1-min\n"
                             "       native_expert_parity --bf16-embd\n"
                             "       native_expert_parity --load-check <pack> <model.gguf>  (start-up checks, no GPU load)\n"
                             "       native_expert_parity --gguf-embd <model.gguf> [rows]   (token_embd rows vs to_float)\n"
                             "       native_expert_parity --gguf-ple <model.gguf> [tokens]  (PLE rows via PleTable vs to_float)\n");
        return 2;
    }
    // #152's width check tests the opt-in rule (the multi-token kernels from one token on)
    if (std::getenv("STRATA_IQ_MT_MIN") == nullptr) {
#ifdef _WIN32
        _putenv_s("STRATA_IQ_MT_MIN", "1");
#else
        setenv("STRATA_IQ_MT_MIN", "1", 1);
#endif
    }
    int failures = 0;
    cudaStream_t s;
    cudaStreamCreate(&s);
    const std::string mode = argv[1];
    if (mode == "--load-check" && argc >= 4) {
        std::printf("load check: pack %s, GGUF %s\n", argv[2], argv[3]);
        failures += check_load(argv[2], argv[3]);
    } else if (mode == "--gguf-embd" && argc >= 3) {
        failures += check_gguf_embd(argv[2], argc > 3 ? std::atoi(argv[3]) : 300, s);
    } else if (mode == "--gguf-ple" && argc >= 3) {
        failures += check_gguf_ple(argv[2], argc > 3 ? std::atoi(argv[3]) : 20);
    } else if (mode == "--q5_1-min") {
        failures += check_q5_1_min(s);
    } else if (mode == "--bf16-embd") {
        failures += check_bf16_embd(s);
    } else if (mode == "--synthetic" || mode == "--synthetic-hadamard") {
        // --synthetic-hadamard: the same with the layer Hadamard-folded (APR, prism.hadamard): random signs, block 128
        const bool had = mode == "--synthetic-hadamard";
        static std::vector<float> sx((size_t) H), sh((size_t) FF);
        std::mt19937 srng(5);
        for (auto& v : sx) v = (srng() & 1) ? 1.f : -1.f;
        for (auto& v : sh) v = (srng() & 1) ? 1.f : -1.f;
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            const size_t slash = arg.find('/');
            const int gu = slash == std::string::npos ? -1 : type_of(arg.substr(0, slash));
            const int dn = slash == std::string::npos ? -1 : type_of(arg.substr(slash + 1));
            cpu::NativeFmt f;
            std::string err;
            // The synthetic check quantizes the blob with ggml and compares against ggml's to_float.  An upstream
            // ggml (the MMQ overlay) has no traits for the trellis types (they sit above GGML_TYPE_COUNT), so the
            // check needs a ggml that carries them (a build with -DSTRATA_GGML_DIR=<the agention llama.cpp>): skip it
            // here with a message instead of reading past the type table.
            if (gu >= (int) GGML_TYPE_COUNT || dn >= (int) GGML_TYPE_COUNT) {   // (ggml_type_name of an id past the table is not a null pointer: it reads past it)
                std::printf("%s: skipped (this ggml has no trellis type traits: the check needs STRATA_GGML_DIR=<a "
                            "llama.cpp with the trellis types>)\n", arg.c_str());
                continue;
            }
            if (gu < 0 || dn < 0 || !cpu::native_fmt(gu, dn, H, FF, f, err) ||
                !strata::kernels::native_expert_supported(gu, dn, H, FF)) {
                std::printf("%s: %s\n", arg.c_str(), err.empty() ? "not a pair the GPU expert kernels take" : err.c_str());
                ++failures;
                continue;
            }
            if (had) {
                f.had_block = 128;
                f.had_gu = f.had_d = true;
                f.gu_signs = sx.data();
                f.d_signs = sh.data();
            }
            failures += check_blob(f, synthetic_blob(f, i), i, had ? "hadamard" : "synthetic", s);
        }
    } else {
        const strata::GgufModel model(strata::gguf_split_paths(argv[1]));
        std::vector<int> layers;
        for (int i = 2; i < argc; ++i) layers.push_back(std::atoi(argv[i]));
        if (layers.empty()) layers = {0, 1, 2, 3, 20, 47};
        for (int l : layers) {
            const strata::TensorInfo* t[3] = {};
            const uint8_t* data[3] = {};
            const char* roles[3] = {"gate", "up", "down"};
            for (int r = 0; r < 3; ++r) {   // each role from the shard that holds it
                size_t at = 0;
                t[r] = model.find("blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight", &at);
                if (t[r]) data[r] = model.shard(at).tensor_data(*t[r]);
            }
            if (!t[0] || !t[1] || !t[2]) { std::printf("layer %d: no expert tensors\n", l); ++failures; continue; }
            cpu::NativeFmt f;
            std::string err;
            if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err)) {
                std::printf("layer %d: %s\n", l, err.c_str()); ++failures; continue;
            }
            std::vector<uint8_t> blob(f.bytes);
            std::memcpy(blob.data(), data[0] + (size_t) E * f.up_off, f.up_off);
            std::memcpy(blob.data() + f.up_off, data[1] + (size_t) E * f.up_off, f.up_off);
            std::memcpy(blob.data() + f.down_off, data[2] + (size_t) E * (f.bytes - f.down_off),
                        f.bytes - f.down_off);
            failures += check_blob(f, blob, l, "layer " + std::to_string(l), s);
        }
    }
    std::printf("native_expert_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
