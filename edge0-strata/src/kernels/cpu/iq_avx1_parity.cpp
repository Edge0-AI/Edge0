// src/kernels/cpu/iq_avx1_parity.cpp - the AVX1 expert kernels (iq_avx1.cpp) against ggml-cpu's dots and
// against themselves at every token count.  No GPU, no model: random blocks (every grid index and sign
// pattern occurs) and random activations quantized the way the engine quantizes them.
//
//     iq_avx1_parity            IQ3_XXS/IQ3_S/IQ2_S/IQ4_XS gate/up rows and IQ4_NL/Q2_0 down rows:
//                               every row bit for bit the same kernel's one-token row (#152), and rel <= 1e-5
//                               against ggml-cpu's vec_dot (the scalar _generic the engine runs without these).
//     iq_avx1_parity --bench [--mb 128] [--nt 1,2,4] [--cpu N]
//                               ms per expert on one thread: ggml-cpu's per-token dot vs the iq128 kernel over
//                               --mb MB of expert blobs (more than any L3: the weights stream from DRAM as in
//                               decode), for the nt in --nt.
//
// Runs on any CPU with AVX (the kernels are compiled for SSE4.2 + AVX); on an AVX2 machine it checks the
// older-CPU path exactly as an AVX-only machine runs it (the engine's dispatch gate is separate: this calls
// the kernels directly).
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sched.h>
#endif

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kH = 2560, kFF = 640;   // the experts' geometry: n_embd, expert width (as iq_avx2_parity)
constexpr int kMaxT = 8;

uint64_t mix(uint64_t& s) {   // splitmix64
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

const ggml_type_traits_cpu* traits_cpu(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

// random bytes, then a sane fp16 scale at the front of every block (iq_avx2_parity's)
void fill_blocks(uint8_t* p, size_t bytes, int type, uint64_t& seed) {
    for (size_t i = 0; i < bytes; i += 8) {
        const uint64_t v = mix(seed);
        std::memcpy(p + i, &v, std::min<size_t>(8, bytes - i));
    }
    const size_t bs = ggml_type_size((ggml_type) type);
    for (size_t o = 0; o + bs <= bytes; o += bs) {
        const float d = (0.5f + (float) (mix(seed) >> 40) * (1.f / 16777216.f)) * 1e-3f;
        const ggml_fp16_t h = ggml_fp32_to_fp16(d);
        std::memcpy(p + o, &h, 2);
    }
}

void fill_blob(uint8_t* blob, const cpu::NativeFmt& f, uint64_t& seed) {
    fill_blocks(blob, f.down_off, f.gu_type, seed);                        // the gate and up rows
    fill_blocks(blob + f.down_off, f.bytes - f.down_off, f.d_type, seed);  // the down rows
}

struct Case {
    int gu, dn;
    const char* name;
};

const Case kCases[] = {
    {18, 20, "iq3_xxs/iq4_nl"}, {21, 20, "iq3_s/iq4_nl"}, {22, 20, "iq2_s/iq4_nl"}, {23, 20, "iq4_xs/iq4_nl"},
    {18, 42, "iq3_xxs/q2_0"},   {21, 42, "iq3_s/q2_0"},   {22, 42, "iq2_s/q2_0"},   {23, 42, "iq4_xs/q2_0"},
};

// one expert, nt tokens of quantized activations, and the reference rows
struct Fixture {
    cpu::NativeFmt f;
    std::vector<uint8_t> blob;
    std::vector<std::vector<uint8_t>> act, hq;          // per token
    std::vector<std::vector<float>> x, h;
    std::vector<float> ff_ref, out_ref;                 // [t][row]
};

void make_fixture(Fixture& fx, int gu, int dn, int nt, uint64_t& seed) {
    std::string err;
    if (!cpu::native_fmt(gu, dn, kH, kFF, fx.f, err)) { std::fprintf(stderr, "native_fmt: %s\n", err.c_str()); std::exit(1); }
    fx.blob.resize(fx.f.bytes);
    fill_blob(fx.blob.data(), fx.f, seed);
    fx.act.resize(nt); fx.hq.resize(nt); fx.x.resize(nt); fx.h.resize(nt);
    for (int t = 0; t < nt; ++t) {
        fx.act[t].resize(fx.f.act_bytes); fx.hq[t].resize(fx.f.h_bytes);
        fx.x[t].resize(kH); fx.h[t].resize(kFF);
        for (int i = 0; i < kH; ++i) fx.x[t][i] = (float) ((mix(seed) >> 40) * 0x1p-24 * 2.0 - 1.0);
        for (int i = 0; i < kFF; ++i) fx.h[t][i] = (float) ((mix(seed) >> 40) * 0x1p-24 * 2.0 - 1.0);
        cpu::native_quant_act(fx.f, fx.x[t].data(), fx.act[t].data());
        cpu::native_quant_h(fx.f, fx.h[t].data(), fx.hq[t].data());
    }
    // reference rows: ggml-cpu's own dots, per token (what the engine runs without the iq128 kernels)
    const ggml_vec_dot_t gu_dot = traits_cpu(gu)->vec_dot;
    const ggml_vec_dot_t dn_dot = traits_cpu(dn)->vec_dot;
    fx.ff_ref.resize((size_t) nt * kFF); fx.out_ref.resize((size_t) nt * kH);
    for (int t = 0; t < nt; ++t) {
        for (int r = 0; r < kFF; ++r) {
            float g = 0.f, u = 0.f;
            gu_dot(kH, &g, 0, fx.blob.data() + (size_t) r * fx.f.gu_row, 0, fx.act[t].data(), 0, 1);
            gu_dot(kH, &u, 0, fx.blob.data() + fx.f.up_off + (size_t) r * fx.f.gu_row, 0, fx.act[t].data(), 0, 1);
            fx.ff_ref[(size_t) t * kFF + r] = (g / (1.f + std::exp(-g))) * u;
        }
        for (int r = 0; r < kH; ++r) {
            float s = 0.f;
            dn_dot(kFF, &s, 0, fx.blob.data() + fx.f.down_off + (size_t) r * fx.f.d_row, 0, fx.hq[t].data(), 0, 1);
            fx.out_ref[(size_t) t * kH + r] = s;
        }
    }
}

int failures = 0;

void check(const char* what, const std::vector<float>& got, const std::vector<float>& ref, bool exact) {
    if (exact) {   // bit-exact (iq_avx2_parity's rows_differ)
        size_t d = 0;
        for (size_t i = 0; i < ref.size(); ++i) d += std::memcmp(&got[i], &ref[i], sizeof(float)) != 0;
        if (d) { std::printf("  FAIL %s: %zu of %zu rows differ\n", what, d, ref.size()); ++failures; }
        return;
    }
    // the repo's bar (iq_avx2_parity's rel): aggregate sum|diff| / sum|ref| <= 1e-5
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        num += std::fabs((double) got[i] - (double) ref[i]);
        den += std::fabs((double) ref[i]);
    }
    const double r = num / (den + 1e-30);
    if (!(r <= 1e-5)) { std::printf("  FAIL %s: rel %.3e > 1e-5 vs ggml-cpu\n", what, r); ++failures; }
}

// ---- the checks
void run_checks() {
    for (const Case& c : kCases) {
        uint64_t seed = 0x5EED1234u + (uint64_t) c.gu * 1000 + (uint64_t) c.dn;
        Fixture fx;
        make_fixture(fx, c.gu, c.dn, kMaxT, seed);
        const void* actp[kMaxT]; const void* hqp[kMaxT];
        std::vector<float> ff[kMaxT], out[kMaxT];
        float* ffp[kMaxT]; float* outp[kMaxT];
        for (int t = 0; t < kMaxT; ++t) {
            actp[t] = fx.act[t].data(); hqp[t] = fx.hq[t].data();
            ff[t].resize(kFF); out[t].resize(kH);
            ffp[t] = ff[t].data(); outp[t] = out[t].data();
        }
        // gate/up: nt = 1..8, every width bit-identical to that token's nt = 1 row (#152), all rel <= 1e-5 vs ggml
        std::vector<std::vector<float>> ff1(kMaxT), out1(kMaxT);
        for (int nt = 1; nt <= kMaxT; ++nt) {
            cpu::iq128_gu_rows(c.gu, fx.blob.data(), fx.f.gu_row, fx.f.up_off, kH, actp, nt, ffp, 0, kFF);
            for (int t = 0; t < nt; ++t) {
                char what[96];
                std::snprintf(what, sizeof what, "%s gu nt=%d vs ggml", c.name, nt);
                check(what, ff[t], std::vector<float>(fx.ff_ref.begin() + (size_t) t * kFF,
                                                      fx.ff_ref.begin() + (size_t) (t + 1) * kFF), false);
                if (nt == 1) ff1[t] = ff[t];
                else {
                    std::snprintf(what, sizeof what, "%s gu nt=%d token %d vs its nt=1 row", c.name, nt, t);
                    check(what, ff[t], ff1[t], true);
                }
            }
        }
        // down
        for (int nt = 1; nt <= kMaxT; ++nt) {
            cpu::iq128_down_rows(c.dn, fx.blob.data() + fx.f.down_off, fx.f.d_row, kFF, hqp, nt, outp, 0, kH);
            for (int t = 0; t < nt; ++t) {
                char what[96];
                std::snprintf(what, sizeof what, "%s down nt=%d vs ggml", c.name, nt);
                check(what, out[t], std::vector<float>(fx.out_ref.begin() + (size_t) t * kH,
                                                      fx.out_ref.begin() + (size_t) (t + 1) * kH), false);
                if (nt == 1) out1[t] = out[t];
                else {
                    std::snprintf(what, sizeof what, "%s down nt=%d token %d vs its nt=1 row", c.name, nt, t);
                    check(what, out[t], out1[t], true);
                }
            }
        }
        std::printf("  %-16s checked (gu %d, down %d)\n", c.name, c.gu, c.dn);
    }
}

// ---- the bench: ms per expert, the weights streamed from DRAM
void run_bench(int mb, const std::vector<int>& nts) {
    for (const Case& c : kCases) {
        uint64_t seed = 0xBEEF0000u + (uint64_t) c.gu * 1000 + (uint64_t) c.dn;
        Fixture fx;
        make_fixture(fx, c.gu, c.dn, 1, seed);
        const size_t per = fx.f.bytes;
        const int nexp = (int) ((size_t) mb * 1024 * 1024 / per);
        std::vector<uint8_t> blob((size_t) nexp * per);
        for (int e = 0; e < nexp; ++e) std::memcpy(blob.data() + (size_t) e * per, fx.blob.data(), per);
        // vary the blocks a little so the run is not one blob repeated (the same bytes hit the same caches)
        for (int e = 1; e < nexp; ++e) {
            uint64_t s2 = seed + (uint64_t) e * 7919;
            const size_t o = (size_t) e * per;
            for (size_t i = 0; i + 8 <= per; i += 128) {
                const uint64_t v = mix(s2);
                std::memcpy(blob.data() + o + i, &v, 8);   // keep block scales (first 2 bytes of each block)
            }
        }
        for (int nt : nts) {
            if (nt > kMaxT) continue;
            const void* actp[kMaxT]; const void* hqp[kMaxT];
            float* ffp[kMaxT]; float* outp[kMaxT];
            std::vector<std::vector<float>> ff(kMaxT, std::vector<float>(kFF)), out(kMaxT, std::vector<float>(kH));
            Fixture fx8;   // nt tokens of activations
            uint64_t s8 = seed + 0x1234;
            make_fixture(fx8, c.gu, c.dn, nt, s8);
            for (int t = 0; t < nt; ++t) {
                actp[t] = fx8.act[t].data(); hqp[t] = fx8.hq[t].data();
                ffp[t] = ff[t].data(); outp[t] = out[t].data();
            }
            const ggml_vec_dot_t gu_dot = traits_cpu(c.gu)->vec_dot, dn_dot = traits_cpu(c.dn)->vec_dot;
            // ggml-cpu's path (what the engine runs today on this CPU): per-token dots over every row
            auto t0 = Clock::now();
            for (int e = 0; e < nexp; ++e) {
                for (int r = 0; r < kFF; ++r) {
                    for (int t = 0; t < nt; ++t) {
                        float g = 0.f, u = 0.f;
                        gu_dot(kH, &g, 0, blob.data() + (size_t) e * per + (size_t) r * fx.f.gu_row, 0, (const uint8_t*) actp[t], 0, 1);
                        gu_dot(kH, &u, 0, blob.data() + (size_t) e * per + fx.f.up_off + (size_t) r * fx.f.gu_row, 0, (const uint8_t*) actp[t], 0, 1);
                        ff[t][r] = (g / (1.f + std::exp(-g))) * u;
                    }
                }
                for (int r = 0; r < kH; ++r) {
                    for (int t = 0; t < nt; ++t) {
                        float s = 0.f;
                        dn_dot(kFF, &s, 0, blob.data() + (size_t) e * per + fx.f.down_off + (size_t) r * fx.f.d_row, 0, (const uint8_t*) hqp[t], 0, 1);
                        out[t][r] = s;
                    }
                }
            }
            const double ms_ggml = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            t0 = Clock::now();
            for (int e = 0; e < nexp; ++e) {
                cpu::iq128_gu_rows(c.gu, blob.data() + (size_t) e * per, fx.f.gu_row, fx.f.up_off, kH, actp, nt, ffp, 0, kFF);
                cpu::iq128_down_rows(c.dn, blob.data() + (size_t) e * per + fx.f.down_off, fx.f.d_row, kFF, hqp, nt, outp, 0, kH);
            }
            const double ms_iq = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            std::printf("  %-16s nt=%d  ggml %8.2f ms/expert   iq128 %6.2f ms/expert   %.1fx\n",
                        c.name, nt, ms_ggml / nexp, ms_iq / nexp, ms_ggml / ms_iq);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    int mb = 128, cpu_pin = -1;
    std::vector<int> nts = {1, 2, 4};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bench") bench = true;
        else if (a == "--mb" && i + 1 < argc) mb = std::atoi(argv[++i]);
        else if (a == "--cpu" && i + 1 < argc) cpu_pin = std::atoi(argv[++i]);
        else if (a == "--nt" && i + 1 < argc) {
            nts.clear();
            const char* p = argv[++i];
            while (*p) { nts.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p == ',') ++p; }
        } else {
            std::printf("usage: iq_avx1_parity [--bench [--mb N] [--nt 1,2,4] [--cpu N]]\n");
            return 2;
        }
    }
#if !defined(_WIN32)
    if (cpu_pin >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu_pin, &set);
        if (sched_setaffinity(0, sizeof set, &set) != 0) std::fprintf(stderr, "pinning to CPU %d failed\n", cpu_pin);
    }
#endif
    if (!cpu::cpu_avx1_ok()) { std::printf("iq_avx1_parity: no AVX here, the kernels would not run\n"); return 0; }
    std::printf("iq_avx1_parity: %s\n", cpu::cpu_avx1_ok() ? "AVX1 kernels checked (128-bit lanes)" : "");
    if (bench) { run_bench(mb, nts); return failures != 0; }
    run_checks();
    std::printf(failures ? "iq_avx1_parity: %d FAILURES\n" : "iq_avx1_parity: all rows match\n", failures);
    return failures != 0;
}
