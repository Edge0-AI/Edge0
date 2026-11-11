// src/kernels/expdec_bench.cpp - the VRAM experts of a decode verify window: the default grouped kernels against
// EXPDEC (STRATA_EXPDEC=1, docs/EXPERTDEC.md), bit for bit and in microseconds, per card, per T, per format pair.
//
//     build/expdec_bench [reps = 300] [device = -1 (all)] [groups = 20] [pairs = all] [choices = def,dec,dec:r2,dec:b4,dec:r2:b4] [shape = real]
//
// The model's routed expert formats at its geometry (n_embd 2560, n_ff 640): gate/up IQ2_XXS (16), IQ2_XS (17), IQ3_XXS
// (18), IQ3_S (21), IQ2_S (22); down IQ4_NL (20), Q2_0 (42).  `pairs` is a comma list of gu/down type ids ("21/20,16/42")
// or "all" (the 10 pairs).  `choices` is a comma list of kernel choices:
//   def                     the default kernels (STRATA_EXPDEC off); the reference for the bitwise check
//   dec[:r<n>][:d<n>][:b<n>][:w<n>][:x<n>][:g<n>][:h<n>][:m<n>]
//                           EXPDEC with r = gate/up rows per lane (1, 2), d = down rows per lane (1, 2, 4), b = the
//                           gate/up kernel's min blocks per SM (2 | 4), w = 1 EXPDEC for the down too (the engine's
//                           default is the gate/up only), x = 1 also for the IQ4_NL down, g / h = gate/up / down rows
//                           per block (0 = auto), m = the auto rule's minimum blocks; a missing field takes the
//                           engine's value (the environment's STRATA_EXPDEC_* or its default)
// `shape`: real = every token picks 10 experts out of `groups` distinct ones (as fp_expert_bench: T=1 10 groups x 1
// entry, T=2 20 x 1, T=4 20 groups with 40 entries); single = 10 T groups of one entry each (the same entries as `real`
// spread over distinct experts); both = the two, plus the decomposition at the end: per-expert cost a = (single T=4 -
// real T=2) / 20 and per-entry cost b = (real T=4 - real T=2) / 20, in us, for every choice (groups = 20 only).
//
// Per (device, pair, T, shape, choice): the output buffer against `def`'s, bit for bit (memcmp), and the median of
// 5 rounds of `reps` back-to-back `native_expert_grouped` calls with CUDA events: the whole call (gate/up + SwiGLU +
// q8_1 + down), then the gate/up phase and the down phase alone (native_expert_set_mode's phase, 3 rounds each).
// The experts rotate over a pool of >= 96 MB of distinct synthetic blobs (random bytes, sane fp16 block scales), so
// every call reads its weights cold from DRAM as the decode does each layer.
//
// One line per cell:  dev T shape gu/d choice all_us gu_us down_us GB/s pct identical|DIFFERENT   (GB/s = groups x blob
// bytes / all_us; pct = of the peak DRAM bandwidth 2 x memory clock x bus width / 8).  Lines starting with '#' are
// context.  Exit code 1 if any choice differs or a CUDA error occurred.  GPU time only.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace K = strata::kernels;

namespace {

constexpr int kEmbd = 2560, kFf = 640, kTop = 10, kMaxT = 4;
constexpr int kMaxEntries = kMaxT * kTop;

struct Pair { int gu, d; };
const Pair kModelPairs[] = { {16, 20}, {17, 20}, {18, 20}, {21, 20}, {22, 20}, {16, 42}, {17, 42}, {18, 42}, {21, 42}, {22, 42} };

const char* type_name(int t) {
    switch (t) {
        case 16: return "IQ2_XXS"; case 17: return "IQ2_XS"; case 18: return "IQ3_XXS"; case 21: return "IQ3_S";
        case 22: return "IQ2_S"; case 20: return "IQ4_NL"; case 42: return "Q2_0"; default: return "?";
    }
}
int qk_of(int t) { return t == 20 ? 32 : t == 42 ? 64 : 256; }

struct Choice {
    std::string name;
    bool dec = false;
    K::NativeExpertExpdecParams p;   // every field set: the engine's startup values, then the choice's fields
};
K::NativeExpertExpdecParams g_engine_params;   // the engine's values at startup (environment or defaults)

// "def" | "dec[:r<n>][:d<n>][:b<n>][:x<n>][:g<n>][:h<n>][:m<n>]"; false on anything else
bool parse_choice(const std::string& s, Choice& c) {
    c = Choice{};
    c.name = s;
    c.p = g_engine_params;
    if (s == "def") return true;
    if (s.rfind("dec", 0) != 0) return false;
    c.dec = true;
    size_t i = 3;
    while (i < s.size()) {
        if (s[i] != ':') return false;
        size_t j = s.find(':', i + 1);
        if (j == std::string::npos) j = s.size();
        if (j < i + 3) return false;
        const char key = s[i + 1];
        const int v = std::atoi(s.substr(i + 2, j - i - 2).c_str());
        switch (key) {
            case 'r': if (v != 1 && v != 2) return false; c.p.gu_rows_per_lane = v; break;
            case 'd': if (v != 1 && v != 2 && v != 4) return false; c.p.down_rows_per_lane = v; break;
            case 'b': if (v != 2 && v != 4) return false; c.p.gu_min_blocks_per_sm = v; break;
            case 'x': if (v != 0 && v != 1) return false; c.p.down_iq4_nl = v; break;
            case 'w': if (v != 0 && v != 1) return false; c.p.down = v; break;   // EXPDEC for the down (engine default 0)
            case 'g': if (v < 0) return false; c.p.gu_rows_per_block = v; break;
            case 'h': if (v < 0) return false; c.p.down_rows_per_block = v; break;
            case 'm': if (v <= 0) return false; c.p.min_blocks = v; break;
            default: return false;
        }
        i = j;
    }
    return true;
}

void apply_choice(const Choice& c) {
    K::native_expert_set_expdec(c.dec ? 1 : 0);
    K::native_expert_set_expdec_params(c.p);
}

bool g_cuda_error = false;

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    g_cuda_error = true;
    return false;
}

uint16_t sane_half(std::mt19937& rng) {   // a finite fp16 of magnitude ~2^-10 .. 2^-6
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

// One expert blob [gate rows | up rows | down rows]: random bytes (any qs / qh / signs / scales nibble / grid index /
// IQ4_NL nibble / Q2_0 pair is a valid code) with a sane fp16 `d` in every block (offset 0 of every format's block).
void fill_blob(uint8_t* b, const K::NativeExpertLayout& L, std::mt19937& rng) {
    for (size_t i = 0; i + 4 <= L.bytes; i += 4) {
        const uint32_t r = rng();
        std::memcpy(b + i, &r, 4);
    }
    const size_t gu_block = K::iq_row_bytes(L.gu_type, qk_of(L.gu_type)), d_block = K::iq_row_bytes(L.d_type, qk_of(L.d_type));
    const size_t gu_blocks = (size_t) 2 * (size_t) L.n_ff * (size_t) (L.n_embd / qk_of(L.gu_type));
    for (size_t k = 0; k < gu_blocks; ++k) {
        const uint16_t h = sane_half(rng);
        std::memcpy(b + k * gu_block, &h, 2);
    }
    const size_t d_blocks = (size_t) L.n_embd * (size_t) (L.n_ff / qk_of(L.d_type));
    for (size_t k = 0; k < d_blocks; ++k) {
        const uint16_t h = sane_half(rng);
        std::memcpy(b + L.down_off + k * d_block, &h, 2);
    }
}

struct Window {
    int groups = 0, entries = 0;
    std::vector<int32_t> start, tok, dst;   // group g: entries [start[g], start[g + 1]); tok = token, dst = out row
};

// real: T tokens, kTop experts each, `groups_req` distinct experts in all (capped at kTop * T, floored at kTop).
// Expert j is forced onto token j % T so that every group has at least one entry; the rest of each token's picks are
// random.
Window build_real(int T, int groups_req, std::mt19937& rng) {
    Window w;
    const int G = std::min(std::max(groups_req, kTop), kTop * T);
    std::vector<std::vector<int>> picks((size_t) G);
    for (int k = 0; k < T; ++k) {
        std::vector<char> in((size_t) G, 0);
        int cnt = 0;
        for (int j = k; j < G; j += T) { in[(size_t) j] = 1; ++cnt; }
        std::vector<int> rest;
        for (int j = 0; j < G; ++j)
            if (!in[(size_t) j]) rest.push_back(j);
        std::shuffle(rest.begin(), rest.end(), rng);
        for (size_t i = 0; cnt < kTop; ++i, ++cnt) in[(size_t) rest[i]] = 1;
        for (int j = 0; j < G; ++j)
            if (in[(size_t) j]) picks[(size_t) j].push_back(k);
    }
    w.groups = G;
    w.start.push_back(0);
    for (int g = 0; g < G; ++g) {
        for (int k : picks[(size_t) g]) {
            w.tok.push_back(k);
            w.dst.push_back((int32_t) w.dst.size());
        }
        w.start.push_back((int32_t) w.dst.size());
    }
    w.entries = (int) w.dst.size();
    return w;
}
// single: kTop T groups of one entry each (token g % T)
Window build_single(int T) {
    Window w;
    w.groups = kTop * T;
    w.start.push_back(0);
    for (int g = 0; g < w.groups; ++g) {
        w.tok.push_back(g % T);
        w.dst.push_back(g);
        w.start.push_back(g + 1);
    }
    w.entries = w.groups;
    return w;
}

struct DevMem {
    std::vector<void*> ptrs;
    ~DevMem() { for (void* p : ptrs) cudaFree(p); }
    void* alloc(size_t n) {
        void* p = nullptr;
        if (!ck(cudaMalloc(&p, n), "malloc")) return nullptr;
        ptrs.push_back(p);
        return p;
    }
};

struct StreamEvents {
    cudaStream_t s = nullptr;
    cudaEvent_t a = nullptr, b = nullptr;
    ~StreamEvents() {
        if (a) cudaEventDestroy(a);
        if (b) cudaEventDestroy(b);
        if (s) cudaStreamDestroy(s);
    }
};

// (device, pair gu, pair d, T, shape, choice) -> {all, gu, down} us
using Key = std::tuple<int, int, int, int, std::string, std::string>;
std::map<Key, std::array<double, 3>> g_times;

bool run_device(int dev, int reps, int groups_req, const std::vector<uint8_t>& pool, size_t P, size_t stride,
                const std::vector<float>& xh, const K::NativeExpertLayout& L, const Pair& pair,
                const std::vector<Choice>& choices, const std::vector<std::string>& shapes, long long& bad) {
    if (!ck(cudaSetDevice(dev), "set device")) return false;
    cudaDeviceProp pr{};
    cudaGetDeviceProperties(&pr, dev);
    std::printf("# device %d: %s (sm_%d%d, %d SMs, L2 %d KB), %d reps x 5 rounds, <= %d groups, pool %d blobs x %.2f MB\n", dev,
                pr.name, pr.major, pr.minor, pr.multiProcessorCount, pr.l2CacheSize / 1024, reps, groups_req, (int) P,
                (double) L.bytes / 1e6);
    int mclk_khz = 0, bus_bits = 0;
    cudaDeviceGetAttribute(&mclk_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
    const double peak = 2.0 * (double) mclk_khz * 1e3 * (double) bus_bits / 8.0;   // bytes / s
    std::printf("# device %d peak DRAM bandwidth %.1f GB/s (memory clock %d kHz, bus %d bit)\n", dev, peak / 1e9, mclk_khz, bus_bits);
    const int gmax = kMaxEntries;
    DevMem mem;
    auto* dpool = (uint8_t*) mem.alloc(pool.size());
    auto* dx = (float*) mem.alloc(xh.size() * sizeof(float));
    void* dxq = mem.alloc((size_t) kMaxT * kEmbd / 32 * 36);
    void* dscr = mem.alloc(K::native_expert_scratch_bytes(kMaxEntries, kFf));
    auto* dout = (float*) mem.alloc((size_t) kMaxEntries * kEmbd * sizeof(float));
    auto* dptr = (unsigned long long*) mem.alloc(P * (size_t) gmax * sizeof(unsigned long long));
    auto* dstart = (int32_t*) mem.alloc((size_t) (gmax + 1) * sizeof(int32_t));
    auto* dn = (int32_t*) mem.alloc(sizeof(int32_t));
    auto* ddst = (int32_t*) mem.alloc((size_t) kMaxEntries * sizeof(int32_t));
    auto* dtok = (int32_t*) mem.alloc((size_t) kMaxEntries * sizeof(int32_t));
    if (!dpool || !dx || !dxq || !dscr || !dout || !dptr || !dstart || !dn || !ddst || !dtok) return false;
    StreamEvents se;
    if (!ck(cudaStreamCreate(&se.s), "stream") || !ck(cudaEventCreate(&se.a), "event") ||
        !ck(cudaEventCreate(&se.b), "event"))
        return false;
    void* stream = (void*) se.s;
    if (!ck(cudaMemcpy(dpool, pool.data(), pool.size(), cudaMemcpyHostToDevice), "copy pool") ||
        !ck(cudaMemcpy(dx, xh.data(), xh.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x"))
        return false;
    K::quantize_q8_1_rows(dx, kMaxT, kEmbd, dxq, stream);
    if (!ck(cudaStreamSynchronize(se.s), "quantize")) return false;

    std::mt19937 rng(20261012u);
    bool ok = true;
    for (int T = 1; T <= kMaxT && ok; ++T) {
        for (const std::string& shape : shapes) {
            if (!ok) break;
            const Window w = shape == "single" ? build_single(T) : build_real(T, groups_req, rng);
            const int G = w.groups, NE = w.entries;
            // call i reads G experts starting at blob (i * G) % P: a fresh set until the pool wraps (>= 96 MB later)
            std::vector<unsigned long long> hp(P * (size_t) G);
            for (size_t i = 0; i < P; ++i)
                for (int g = 0; g < G; ++g)
                    hp[i * (size_t) G + (size_t) g] = (unsigned long long) (dpool + ((i * (size_t) G + (size_t) g) % P) * stride);
            ok = ck(cudaMemcpy(dptr, hp.data(), hp.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice), "copy ptrs") &&
                 ck(cudaMemcpy(dstart, w.start.data(), w.start.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy start") &&
                 ck(cudaMemcpy(dn, &G, sizeof(int32_t), cudaMemcpyHostToDevice), "copy n") &&
                 ck(cudaMemcpy(ddst, w.dst.data(), w.dst.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy dst") &&
                 ck(cudaMemcpy(dtok, w.tok.data(), w.tok.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy tok");
            if (!ok) break;
            std::printf("# T=%d %s: %d groups, %d entries, %.1f MB of weights per call\n", T, shape.c_str(), G, NE,
                        (double) G * (double) L.bytes / 1e6);
            auto call = [&](size_t i) {
                K::native_expert_grouped(L, dptr + (i % P) * (size_t) G, dstart, dn, ddst, dtok, G, NE, dxq, dscr, dout, stream, 0);
            };
            auto timed = [&](int phase, int rounds, double& out_us) {
                K::native_expert_set_mode(-1, phase);
                for (size_t i = 0; i < (phase == 0 ? 100u : 20u); ++i) call(i);
                std::vector<double> rs((size_t) rounds);
                for (double& r : rs) {
                    cudaEventRecord(se.a, se.s);
                    for (int i = 0; i < reps; ++i) call((size_t) i);
                    cudaEventRecord(se.b, se.s);
                    cudaEventSynchronize(se.b);
                    float ms = 0.0f;
                    ok = ok && ck(cudaEventElapsedTime(&ms, se.a, se.b), "elapsed") && ck(cudaGetLastError(), "kernel");
                    r = 1000.0 * (double) ms / reps;
                }
                std::sort(rs.begin(), rs.end());
                out_us = rs[(size_t) rounds / 2];
                K::native_expert_set_mode(-1, 0);
            };
            std::vector<float> ref, res((size_t) NE * kEmbd);
            for (size_t vi = 0; vi < choices.size() && ok; ++vi) {
                const Choice& c = choices[vi];
                apply_choice(c);
                K::native_expert_set_mode(-1, 0);
                ok = ck(cudaMemset(dout, 0, (size_t) NE * kEmbd * sizeof(float)), "memset out");
                call(0);
                ok = ok && ck(cudaStreamSynchronize(se.s), "run") && ck(cudaGetLastError(), "kernel") &&
                     ck(cudaMemcpy(res.data(), dout, res.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out");
                if (!ok) break;
                bool same = true;
                if (vi == 0) {
                    ref = res;
                    long long nonfinite = 0;
                    for (float f : ref) nonfinite += std::isfinite(f) ? 0 : 1;
                    if (nonfinite) std::printf("# T=%d: %s output has %lld non-finite values\n", T, c.name.c_str(), nonfinite);
                } else {
                    same = std::memcmp(ref.data(), res.data(), ref.size() * sizeof(float)) == 0;
                }
                double us[3] = {0.0, 0.0, 0.0};
                timed(0, 5, us[0]);
                timed(1, 3, us[1]);
                timed(2, 3, us[2]);
                const double gbps = (double) G * (double) L.bytes / (us[0] * 1e-6) / 1e9;
                std::printf("%d %d %s %d/%d %s %.1f %.1f %.1f %.1f %.1f%% %s\n", dev, T, shape.c_str(), L.gu_type, L.d_type,
                            c.name.c_str(), us[0], us[1], us[2], gbps, peak > 0.0 ? 100.0 * gbps * 1e9 / peak : 0.0,
                            same ? "identical" : "DIFFERENT");
                g_times[Key(dev, L.gu_type, L.d_type, T, shape, c.name)] = {us[0], us[1], us[2]};
                if (!same) ++bad;
            }
        }
    }
    Choice off;
    off.p = g_engine_params;
    apply_choice(off);
    K::native_expert_set_mode(-1, 0);
    return ok;
}

// "gu/d,gu/d,..." -> pairs; false on a malformed item
bool parse_pairs(const std::string& spec, std::vector<Pair>& out) {
    size_t i = 0;
    while (i < spec.size()) {
        size_t j = spec.find(',', i);
        if (j == std::string::npos) j = spec.size();
        const std::string item = spec.substr(i, j - i);
        i = j + 1;
        if (item.empty()) continue;
        const size_t s = item.find('/');
        if (s == std::string::npos || s == 0 || s + 1 >= item.size()) return false;
        Pair p;
        p.gu = std::atoi(item.substr(0, s).c_str());
        p.d = std::atoi(item.substr(s + 1).c_str());
        if (p.gu <= 0 || p.d <= 0) return false;
        out.push_back(p);
    }
    return !out.empty();
}

std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j + 1;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int reps = argc > 1 ? std::max(20, std::atoi(argv[1])) : 300;
    const int only = argc > 2 ? std::atoi(argv[2]) : -1;
    const int groups_req = argc > 3 ? std::min(kMaxEntries, std::max(kTop, std::atoi(argv[3]))) : 20;
    std::vector<Pair> pairs;
    if (argc > 4 && std::string(argv[4]) != "all") {
        if (!parse_pairs(argv[4], pairs)) {
            std::printf("expdec_bench: pairs '%s' is not a comma list of gu/down ggml type ids\n", argv[4]);
            return 1;
        }
    } else {
        pairs.assign(std::begin(kModelPairs), std::end(kModelPairs));
    }
    g_engine_params = K::native_expert_expdec_params();
    std::vector<Choice> choices;
    for (const std::string& s : split_commas(argc > 5 ? argv[5] : "def,dec,dec:r2,dec:b4,dec:r2:b4")) {
        Choice c;
        if (!parse_choice(s, c)) {
            std::printf("expdec_bench: choice '%s' is not def or dec[:r<1|2>][:d<1|2|4>][:b<2|4>][:x<0|1>][:g<n>][:h<n>][:m<n>]\n", s.c_str());
            return 1;
        }
        choices.push_back(c);
    }
    if (choices.empty() || choices[0].dec) {
        Choice d;
        parse_choice("def", d);
        choices.insert(choices.begin(), d);   // the reference first
    }
    std::vector<std::string> shapes;
    {
        const std::string sh = argc > 6 ? argv[6] : "real";
        if (sh == "both") shapes = {"real", "single"};
        else if (sh == "real" || sh == "single") shapes = {sh};
        else { std::printf("expdec_bench: shape '%s' is not real, single or both\n", sh.c_str()); return 1; }
    }
    int n_dev = 0;
    if (!ck(cudaGetDeviceCount(&n_dev), "device count") || n_dev == 0) {
        std::printf("expdec_bench: no CUDA device\n");
        return 1;
    }
    if (only >= n_dev) {
        std::printf("expdec_bench: device %d, only %d present\n", only, n_dev);
        return 1;
    }
    for (const Pair& p : pairs) {
        if (!K::native_expert_supported(p.gu, p.d, kEmbd, kFf)) {
            std::printf("expdec_bench: native_expert_supported(%d, %d, %d, %d) is false\n", p.gu, p.d, kEmbd, kFf);
            return 1;
        }
        const K::NativeExpertLayout L = K::native_expert_layout(p.gu, p.d, kEmbd, kFf);
        bool gu = false, down = false;
        K::native_expert_expdec_takes(L, gu, down);
        if (!gu || !down) std::printf("# warning: EXPDEC has no %s kernel for pair %d/%d: that phase runs the default\n",
                                      !gu ? "gate/up" : "down", p.gu, p.d);
    }
    for (const char* name : {"STRATA_NO_SUB16_GU", "STRATA_EXPERT_V2", "STRATA_OLD_IQ_MMVQ", "STRATA_EXP_MODE"}) {
        const char* v = std::getenv(name);
        if (v != nullptr && v[0] != '\0' && v[0] != '0') std::printf("# warning: %s is set, the kernel choice is bypassed\n", name);
    }
    K::iq_set_old_kernels(false);
    std::printf("# n_embd %d, n_ff %d, %d pair(s), choices:", kEmbd, kFf, (int) pairs.size());
    for (const Choice& c : choices) {
        if (c.dec) std::printf(" %s(gu R %d minb %d, down R %d iq4_nl %d, rb %d drb %d, min blocks %d)", c.name.c_str(),
                               c.p.gu_rows_per_lane, c.p.gu_min_blocks_per_sm, c.p.down_rows_per_lane, c.p.down_iq4_nl,
                               c.p.gu_rows_per_block, c.p.down_rows_per_block, c.p.min_blocks);
        else std::printf(" %s", c.name.c_str());
    }
    std::printf("\n# dev T shape gu/d choice all_us gu_us down_us GB/s pct identical|DIFFERENT\n");
    long long bad = 0;
    bool ok = true;
    for (const Pair& p : pairs) {
        if (!ok) break;
        const K::NativeExpertLayout L = K::native_expert_layout(p.gu, p.d, kEmbd, kFf);
        const size_t stride = (L.bytes + 255) / 256 * 256;
        const size_t P = std::max<size_t>((size_t) (96u << 20) / stride + 1, (size_t) 2 * (size_t) kMaxEntries);
        std::printf("# pair %d/%d: gate/up %s, down %s: blob %.2f MB (gu_row %d, d_row %d), pool %.0f MB\n", p.gu, p.d,
                    type_name(p.gu), type_name(p.d), (double) L.bytes / 1e6, (int) L.gu_row, (int) L.d_row,
                    (double) (P * stride) / 1e6);
        std::mt19937 rng(77u + (unsigned) (p.gu * 131 + p.d));
        std::vector<uint8_t> pool(P * stride, 0);
        for (size_t i = 0; i < P; ++i) fill_blob(pool.data() + i * stride, L, rng);
        std::vector<float> xh((size_t) kMaxT * kEmbd);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float& v : xh) v = nd(rng);
        for (int dev = 0; dev < n_dev; ++dev) {
            if (only >= 0 && dev != only) continue;
            ok = run_device(dev, reps, groups_req, pool, P, stride, xh, L, p, choices, shapes, bad) && ok;
        }
    }
    if (shapes.size() == 2 && groups_req == 20) {
        std::printf("\n# decomposition (us per layer call): a = per expert = (single T=4 - real T=2) / 20, b = per entry = (real T=4 - real T=2) / 20\n");
        std::printf("# dev gu/d choice a_all a_gu a_down b_all b_gu b_down\n");
        for (int dev = 0; dev < n_dev; ++dev) {
            if (only >= 0 && dev != only) continue;
            for (const Pair& p : pairs)
                for (const Choice& c : choices) {
                    auto f2 = g_times.find(Key(dev, p.gu, p.d, 2, "real", c.name));
                    auto f4 = g_times.find(Key(dev, p.gu, p.d, 4, "real", c.name));
                    auto s4 = g_times.find(Key(dev, p.gu, p.d, 4, "single", c.name));
                    if (f2 == g_times.end() || f4 == g_times.end() || s4 == g_times.end()) continue;
                    std::printf("%d %d/%d %s", dev, p.gu, p.d, c.name.c_str());
                    for (int k = 0; k < 3; ++k) std::printf(" %.2f", (s4->second[k] - f2->second[k]) / 20.0);
                    for (int k = 0; k < 3; ++k) std::printf(" %.2f", (f4->second[k] - f2->second[k]) / 20.0);
                    std::printf("\n");
                }
        }
    }
    std::printf("\nexpdec_bench: %s (%lld choice comparisons differ%s)\n", (bad || !ok || g_cuda_error) ? "FAILED" : "ok", bad,
                (!ok || g_cuda_error) ? ", CUDA error" : "");
    return (bad || !ok || g_cuda_error) ? 1 : 0;
}
