// src/kernels/native_expert_bench.cpp - the grouped native experts on real GGUF rows: one AMD layout against
// another, bit for bit, and each one's time.
//
//     build/native_expert_bench <shard1.gguf> <layer[,layer...]> <groups> <tokens per group> <ref mode> <mode> [iters]
//     build/native_expert_bench --nibble <shard.gguf> <layer[,layer...]> [--t 1,2,3,4,5,6] [--share 0,50] [--plans 200]
//                               [--trials 7] [--mode -1|0] [--variant -1|0|1]
//
// Modes are STRATA_EXP_MODE values.  A group is one expert (distinct blobs, ~2 MB each, so G >= 8 does not fit
// in L2); its entries read distinct tokens of an 8-token window.  The output must be bitwise equal to the
// reference mode's: the verify window's text depends on it.
//
// --nibble (STRATA_EXPERT_NIBBLE, strata/core/expert_nibble.hpp): the layer's 512 experts as GGUF blobs (P, the
// production kernels) and as 4-bit slots made by the engine's transcoder (N: native_expert_layout_nib); the verify
// window's whole expert step (gate/up + SwiGLU/q8_1 + down) and its gate/up phase alone, per (T, sharing) over
// `--plans` top-10 routings (0% or 50% of a token's experts shared with the previous token's), every plan's out rows
// compared byte for byte first; interleaved trials, medians.  `--mode`: both arms' layout (-1 the card's default, a
// GP100's mode 8; 0 the CUDA layout); `--variant`: the 4-bit gate/up kernel (-1 per format, 0 S, 1 P).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_nibble.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace K = strata::kernels;

int nibble_main(int argc, char** argv);

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc >= 2 && std::strcmp(argv[1], "--nibble") == 0) return nibble_main(argc, argv);
    if (argc < 7) {
        std::fprintf(stderr, "usage: native_expert_bench <shard1.gguf> <layer[,layer]> <groups> <tokens> <ref mode> <mode> [iters]\n");
        return 2;
    }
    strata::GgufFile gguf(argv[1]);
    std::vector<int> layers;
    {
        std::stringstream ss(argv[2]);
        std::string t;
        while (std::getline(ss, t, ',')) layers.push_back(std::atoi(t.c_str()));
    }
    const int G = std::atoi(argv[3]), T = std::atoi(argv[4]), mref = std::atoi(argv[5]), mode = std::atoi(argv[6]);
    const int iters = argc > 7 ? std::atoi(argv[7]) : 200;
    const int NTOK = 8;
    if (T < 1 || T > NTOK || G < 1) { std::fprintf(stderr, "tokens 1..8, groups >= 1\n"); return 2; }
    const int64_t H = 2560, FF_FULL = 640, FF = FF_FULL;   // Flash-Next's expert geometry
    int failures = 0;
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int l : layers) {
        const strata::TensorInfo* t[3] = {};
        const char* roles[3] = {"gate", "up", "down"};
        for (const auto& ti : gguf.tensors())
            for (int r = 0; r < 3; ++r)
                if (ti.name == "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight") t[r] = &ti;
        if (!t[0] || !t[1] || !t[2]) { std::printf("layer %d: no expert tensors\n", l); ++failures; continue; }
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err)) {
            std::printf("layer %d: %s\n", l, err.c_str()); ++failures; continue;
        }
        cpu::NativeFmt ff;    // the full expert's geometry, for strides into the GGUF
        if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF_FULL, ff, err)) {
            std::printf("layer %d: %s\n", l, err.c_str()); ++failures; continue;
        }
        const size_t dsz = f.bytes - f.down_off, dsz_full = ff.bytes - ff.down_off;
        const size_t drow = dsz / H, drow_full = dsz_full / H;
        std::vector<uint8_t> blobs((size_t) G * f.bytes);
        for (int g = 0; g < G; ++g) {
            const size_t E = (size_t) ((g * 37 + 5) % 256);
            uint8_t* b = blobs.data() + (size_t) g * f.bytes;
            std::memcpy(b, gguf.tensor_data(*t[0]) + E * ff.up_off, f.up_off);
            std::memcpy(b + f.up_off, gguf.tensor_data(*t[1]) + E * ff.up_off, f.up_off);
            for (int64_t r = 0; r < H; ++r)
                std::memcpy(b + f.down_off + r * drow, gguf.tensor_data(*t[2]) + E * dsz_full + r * drow_full, drow);
        }
        std::mt19937 rng(17 + l);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) NTOK * H);
        for (auto& v : x) v = nd(rng);
        const int NE = G * T;
        // BENCH_CAP=n: the launch is sized for n possible groups (the engine passes n_tokens * K), G of them real: what the empty block rows cost
        const int CAP = std::getenv("BENCH_CAP") ? std::max(G, std::atoi(std::getenv("BENCH_CAP"))) : G, CAPE = std::max(NE, CAP);
        std::vector<unsigned long long> ptr(G);
        std::vector<int32_t> start(G + 1), dst(NE), tok(NE);
        const auto L = K::native_expert_layout(f.gu_type, f.d_type, H, FF);
        uint8_t* dblob;
        void *dx, *dxq, *dscr;
        float* dout;
        unsigned long long* dptr;
        int32_t *dstart, *dn, *ddst, *dtok;
        cudaMalloc((void**) &dblob, blobs.size());
        cudaMalloc(&dx, x.size() * 4);
        cudaMalloc(&dxq, (size_t) NTOK * H / 32 * 36);
        cudaMalloc(&dscr, K::native_expert_scratch_bytes(CAPE, FF));
        cudaMalloc((void**) &dout, (size_t) NE * H * 4);
        cudaMalloc((void**) &dptr, (size_t) G * 8);
        cudaMalloc((void**) &dstart, (size_t) (G + 1) * 4);
        cudaMalloc((void**) &dn, 4);
        cudaMalloc((void**) &ddst, (size_t) NE * 4);
        cudaMalloc((void**) &dtok, (size_t) NE * 4);
        for (int g = 0; g < G; ++g) {
            ptr[g] = (unsigned long long) (dblob + (size_t) g * f.bytes);
            start[g] = g * T;
            for (int j = 0; j < T; ++j) {
                dst[g * T + j] = g * T + j;
                tok[g * T + j] = (g * 3 + j) % NTOK;
            }
        }
        start[G] = NE;
        cudaMemcpy(dblob, blobs.data(), blobs.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dptr, ptr.data(), (size_t) G * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dstart, start.data(), (size_t) (G + 1) * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dn, &G, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(ddst, dst.data(), (size_t) NE * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dtok, tok.data(), (size_t) NE * 4, cudaMemcpyHostToDevice);
        K::quantize_q8_1_rows((const float*) dx, NTOK, H, dxq, s);
        auto run = [&](int m, int phase) {
            K::native_expert_set_mode(m, phase);
            K::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, CAP, CAPE, dxq, dscr, dout, s);
        };
        std::vector<float> ref((size_t) NE * H), got((size_t) NE * H);
        cudaMemsetAsync(dout, 0xff, got.size() * 4, s);
        run(mref, 0);
        cudaMemcpyAsync(ref.data(), dout, ref.size() * 4, cudaMemcpyDeviceToHost, s);
        cudaMemsetAsync(dout, 0xff, got.size() * 4, s);
        run(mode, 0);
        cudaMemcpyAsync(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost, s);
        if (cudaStreamSynchronize(s) != cudaSuccess) { std::printf("layer %d: CUDA error\n", l); return 1; }
        size_t ndiff = 0;
        double worst = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            if (std::memcmp(&ref[i], &got[i], 4) != 0) {
                ++ndiff;
                worst = std::fmax(worst, std::fabs((double) ref[i] - got[i]) / (std::fabs((double) ref[i]) + 1e-6));
            }
        auto time = [&](int m, int phase) {
            for (int i = 0; i < 10; ++i) run(m, phase);
            cudaEventRecord(e0, s);
            for (int i = 0; i < iters; ++i) run(m, phase);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            return 1e3 * ms / iters;
        };
        const double gu_b = (double) G * 2 * f.up_off, d_b = (double) G * dsz;
        double us[2][3] = {{1e30, 1e30, 1e30}, {1e30, 1e30, 1e30}};
        for (int round = 0; round < 3; ++round)   // interleaved, the minimum: the clocks wander between runs
            for (int k = 0; k < 2; ++k)
                for (int p = 0; p < 3; ++p) us[k][p] = std::fmin(us[k][p], time(k ? mode : mref, p));
        uint64_t fnv = 1469598103934665603ull;   // FNV-1a of the output bytes: equal hashes across two builds = bitwise equal outputs
        for (size_t i = 0; i < got.size() * 4; ++i) fnv = (fnv ^ reinterpret_cast<const uint8_t*>(got.data())[i]) * 1099511628211ull;
        std::printf("hash %016llx\n", (unsigned long long) fnv);
        std::printf("layer %2d %-7s/%-6s G %d T %d | mode %d: all %7.1f us  gu %7.1f us (%3.0f GB/s)  down %6.1f us (%3.0f GB/s)"
                    " | mode %d: all %7.1f us  gu %7.1f us (%3.0f GB/s)  down %6.1f us (%3.0f GB/s) | %s\n",
                    l, ggml_type_name((ggml_type) f.gu_type), ggml_type_name((ggml_type) f.d_type), G, T,
                    mref, us[0][0], us[0][1], gu_b / us[0][1] / 1e3, us[0][2], d_b / us[0][2] / 1e3,
                    mode, us[1][0], us[1][1], gu_b / us[1][1] / 1e3, us[1][2], d_b / us[1][2] / 1e3,
                    ndiff ? "DIFFERS" : "bitwise equal");
        if (ndiff) {
            std::printf("          %zu of %zu outputs differ, worst rel %.2e\n", ndiff, ref.size(), worst);
            ++failures;
        }
        K::native_expert_set_mode(-1, 0);
        cudaFree(dblob); cudaFree(dx); cudaFree(dxq); cudaFree(dscr); cudaFree(dout); cudaFree(dptr);
        cudaFree(dstart); cudaFree(dn); cudaFree(ddst); cudaFree(dtok);
    }
    std::printf("native_expert_bench: %d differing\n", failures);
    return failures ? 1 : 0;
}

// ------------------------------------------------------------------------------------------------ --nibble
#if STRATA_EXPERT_NIBBLE_BUILT
namespace {

namespace nib = strata::core::nibble;

#define NB_CK(x)                                                                                                    \
    do {                                                                                                            \
        cudaError_t e_ = (x);                                                                                       \
        if (e_ != cudaSuccess) {                                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));                 \
            std::exit(2);                                                                                           \
        }                                                                                                           \
    } while (0)

struct NbPlan {
    int ng = 0, ne = 0, t = 0;
    std::vector<int> gid, start, tok, dst;
};

// synthetic routing: t tokens of top-10, `share` % of each token's experts the previous token's; groups by first
// appearance (resident_plan_kernel's order), entries in token order
NbPlan nb_route(std::mt19937& rng, int t, int share) {
    std::vector<int> ids, prev, cur;
    for (int k = 0; k < t; ++k) {
        cur.clear();
        const bool sh = k > 0 && share > 0;
        if (sh) {
            std::vector<int> p = prev;
            std::shuffle(p.begin(), p.end(), rng);
            cur.assign(p.begin(), p.begin() + 10 * share / 100);
        }
        while ((int) cur.size() < 10) {
            const int e = (int) (rng() % 512u);
            if (std::find(cur.begin(), cur.end(), e) != cur.end()) continue;
            if (sh && std::find(prev.begin(), prev.end(), e) != prev.end()) continue;
            cur.push_back(e);
        }
        std::shuffle(cur.begin(), cur.end(), rng);
        ids.insert(ids.end(), cur.begin(), cur.end());
        prev = cur;
    }
    NbPlan p;
    p.t = t;
    std::map<int, std::vector<int>> by;   // expert -> its entries (j = token * 10 + k)
    std::vector<int> first;
    for (int j = 0; j < (int) ids.size(); ++j) {
        if (by.find(ids[j]) == by.end()) first.push_back(ids[j]);
        by[ids[j]].push_back(j);
    }
    for (int e : first) {
        p.gid.push_back(e);
        p.start.push_back(p.ne);
        for (int j : by[e]) {
            p.tok.push_back(j / 10);
            p.dst.push_back(j);
            ++p.ne;
        }
    }
    p.start.push_back(p.ne);
    p.ng = (int) p.gid.size();
    return p;
}

struct NbDev {
    unsigned long long* ptr = nullptr;
    int32_t *start = nullptr, *ng = nullptr, *tok = nullptr, *dst = nullptr;
};
NbDev nb_upload(const NbPlan& p, const uint8_t* base, size_t stride) {
    NbDev d;
    std::vector<unsigned long long> hp(std::max(1, p.ng));
    for (int g = 0; g < p.ng; ++g) hp[g] = (unsigned long long) (base + (size_t) p.gid[g] * stride);
    NB_CK(cudaMalloc((void**) &d.ptr, hp.size() * 8));
    NB_CK(cudaMalloc((void**) &d.start, (size_t) (p.ng + 1) * 4));
    NB_CK(cudaMalloc((void**) &d.ng, 4));
    NB_CK(cudaMalloc((void**) &d.tok, (size_t) std::max(1, p.ne) * 4));
    NB_CK(cudaMalloc((void**) &d.dst, (size_t) std::max(1, p.ne) * 4));
    NB_CK(cudaMemcpy(d.ptr, hp.data(), hp.size() * 8, cudaMemcpyHostToDevice));
    NB_CK(cudaMemcpy(d.start, p.start.data(), (size_t) (p.ng + 1) * 4, cudaMemcpyHostToDevice));
    NB_CK(cudaMemcpy(d.ng, &p.ng, 4, cudaMemcpyHostToDevice));
    if (p.ne) {
        NB_CK(cudaMemcpy(d.tok, p.tok.data(), (size_t) p.ne * 4, cudaMemcpyHostToDevice));
        NB_CK(cudaMemcpy(d.dst, p.dst.data(), (size_t) p.ne * 4, cudaMemcpyHostToDevice));
    }
    return d;
}
void nb_free(NbDev& d) {
    cudaFree(d.ptr);
    cudaFree(d.start);
    cudaFree(d.ng);
    cudaFree(d.tok);
    cudaFree(d.dst);
}

std::vector<int> nb_ints(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(std::atoi(t.c_str()));
    return v;
}

}  // namespace

int nibble_main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: native_expert_bench --nibble <shard.gguf> <layer[,layer]> [--t 1,2,3,4,5,6] [--share 0,50] "
                             "[--plans 200] [--trials 7] [--mode -1|0] [--variant -1|0|1]\n");
        return 2;
    }
    const std::string shard = argv[2];
    const std::vector<int> layers = nb_ints(argv[3]);
    std::vector<int> Ts = { 1, 2, 3, 4, 5, 6 }, shares = { 0, 50 };
    int plans = 200, trials = 7, mode = -1, variant = -1;
    std::vector<int> vs;   // gu-proto: the nibble gate/up kernels compared with the production one (-1), gate/up only
    for (int i = 4; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--t") Ts = nb_ints(argv[i + 1]);
        else if (a == "--share") shares = nb_ints(argv[i + 1]);
        else if (a == "--plans") plans = std::atoi(argv[i + 1]);
        else if (a == "--trials") trials = std::atoi(argv[i + 1]);
        else if (a == "--mode") mode = std::atoi(argv[i + 1]);
        else if (a == "--variant") variant = std::atoi(argv[i + 1]);
        else if (a == "--vs") vs = nb_ints(argv[i + 1]);
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    std::unique_ptr<strata::GgufModel> model;
    try {
        model = std::make_unique<strata::GgufModel>(strata::GgufModel::open(shard));
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "native_expert_bench: %s\n", ex.what());
        return 2;
    }
    cudaDeviceProp prop{};
    NB_CK(cudaGetDeviceProperties(&prop, 0));
    char bus[64] = {};
    NB_CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
    std::printf("device 0: %s, bus %s, cc %d.%d; layout %s, 4-bit kernel %s\n", prop.name, bus, prop.major, prop.minor,
                mode < 0 ? "the card's default" : ("mode " + std::to_string(mode)).c_str(),
                variant < 0 ? "per format" : variant == 1 ? "P" : "S");
    K::native_expert_nib_set_variant(variant);
    const int64_t H = nib::kNEmbd, FF = nib::kNFf;
    constexpr int kTokMax = 8;
    int failures = 0;
    cudaStream_t s;
    NB_CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    NB_CK(cudaEventCreate(&e0));
    NB_CK(cudaEventCreate(&e1));
    // the activations: kTokMax tokens of normal floats, quantized as the verify window quantizes them
    std::mt19937 rng(20261019);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) kTokMax * H);
    for (auto& v : x) v = nd(rng);
    void *dx = nullptr, *dxq = nullptr, *dscr = nullptr;
    float* dout = nullptr;
    const int64_t cap = (int64_t) kTokMax * 10;
    NB_CK(cudaMalloc(&dx, x.size() * 4));
    NB_CK(cudaMalloc(&dxq, (size_t) kTokMax * H / 32 * 36));
    NB_CK(cudaMalloc(&dscr, K::native_expert_scratch_bytes(cap, FF)));
    NB_CK(cudaMalloc((void**) &dout, (size_t) cap * H * 4));
    NB_CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    K::quantize_q8_1_rows((const float*) dx, kTokMax, H, dxq, s);
    for (int l : layers) {
        const char* roles[3] = { "gate", "up", "down" };
        const strata::TensorInfo* t[3] = {};
        size_t sh[3] = {};
        for (int r = 0; r < 3; ++r) t[r] = model->find("blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight", &sh[r]);
        if (!t[0] || !t[1] || !t[2]) {
            std::printf("layer %d: no expert tensors\n", l);
            ++failures;
            continue;
        }
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err)) {
            std::printf("layer %d: %s\n", l, err.c_str());
            ++failures;
            continue;
        }
        nib::Config c;
        c.on = true;
        const nib::LayerVram V = nib::plan_layer(l, f, c);
        if (!V.on) {
            std::printf("layer %d: %s is not converted\n", l, ggml_type_name((ggml_type) f.gu_type));
            continue;
        }
        // the 512 blobs (P) and their 4-bit slots (N, the engine's transcoder)
        std::vector<uint8_t> host((size_t) 512 * f.bytes);
        const size_t dn_role = f.bytes - f.down_off;
        for (int e = 0; e < 512; ++e) {
            uint8_t* b = host.data() + (size_t) e * f.bytes;
            std::memcpy(b, model->shard(sh[0]).tensor_data(*t[0]) + (size_t) e * f.up_off, f.up_off);
            std::memcpy(b + f.up_off, model->shard(sh[1]).tensor_data(*t[1]) + (size_t) e * f.up_off, f.up_off);
            std::memcpy(b + f.down_off, model->shard(sh[2]).tensor_data(*t[2]) + (size_t) e * dn_role, dn_role);
        }
        uint8_t *raw = nullptr, *slots = nullptr;
        unsigned long long *dp = nullptr, *dbad = nullptr;
        NB_CK(cudaMalloc((void**) &raw, host.size()));
        NB_CK(cudaMalloc((void**) &slots, (size_t) 512 * V.slot_bytes));
        NB_CK(cudaMalloc((void**) &dp, 512 * 8));
        NB_CK(cudaMalloc((void**) &dbad, 8));
        NB_CK(cudaMemcpy(raw, host.data(), host.size(), cudaMemcpyHostToDevice));
        std::vector<unsigned long long> hp(512);
        for (int e = 0; e < 512; ++e) hp[e] = (unsigned long long) (slots + (size_t) e * V.slot_bytes);
        NB_CK(cudaMemcpy(dp, hp.data(), 512 * 8, cudaMemcpyHostToDevice));
        NB_CK(cudaMemset(dbad, 0, 8));
        for (int b0 = 0; b0 < 512; b0 += 64)
            if (nib::transcode(V, raw + (size_t) b0 * V.raw_bytes, 64, dp + b0, dbad, s) != 0) return 2;
        for (int e = 0; e < 512; ++e)
            NB_CK(cudaMemcpyAsync(slots + (size_t) e * V.slot_bytes + nib::kDownOff, raw + (size_t) e * V.raw_bytes + V.raw_down_off,
                                  V.down_bytes, cudaMemcpyDeviceToDevice, s));
        unsigned long long bad = 0;
        NB_CK(cudaMemcpyAsync(&bad, dbad, 8, cudaMemcpyDeviceToHost, s));
        NB_CK(cudaStreamSynchronize(s));
        if (bad) {
            std::printf("layer %d: %llu items do not round-trip\n", l, bad);
            ++failures;
            continue;
        }
        const auto Lr = K::native_expert_layout(f.gu_type, f.d_type, H, FF);
        const auto Ln = K::native_expert_layout_nib(f.gu_type, f.d_type, H, FF);
        std::printf("layer %2d %-7s/%-6s: 512 experts, blob %zu B, 4-bit slot %llu B\n", l, nib::fmt_name(f.gu_type),
                    ggml_type_name((ggml_type) f.d_type), f.bytes, (unsigned long long) V.slot_bytes);
        // one set of plans: every plan's out rows compared first, then P and N timed interleaved, phase 0 (the whole
        // step) and 1 (gate/up + SwiGLU/q8_1), us per launch (median of the trials)
        auto bench = [&](const std::vector<NbPlan>& ps, int tt, double us[2][2]) -> long long {
            std::vector<NbDev> dr(ps.size()), dn(ps.size());
            for (size_t i = 0; i < ps.size(); ++i) {
                dr[i] = nb_upload(ps[i], raw, f.bytes);
                dn[i] = nb_upload(ps[i], slots, V.slot_bytes);
            }
            const int64_t cp = std::max<int64_t>((int64_t) tt * 10, 1);
            auto launch = [&](int arm, size_t i) {
                const NbDev& d = arm ? dn[i] : dr[i];
                K::native_expert_grouped(arm ? Ln : Lr, d.ptr, d.start, d.ng, d.dst, d.tok, std::max<int64_t>(cp, ps[i].ng),
                                         std::max<int64_t>(cp, ps[i].ne), dxq, dscr, dout, s, 0);
            };
            long long diff = 0;
            std::vector<uint8_t> a((size_t) cap * H * 4), b(a.size());
            K::native_expert_set_mode(mode, 0);
            for (size_t i = 0; i < ps.size(); ++i) {
                const size_t n = (size_t) ps[i].ne * H * 4;
                NB_CK(cudaMemsetAsync(dout, 0xFF, a.size(), s));
                launch(0, i);
                NB_CK(cudaMemcpyAsync(a.data(), dout, n, cudaMemcpyDeviceToHost, s));
                NB_CK(cudaMemsetAsync(dout, 0x00, a.size(), s));
                launch(1, i);
                NB_CK(cudaMemcpyAsync(b.data(), dout, n, cudaMemcpyDeviceToHost, s));
                NB_CK(cudaStreamSynchronize(s));
                diff += std::memcmp(a.data(), b.data(), n) != 0;
            }
            for (int ph = 0; ph < 2; ++ph) {
                K::native_expert_set_mode(mode, ph);
                std::vector<float> tm[2];
                for (int arm = 0; arm < 2; ++arm)
                    for (size_t i = 0; i < std::min<size_t>(ps.size(), 20); ++i) launch(arm, i);
                for (int tr = 0; tr < trials; ++tr)
                    for (int arm = 0; arm < 2; ++arm) {
                        NB_CK(cudaEventRecord(e0, s));
                        for (size_t i = 0; i < ps.size(); ++i) launch(arm, i);
                        NB_CK(cudaEventRecord(e1, s));
                        NB_CK(cudaEventSynchronize(e1));
                        float ms = 0;
                        NB_CK(cudaEventElapsedTime(&ms, e0, e1));
                        tm[arm].push_back(1e3f * ms / (float) ps.size());
                    }
                for (int arm = 0; arm < 2; ++arm) {
                    std::sort(tm[arm].begin(), tm[arm].end());
                    us[ph][arm] = tm[arm][tm[arm].size() / 2];
                }
            }
            K::native_expert_set_mode(mode, 0);
            for (size_t i = 0; i < ps.size(); ++i) {
                nb_free(dr[i]);
                nb_free(dn[i]);
            }
            return diff;
        };
        if (!vs.empty()) {   // gu-proto: nibble production (-1) against each id in --vs, gate/up only (phase 3)
            std::vector<int> arms = { -1 };
            arms.insert(arms.end(), vs.begin(), vs.end());
            for (int v : vs) {
                int regs = 0, loc = 0, occ = 0;
                if (K::native_expert_nib_variant_attr(v, f.gu_type, &regs, &loc, &occ))
                    std::printf("  variant %d: %d registers, %d B local, %d blocks/SM\n", v, regs, loc, occ);
            }
            const size_t fsz = (size_t) cap * FF * 4, fa = (fsz + 255) & ~(size_t) 255;
            for (int tt : Ts) {
                if (tt < 1 || tt > kTokMax) continue;
                for (int share : shares) {
                    std::vector<NbPlan> ps;
                    double G = 0;
                    for (int i = 0; i < plans; ++i) {
                        ps.push_back(nb_route(rng, tt, share));
                        G += ps.back().ng;
                    }
                    std::vector<NbDev> dn(ps.size());
                    for (size_t i = 0; i < ps.size(); ++i) dn[i] = nb_upload(ps[i], slots, V.slot_bytes);
                    const int64_t cp = std::max<int64_t>((int64_t) tt * 10, 1);
                    auto launch = [&](int v, size_t i) {
                        K::native_expert_nib_set_variant(v);
                        K::native_expert_grouped(Ln, dn[i].ptr, dn[i].start, dn[i].ng, dn[i].dst, dn[i].tok,
                                                 std::max<int64_t>(cp, ps[i].ng), std::max<int64_t>(cp, ps[i].ne), dxq, dscr,
                                                 dout, s, 0);
                    };
                    K::native_expert_set_mode(mode, 3);
                    // every plan: each arm's gate and up rows (the plan's entries) byte for byte the production arm's
                    std::vector<long long> drows(arms.size(), 0);
                    std::vector<uint8_t> ref(2 * fsz), got(2 * fsz);
                    for (size_t i = 0; i < ps.size(); ++i) {
                        const size_t n = (size_t) ps[i].ne * FF * 4;
                        // native_expert_grouped's up rows start after cap_entries gate rows (256-aligned)
                        const size_t fi = (size_t) std::max<int64_t>(cp, ps[i].ne) * FF * 4, fai = (fi + 255) & ~(size_t) 255;
                        for (size_t a = 0; a < arms.size(); ++a) {
                            NB_CK(cudaMemsetAsync(dscr, a == 0 ? 0xFF : 0x00, 2 * fa, s));
                            launch(arms[a], i);
                            std::vector<uint8_t>& o = a == 0 ? ref : got;
                            NB_CK(cudaMemcpyAsync(o.data(), dscr, n, cudaMemcpyDeviceToHost, s));
                            NB_CK(cudaMemcpyAsync(o.data() + fsz, (const uint8_t*) dscr + fai, n, cudaMemcpyDeviceToHost, s));
                            NB_CK(cudaStreamSynchronize(s));
                            if (a == 0) continue;
                            for (size_t r = 0; r < (size_t) ps[i].ne; ++r) {
                                drows[a] += std::memcmp(ref.data() + r * FF * 4, got.data() + r * FF * 4, FF * 4) != 0;
                                drows[a] += std::memcmp(ref.data() + fsz + r * FF * 4, got.data() + fsz + r * FF * 4, FF * 4) != 0;
                            }
                        }
                    }
                    long long rows = 0;
                    for (const NbPlan& p : ps) rows += 2LL * p.ne;
                    std::vector<std::vector<float>> tm(arms.size());
                    for (size_t a = 0; a < arms.size(); ++a)
                        for (size_t i = 0; i < std::min<size_t>(ps.size(), 20); ++i) launch(arms[a], i);
                    for (int tr = 0; tr < trials; ++tr)
                        for (size_t a = 0; a < arms.size(); ++a) {
                            NB_CK(cudaEventRecord(e0, s));
                            for (size_t i = 0; i < ps.size(); ++i) launch(arms[a], i);
                            NB_CK(cudaEventRecord(e1, s));
                            NB_CK(cudaEventSynchronize(e1));
                            float ms = 0;
                            NB_CK(cudaEventElapsedTime(&ms, e0, e1));
                            tm[a].push_back(1e3f * ms / (float) ps.size());
                        }
                    K::native_expert_set_mode(mode, 0);
                    K::native_expert_nib_set_variant(variant);
                    std::printf("  T=%d share %2d%% (G %.2f, %lld gate/up rows a plan set), gate/up us per launch (median, min..max of %d):\n",
                                tt, share, G / plans, rows, trials);
                    double base = 0;
                    for (size_t a = 0; a < arms.size(); ++a) {
                        std::sort(tm[a].begin(), tm[a].end());
                        const double med = tm[a][tm[a].size() / 2];
                        if (a == 0) base = med;
                        std::printf("    %4d: %8.2f  (%7.2f..%7.2f)  x%.3f  %s\n", arms[a], med, tm[a].front(), tm[a].back(),
                                    base / med, a == 0 ? "reference" : drows[a] ? ("rows differ: " + std::to_string(drows[a])).c_str() : "bitwise equal");
                    }
                    for (size_t i = 0; i < ps.size(); ++i) nb_free(dn[i]);
                }
            }
            cudaFree(raw);
            cudaFree(slots);
            cudaFree(dp);
            cudaFree(dbad);
            continue;
        }
        for (int tt : Ts) {
            if (tt < 1 || tt > kTokMax) continue;
            for (int share : shares) {
                std::vector<NbPlan> ps;
                double G = 0;
                for (int i = 0; i < plans; ++i) {
                    ps.push_back(nb_route(rng, tt, share));
                    G += ps.back().ng;
                }
                double us[2][2];
                const long long diff = bench(ps, tt, us);
                std::printf("  T=%d share %2d%% (G %.2f): step P %7.2f N %7.2f us x%.3f | gate/up+swiglu P %7.2f N %7.2f us x%.3f | %s\n",
                            tt, share, G / plans, us[0][0], us[0][1], us[0][0] / us[0][1], us[1][0], us[1][1],
                            us[1][0] / us[1][1], diff ? "DIFFERS" : "bitwise equal");
                if (diff) {
                    std::printf("      %lld of %d plans differ\n", diff, plans);
                    ++failures;
                }
            }
        }
        cudaFree(raw);
        cudaFree(slots);
        cudaFree(dp);
        cudaFree(dbad);
    }
    K::native_expert_set_mode(-1, 0);
    K::native_expert_nib_set_variant(-1);
    std::printf("native_expert_bench --nibble: %d failing\n", failures);
    return failures ? 1 : 0;
}
#else
int nibble_main(int, char**) {
    std::fprintf(stderr, "native_expert_bench --nibble: no 4-bit kernels in this build (STRATA_EXPERIMENTAL_SM60, CUDA)\n");
    return 2;
}
#endif
