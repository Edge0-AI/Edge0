// src/kernels/tq_pascal_bench.cpp - the Gyro trellis decode kernels (TQ2_T 144, TQK6 145, TQK7 146 and Gyro-L's TQP5-7 147-149) on synthetic blocks.
//
//     build/tq_pascal_bench            checks: the decode-once kernels against the per-column ones (bitwise), and a
//                                      checksum of every output, so two builds can be compared byte for byte
//     build/tq_pascal_bench --bench    the checks, then microseconds per call
//
// Random bytes are valid trellis codes (any 15-bit state indexes the codebook); only the fp16 block scales are set,
// small enough that the grouped path's SwiGLU output keeps a finite fp16 q8_1 scale.  No model, needs the GPU.
// The checksum lines ("SUM ...") are what two builds are compared by: same lines, same bits.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml-common-tq.h"   // tq_lut_f16: the codebook, as host data

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template<typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}

const char* name_of(int t) {
    return t == 144 ? "TQ2_T" : t == 145 ? "TQK6" : t == 146 ? "TQK7" : t == 147 ? "TQP5" : t == 148 ? "TQP6" : t == 149 ? "TQP7"
         : t == 42 ? "Q2_0" : "?";
}

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const int bv = t == 42 ? 64 : 128;
    const size_t rb = k::iq_row_bytes(t, n), bs = k::iq_row_bytes(t, bv);
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 3);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        const uint16_t h = (uint16_t) ((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));   // 2^-13 .. 2^-6
        std::memcpy(&w[o], &h, 2);
    }
    return w;
}

std::vector<float> random_x(size_t n, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(n);
    for (auto& v : x) v = nd(rng);
    return x;
}

void check_mmvq(int t, int n_in, int n_out, cudaStream_t s, std::mt19937& rng) {
    const auto w = random_rows(t, n_out, n_in, rng);
    uint8_t* dw = dalloc<uint8_t>(w.size());
    ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    const int max_cols = 8;
    const auto x = random_x((size_t) max_cols * n_in, rng);
    float* dx = dalloc<float>(x.size());
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    uint8_t* dxq = dalloc<uint8_t>((size_t) max_cols * (n_in / 32) * 36);
    k::quantize_q8_1_rows(dx, max_cols, n_in, dxq, s);
    float* dy_old = dalloc<float>((size_t) max_cols * n_out);
    float* dy_new = dalloc<float>((size_t) max_cols * n_out);
    std::vector<float> y_old((size_t) max_cols * n_out), y_new(y_old.size());
    uint64_t sum = 1469598103934665603ull;
    int bad = 0;
    for (int nc = 1; nc <= max_cols; ++nc) {
        ck(cudaMemset(dy_old, 0xFF, y_old.size() * 4), "memset");
        ck(cudaMemset(dy_new, 0xFF, y_new.size() * 4), "memset");
        k::iq_set_old_kernels(true);
        k::iq_mmvq(t, dw, dxq, dy_old, n_in, n_out, nc, s);
        k::iq_set_old_kernels(false);
        k::iq_mmvq(t, dw, dxq, dy_new, n_in, n_out, nc, s);
        ck(cudaStreamSynchronize(s), "sync");
        ck(cudaMemcpy(y_old.data(), dy_old, y_old.size() * 4, cudaMemcpyDeviceToHost), "y_old");
        ck(cudaMemcpy(y_new.data(), dy_new, y_new.size() * 4, cudaMemcpyDeviceToHost), "y_new");
        size_t diff = 0;
        bool finite = true;
        for (size_t i = 0; i < (size_t) nc * n_out; ++i) {
            diff += std::memcmp(&y_old[i], &y_new[i], 4) != 0;
            finite = finite && std::isfinite(y_new[i]);
        }
        if (diff || !finite) { std::printf("  %s mmvq nc %d: %zu differ%s FAIL\n", name_of(t), nc, diff, finite ? "" : ", non-finite"); ++bad; }
        sum = fnv(y_new.data(), (size_t) nc * n_out * 4, sum);
    }
    std::printf("SUM mmvq %-6s %5d x %5d  %016llx  %s\n", name_of(t), n_out, n_in, (unsigned long long) sum,
                bad ? "FAIL" : "decode-once == per-column");
    g_fail += bad;
    cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dy_old); cudaFree(dy_new);
}

struct Grouped {
    k::NativeExpertLayout L;
    int n_groups = 0, n_ent = 0, cap_groups = 0, cap_ent = 0, n_tok = 0;
    std::vector<uint8_t*> blobs;
    unsigned long long* dptr = nullptr;
    int32_t *dstart = nullptr, *dn = nullptr, *ddst = nullptr, *dtok = nullptr;
    float* dx = nullptr;
    uint8_t *dxq = nullptr, *dscr = nullptr;
    float* dout = nullptr;
    size_t out_floats = 0;

    Grouped(int gu, int dt, int64_t H, int64_t FF, const std::vector<int>& counts, int tokens, std::mt19937& rng) {
        L = k::native_expert_layout(gu, dt, H, FF);
        n_groups = (int) counts.size();
        cap_groups = n_groups + 2;
        n_tok = tokens;
        std::vector<int32_t> start(1, 0), tok, dst;
        for (int c : counts) start.push_back(start.back() + c);
        n_ent = start.back();
        cap_ent = n_ent + 3;
        std::uniform_int_distribution<int> tk(0, tokens - 1);
        for (int e = 0; e < n_ent; ++e) tok.push_back(tk(rng));
        dst.resize(n_ent);
        std::iota(dst.begin(), dst.end(), 0);
        std::shuffle(dst.begin(), dst.end(), rng);
        std::vector<unsigned long long> ptr;
        for (int g = 0; g < n_groups; ++g) {
            auto b = random_rows(gu, 2 * FF, H, rng);
            const auto d = random_rows(dt, H, FF, rng);
            b.insert(b.end(), d.begin(), d.end());
            if (b.size() != L.bytes) { std::fprintf(stderr, "blob size %zu != %zu\n", b.size(), L.bytes); std::exit(2); }
            uint8_t* db = dalloc<uint8_t>(b.size());
            ck(cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice), "blob");
            blobs.push_back(db);
            ptr.push_back((unsigned long long) db);
        }
        dptr = dalloc<unsigned long long>(cap_groups);
        dstart = dalloc<int32_t>(cap_groups + 1);
        dn = dalloc<int32_t>(1);
        ddst = dalloc<int32_t>(cap_ent);
        dtok = dalloc<int32_t>(cap_ent);
        ck(cudaMemcpy(dptr, ptr.data(), ptr.size() * 8, cudaMemcpyHostToDevice), "ptr");
        ck(cudaMemcpy(dstart, start.data(), start.size() * 4, cudaMemcpyHostToDevice), "start");
        ck(cudaMemcpy(dn, &n_groups, 4, cudaMemcpyHostToDevice), "n");
        if (n_ent) {
            ck(cudaMemcpy(ddst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "dst");
            ck(cudaMemcpy(dtok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice), "tok");
        }
        const auto x = random_x((size_t) tokens * H, rng);
        dx = dalloc<float>(x.size());
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        dxq = dalloc<uint8_t>((size_t) tokens * (H / 32) * 36);
        dscr = dalloc<uint8_t>(k::native_expert_scratch_bytes(cap_ent, FF));
        out_floats = (size_t) cap_ent * H;
        dout = dalloc<float>(out_floats);
    }
    ~Grouped() {
        for (auto* b : blobs) cudaFree(b);
        cudaFree(dptr); cudaFree(dstart); cudaFree(dn); cudaFree(ddst); cudaFree(dtok);
        cudaFree(dx); cudaFree(dxq); cudaFree(dscr); cudaFree(dout);
    }
    void run(bool old, cudaStream_t s) {
        k::iq_set_old_kernels(old);
        k::quantize_q8_1_rows(dx, n_tok, L.n_embd, dxq, s);
        k::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, cap_groups, cap_ent, dxq, dscr, dout, s);
        k::iq_set_old_kernels(false);
    }
    std::vector<float> result(bool old, cudaStream_t s) {
        ck(cudaMemset(dout, 0xFF, out_floats * 4), "memset");
        ck(cudaMemset(dscr, 0, k::native_expert_scratch_bytes(cap_ent, L.n_ff)), "memset");
        run(old, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> o(out_floats);
        ck(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost), "out");
        return o;
    }
};

void check_grouped(int gu, int dt, int64_t H, int64_t FF, cudaStream_t s, std::mt19937& rng) {
    Grouped G(gu, dt, H, FF, {1, 3, 4, 5, 0, 8, 2, 11, 1}, 8, rng);
    const auto a = G.result(true, s), b = G.result(false, s);
    size_t diff = 0;
    bool finite = true;
    for (size_t i = 0; i < a.size(); ++i) {
        diff += std::memcmp(&a[i], &b[i], 4) != 0;
        if (i < (size_t) G.n_ent * H) finite = finite && std::isfinite(b[i]);
    }
    const bool ok = diff == 0 && finite;
    std::printf("SUM grouped %-6s/%-6s %5lld x %4lld  %016llx  %s\n", name_of(gu), name_of(dt), (long long) H, (long long) FF,
                (unsigned long long) fnv(b.data(), b.size() * 4), ok ? "decode-once == per-column" : "FAIL");
    if (!ok) { std::printf("  %zu of %zu floats differ%s\n", diff, a.size(), finite ? "" : ", non-finite outputs"); ++g_fail; }
}

// ---- Gyro-L pair trellis (TQP5/6/7, ggml 147-149) against a reference written from the format definition (no ggml):
// state t of a block = the 16 stream bits that start at bit (63 - t) * K, LSB first, circular over 64 * K bits (15 bits
// are used); the step's two weights are the first pair of the state's codebook entry, times the block's fp16 scale.
uint32_t tqp_ref_state(const uint8_t* qs, int K, int t) {
    const int nbits = 64 * K;
    uint32_t s = 0;
    for (int b = 0; b < 16; ++b) {
        const int pos = ((63 - t) * K + b) % nbits;
        s |= (uint32_t) ((qs[pos >> 3] >> (pos & 7)) & 1) << b;
    }
    return s & 0x7FFFu;
}

float half_to_f(uint16_t h) {   // normal and subnormal fp16 (no inf / nan in these tests)
    const int e = (h >> 10) & 31, m = h & 1023;
    const float v = e == 0 ? std::ldexp((float) m, -24) : std::ldexp((float) (1024 + m), e - 25);
    return (h & 0x8000) ? -v : v;
}

uint16_t f_to_half_rn(float f) {   // round to nearest even, the GPU's __float2half_rn
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int e = (int) ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFFu;
    if (e >= 31) return (uint16_t) (sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const int shift = 14 - e;
        uint32_t half = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1u))) ++half;
        return (uint16_t) (sign | half);
    }
    uint32_t half = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;
    return (uint16_t) (sign | half);
}

// the weights of `rows` rows of `n` values, as floats (d * codebook, unrounded)
std::vector<float> tqp_ref_weights(int t, const std::vector<uint8_t>& w, int64_t rows, int64_t n) {
    const int K = t - 142;   // 147 -> 5, 148 -> 6, 149 -> 7
    const size_t bs = k::iq_row_bytes(t, 128);
    std::vector<float> out((size_t) (rows * n));
    for (size_t b = 0; b < (size_t) (rows * n / 128); ++b) {
        const uint8_t* blk = &w[b * bs];
        uint16_t hd;
        std::memcpy(&hd, blk, 2);
        const float d = half_to_f(hd);
        for (int step = 0; step < 64; ++step) {
            const uint32_t s = tqp_ref_state(blk + 2, K, step);
            const uint32_t x = (s & 0x7FFFu) * 0x9e3779b1u;
            const uint32_t pt = x >> 21;
            out[b * 128 + 2 * step + 0] = d * half_to_f(tq_lut_f16[2 * pt + 0]);
            out[b * 128 + 2 * step + 1] = d * half_to_f(tq_lut_f16[2 * pt + 1]);
        }
    }
    return out;
}

void check_tqp_reference(int t, int n_in, int n_out, cudaStream_t s, std::mt19937& rng) {
    const auto w = random_rows(t, n_out, n_in, rng);
    const auto ref = tqp_ref_weights(t, w, n_out, n_in);
    uint8_t* dw = dalloc<uint8_t>(w.size());
    ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    // (1) the prompt path's fp16 dequantizer: bit for bit the fp16 of the reference weights
    const int64_t nv = (int64_t) n_out * n_in;
    uint16_t* dh = dalloc<uint16_t>((size_t) nv);
    k::iq_dequant_f16(t, dw, nv, dh, s);
    ck(cudaStreamSynchronize(s), "sync");
    std::vector<uint16_t> h((size_t) nv);
    ck(cudaMemcpy(h.data(), dh, h.size() * 2, cudaMemcpyDeviceToHost), "h");
    size_t bad = 0;
    for (size_t i = 0; i < h.size(); ++i) bad += h[i] != f_to_half_rn(ref[i]);
    std::printf("tqp ref %-6s dequant_f16 %5d x %5d: %zu of %zu fp16 values differ from the definition  %s\n", name_of(t), n_out, n_in,
                bad, h.size(), bad ? "FAIL" : "bitwise equal");
    g_fail += bad != 0;
    // (2) the decode kernels (int8 codebook, q8_1 activations): relative L1 error against the float weights . the
    // dequantized q8_1 activations, in double, below the 3e-2 every expert type gets in native_expert_parity
    const int nc = 3;
    const auto x = random_x((size_t) nc * n_in, rng);
    float* dx = dalloc<float>(x.size());
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    uint8_t* dxq = dalloc<uint8_t>((size_t) nc * (n_in / 32) * 36);
    k::quantize_q8_1_rows(dx, nc, n_in, dxq, s);
    float* dy = dalloc<float>((size_t) nc * n_out);
    k::iq_mmvq(t, dw, dxq, dy, n_in, n_out, nc, s);
    ck(cudaStreamSynchronize(s), "sync");
    std::vector<float> y((size_t) nc * n_out);
    ck(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost), "y");
    std::vector<uint8_t> xq((size_t) nc * (n_in / 32) * 36);
    ck(cudaMemcpy(xq.data(), dxq, xq.size(), cudaMemcpyDeviceToHost), "xq");
    double l1 = 0, l1ref = 0;
    for (int c = 0; c < nc; ++c) {
        std::vector<double> xd((size_t) n_in);
        for (int b = 0; b < n_in / 32; ++b) {
            const uint8_t* blk = &xq[((size_t) c * (n_in / 32) + b) * 36];
            uint16_t hd;
            std::memcpy(&hd, blk, 2);
            const double d = half_to_f(hd);
            for (int j = 0; j < 32; ++j) xd[(size_t) b * 32 + j] = d * (double) (int8_t) blk[4 + j];
        }
        for (int r = 0; r < n_out; ++r) {
            double acc = 0;
            for (int i = 0; i < n_in; ++i) acc += (double) ref[(size_t) r * n_in + i] * xd[(size_t) i];
            l1 += std::fabs((double) y[(size_t) c * n_out + r] - acc);
            l1ref += std::fabs(acc);
        }
    }
    const double rel = l1 / l1ref;
    std::printf("tqp ref %-6s mmvq %5d x %5d x %d cols: relative L1 error %.2e (bound 3e-2)  %s\n", name_of(t), n_out, n_in, nc, rel,
                rel < 3e-2 ? "ok" : "FAIL");
    g_fail += !(rel < 3e-2);
    cudaFree(dw); cudaFree(dh); cudaFree(dx); cudaFree(dxq); cudaFree(dy);
}

float time_ms(cudaStream_t s, int it, const auto& fn) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    fn();
    ck(cudaStreamSynchronize(s), "warm");
    cudaEventRecord(e0, s);
    for (int i = 0; i < it; ++i) fn();
    cudaEventRecord(e1, s);
    ck(cudaEventSynchronize(e1), "bench");
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / it;
}

void bench(cudaStream_t s, std::mt19937& rng) {
    std::printf("\n--bench: microseconds per call (idle GPU assumed)\n");
    const int n_in = 2560, n_out = 8192, it = 100;
    for (int t : {144, 145, 146, 42}) {
        const auto w = random_rows(t, n_out, n_in, rng);
        uint8_t* dw = dalloc<uint8_t>(w.size());
        ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
        const auto x = random_x((size_t) 8 * n_in, rng);
        float* dx = dalloc<float>(x.size());
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        uint8_t* dxq = dalloc<uint8_t>((size_t) 8 * (n_in / 32) * 36);
        k::quantize_q8_1_rows(dx, 8, n_in, dxq, s);
        float* dy = dalloc<float>((size_t) 8 * n_out);
        const double gb = (double) w.size() / 1e9;
        std::printf("BENCH mmvq %-6s %d x %d (%.3f GB):", name_of(t), n_out, n_in, gb);
        for (int nc : {1, 2, 3, 4, 8}) {
            const float us = 1e3f * time_ms(s, it, [&] { k::iq_mmvq(t, dw, dxq, dy, n_in, n_out, nc, s); });
            std::printf("  nc%d %.1f us (%.0f GB/s)", nc, us, gb / (us * 1e-6));
        }
        std::printf("\n");
        cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dy);
    }
    // the same 21M weights as rows of different lengths: a per-row cost shows as a falling time with longer rows
    for (int t : {145}) {
        for (auto rc : {std::pair<int, int>{8192, 2560}, {4096, 5120}, {2048, 10240}, {1024, 20480}}) {
            const int ni = rc.second, no = rc.first;
            const auto w = random_rows(t, no, ni, rng);
            uint8_t* dw = dalloc<uint8_t>(w.size());
            ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
            const auto x = random_x((size_t) 8 * ni, rng);
            float* dx = dalloc<float>(x.size());
            ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
            uint8_t* dxq = dalloc<uint8_t>((size_t) 8 * (ni / 32) * 36);
            k::quantize_q8_1_rows(dx, 8, ni, dxq, s);
            float* dy = dalloc<float>((size_t) 8 * no);
            const float us = 1e3f * time_ms(s, it, [&] { k::iq_mmvq(t, dw, dxq, dy, ni, no, 1, s); });
            std::printf("BENCH rows %-6s %5d x %5d nc1 %.1f us\n", name_of(t), no, ni, us);
            cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dy);
        }
    }
    // the verify window's shape: 2560 x 640 experts, 16 groups of m entries each
    for (auto gd : {std::pair<int, int>{145, 146}, {145, 144}, {144, 144}, {144, 146}}) {
        std::printf("BENCH grouped %-6s/%-6s 16 groups of m entries:", name_of(gd.first), name_of(gd.second));
        for (int m : {1, 2, 3, 4, 8}) {
            Grouped G(gd.first, gd.second, 2560, 640, std::vector<int>(16, m), 8, rng);
            const float us = 1e3f * time_ms(s, 30, [&] { G.run(false, s); });
            const double gb = 16.0 * (double) G.L.bytes / 1e9;
            std::printf("  m=%d %.1f us (%.0f GB/s)", m, us, gb / (us * 1e-6));
        }
        std::printf("\n");
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device %s cc %d.%d\n", prop.name, prop.major, prop.minor);
    std::mt19937 rng(18);
    // Gyro-S / Gyro-M types first, on the generator they always had: their "SUM" lines are the same bits as before the pair types
    for (int t : {144, 145, 146}) {
        check_mmvq(t, 2560, 67, s, rng);
        check_mmvq(t, 1024, 5, s, rng);
    }
    for (int gu : {144, 145})
        for (int dt : {144, 146}) check_grouped(gu, dt, 2560, 640, s, rng);
    // other geometries: a longer hidden row (the staged gate/up falls back from its 80-block activation buffer), a shorter
    // expert row (the 8-lane down rows need n_ff 640)
    for (int gu : {144, 145}) {
        check_grouped(gu, 146, 5120, 640, s, rng);
        check_grouped(gu, 144, 2560, 512, s, rng);
        check_grouped(gu, 146, 2048, 256, s, rng);
    }
    // Gyro-L's pair trellis (own generator): decode-once == per-column, then a reference from the format definition
    // (fp16 dequantizer bit for bit, decode kernels to 3e-2)
    std::mt19937 rng_p(19);
    for (int t : {147, 148, 149}) {
        check_mmvq(t, 2560, 67, s, rng_p);
        check_mmvq(t, 1024, 5, s, rng_p);
        check_tqp_reference(t, 2560, 67, s, rng_p);
        check_tqp_reference(t, 1024, 5, s, rng_p);
        check_grouped(t, t, 2560, 640, s, rng_p);   // the pair trellis in both roles, as its file has it (tqp6 / tqp6)
        check_grouped(t, t, 5120, 640, s, rng_p);
    }
    check_grouped(148, 146, 2560, 640, s, rng_p);
    check_grouped(145, 148, 2560, 640, s, rng_p);
    if (do_bench) bench(s, rng);
    std::printf("tq_pascal_bench: %d failures\n", g_fail);
    cudaStreamDestroy(s);
    return g_fail ? 1 : 0;
}
