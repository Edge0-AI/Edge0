// src/kernels/dequant_wide_parity.cpp - the wide dequant kernels (a thread per 8 values, dequant_bf16.cu) against the
// per-group ones, bitwise, for Q4_K / Q5_K / Q6_K / IQ4_XS into FP16, BF16 and FP32: whole tensors, row slices and a
// padded row stride, on blocks with sane fields and on raw random bytes (NaN and Inf scales included).
//
//     build/dequant_wide_parity [--bench]
// --bench adds each kernel's time per call on the prompt path's shapes (FP16 out).  Exit 0: every case bitwise equal.
#include "strata/kernels/dequant_bf16.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "dequant_wide_parity: %s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
static uint16_t f2h(float f) {   // FP32 -> FP16 bits, round to nearest even (normal range only: the scales below)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    int e = (int) ((u >> 23) & 0xff) - 127 + 15;
    uint32_t m = u & 0x7fffffu;
    if (e <= 0) return (uint16_t) sign;
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    if ((m & 0x1fffu) > 0x1000u || ((m & 0x1fffu) == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}
struct Type { int id; const char* name; int block_bytes; };
static const Type kTypes[] = {{12, "Q4_K", 144}, {13, "Q5_K", 176}, {14, "Q6_K", 210}, {23, "IQ4_XS", 136}};

// `sane`: random codes and scales with a block scale in a weight's range; else raw random bytes
static std::vector<uint8_t> make_blocks(const Type& t, int64_t rows, int64_t cols, bool sane, uint32_t seed) {
    std::mt19937 rng(seed);
    const int64_t nb = rows * (cols / 256);
    std::vector<uint8_t> b((size_t) nb * t.block_bytes);
    for (size_t i = 0; i + 4 <= b.size(); i += 4) { const uint32_t r = rng(); std::memcpy(&b[i], &r, 4); }
    for (size_t i = b.size() & ~(size_t) 3; i < b.size(); ++i) b[i] = (uint8_t) rng();
    if (sane) {
        std::uniform_real_distribution<float> ud(1e-4f, 3e-2f);
        for (int64_t k = 0; k < nb; ++k) {
            uint8_t* p = b.data() + (size_t) k * t.block_bytes;
            const uint16_t d = f2h((rng() & 1 ? 1.0f : -1.0f) * ud(rng)), dm = f2h(-ud(rng));
            if (t.id == 14) std::memcpy(p + 208, &d, 2);
            else if (t.id == 23) std::memcpy(p, &d, 2);
            else { std::memcpy(p, &d, 2); std::memcpy(p + 2, &dm, 2); }
        }
    }
    return b;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    struct Shape { int64_t rows, cols; };
    const Shape shapes[] = {{10240, 2560}, {6144, 2560}, {2560, 6144}, {12288, 2560}, {512, 2560}, {2560, 4096}, {320, 10240}, {7, 256}, {33, 768}};
    int cases = 0, failures = 0;
    for (const Type& t : kTypes) {
        for (const Shape& sh : shapes) {
            for (int sane = 1; sane >= 0; --sane) {
                const int64_t rows = sh.rows, cols = sh.cols, n = rows * cols;
                const std::vector<uint8_t> hb = make_blocks(t, rows, cols, sane != 0, (uint32_t) (t.id * 1000003 + rows * 31 + cols + sane));
                uint8_t* db = nullptr;
                ck(cudaMalloc((void**) &db, hb.size()), "blocks");
                ck(cudaMemcpy(db, hb.data(), hb.size(), cudaMemcpyHostToDevice), "upload");
                const int64_t ldp = cols + 64;                      // the padded stride (dequant_f16_ld)
                const size_t cap = (size_t) rows * (size_t) ldp * 4;
                uint8_t *o0 = nullptr, *o1 = nullptr;
                ck(cudaMalloc((void**) &o0, cap), "out0");
                ck(cudaMalloc((void**) &o1, cap), "out1");
                std::vector<uint8_t> h0(cap), h1(cap);
                // what: 0 FP16, 1 BF16, 2 FP32, 3 FP16 with the padded stride, 4 FP16 of a row slice
                for (int what = 0; what < 5; ++what) {
                    const int64_t r0 = what == 4 ? rows / 3 : 0, nr = what == 4 ? rows - rows / 3 - rows / 5 : rows;
                    if (nr <= 0) continue;
                    for (int wide = 0; wide < 2; ++wide) {
                        uint8_t* o = wide ? o1 : o0;
                        ck(cudaMemsetAsync(o, 0xA5, cap, s), "memset");
                        K::dequant_set_wide(wide);
                        if (what == 0 || what == 4) K::dequant_f16(t.id, db, r0, nr, cols, (uint16_t*) o, s);
                        else if (what == 1) K::dequant_bf16(t.id, db, r0, nr, cols, (uint16_t*) o, s);
                        else if (what == 2) K::dequant_f32(t.id, db, r0, nr, cols, (float*) o, s);
                        else if (!K::dequant_f16_ld(t.id, db, r0, nr, cols, ldp, (uint16_t*) o, s)) { std::printf("FAIL %s: dequant_f16_ld refused\n", t.name); ++failures; }
                        ck(cudaMemcpyAsync(wide ? h1.data() : h0.data(), o, cap, cudaMemcpyDeviceToHost, s), "download");
                    }
                    ck(cudaStreamSynchronize(s), "sync");
                    ++cases;
                    if (std::memcmp(h0.data(), h1.data(), cap) != 0) {
                        size_t nd = 0, first = 0;
                        for (size_t i = 0; i < cap; ++i) if (h0[i] != h1[i]) { if (!nd) first = i; ++nd; }
                        std::printf("FAIL %s %lldx%lld %s what %d: %zu bytes differ, first at %zu\n", t.name, (long long) rows, (long long) cols,
                                    sane ? "sane" : "raw", what, nd, first);
                        ++failures;
                    }
                    // the default-pattern bytes outside the written values must be untouched by both (the padded stride's gaps)
                    if (what == 3) {
                        size_t bad = 0;
                        for (int64_t r = 0; r < nr; ++r)
                            for (int64_t c = cols; c < ldp; ++c)
                                if (h1[(size_t) (r * ldp + c) * 2] != 0xA5 || h1[(size_t) (r * ldp + c) * 2 + 1] != 0xA5) ++bad;
                        if (bad) { std::printf("FAIL %s %lldx%lld: the wide kernel wrote %zu values into the row padding\n", t.name, (long long) rows, (long long) cols, bad); ++failures; }
                    }
                }
                if (bench && sane && rows >= 512) {
                    cudaEvent_t e0, e1;
                    cudaEventCreate(&e0);
                    cudaEventCreate(&e1);
                    double ms[2] = {0, 0};
                    const int reps = 20;
                    for (int wide = 0; wide < 2; ++wide) {
                        K::dequant_set_wide(wide);
                        for (int i = 0; i < 3; ++i) K::dequant_f16(t.id, db, 0, rows, cols, (uint16_t*) o0, s);
                        cudaEventRecord(e0, s);
                        for (int i = 0; i < reps; ++i) K::dequant_f16(t.id, db, 0, rows, cols, (uint16_t*) o0, s);
                        cudaEventRecord(e1, s);
                        ck(cudaEventSynchronize(e1), "bench sync");
                        float tms = 0;
                        cudaEventElapsedTime(&tms, e0, e1);
                        ms[wide] = tms / reps;
                    }
                    const double bytes = (double) hb.size() + (double) n * 2;
                    std::printf("BENCH %-6s %6lldx%-6lld per-group %8.3f ms (%5.1f GB/s) | wide %8.3f ms (%5.1f GB/s) | %.2fx\n", t.name, (long long) rows,
                                (long long) cols, ms[0], bytes / ms[0] / 1e6, ms[1], bytes / ms[1] / 1e6, ms[0] / ms[1]);
                    cudaEventDestroy(e0);
                    cudaEventDestroy(e1);
                }
                cudaFree(db);
                cudaFree(o0);
                cudaFree(o1);
            }
        }
    }
    K::dequant_set_wide(-1);
    std::printf("dequant_wide_parity: %d cases, %s\n", cases, failures ? "FAILED" : "OK (bitwise)");
    return failures ? 1 : 0;
}
