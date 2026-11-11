// src/core/overlap_main.cpp - `strata-overlap`: does streaming an expert overlap with computing on it?
//
// THE ARCHITECTURE'S CENTRAL PREMISE.  Only (1 - h) of a token's expert weights are in VRAM, so the rest must
// cross the bus while the GPU is working.  If the copy and the compute do not overlap, the per-token cost is
// their SUM; if they do, it is their MAXIMUM.  L3 measured the stream at 4.07 ms/token (design h) and L5 the
// matvec at 27.9 ms/token, so serial would be ~32 ms and overlapped ~28 - but those are the STEADY-STATE
// numbers for a whole token; at one expert's granularity the ratio is completely different and that is what
// decides how the loop must be built.
//
// Measured against a SERIAL baseline in the same process, because the interesting quantity is the ratio and
// not the absolute figure - and L6 established that absolute figures on this machine move by 20%.
#include "strata/core/pinned.hpp"
#include "strata/kernels/s_gemv.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Timing {
    double seconds = 0;
    const char* name = "";
};

}  // namespace

int main(int argc, char** argv) {
    std::string path;
    int experts = 64;
    int threads_per_row = 32;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--file") path = val();
        else if (a == "--experts") experts = std::atoi(val());
        else if (a == "--tpr") threads_per_row = std::atoi(val());
        else {
            std::fprintf(stderr, "usage: strata-overlap --file experts.bin [--experts N] [--tpr N]\n");
            return 2;
        }
    }
    if (path.empty()) { std::fprintf(stderr, "--file is required\n"); return 2; }

    // One ROLE's worth of S2 weights: gate is [n_embd 2560] x [ff 640] = 1,638,400 weights = 460,800 B of
    // codes at 4 codes per byte, plus 40 scales per row.
    const long long n_in = 2560, n_out = 640;
    const uint64_t role_codes = (uint64_t) n_out * n_in / 4;
    const uint64_t role_bytes = role_codes + (uint64_t) n_out * (n_in / 64) * 4;
    std::printf("per role: %lld x %lld S2 weights, %.1f KB of planes\n", n_in, n_out, role_bytes / 1024.0);

    // host side: `experts` blobs read into a pinned arena, and the S2 planes the GEMV will read
    strata::core::PinnedArena arena((uint64_t) experts * role_codes);
    if (!arena.valid()) { std::fprintf(stderr, "arena allocation failed\n"); return 1; }
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return 1; }
        // take the first `experts` blobs' worth of CODE bytes, pretending each is a role's plane: the bytes
        // are real Q2_0 codes, so the kernel decodes real data and no cache-friendly pattern is invented
        for (int i = 0; i < experts; ++i) {
            f.seekg((std::streamoff) i * 1382400);
            f.read((char*) arena.data() + (uint64_t) i * role_codes, (std::streamsize) role_codes);
            if ((uint64_t) f.gcount() != role_codes) {
                std::fprintf(stderr, "short read on expert %d\n", i);
                return 1;
            }
        }
    }
    std::printf("loaded %d role-sized planes into a %s arena\n", experts, arena.note.c_str());

    // device side: double buffered codes, plus x / scales / y
    uint8_t* d_codes[2] = {nullptr, nullptr};
    for (int b = 0; b < 2; ++b) check(cudaMalloc(&d_codes[b], role_codes), "cudaMalloc codes");
    uint16_t* d_x = nullptr;
    float *d_scales = nullptr, *d_y = nullptr;
    check(cudaMalloc(&d_x, n_in * sizeof(uint16_t)), "cudaMalloc x");
    check(cudaMalloc(&d_scales, (size_t) n_out * (n_in / 64) * sizeof(float)), "cudaMalloc scales");
    check(cudaMalloc(&d_y, n_out * sizeof(float)), "cudaMalloc y");
    std::vector<uint16_t> hx((size_t) n_in, 0x3C00);
    std::vector<float> hs((size_t) n_out * (size_t) (n_in / 64), 0.001f);
    check(cudaMemcpy(d_x, hx.data(), hx.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy x");
    check(cudaMemcpy(d_scales, hs.data(), hs.size() * sizeof(float), cudaMemcpyHostToDevice), "copy scales");

    cudaStream_t s_copy{}, s_comp{};
    check(cudaStreamCreate(&s_copy), "stream copy");
    check(cudaStreamCreate(&s_comp), "stream compute");
    const strata::kernels::SForm form{2, -1, 64, strata::kernels::Codebook::Affine, false};

    // ---- SERIAL: the same operations on the SAME stream, so the GPU runs them one after another ----
    //
    // NOT a sync-per-iteration loop.  The first version waited with `cudaStreamSynchronize` after every copy and
    // measured 234.8 us/expert where a copy plus a GEMV is ~43 us: the baseline was dominated by the CPU
    // round-trip, so the ratio of 0.095 mostly measured "with syncs" against "without syncs".  Putting both
    // operations on one stream expresses the same dependency with NO CPU involvement and ONE sync at the end,
    // which is what makes the comparison about overlap rather than about synchronisation.
    check(cudaStreamSynchronize(0), "pre sync");
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < experts; ++i) {
        check(cudaMemcpyAsync(d_codes[0], arena.data() + (uint64_t) i * role_codes, role_codes,
                              cudaMemcpyHostToDevice, 0),
              "serial copy");
        strata::kernels::s_gemv_split_async(d_x, d_codes[0], d_scales, nullptr, d_y, n_in, n_out, form,
                                            threads_per_row, (void*) (cudaStream_t) 0);
    }
    check(cudaStreamSynchronize(0), "serial final sync");
    const double serial = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // ---- OVERLAPPED: copy the next one while computing on the current one ----
    check(cudaStreamSynchronize(0), "pre sync 2");
    cudaEvent_t ev[2];
    for (int b = 0; b < 2; ++b) check(cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming), "event");
    t0 = std::chrono::steady_clock::now();
    check(cudaMemcpyAsync(d_codes[0], arena.data(), role_codes, cudaMemcpyHostToDevice, s_copy), "prologue");
    check(cudaEventRecord(ev[0], s_copy), "prologue event");
    for (int i = 0; i < experts; ++i) {
        const int cur = i & 1, nxt = (i + 1) & 1;
        if (i + 1 < experts) {
            check(cudaMemcpyAsync(d_codes[nxt], arena.data() + (uint64_t) (i + 1) * role_codes, role_codes,
                                  cudaMemcpyHostToDevice, s_copy),
                  "overlap copy");
            check(cudaEventRecord(ev[nxt], s_copy), "overlap event");
        }
        // the compute stream must not start row i until its codes have landed
        check(cudaStreamWaitEvent(s_comp, ev[cur], 0), "wait");
        // a hand-written launch so it goes on s_comp; `s_gemv` synchronises on the default stream
        strata::kernels::s_gemv_split_async(d_x, d_codes[cur], d_scales, nullptr, d_y, n_in, n_out, form,
                                            threads_per_row, (void*) s_comp);
    }
    check(cudaStreamSynchronize(s_comp), "final sync");
    check(cudaStreamSynchronize(s_copy), "final sync 2");
    const double overlap = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("\n  serial     %8.3f ms for %d experts  (%.1f us/expert)\n", serial * 1000, experts,
                serial * 1e6 / experts);
    std::printf("  overlapped %8.3f ms for %d experts  (%.1f us/expert)\n", overlap * 1000, experts,
                overlap * 1e6 / experts);
    std::printf("  ratio      %.3f  (1.00 = no overlap at all, <1.00 = overlap)\n", overlap / serial);
    std::printf("\nNOTE: the OVERLAPPED loop launches on `s_comp` without synchronising, so the copy and the\n"
                "      compute really are concurrent; the serial loop waits after each step.  What this does NOT\n"
                "      model is the CPU expert path, the router, or the fact that a real token has 1440 roles to\n"
                "      stream rather than %d.\n", experts);

    for (int b = 0; b < 2; ++b) {
        cudaFree(d_codes[b]);
        cudaEventDestroy(ev[b]);
    }
    cudaFree(d_x);
    cudaFree(d_scales);
    cudaFree(d_y);
    cudaStreamDestroy(s_copy);
    cudaStreamDestroy(s_comp);
    return 0;
}
