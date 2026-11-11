// src/kernels/ple_iq4nl_dequant_test.cpp - STRATA_PLE_COMPACT: ple_iq4nl_dequant (raw IQ4_NL PLE rows to floats on the
// device) against iq4nl_dequant_row (the host decode the prompt path used before), bit for bit.  Synthetic bytes, no model.
//
//   - every one of the 65,536 fp16 scale patterns (infinities and NaNs included: those are compared as "both NaN") in
//     every block position, with random codes;
//   - rows of random bytes;
//   - the row the table gives for an index past its end (scale 0, code 8 in every nibble: +0.0f, what PleTable::read_row
//     writes for such a row);
//   - an odd row count (not a multiple of the 256-thread block).
// Exit 77 without a CUDA device.
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
using namespace strata::kernels;

int check(const std::vector<uint8_t>& raw, int64_t rows, const char* what) {
    float* dout = nullptr;
    uint8_t* draw = nullptr;
    if (cudaMalloc((void**) &dout, (size_t) rows * PLE_HEAD_DIM * 4) != cudaSuccess ||
        cudaMalloc((void**) &draw, raw.size()) != cudaSuccess ||
        cudaMemcpy(draw, raw.data(), raw.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::printf("CUDA setup failed for %s\n", what);
        return 1;
    }
    cudaMemset(dout, 0xff, (size_t) rows * PLE_HEAD_DIM * 4);
    ple_iq4nl_dequant(draw, rows, dout, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("CUDA error for %s: %s\n", what, cudaGetErrorString(cudaGetLastError())); return 1; }
    std::vector<float> got((size_t) rows * PLE_HEAD_DIM), want(got.size());
    cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
    cudaFree(dout);
    cudaFree(draw);
    for (int64_t r = 0; r < rows; ++r) iq4nl_dequant_row(raw.data() + (size_t) r * PLE_ROW_BYTES, want.data() + (size_t) r * PLE_HEAD_DIM);
    long long diff = 0, first = -1;
    for (size_t i = 0; i < got.size(); ++i) {
        uint32_t a, b;
        std::memcpy(&a, &got[i], 4);
        std::memcpy(&b, &want[i], 4);
        if (std::isnan(got[i]) && std::isnan(want[i])) continue;   // a NaN scale: any payload is a NaN
        if (a != b) { if (first < 0) first = (long long) i; ++diff; }
    }
    std::printf("  %-52s %lld rows: %s\n", what, (long long) rows, diff ? "*** DIFFERS ***" : "bitwise equal");
    if (diff) std::printf("      %lld values differ, first at %lld\n", diff, first);
    return diff ? 1 : 0;
}
}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) { std::printf("no CUDA device (skipped)\n"); return 77; }
    int bad = 0;
    std::mt19937 rng(20261026u);
    {   // all fp16 patterns: 65,536 scales over 5 blocks a row = 13,108 rows (the last row pads with 0)
        const int64_t rows = (65536 + 4) / 5;
        std::vector<uint8_t> raw((size_t) rows * PLE_ROW_BYTES);
        for (auto& b : raw) b = (uint8_t) (rng() & 0xff);
        uint32_t d = 0;
        for (int64_t r = 0; r < rows; ++r)
            for (int b = 0; b < PLE_HEAD_DIM / 32; ++b, ++d) {
                const uint16_t h = (uint16_t) (d & 0xffff);
                std::memcpy(&raw[(size_t) r * PLE_ROW_BYTES + (size_t) b * 18], &h, 2);
            }
        bad += check(raw, rows, "every fp16 scale pattern, random codes");
    }
    {   // random rows
        const int64_t rows = 4096 + 37;
        std::vector<uint8_t> raw((size_t) rows * PLE_ROW_BYTES);
        for (auto& b : raw) b = (uint8_t) (rng() & 0xff);
        bad += check(raw, rows, "random bytes (odd row count)");
    }
    {   // the out-of-table row of PleTable::gather_raw_iq4nl
        const int64_t rows = 16;
        std::vector<uint8_t> raw((size_t) rows * PLE_ROW_BYTES);
        for (int64_t r = 0; r < rows; ++r)
            for (int b = 0; b < PLE_HEAD_DIM / 32; ++b) {
                uint8_t* blk = &raw[(size_t) r * PLE_ROW_BYTES + (size_t) b * 18];
                std::memset(blk, 0, 2);
                std::memset(blk + 2, 0x88, 16);
            }
        bad += check(raw, rows, "out-of-table row (scale 0, code 8)");
        // and it is +0.0f, not -0.0f, on both sides
        std::vector<float> want(PLE_HEAD_DIM);
        iq4nl_dequant_row(raw.data(), want.data());
        for (float v : want) {
            uint32_t a;
            std::memcpy(&a, &v, 4);
            if (a != 0) { std::printf("  *** the out-of-table row does not decode to +0.0f\n"); ++bad; break; }
        }
    }
    std::printf("ple_iq4nl_dequant_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
