// src/kernels/dequant_bf16_test.cpp - plan v0.3 P5: the device dequantizers against the artifact's validated CPU
// dequantizers (`strata/artifact/dequant.hpp`), on one real tensor of every type the model uses.
//
//     dequant_bf16_test SHARD1.gguf [SHARD2.gguf]
//
// For each type: the first 4 rows of the first tensor of that type, FP32 path compared with a relative tolerance of
// 1e-6 (the products are the reference's, reordered at most), BF16 path within half a BF16 ulp of the CPU value.
#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

bool cpu_block(int type, const uint8_t* b, float* out) {
    using namespace strata;
    switch (type) {
    case 2: dequantize_q4_0(b, out); return true;
    case 6: dequantize_q5_0(b, out); return true;
    case 7: dequantize_q5_1(b, out); return true;
    case 8: dequantize_q8_0(b, out); return true;
    case 11: dequantize_q3_K(b, out); return true;
    case 12: dequantize_q4_K(b, out); return true;
    case 13: dequantize_q5_K(b, out); return true;
    case 14: dequantize_q6_K(b, out); return true;
    case 20: dequantize_iq4_nl(b, out); return true;
    case 23: dequantize_iq4_xs(b, out); return true;
    case 42: dequantize_q2_0(b, out); return true;
    default: return false;
    }
}

float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: dequant_bf16_test SHARD.gguf [SHARD2.gguf]\n");
        return 2;
    }
    std::map<int, bool> done;
    int fails = 0, checked = 0;
    for (int a = 1; a < argc; ++a) {
        strata::GgufFile gguf(argv[a]);
        for (const auto& t : gguf.tensors()) {
            const int type = (int) t.type;
            if (done.count(type) || !strata::kernels::dequant_bf16_supported(type) || t.shape.size() != 2) continue;
            int be = 0, bb = 0;
            if (!strata::block_geometry(t.type, be, bb)) continue;
            const int64_t cols = (int64_t) t.shape[0];
            const int64_t rows = (std::min<int64_t>)(4, (int64_t) t.shape[1]);
            const int64_t row_bytes = cols / be * bb;
            const uint8_t* host = (const uint8_t*) gguf.tensor_data(t);
            void* d_blocks = nullptr;
            float* d_f = nullptr;
            uint16_t* d_h = nullptr;
            cudaMalloc(&d_blocks, (size_t) (rows * row_bytes));
            cudaMalloc(&d_f, (size_t) (rows * cols) * 4);
            cudaMalloc(&d_h, (size_t) (rows * cols) * 2);
            cudaMemcpy(d_blocks, host, (size_t) (rows * row_bytes), cudaMemcpyHostToDevice);
            strata::kernels::dequant_f32(type, d_blocks, 0, rows, cols, d_f, nullptr);
            strata::kernels::dequant_bf16(type, d_blocks, 0, rows, cols, d_h, nullptr);
            std::vector<float> gf((size_t) (rows * cols));
            std::vector<uint16_t> gh((size_t) (rows * cols));
            cudaMemcpy(gf.data(), d_f, gf.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(gh.data(), d_h, gh.size() * 2, cudaMemcpyDeviceToHost);
            cudaFree(d_blocks);
            cudaFree(d_f);
            cudaFree(d_h);
            std::vector<float> ref((size_t) (rows * cols));
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t b = 0; b < cols / be; ++b)
                    cpu_block(type, host + r * row_bytes + b * bb, ref.data() + r * cols + b * be);
            int64_t bad_f = 0, bad_h = 0;
            double worst = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                const double e = std::fabs((double) gf[i] - ref[i]);
                const double tol = 1e-6 * std::fabs((double) ref[i]) + 1e-30;
                if (e > tol) ++bad_f;
                worst = (std::max)(worst, e / (std::fabs((double) ref[i]) + 1e-30));
                const double eh = std::fabs((double) bf16_to_f32(gh[i]) - ref[i]);
                if (eh > std::ldexp(std::fabs((double) ref[i]), -8) + 1e-30) ++bad_h;
            }
            ++checked;
            done[type] = true;
            const bool ok = bad_f == 0 && bad_h == 0;
            if (!ok) ++fails;
            std::printf("type %2d  %-40s %lld x %lld: f32 %s (%lld off, worst rel %.2e), bf16 %s (%lld off)\n", type,
                        t.name.c_str(), (long long) rows, (long long) cols, bad_f ? "FAIL" : "ok", (long long) bad_f,
                        worst, bad_h ? "FAIL" : "ok", (long long) bad_h);
        }
    }
    std::printf("dequant_bf16_test: %d types checked, %s\n", checked, fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
