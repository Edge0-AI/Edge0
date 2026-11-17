// src/kernels/b6_mmvq_bench.cpp - B6: what a verify row costs in the dense projections, per kernel, per card.
//
//     build/b6_mmvq_bench [reps] [device, -1 = all] [B rows mode: 1 = 2 rows, or 2 / 4 / 8]
//
// The decode's dense projections at the model's own shapes and formats (qwen3.8-flash-next GSQ-RCO IQ3_XXS:
// GDN qkv / gate / out, QSA q / out, shared expert, the head; the BF16 router), called with ncols = 1..5 and 8 as a
// verify window of that many rows calls them.  Per call: CUDA events around each launch, every launch queued back to
// back (no host gap inside a measurement), the median of `reps` (default 500) after a warm-up.  The weights rotate
// over enough copies to exceed the L2, as the decode reads each layer's weights cold.  Synthetic bytes with sane
// block scales (as mmvq_multi_parity): the time does not depend on the values.
//
// Two layouts side by side: the default exact layout (A) and STRATA_B6_MMVQ_ROWS (B, native_mmvq_set_b6_rows).  For
// every case and ncols: B's output against A's, bit for bit (memcmp), and both against ncols single-column calls.
// Exit code 1 if any comparison differs.  GPU time only: immune to the host and to any generated text.
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

namespace {

struct Case {
    const char* name;
    int type;          // GGML type id; 30 = BF16 (bf16_gemv_fp32_mmvf_multi)
    int n_in, n_out;
    int block_elems, block_bytes;
    int scale_at[2];   // byte offsets of the fp16 scales in a block; -1 = none
};

const Case CASES[] = {
    {"gdn qkv Q6_K", 14, 2560, 10240, 256, 210, {208, -1}},
    {"gdn qkv Q4_K", 12, 2560, 10240, 256, 144, {0, 2}},
    {"gdn qkv Q5_K", 13, 2560, 10240, 256, 176, {0, 2}},
    {"gdn qkv IQ4_XS", 23, 2560, 10240, 256, 136, {0, -1}},
    {"gdn gate Q6_K", 14, 2560, 6144, 256, 210, {208, -1}},
    {"gdn out Q6_K", 14, 6144, 2560, 256, 210, {208, -1}},
    {"gdn out IQ4_XS", 23, 6144, 2560, 256, 136, {0, -1}},
    {"gdn out Q4_K", 12, 6144, 2560, 256, 144, {0, 2}},
    {"qsa q IQ4_XS", 23, 2560, 12288, 256, 136, {0, -1}},
    {"qsa q Q6_K", 14, 2560, 12288, 256, 210, {208, -1}},
    {"shexp up IQ3_S", 21, 2560, 640, 256, 110, {0, -1}},
    {"shexp gate Q4_K", 12, 2560, 640, 256, 144, {0, 2}},
    {"shexp down IQ4_NL", 20, 640, 2560, 32, 18, {0, -1}},
    {"router BF16", 30, 2560, 512, 1, 2, {-1, -1}},
    {"head Q5_K", 13, 2560, 248320, 256, 176, {0, 2}},
};
const int WIDTHS[] = {1, 2, 3, 4, 5, 8};

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}
uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

struct Buf {
    std::vector<void*> w;   // weight copies
    std::size_t wbytes = 0;
    void* xq = nullptr;     // Q8_1 of 8 columns
    float* xf = nullptr;    // the float activations (BF16 path)
    float* y = nullptr;     // 8 columns of output
};

// one call of the case at `ncols` on weight copy `k` into y
void call(const Case& c, const Buf& b, int k, int ncols, float* y, cudaStream_t s) {
    if (c.type == 30)
        K::bf16_gemv_fp32_mmvf_multi(b.xf, c.n_in, (const uint16_t*) b.w[(size_t) k], y, c.n_out, c.n_in, c.n_out, ncols, s);
    else
        K::native_mmvq(c.type, b.w[(size_t) k], b.xq, y, c.n_in, c.n_out, ncols, s);
}
// column j alone (ncols = 1), into y + j * n_out
void call_col(const Case& c, const Buf& b, int j, float* y, cudaStream_t s) {
    if (c.type == 30)
        K::bf16_gemv_fp32_mmvf_multi(b.xf + (size_t) j * c.n_in, c.n_in, (const uint16_t*) b.w[0],
                                     y + (size_t) j * c.n_out, c.n_out, c.n_in, c.n_out, 1, s);
    else
        K::native_mmvq(c.type, b.w[0], (const uint8_t*) b.xq + K::native_q8_1_bytes(c.n_in, 1) * (size_t) j,
                       y + (size_t) j * c.n_out, c.n_in, c.n_out, 1, s);
}

double median_ms(const Case& c, const Buf& b, int ncols, int reps, cudaStream_t s, std::vector<cudaEvent_t>& ev) {
    const int nk = (int) b.w.size();
    for (int r = 0; r < 30; ++r) call(c, b, r % nk, ncols, b.y, s);   // warm-up
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(ev[(size_t) 2 * r], s);
        call(c, b, r % nk, ncols, b.y, s);
        cudaEventRecord(ev[(size_t) 2 * r + 1], s);
    }
    cudaStreamSynchronize(s);
    std::vector<float> t((size_t) reps);
    for (int r = 0; r < reps; ++r) cudaEventElapsedTime(&t[(size_t) r], ev[(size_t) 2 * r], ev[(size_t) 2 * r + 1]);
    std::nth_element(t.begin(), t.begin() + reps / 2, t.end());
    return t[(size_t) reps / 2];
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int reps = argc > 1 ? std::max(50, std::atoi(argv[1])) : 500;
    int n_dev = 0;
    if (!ck(cudaGetDeviceCount(&n_dev), "device count")) return 1;
    const int only = argc > 2 ? std::atoi(argv[2]) : -1;
    const int bmode = argc > 3 ? std::atoi(argv[3]) : 1;
    long long bad = 0;
    for (int dev = 0; dev < n_dev; ++dev) {
        if (only >= 0 && dev != only) continue;
        cudaSetDevice(dev);
        cudaDeviceProp pr{};
        cudaGetDeviceProperties(&pr, dev);
        std::printf("\n=== device %d: %s (sm_%d%d, L2 %d KB), %d reps, median ms per call, B = rows mode %d ===\n", dev, pr.name,
                    pr.major, pr.minor, pr.l2CacheSize / 1024, reps, bmode);
        std::printf("%-18s %4s %9s %9s %8s %8s  %s\n", "case", "cols", "A ms", "B ms", "A +col", "B +col",
                    "bitwise B=A, A=1col, B=1col");
        cudaStream_t s;
        if (!ck(cudaStreamCreate(&s), "stream")) return 1;
        std::vector<cudaEvent_t> ev((size_t) 2 * reps);
        for (auto& e : ev) cudaEventCreate(&e);
        for (const Case& c : CASES) {
            if (c.type != 30 && !K::native_mmvq_supported(c.type)) {
                std::printf("%-18s unsupported type %d\n", c.name, c.type);
                continue;
            }
            std::mt19937 rng(77u + (unsigned) c.type * 131u + (unsigned) c.n_out);
            Buf b;
            b.wbytes = c.type == 30 ? (size_t) c.n_in * c.n_out * 2 : K::native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
            std::vector<uint8_t> w(b.wbytes);
            if (c.type == 30) {
                std::normal_distribution<float> nd(0.f, 0.05f);
                auto* p = (uint16_t*) w.data();
                for (size_t i = 0; i < b.wbytes / 2; ++i) p[i] = bf16(nd(rng));
            } else {
                for (auto& x : w) x = (uint8_t) (rng() & 0xff);
                const size_t nb = (size_t) c.n_out * (size_t) (c.n_in / c.block_elems);
                for (size_t k = 0; k < nb; ++k)
                    for (int at : c.scale_at)
                        if (at >= 0) {
                            const uint16_t h = sane_half(rng);
                            std::memcpy(&w[k * (size_t) c.block_bytes + (size_t) at], &h, 2);
                        }
            }
            const size_t copies = std::max<size_t>(1, std::min<size_t>(8, (96u << 20) / b.wbytes + 1));
            bool ok = true;
            for (size_t k = 0; k < copies && ok; ++k) {
                void* d = nullptr;
                ok = ck(cudaMalloc(&d, b.wbytes), "malloc w") && ck(cudaMemcpy(d, w.data(), b.wbytes, cudaMemcpyHostToDevice), "copy w");
                b.w.push_back(d);
            }
            std::vector<float> x((size_t) 8 * c.n_in);
            std::normal_distribution<float> nx(0.f, 1.f);
            for (auto& v : x) v = nx(rng);
            const size_t ybytes = (size_t) 8 * c.n_out * 4;
            ok = ok && ck(cudaMalloc(&b.xf, x.size() * 4), "malloc x") &&
                 ck(cudaMemcpy(b.xf, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x") &&
                 ck(cudaMalloc(&b.y, ybytes), "malloc y");
            if (ok && c.type != 30) {
                ok = ck(cudaMalloc(&b.xq, K::native_q8_1_bytes(c.n_in, 8)), "malloc xq");
                if (ok) K::quantize_q8_1_rows(b.xf, 8, c.n_in, b.xq, s);
            }
            float *ya = nullptr, *yb = nullptr, *y1 = nullptr;
            ok = ok && ck(cudaMalloc(&ya, ybytes), "ya") && ck(cudaMalloc(&yb, ybytes), "yb") && ck(cudaMalloc(&y1, ybytes), "y1");
            if (!ok) return 1;
            double a1 = 0, b1 = 0;
            for (int T : WIDTHS) {
                std::vector<uint32_t> ha((size_t) T * c.n_out), hb(ha.size()), h1(ha.size());
                K::native_mmvq_set_b6_rows(0);
                const double ta = median_ms(c, b, T, reps, s, ev);
                call(c, b, 0, T, ya, s);
                for (int j = 0; j < T; ++j) call_col(c, b, j, y1, s);
                K::native_mmvq_set_b6_rows(bmode);
                const double tb = median_ms(c, b, T, reps, s, ev);
                call(c, b, 0, T, yb, s);
                K::native_mmvq_set_b6_rows(0);
                if (!ck(cudaStreamSynchronize(s), "run")) return 1;
                cudaMemcpy(ha.data(), ya, ha.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(hb.data(), yb, hb.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(h1.data(), y1, h1.size() * 4, cudaMemcpyDeviceToHost);
                const bool ba = std::memcmp(ha.data(), hb.data(), ha.size() * 4) == 0;
                const bool a_1 = std::memcmp(ha.data(), h1.data(), ha.size() * 4) == 0;
                const bool b_1 = std::memcmp(hb.data(), h1.data(), hb.size() * 4) == 0;
                long long nonfinite = 0;
                for (uint32_t u : ha) { float f; std::memcpy(&f, &u, 4); if (!std::isfinite(f)) ++nonfinite; }
                if (!ba || !b_1 || nonfinite) ++bad;
                if (T == 1) { a1 = ta; b1 = tb; }
                std::printf("%-18s %4d %9.4f %9.4f %8.4f %8.4f  %s %s %s%s\n", c.name, T, ta, tb,
                            T > 1 ? (ta - a1) / (T - 1) : 0.0, T > 1 ? (tb - b1) / (T - 1) : 0.0, ba ? "ok" : "DIFF",
                            a_1 ? "ok" : "diff", b_1 ? "ok" : "DIFF", nonfinite ? " NONFINITE" : "");
            }
            for (void* d : b.w) cudaFree(d);
            cudaFree(b.xq); cudaFree(b.xf); cudaFree(b.y); cudaFree(ya); cudaFree(yb); cudaFree(y1);
        }
        for (auto& e : ev) cudaEventDestroy(e);
        cudaStreamDestroy(s);
    }
    std::printf("\nb6_mmvq_bench: %s (%lld case/width comparisons failed)\n", bad ? "FAILED" : "ok", bad);
    return bad ? 1 : 0;
}
