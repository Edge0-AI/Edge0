// src/kernels/latent_layer0_verify.cpp - layer0 latent-MoE pipeline on the real checkpoint.
//
//     latent_layer0_verify <latent_layer0_test.bin> <experts_layer0_native.bin>
//
// The A pack must be in the native block_nvfp4 nibble layout (d[4]+qs[32],
// sub-block low/high split), not a verbatim copy of the checkpoint's even/odd
// q bytes; make_native_pack.py does that permutation.
//
// D (Q8_0) -> q8_1 -> A (native NVFP4 grouped experts) -> U (Q8_0) -> combine,
// against the Python float32 reference written in the test bundle.  This is the
// first module of the latent adaptation: kernels + routing + two-stage output,
// without the Strata verify/cache integration yet.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

template<typename T>
std::vector<T> read_n(std::ifstream& f, size_t n, const char* what) {
    std::vector<T> v(n);
    if (!f.read((char*) v.data(), (std::streamsize) (n * sizeof(T)))) {
        std::fprintf(stderr, "short read: %s\n", what);
        std::exit(1);
    }
    return v;
}

double rel_l2(const std::vector<float>& got, const std::vector<float>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double) got[i] - (double) ref[i];
        num += d * d;
        den += (double) ref[i] * (double) ref[i];
    }
    return std::sqrt(num / (den + 1e-30));
}

float max_abs(const std::vector<float>& got, const std::vector<float>& ref) {
    float m = 0;
    for (size_t i = 0; i < got.size(); ++i) m = std::fmax(m, std::fabs(got[i] - ref[i]));
    return m;
}

void report(const char* name, const std::vector<float>& got, const std::vector<float>& ref) {
    std::printf("  %-4s relL2 %.4f%%  max_abs %.6g\n", name, 100 * rel_l2(got, ref), max_abs(got, ref));
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s test.bin experts_layer0.bin\n", argv[0]); return 2; }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "LAT0", 4) != 0) { std::fprintf(stderr, "bad magic\n"); return 2; }
    int32_t hdr[7];
    if (!f.read((char*) hdr, sizeof hdr)) return 2;
    const int T = hdr[0], K = hdr[1], G = hdr[2], E = hdr[3], H = hdr[4], ELL = hdr[5], I = hdr[6];
    std::printf("T=%d K=%d G=%d E=%d H=%d ELL=%d I=%d\n", T, K, G, E, H, ELL, I);
    if (T * K > 128) { std::fprintf(stderr, "cap too large for this test\n"); return 2; }

    const auto x = read_n<float>(f, (size_t) T * H, "x");
    const auto topk = read_n<int32_t>(f, (size_t) T * K, "topk");
    const auto w = read_n<float>(f, (size_t) T * K, "w");
    const auto labels = read_n<int32_t>(f, (size_t) E, "labels");
    const size_t drow = (size_t) H / 32 * 34, urow = (size_t) ELL / 32 * 34;
    const auto dq8 = read_n<uint8_t>(f, (size_t) G * ELL * drow, "Dq8");
    const auto uq8 = read_n<uint8_t>(f, (size_t) G * H * urow, "Uq8");
    const auto ref_xl = read_n<float>(f, (size_t) T * G * ELL, "ref_xl");
    const auto ref_yl = read_n<float>(f, (size_t) T * K * ELL, "ref_yl");
    const auto ref_y = read_n<float>(f, (size_t) T * K * H, "ref_y");
    const auto ref_out = read_n<float>(f, (size_t) T * H, "ref_out");

    const k::NativeExpertLayout L = k::native_expert_layout(40, 40, ELL, I);
    if (!k::native_expert_supported(40, 40, ELL, I)) { std::fprintf(stderr, "native layout unsupported\n"); return 2; }
    std::ifstream af(argv[2], std::ios::binary | std::ios::ate);
    if (!af) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
    const size_t asz = (size_t) af.tellg();
    if (asz != (size_t) E * L.bytes) {
        std::fprintf(stderr, "A pack size %zu != %zu\n", asz, (size_t) E * L.bytes);
        return 2;
    }
    af.seekg(0);
    auto ablob = read_n<uint8_t>(af, asz, "A blobs");

    const int32_t cap = T * K;
    // route plan grouped by expert: entries are the route indices, activations are token*G+group
    std::vector<int32_t> start(E + 1, 0), dst(cap), tok(cap), pos(E);
    std::vector<int32_t> counts(E, 0);
    for (int32_t r = 0; r < cap; ++r) counts[topk[r]]++;
    for (int e = 0; e < E; ++e) start[e + 1] = start[e] + counts[e];
    pos = start;
    for (int32_t r = 0; r < cap; ++r) {
        const int e = topk[r];
        const int p = pos[e]++;
        dst[p] = r;
        tok[p] = (r / K) * G + labels[e];
    }

    void *dxq = nullptr, *dxlq = nullptr, *dscr = nullptr;
    float *dx = nullptr, *dD = nullptr, *dU = nullptr, *dXl = nullptr, *dYl = nullptr, *dY = nullptr;
    float *dUIn = nullptr, *dUOut = nullptr;
    void* dUq = nullptr;
    uint8_t* dA = nullptr;
    unsigned long long* dPtr = nullptr;
    int32_t *dStart = nullptr, *dN = nullptr, *dDst = nullptr, *dTok = nullptr;

    ck(cudaMalloc(&dx, (size_t) T * H * 4), "x");
    ck(cudaMalloc(&dxq, (size_t) T * (H / 32) * 36), "xq");
    ck(cudaMalloc(&dD, (size_t) G * ELL * drow), "D");
    ck(cudaMalloc(&dU, (size_t) G * H * urow), "U");
    ck(cudaMalloc(&dXl, (size_t) T * G * ELL * 4), "xl");
    ck(cudaMalloc(&dxlq, (size_t) T * G * (ELL / 32) * 36), "xlq");
    ck(cudaMalloc(&dA, asz), "A");
    ck(cudaMalloc(&dPtr, (size_t) E * 8), "ptr");
    ck(cudaMalloc(&dStart, (size_t) (E + 1) * 4), "start");
    ck(cudaMalloc(&dN, 4), "n");
    ck(cudaMalloc(&dDst, (size_t) cap * 4), "dst");
    ck(cudaMalloc(&dTok, (size_t) cap * 4), "tok");
    ck(cudaMalloc(&dYl, (size_t) cap * ELL * 4), "yl");
    ck(cudaMalloc(&dY, (size_t) cap * H * 4), "y");
    ck(cudaMalloc(&dUIn, (size_t) cap * ELL * 4), "uin");
    ck(cudaMalloc(&dUOut, (size_t) cap * H * 4), "uout");
    ck(cudaMalloc(&dUq, (size_t) cap * (ELL / 32) * 36), "uq");
    ck(cudaMalloc(&dscr, k::native_expert_scratch_bytes(cap, I)), "scratch");

    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(dD, dq8.data(), dq8.size(), cudaMemcpyHostToDevice), "D");
    ck(cudaMemcpy(dU, uq8.data(), uq8.size(), cudaMemcpyHostToDevice), "U");
    ck(cudaMemcpy(dA, ablob.data(), ablob.size(), cudaMemcpyHostToDevice), "A");
    std::vector<unsigned long long> ptr(E);
    for (int e = 0; e < E; ++e) ptr[e] = (unsigned long long) (dA + (size_t) e * L.bytes);
    const int32_t n_groups = E;
    ck(cudaMemcpy(dPtr, ptr.data(), ptr.size() * 8, cudaMemcpyHostToDevice), "ptr");
    ck(cudaMemcpy(dStart, start.data(), start.size() * 4, cudaMemcpyHostToDevice), "start");
    ck(cudaMemcpy(dN, &n_groups, 4, cudaMemcpyHostToDevice), "n");
    ck(cudaMemcpy(dDst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "dst");
    ck(cudaMemcpy(dTok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice), "tok");

    // D: [T,H] x [G*ELL,H]^T -> [T,G*ELL]
    k::quantize_q8_1_rows(dx, T, H, dxq, nullptr);
    ck(cudaDeviceSynchronize(), "D quantize");
    k::iq_mmvq(8, dD, dxq, dXl, H, G * ELL, T, nullptr);
    ck(cudaDeviceSynchronize(), "D");
    k::quantize_q8_1_rows(dXl, T * G, ELL, dxlq, nullptr);
    ck(cudaDeviceSynchronize(), "xl quantize");

    // A: every expert is a group; entries are grouped by expert, activation row = token*G+group.
    k::native_expert_grouped(L, dPtr, dStart, dN, dDst, dTok, E, cap, dxlq, dscr, dYl, nullptr, E);
    ck(cudaDeviceSynchronize(), "A");
    // U: per group, gather the route rows, Q8 GEMM, scatter back.
    for (int g = 0; g < G; ++g) {
        std::vector<int32_t> routes;
        for (int r = 0; r < cap; ++r) if (labels[topk[r]] == g) routes.push_back(r);
        const int n = (int) routes.size();
        if (!n) continue;
        for (int i = 0; i < n; ++i)
            ck(cudaMemcpy(dUIn + (size_t) i * ELL, dYl + (size_t) routes[i] * ELL, (size_t) ELL * 4,
                          cudaMemcpyDeviceToDevice), "gather");
        k::quantize_q8_1_rows(dUIn, n, ELL, dUq, nullptr);
        ck(cudaDeviceSynchronize(), "U quantize");
        k::iq_mmvq(8, (const uint8_t*) dU + (size_t) g * H * urow, dUq, dUOut, ELL, H, n, nullptr);
        ck(cudaDeviceSynchronize(), "U");
        for (int i = 0; i < n; ++i)
            ck(cudaMemcpy(dY + (size_t) routes[i] * H, dUOut + (size_t) i * H, (size_t) H * 4,
                          cudaMemcpyDeviceToDevice), "scatter");
    }
    ck(cudaDeviceSynchronize(), "run");

    std::vector<float> xl((size_t) T * G * ELL), yl((size_t) cap * ELL), y((size_t) cap * H), out((size_t) T * H, 0.f);
    ck(cudaMemcpy(xl.data(), dXl, xl.size() * 4, cudaMemcpyDeviceToHost), "xl");
    ck(cudaMemcpy(yl.data(), dYl, yl.size() * 4, cudaMemcpyDeviceToHost), "yl");
    ck(cudaMemcpy(y.data(), dY, y.size() * 4, cudaMemcpyDeviceToHost), "y");
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < K; ++j)
            for (int h = 0; h < H; ++h)
                out[(size_t) t * H + h] += w[(size_t) t * K + j] * y[(size_t) (t * K + j) * H + h];

    std::printf("latent layer0 (D Q8 -> A NVFP4 -> U Q8):\n");
    report("xl", xl, ref_xl);
    report("yl", yl, ref_yl);
    report("y", y, ref_y);
    report("out", out, ref_out);
    const double eo = rel_l2(out, ref_out);
    std::printf("RESULT: %s (combine relL2 %.3f%%)\n", eo < 0.03 ? "ok" : "TOO FAR", 100 * eo);

    for (void* p : {(void*) dx, dxq, (void*) dD, (void*) dU, (void*) dXl, dxlq, (void*) dA, (void*) dPtr,
                    (void*) dStart, (void*) dN, (void*) dDst, (void*) dTok, (void*) dYl, (void*) dY,
                    (void*) dUIn, (void*) dUOut, dUq, dscr})
        cudaFree(p);
    return eo < 0.03 ? 0 : 1;
}
