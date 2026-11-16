// src/kernels/latent_tables_check.cpp - parity/unit: LatentTables (per-device U + labels upload).
//
// Usage:  STRATA_LATENT_DENSE_FMT=8 latent_tables_check <dense.bin> <ELL> <FMT>
//   dense.bin : a real latent_dense_q{4,8}.bin (the file already matches n_layers=48, G=8, H=2560)
//   ELL       : latent dim (1792 or 1280)
//   FMT       : 8 (Q8_0) or 4 (Q4_0), must match the file's format
//
// Checks, in order:
//   1. ensure(0, layer 0) uploads U[0] and the labels table, and both re-download byte-identically.
//   2. ensure(0, layer 1) re-uses the single U slot (u_dev(0,0) is dropped) and U[1] matches.
//   3. fmt() reports the pack's format.
//   4. (two GPUs) device 1 gets its OWN labels and U pointers, independent of device 0.
// Exit 0 = pass, non-zero = fail (prints the failing step).
#include "strata/core/latent_dense.hpp"
#include "strata/core/latent_tables.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
int g_n_layers = 48, g_G = 8, g_H = 2560, g_E = 512;

struct On {
    int prev = 0;
    explicit On(int dev) { cudaGetDevice(&prev); if (prev != dev) cudaSetDevice(dev); }
    ~On() { int cur = 0; cudaGetDevice(&cur); if (cur != prev) cudaSetDevice(prev); }
};

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return false;
}

std::string step = "";

void fail(const std::string& why) {
    std::fprintf(stderr, "FAIL [%s]: %s\n", step.c_str(), why.c_str());
    std::exit(1);
}

std::vector<int32_t> make_labels() {
    std::vector<int32_t> v((size_t) g_n_layers * g_E);
    for (size_t i = 0; i < v.size(); ++i) v[i] = (int32_t) ((i * 7) % g_G);
    return v;
}

bool blob_equal(const void* a, const void* b, size_t n) { return std::memcmp(a, b, n) == 0; }

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: latent_tables_check <dense.bin> <ELL> <FMT>\n");
        return 2;
    }
    const std::string dense_path = argv[1];
    const int64_t ELL = std::atoll(argv[2]);
    const int fmt = std::atoi(argv[3]);
    if (fmt != 4 && fmt != 8) { std::fprintf(stderr, "FMT must be 4 or 8\n"); return 2; }

    // LatentDense::open reads the format from STRATA_LATENT_DENSE_FMT, so pin it to the file under test.
    setenv("STRATA_LATENT_DENSE_FMT", std::to_string(fmt).c_str(), 1);

    int count = 0;
    if (!ck(cudaGetDeviceCount(&count), "cudaGetDeviceCount")) return 1;

    strata::core::LatentDense dense;
    std::string err;
    if (!dense.open(dense_path, g_n_layers, g_G, g_H, ELL, err)) {
        std::fprintf(stderr, "open %s: %s\n", dense_path.c_str(), err.c_str());
        return 2;
    }
    const auto labels = make_labels();

    strata::core::LatentTables tab;
    tab.bind(&dense, labels.data(), g_n_layers, g_E, g_H, ELL);

    // ---- 1. device 0, layer 0 ----
    step = "ensure(0,0)";
    if (!tab.ensure(0, 0, err)) fail(err);
    if (tab.fmt() != fmt) fail("fmt() mismatch");
    if (tab.u_dev(0, 0) == nullptr) fail("u_dev(0,0) null");
    if (tab.labels_dev(0) == nullptr) fail("labels_dev(0) null");

    step = "U0 roundtrip";
    {
        On on(0);
        std::vector<uint8_t> back(dense.u_bytes());
        if (!ck(cudaMemcpy(back.data(), tab.u_dev(0, 0), back.size(), cudaMemcpyDeviceToHost), "U0 D2H")) return 1;
        if (!blob_equal(back.data(), dense.u(0), back.size())) fail("U[0] differs from the dense file");
    }
    step = "labels roundtrip (dev 0)";
    {
        On on(0);
        std::vector<int32_t> back((size_t) g_n_layers * g_E);
        if (!ck(cudaMemcpy(back.data(), tab.labels_dev(0), back.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
           "labels D2H")) return 1;
        if (!blob_equal(back.data(), labels.data(), back.size() * sizeof(int32_t))) fail("labels differ");
    }

    // ---- 2. same device, next layer re-uses the slot ----
    step = "ensure(0,1)";
    if (!tab.ensure(0, 1, err)) fail(err);
    if (tab.u_dev(0, 1) == nullptr) fail("u_dev(0,1) null");
    if (tab.u_dev(0, 0) != nullptr) fail("single-slot did not drop layer 0");
    step = "U1 roundtrip";
    {
        On on(0);
        std::vector<uint8_t> back(dense.u_bytes());
        if (!ck(cudaMemcpy(back.data(), tab.u_dev(0, 1), back.size(), cudaMemcpyDeviceToHost), "U1 D2H")) return 1;
        if (!blob_equal(back.data(), dense.u(1), back.size())) fail("U[1] differs");
    }

    // ---- 3. a second device is independent ----
    if (count >= 2) {
        step = "ensure(1,0)";
        if (!tab.ensure(1, 0, err)) fail(err);
        if (tab.u_dev(1, 0) == nullptr) fail("u_dev(1,0) null");
        if (tab.labels_dev(1) == nullptr) fail("labels_dev(1) null");
        if (tab.labels_dev(1) == tab.labels_dev(0)) fail("labels pointer shared across devices");
        step = "dev1 U0 roundtrip";
        {
            On on(1);
            std::vector<uint8_t> back(dense.u_bytes());
            if (!ck(cudaMemcpy(back.data(), tab.u_dev(1, 0), back.size(), cudaMemcpyDeviceToHost), "dev1 U0 D2H")) return 1;
            if (!blob_equal(back.data(), dense.u(0), back.size())) fail("dev1 U[0] differs");
        }
        // dev 0 must still hold layer 1, untouched by dev 1's ensure.
        if (tab.u_dev(0, 1) == nullptr) fail("dev0's layer-1 slot was disturbed by dev1");
    }

    tab.close();
    std::printf("PASS: latent_tables (ELL=%lld fmt=%d, %s)\n", (long long) ELL, fmt,
                count >= 2 ? "2 devices" : "1 device");
    return 0;
}