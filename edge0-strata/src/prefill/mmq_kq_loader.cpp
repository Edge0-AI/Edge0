// src/prefill/mmq_kq_loader.cpp - opens strata_kq (src/prefill/mmq_kq_lib.cu), the library with the K-quant MMQ prompt
// kernels, from the engine's own folder - once, for the first model format that asks for it (moe_mmq.cu's supported()),
// which the engine does before it sizes the expert cache (src/program/generate.cpp).  Without the library, or with an
// interface of another version, the K-quants keep the FP16 prompt path and the engine says so once.
#include "strata/prefill/mmq_kq_abi.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace strata::prefill::mmq::kq {
namespace {

struct Lib {
    bool ok = false;
    strata_kq_run_fn run = nullptr;
};

std::string own_folder() {
#if defined(_WIN32)
    char p[MAX_PATH] = {};
    const DWORD n = GetModuleFileNameA(nullptr, p, MAX_PATH);
    std::string s(p, n);
    const size_t k = s.find_last_of("\\/");
    return k == std::string::npos ? std::string() : s.substr(0, k + 1);
#else
    char p[4096] = {};
    const ssize_t n = readlink("/proc/self/exe", p, sizeof(p) - 1);
    std::string s(p, n > 0 ? (size_t) n : 0);
    const size_t k = s.find_last_of('/');
    return k == std::string::npos ? std::string() : s.substr(0, k + 1);
#endif
}

void* sym(void* h, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress((HMODULE) h, name));
#else
    return dlsym(h, name);
#endif
}

Lib open_lib() {
    Lib L;
    const std::string path = own_folder() + STRATA_KQ_LIB_NAME;
#if defined(_WIN32)
    void* h = (void*) LoadLibraryA(path.c_str());
#else
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (h == nullptr) {
        std::fprintf(stderr, "strata: K-quant prompt kernels (STRATA_MMQ_KQ=1): %s not found next to the engine - "
                             "these experts take the FP16 prompt path\n", STRATA_KQ_LIB_NAME);
        return L;
    }
    const auto abi = (strata_kq_abi_fn) sym(h, "strata_kq_abi");
    const auto init = (strata_kq_init_fn) sym(h, "strata_kq_init");
    L.run = (strata_kq_run_fn) sym(h, "strata_kq_run");
    if (abi == nullptr || init == nullptr || L.run == nullptr || abi() / 100000 != STRATA_KQ_ABI_VERSION ||
        abi() % 100000 != abi_args_size()) {
        std::fprintf(stderr, "strata: K-quant prompt kernels: %s is from another engine version - these experts take "
                             "the FP16 prompt path\n", path.c_str());
        L.run = nullptr;
        return L;
    }
    // every GPU this engine sees: each context loads the library's kernels now, while the VRAM is still free
    int n = 0, cur = 0;
    cudaGetDevice(&cur);
    if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
    for (int d = 0; d < n; ++d) {
        const int e = init(d);
        if (e != 0) {
            std::fprintf(stderr, "strata: K-quant prompt kernels: %s did not start on GPU %d (CUDA error %d) - these "
                                 "experts take the FP16 prompt path\n", path.c_str(), d, e);
            L.run = nullptr;
            cudaSetDevice(cur);
            return L;
        }
    }
    cudaSetDevice(cur);
    std::fprintf(stderr, "strata: K-quant prompt kernels from %s (STRATA_MMQ_KQ=1)\n", path.c_str());
    L.ok = true;
    return L;
}

const Lib& lib() {
    static const Lib L = open_lib();
    return L;
}

}  // namespace

bool wanted() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_MMQ_KQ");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

bool available() { return wanted() && lib().ok; }

int run(int type, const void* args, void* stream) {
    int dev = 0;
    cudaGetDevice(&dev);
    return lib().ok ? lib().run(dev, type, args, stream) : -1;
}

}  // namespace strata::prefill::mmq::kq
