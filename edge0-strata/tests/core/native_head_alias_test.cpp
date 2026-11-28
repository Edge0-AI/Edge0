// tests/core/native_head_alias_test.cpp - the memory contract --prefill-head-alias needs from the output head.
//
// The head is unloaded while each fresh prompt is read (its VRAM is lent to the prompt path), then reloaded before
// the first decode window.  The verify windows' CUDA graphs are captured during that decode and hold the head's
// addresses, so a reload that lands elsewhere would run them against freed memory.  VmmBuffer keeps the addresses
// reserved while the physical memory goes back to the pool, so the reload lands on them even while another
// allocation (the prompt path's own buffers, an elastic K/V growth) holds physical memory taken from the lend.  A
// cudaMalloc block makes no such promise: that is what stopped a session with
// "ERR --prefill-head-alias: the output head was reloaded at another address".
#include "strata/core/vmm.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

using strata::core::VmmBuffer;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
size_t free_vram() {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    return free_b;
}
// The free-memory counter may lag a release by a moment (vmm_test waits the same way, for WDDM's sake).
bool wait_free_at_least(size_t target) {
    for (int i = 0; i < 20; ++i) {
        if (free_vram() >= target) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}
}  // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess || !strata::core::vmm_available()) {
        std::printf("native_head_alias_test: no CUDA device with virtual memory - skipped\n");
        return 0;
    }
    const uint64_t gran = strata::core::vmm_granularity();
    const uint64_t bytes = 8 * gran;
    VmmBuffer head;
    check(head.ensure(bytes) && head.reserved() && head.bytes() >= bytes, "the addresses are reserved once");
    check(head.map() && head.mapped(), "the head's physical memory is mapped at them");
    const void* first = head.base();
    const size_t mapped = free_vram();
    std::vector<uint8_t> pattern((size_t) bytes);
    for (size_t i = 0; i < pattern.size(); ++i) pattern[i] = (uint8_t) (i * 131u + 7u);
    check(cudaMemcpy(head.base(), pattern.data(), (size_t) bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaDeviceSynchronize() == cudaSuccess,
          "the head is uploaded and the copy lands");
    head.unmap();
    check(!head.mapped() && head.base() == first, "unload frees the physical memory and keeps the addresses");
    const bool lent_back = wait_free_at_least(mapped + bytes);
    check(lent_back, "the lend returns the VRAM (the prompt path can really use it)");
    // Between the unload and the reload, a running session allocates: the prompt path's own buffers, or the elastic
    // K/V growing with the prompt.  Those allocations take VRAM from the lend; a cudaMalloc'd head would now come
    // back wherever the free list puts it.
    void* squatter = nullptr;
    const bool squatted = cudaMalloc(&squatter, (size_t) bytes) == cudaSuccess;
    check(squatted, "another allocation takes VRAM between the unload and the reload");
    check(head.map() && head.mapped(), "the reload maps physical memory again");
    check(head.base() == first, "the reload lands on the addresses the verify graphs captured");
    std::vector<uint8_t> back((size_t) bytes);
    check(cudaMemcpy(back.data(), head.base(), (size_t) bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
          "the reloaded head reads back");
    if (squatted) cudaFree(squatter);
    check(cudaMemset(head.base(), 0x5a, (size_t) bytes) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess,
          "the reloaded head is writable by the kernels");
    head.unmap();
    check(!head.mapped() && head.base() == first, "a second lend keeps the addresses too");
    check(wait_free_at_least(mapped), "the second lend returns the VRAM again");
    head.release();
    check(!head.reserved() && head.base() == nullptr, "release gives the addresses back");
    check(cudaGetLastError() == cudaSuccess, "no CUDA error was left behind");
    std::printf("native_head_alias_test: %s\n", g_fail ? "FAILED" : "ok");
    return g_fail ? 1 : 0;
}
