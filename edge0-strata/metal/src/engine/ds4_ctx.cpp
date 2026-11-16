// metal/src/engine/ds4_ctx.cpp - the one-Context allocator the dsv41 engine can opt into.
//
// Why: a Metal kernel must be handed the gpuAddress of a buffer that the SAME mtl::Context encodes the dispatch for (its
// residency set).  strata::gpu, which metal_compat's cudaMalloc maps to, is a CUDA-shaped facade with its own device state, so a
// dsv41 launch would read a foreign residency set.  This file gives metal_compat an alternative: allocate inside one Context and
// remember gpuAddress -> Buffer, so wrap()/Arg::B can adopt the memory without a copy.
//
// Off by default: the engine's default path stays exactly as it was.  Set STRATA_DS4_METAL_CTX=1 to route allocations here -
// which becomes testable the moment a pack exists (the engine allocates only after it has read ds4.txt).
#include "mtl_rt.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace strata::dsv41::metal_ctx {

namespace {
struct Entry {
    mtl::Buffer buf;
    size_t bytes = 0;
};

std::mutex g_mu;
std::unordered_map<void*, Entry>* g_map = nullptr;   // gpuAddress -> buffer
mtl::Context* g_cx = nullptr;

bool enabled() {
    static const bool on = std::getenv("STRATA_DS4_METAL_CTX") != nullptr;
    return on;
}

mtl::Context& cx() {
    if (!g_cx) g_cx = new mtl::Context();
    return *g_cx;
}
}  // namespace

bool active() { return enabled(); }

void* alloc(size_t bytes) {
    if (!enabled()) return nullptr;
    const size_t page = 16384;
    const size_t rounded = (bytes + page - 1) / page * page;
    mtl::Buffer b = cx().alloc(rounded);
    if (!b.handle || !b.ptr) {
        std::fprintf(stderr, "dsv41 metal: Context::alloc(%zu) failed\n", bytes);
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_map) g_map = new std::unordered_map<void*, Entry>();
    (*g_map)[b.ptr] = Entry{b, rounded};
    return b.ptr;   // the CPU address doubles as the handle's key; the device pointer is b.handle's gpuAddress
}

void* host_view_of(void* p) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_map) return nullptr;
    auto it = g_map->find(p);
    return it == g_map->end() ? nullptr : it->second.buf.ptr;
}

const mtl::Buffer* buffer_of(void* p) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_map) return nullptr;
    auto it = g_map->find(p);
    return it == g_map->end() ? nullptr : &it->second.buf;
}

void release(void* p) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_map) return;
    auto it = g_map->find(p);
    if (it == g_map->end()) return;
    cx().free(it->second.buf);
    g_map->erase(it);
}

}  // namespace strata::dsv41::metal_ctx
