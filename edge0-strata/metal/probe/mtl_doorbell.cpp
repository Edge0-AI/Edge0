// mtl_doorbell.cpp - can Strata's doorbell protocol run on Metal?
//  A. in-kernel spin: a running kernel polls a shared-memory word that the CPU writes later (the CUDA doorbell model)
//  B. MTLSharedEvent ping-pong: the GPU signals "routing ready" (value 2l+1), the host sees it, does its part, sets
//     2l+2 from the CPU, and the GPU (encodeWait inside the same, already committed command buffer) goes on.
//     Measures the round trip per "layer".
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include <algorithm>

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct SpinArgs { device atomic_uint* flag; device uint* out; uint want; uint max_iter; };
kernel void spin(constant SpinArgs& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i != 0) return;
    uint it = 0;
    while (atomic_load_explicit(p.flag, memory_order_relaxed) != p.want && it < p.max_iter) ++it;
    p.out[0] = it;
    p.out[1] = atomic_load_explicit(p.flag, memory_order_relaxed);
}
struct TickArgs { device uint* x; };
kernel void tick(constant TickArgs& p [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0) p.x[0] += 1; }
)MSL";
struct SpinArgs { uint64_t flag, out; uint32_t want, max_iter; };
struct TickArgs { uint64_t x; };

int main() {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* dev = MTL::CreateSystemDefaultDevice();
    NS::Error* e = nullptr;
    MTL::Library* lib = dev->newLibrary(NS::String::string(kSrc, NS::UTF8StringEncoding), nullptr, &e);
    if (!lib) { std::printf("compile failed\n"); return 1; }
    auto pipe = [&](const char* n) { return dev->newComputePipelineState(lib->newFunction(NS::String::string(n, NS::UTF8StringEncoding)), &e); };
    MTL::ComputePipelineState* p_spin = pipe("spin");
    MTL::ComputePipelineState* p_tick = pipe("tick");
    MTL::CommandQueue* q = dev->newCommandQueue();
    MTL::Buffer* bf = dev->newBuffer(16384, MTL::ResourceStorageModeShared);
    MTL::Buffer* bo = dev->newBuffer(16384, MTL::ResourceStorageModeShared);
    MTL::ResidencySet* rs = dev->newResidencySet(MTL::ResidencySetDescriptor::alloc()->init(), &e);
    rs->addAllocation(bf); rs->addAllocation(bo); rs->commit(); q->addResidencySet(rs);
    volatile uint32_t* flag = (volatile uint32_t*) bf->contents();
    uint32_t* out = (uint32_t*) bo->contents();

    // ---- A: in-kernel spin on a word the CPU writes 5 ms after commit
    for (int rep = 0; rep < 3; ++rep) {
        *flag = 0; out[0] = out[1] = 0;
        SpinArgs a{bf->gpuAddress(), bo->gpuAddress(), 7, 200000000u};
        MTL::CommandBuffer* cb = q->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(p_spin);
        enc->setBytes(&a, sizeof a, 0);
        enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
        enc->endEncoding();
        const double t0 = now_us();
        cb->commit();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *flag = 7;
        const double tw = now_us();
        cb->waitUntilCompleted();
        std::printf("A in-kernel spin: GPU saw the CPU store: %s (iterations %u, flag %u), done %.0f us after the store, status %d\n",
                    out[1] == 7 ? "YES" : "NO (timed out)", out[0], out[1], now_us() - tw, (int) cb->status());
        (void) t0;
    }

    // ---- B: shared-event ping-pong, L "layers" in ONE committed command buffer
    for (int rep = 0; rep < 3; ++rep) {
        const int L = 48;
        MTL::SharedEvent* ev = dev->newSharedEvent();
        ev->setSignaledValue(0);
        out[0] = 0;
        MTL::CommandBuffer* cb = q->commandBuffer();
        TickArgs ta{bo->gpuAddress()};
        for (int l = 0; l < L; ++l) {
            MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
            enc->setComputePipelineState(p_tick);
            enc->setBytes(&ta, sizeof ta, 0);
            enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
            enc->endEncoding();
            cb->encodeSignalEvent(ev, 2 * l + 1);   // "the miss list of layer l is ready"
            cb->encodeWait(ev, 2 * l + 2);          // "the host has answered layer l"
        }
        std::vector<double> rt;
        const double t0 = now_us();
        cb->commit();
        for (int l = 0; l < L; ++l) {
            while (ev->signaledValue() < (uint64_t) (2 * l + 1)) {}
            const double ts = now_us();
            if (out[0] != (uint32_t) (l + 1)) std::printf("  layer %d: host read %u (stale?)\n", l, out[0]);
            ev->setSignaledValue(2 * l + 2);
            rt.push_back(ts);
        }
        cb->waitUntilCompleted();
        const double total = now_us() - t0;
        std::vector<double> gaps;
        for (int l = 1; l < L; ++l) gaps.push_back(rt[l] - rt[l - 1]);
        std::sort(gaps.begin(), gaps.end());
        std::printf("B shared-event round trip: %d layers in %.0f us, per layer median %.1f us (min %.1f, max %.1f), ticks %u, status %d\n",
                    L, total, gaps[gaps.size() / 2], gaps.front(), gaps.back(), out[0], (int) cb->status());
        ev->release();
    }
    pool->release();
    return 0;
}
