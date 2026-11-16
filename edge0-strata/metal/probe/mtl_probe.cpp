// mtl_probe.cpp - M5 Max checks for the Metal device-interface design (mac-design stream).
//  1. device facts: unified memory, maxBufferLength, recommendedMaxWorkingSetSize, GPU families
//  2. MSL compiled at run time (no offline `metal` tool in the Command Line Tools)
//  3. a hello kernel that takes RAW GPU ADDRESSES in a setBytes struct (the CUDA "void* device pointer" model)
//  4. gpuAddress vs contents(): are they equal?
//  5. newBufferWithBytesNoCopy over an mmapped file > 30 GB: alignment, max length, split into chunks, GPU reads it
// Build: clang++ -std=c++20 -O2 -I metal-cpp mtl_probe.cpp -framework Metal -framework Foundation -framework QuartzCore
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct AxpyArgs { device const float* x; device float* y; float a; uint n; };
kernel void axpy(constant AxpyArgs& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) p.y[i] = p.a * p.x[i] + p.y[i];
}
// reads one uint32 at each probe offset of a (possibly huge) buffer given by raw GPU address
struct ProbeArgs { device const uint* base; device const ulong* offs; device uint* out; uint n; };
kernel void probe(constant ProbeArgs& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) p.out[i] = p.base[p.offs[i] / 4];
}
// streaming read of a buffer range: sum of uint32 (bandwidth through a NoCopy file buffer)
struct SumArgs { device const uint4* base; device atomic_uint* out; ulong n4; };
kernel void sum(constant SumArgs& p [[buffer(0)]], uint i [[thread_position_in_grid]], uint g [[threads_per_grid]]) {
    uint s = 0;
    for (ulong k = i; k < p.n4; k += g) { uint4 v = p.base[k]; s += v.x ^ v.y ^ v.z ^ v.w; }
    atomic_fetch_add_explicit(p.out, s, memory_order_relaxed);
}
)MSL";

struct AxpyArgs { uint64_t x, y; float a; uint32_t n; };
struct ProbeArgs { uint64_t base, offs, out; uint32_t n; };
struct SumArgs { uint64_t base, out; uint64_t n4; };

int main(int argc, char** argv) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* dev = MTL::CreateSystemDefaultDevice();
    if (!dev) { std::printf("no Metal device\n"); return 1; }
    std::printf("device: %s\n", dev->name()->utf8String());
    std::printf("unified memory: %d\n", (int) dev->hasUnifiedMemory());
    std::printf("maxBufferLength: %llu (%.2f GiB)\n", (unsigned long long) dev->maxBufferLength(), dev->maxBufferLength() / 1073741824.0);
    std::printf("recommendedMaxWorkingSetSize: %llu (%.2f GiB)\n", (unsigned long long) dev->recommendedMaxWorkingSetSize(),
                dev->recommendedMaxWorkingSetSize() / 1073741824.0);
    std::printf("maxThreadgroupMemoryLength: %llu\n", (unsigned long long) dev->maxThreadgroupMemoryLength());
    std::printf("families: apple7 %d apple8 %d apple9 %d apple10 %d metal3 %d metal4 %d\n", (int) dev->supportsFamily(MTL::GPUFamilyApple7),
                (int) dev->supportsFamily(MTL::GPUFamilyApple8), (int) dev->supportsFamily(MTL::GPUFamilyApple9),
                (int) dev->supportsFamily(MTL::GPUFamilyApple10), (int) dev->supportsFamily(MTL::GPUFamilyMetal3),
                (int) dev->supportsFamily(MTL::GPUFamilyMetal4));
    std::printf("page size: %ld\n", sysconf(_SC_PAGESIZE));

    // ---- 2. run-time MSL compile
    NS::Error* e = nullptr;
    double t0 = now_ms();
    MTL::CompileOptions* co = MTL::CompileOptions::alloc()->init();
    MTL::Library* lib = dev->newLibrary(NS::String::string(kSrc, NS::UTF8StringEncoding), co, &e);
    if (!lib) { std::printf("MSL compile failed: %s\n", e ? e->localizedDescription()->utf8String() : "?"); return 1; }
    std::printf("MSL run-time compile: %.1f ms\n", now_ms() - t0);
    auto pipe = [&](const char* n) {
        MTL::Function* f = lib->newFunction(NS::String::string(n, NS::UTF8StringEncoding));
        MTL::ComputePipelineState* p = dev->newComputePipelineState(f, &e);
        if (!p) { std::printf("pipeline %s failed\n", n); std::exit(1); }
        return p;
    };
    MTL::ComputePipelineState* p_axpy = pipe("axpy");
    MTL::ComputePipelineState* p_probe = pipe("probe");
    MTL::ComputePipelineState* p_sum = pipe("sum");
    MTL::CommandQueue* q = dev->newCommandQueue();

    // ---- 3/4. hello kernel through raw GPU addresses; residency through a residency set
    const uint32_t N = 1 << 20;
    MTL::Buffer* bx = dev->newBuffer(N * 4, MTL::ResourceStorageModeShared);
    MTL::Buffer* by = dev->newBuffer(N * 4, MTL::ResourceStorageModeShared);
    std::printf("buffer x: contents %p gpuAddress 0x%llx (%s)\n", bx->contents(), (unsigned long long) bx->gpuAddress(),
                (uint64_t) bx->contents() == bx->gpuAddress() ? "EQUAL" : "different");
    float* hx = (float*) bx->contents();
    float* hy = (float*) by->contents();
    for (uint32_t i = 0; i < N; ++i) { hx[i] = (float) i; hy[i] = 1.0f; }
    MTL::ResidencySetDescriptor* rd = MTL::ResidencySetDescriptor::alloc()->init();
    MTL::ResidencySet* rs = dev->newResidencySet(rd, &e);
    if (!rs) { std::printf("residency set failed\n"); return 1; }
    rs->addAllocation(bx); rs->addAllocation(by); rs->commit();
    q->addResidencySet(rs);
    {
        AxpyArgs a{bx->gpuAddress(), by->gpuAddress(), 2.0f, N};
        MTL::CommandBuffer* cb = q->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(p_axpy);
        enc->setBytes(&a, sizeof a, 0);
        enc->dispatchThreads(MTL::Size(N, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        int bad = 0;
        for (uint32_t i = 0; i < N; ++i) if (hy[i] != 2.0f * i + 1.0f) ++bad;
        std::printf("hello axpy via raw GPU addresses + residency set: %s (%d bad)\n", bad ? "FAIL" : "OK", bad);
    }
    // MTLSharedEvent as the CUDA event: signal from the GPU, poll from the host
    {
        MTL::SharedEvent* ev = dev->newSharedEvent();
        MTL::CommandBuffer* cb = q->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        AxpyArgs a{bx->gpuAddress(), by->gpuAddress(), 1.0f, N};
        enc->setComputePipelineState(p_axpy);
        enc->setBytes(&a, sizeof a, 0);
        enc->dispatchThreads(MTL::Size(N, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
        cb->encodeSignalEvent(ev, 1);
        double ts = now_ms();
        cb->commit();
        long spins = 0;
        while (ev->signaledValue() < 1) ++spins;
        std::printf("shared event: signalled after %.3f ms, %ld host polls\n", now_ms() - ts, spins);
        ev->release();
    }
    // dispatch cost: 1,000 tiny dispatches in one command buffer (the window-replay cost model)
    {
        AxpyArgs a{bx->gpuAddress(), by->gpuAddress(), 0.0f, 256};
        for (int rep = 0; rep < 3; ++rep) {
            double ts = now_ms();
            MTL::CommandBuffer* cb = q->commandBufferWithUnretainedReferences();
            MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
            for (int k = 0; k < 1000; ++k) {
                enc->setComputePipelineState(p_axpy);
                enc->setBytes(&a, sizeof a, 0);
                enc->dispatchThreads(MTL::Size(256, 1, 1), MTL::Size(256, 1, 1));
            }
            enc->endEncoding();
            const double te = now_ms();
            cb->commit();
            cb->waitUntilCompleted();
            std::printf("1000 dependent tiny dispatches: encode %.3f ms, encode+run %.3f ms\n", te - ts, now_ms() - ts);
        }
    }

    // ---- 5. a > 30 GB mmapped file wrapped with newBufferWithBytesNoCopy
    if (argc < 2) { pool->release(); return 0; }
    const char* path = argv[1];
    const int fd = open(path, O_RDONLY);
    if (fd < 0) { std::printf("open %s failed\n", path); return 1; }
    struct stat sb; fstat(fd, &sb);
    const uint64_t fsize = (uint64_t) sb.st_size;
    const uint64_t pg = (uint64_t) sysconf(_SC_PAGESIZE);
    const uint64_t mlen = (fsize + pg - 1) / pg * pg;   // NoCopy wants a page-multiple length
    std::printf("file %s: %.2f GiB\n", path, fsize / 1073741824.0);
    for (int mode = 0; mode < 2; ++mode) {
        // mode 0: PROT_READ MAP_SHARED (read-only, what the engine wants); mode 1: PROT_READ|PROT_WRITE MAP_PRIVATE
        const int prot = mode == 0 ? PROT_READ : (PROT_READ | PROT_WRITE);
        const int flags = mode == 0 ? MAP_SHARED : MAP_PRIVATE;
        void* m = mmap(nullptr, mlen, prot, flags, fd, 0);
        if (m == MAP_FAILED) { std::printf("mmap mode %d failed\n", mode); continue; }
        std::printf("-- mode %d (%s): mmap at %p (page aligned %d)\n", mode, mode == 0 ? "PROT_READ MAP_SHARED" : "RW MAP_PRIVATE", m,
                    (int) (((uintptr_t) m % pg) == 0));
        double ts = now_ms();
        MTL::Buffer* whole = dev->newBuffer(m, mlen, MTL::ResourceStorageModeShared, nullptr);
        std::printf("   one NoCopy buffer of %.2f GiB: %s (%.1f ms)\n", mlen / 1073741824.0, whole ? "OK" : "nil", now_ms() - ts);
        // chunks of at most maxBufferLength / 2 (and a 4 GiB variant: one chunk per few layers)
        std::vector<MTL::Buffer*> chunks;
        const uint64_t chunk = argc > 2 ? (std::strtoull(argv[2], nullptr, 10) << 20) : (4ull << 30);
        ts = now_ms();
        bool all = true;
        for (uint64_t off = 0; off < mlen; off += chunk) {
            const uint64_t len = std::min(chunk, mlen - off);
            MTL::Buffer* b = dev->newBuffer((char*) m + off, len, MTL::ResourceStorageModeShared, nullptr);
            if (!b) { all = false; break; }
            chunks.push_back(b);
        }
        std::printf("   %zu NoCopy chunks of %.2f GiB: %s (%.1f ms)\n", chunks.size(), chunk / 1073741824.0, all ? "OK" : "FAILED", now_ms() - ts);
        // GPU probe reads at 64 offsets spread over the file (the file's pattern: uint32 at offset o equals (o>>12) ^ 0x5a5a5a5a
        // where written; holes read 0)
        const uint32_t NP = 64;
        MTL::Buffer* boffs = dev->newBuffer(NP * 8, MTL::ResourceStorageModeShared);
        MTL::Buffer* bout = dev->newBuffer(NP * 4, MTL::ResourceStorageModeShared);
        uint64_t* offs = (uint64_t*) boffs->contents();
        for (uint32_t i = 0; i < NP; ++i) offs[i] = ((fsize / NP) * i) & ~(uint64_t) 4095;
        auto run_probe = [&](MTL::Buffer* b, uint64_t base_off, const char* what) {
            // offsets relative to this buffer's base
            std::vector<uint64_t> keep(offs, offs + NP);
            uint32_t n = 0;
            std::vector<uint32_t> idx;
            for (uint32_t i = 0; i < NP; ++i)
                if (keep[i] >= base_off && keep[i] < base_off + b->length()) { offs[n++] = keep[i] - base_off; idx.push_back(i); }
            MTL::ResidencySet* r2 = dev->newResidencySet(rd, &e);
            r2->addAllocation(b); r2->addAllocation(boffs); r2->addAllocation(bout); r2->commit();
            double tr = now_ms();
            r2->requestResidency();
            const double treq = now_ms() - tr;
            ProbeArgs a{b->gpuAddress(), boffs->gpuAddress(), bout->gpuAddress(), n};
            MTL::CommandBuffer* cb = q->commandBuffer();
            cb->useResidencySet(r2);
            MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
            enc->setComputePipelineState(p_probe);
            enc->setBytes(&a, sizeof a, 0);
            enc->dispatchThreads(MTL::Size(n ? n : 1, 1, 1), MTL::Size(64, 1, 1));
            enc->endEncoding();
            tr = now_ms();
            cb->commit();
            cb->waitUntilCompleted();
            const double trun = now_ms() - tr;
            int okc = 0, bad = 0;
            const uint32_t* out = (const uint32_t*) bout->contents();
            for (uint32_t k = 0; k < n; ++k) {
                const uint64_t fo = keep[idx[k]];
                uint32_t want = 0;
                pread(fd, &want, 4, (off_t) fo);
                (out[k] == want ? okc : bad)++;
            }
            std::printf("   GPU probe %s: %u reads, %d match pread, %d differ (requestResidency %.1f ms, run %.1f ms, status %d)\n", what, n,
                        okc, bad, treq, trun, (int) cb->status());
            for (uint32_t i = 0; i < NP; ++i) offs[i] = keep[i];
            r2->release();
        };
        if (whole) run_probe(whole, 0, "whole buffer");
        if (all && !chunks.empty()) run_probe(chunks.back(), (chunks.size() - 1) * chunk, "last chunk");
        // a streaming GPU read over one chunk, cold then warm (page cache)
        if (all && !chunks.empty()) {
            MTL::Buffer* b = chunks[chunks.size() / 2];
            MTL::Buffer* bs = dev->newBuffer(4, MTL::ResourceStorageModeShared);
            for (int rep = 0; rep < 2; ++rep) {
                *(uint32_t*) bs->contents() = 0;
                MTL::ResidencySet* r2 = dev->newResidencySet(rd, &e);
                r2->addAllocation(b); r2->addAllocation(bs); r2->commit();
                SumArgs a{b->gpuAddress(), bs->gpuAddress(), b->length() / 16};
                double tr = now_ms();
                MTL::CommandBuffer* cb = q->commandBuffer();
                cb->useResidencySet(r2);
                MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
                enc->setComputePipelineState(p_sum);
                enc->setBytes(&a, sizeof a, 0);
                enc->dispatchThreads(MTL::Size(40 * 1024, 1, 1), MTL::Size(256, 1, 1));
                enc->endEncoding();
                cb->commit();
                cb->waitUntilCompleted();
                const double dt = now_ms() - tr;
                std::printf("   GPU streaming read of a %.2f GiB chunk (%s): %.1f ms = %.1f GB/s\n", b->length() / 1073741824.0,
                            rep == 0 ? "first touch" : "again", dt, b->length() / dt / 1e6);
                r2->release();
            }
            bs->release();
        }
        for (auto* b : chunks) b->release();
        if (whole) whole->release();
        boffs->release(); bout->release();
        munmap(m, mlen);
    }
    close(fd);
    pool->release();
    return 0;
}
