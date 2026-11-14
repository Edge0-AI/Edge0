// metal/src/mtl_rt.mm - see mtl_rt.h.  Built with ARC (-fobjc-arc).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "mtl_rt.h"
#include "resource_path.h"

#include <chrono>
#if defined(STRATA_DS4_RUNTIME)
#include <execinfo.h>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <string>
#include <deque>
#include <mach/mach_time.h>
#include <map>
#include <thread>
#endif
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace mtl {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "mtl: cannot read %s\n", path.c_str());
        std::exit(2);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

#if defined(STRATA_DS4_RUNTIME)
extern "C" void* ds4_gpu_residency_set() __attribute__((weak_import));
#endif
struct Context::Impl {
#if defined(STRATA_DS4_RUNTIME)
    id<MTLCommandBuffer> cb_def = nil;   // the open command buffer single dispatches accumulate in
    id<MTLComputeCommandEncoder> enc_def = nil;
    int def_n = 0;
    std::string def_names, inflight_names;   // (DS4_ERRINFO) the kernels of the open / the committed command buffer
    id<MTLSharedEvent> qev = nil;          // consecutive deferred buffers are chained through it (untracked buffers: the queue alone does not order them)
    uint64_t qev_v = 0;
    id<MTLCommandBuffer> inflight = nil;   // the last committed-but-not-waited deferred buffer (the queue runs them in order)
    std::vector<void*> graveyard;   // buffers freed while the deferred command buffer still references them (it holds none)
    std::recursive_mutex mu;   // helper threads (the stager, the expert cache) flush too: one writer on the open encoder
#endif
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    std::vector<id<MTLLibrary>> libs;
    id<MTLSharedEvent> ev = nil;   // run_graph_seg's handshake timeline
    uint64_t ev_v = 0;
};

static inline id<MTLBuffer> B(const Buffer& b) { return (__bridge id<MTLBuffer>)b.handle; }
// the MTLBuffer and offset that bind byte `off` of b (a chunked buffer: the chunk the offset selects)
static inline id<MTLBuffer> BO(const Buffer& b, size_t off, size_t& o) {
    if (!b.chunks) { o = off + b.bias; return B(b); }   // (bias: a view onto part of a bigger allocation; 0 otherwise)
    const size_t k = std::min(off / b.stride, b.chunks->size() - 1);
    o = off - k * b.stride;
    return (__bridge id<MTLBuffer>)(*b.chunks)[k];
}
static inline id<MTLComputePipelineState> P(const Pipeline* p) { return (__bridge id<MTLComputePipelineState>)p->handle; }
static std::string g_device_name = "unknown GPU";

#if defined(STRATA_DS4_RUNTIME)

// ---- DS4_TIMELINE=1: every deferred dispatch in its own encoder with start / end GPU timestamps; the report at exit gives, per kernel,
// its GPU time and the gap the GPU sat idle before it (host waits, event waits, launch).  DS4_TL_FROM=N: only from dispatch N on.
static std::atomic<uint64_t> g_disp_count{0};   // dispatches encoded
namespace {
struct Timeline {
    bool on = false;
    id<MTLCounterSet> tset = nil;
    std::vector<id<MTLCounterSampleBuffer>> bufs;   // a sample buffer holds only ~1500 encoders: they rotate
    size_t per = 0, cap = 0, n = 0;        // encoders per buffer / in all (the buffers' limit) / recorded
    long after = 0;                        // DS4_TL_AFTER: record from the Nth dispatch on
    std::vector<std::string> names;        // per encoder
    uint64_t cpu0 = 0, gpu0 = 0;
    bool reg = false;
    bool gate = true;                      // ds4_timeline_gate: record only while open
    bool att = false;                      // the encoder just made carries a sample slot
} g_tl;

id<MTLComputeCommandEncoder> tl_make_encoder(id<MTLDevice> dev, id<MTLCommandBuffer> cb, MTLDispatchType dt) {
    g_tl.att = false;
    if (!g_tl.on) return [cb computeCommandEncoderWithDispatchType:dt];
    if (g_tl.tset == nil) {
        for (id<MTLCounterSet> cs in dev.counterSets)
            if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) g_tl.tset = cs;
        if (g_tl.tset == nil) {
            std::fprintf(stderr, "mtl: DS4_TIMELINE: no timestamp counter set\n");
            g_tl.on = false;
            return [cb computeCommandEncoderWithDispatchType:dt];
        }
        for (size_t c = 4096; c >= 64 && g_tl.per == 0; c /= 2) {
            MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
            d.counterSet = g_tl.tset;
            d.storageMode = MTLStorageModeShared;
            d.sampleCount = c * 2;
            NSError* err = nil;
            id<MTLCounterSampleBuffer> b = [dev newCounterSampleBufferWithDescriptor:d error:&err];
            if (b != nil) {
                g_tl.per = c;
                g_tl.bufs.push_back(b);
            }
        }
        if (g_tl.per == 0) {
            std::fprintf(stderr, "mtl: DS4_TIMELINE: no counter sample buffer\n");
            g_tl.on = false;
            return [cb computeCommandEncoderWithDispatchType:dt];
        }
        g_tl.cap = g_tl.per * 120;
        MTLTimestamp c0 = 0, g0 = 0;
        [dev sampleTimestamps:&c0 gpuTimestamp:&g0];
        g_tl.cpu0 = c0;
        g_tl.gpu0 = g0;
        g_tl.names.assign(g_tl.cap, std::string());
        g_tl.after = std::getenv("DS4_TL_AFTER") ? std::atol(std::getenv("DS4_TL_AFTER")) : 0;
        if (std::getenv("DS4_TL_DRAFT")) g_tl.gate = false;
    }
    if (!g_tl.gate || g_tl.n >= g_tl.cap || (long) g_disp_count.load() < g_tl.after) return [cb computeCommandEncoderWithDispatchType:dt];
    const size_t bi = g_tl.n / g_tl.per, si = g_tl.n % g_tl.per;
    if (bi >= g_tl.bufs.size()) {
        MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
        d.counterSet = g_tl.tset;
        d.storageMode = MTLStorageModeShared;
        d.sampleCount = g_tl.per * 2;
        NSError* err = nil;
        id<MTLCounterSampleBuffer> b = [dev newCounterSampleBufferWithDescriptor:d error:&err];
        if (b == nil) {
            g_tl.cap = g_tl.n;
            return [cb computeCommandEncoderWithDispatchType:dt];
        }
        g_tl.bufs.push_back(b);
    }
    MTLComputePassDescriptor* pd = [MTLComputePassDescriptor computePassDescriptor];
    pd.dispatchType = dt;
    pd.sampleBufferAttachments[0].sampleBuffer = g_tl.bufs[bi];
    pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * si;
    pd.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * si + 1;
    g_tl.att = true;
    return [cb computeCommandEncoderWithDescriptor:pd];
}

void tl_report(id<MTLDevice> dev) {
    if (!g_tl.on && g_tl.bufs.empty()) return;
    const size_t n = std::min(g_tl.n, g_tl.cap);
    if (n < 2) return;
    std::vector<MTLCounterResultTimestamp> tsv(2 * n);
    for (size_t b = 0; b * g_tl.per < n; ++b) {
        const size_t cnt = std::min(g_tl.per, n - b * g_tl.per);
        NSData* data = [g_tl.bufs[b] resolveCounterRange:NSMakeRange(0, 2 * cnt)];
        if (data == nil) { std::fprintf(stderr, "mtl: DS4_TIMELINE: resolve failed\n"); return; }
        std::memcpy(&tsv[2 * b * g_tl.per], [data bytes], 2 * cnt * sizeof(MTLCounterResultTimestamp));
    }
    const MTLCounterResultTimestamp* ts = tsv.data();
    MTLTimestamp c1 = 0, g1 = 0;
    [dev sampleTimestamps:&c1 gpuTimestamp:&g1];
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    const double cpu_ns_per_tick = (double) tb.numer / (double) tb.denom;
    (void) cpu_ns_per_tick;   // (sampleTimestamps' CPU value is in ns already)
    const double ns_per_gpu = (g1 > g_tl.gpu0) ? (double) (c1 - g_tl.cpu0) / (double) (g1 - g_tl.gpu0) : 1.0;
    std::fprintf(stderr, "timeline: cap %zu n %zu scale %.6f ns/tick first ts %llu %llu last %llu\n", g_tl.cap, g_tl.n, ns_per_gpu, (unsigned long long) ts[0].timestamp, (unsigned long long) ts[1].timestamp, (unsigned long long) ts[2 * n - 1].timestamp);
    size_t from = std::getenv("DS4_TL_FROM") ? (size_t) std::atoll(std::getenv("DS4_TL_FROM")) : 0;
    struct Acc { double busy = 0, gap = 0, gap_big = 0; size_t n = 0, n_big = 0; };
    std::map<std::string, Acc> acc;
    double t_first = 0, t_last = 0, total_busy = 0, total_gap = 0, total_big = 0;
    double prev_end = -1, ovl_us = 0;
    size_t used = 0, n_ovl = 0;
    for (size_t k = from; k < n; ++k) {
        if (g_tl.names[k].empty()) continue;
        const double st = (double) ts[2 * k].timestamp * ns_per_gpu * 1e-3, en = (double) ts[2 * k + 1].timestamp * ns_per_gpu * 1e-3;   // us
        if (!(en >= st)) continue;
        if (prev_end < 0) t_first = st;
        Acc& a = acc[g_tl.names[k]];
        a.busy += en - st;
        ++a.n;
        total_busy += en - st;
        if (prev_end >= 0 && st < prev_end) { ++n_ovl; ovl_us += prev_end - st; }
        if (prev_end >= 0 && st >= prev_end) {
            const double gp = st - prev_end;
            a.gap += gp;
            total_gap += gp;
            if (gp > 300.0) { a.gap_big += gp; ++a.n_big; total_big += gp; }
        }
        prev_end = en;
        t_last = en;
        ++used;
    }
    std::vector<std::pair<double, std::string>> v;
    for (auto& kv : acc) v.emplace_back(kv.second.busy + kv.second.gap, kv.first);
    std::sort(v.begin(), v.end(), [](auto& x, auto& y) { return x.first > y.first; });
    std::fprintf(stderr, "timeline: %zu dispatches, wall %.1f ms, GPU busy %.1f ms, idle gaps %.1f ms (of it > 300 us: %.1f ms)\n", used,
                 (t_last - t_first) / 1000.0, total_busy / 1000.0, total_gap / 1000.0, total_big / 1000.0);
    std::fprintf(stderr, "timeline: %zu dispatches started before the previous one ended (%.1f ms overlapped)\n", n_ovl, ovl_us / 1000.0);
    std::fprintf(stderr, "timeline: kernel                          n   busy ms  us/disp   gap ms  gap/disp  big-gap ms (n)\n");
    for (size_t i = 0; i < v.size() && i < 40; ++i) {
        const Acc& a = acc[v[i].second];
        std::fprintf(stderr, "timeline: %-28s %6zu %9.1f %8.1f %8.1f %8.1f %10.1f (%zu)\n", v[i].second.c_str(), a.n, a.busy / 1000.0, a.busy / (double) a.n,
                     a.gap / 1000.0, a.gap / (double) a.n, a.gap_big / 1000.0, a.n_big);
    }
}
}  // namespace

extern "C" void ds4_timeline_gate(int open) { g_tl.gate = open != 0; }


// ---- ds4_busy_begin / ds4_busy_end: a one-threadgroup kernel on a second queue that spins on a shared flag while the host works
// (the CPU lane), so the GPU is not idle between two bursts (idle -> the next kernels run 2-3x slower); the host sets the flag the
// moment it is done and the kernel leaves within microseconds.  DS4_BUSY=0 turns it off.
namespace {
struct Busy {
    bool init = false, ok = false;
    id<MTLCommandQueue> q = nil;
    id<MTLComputePipelineState> ps = nil;
    id<MTLBuffer> flag = nil;
    id<MTLCommandBuffer> last = nil;
    bool running = false;
};
Busy g_busy;
}  // namespace

extern "C" void ds4_busy_begin() {
    static const bool off = std::getenv("DS4_BUSY") && std::string(std::getenv("DS4_BUSY")) == "0";
    if (off) return;
    @autoreleasepool {
        if (!g_busy.init) {
            g_busy.init = true;
            id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void busy_spin(device volatile atomic_uint* flag [[buffer(0)]], constant uint& cap [[buffer(1)]], uint t [[thread_position_in_grid]]) {
    uint x = t;
    float f = (float) t * 1e-9f;
    for (uint i = 0; i < cap; ++i) {
        for (uint j = 0; j < 64; ++j) x = x * 1664525u + 1013904223u;
        for (uint j = 0; j < 16; ++j) f = fma(f, 1.0000001f, 1e-7f);   // (a real FP chain: keeps the ALU clock up)
        x ^= as_type<uint>(f);
        if (atomic_load_explicit(flag, memory_order_relaxed) != 0u) break;
    }
    if (x == 0xdeadbeefu) atomic_store_explicit(flag, 2u, memory_order_relaxed);
}
)MSL";
            NSError* err = nil;
            id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:nil error:&err];
            if (lib) g_busy.ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"busy_spin"] error:&err];
            if (g_busy.ps) {
                g_busy.q = [dev newCommandQueue];
                g_busy.flag = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
                g_busy.ok = true;
            }
        }
        if (!g_busy.ok || g_busy.running) return;
        g_busy.running = true;
        if (g_busy.last != nil) [g_busy.last waitUntilCompleted];   // (the previous spin has left)
        *(volatile uint32_t*) [g_busy.flag contents] = 0;
        uint32_t cap = std::getenv("DS4_BUSY_CAP") ? (uint32_t) std::atoi(std::getenv("DS4_BUSY_CAP")) : 20000u;   // ~ tens of ms at most
        // the spin's width: one 256-thread threadgroup leaves most of the GPU idle and the clock drops anyway; DS4_BUSY_W
        // threadgroups (256 threads each) keep more of the card's ALUs fed, which is what holds the performance state.
        static const uint32_t width = std::getenv("DS4_BUSY_W") ? (uint32_t) std::max(1, std::atoi(std::getenv("DS4_BUSY_W"))) : 1u;
        id<MTLCommandBuffer> cb = [g_busy.q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_busy.ps];
        [enc setBuffer:g_busy.flag offset:0 atIndex:0];
        [enc setBytes:&cap length:4 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(256 * width, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        g_busy.last = cb;
    }
}
extern "C" void ds4_busy_end() {
    if (!g_busy.ok || g_busy.last == nil || !g_busy.running) return;
    *(volatile uint32_t*) [g_busy.flag contents] = 1;
    g_busy.running = false;
}

#endif
Context::Context() : impl_(new Impl) {
    impl_->dev = MTLCreateSystemDefaultDevice();
#if defined(STRATA_DS4_RUNTIME)
    if (std::getenv("DS4_TIMELINE") && !g_tl.reg) {
        g_tl.on = true;
        g_tl.reg = true;
        static id<MTLDevice> tl_dev = impl_->dev;
        std::atexit([] { tl_report(tl_dev); });
    }
#endif
    if (!impl_->dev) {
        std::fprintf(stderr, "mtl: no Metal device\n");
        std::exit(2);
    }
    impl_->q = [impl_->dev newCommandQueue];
#if defined(STRATA_DS4_RUNTIME)
    if (ds4_gpu_residency_set && !std::getenv("DS4_NO_RESIDENCY")) {
        id<MTLResidencySet> rs = (__bridge id<MTLResidencySet>)ds4_gpu_residency_set();
        if (rs) [impl_->q addResidencySet:rs];
    }
#endif
    // env, then <exe dir>/metal/ (a component package), then the compiled-in source-tree path
    ggml_common_path = strata_metal::resource("STRATA_GGML_COMMON", "ggml-common.h", STRATA_GGML_COMMON_DEFAULT);
    kernel_dir = strata_metal::resource("STRATA_METAL_KERNELS", "kernels", STRATA_METAL_KERNELS_DEFAULT);
}

Context::~Context() { delete impl_; }

std::string Context::device_name() const { return [[impl_->dev name] UTF8String]; }
bool Context::has_family_apple(int n) const {
    return [impl_->dev supportsFamily:(MTLGPUFamily)(MTLGPUFamilyApple1 + (n - 1))];
}
bool Context::supports_metal4() const {
    if (@available(macOS 26.0, *)) return [impl_->dev supportsFamily:MTLGPUFamilyMetal4];
    return false;
}
uint64_t Context::recommended_working_set() const { return [impl_->dev recommendedMaxWorkingSetSize]; }
uint64_t Context::max_buffer_length() const { return [impl_->dev maxBufferLength]; }

int Context::compile(const std::vector<std::string>& files, const std::string& prelude, bool fast_math) {
    std::string src = prelude + "\n#include <metal_stdlib>\nusing namespace metal;\n";
    if (!ggml_common_path.empty() && prelude.find("NO_GGML_COMMON") == std::string::npos) {
        src += "#define GGML_COMMON_DECL_METAL\n#define GGML_COMMON_IMPL_METAL\n";
        src += read_file(ggml_common_path);
        src += "\n";
    }
    for (const auto& f : files) {
        std::string p = (f.size() && f[0] == '/') ? f : kernel_dir + "/" + f;
        src += "#line 1 \"" + f + "\"\n" + read_file(p) + "\n";
    }
    MTLCompileOptions* o = [MTLCompileOptions new];
    // Safe math by default: no reassociation, so every token of a window is bitwise what a one-token call gives
    // (the kernels write their contractions as explicit fma).  Measured free on the M5 Max.
    fast_math = false;
    if (const char* m = std::getenv("STRATA_METAL_MATH")) fast_math = std::string(m) == "fast";
    if (@available(macOS 15.0, *)) {
        o.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;
    } else {
        o.fastMathEnabled = fast_math;
    }
    NSError* err = nil;
    id<MTLLibrary> lib = [impl_->dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:o error:&err];
    if (!lib) {
        std::fprintf(stderr, "mtl: compile failed (%s, macOS %s; %s):\n%s\n", [[impl_->dev name] UTF8String],
                     [[[NSProcessInfo processInfo] operatingSystemVersionString] UTF8String],
                     files.empty() ? "?" : files.back().c_str(), [[err localizedDescription] UTF8String]);
        std::exit(3);
    }
    impl_->libs.push_back(lib);
    return (int)impl_->libs.size() - 1;
}

#if defined(STRATA_DS4_RUNTIME)

// ---- the keep-alive spin (see mtl_rt.h)
static std::atomic<int64_t> g_last_act_ns{0};
static inline int64_t ka_now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::atomic<bool> g_ka_on{false};

void Context::start_keepalive() {
    if (!std::getenv("DS4_KEEPALIVE") || g_ka_on.exchange(true)) return;   // (measured slower in the decode loop)
    id<MTLDevice> dev = impl_->dev;
    std::thread([dev]() {
        @autoreleasepool {
            static const char* kSpin = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void ka_spin(device atomic_uint* c [[buffer(0)]], constant uint& n [[buffer(1)]], uint t [[thread_position_in_grid]]) {
    uint x = t;
    for (uint i = 0; i < n; ++i) x = x * 1664525u + 1013904223u;
    if (x == 0xdeadbeefu) atomic_fetch_add_explicit(c, 1u, memory_order_relaxed);
}
)MSL";
            NSError* err = nil;
            id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSpin] options:nil error:&err];
            if (!lib) return;
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"ka_spin"] error:&err];
            if (!ps) return;
            id<MTLCommandQueue> q = [dev newCommandQueue];
            id<MTLBuffer> cbuf = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
            uint32_t iters = std::getenv("DS4_KEEPALIVE_ITERS") ? (uint32_t) std::atoi(std::getenv("DS4_KEEPALIVE_ITERS")) : 20000u;
            for (;;) {
                if (ka_now_ns() - g_last_act_ns.load(std::memory_order_relaxed) > 50000000LL) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    continue;
                }
                @autoreleasepool {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:ps];
                    [enc setBuffer:cbuf offset:0 atIndex:0];
                    [enc setBytes:&iters length:4 atIndex:1];
                    [enc dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                }
            }
        }
    }).detach();
}


// ---- zero-copy weights (see mtl_rt.h)
namespace {
struct ZcBuf { size_t len; void* handle; uint64_t addr; };
std::map<uintptr_t, ZcBuf> g_zc;
std::mutex g_zc_mu;
}  // namespace

uint64_t Context::zc_map(const void* p, size_t bytes) {
    const size_t pg = 16384;
    const uintptr_t lo = (uintptr_t) p & ~(uintptr_t) (pg - 1);
    const uintptr_t hi = ((uintptr_t) p + bytes + pg - 1) & ~(uintptr_t) (pg - 1);
    std::lock_guard<std::mutex> lk(g_zc_mu);
    auto it = g_zc.upper_bound(lo);
    if (it != g_zc.begin()) {
        --it;
        if (it->first <= lo && hi <= it->first + it->second.len) return it->second.addr + ((uintptr_t) p - it->first);
    }
    static const size_t cap = std::getenv("DS4_ZC_MAX") ? (size_t) std::atoll(std::getenv("DS4_ZC_MAX")) : 9000;
    if (g_zc.size() >= cap) return 0;
    id<MTLBuffer> b = [impl_->dev newBufferWithBytesNoCopy:(void*) lo length:hi - lo options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) return 0;
    if (ds4_gpu_residency_set) {
        id<MTLResidencySet> rs = (__bridge id<MTLResidencySet>)ds4_gpu_residency_set();
        if (rs) {
            [rs addAllocation:b];
            [rs commit];
        }
    }
    ZcBuf z{hi - lo, (__bridge_retained void*) b, (uint64_t) [b gpuAddress]};
    g_zc[lo] = z;
    return z.addr + ((uintptr_t) p - lo);
}
size_t Context::zc_count() const { std::lock_guard<std::mutex> lk(g_zc_mu); return g_zc.size(); }

#endif
Pipeline Context::pipeline(int lib, const std::string& fn, const Constants& c) {
#if defined(STRATA_DS4_RUNTIME)
    // (cached: the host helpers ask for the same pipeline a dispatch at a time; newComputePipelineState costs ~0.3 ms and
    // [MTLDevice name] reads the preferences - the prompt path's expert copies (3 a streamed expert) spent seconds there)
    static std::mutex cache_mu;
    static std::map<std::string, Pipeline> cache;
    std::string key = std::to_string(lib) + "|" + fn;
    for (const auto& v : c.vals) key += "|" + std::to_string(v.index) + ":" + std::to_string(v.type) + ":" + std::to_string(v.v);
    key += "|" + std::to_string((uintptr_t) this);
    {
        std::lock_guard<std::mutex> lk(cache_mu);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
    }
    Pipeline made = pipeline_uncached(lib, fn, c);
    {
        std::lock_guard<std::mutex> lk(cache_mu);
        cache[key] = made;
    }
    return made;
}

Pipeline Context::pipeline_uncached(int lib, const std::string& fn, const Constants& c) {
#endif
    NSError* err = nil;
    id<MTLFunction> f;
    NSString* name = [NSString stringWithUTF8String:fn.c_str()];
    if (c.vals.empty()) {
        f = [impl_->libs[lib] newFunctionWithName:name];
        // a function with (optional) function constants needs a specialization even when none is set
        if (f && f.functionConstantsDictionary.count > 0)
            f = [impl_->libs[lib] newFunctionWithName:name constantValues:[MTLFunctionConstantValues new] error:&err];
    } else {
        MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
        for (const auto& v : c.vals) {
            int32_t iv = (int32_t)v.v;
            bool bv = v.v != 0;
            uint32_t uv = v.v;
            switch (v.type) {
                case 0: [cv setConstantValue:&iv type:MTLDataTypeInt atIndex:v.index]; break;
                case 1: [cv setConstantValue:&bv type:MTLDataTypeBool atIndex:v.index]; break;
                default: [cv setConstantValue:&uv type:MTLDataTypeUInt atIndex:v.index]; break;
            }
        }
        f = [impl_->libs[lib] newFunctionWithName:name constantValues:cv error:&err];
    }
    if (!f) {
        std::fprintf(stderr, "mtl: no function %s %s\n", fn.c_str(), err ? [[err localizedDescription] UTF8String] : "");
        std::exit(3);
    }
    MTLComputePipelineDescriptor* d = [MTLComputePipelineDescriptor new];
    d.computeFunction = f;
    d.supportIndirectCommandBuffers = YES;
    id<MTLComputePipelineState> ps = [impl_->dev newComputePipelineStateWithDescriptor:d
                                                                               options:MTLPipelineOptionNone
                                                                            reflection:nil
                                                                                 error:&err];
    if (!ps) {
        std::fprintf(stderr, "mtl: pipeline %s: %s\n", fn.c_str(), [[err localizedDescription] UTF8String]);
        std::exit(3);
    }
    Pipeline p;
    p.handle = (__bridge_retained void*)ps;
    p.max_threads = (int)ps.maxTotalThreadsPerThreadgroup;
    if (const char* cap = std::getenv("STRATA_METAL_MAX_THREADS_CAP")) {   // test hook, see require_fit
        const int lim = std::atoi(cap);
        if (lim > 0 && lim < p.max_threads) p.max_threads = lim;
    }
    p.simd_width = (int)ps.threadExecutionWidth;
    p.static_tg_mem = (int)ps.staticThreadgroupMemoryLength;
    p.name = fn;
    if (!c.vals.empty()) {
        p.name += "{";
        for (size_t i = 0; i < c.vals.size(); ++i)
            p.name += (i ? "," : "") + std::to_string(c.vals[i].index) + "=" + std::to_string((int)c.vals[i].v);
        p.name += "}";
    }
#if defined(STRATA_DS4_RUNTIME)
    if (g_device_name == "unknown GPU") g_device_name = [[impl_->dev name] UTF8String];
#else
    g_device_name = [[impl_->dev name] UTF8String];
#endif
    return p;
}

void require_fit(const Pipeline& p, uint32_t threads, const char* where) {
    if (threads == 0 || (int64_t)threads <= (int64_t)p.max_threads) return;
    std::fprintf(stderr,
                 "strata metal: the kernel %s can run at most %d threads per threadgroup on this GPU (%s), but the %s asks "
                 "for %u.\nMetal skips such a dispatch without any error, so every answer would be wrong; stopping here "
                 "instead.  Please report this (the GPU, macOS version and this line) at "
                 "https://github.com/Niko1221/Strata/issues .  (#2323)\n",
                 p.name.empty() ? "?" : p.name.c_str(), p.max_threads, g_device_name.c_str(), where, threads);
    std::fflush(stderr);
    std::exit(6);
}

Buffer Context::alloc(size_t bytes) {
#if defined(STRATA_DS4_RUNTIME)
    // Metal returns the new buffer with one more reference parked in the calling thread's autorelease pool.  The engine's host
    // threads have no pool (only Context::run opens one), so that reference was never given back: every buffer made here lived for
    // the rest of the process, free() or not (a 16 KB job array per expert matvec, ~90K a request: +1.3 GB and a slower queue every
    // call, 16 -> 3 tok/s after a few requests on a big Mac).
    Buffer r;
    @autoreleasepool {
        id<MTLBuffer> b = [impl_->dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!b) {
            std::fprintf(stderr, "mtl: alloc %zu failed\n", bytes);
            std::exit(4);
        }
        std::memset([b contents], 0, bytes);
        r.handle = (__bridge_retained void*)b;
        r.ptr = [b contents];
    }
    allocated_ += bytes;
    r.bytes = bytes;
    return r;
#else
    id<MTLBuffer> b = [impl_->dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) {
        std::fprintf(stderr, "mtl: alloc %zu failed\n", bytes);
        std::exit(4);
    }
    std::memset([b contents], 0, bytes);
    allocated_ += bytes;
    Buffer r;
    r.handle = (__bridge_retained void*)b;
    r.ptr = [b contents];
    r.bytes = bytes;
    return r;
#endif
}

Buffer Context::wrap(void* p, size_t bytes, size_t max_span) {
    const size_t pg = 16384, maxlen = [impl_->dev maxBufferLength] / pg * pg;
    if (bytes <= maxlen || std::getenv("STRATA_METAL_WRAP_SPLIT") && std::string(std::getenv("STRATA_METAL_WRAP_SPLIT")) == "0")
        return wrap(p, bytes);
    const size_t span = (max_span + pg - 1) / pg * pg;
    if (span >= maxlen / 2) {
        std::fprintf(stderr, "mtl: wrap %zu: a tensor of %zu bytes does not fit the chunks\n", bytes, max_span);
        std::exit(4);
    }
    Buffer r;
    r.chunks = std::make_shared<std::vector<void*>>();
    r.stride = maxlen - span;
    for (size_t base = 0; base < bytes; base += r.stride) {
        const size_t len = std::min(maxlen, (bytes - base + pg - 1) / pg * pg);
        id<MTLBuffer> b = [impl_->dev newBufferWithBytesNoCopy:(uint8_t*) p + base length:len
                                                        options:MTLResourceStorageModeShared deallocator:nil];
        if (!b) {
            std::fprintf(stderr, "mtl: wrap chunk %zu+%zu failed\n", base, len);
            std::exit(4);
        }
        r.chunks->push_back((__bridge_retained void*)b);
        if (base + len >= bytes) break;
    }
    allocated_ += bytes;
    r.handle = (*r.chunks)[0];
    r.ptr = p;
    r.bytes = bytes;
    std::fprintf(stderr, "mtl: wrapped %.2f GB as %zu overlapping buffers (maxBufferLength %.2f GB, overlap %.2f GB)\n",
                 bytes / 1e9, r.chunks->size(), maxlen / 1e9, span / 1e9);
    return r;
}

#if defined(STRATA_DS4_RUNTIME)
// The runtime's own MTLBuffer under a CPU address (strata::gpu's allocations), when [p, p + bytes) lies inside one.  Binding it
// - the same MTLBuffer, so the same GPU address, as every other kernel that touches this memory - instead of a fresh no-copy
// buffer per call.  (Each fresh wrapper is a second GPU mapping of the same pages: the quantiser read the activations through
// one mapping while the norm that produced them had written through another, and saw the previous layer's.)
extern "C" void* ds4_gpu_buffer_for_cpu(const void* cpu, size_t bytes, size_t* off) __attribute__((weak_import));
extern "C" size_t ds4_gpu_alloc_remaining(const void* cpu) __attribute__((weak_import));
#endif
Buffer Context::wrap(void* p, size_t bytes) {
#if defined(STRATA_DS4_RUNTIME)
    if (ds4_gpu_buffer_for_cpu && !std::getenv("DS4_WRAP_COPY")) {
        size_t off = 0;
        if (void* mb = ds4_gpu_buffer_for_cpu(p, bytes, &off)) {
            Buffer r;
            r.handle = (__bridge_retained void*)((__bridge id<MTLBuffer>)mb);
            r.ptr = p;
            r.bytes = bytes;
            r.bias = off;
            return r;
        }
    }
#endif
#if defined(STRATA_DS4_RUNTIME)
    if (std::getenv("DS4_WRAP_LOG"))   // debugging: a wrap that is not inside one runtime allocation gets a second GPU mapping of the pages
        std::fprintf(stderr, "mtl: wrap FALLBACK p=%p bytes=%zu alloc_remaining=%zu\n", p, bytes, ds4_gpu_alloc_remaining ? ds4_gpu_alloc_remaining(p) : (size_t) 0);
#endif
#if defined(STRATA_DS4_RUNTIME)
    Buffer r;
    @autoreleasepool {   // (see alloc: the engine's host threads have no pool of their own)
        id<MTLBuffer> b = [impl_->dev newBufferWithBytesNoCopy:p length:bytes options:MTLResourceStorageModeShared deallocator:nil];
        if (!b) {
            std::fprintf(stderr, "mtl: wrap %zu failed\n", bytes);
            std::exit(4);
        }
        r.handle = (__bridge_retained void*)b;
    }
    allocated_ += bytes;
    r.ptr = p;
    r.bytes = bytes;
    return r;
#else
    id<MTLBuffer> b = [impl_->dev newBufferWithBytesNoCopy:p length:bytes options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) {
        std::fprintf(stderr, "mtl: wrap %zu failed\n", bytes);
        std::exit(4);
    }
    allocated_ += bytes;
    Buffer r;
    r.handle = (__bridge_retained void*)b;
    r.ptr = p;
    r.bytes = bytes;
    return r;
#endif
}

void Context::free(Buffer& b) {
#if defined(STRATA_DS4_RUNTIME)
    // The deferred command buffer is made WITHOUT retained references: a buffer released here while a dispatch that names it
    // is still queued is read after it is gone (or after another alloc took its memory: the job arrays of iq_mv read the
    // NEXT call's jobs, measured as rows of zeros and different tokens every run).  Hold it until the buffer has completed.
    std::lock_guard<std::recursive_mutex> lk_(impl_->mu);
    const bool hold = impl_->cb_def != nil || impl_->inflight != nil;
#endif
    if (b.chunks) {
#if defined(STRATA_DS4_RUNTIME)
        for (void* h : *b.chunks) {
            if (hold) impl_->graveyard.push_back(h);
            else { id<MTLBuffer> x = (__bridge_transfer id<MTLBuffer>)h; x = nil; }
        }
#else
        for (void* h : *b.chunks) { id<MTLBuffer> x = (__bridge_transfer id<MTLBuffer>)h; x = nil; }
#endif
        b = Buffer{};
        return;
    }
    if (b.handle) {
#if defined(STRATA_DS4_RUNTIME)
        if (hold) impl_->graveyard.push_back(b.handle);
        else { id<MTLBuffer> x = (__bridge_transfer id<MTLBuffer>)b.handle; x = nil; }
#else
        id<MTLBuffer> x = (__bridge_transfer id<MTLBuffer>)b.handle;
        x = nil;
#endif
    }
    b = Buffer{};
}

#if defined(STRATA_DS4_RUNTIME)
static std::mutex g_hist_mu;
static std::unordered_map<void*, std::string> g_pipe_names;
static std::unordered_map<std::string, uint64_t> g_kernel_counts;
// DS4_ENC_TIMES: where a dispatch's encoding time goes - the pipeline state, the arguments, the dispatch call
static std::atomic<uint64_t> g_t_pipe{0}, g_t_args{0}, g_t_disp{0};
#endif
static void encode(id<MTLComputeCommandEncoder> enc, const Dispatch& d) {
    require_fit(d, "dispatch");
#if defined(STRATA_DS4_RUNTIME)
    static const bool enc_times = std::getenv("DS4_ENC_TIMES") != nullptr;
    const auto t_a = std::chrono::steady_clock::now();
#endif
    [enc setComputePipelineState:P(d.p)];
#if defined(STRATA_DS4_RUNTIME)
    const auto t_b = std::chrono::steady_clock::now();
#endif
    for (size_t i = 0; i < d.args.size(); ++i) {
        const Arg& a = d.args[i];
        if (a.buf) {
            size_t o;
            id<MTLBuffer> mb = BO(*a.buf, a.off, o);
#if defined(STRATA_DS4_RUNTIME)
            static const bool chkbuf = std::getenv("DS4_CHECK_BUF") != nullptr;
            if (chkbuf && o >= [mb length]) std::fprintf(stderr, "mtl: %s arg %zu: offset %zu past the buffer (%zu bytes; Buffer.bytes %zu bias %zu off %zu)\n", d.p->name.c_str(), i, o, (size_t) [mb length], a.buf->bytes, a.buf->bias, (size_t) a.off);
#endif
            [enc setBuffer:mb offset:o atIndex:i];
        }
        else
            [enc setBytes:a.bytes.data() length:a.bytes.size() atIndex:i];
    }
#if defined(STRATA_DS4_RUNTIME)
    const auto t_c = std::chrono::steady_clock::now();
#endif
    if (d.tg_mem) [enc setThreadgroupMemoryLength:(d.tg_mem + 15) / 16 * 16 atIndex:0];
    if (d.indirect)
        [enc dispatchThreadgroupsWithIndirectBuffer:B(*d.indirect)
                               indirectBufferOffset:d.ind_off
                              threadsPerThreadgroup:MTLSizeMake(d.tg[0], d.tg[1], d.tg[2])];
    else
        [enc dispatchThreadgroups:MTLSizeMake(d.grid[0], d.grid[1], d.grid[2])
            threadsPerThreadgroup:MTLSizeMake(d.tg[0], d.tg[1], d.tg[2])];
#if defined(STRATA_DS4_RUNTIME)
    if (enc_times) {
        const auto t_d = std::chrono::steady_clock::now();
        g_t_pipe.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t_b - t_a).count(), std::memory_order_relaxed);
        g_t_args.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t_c - t_b).count(), std::memory_order_relaxed);
        g_t_disp.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t_d - t_c).count(), std::memory_order_relaxed);
    }
}

static std::deque<std::string> g_recent;   // the last dispatches encoded (printed with a command buffer failure)
static std::string recent_str() { std::string r; for (auto& n : g_recent) r += n + " "; return r; }
static double g_gpu_ms = 0.0;
static std::atomic<uint64_t> g_gpu_busy_us{0};   // DS4_GPU_TOTAL: the GPU time of every committed command buffer
static void add_busy_handler(id<MTLCommandBuffer> cb) {
    static const bool on = std::getenv("DS4_GPU_TOTAL") != nullptr;
    if (on) [cb addCompletedHandler:^(id<MTLCommandBuffer> c) { const double d = c.GPUEndTime - c.GPUStartTime; if (d > 0) g_gpu_busy_us += (uint64_t) (d * 1e6); }];
}   // GPU time of every committed command buffer, summed
static std::atomic<uint64_t> g_wait_ns{0}, g_enc_ns{0}, g_wait_n{0};   // host time: waiting on command buffers; encoding dispatches
static std::mutex g_prof_mu;
static std::unordered_map<std::string, std::pair<double, uint64_t>> g_prof;   // DS4_PROF: kernel -> (GPU us, dispatches)
static std::string g_prof_last;
static std::atomic<uint64_t> g_cb_count{0};
static bool g_count_reg = false;
static void ds4_report_counts() {
    if (g_gpu_busy_us.load()) std::fprintf(stderr, "mtl: GPU busy (every command buffer) %.1f ms\n", g_gpu_busy_us.load() / 1000.0);
    std::fprintf(stderr, "mtl: %.1f ms GPU, %llu dispatches, %llu command buffers\n", g_gpu_ms,
                 (unsigned long long) g_disp_count.load(), (unsigned long long) g_cb_count.load());
    std::fprintf(stderr, "mtl: host waited %.1f ms for the GPU in %llu waits; encoding the dispatches took %.1f ms\n", g_wait_ns.load() / 1e6, (unsigned long long) g_wait_n.load(), g_enc_ns.load() / 1e6);
    if (g_t_pipe.load() || g_t_args.load() || g_t_disp.load())
        std::fprintf(stderr, "mtl: of the encoding: pipeline %.1f ms, arguments %.1f ms, dispatch %.1f ms\n",
                     g_t_pipe.load() / 1e6, g_t_args.load() / 1e6, g_t_disp.load() / 1e6);
    if (!g_prof.empty()) {
        std::vector<std::pair<double, std::string>> pv;
        double tot = 0;
        for (auto& kv : g_prof) { pv.emplace_back(kv.second.first, kv.first); tot += kv.second.first; }
        std::sort(pv.begin(), pv.end(), [](auto& a, auto& b) { return a.first > b.first; });
        std::fprintf(stderr, "prof: kernel                          GPU ms    n   us/dispatch   (total %.1f ms)\n", tot / 1000.0);
        for (size_t i = 0; i < pv.size() && i < 30; ++i) {
            auto& e = g_prof[pv[i].second];
            std::fprintf(stderr, "prof: %-30s %8.1f %6llu %8.1f\n", pv[i].second.c_str(), e.first / 1000.0, (unsigned long long) e.second, e.first / (double) e.second);
        }
    }
    std::vector<std::pair<uint64_t, std::string>> v;
    { std::lock_guard<std::mutex> lk(g_hist_mu); for (auto& kv : g_kernel_counts) v.emplace_back(kv.second, kv.first); }
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < v.size() && i < 14; ++i)
        std::fprintf(stderr, "  %6llu  %s\n", (unsigned long long) v[i].first, v[i].second.c_str());
}

extern "C" unsigned long long ds4_dispatch_seq() { return g_disp_count.load(); }

extern "C" void ds4_gpu_sync_all_streams_weak() __asm("_ds4_gpu_sync_all_streams") __attribute__((weak_import));
void Context::split_runs() {
    // (the streams' pending copies come first, as in a full flush: they are queued on another MTLCommandQueue and would
    // otherwise not have run when the commands committed here read what they write)
    if (ds4_gpu_sync_all_streams_weak) ds4_gpu_sync_all_streams_weak();
    // Commit the batched dispatches as their own command buffer WITHOUT waiting: the queue runs buffers in order and makes each
    // one's writes visible to the next.  (Measured: a producer kernel's output was read stale by the next dispatch of the same
    // buffer for two kernels in a layer; ending the buffer after them is enough, no wait needed.)
    std::lock_guard<std::recursive_mutex> lk_(impl_->mu);
    if (impl_->cb_def == nil) return;
    ++g_cb_count;
    [impl_->enc_def endEncoding];
    if (impl_->qev == nil) impl_->qev = [impl_->dev newSharedEvent];
    [impl_->cb_def encodeSignalEvent:impl_->qev value:++impl_->qev_v];
    add_busy_handler(impl_->cb_def);
    [impl_->cb_def commit];
    impl_->inflight = impl_->cb_def;
    impl_->inflight_names = std::move(impl_->def_names); impl_->def_names.clear();
    impl_->cb_def = nil;
    impl_->enc_def = nil;
    impl_->def_n = 0;
}

// Waiting for a committed command buffer: spin on its status for a while (the thread wake of waitUntilCompleted costs
// ~1 ms a wait; the decode token waits ~100 times), then block.  DS4_NO_SPIN=1: always block.
static void wait_spin(id<MTLCommandBuffer> cb) {
    static const bool no_spin = std::getenv("DS4_NO_SPIN") != nullptr;
    static const double budget = std::getenv("DS4_SPIN_US") ? std::atof(std::getenv("DS4_SPIN_US")) * 1e-6 : 20e-3;
    if (!no_spin) {
        const auto t0 = std::chrono::steady_clock::now();
        while (cb.status < MTLCommandBufferStatusCompleted) {
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > budget) break;
        }
    }
    if (cb.status < MTLCommandBufferStatusCompleted) [cb waitUntilCompleted];
}

void* Context::mark() {
    std::lock_guard<std::recursive_mutex> lk_(impl_->mu);
    if (impl_->cb_def != nil) split_runs();   // (the streams' pending work goes first, inside)
    if (impl_->inflight == nil) return nullptr;
    return (__bridge_retained void*) impl_->inflight;
}

void Context::drop_mark(void* token) {
    // The token holds a +1 on a committed command buffer; one nobody waits for is never freed.
    if (token) { id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>) token; cb = nil; }
}

void Context::wait_mark(void* token) {
    id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>) token;
    const auto tw0 = std::chrono::steady_clock::now();
    wait_spin(cb);
    g_wait_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - tw0).count();
    ++g_wait_n;
    if (cb.status != MTLCommandBufferStatusCompleted)
        std::fprintf(stderr, "mtl: deferred command buffer failed: %s\n", [[cb.error localizedDescription] UTF8String]);
    if (cb.status != MTLCommandBufferStatusCompleted && std::getenv("DS4_ERRINFO")) std::fprintf(stderr, "mtl: recent: %s\n", recent_str().c_str()), std::fprintf(stderr, "mtl: info: %s\n", [[[cb.error userInfo] description] UTF8String]);
}

void Context::flush_runs() {
    if (std::getenv("DS4_TRACE_DISP") && impl_->cb_def != nil) std::fprintf(stderr, "TRACE flush_runs (%d queued)\n", impl_->def_n);
    static const bool cb_log = std::getenv("DS4_CB_LOG") != nullptr;   // debugging: one line a command buffer (what it held)
    if (cb_log && impl_->cb_def != nil) {
        std::string first, last;
        if (!impl_->def_names.empty()) {
            size_t sp = impl_->def_names.find(' ');
            first = impl_->def_names.substr(0, sp);
            size_t e = impl_->def_names.find_last_not_of(' ');
            size_t b = impl_->def_names.find_last_of(' ', e);
            last = impl_->def_names.substr(b == std::string::npos ? 0 : b + 1, e - (b == std::string::npos ? 0 : b));
        }
        std::fprintf(stderr, "CB %d disp  first=%-24s last=%s\n", impl_->def_n, first.c_str(), last.c_str());
    }
    std::lock_guard<std::recursive_mutex> lk_(impl_->mu);
    if (!g_count_reg) { g_count_reg = true; std::atexit(ds4_report_counts); }
    if (impl_->cb_def != nil) {
        ++g_cb_count;
        [impl_->enc_def endEncoding];
        if (impl_->qev == nil) impl_->qev = [impl_->dev newSharedEvent];
        [impl_->cb_def encodeSignalEvent:impl_->qev value:++impl_->qev_v];
    add_busy_handler(impl_->cb_def);
        [impl_->cb_def commit];
        impl_->inflight = impl_->cb_def;
        impl_->inflight_names = std::move(impl_->def_names); impl_->def_names.clear();
        impl_->cb_def = nil;
        impl_->enc_def = nil;
        impl_->def_n = 0;
    }
    if (impl_->inflight != nil) {
        if (std::getenv("DS4_TRACE_WAIT")) { void* fr[8]; const int nf = backtrace(fr, 8); std::fprintf(stderr, "WAIT flush_runs:\n"); backtrace_symbols_fd(fr + 1, nf - 1, 2); }
        const auto tw0 = std::chrono::steady_clock::now();
        wait_spin(impl_->inflight);
        g_wait_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - tw0).count();
        ++g_wait_n;
        { const double d = impl_->inflight.GPUEndTime - impl_->inflight.GPUStartTime; if (d > 0) g_gpu_ms += d * 1000.0;
          if (!g_prof_last.empty()) { std::lock_guard<std::mutex> pl(g_prof_mu); auto& e = g_prof[g_prof_last]; e.first += d * 1e6; e.second += 1; g_prof_last.clear(); } }
        if (impl_->inflight.status != MTLCommandBufferStatusCompleted)
            std::fprintf(stderr, "mtl: deferred command buffer failed: %s\n", [[impl_->inflight.error localizedDescription] UTF8String]);
        if (impl_->inflight.status != MTLCommandBufferStatusCompleted && std::getenv("DS4_ERRINFO")) std::fprintf(stderr, "mtl: failed cb held: %s\n", impl_->inflight_names.c_str());
        impl_->inflight = nil;
    }
    for (void* h : impl_->graveyard) { id<MTLBuffer> x = (__bridge_transfer id<MTLBuffer>)h; x = nil; }
    impl_->graveyard.clear();
#endif
}

double Context::run(const std::vector<Dispatch>& ds, int reps, bool concurrent, bool barrier_between) {
#if defined(STRATA_DS4_RUNTIME)
    std::lock_guard<std::recursive_mutex> lk_(impl_->mu);
#endif
    @autoreleasepool {
#if defined(STRATA_DS4_RUNTIME)
        if (ds.size() == 1 && reps == 1 && !barrier_between) {
            { static const std::string skipk = std::getenv("DS4_SKIP_K") ? std::string(",") + std::getenv("DS4_SKIP_K") + "," : std::string();   // debugging: drop these kernels (timing only)
              if (!skipk.empty() && skipk.find("," + ds[0].p->name + ",") != std::string::npos) return 0.0; }
            if (impl_->cb_def == nil || impl_->def_n >= 2000000) {
                if (impl_->cb_def != nil) split_runs();   // (a buffer committed earlier may still be running: the queue keeps the order)
                if (std::getenv("DS4_ERRINFO")) {
                    MTLCommandBufferDescriptor* cd = [MTLCommandBufferDescriptor new];
                    cd.retainedReferences = NO;
                    cd.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
                    impl_->cb_def = [impl_->q commandBufferWithDescriptor:cd];
                } else
                impl_->cb_def = [impl_->q commandBufferWithUnretainedReferences];
                // (the wait on our own event is only for a *different* queue: this queue already runs the previous buffer
                // before this one, and the wait costs a full GPU round trip at every buffer - 107 buffers a decode token,
                // 0.28 ms each.  DS4_QEV_WAIT=1 restores it.)
                static const bool qev_wait = std::getenv("DS4_NO_QEV_WAIT") == nullptr;   // (measured: off is slower - keep it)
                if (qev_wait && impl_->qev != nil && impl_->qev_v > 0) [impl_->cb_def encodeWaitForEvent:impl_->qev value:impl_->qev_v];
                impl_->enc_def = tl_make_encoder(impl_->dev, impl_->cb_def, std::getenv("DS4_CONCURRENT") ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial);
            }
            const auto te0 = std::chrono::steady_clock::now();
            encode(impl_->enc_def, ds[0]);
            if (g_tl.on && g_tl.att && g_tl.n < g_tl.cap) { g_tl.names[g_tl.n++] = ds[0].p->name; g_tl.att = false; }
            g_last_act_ns.store(ka_now_ns(), std::memory_order_relaxed);
            g_recent.push_back(ds[0].p->name); if (g_recent.size() > 8) g_recent.pop_front();
            if (std::getenv("DS4_ERRINFO")) impl_->def_names += ds[0].p->name + " ";
            g_enc_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - te0).count();
            static const bool trace = std::getenv("DS4_TRACE_DISP") != nullptr;
            if (trace) { std::lock_guard<std::mutex> lk(g_hist_mu); std::fprintf(stderr, "TRACE disp %s\n", ds[0].p->name.c_str()); }
            // Every wrapper binds its own MTLBuffer over the same memory, so Metal's hazard tracking cannot see that one
            // dispatch reads what the previous wrote: without an explicit barrier the reader saw the stale value (measured: the
            // second matvec's activation quantiser read the previous layer's x).
            static const bool no_mb = std::getenv("DS4_NO_MBARRIER") != nullptr;
            if (!no_mb) [impl_->enc_def memoryBarrierWithScope:MTLBarrierScopeBuffers];
            static const bool enc_each = std::getenv("DS4_ENC_EACH") != nullptr || g_tl.on;   // a new encoder per dispatch (debugging, the timeline)
            if (enc_each) {
                [impl_->enc_def endEncoding];
                impl_->enc_def = tl_make_encoder(impl_->dev, impl_->cb_def, (std::getenv("DS4_CONCURRENT") ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial));
            }
            ++impl_->def_n;
            ++g_disp_count;
            static const bool prof = std::getenv("DS4_PROF") != nullptr;   // every dispatch alone, its GPU time kept per kernel
            if (prof) { g_prof_last = ds[0].p->name; if (ds4_gpu_sync_all_streams_weak) ds4_gpu_sync_all_streams_weak(); flush_runs(); return 0.0; }
            static const bool each = std::getenv("DS4_FLUSH_EACH") != nullptr;   // debugging: every dispatch alone, in order
            static const std::string only = std::getenv("DS4_FLUSH_AFTER") ? std::getenv("DS4_FLUSH_AFTER") : "";   // ",name1,name2,"
            const bool listed = !only.empty() && only.find("," + ds[0].p->name + ",") != std::string::npos;
            if (each || listed) {
                if (ds4_gpu_sync_all_streams_weak) ds4_gpu_sync_all_streams_weak();
                flush_runs();
            }
            return 0.0;
        }
        flush_runs();
#endif
        id<MTLCommandBuffer> cb = [impl_->q commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc =
            [cb computeCommandEncoderWithDispatchType:concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
        for (int r = 0; r < reps; ++r)
            for (const auto& d : ds) {
                encode(enc, d);
#if defined(STRATA_DS4_RUNTIME)
                ++g_disp_count;
#endif
                if (barrier_between) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "mtl: command buffer failed: %s\n", [[cb.error localizedDescription] UTF8String]);
            std::exit(5);
        }
        return cb.GPUEndTime - cb.GPUStartTime;
    }
}

double Context::run_graph(const std::vector<Dispatch>& ds) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [impl_->q commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        for (size_t i = 0; i < ds.size(); ++i) {
            if (i > 0 && ds[i].barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode(enc, ds[i]);
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "mtl: command buffer failed: %s\n", [[cb.error localizedDescription] UTF8String]);
            std::exit(5);
        }
        return cb.GPUEndTime - cb.GPUStartTime;
    }
}

double Context::run_graph_seg(const std::vector<Dispatch>& ds, const std::vector<int>& cuts, void (*at_cut)(void*, int),
                              void* user) {
    @autoreleasepool {
        if (impl_->ev == nil) impl_->ev = [impl_->dev newSharedEvent];
        id<MTLSharedEvent> ev = impl_->ev;
        const uint64_t base = impl_->ev_v;
        id<MTLCommandBuffer> cb = [impl_->q commandBufferWithUnretainedReferences];
        size_t i = 0;
        for (size_t s = 0; s <= cuts.size(); ++s) {
            const size_t end = s < cuts.size() ? (size_t) cuts[s] : ds.size();
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            const size_t first = i;
            for (; i < end; ++i) {
                if (i > first && ds[i].barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encode(enc, ds[i]);
            }
            [enc endEncoding];
            if (s < cuts.size()) {
                [cb encodeSignalEvent:ev value:base + 2 * s + 1];
                [cb encodeWaitForEvent:ev value:base + 2 * s + 2];
            }
        }
        [cb commit];
        for (size_t s = 0; s < cuts.size(); ++s) {
            const uint64_t want = base + 2 * s + 1;
            while (ev.signaledValue < want) {
                if (cb.status >= MTLCommandBufferStatusCompleted) break;   // an error: the wait below reports it
            }
            at_cut(user, (int) s);
            ev.signaledValue = base + 2 * s + 2;
        }
        impl_->ev_v = base + 2 * cuts.size() + 2;
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "mtl: command buffer failed: %s\n", [[cb.error localizedDescription] UTF8String]);
            std::exit(5);
        }
        return cb.GPUEndTime - cb.GPUStartTime;
    }
}

double Context::run_wall(const std::vector<Dispatch>& ds, int reps, double* gpu_seconds) {
    double gpu = 0;
    const double t0 = now_s();
    for (int r = 0; r < reps; ++r) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [impl_->q commandBufferWithUnretainedReferences];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (const auto& d : ds) encode(enc, d);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            gpu += cb.GPUEndTime - cb.GPUStartTime;
        }
    }
    if (gpu_seconds) *gpu_seconds = gpu;
    return now_s() - t0;
}

void Context::warm(const std::vector<Dispatch>& ds, double seconds) {
    const double t0 = now_s();
    while (now_s() - t0 < seconds) run(ds);
}

double Context::run_icb(const std::vector<Dispatch>& ds, int reps, bool barrier_between) {
    @autoreleasepool {
        size_t maxb = 1;
        for (const auto& d : ds) maxb = std::max(maxb, d.args.size());
        MTLIndirectCommandBufferDescriptor* desc = [MTLIndirectCommandBufferDescriptor new];
        desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
        desc.inheritBuffers = NO;
        desc.inheritPipelineState = NO;
        desc.maxKernelBufferBindCount = maxb;
        id<MTLIndirectCommandBuffer> icb = [impl_->dev newIndirectCommandBufferWithDescriptor:desc
                                                                              maxCommandCount:ds.size()
                                                                                      options:0];
        NSMutableSet* used = [NSMutableSet new];
        for (size_t i = 0; i < ds.size(); ++i) {
            const Dispatch& d = ds[i];
            require_fit(d, "indirect-command dispatch");
            id<MTLIndirectComputeCommand> c = [icb indirectComputeCommandAtIndex:i];
            [c setComputePipelineState:P(d.p)];
            for (size_t a = 0; a < d.args.size(); ++a) {
                if (!d.args[a].buf) {
                    std::fprintf(stderr, "mtl: run_icb needs buffer arguments only\n");
                    std::exit(5);
                }
                size_t o;
                id<MTLBuffer> mb = BO(*d.args[a].buf, d.args[a].off, o);
                [c setKernelBuffer:mb offset:o atIndex:a];
                [used addObject:mb];
            }
            if (d.tg_mem) [c setThreadgroupMemoryLength:(d.tg_mem + 15) / 16 * 16 atIndex:0];
            [c concurrentDispatchThreadgroups:MTLSizeMake(d.grid[0], d.grid[1], d.grid[2])
                        threadsPerThreadgroup:MTLSizeMake(d.tg[0], d.tg[1], d.tg[2])];
            if (barrier_between) [c setBarrier];
        }
        id<MTLCommandBuffer> cb = [impl_->q commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        for (id<MTLBuffer> b in used) [enc useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        for (int r = 0; r < reps; ++r) {
            [enc executeCommandsInBuffer:icb withRange:NSMakeRange(0, ds.size())];
            if (barrier_between) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "mtl: icb command buffer failed: %s\n", [[cb.error localizedDescription] UTF8String]);
            std::exit(5);
        }
        return cb.GPUEndTime - cb.GPUStartTime;
    }
}

}  // namespace mtl
