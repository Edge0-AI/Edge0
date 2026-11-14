// metal/src/mtl_rt.h - a thin C++ face over Metal for the standalone kernel programs (parity, benches).
//
// The implementation (mtl_rt.mm) is the only Objective-C++ file; every program and test is plain C++ against this
// header.  Kernels are MSL source files compiled at run time (newLibraryWithSource): the Command Line Tools have no
// offline `metal` compiler, and the runtime compiler is part of the OS.  ggml-common.h (MIT, third_party/ggml) is
// prepended for the i-quant grids and block structs.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace mtl {

struct Buffer {
    void* handle = nullptr;   // id<MTLBuffer> (retained); the first chunk of a chunked buffer
    void* ptr = nullptr;      // CPU address (shared storage)
    size_t bytes = 0;
    // a buffer longer than maxBufferLength: overlapping chunks every `stride` bytes (see Context::wrap)
    std::shared_ptr<std::vector<void*>> chunks;
    size_t stride = 0;
    size_t bias = 0;          // bytes from the MTLBuffer's start to ptr (a view onto an allocation the runtime owns)
    template <class T> T* as() const { return static_cast<T*>(ptr); }
};

struct Pipeline {
    void* handle = nullptr;   // id<MTLComputePipelineState>
    int max_threads = 0;      // maxTotalThreadsPerThreadgroup
    int simd_width = 0;       // threadExecutionWidth
    int static_tg_mem = 0;    // staticThreadgroupMemoryLength
    std::string name;         // the function (+ its constants), for messages
};

// A function-constant value set (index -> int / bool / uint).
struct Constants {
    struct V { int index; int type; uint32_t v; };  // type 0 int, 1 bool, 2 uint, 3 short, 4 ushort
    std::vector<V> vals;
    Constants& i(int idx, int v) { vals.push_back({idx, 0, (uint32_t)v}); return *this; }
    Constants& b(int idx, bool v) { vals.push_back({idx, 1, (uint32_t)v}); return *this; }
    Constants& u(int idx, uint32_t v) { vals.push_back({idx, 2, v}); return *this; }
};

struct Arg {           // one argument of a dispatch: a buffer (+ offset) or inline bytes
    const Buffer* buf = nullptr;
    size_t off = 0;
    std::vector<uint8_t> bytes;
    static Arg B(const Buffer& b, size_t off = 0) { Arg a; a.buf = &b; a.off = off; return a; }
    template <class T> static Arg V(const T& v) {
        Arg a; a.bytes.resize(sizeof(T)); std::memcpy(a.bytes.data(), &v, sizeof(T)); return a;
    }
};

struct Dispatch {
    const Pipeline* p = nullptr;
    std::vector<Arg> args;
    uint32_t grid[3] = {1, 1, 1};     // threadgroups
    uint32_t tg[3] = {32, 1, 1};      // threads per threadgroup
    uint32_t tg_mem = 0;              // dynamic threadgroup memory at index 0 (bytes)
    const Buffer* indirect = nullptr; // grid from a GPU buffer (3 x uint32 threadgroup counts) at ind_off
    size_t ind_off = 0;
    bool barrier = true;              // run_graph: a buffer memory barrier before this dispatch
};

// A pipeline whose maxTotalThreadsPerThreadgroup is below the threadgroup size a dispatch asks for is NOT executed by
// Metal and raises no error (0.1.44, #2323: gdn_step needs 512 threads, an M2 Max gave 448 for the GDN_NTOK >= 2
// specializations - the layers silently kept their old outputs and every answer was garbage).  So every dispatch is
// checked: too few threads is fatal, with a message that names the kernel, the limit and the size needed.
// STRATA_METAL_MAX_THREADS_CAP=N lowers every pipeline's reported limit to N (a test hook: it proves this check).
void require_fit(const Pipeline& p, uint32_t threads, const char* where);
inline void require_fit(const Dispatch& d, const char* where = "dispatch") {
    require_fit(*d.p, d.tg[0] * d.tg[1] * d.tg[2], where);
}

class Context {
public:
    Context();
    ~Context();
    std::string device_name() const;
    bool has_family_apple(int n) const;          // MTLGPUFamilyApple<n>
    bool supports_metal4() const;
    uint64_t recommended_working_set() const;
    uint64_t max_buffer_length() const;
    // Keeps the GPU out of its low-clock idle state while dispatches keep coming (a lone kernel in a quiet GPU runs 2-3x slower:
    // measured 205 vs 69 us for the same matvec): a spin kernel of one threadgroup on a second queue, stopped 50 ms after the
    // last dispatch.  DS4_NO_KEEPALIVE=1: off.
    void start_keepalive();
    // Zero-copy weights: the GPU address of host memory [p, p + bytes) (file-backed pages of the model, RAM slots) through a
    // no-copy buffer over the pages it spans, made resident.  0: not mappable (the cap on buffers is reached, or the pages are not
    // ours).  The first dispatch that reads cold pages waits for them (the buffer is wired on use).
    uint64_t zc_map(const void* p, size_t bytes);
    Pipeline pipeline_uncached(int lib, const std::string& fn, const Constants& c);
    size_t zc_count() const;
    uint64_t allocated() const { return allocated_; }   // bytes of alloc() + wrap() buffers so far

    // Compiles a library from MSL source files (paths), prepended by `prelude` (defines) and, if include_ggml,
    // ggml-common.h with GGML_COMMON_DECL_METAL/IMPL_METAL.  Returns a library id (index); aborts with the
    // compiler log on error.
    // Safe math unless STRATA_METAL_MATH=fast (the argument is ignored; kept for callers).
    int compile(const std::vector<std::string>& files, const std::string& prelude = "", bool fast_math = false);
    Pipeline pipeline(int lib, const std::string& fn, const Constants& c = {});

    Buffer alloc(size_t bytes);                  // shared storage, zeroed
    Buffer wrap(void* page_aligned, size_t bytes);  // no-copy wrap (page-aligned, size multiple of page)
    // the same, also when bytes > maxBufferLength: then overlapping chunks, each holding any range of up to
    // max_span bytes that starts in it (max_span = the largest tensor bound at one offset)
    Buffer wrap(void* page_aligned, size_t bytes, size_t max_span);
    void free(Buffer& b);

    // Encodes `ds` into ONE compute encoder (serial or concurrent dispatch type) and ONE command buffer, `reps`
    // times over; commits and waits.  Returns the GPU time (GPUEndTime - GPUStartTime) in seconds.
    void* mark();                 // split_runs() that returns a token for the buffer just committed (nullptr: nothing was queued)
    void wait_mark(void* token);  // block until that buffer has completed, then drop the token
    void drop_mark(void* token);  // drop a token without waiting (an event recorded again, or destroyed, before it was waited on)
    void split_runs();   // commit the batched dispatches as their own command buffer, without waiting
    void flush_runs();   // commit and wait for the dispatches batched since the last flush
    double run(const std::vector<Dispatch>& ds, int reps = 1, bool concurrent = false, bool barrier_between = false);
    // A dependency graph in ONE concurrent encoder: a memory barrier only before dispatches with `barrier` set, so
    // the dispatches between two barriers may overlap.  Returns the GPU time in seconds.
    double run_graph(const std::vector<Dispatch>& ds);
    // run_graph in segments: the dispatches [cuts[s-1], cuts[s]) are one concurrent encoder; after segment s (for
    // s < cuts.size()) the GPU signals a shared event and waits until the host has run at_cut(s) - the host's turn
    // between two parts of one committed command buffer (the SSD-streaming handshake: the routing is read, the
    // missing experts are put in place).  Returns the GPU time in seconds.
    double run_graph_seg(const std::vector<Dispatch>& ds, const std::vector<int>& cuts, void (*at_cut)(void*, int),
                         void* user);
    // One command buffer per `run_cb` call, empty or not: measures commit+wait round trips.
    double run_wall(const std::vector<Dispatch>& ds, int reps, double* gpu_seconds = nullptr);

    // Indirect command buffer: the dispatches recorded once, executed `reps` times in one command buffer.
    // Buffers are bound per command (setKernelBuffer); inline bytes are not allowed (use buffers).
    double run_icb(const std::vector<Dispatch>& ds, int reps, bool barrier_between);
    // Runs `ds` repeatedly for about `seconds` of wall time: lifts the GPU out of its idle clocks before a timing.
    void warm(const std::vector<Dispatch>& ds, double seconds = 0.3);

    std::string ggml_common_path;                // where ggml-common.h is read from
    std::string kernel_dir;                      // where relative kernel paths resolve

private:
    struct Impl;
    Impl* impl_;
    uint64_t allocated_ = 0;
};

double now_s();                                  // monotonic wall clock (seconds)
std::string read_file(const std::string& path);

}  // namespace mtl
