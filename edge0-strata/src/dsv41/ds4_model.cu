// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_model.cu - DeepSeek-V4.1-Flash on Maya (see ds4_model.hpp).
//
// The layer, for one token at position p (DeepSeek's inference/model.py, Block.forward with the lagged pre-mix):
//   [Engram, layers 1 and 14]  h += gate(h, key) * value from 24 n-gram rows of the FP8 tables
//   a_pre, a_post, a_comb = hc_mixes(h);  x = rmsnorm(sum_s pre[s] h[s])   (pre: the previous layer's f_pre)
//   attention(x): q from the low-rank wq_a / wq_b, one shared K = V row per token in a 128-slot window ring, plus
//                 (compress ratio > 0) up to 512 compressed rows the indexer picks; a learned sink per head; the
//                 output's rotary part rotated back, rounded to FP8, through the grouped wo_a and wo_b
//   h = post * attn + comb^T h;  f_* = hc_mixes(h);  x = rmsnorm(sum_s a_pre[s] h[s])
//   moe(x): 6 of 384 routed experts (sqrt-softplus scores, biased top-k) + the shared expert
//   h = post * moe + comb^T h;  pre = f_pre
// and after the last layer logits = head(rmsnorm(sum_s pre[s] h[s])).
#include "ds4_launch.h"   // strata_ds4_launch: <<<>>> on CUDA and HIP, a dispatch on Metal
#include "ds4_model.hpp"
#include "ds4_pick_rule.hpp"
#include "ds4_experts_rule.hpp"
#include "ds4_engram_io.hpp"

#include "ds4_gguf.cuh"
#include "ds4_iq.cuh"
#include "ds4_gguf_cpu.hpp"
#include "ds4_kernels.cuh"
#include "ds4_os.hpp"
#include "ds4_pool.hpp"
#include "ds4_store.hpp"
#include "ds4_vision.cuh"
#include "strata/dsv41/exl3_cpu.hpp"
#include "strata/dsv41/exl3.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/dsv41/moe_win.hpp"
#include "strata/dsv41/sampler.hpp"
#if defined(STRATA_USE_HIP)
#include "wmma_gemm.h"   // strata_pf_gemm_f16_ld: the RDNA3 WMMA dense GEMM (gemm() prefers it over hipBLAS on gfx11; -Isrc/prefill)
#endif

#include <cublas_v2.h>
#include <cuda_runtime.h>

#if defined(STRATA_METAL)
extern "C" void ds4_busy_begin();
extern "C" void ds4_busy_end();
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif
#include <algorithm>
#include <array>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_map>

#if defined(_MSC_VER)
#define strtok_r strtok_s
#endif

namespace strata::ds4 {
namespace x3 = strata::dsv41k::exl3;
namespace kc = strata::dsv41k::cpu;
namespace gf = strata::dsv41k::glmf;
namespace gg = strata::ds4::gguf;

#define CK(call)                                                                            \
    do {                                                                                    \
        cudaError_t e_ = (call);                                                            \
        if (e_ != cudaSuccess) {                                                            \
            err = std::string(#call) + ": " + cudaGetErrorString(e_);                       \
            return false;                                                                   \
        }                                                                                   \
    } while (0)

namespace {

// A HIP build (the share-based defaults below are gated on the devices being RDNA4, Impl::rdna4()).
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
constexpr bool kHipBuild = true;
#else
constexpr bool kHipBuild = false;
#endif

// an environment variable as 0 / 1, or -1 when it is not set (the rules in ds4_experts_rule.hpp take this form)
int env_tri(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr ? (std::atoi(v) != 0 ? 1 : 0) : -1;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// rows past the chunk that the Metal expert tile may read (its 16-row load of the last partial tile); CUDA and HIP allocate none
#if defined(STRATA_METAL)
constexpr size_t kH16Pad = 16;
#else
constexpr size_t kH16Pad = 0;
#endif
// the embedding rows' upload: asynchronous on the device's stream; on Metal a synchronous copy (the batched dispatches read the rows when
// they are committed, and h_emb is written again by the next window)
inline cudaError_t emb_upload(void* dst, const void* src, size_t n, cudaStream_t s) {
#if defined(STRATA_METAL)
    (void) s;
    return cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice);
#else
    return cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, s);
#endif
}

size_t align256(size_t n) { return (n + 255) & ~(size_t) 255; }

// A run of chunks' device-side waits for another group (xfer_get, wait_flags: flag >= seq) are queued only once the
// host sees the flag up, so the kernel finds it up and never spins.  A wait kernel spinning on a device while its
// thread goes on queueing the layer stalled that thread's first cuBLAS call behind it, and with it the other
// devices' (two H100s, the IQ2_XS pack on a layer split: both threads stopped in cublasSgemm, hc_mix_chunk).
void host_wait_at_least(const unsigned* flag, unsigned seq) {
    for (int spin = 0; (int) (*(volatile const unsigned*) flag - seq) < 0; ++spin) {
        if (spin < 1000) std::this_thread::yield();
        else std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
}

// part p of np of `whole` (a multiple of 128) in 128-blocks, as even as they go (whole / 128 % np parts a block more:
// those from part rot on, cyclically - so a group's layers, each its own rot, even out what each member holds):
// tensor parallel's columns of an FFN or of the head - an even split is what whole / np gave before
void split128(int whole, int np, int p, int& off, int& n, int rot = 0) {
    const int b = whole / 128, q = b / np, r = b % np;
    auto blocks = [&](int i) { return q + (((i - rot) % np + np) % np < r ? 1 : 0); };
    int at = 0;
    for (int i = 0; i < p; ++i) at += blocks(i);
    off = 128 * at;
    n = 128 * blocks(p);
}

std::string read_file(const std::string& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
std::vector<int> cpu_list(const std::string& s) {   // "0-3,8,10-11"
    std::vector<int> v;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty() || part == "\n") continue;
        const size_t dash = part.find('-');
        const int a = std::atoi(part.c_str());
        const int b = dash == std::string::npos ? a : std::atoi(part.c_str() + dash + 1);
        for (int c = a; c <= b; ++c) v.push_back(c);
    }
    return v;
}
// each NUMA node's physical cores (the first logical CPU of each core), in order
std::vector<std::vector<int>> node_cores() {
#if defined(_WIN32)
    static const bool smt = [] {   // DS4_CPU_SMT=1: the second hardware thread of every core as well (memory-bound rows gain from it)
        const char* v = std::getenv("DS4_CPU_SMT");
        return v != nullptr && std::atoi(v) != 0;
    }();
    std::vector<std::vector<int>> nodes = os::node_cores(smt);
#else
    std::vector<std::vector<int>> nodes;
    for (int n = 0; n < 64; ++n) {
        const std::string cl = read_file("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist");
        if (cl.empty()) break;
        std::vector<int> cores;
        for (int c : cpu_list(cl)) {
            const std::vector<int> sib = cpu_list(
                read_file("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"));
            if (sib.empty() || sib[0] == c) cores.push_back(c);
        }
        if (!cores.empty()) nodes.push_back(cores);
    }
#endif
    if (nodes.empty()) {
        std::vector<int> all;
        for (unsigned c = 0; c < std::max(1u, std::thread::hardware_concurrency() / 2); ++c) all.push_back((int) c);
        nodes.push_back(all);
    }
    return nodes;
}
// anonymous memory on one NUMA node (node < 0: wherever the first touch puts it), transparent huge pages
void* node_map(size_t bytes, int node, int n_nodes) { return os::anon_alloc(bytes, node, n_nodes); }
void pin_self(int cpu) { os::pin_self(cpu); }
// the RAM this process can still get: MemAvailable, and below it the tightest cgroup v2 limit on the way to the root
// (its limit less what is charged, the page cache counted as free: the kernel takes it back); ~0: not known
uint64_t ram_avail() {
#if defined(_WIN32)
    return os::ram_avail();
#endif
    uint64_t avail = ~uint64_t{0};
    {
        std::istringstream mi(read_file("/proc/meminfo"));
        std::string k;
        uint64_t kb = 0;
        while (mi >> k >> kb) {
            if (k == "MemAvailable:") avail = kb << 10;
            mi.ignore(64, '\n');
        }
    }
    std::string cg = read_file("/proc/self/cgroup");   // "0::/user.slice/..."
    const size_t at = cg.find("0::");
    if (at == std::string::npos) return avail;
    cg = cg.substr(at + 3);
    cg = cg.substr(0, cg.find('\n'));
    for (;;) {
        const std::string dir = "/sys/fs/cgroup" + (cg == "/" ? std::string() : cg);
        const std::string mx = read_file(dir + "/memory.max");
        if (!mx.empty() && mx.compare(0, 3, "max") != 0) {
            const uint64_t lim = std::strtoull(mx.c_str(), nullptr, 10);
            const uint64_t cur = std::strtoull(read_file(dir + "/memory.current").c_str(), nullptr, 10);
            uint64_t file = 0;
            std::istringstream st(read_file(dir + "/memory.stat"));
            std::string k;
            uint64_t v = 0;
            while (st >> k >> v)
                if (k == "file") file = v;
            const uint64_t used = cur > file ? cur - file : 0;
            avail = std::min(avail, lim > used ? lim - used : 0);
        }
        if (cg.empty() || cg == "/") break;
        const size_t sl = cg.rfind('/');
        cg = sl == 0 || sl == std::string::npos ? "/" : cg.substr(0, sl);
    }
    return avail;
}

// the device memory the experts' placement may count on: cudaMemGetInfo's free, but on an integrated NVIDIA GPU on Linux
// (a DGX Spark's GB10, one memory with the CPU) what the kernel can still give back, MemAvailable less
// DS4_UMA_HEADROOM_GIB (6): cudaMemGetInfo counts only MemFree there, and the model's files just read fill the page cache
// (measured on a GB10: 55 GB "free" with 115 GB available).  Not on HIP: a Strix Halo's GPU memory is its carve-out plus
// the GTT pool, which hipMemGetInfo already counts and MemAvailable may exceed
size_t uma_free(int dev, size_t fr) {
#if !defined(_WIN32) && !defined(STRATA_USE_HIP)
    int integrated = 0;
    if (cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return fr;
    }
    if (!integrated) return fr;
    const char* h = std::getenv("DS4_UMA_HEADROOM_GIB");
    const uint64_t head = (uint64_t) (h ? std::atof(h) : 6.0) << 30, avail = ram_avail();
    if (avail != ~uint64_t{0} && avail > head && avail - head > fr) return (size_t) (avail - head);
#else
    (void) dev;
#endif
    return fr;
}

float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else {   // subnormal
            int ee = -1;
            uint32_t mm = m;
            do {
                ++ee;
                mm <<= 1;
            } while ((mm & 0x400) == 0);
            u = s | (uint32_t) (127 - 15 - ee) << 23 | (mm & 0x3ff) << 13;
        }
    } else if (e == 31) u = s | 0x7f800000u | m << 13;
    else u = s | (e - 15 + 127) << 23 | m << 13;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ------------------------------------------------------------------ the pack
struct Tensor {
    std::string dtype;
    int file = -1;
    uint64_t off = 0, bytes = 0;
    std::vector<int64_t> shape;
};

struct Pack {
    std::string dir, model_dir;
    std::unordered_map<std::string, std::string> kv;
    std::vector<std::string> files;
    std::vector<os::fd_t> fds;
    std::unordered_map<std::string, Tensor> t;
    int eg_vocab = 0, eg_layers = 0, eg_ngram = 0, eg_heads = 0, eg_hd = 0, eg_pad = 0;
    std::vector<int32_t> eg_map;
    std::vector<int64_t> eg_mult, eg_primes, eg_off;
    // format gguf (tools/ds4/pack_gguf.py): the files are a GGUF's shards, every one mapped at load (map()) - the
    // routed experts are always read in place (the OS page cache their RAM tier, the disk the rest)
    bool gguf = false;

    std::vector<std::pair<uint8_t*, size_t>> maps;   // file -> its read-only mapping (map(): the experts in place)
    std::vector<os::fd_t> dfds;   // (gguf) file -> unbuffered (os::open_direct; kBadFd: buffered reads only)
    std::vector<os::fd_t> pfds;   // (gguf) file -> for reads from many threads at once (os::open_par)

    ~Pack() {
        for (auto& m : maps) os::unmap(m.first, m.second);
        for (os::fd_t fd : fds) os::close_fd(fd);
        for (os::fd_t fd : dfds) os::close_fd(fd);
        for (os::fd_t fd : pfds) os::close_fd(fd);
    }
    // file i mapped whole, read-only (once; not thread-safe: called while loading)
    const uint8_t* map(int file) {
        if (maps.size() < fds.size()) maps.resize(fds.size(), {nullptr, 0});
        auto& m = maps[(size_t) file];
        if (m.first == nullptr) {
            const uint64_t size = os::file_size(fds[(size_t) file]);
            void* p = os::map_ro(fds[(size_t) file], size);
            if (p == nullptr) return nullptr;
            m = {(uint8_t*) p, (size_t) size};
        }
        return m.first;
    }

    bool load(const std::string& d, std::string& err) {
        dir = d;
        {
            std::ifstream f(d + "/ds4.txt");
            if (!f) {
                err = d + "/ds4.txt: not found (run tools/ds4/pack.py --model <model>)";
                return false;
            }
            std::string line;
            while (std::getline(f, line)) {
                const size_t sp = line.find(' ');
                if (sp == std::string::npos) continue;
                kv[line.substr(0, sp)] = line.substr(sp + 1);
            }
        }
        model_dir = str("model_dir");
        std::FILE* f = std::fopen((d + "/tensors.txt").c_str(), "r");
        if (f == nullptr) {
            err = d + "/tensors.txt: not found";
            return false;
        }
        std::vector<char> buf(1 << 16);
        while (std::fgets(buf.data(), (int) buf.size(), f) != nullptr) {
            char* s = buf.data();
            if (s[0] == 'f') {   // file <i> <name>
                char name[4096];
                int i = 0;
                if (std::sscanf(s, "file %d %4095s", &i, name) == 2) {
                    if ((int) files.size() <= i) files.resize((size_t) i + 1);
                    files[(size_t) i] = name;
                }
                continue;
            }
            if (s[0] != 't') continue;
            char* save = nullptr;
            ::strtok_r(s, " \n", &save);
            const char* name = ::strtok_r(nullptr, " \n", &save);
            const char* dt = ::strtok_r(nullptr, " \n", &save);
            const char* fi = ::strtok_r(nullptr, " \n", &save);
            const char* off = ::strtok_r(nullptr, " \n", &save);
            const char* nb = ::strtok_r(nullptr, " \n", &save);
            if (name == nullptr || nb == nullptr) continue;
            Tensor x;
            x.dtype = dt;
            x.file = std::atoi(fi);
            x.off = std::strtoull(off, nullptr, 10);
            x.bytes = std::strtoull(nb, nullptr, 10);
            while (const char* dim = ::strtok_r(nullptr, " \n", &save)) x.shape.push_back(std::atoll(dim));
            t.emplace(name, std::move(x));
        }
        std::fclose(f);
        for (const std::string& fn : files) {
            const std::string p = model_dir + "/" + fn;
            const os::fd_t fd = os::open_ro(p);
            if (fd == os::kBadFd) {
                err = p + ": cannot open";
                return false;
            }
            fds.push_back(fd);
        }
        gguf = str("format") == "gguf";
        if (gguf)
            for (size_t i = 0; i < fds.size(); ++i) {
                if (map((int) i) == nullptr) {
                    err = model_dir + "/" + files[i] + ": cannot map";
                    return false;
                }
                dfds.push_back(os::open_direct(model_dir + "/" + files[i]));
                pfds.push_back(os::open_par(model_dir + "/" + files[i]));
            }
        // the Engram tables
        std::FILE* e = std::fopen((d + "/engram.bin").c_str(), "rb");
        if (e == nullptr) {
            err = d + "/engram.bin: not found";
            return false;
        }
        char magic[4];
        int32_t hdr[6];
        bool ok = std::fread(magic, 1, 4, e) == 4 && std::memcmp(magic, "DSE1", 4) == 0 &&
                  std::fread(hdr, 4, 6, e) == 6;
        if (ok) {
            eg_vocab = hdr[0], eg_layers = hdr[1], eg_ngram = hdr[2], eg_heads = hdr[3], eg_hd = hdr[4], eg_pad = hdr[5];
            const int cols = (eg_ngram - 1) * eg_heads;
            eg_map.resize((size_t) eg_vocab);
            eg_mult.resize((size_t) eg_layers * eg_ngram);
            eg_primes.resize((size_t) eg_layers * cols);
            eg_off.resize((size_t) eg_layers * cols);
            ok = std::fread(eg_map.data(), 4, eg_map.size(), e) == eg_map.size() &&
                 std::fread(eg_mult.data(), 8, eg_mult.size(), e) == eg_mult.size() &&
                 std::fread(eg_primes.data(), 8, eg_primes.size(), e) == eg_primes.size() &&
                 std::fread(eg_off.data(), 8, eg_off.size(), e) == eg_off.size();
        }
        std::fclose(e);
        if (!ok) {
            err = d + "/engram.bin: unreadable";
            return false;
        }
        return true;
    }

    std::string str(const std::string& k, const std::string& def = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }
    double num(const std::string& k, double def) const {
        auto it = kv.find(k);
        return it == kv.end() ? def : std::atof(it->second.c_str());
    }
    std::vector<int> list(const std::string& k) const {
        std::vector<int> v;
        std::stringstream ss(str(k));
        std::string x;
        while (std::getline(ss, x, ','))
            if (!x.empty()) v.push_back(std::atoi(x.c_str()));
        return v;
    }
    const Tensor* get(const std::string& n) const {
        auto it = t.find(n);
        return it == t.end() ? nullptr : &it->second;
    }
    bool direct_reads = false;   // (after drop_views: the reads go unbuffered, the cached ones are what the slow path is)
    bool read_at(int file, uint64_t off, uint64_t n, void* dst) const {   // (any thread; side by side where pfds is)
        if (direct_reads && (size_t) file < dfds.size() && dfds[(size_t) file] != os::kBadFd) {
            thread_local std::vector<uint8_t> bounce;
            const uint64_t span = os::direct_span(off, n);
            if (bounce.size() < span + 2 * os::kDirectAlign) bounce.resize((size_t) span + 2 * os::kDirectAlign);
            uint8_t* bp = (uint8_t*) (((uintptr_t) bounce.data() + os::kDirectAlign - 1) & ~(uintptr_t) (os::kDirectAlign - 1));
            if (os::pread_direct(dfds[(size_t) file], bp, n, off)) {
                std::memcpy(dst, bp + off % os::kDirectAlign, (size_t) n);
                return true;
            }
        }
        if ((size_t) file < pfds.size() && pfds[(size_t) file] != os::kBadFd)
            return os::pread_par(pfds[(size_t) file], dst, n, off);
        return os::pread_full(fds[(size_t) file], dst, n, off);
    }
    bool read(const Tensor& x, void* dst) const { return read_at(x.file, x.off, x.bytes, dst); }
    const uint8_t* at(int file, uint64_t off) const {   // (gguf: mapped; null once the views are dropped)
        return (size_t) file < maps.size() && maps[(size_t) file].first != nullptr ? maps[(size_t) file].first + off : nullptr;
    }
    // Windows: unbuffered reads of a file that has ANY view in this process run at half the drive's rate (a probe on this
    // box: 6.5 GB/s with no view, 3.2 with a view of the whole file or of its first 1 GB, 6.7 on a copy while the original
    // is mapped).  Once the load is done the experts reach the CPU lane through the store's slots, so the views go.
    // A file with cached (buffered) reads and a handle still open is slow the same way (probe: 3.3 GB/s with a buffered
    // handle held open after one read, 6.5 once it is closed), so the load's buffered handles are closed and fresh ones
    // (never read from) stand in for the fallbacks; the reads that remain go unbuffered (direct_reads).
    void drop_views() {
        for (auto& m : maps)
            if (m.first != nullptr) {
                os::unmap(m.first, m.second);
                m = {nullptr, 0};
            }
        for (size_t i = 0; i < fds.size(); ++i) {
            const std::string path = model_dir + "/" + files[i];
            os::close_fd(fds[i]);
            fds[i] = os::open_ro(path);
            if (i < pfds.size()) {
                os::close_fd(pfds[i]);
                pfds[i] = os::open_par(path);
            }
        }
        direct_reads = true;
    }
};

// ------------------------------------------------------------------ geometry
struct Geo {
    int vocab = 0, dim = 0, nl = 0, heads = 0, hd = 0, rd = 0, q_lora = 0, o_lora = 0, o_groups = 0, n_exp = 0,
        k_exp = 0, ff = 0, win = 0, hc = 0, sinkhorn = 0, ix_heads = 0, ix_hd = 0, ix_topk = 0, cand_src = -1,
        cand_topk = 0, cand_blk = 0, orig_ctx = 0, eos = 1;
    float eps = 1e-20f, hc_eps = 1e-6f, route_scale = 1.0f, lim = 0.0f;
    double rope_theta = 1e4, comp_theta = 1e4, factor = 1.0, beta_fast = 32, beta_slow = 1;
    std::vector<int> ratios, kv_src, ix_src, eg_layers;
    // the DSpark drafter (the checkpoint's mtp.* blocks): a block of [seed, noise x (block - 1)] through n_mtp blocks
    int n_mtp = 0, d_block = 0, d_noise = 0, d_rank = 0, d_n_exp = 0, d_k = 0;
    std::vector<int> d_targets;

    void from(const Pack& p) {
        vocab = (int) p.num("vocab_size", 0), dim = (int) p.num("hidden_size", 0);
        nl = (int) p.num("num_hidden_layers", 0), heads = (int) p.num("num_attention_heads", 0);
        hd = (int) p.num("head_dim", 0), rd = (int) p.num("qk_rope_head_dim", 0);
        q_lora = (int) p.num("q_lora_rank", 0), o_lora = (int) p.num("o_lora_rank", 0);
        o_groups = (int) p.num("o_groups", 0), n_exp = (int) p.num("n_routed_experts", 0);
        k_exp = (int) p.num("num_experts_per_tok", 0), ff = (int) p.num("moe_intermediate_size", 0);
        win = (int) p.num("sliding_window", 0), hc = (int) p.num("hc_mult", 0);
        sinkhorn = (int) p.num("hc_sinkhorn_iters", 0), ix_heads = (int) p.num("index_n_heads", 0);
        ix_hd = (int) p.num("index_head_dim", 0), ix_topk = (int) p.num("index_topk", 0);
        cand_src = (int) p.num("candidate_source_layer_id", -1), cand_topk = (int) p.num("candidate_topk_blocks", 0);
        cand_blk = (int) p.num("candidate_block_size", 0);
        orig_ctx = (int) p.num("rope_scaling.original_max_position_embeddings", 0);
        eos = (int) p.num("eos_token_id", 1);
        eps = (float) p.num("rms_norm_eps", 1e-20), hc_eps = (float) p.num("hc_eps", 1e-6);
        route_scale = (float) p.num("routed_scaling_factor", 1.0), lim = (float) p.num("swiglu_limit", 0.0);
        rope_theta = p.num("rope_theta", 1e4), comp_theta = p.num("compress_rope_theta", 1e4);
        factor = p.num("rope_scaling.factor", 1.0), beta_fast = p.num("rope_scaling.beta_fast", 32);
        beta_slow = p.num("rope_scaling.beta_slow", 1);
        ratios = p.list("compress_ratios"), kv_src = p.list("kv_source_layer_ids");
        ix_src = p.list("index_source_layer_ids"), eg_layers = p.list("engram_layer_ids");
        d_block = (int) p.num("dspark_block_size", 0), d_noise = (int) p.num("dspark_noise_token_id", 0);
        d_rank = (int) p.num("dspark_markov_rank", 0), d_n_exp = (int) p.num("dspark_n_routed_experts", 0);
        d_k = (int) p.num("dspark_num_experts_per_tok", 0), d_targets = p.list("dspark_target_layer_ids");
        // the DSpark blocks: an EXL3 pack's, or a GGUF pack's from DeepSeek's checkpoint (tools/ds4/drafter_pack.py)
        n_mtp = d_block > 0 && (int) ratios.size() > nl &&
                        (p.get("mtp.0.attn.wq_a.trellis") != nullptr || p.get("mtp.0.attn.wq_a.weight") != nullptr)
                    ? (int) ratios.size() - nl : 0;
    }
};

// DeepSeek's precompute_freqs_cis frequencies (YaRN when original_seq_len > 0), in F32 as torch computes them
std::vector<float> rope_freqs(int dim, int original, double base, double factor, double beta_fast, double beta_slow) {
    std::vector<float> f((size_t) dim / 2);
    for (int i = 0; i < dim / 2; ++i) {
        const float e = (float) (2 * i) / (float) dim;
        f[(size_t) i] = 1.0f / std::pow((float) base, e);
    }
    if (original > 0) {
        auto corr = [&](double rot) {
            return dim * std::log(original / (rot * 2 * 3.14159265358979323846)) / (2 * std::log(base));   // (M_PI: not MSVC's)
        };
        const double low = std::max(std::floor(corr(beta_fast)), 0.0);
        const double high = std::min(std::ceil(corr(beta_slow)), (double) dim - 1);
        const float den = (float) std::max(high - low, 1e-3);
        for (int i = 0; i < dim / 2; ++i) {
            float ramp = ((float) i - (float) low) / den;
            ramp = std::min(1.0f, std::max(0.0f, ramp));
            const float smooth = 1.0f - ramp;
            f[(size_t) i] = f[(size_t) i] / (float) factor * (1.0f - smooth) + f[(size_t) i] * smooth;
        }
    }
    return f;
}

// ------------------------------------------------------------------ devices, layers
struct Dev {
    int id = 0;
    cudaStream_t st = nullptr;
    cudaEvent_t ev = nullptr, ev2 = nullptr;   // (ev2: the lookahead's guesses are on the host)
    int* ridx_a = nullptr;                      // (Metal) the next layers' guessed picks (2 x kPicks)
    float* rw_a = nullptr;
    size_t free0 = 0, total = 0, weights = 0, experts = 0, caches = 0, scratch = 0;
    std::vector<void*> allocs;
    // a window's scratch, rows of up to kWin tokens (the routed experts' rows: kWin * kPicks)
    float *h = nullptr, *h2 = nullptr, *x = nullptr, *qr = nullptr, *q = nullptr, *kvr = nullptr, *ckr = nullptr,
          *cgr = nullptr, *latent = nullptr, *ikr = nullptr, *iq = nullptr, *iw = nullptr, *scores = nullptr,
          *bm = nullptr, *o = nullptr, *oa = nullptr, *ao = nullptr, *mix = nullptr, *a_pre = nullptr,
          *a_post = nullptr, *a_comb = nullptr, *f_pre = nullptr, *f_post = nullptr, *f_comb = nullptr, *pre = nullptr,
          *glog = nullptr, *rw = nullptr, *moe_out = nullptr, *sg = nullptr, *su = nullptr, *sh = nullptr,
          *sh_out = nullptr, *cpu_part = nullptr, *eg_x = nullptr, *eg_kv = nullptr, *logits = nullptr,
          *freqs_win = nullptr, *freqs_comp = nullptr, *xe = nullptr, *xg = nullptr, *xu = nullptr, *xhh = nullptr,
          *ed = nullptr;
    float* glog_a = nullptr;   // lookahead: the next two layers' gate logits on this layer's FFN input (rows 1024 apart)
    int *ridx = nullptr, *topk = nullptr, *cand = nullptr, *tok = nullptr;
    unsigned long long* plan_ptr = nullptr;           // the decode path's VRAM experts (0: on the CPU lane)
    uint8_t* eg_rows = nullptr;
    uint16_t *emb_rows = nullptr, *ext = nullptr;   // the window's embedding rows; the window attention's rows
    AttnScratch attn_scr;   // a few queries' selected compressed rows decoded (sparse_attn)
    // the hand-off to the next device through pinned host memory (these cards copy peer to peer only through the
    // driver's staging, ~1.7 ms a copy): h | pre | topk | cand
    uint8_t* bounce = nullptr;
    cudaEvent_t ev_out = nullptr;
    float* tap = nullptr;   // a window's stream means before a DSpark target layer
    // threads / tensor parallelism: its group and its index in it, the group's all-reduces it took part in, xfer_put's
    // block counter, the hand-off to the next group (host-mapped: h | pre | topk | cand, then its flag), the window
    // MoE's grouping scratch
    int grp = 0, mi = 0;
    unsigned ar_seq = 0;
    unsigned* xcnt = nullptr;
    unsigned* hccnt = nullptr;   // [kWin] hc_mix_split's per-token block counters (zeroed at allocation, left zero)
    // the head group's members (the last layer's group): each its columns (split128 of V) of the head and the norms;
    // logits_part: a member's share of the logits (the lead's: where the others' land before they join its rows)
    x3::Mat head;
    float *norm_w = nullptr, *d_norm = nullptr, *logits_part = nullptr;
    unsigned head_seq = 0;
    // the drafter's markov bias over the head group (markov_share): this member's rows [mk_v0, mk_v0 + mk_vn) of the
    // markov head, the previous token's E row, the drafted tokens (the lead's: d_ints)
    uint16_t *mk_share = nullptr, *mk_erow = nullptr;
    int* mk_ints = nullptr;
    int mk_v0 = 0, mk_vn = 0;
    uint8_t* hand = nullptr;
    unsigned* hand_flag = nullptr;
    unsigned long long* grp_ptr = nullptr;
    int *grp_start = nullptr, *grp_ent = nullptr, *n_groups = nullptr;
};

// the chunked prompt path's buffers on a device: a chunk of C tokens through a layer at once (the attention in slabs of
// A queries, the indexer in slabs of AI), its GEMMs through FP16 reconstructions of the weights and cuBLAS
// DS4_PRED_DUMP: a fixed +-1 random projection of the router input (D -> 256), one block a token
__global__ void pred_proj_k(const float* x, int D, float* out) {
    const int t = blockIdx.x, j = threadIdx.x;   // 256 threads
    const float* xr = x + (size_t) t * D;
    float a = 0.0f;
    for (int d = 0; d < D; ++d) {
        unsigned h = (unsigned) d * 2654435761u ^ ((unsigned) j + 1u) * 40503u;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
        a += (h & 1u) ? xr[d] : -xr[d];
    }
    out[(size_t) t * 256 + j] = a * rsqrtf((float) D);
}

struct Chunk {
    int C = 0, A = 0, AI = 0;
    float *h = nullptr, *x = nullptr, *acc = nullptr, *qr = nullptr, *kvr = nullptr, *ckr = nullptr, *cgr = nullptr,
          *latent = nullptr, *ikr = nullptr, *q = nullptr, *o = nullptr, *oa = nullptr, *iq = nullptr, *iw = nullptr,
          *scores = nullptr, *bm = nullptr, *mix = nullptr, *a_pre = nullptr, *a_post = nullptr, *a_comb = nullptr,
          *f_pre = nullptr, *f_post = nullptr, *f_comb = nullptr, *pre = nullptr, *glog = nullptr, *rw = nullptr,
          *sg = nullptr, *su = nullptr, *y = nullptr, *eg_x = nullptr, *eg_kv = nullptr;
    uint16_t *x16 = nullptr, *qr16 = nullptr, *o16 = nullptr, *oa16 = nullptr, *lat16 = nullptr, *sh16 = nullptr,
             *h16 = nullptr, *eg_x16 = nullptr, *ext = nullptr, *w16 = nullptr, *emb = nullptr;
    size_t w16_elems = 0;
    int *ridx = nullptr, *topk = nullptr, *cand = nullptr, *bounds = nullptr, *row_tok = nullptr, *pos = nullptr;
    uint8_t* eg_rows = nullptr;
    int* bounds_pin = nullptr;   // pinned: the sort's bounds, read back
    // the CPU lane's experts streamed in: ring_n VRAM slots of ring_stride bytes in kParts parts, a batch a part,
    // kParts batches in flight on the copy stream (filled / freed: each part's last copy, its last use)
    static constexpr int kParts = 4;
    uint8_t* ring = nullptr;
    size_t ring_stride = 0;
    int ring_n = 0;
    cudaStream_t cs = nullptr;
    cudaEvent_t filled[kParts] = {}, freed[kParts] = {};
    // in place (file_experts): each streamed expert's file region read into pinned staging (ring_n slots of
    // stage_stride bytes, a part's slots free again once its `filled` copies are done) by the readers, in parallel,
    // and DMA'd from there - a copy from the files' pages goes through the driver's staging, one thread at ~1.3 GB/s
    uint8_t* stage = nullptr;
    size_t stage_stride = 0;
    std::unique_ptr<Pool> readers;
    cublasHandle_t blas = nullptr;
    // the busy experts' path (run_set, DS4_FAT_ROWS): one expert's weights as FP16 (gate and up, then down), the rows
    // gathered for its GEMM, the GEMM's gate | up output - fat_r rows a piece; allocated at the first use
    uint16_t *fat_w = nullptr, *fat_x = nullptr;
    float* fat_gu = nullptr;
    int fat_r = 0;
    bool fat_off = false;
    // host-mapped: a chunk's hand-off to the next group (h | pre | topk | cand), two in flight (slot = chunk % 2), each
    // with its flag, and the next group's members' acknowledgements [slot][member] (32 apart)
    uint8_t* hand[2] = {nullptr, nullptr};
    unsigned* hand_flag[2] = {nullptr, nullptr};
    unsigned* hand_ack = nullptr;
    unsigned arb_seq = 0;        // the group's chunk all-reduces
    // pinned, two chunks in flight: the embedding rows (the first layer's device), the Engram rows (an Engram layer's)
    uint16_t* emb_host[2] = {nullptr, nullptr};
    uint8_t* eg_host[2] = {nullptr, nullptr};
    cudaEvent_t emb_ev[2] = {nullptr, nullptr}, eg_ev[2] = {nullptr, nullptr};
    double stream_bytes = 0;     // the experts streamed in, in all
    // the flow (a GGUF's in-place experts, every layer of the device on it alone; DS4_FLOW=0: the batches above): a
    // (chunk, layer)'s streamed experts are those it routes to - its likely ones (the profile's chance of a pick by the
    // chunk's tokens) queued once the layer before it routed, read while that one computes, the rest of its picks once
    // it routed itself; never an expert no token of the chunk picked unless it was a likely one.  Each item read by io
    // (unbuffered: the disk straight into its staging slot, every part of every item in flight at once - none waits
    // for a batch) or taken from its RAM-tier slot as it lies, then DMA'd into the ring in order, item i into slot
    // i % ring_n; the compute takes the items as they land, in runs of what has.
    struct Item {
        int il = 0, k = 0, e = 0, v = 0, stage = -1;   // layer, chunk, expert, virtual slot, staging slot (-1: RAM)
        int mi = 0;                                    // the device's member index in the layer's group
        bool direct = false;                           // (Metal) read straight into its ring slot: no staging, no DMA
        const uint8_t* src[3] = {nullptr, nullptr, nullptr};   // where its gate / up / down (its parts) lie once read
        const uint8_t* ram = nullptr;                  // its RAM-tier blob, copied into staging (not pinned there)
        uint8_t* keep = nullptr;                       // (layer-major) its layer-cache slot: what is read kept there
        int64_t after = -1;                            // (layer-major) the items up to this one read first
        std::atomic<int> left{0};                      // parts still being read
        std::atomic<bool> bad{false};
    };
    bool flow = false;
    bool direct = false;                     // (Metal, unified memory) the flow reads the files straight into the ring
    int dev = -1;                            // the device (index into devs) this chunk state belongs to
    std::unique_ptr<IoQueue> io;
    std::deque<Item> items;                  // items[0] is item `base`; the indices below are absolute
    size_t base = 0, n_read = 0, n_dma = 0, n_comp = 0;
    uint64_t n_file = 0;                     // file items so far (staging slot = ordinal % S)
    int S = 0;                               // staging slots, stage_stride apart
    std::vector<int64_t> stage_item;         // staging slot -> the item last read into it (-1: none)
    std::vector<cudaEvent_t> stage_ev, ring_fill, ring_free;   // a staging slot's DMA out; a ring slot's DMA in, its use
    int spec_k = -1, spec_il = -1, n_spec = 0;   // the likely items queued last: (chunk, layer), how many, from where
    size_t spec_begin = 0;
    std::vector<int> fl_layers, fl_T;        // the device's trunk layers in order; the run's chunk sizes
    int fl_yoco = -1, fl_cut_k = 0, fl_cut_r = 0;   // Run::yoco / cut_k / cut_r: what a layer's chunk streams for
    int bc_n = 0, bc_w = 0;                  // a batch's compacted bounds: bc_n regions of bc_w, by batch count
    int *bc_pin = nullptr, *bc_dev = nullptr;
    uint64_t n_batch = 0;
    double wait_ms = 0;                      // the compute waiting for a read
    // (DS4_PROF: where copy_ahead's time goes - the host staging copy, the DMA issues, the waits for a slot)
    double dbg_stage_ms = 0, dbg_dma_ms = 0, dbg_evt_ms = 0;
    uint64_t it_file = 0, it_ram = 0, it_spec = 0, it_spec_unused = 0;
    uint64_t io_d0 = 0, io_b0 = 0;           // io's unbuffered / buffered bytes when the run started
    double io_busy0 = 0;                     // io's busy time when the run started (DS4_PROF)
    // layer-major (a run of several chunks, every layer on this device): the layers one after another, each through
    // all the chunks; a chunk's state between its layers (h | pre | topk | cand, chunk k at k chunk_hand_bytes()) in
    // host memory, and the layer cache - RAM slots where the experts a layer's chunk reads from the files are kept for
    // its later chunks, so each is read once a run
    bool lm = false, lm_pinned = false;
    uint8_t* lm_state = nullptr;
    size_t lm_state_bytes = 0;
    uint8_t* lc = nullptr;                   // lc_bytes of slots, a layer's experts' blobs side by side
    size_t lc_bytes = 0;
    int lc_il = -1, lc_used = 0, lc_n = 0;   // the layer the cache holds, its slots taken, its slots in all
    size_t lc_first = 0, lc_chk = 0;         // the layer's first item; the items before lc_chk all read
    std::vector<int> lc_of;                  // expert -> its slot (-1: none)
    std::vector<int64_t> lc_writer;          // slot -> the item reading its expert into it
    uint64_t it_lc = 0;                      // items taken from the layer cache
};

struct Layer {
    int il = 0, dev = 0, ratio = 0;
    // its group (tensor parallel: nm members, each a copy of the device state in mates, member mi's share of the
    // heads, of wo_a's groups, of wo_b's rows and of the FFNs: gate / up columns, down rows; its part of an expert
    // laid out as hlay, hblob bytes - the host state, the CPU lane and the full blob layout stay with member 0)
    int grp = 0, nm = 1, mi = 0;
    std::vector<Layer> mates;
    x3::ExpertLayout hlay;
    size_t hblob = 0;
    int ff0 = 0, ffn = 0;     // this member's FFN columns [ff0, ff0 + ffn) (split128 of n_ff; the whole one alone)
    // this member's heads [ah0, ah0 + ahn) and wo_a groups [og0, og0 + ogn) - all of them alone; in a group of more
    // members than the heads split among evenly (head_members: 9, 7 or 6 over 64 heads), the first ones none: their
    // attention adds nothing to the sum
    int ah0 = 0, ahn = 0, og0 = 0, ogn = 0;
    // the chunked prompt path: each expert's virtual slot (its VRAM slot, or n_slots + its place among the CPU lane's
    // experts, which a chunk streams in that order) - pinned on member 0, a device copy on each member
    int* vs_pin = nullptr;
    int* vs_dev = nullptr;
    std::vector<int> host_list;
    std::string name;         // "layers.N." or, for the DSpark drafter's blocks, "mtp.N."
    int n_exp = 0, k_exp = 0; // routed experts, and how many a token takes (the drafter's: 128, 3)
    bool mtp = false;         // a DSpark block (index stage = il - n_layers)
    bool kv_src = false, ix_src = false, owns_k = false, cand_src = false, uses_cand = false;
    int src = -1;      // the kv source whose compressed KV this layer reads (index into Impl::srcs)
    int ik_src = -1;   // the kv source whose index keys its indexer reads
    int eg = -1;       // engram layer index, or -1
    x3::Mat wq_a, wq_b, wkv, wo_a[16], wo_b, s1, s2, s3, c_wkv, c_wgate, i_wq_b, i_wk, eg_wkv;
    float *q_norm = nullptr, *kv_norm = nullptr, *attn_norm = nullptr, *ffn_norm = nullptr, *c_norm = nullptr,
          *ik_norm = nullptr, *sink = nullptr, *hca_fn = nullptr, *hca_scale = nullptr, *hca_base = nullptr,
          *hcf_fn = nullptr, *hcf_scale = nullptr, *hcf_base = nullptr, *gate_bias = nullptr, *eg_w = nullptr,
          *gate_bias_vl = nullptr;   // (the image tokens' routing bias, with the vision tower)
    uint16_t* i_wproj = nullptr;
    float* gate_w = nullptr;   // the router's weights, F32 (DeepSeek's inference computes the gate in F32)
    uint8_t* win = nullptr;                     // [win] rows of kKvFp8 (kv_row_bytes)
    float *c_kv = nullptr, *c_score = nullptr;  // the compressor's partial group (ratio > 1)
    // a run of chunks' mid-run checkpoint (feed_mark): the ring and the partial group as they were before a row
    uint8_t* snap_win = nullptr;
    float *snap_kv = nullptr, *snap_sc = nullptr;
    // routed experts
    x3::ExpertLayout lay;
    size_t blob = 0;                            // bytes of one expert's blob
    int n_vram = 0;
    bool cpu_only = false;   // a role's type has no device kernel: every expert of this layer is the CPU lane's
    uint8_t* vram_base = nullptr;
    std::vector<int> slot;                      // expert -> VRAM slot or -1
    std::vector<int> ram;                       // expert -> RAM-tier slot or -1 (a GGUF's in-place experts; empty: none)
    std::vector<uint8_t*> host;                 // expert -> host (non-null: on the CPU lane; its halves in hh)
    // the CPU lane keeps an expert as NH column parts, part X on NUMA node X: gate / up columns [X n_ff / NH, ..),
    // down columns [X dim / NH, ..), each a standalone EXL3 matrix (its own trellis, the full suh, its part of svh)
    struct Half {
        size_t trellis[3], suh[3], svh[3], bytes;
        int k[3], n[3];
    } hl;
    std::vector<std::array<uint8_t*, 4>> hh;     // expert -> its parts on the nodes
    // the expert cache: one VRAM slot and one set of host parts kept free, so an expert can move in while the one it
    // replaces moves out; heat is how much each expert was used lately (the profile's share to start with)
    int n_slots = 0, spare_vram = -1;
    std::array<uint8_t*, 4> spare_host = {nullptr, nullptr, nullptr, nullptr};
    std::vector<float> heat;
    std::vector<int> cpu_hits;                  // CPU-lane experts used since the cache last looked
    bool swap_pending = false;
    unsigned long long* tab_pin = nullptr;      // pinned: the table entries a swap writes
    // a speculative window's backups (trunk layers): the ring rows it replaced, and a compressor's state before it and
    // the window's raw rows (kv, then score) to replay the accepted ones
    uint8_t* ring_bak = nullptr;
    float *c_snap = nullptr, *c_rows = nullptr;
    unsigned long long* vram_tab = nullptr;     // device [n_exp]
    std::vector<uint8_t> zc;                    // (the RAM store) expert -> 1: its store slot is GPU-visible, the device computes it over PCIe
    unsigned long long* zc_tab = nullptr;       // device [n_exp]: those slots' device addresses (0: none) - the decode token's zc_stage reads it
    std::vector<std::array<x3::Mat, 3>> vm;     // expert -> its gate, up, down as EXL3 views of its VRAM blob
    std::vector<uint64_t> region_off;           // expert -> file region (offset, bytes, file)
    std::vector<uint64_t> region_bytes;
    std::vector<int> region_file;
    size_t piece_src[3][3];                     // part (gate, up, down) x (trellis, suh, svh): offset in the region
    size_t piece_bytes[3][3];
    size_t dense_bytes = 0;
    // a GGUF's MXFP4 experts (lay.fmt == gg::kFmtMXFP4): matrix r (gate, up, down) of expert e lies at mx_off[r] +
    // e mx_stride[r] in file mx_file[r] - loaded from there, and read in place (the file's mapping) by the CPU lane
    int mx_file[3] = {-1, -1, -1};
    uint64_t mx_off[3] = {0, 0, 0}, mx_stride[3] = {0, 0, 0};
};

struct Src {
    int layer = 0, ratio = 1;
    int64_t cap = 0;
    // per device (nullptr where no layer reads it): rows of kKvFp4 (the compressed KV), kKvFp4p (the index keys)
    std::vector<uint8_t*> ckv, ik;
    uint8_t *h_ckv = nullptr, *h_ik = nullptr;   // pinned: a window's new rows, on their way to the other devices
    cudaEvent_t ev_ckv = nullptr, ev_ik = nullptr;
    // threads: whether devices outside the source's group read it, and the host-mapped hand-off of a window's new rows
    // (ckv rows, then ik rows at kWin * hd) and its flag
    bool remote = false;
    uint8_t* x_stage = nullptr;
    unsigned* x_flag = nullptr;
    // pipelined chunks: a second staging slot (slot 1; slot 0 is x_stage), its flag, and the readers' acknowledgements
    // [slot][device] (32 apart); readers: the devices outside the source's group that read it
    uint8_t* x_stage1 = nullptr;
    unsigned* x_flag1 = nullptr;
    unsigned* x_ack = nullptr;
    std::vector<int> readers;
    uint8_t* stage(int slot) const { return slot ? x_stage1 : x_stage; }
    unsigned* flag(int slot) const { return slot ? x_flag1 : x_flag; }
};

// a host thread that enqueues one device's work (threads mode)
struct Worker {
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    std::function<void()> job;
    bool busy = false, quit = false;
    void start() {
        th = std::thread([this] {
            for (;;) {
                std::function<void()> j;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return quit || job != nullptr; });
                    if (quit) return;
                    j = std::move(job);
                    job = nullptr;
                }
                j();
                {
                    std::lock_guard<std::mutex> lk(mu);
                    busy = false;
                }
                cv.notify_all();
            }
        });
    }
    void submit(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(mu);
            job = std::move(f);
            busy = true;
        }
        cv.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !busy; });
    }
    bool idle() {
        std::lock_guard<std::mutex> lk(mu);
        return !busy;
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv.notify_all();
        if (th.joinable()) th.join();
    }
};

struct Group {
    std::vector<int> mem;   // device indices, member order (member 0 the lead)
    ArGroup ar, arb;        // a window's all-reduces, a chunk's
    // the head group: the members' logits shares on their way to the lead, two slots (by seq), their flags and the
    // lead's acknowledgements (host-mapped)
    float* lg_stage = nullptr;
    unsigned *lg_flag = nullptr, *lg_ack = nullptr;
};

constexpr int kMaxL = 64;   // layers (the trunk's and the drafter's) a window plan covers

// what a window's layers need from the host, decided before they run (threads enqueue the layers concurrently): the
// mailbox seq of each layer (async: the device publishes its routing), the indexer's selection sizes and the candidate
// blocks as each layer sees them, the window's hand-off seq
struct WinPlan {
    bool par = false, async = false;
    unsigned win = 0;
    int k = 0;              // (a run of chunks) the chunk's index: its hand-off slot k % 2
    int snap_t = -1;        // (a run of chunks) the chunk's row whose state before it the layers keep (-1: none)
    int eg_slot = -1;       // (a run of chunks, layer-major) the Engram rows' slot (-1: k % 2)
    unsigned vl = 0;        // (a window) bit t: token t is inside a picture's span (routed with the image bias)
    unsigned seq[kMaxL] = {};
    int topk[kMaxL][kWin] = {}, cand[kMaxL][kWin] = {};
};

}  // namespace

// ------------------------------------------------------------------ the model
struct Model::Impl {
    Options o;
    Pack pk;
    Geo g;
    std::vector<Dev> devs;
    std::vector<Layer> L;
    std::vector<Src> srcs;
    std::vector<Group> groups;
    std::vector<std::unique_ptr<Worker>> workers;   // threads mode: one per device
    bool par = false;                 // threads mode (the windows: decode)
    mutable int all_gfx12 = -1;       // rdna4(): -1 not asked yet
    // Every GPU the engine uses is a gfx1200 / gfx1201 (RDNA4: R9700, RX 9000) in a HIP build.  The DeepSeek engine's share-based defaults (the small
    // chunk for multi-card tp runs, the whole-chunk slab, the small staging ring, the pinned copy at share >= 0.5) and the drafter's cumulative cut are
    // the 4 x R9700 measurements, so only RDNA4 takes them until another AMD part is tested (Strix Halo next).
    bool rdna4() const {
        if (all_gfx12 < 0) {
            bool yes = false;
#if defined(STRATA_USE_HIP)
            yes = !devs.empty();
            for (const auto& d : devs) {
                cudaDeviceProp prop{};
                yes = yes && cudaGetDeviceProperties(&prop, d.id) == cudaSuccess &&
                      (std::strncmp(prop.gcnArchName, "gfx1200", 7) == 0 || std::strncmp(prop.gcnArchName, "gfx1201", 7) == 0);
            }
#endif
            all_gfx12 = yes ? 1 : 0;
        }
        return all_gfx12 == 1;
    }
    bool prompt_par = false;          // the prompt path's chunks (a thread a GPU), as --tp / --threads asked
    unsigned win_seq = 0;             // the windows the threads ran (the hand-offs' seq)
    // the chunked prompt path (threads mode): chunk tokens a pass, runs shorter than chunk_min in windows
    int chunk = 0, chunk_min = 512;
    static constexpr bool kChunkedPromptDefault = true;   // the layer-major prompt is the default (gate: KL 0.1685 vs 0.1686, 8K prompt 6x, decode within 2% - ds4 notes-dsv41-r97)
    static constexpr int kFlowChunkMin = 2;   // chunk_min with every device's layers on the flow (see load)
    // --prefill-chunk at most (route() runs in kVlRows-row slabs).  The default stays kVlRows: a 16384-token chunk
    // reads a 16K prompt's experts once instead of twice (V100: 461.8 -> 321.3 s, 736 -> 506 GB read) but its buffers
    // take ~1.9 GB of VRAM experts (745 -> 644 on a 32 GB V100) from every decoded token
    static constexpr int kMaxChunk = 4 * kVlRows;
    std::vector<std::unique_ptr<Chunk>> chunks;   // per device
    size_t eg_cap = kWin;             // tokens h_eg holds
    int tap_cap = kWin;               // rows h_taps holds (a chunk seeds the drafter with its last 128)
    static constexpr int kYocoMin = 1024;   // a long prompt's skip (forward_chunks): the fewest rows worth skipping
    bool logits_all = false;          // the last feed was a chunk with logits for every row (chunk_logits)
    std::vector<float> chunk_logits;
    std::function<void(int64_t)> progress;   // a run of chunks: the tokens done so far (the main thread, while it waits)
    // a Stop while a run of chunks goes through the layers (Model::set_cancel, asked while forward_chunks waits): one
    // GPU stops after the layer (or the layer's chunk) it is on (halt); several stop at a chunk every one of them
    // reaches - none starts chunk halt_k or later, halt_k one past the latest any started - so no device waits on a
    // hand-off that never comes (Project Maya v1.1.20)
    std::function<bool()> cancel;
    std::atomic<bool> halt{false};
    std::mutex halt_mu;
    int halt_k = INT_MAX, halt_seen = -1;
    bool halted = false;                      // the last run ended early on a Stop
    static constexpr const char* kHalted = "stopped";
    bool chunk_go(int k) {   // chunk k may start on this device (several GPUs)
        std::lock_guard<std::mutex> lk(halt_mu);
        if (k >= halt_k) return false;
        halt_seen = std::max(halt_seen, k);
        return true;
    }
    bool halt_now() const { return devs.size() == 1 && halt.load(std::memory_order_acquire); }
    // a run's early end on device di: the expert reads still in flight land first (they count down items the next run
    // clears), then its streams are idle
    bool halt_end(int di, std::string& err) {
        Chunk& c = *chunks[(size_t) di];
        cudaSetDevice(devs[(size_t) di].id);
        if (c.flow)
            for (size_t i = c.base; i < c.n_read; ++i)
                while (fit(c, i).left.load(std::memory_order_acquire) != 0) c.io->wait_any(500);
        if (c.cs != nullptr) cudaStreamSynchronize(c.cs);
        cudaStreamSynchronize(devs[(size_t) di].st);
        err = kHalted;
        return false;
    }
    std::vector<cudaEvent_t> chunk_done;     // on the last layer's device: chunk k through every layer
    std::unique_ptr<std::atomic<int>[]> chunk_rec;   // chunk k's event recorded
    // a prompt the flow could take: the flow or windows of 16, whichever this PC runs it faster (pick_path) - this
    // PC's speeds measured at load (calibrate_paths), each path's predictions scaled by how long it really took
    struct PathModel {
        bool on = false;
        double dma = 0, copy = 0, disk = 0, cpu = 0, cpu_cold = 0;   // bytes/s (disk, cpu_cold 0: not measured)
        // the flow's taken / predicted so far [0] by the prompt's size class (2-31, 32-127, 128-511, 512 tokens and
        // more: a short one's fixed costs weigh more), and how many prompts set it ([1] unused)
        double k[2][4] = {{1, 1, 1, 1}, {1, 1, 1, 1}};
        int64_t kn[2][4] = {};
        double win_s = 0;                                              // windows: seconds a window took, so far
        int64_t win_n = 0;
        int last_cls = 0, last_nw = 0;                                 // the last pick's size class, its windows
        double m = 1;                                                  // the picks' spread, against the profile's
        int64_t n_m = 0;
        int last = -1;                                                 // the last pick: 1 the flow, 0 windows
        double last_pred = 0;                                          // its prediction (s, before its k)
    } pm;
    static constexpr int kPickMax = 2048;   // (from here on the flow: windows' work grows with every 16 tokens)
    static constexpr double kPickRes = 0.9;  // (this share of the CPU-lane experts in RAM: windows if predicted faster)
    std::atomic<int> lm_layers{0};           // (layer-major) the layers every chunk went through
    WinPlan wplan;
    int head_dev = 0;
    x3::Mat head;
    float* norm_w = nullptr;          // on the head device
    // the drafter: main_proj / main_norm (on its first block's device), its final norm, the markov head and the
    // confidence head (on the head device: the drafts' logits come from the trunk's head)
    x3::Mat d_main_proj;
    float *d_main_norm = nullptr, *d_norm = nullptr, *d_conf = nullptr;
    uint16_t *d_mk_embed = nullptr, *d_mk_head = nullptr;   // BF16 [vocab][rank], F16 [vocab][rank]
    uint8_t* d_mk_head_q8 = nullptr;   // the markov head as Q8_0 blocks (the same bias, half the bytes a step)
    // a head group of two or more: the markov steps over its members (each its rows of the bias and an argmax of its
    // share, the global one through host-mapped slots [step][member] {value, index, seq}) in place of the lead's whole
    // rows - E host-mapped (each member reads the picked token's row), a draft's seq in mk_seq
    bool mk_split = false;
    uint16_t* h_mk_embed = nullptr;
    unsigned* mk_slots = nullptr;
    unsigned mk_seq = 0;
    int d_dev = 0;                    // the drafter's first block's device
    bool spec_win = false;            // the window being run is a draft's verification (layer() keeps backups)
    float* h_taps = nullptr;          // pinned [kWin][targets][dim]: the last window's taps
    cudaEvent_t ev_seed = nullptr;    // the drafter's seeding read h_taps: the next window's taps wait for it
    float* d_confv = nullptr;         // on the head device
    int* d_ints = nullptr;            // on the head device: drafts in, picks out
    float conf_threshold = std::getenv("DS4_DSPARK_CONF") ? (float) std::atof(std::getenv("DS4_DSPARK_CONF")) : 0.5f;
    // a sampled round's drafts: the distribution q each one was drawn from (its kept list, on the head device) for
    // verify()'s rule - a point at the draft for the drafter's argmax; and the chain they were drawn with (verify()
    // checks it is its own)
    strata::dsv41k::SpecList* d_specq = nullptr;
    std::vector<strata::dsv41k::SpecList> h_specq = std::vector<strata::dsv41k::SpecList>(kWin);
    bool drafts_q = false;
    float dq_temp = 0.0f, dq_top_p = 1.0f, dq_min_p = 0.0f;
    int dq_top_k = 0;
    uint64_t dq_seed = 0, dq_counter = 0;
    // (tests) DS4_DSPARK_SAMPLE=1: a sampled request's drafts drawn from the drafter's distribution through the
    // request's chain, as DeepSeek's reference does (else its argmax: the same acceptance on Uranus, 31-32% of 5
    // drafts a round either way); DS4_DSPARK_QTEMP=f: those draws at f x the request's temperature (any q keeps the
    // rule exact); DS4_DSPARK_MAX=n: at most n drafts a round; DS4_DSPARK_DUMP=file: the drafter's inputs and outputs
    // (tools/ds4/dspark_ref.py checks them against DeepSeek's checkpoint)
    bool sample_drafts = std::getenv("DS4_DSPARK_SAMPLE") != nullptr && std::atoi(std::getenv("DS4_DSPARK_SAMPLE")) != 0;
    float q_temp_scale = std::getenv("DS4_DSPARK_QTEMP") ? (float) std::atof(std::getenv("DS4_DSPARK_QTEMP")) : 1.0f;
    int max_drafts = std::getenv("DS4_DSPARK_MAX") ? std::atoi(std::getenv("DS4_DSPARK_MAX")) : 0;
    std::FILE* dump_d = std::getenv("DS4_DSPARK_DUMP") ? std::fopen(std::getenv("DS4_DSPARK_DUMP"), "wb") : nullptr;
    float* h_dump = nullptr;          // (the dump's pinned copies: each block's stream, each block's ring)
    uint16_t* h_dump_ring = nullptr;

    uint16_t* embed_pinned = nullptr;
    std::vector<float> freqs_win, freqs_comp;
    std::vector<std::unique_ptr<Pool>> pools;   // one per NUMA node, its workers on that node's cores
    std::vector<std::vector<int>> cores;        // each node's physical cores
    int NH = 1;                                 // the CPU lane's parts of an expert (the NUMA nodes)
    // the CPU lane's experts read in place from the model's files (one part, the files' layout; the page cache keeps
    // what fits) instead of copied into RAM, and how much of them the load reads ahead (the most-routed first)
    bool file_experts = false;
    bool views = true;   // false: the shards' views are dropped after the load (DS4_NO_VIEW=1)
    uint64_t warm_bytes = 0;
    static constexpr uint64_t kRamMargin = 8ull << 30;   // RAM left to the rest (the system, the page cache's churn)
    uint8_t* file_part(const Layer& l, int e) const {
        return pk.maps[(size_t) l.region_file[(size_t) e]].first + l.region_off[(size_t) e];
    }
    // host side of the CPU lane, a window's rows
    float *h_x = nullptr, *h_w = nullptr, *h_part = nullptr;
    int* h_idx = nullptr;
    int* h_idx_a = nullptr;   // (Metal) the lookahead's guessed picks, 2 x kPicks
    uint8_t* h_eg = nullptr;          // the window's Engram rows: [engram layer][token] of [cols][hd] then the scales
    // the Engram rows' reader (its own threads and file handles), the rows already fetched, the window's batches (one
    // an Engram layer: asked for when the window's tokens are known, collected right before that layer)
    EgReader eg_io;
    EgRowCache eg_cache;
    std::mutex eg_cache_mu;
    std::vector<std::unique_ptr<EgBatch>> eg_win;
    // Engram layer li's embed rows, as the pack stores them: 0 MXFP8 ([scale, 32 codes] blocks, 33 bytes a 32), 1
    // ggml Q6_K (210 bytes a 256), 2 ggml IQ4_NL (18 bytes a 32: an fp16 d and 16 nibble bytes), 3 an old pack's FP8
    // codes with their E8M0 scales in a tensor of their own (see eg_issue)
    std::vector<uint8_t> eg_fmt;
    std::atomic<int64_t> eg_rows_n{0}, eg_hits_n{0}, eg_reads_n{0}, eg_wait_ns{0}, eg_io_ns{0};
    uint16_t* h_emb = nullptr;        // the window's embedding rows
    std::vector<float> c_xh, c_xsb, c_acc, c_xhd, c_xsd, c_accd, c_yd;
    // the state
    int64_t pos = 0;
    std::vector<int32_t> toks, ctoks;
    struct Check {                    // the state at a position: every window ring, the compressors' partial groups
        int64_t pos = -1;
        std::vector<std::vector<uint8_t>> win;
        std::vector<std::vector<float>> comp;
    };
    static constexpr int kChecks = 4;
    Check ck[kChecks];                // the end of the last prompt, the position before it, the anchor (the position
                                      // before the end of the first prompt of the run of prompts that each continued
                                      // the last in place: a chat template that drops earlier turns' reasoning at a new
                                      // user message re-renders the history from about there) and the prefix (where the
                                      // server said the prompt's shared start - system prompt and tools - ends: another
                                      // agent behind the same start goes back to it)
    bool slot_held = false;           // slot_save took this state and nothing has changed it since
    std::vector<uint32_t> prof;       // [nl][n_exp] routed-expert hits
    Stats st;
    int cur_topk_n[kWin] = {}, cur_cand_n[kWin] = {};   // what the latest indexer / candidate source left, per query
                                                        // (plan_window's: the windows' plans are made in order)
    int n_logits = 0;                 // the last forward's logit rows
    // the decode path (windows of one): the device publishes each layer's routing into reqs, the service thread
    // answers in resps with the CPU lane's part, and the device waits for the answer only before the expert sum
    Req* reqs = nullptr;
    Resp* resps = nullptr;
    unsigned seq_issued = 0;
    std::atomic<int> layer_of[kRing];
    std::thread service;
    std::atomic<bool> quit{false};
    std::thread store_bg;   // the RAM store's fill beyond the first 48 GB, read after the model is ready (preload_store)
#if defined(STRATA_METAL)
    // (the async decode has the GPU wait on flags the CPU lane sets while it runs: the Metal queue cannot be waited on in the
    // middle of a command buffer, so a decode step takes the host-routed path, as the prompt windows do)
    bool sync_decode = std::getenv("DS4_ASYNC") == nullptr;
#else
    bool sync_decode = std::getenv("DS4_SYNC") != nullptr;
#endif
    // a window of one through moe_gate_up / moe_down (DS4_MOE_TOKEN=0: win_experts, whose down rows run
    // before the wait for the CPU lane - the same bits, no faster)
    bool moe_token_path = std::getenv("DS4_MOE_TOKEN") == nullptr || std::atoi(std::getenv("DS4_MOE_TOKEN")) != 0;
    int async_max_t = std::getenv("DS4_ASYNC_T") ? std::atoi(std::getenv("DS4_ASYNC_T")) : 8;   // windows up to this
    // lookahead: while layer L runs, the device guesses the next layer's picks (its gate on L's FFN input, the same
    // choice) and publishes them with L's routing; the CPU lane asks for the guessed experts it will read from the files
    // at once (a thread of its own, os::prefetch), so the SSD reads them while L computes instead of on the next
    // layer's critical path.  Only the page cache sees it - the outputs are the same.  DS4_AHEAD=0: off, 2: also the
    // layer after (from L's input too); DS4_AHEAD_EXTRA: guesses past the next layer's top k (< 0: fewer - the guess's
    // picks by rank on Sun: 97 91 81 70 57 41 %, past the top 6 29 20 %: reads wasted on an SSD-bound PC).
    int ahead_mode = std::getenv("DS4_AHEAD") ? std::atoi(std::getenv("DS4_AHEAD")) : 1;
    int ahead_extra = std::getenv("DS4_AHEAD_EXTRA") ? std::atoi(std::getenv("DS4_AHEAD_EXTRA")) : 0;
    // (DS4_AHEAD_ORDER=0: the CPU lane computes a layer's experts in expert order, not the ones asked for early first)
    bool ahead_order = std::getenv("DS4_AHEAD_ORDER") == nullptr || std::atoi(std::getenv("DS4_AHEAD_ORDER")) != 0;
    std::vector<uint8_t> ahead_fl;     // [layer][1024]: 1 the top-k guess, 2 any guess, a layer before; 4 two layers before;
                                       // 8 asked for (prefetch)
    std::vector<uint8_t> ahead_rank;   // [layer][1024]: 1 + the expert's rank in the guess a layer before (0: not guessed)
    std::vector<unsigned> ahead_for;   // [layer]: the request (seq) its flags are for
    std::thread pf_th;
    std::mutex pf_mu;
    std::condition_variable pf_cv;
    std::vector<std::pair<const void*, size_t>> pf_q;
    // the expert cache: swaps prepared on threads of their own (DS4_CACHE_THREADS, 1 by default - 4 measured no faster
    // on one V100 - more for a PC to try: each its staging buffers and
    // copy streams), switched between windows
    struct Swap {
        int layer = 0, in = -1, out = -1;
        int ram = -1;   // (copy_mode, `in` held in RAM) the spare slot `out` is copied into from VRAM
    };
    static constexpr int kMaxSwaps = 8;
    // the swap rule: a CPU-lane expert in when its heat passes the coldest VRAM one's x margin + bonus; heat x decay
    // every 256 tokens; at most `inflight` swaps under way (DS4_CACHE_MARGIN / _BONUS / _DECAY / _INFLIGHT)
    static float env_f(const char* k, float d) { return std::getenv(k) ? (float) std::atof(std::getenv(k)) : d; }
    const float c_margin = env_f("DS4_CACHE_MARGIN", 1.25f), c_bonus = env_f("DS4_CACHE_BONUS", 1.0f),
                c_decay = env_f("DS4_CACHE_DECAY", 0.9f);
    const int c_inflight = std::max(1, (int) env_f("DS4_CACHE_INFLIGHT", (float) kMaxSwaps));
    // the RAM tier (a GGUF's in-place experts, as GLM's: the hottest in VRAM, the next hottest here, the coldest in the
    // files): RAM slots of one expert blob each, filled at load with the hottest experts VRAM does not hold, so the
    // CPU lane reads them from RAM; between windows (cache_tick) a hotter expert from the files takes the place of the
    // layer's coldest held one - read into a free slot by ram_th, switched while nothing reads them.  Opt-in
    // (STRATA_DS4_RAM_TIER=1): on a V100 with 64 GB of RAM it was slower than the page cache alone (1.4 against 2.2-2.4
    // tokens/s sampled) - the cache follows a conversation's experts at once, the tier's moves lag behind.
    struct RamSwap {
        int layer = 0, in = -1, out = -1, slot = -1;
    };
    std::vector<uint8_t*> ram_slots, ram_chunks;
    std::vector<size_t> ram_chunk_bytes;
    bool ram_pinned = false;   // the slots are pinned memory (the prompt path DMAs them from where they lie)
    size_t flow_part[3] = {0, 0, 0};   // the flow's staging room for each part: its largest over the trunk's layers
    std::vector<int> ram_free;
    std::deque<RamSwap> ram_queue;
    std::vector<RamSwap> ram_ready;
    std::condition_variable ram_cv;
    std::thread ram_th;
    int ram_inflight = 0, ram_held = 0;
    bool ram_tier = std::getenv("STRATA_DS4_RAM_TIER") != nullptr && std::atoi(std::getenv("STRATA_DS4_RAM_TIER")) != 0;
    // STRATA_DS4_DISK_SKIP_W (experimental opt-in, CHANGES THE ANSWERS; 0 = off): on the small windows (decode, a
    // draft's check - never the prompt path) the CPU lane leaves out a routed expert whose share of its token's route
    // weights is below this and whose pages are not in RAM now (the page cache or the RAM tier; os::in_ram - Linux every
    // page, Windows sampled pages of the working set), so a token waits on the SSD only for the experts that weigh; the
    // weight it carried is lost (no renormalisation).  On layers from STRATA_DS4_DISK_SKIP_FROM (0) on.  An expert left
    // out still counts as used (the expert cache, the profile).
    const float disk_skip_w = [] {
        const char* v = std::getenv("STRATA_DS4_DISK_SKIP_W");
        return v != nullptr ? std::max(0.0f, (float) std::atof(v)) : 0.0f;
    }();
    const int disk_skip_from = [] {
        const char* v = std::getenv("STRATA_DS4_DISK_SKIP_FROM");
        return v != nullptr ? std::max(0, std::atoi(v)) : 0;
    }();
    // the RAM store (--ssd-cache): the CPU lane's experts in slots a policy owns, read unbuffered (null: off)
    std::unique_ptr<ExpertStore> store;
    uint8_t* zc_stage_buf = nullptr;   // device: k staging slots for the store's experts of a decode token
    size_t zc_stride = 0;
    bool zc_copy = true;               // DS4_STORE_STAGE=0: the expert kernels read the arena over PCIe themselves
    bool in_ram(const Layer& l, int e) const {   // (DISK_SKIP_W) expert e's pages in RAM now
        if (!l.ram.empty() && l.ram[(size_t) e] >= 0) return true;
        if (l.lay.fmt == gg::kFmtMXFP4) {
            for (int r = 0; r < 3; ++r)
                if (!os::in_ram(mx_mat(l, e, r), mx_part_bytes(l, r))) return false;
            return true;
        }
        return !file_experts || os::in_ram(file_part(l, e), l.blob);
    }
    // a GGUF's CPU-lane experts copied into RAM (copy_experts, after VRAM took its share: --mmap-experts off - auto reads
    // them in place; DS4_COPY_GB=<n>: the hottest n GB, the rest in place - static, no moves between
    // them): RAM slots (l.ram, as the tier's) of NH parts, part X on NUMA node X with rows [X n / NH, (X + 1) n / NH) of
    // each matrix at cp_lay[layer][r] - whole rows, so the split never cuts a GGUF block (MXFP4's 32, the IQ / K types'
    // 256 values: blocks run along a row); one spare slot a layer for the cache's swaps (the expert VRAM gives back takes
    // the RAM place of the one it takes in)
    // Windows' commit on an integrated GPU: its allocations are charged to the commit (RAM + page file) like any private
    // memory - measured on a Radeon 8065S with 160 GiB of its own: the system's free commit fell by each one, whatever
    // its size (one above ~3 GiB also counts as the process's private bytes, a smaller one does not, the commit the
    // same), so a page file at least the GPU's share is what lets it hold experts at all.  The experts it holds
    // (place_experts) and the copies stay within what Windows can still commit, less max(8 GB, 10% of the RAM) for the
    // other programs - commit_kept: the experts that kept off the GPU, pagefile_hint: the page file (GB) that would hold
    // them (the log's WARNING, a NOTE and INFO for the server)
    double commit_kept = 0, pagefile_hint = 0;
    double commit_margin() const {
        const os::Commit c = os::commit();
        return std::max(8.0 * (double) (1ull << 30), c.ram == ~uint64_t{0} ? 0.0 : 0.1 * (double) c.ram);
    }
    // what may still be committed beside the margin (~inf: nothing limits it; DS4_COMMIT_CAP_GB: at most that, tests)
    double commit_room() const {
        const os::Commit c = os::commit();
        double avail = c.avail == ~uint64_t{0} ? 1e300 : (double) c.avail;
        if (const char* v = std::getenv("DS4_COMMIT_CAP_GB")) avail = std::min(avail, std::atof(v) * 1e9);
        return avail >= 1e300 ? 1e300 : std::max(0.0, avail - commit_margin());
    }
    // a device whose allocations Windows charges to the commit: an integrated GPU (DS4_COMMIT_CAP_GB: any, tests)
    bool commit_charged(const Dev& d) const {
        if (std::getenv("DS4_COMMIT_CAP_GB") != nullptr) return true;
#if defined(_WIN32)
        int v = 0;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrIntegrated, d.id) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        return v != 0;
#else
        (void) d;
        return false;
#endif
    }
    bool copy_mode = false;
    std::vector<std::array<uint8_t*, 2>> ram_part;   // slot -> its parts
    std::vector<std::array<size_t, 4>> cp_lay;       // layer -> part offsets of gate, up, down rows; a part's bytes
    std::vector<int> copy_spare;                      // layer -> its spare slot (-1: none)
    std::mutex sw_mu;
    std::condition_variable sw_cv;
    std::deque<Swap> sw_queue;
    std::vector<Swap> sw_ready;
    struct SwapBufs {
        uint8_t *a = nullptr, *b = nullptr, *h = nullptr;
        std::vector<cudaStream_t> st;   // a copy stream a device
    };
    std::vector<SwapBufs> sw_bufs;
    std::vector<std::thread> cache_ths;
    int sw_inflight = 0;
    int64_t decay_tokens = 0;
    bool check_swaps = std::getenv("DS4_CHECK_SWAPS") != nullptr;
    std::vector<float> host_logits;
    std::vector<void*> host_allocs;
    std::FILE* dump_h = std::getenv("DS4_DUMP_H") ? std::fopen(std::getenv("DS4_DUMP_H"), "wb") : nullptr;
    // DS4_DUMP_L: (threads) every layer's residual streams as records [layer, p0, T, T x hc x dim floats], window or
    // chunk path (tests: where the two part)
    std::FILE* dump_l = std::getenv("DS4_DUMP_L") ? std::fopen(std::getenv("DS4_DUMP_L"), "wb") : nullptr;
    // DS4_DUMP_S: the sub-steps inside the layers named by DS4_DUMP_S_LAYERS (default 0,1,2,3), records
    // [char name[16], int64 il, p0, T, nbytes][nbytes raw bytes] (window path, the same on CUDA and Metal)
    std::FILE* dump_s = std::getenv("DS4_DUMP_S") ? std::fopen(std::getenv("DS4_DUMP_S"), "wb") : nullptr;
    std::vector<int> dump_s_layers = [] {
        std::vector<int> v;
        const char* e = std::getenv("DS4_DUMP_S_LAYERS");
        std::string str = e ? e : "0,1,2,3";
        size_t a = 0;
        while (a < str.size()) {
            size_t b = str.find(',', a);
            if (b == std::string::npos) b = str.size();
            if (b > a) v.push_back(std::atoi(str.substr(a, b - a).c_str()));
            a = b + 1;
        }
        return v;
    }();
    // DS4_DUMP_R: the chunk path's routing, a record a layer [layer, p0, T, k, n_exp, n_vs, n_slots, kPicks] (int64),
    // then the picks' experts [T kPicks] (the first k of each token's), each expert's virtual slot [n_exp], the
    // slots' row bounds [n_vs + 1], the sorted rows' tokens [T k] (int32), the gate's logits [T n_exp] and its bias
    // [n_exp] (F32)
    std::FILE* dump_r = std::getenv("DS4_DUMP_R") ? std::fopen(std::getenv("DS4_DUMP_R"), "wb") : nullptr;
    // DS4_DUMP_Y=<layer>:<file> (layer -1: every layer): its expert sets on the chunk path, a record a piece [layer,
    // row0, rows, n_exp, wbase kind (0 VRAM, 1 ring), max_rows] (int64), its bounds [n_exp + 1], its h rows [rows ff]
    // (F16) and y rows [rows dim] (F32)
    int dump_y_layer = std::getenv("DS4_DUMP_Y") ? std::atoi(std::getenv("DS4_DUMP_Y")) : -1;
    std::FILE* dump_y = [] {
        const char* v = std::getenv("DS4_DUMP_Y");
        const char* f = v != nullptr ? std::strchr(v, ':') : nullptr;
        return f != nullptr ? std::fopen(f + 1, "wb") : nullptr;
    }();
    std::mutex dump_mu;
    // DS4_PROF: the GPU's time a decoded token by stage - events in stream order at a layer's start, after its
    // attention, after the routing and the shared expert, after the experts' gate / up, after the wait for the CPU
    // lane, after the down pass and the post - read after the token's sync, each the time since the mark before it
    bool prof_on = std::getenv("DS4_PROF") != nullptr;
    // DS4_PRED_DUMP=file (predictor data, the chunked prompt path): per layer and chunk u8 layer, u16 T, T x 384 fp16 router
    // logits, T x 256 fp16 projection of the router input.  DS4_XIN_DUMP=file (expert similarity inputs): u8 layer, u16 T,
    // T x D fp32 router inputs, the first DS4_XIN_N (1024) tokens of each layer.
    std::FILE* pred_f = std::getenv("DS4_PRED_DUMP") != nullptr ? std::fopen(std::getenv("DS4_PRED_DUMP"), "wb") : nullptr;
    std::FILE* xin_f = std::getenv("DS4_XIN_DUMP") != nullptr ? std::fopen(std::getenv("DS4_XIN_DUMP"), "wb") : nullptr;
    int xin_n = std::getenv("DS4_XIN_N") ? std::atoi(std::getenv("DS4_XIN_N")) : 1024;
    std::array<int, 64> xin_got{};
    std::FILE* trace_f = std::getenv("DS4_TRACE") != nullptr ? std::fopen(std::getenv("DS4_TRACE"), "wb") : nullptr;
    struct ProfMark {
        const Dev* d;
        cudaEvent_t ev;
        int stage;
    };
    std::mutex prof_mu;
    std::vector<ProfMark> prof_marks;
    std::map<int, std::vector<cudaEvent_t>> prof_pool;   // by device: an event belongs to the device it was made on
    bool prof_window = false;   // (decode windows only: T == 1)
    const bool prof_sync_on = std::getenv("DS4_PROF_SYNC") != nullptr;
    double prof_sync_t = 0;     // DS4_PROF_SYNC: the host clock at the last mark
    // ... and a run of chunks' (prof_chunk): a layer's start, after its attention, after the routing and the shared
    // expert, after its experts (the streamed ones' waits for their reads in it); printed after the run
    bool prof_chunk = false;
    // DS4_PROF stages of the chunked prompt (finer ones than the decode's 0..5): 0 between layers, 1 attention rest (wq_b, wo,
    // hc_post), 2 router-side tail (shared expert), 3 experts (+ waits), 4 Engram, 5 attn mHC (mix, split, norm), 6 attention
    // projections (wq_a, wkv, compressor), 7 compressor + indexer, 8 q_b + rope, 9 sparse attention, 10 ffn mHC + router + sort
    double cprof_ms[16] = {};
    // DS4_PROF: the experts' kernels alone, by type (events around the gate/up and down calls of run_set)
    struct EkMark {
        cudaEvent_t e0, e1, e2;
        int tg, td, nrows, nexp, maxrows, dev, ff;
    };
    std::vector<EkMark> ek_marks;
    std::map<int, std::vector<cudaEvent_t>> ek_pool;   // by device
    cudaEvent_t ek_ev() {
        int cur = 0;
        cudaGetDevice(&cur);
        std::vector<cudaEvent_t>& pl = ek_pool[cur];
        if (!pl.empty()) {
            cudaEvent_t e = pl.back();
            pl.pop_back();
            return e;
        }
        cudaEvent_t e = nullptr;
        cudaEventCreate(&e);
        return e;
    }
    void prof_mark(const Dev& d, int stage) {
        if (ds4_dbg("DS4_TLOG")) {   // the host's own phase times (what the GPU meters do not see)
            static double t_last = 0, acc[5] = {0, 0, 0, 0, 0}; static long n = 0; static bool reg = false;
            if (!reg) {
                reg = true;
                std::atexit([] {
                    std::fprintf(stderr,
                                 "tlog: %ld layers | host ms: attn %.0f  moe_pre %.0f  gate_up %.0f  cpu_wait %.0f  down+post %.0f\n",
                                 n, acc[0], acc[1], acc[2], acc[3], acc[4]);
                });
            }
            const double t = now_ms();
            if (t_last > 0) {
                const int b = stage == 0 ? 4 : stage - 1;
                acc[b >= 0 && b < 5 ? b : 4] += t - t_last;
            }
            t_last = t;
            if (stage == 3) ++n;
        }
        if (!prof_window && !prof_chunk) return;
        if (prof_window && !prof_chunk && prof_sync_on) {   // decode: the stream drained at each mark, host clock (events hang the driver on gfx1151 Windows)
            std::lock_guard<std::mutex> lk(prof_mu);
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
            const double t = now_ms();
            if (prof_sync_t > 0) st.prof_ms[stage] += t - prof_sync_t;
            prof_sync_t = now_ms();
            return;
        }
        std::lock_guard<std::mutex> lk(prof_mu);
        int cur = 0;
        cudaGetDevice(&cur);
        cudaSetDevice(d.id);   // (events of another device are an "invalid resource handle")
        cudaEvent_t e = nullptr;
        std::vector<cudaEvent_t>& pool = prof_pool[d.id];
        if (pool.empty()) cudaEventCreate(&e);
        else {
            e = pool.back();
            pool.pop_back();
        }
        cudaEventRecord(e, d.st);
        cudaSetDevice(cur);
        prof_marks.push_back({&d, e, stage});
    }
    void prof_collect() {
        if (!prof_window && !prof_chunk) return;
        if (prof_window && !prof_chunk && prof_sync_on) {
            std::lock_guard<std::mutex> lk(prof_mu);
            prof_sync_t = 0;
            ++st.prof_tokens;
            return;
        }
        std::lock_guard<std::mutex> lk(prof_mu);
        std::vector<std::pair<const Dev*, cudaEvent_t>> last;
        for (const ProfMark& m : prof_marks) {
            auto it = std::find_if(last.begin(), last.end(), [&](const auto& x) { return x.first == m.d; });
            if (it != last.end()) {
                float ms = 0.0f;
                cudaSetDevice(m.d->id);
                if (cudaEventElapsedTime(&ms, it->second, m.ev) == cudaSuccess)
                    (prof_chunk ? cprof_ms : st.prof_ms)[m.stage] += ms;
                it->second = m.ev;
            } else {
                last.push_back({m.d, m.ev});
            }
        }
        for (const ProfMark& m : prof_marks) prof_pool[m.d->id].push_back(m.ev);
        prof_marks.clear();
        if (!prof_chunk) ++st.prof_tokens;
    }
    void dump_layer(Dev& d, const float* h, int il, int64_t p0, int T) {
        std::vector<float> v((size_t) T * g.hc * g.dim);
        cudaSetDevice(d.id);
#if defined(STRATA_METAL)
        ds4_metal_flush();   // (the residual stream is read back here: commit the batched dispatches first)
        cudaDeviceSynchronize();
#endif
        cudaMemcpyAsync(v.data(), h, v.size() * 4, cudaMemcpyDeviceToHost, d.st);
        cudaStreamSynchronize(d.st);
        std::lock_guard<std::mutex> lk(dump_mu);
        const int64_t hdr[3] = {il, p0, T};
        std::fwrite(hdr, 8, 3, dump_l);
        std::fwrite(v.data(), 4, v.size(), dump_l);
        std::fflush(dump_l);
    }

    bool dump_s_on(int il) const {
        if (dump_s == nullptr) return false;
        for (int x : dump_s_layers)
            if (x == il) return true;
        return false;
    }
    bool dump_s_name(const char* name) const {   // DS4_DUMP_S_NAMES=a,b,c: only these points (default all)
        static const std::string names = [] { const char* e = std::getenv("DS4_DUMP_S_NAMES"); return std::string(e ? "," : "") + (e ? e : "") + (e ? "," : ""); }();
        return names.empty() || names.find(std::string(",") + name + ",") != std::string::npos;
    }
    void dump_sub(Dev& d, const char* name, int il, int64_t p0, int T, const void* dev, size_t nbytes) {
        if (!dump_s_on(il) || !dump_s_name(name)) return;
        std::vector<uint8_t> v(nbytes);
        cudaSetDevice(d.id);
#if defined(STRATA_METAL)
        ds4_metal_flush();
        cudaDeviceSynchronize();   // (the batched dispatches have finished before the read)
#endif
        cudaMemcpyAsync(v.data(), dev, nbytes, cudaMemcpyDeviceToHost, d.st);
        cudaStreamSynchronize(d.st);
        dump_sub_host(name, il, p0, T, v.data(), nbytes);
    }
    void dump_sub_host(const char* name, int il, int64_t p0, int T, const void* host, size_t nbytes) {
        if (!dump_s_on(il) || !dump_s_name(name)) return;
        std::lock_guard<std::mutex> lk(dump_mu);
        char nm[16] = {};
        std::strncpy(nm, name, 15);
        const int64_t hdr[4] = {il, p0, T, (int64_t) nbytes};
        std::fwrite(nm, 1, 16, dump_s);
        std::fwrite(hdr, 8, 4, dump_s);
        std::fwrite(host, 1, nbytes, dump_s);
        std::fflush(dump_s);
    }

    ~Impl() {
        quit.store(true);
        if (warm_bg.joinable()) warm_bg.join();   // (the warm-up uses the BLAS handles destroyed below)
        eg_io.stop();   // (its threads write into h_eg)
        for (auto& w : workers) w->stop();
        for (size_t i = 0; i < chunks.size(); ++i)
            if (chunks[i] && chunks[i]->blas) {
                cudaSetDevice(devs[i].id);
                cublasDestroy(chunks[i]->blas);
            }
        quit.store(true);
        if (service.joinable()) service.join();
        if (store_bg.joinable()) store_bg.join();   // (the store's background fill: it checks quit between its pieces)
        {
            std::lock_guard<std::mutex> lk(pf_mu);
        }
        pf_cv.notify_all();
        if (pf_th.joinable()) pf_th.join();
        {
            std::lock_guard<std::mutex> lk(sw_mu);
        }
        sw_cv.notify_all();
        ram_cv.notify_all();
        for (std::thread& t : cache_ths)
            if (t.joinable()) t.join();
        if (ram_th.joinable()) ram_th.join();
        for (size_t i = 0; i < ram_chunks.size(); ++i) {
            if (ram_pinned) (void) cudaHostUnregister(ram_chunks[i]);
            os::anon_free(ram_chunks[i], ram_chunk_bytes[i]);
        }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            for (void* p : d.allocs) cudaFree(p);
            if (d.st) cudaStreamDestroy(d.st);
            if (d.ev) cudaEventDestroy(d.ev);
        }
        for (void* p : host_allocs) cudaFreeHost(p);
        for (Layer& l : L)
            for (uint8_t* b : l.host)
                if (b != nullptr) {
                    (void) b;   // host blobs live in one mapping per layer, unmapped below
                }
        for (auto& m : host_maps) {
            if (chunk > 0) cudaHostUnregister(m.first);
            os::anon_free(m.first, m.second);
        }
    }
    std::vector<std::pair<void*, size_t>> host_maps;

    template <class T>
    bool dalloc(Dev& d, T*& p, size_t n, std::string& err, size_t* acct = nullptr) {
        void* v = nullptr;
        cudaSetDevice(d.id);
        const size_t b = align256(n * sizeof(T));
        if (cudaMalloc(&v, b) != cudaSuccess) {
            err = "cudaMalloc " + std::to_string(b >> 20) + " MiB on CUDA" + std::to_string(d.id) + " failed";
            return false;
        }
        // zeroed before anything lands in it: the loads copy in on the default stream, which the non-blocking d.st
        // does not order against - a zeroing still queued there could wipe the weights just copied
        if (cudaMemsetAsync(v, 0, b, d.st) != cudaSuccess || cudaStreamSynchronize(d.st) != cudaSuccess) {
            cudaFree(v);
            err = "zeroing " + std::to_string(b >> 20) + " MiB on CUDA" + std::to_string(d.id) + " failed";
            return false;
        }
        d.allocs.push_back(v);
        if (acct) *acct += b;
        else d.scratch += b;
        p = (T*) v;
        return true;
    }
    static Layer& member(Layer& l, int m) { return m == 0 ? l : l.mates[(size_t) m - 1]; }
    static const Layer& member(const Layer& l, int m) { return m == 0 ? l : l.mates[(size_t) m - 1]; }
    // bytes of a VRAM slot of l's experts (a member's part of each, in a group)
    static size_t unit(const Layer& l) { return l.nm > 1 ? l.hblob : l.blob; }
    // the members of a group of nm that take heads: the most the heads and the wo_a groups split among evenly (64
    // heads and 8 groups: 9 -> 8, 7 or 6 -> 4, 5 -> 4, 3 -> 2); the first members the rest, only their FFN share
    int head_members(int nm) const {
        int na = nm;
        while (na > 1 && (g.heads % na != 0 || g.o_groups % na != 0)) --na;
        return na;
    }
    // host-mapped memory (the same address on the host and every device), zeroed
    template <class T>
    bool hmap(T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (cudaHostAlloc(&v, n * sizeof(T) + 64, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            err = "host-mapped memory (" + std::to_string((n * sizeof(T)) >> 20) + " MiB)";
            return false;
        }
        std::memset(v, 0, n * sizeof(T) + 64);
        host_allocs.push_back(v);
        p = (T*) v;
        return true;
    }
    template <class T>
    bool halloc(T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (cudaHostAlloc(&v, n * sizeof(T) + 64, cudaHostAllocPortable) != cudaSuccess) {
            err = "cudaHostAlloc " + std::to_string((n * sizeof(T)) >> 20) + " MiB failed";
            return false;
        }
        host_allocs.push_back(v);
        p = (T*) v;
        return true;
    }

    // ---- weights
    bool upload(Dev& d, const Tensor& t, void* dst, std::string& err) {
        std::vector<uint8_t> tmp(t.bytes);
        if (!pk.read(t, tmp.data())) {
            err = "read failed";
            return false;
        }
        cudaSetDevice(d.id);
        CK(cudaMemcpy(dst, tmp.data(), t.bytes, cudaMemcpyHostToDevice));
        return true;
    }
    // a dense matrix in a ggml block type (x3::Mat fmt kFmtGG, row = k / 256 blocks): whole, or part of its rows / inputs
    bool mat_gg(Dev& d, const std::string& pre, const Tensor& t, int ggt, x3::Mat& m, std::string& err, size_t* acct, int part,
                int np, bool cols) {
        if (t.shape.size() != 2 || t.shape[1] % 256 != 0) {
            err = pre + ": a " + t.dtype + " tensor that is not a matrix of whole blocks";
            return false;
        }
        int n = (int) t.shape[0], k = (int) t.shape[1];
        const size_t rb = gg::type_row_bytes(ggt, k), bb = rb / ((size_t) k / 256);
        std::vector<uint8_t> w((size_t) n * rb);
        if (t.bytes != w.size() || !pk.read(t, w.data())) {
            err = pre + ": read failed";
            return false;
        }
        if (np > 1) {
            const int whole = cols ? n : k, wd = whole / np;
            if (whole % np != 0 || wd % (cols ? 1 : 256) != 0) {
                err = pre + ": " + std::to_string(whole) + (cols ? " outputs" : " inputs") + " do not split in " + std::to_string(np) + " parts";
                return false;
            }
            std::vector<uint8_t> p;
            if (cols) {
                p.assign(w.begin() + (long) ((size_t) part * wd * rb), w.begin() + (long) ((size_t) (part + 1) * wd * rb));
                n = wd;
            } else {
                const size_t pb = (size_t) (wd / 256) * bb, off = (size_t) (part * wd / 256) * bb;
                p.resize((size_t) n * pb);
                for (size_t r = 0; r < (size_t) n; ++r) std::memcpy(p.data() + r * pb, w.data() + r * rb + off, pb);
                k = wd;
            }
            w.swap(p);
        }
        uint8_t* dp = nullptr;
        if (!dalloc(d, dp, w.size(), err, acct)) return false;
        cudaSetDevice(d.id);
        CK(cudaMemcpy(dp, w.data(), w.size(), cudaMemcpyHostToDevice));
        m = x3::Mat{};
        m.fmt = gg::kFmtGG;
        m.K = ggt;
        m.raw = dp;
        m.k = k;
        m.n = n;
        m.ld = (int) gg::type_row_bytes(ggt, k);
        return true;
    }
    // a GGUF's BF16 or Q8_0 matrix ([out][in]) onto the device, as mat() takes an EXL3 one: whole, or part `part` of its
    // columns (outputs: a range of its rows) or of its rows (inputs: a range of each row, whole Q8_0 blocks) - a
    // standalone [n][k] matrix
    bool mat_bf16(Dev& d, const std::string& pre, const Tensor& t, x3::Mat& m, std::string& err, size_t* acct, int part,
                  int np, bool cols) {
        const bool q8 = t.dtype == "Q8_0";
        // a ggml K-quant dense matrix (Q4_K / Q5_K / Q6_K: 256 values a block; the GEMV path of ds4_iq)
        const int ggt = (t.dtype == "Q4_K" || t.dtype == "Q5_K" || t.dtype == "Q6_K") ? gg_type_of(t.dtype) : -1;
        if (ggt > 0) return mat_gg(d, pre, t, ggt, m, err, acct, part, np, cols);
        if (t.shape.size() != 2 || (q8 && t.shape[1] % 32 != 0)) {
            err = pre + ": a " + t.dtype + " tensor that is not a matrix";
            return false;
        }
        int n = (int) t.shape[0], k = (int) t.shape[1];
        // a row's bytes, and the bytes of v of its values (BF16: 2 each; Q8_0: 34 every 32)
        auto bytes_of = [&](size_t v) { return q8 ? v / 32 * gg::kQ8Bytes : v * 2; };
        std::vector<uint8_t> w((size_t) n * bytes_of((size_t) k));
        if (t.bytes != w.size() || !pk.read(t, w.data())) {
            err = pre + ": read failed";
            return false;
        }
        if (np > 1) {
            const int whole = cols ? n : k, wd = whole / np;
            if (whole % np != 0 || wd % (q8 ? 32 : 8) != 0) {
                err = pre + ": " + std::to_string(whole) + (cols ? " outputs" : " inputs") + " do not split in " +
                      std::to_string(np) + " parts";
                return false;
            }
            const size_t rb = bytes_of((size_t) k);
            std::vector<uint8_t> p;
            if (cols) {
                p.assign(w.begin() + (long) ((size_t) part * wd * rb), w.begin() + (long) ((size_t) (part + 1) * wd * rb));
                n = wd;
            } else {
                const size_t pb = bytes_of((size_t) wd);
                p.resize((size_t) n * pb);
                for (size_t r = 0; r < (size_t) n; ++r)
                    std::memcpy(p.data() + r * pb, w.data() + r * rb + bytes_of((size_t) part * wd), pb);
                k = wd;
            }
            w.swap(p);
        }
        uint8_t* dp = nullptr;
        if (!dalloc(d, dp, w.size(), err, acct)) return false;
        cudaSetDevice(d.id);
        CK(cudaMemcpy(dp, w.data(), w.size(), cudaMemcpyHostToDevice));
        m = x3::Mat{};
        m.fmt = q8 ? gg::kFmtQ8_0 : gg::kFmtBF16;
        m.raw = dp;
        m.k = k;
        m.n = n;
        m.ld = q8 ? (int) bytes_of((size_t) k) : k;
        return true;
    }
    // a Q8_0 tensor's values as F32 (host)
    static std::vector<float> q8_values(const uint8_t* q, size_t n) {
        std::vector<float> v(n);
        for (size_t b = 0; b < n / 32; ++b) {
            const uint8_t* p = q + b * gg::kQ8Bytes;
            const float d = f16_to_f32((uint16_t) (p[0] | p[1] << 8));
            for (int j = 0; j < 32; ++j) v[b * 32 + j] = d * (float) (int8_t) p[2 + j];
        }
        return v;
    }
    // an EXL3 matrix onto the device: whole, or (np > 1) part `part` of its columns (cols) or of its rows - a standalone
    // matrix of its own (the rotations' 128-blocks stay whole)
    // (rot: split128's - an EXL3 matrix's uneven 128-block shares going round the members; a GGUF's BF16 / Q8_0 matrix
    // splits evenly, its groups only as wide as that goes)
    bool mat(Dev& d, const std::string& pre, x3::Mat& m, std::string& err, size_t* acct, int part = 0, int np = 1,
             bool cols = true, int rot = 0) {
        if (const Tensor* tw = pk.get(pre + ".weight"); tw != nullptr && (tw->dtype == "BF16" || tw->dtype == "Q8_0" || tw->dtype == "Q4_K" || tw->dtype == "Q5_K" || tw->dtype == "Q6_K"))
            return mat_bf16(d, pre, *tw, m, err, acct, part, np, cols);
        const Tensor* tr = pk.get(pre + ".trellis");
        const Tensor* su = pk.get(pre + ".suh");
        const Tensor* sv = pk.get(pre + ".svh");
        if (tr == nullptr || su == nullptr || sv == nullptr || tr->shape.size() != 3) {
            err = pre + ": not an EXL3 matrix in the pack";
            return false;
        }
        if (pk.get(pre + ".mul1") == nullptr) {
            err = pre + ": only the mul1 codebook is covered";
            return false;
        }
        m.k = (int) tr->shape[0] * 16;
        m.n = (int) tr->shape[1] * 16;
        m.K = (int) tr->shape[2] / 16;
        m.cb = 2;
        m.ld = 0;
        if (np > 1) {
            const int whole = cols ? m.n : m.k;
            int at = 0, w = 0;
            if (whole % 128 != 0 || whole / 128 < np) {
                err = pre + ": " + std::to_string(whole) + (cols ? " columns" : " rows") + " do not split in " +
                      std::to_string(np) + " parts of 128-blocks";
                return false;
            }
            split128(whole, np, part, at, w, rot);
            std::vector<uint8_t> t0(tr->bytes), u0(su->bytes), v0(sv->bytes);
            if (!pk.read(*tr, t0.data()) || !pk.read(*su, u0.data()) || !pk.read(*sv, v0.data())) {
                err = pre + ": read failed";
                return false;
            }
            const size_t tile = tr->bytes / ((size_t) (m.k / 16) * (m.n / 16));
            std::vector<uint8_t> t1, u1, v1;
            if (cols) {   // the tiles [c0 / 16, (c0 + w) / 16) of each 16-row slice, svh [c0, c0 + w)
                const size_t c0 = (size_t) at, nt = (size_t) m.n / 16, nth = (size_t) w / 16;
                t1.resize((size_t) (m.k / 16) * nth * tile);
                for (size_t sl = 0; sl < (size_t) m.k / 16; ++sl)
                    std::memcpy(t1.data() + sl * nth * tile, t0.data() + (sl * nt + c0 / 16) * tile, nth * tile);
                u1 = u0;
                v1.assign(v0.begin() + (long) (c0 * 2), v0.begin() + (long) ((c0 + w) * 2));
                m.n = w;
            } else {      // the 16-row slices [r0 / 16, (r0 + w) / 16), suh [r0, r0 + w)
                const size_t r0 = (size_t) at, row = (size_t) (m.n / 16) * tile;
                t1.assign(t0.begin() + (long) (r0 / 16 * row), t0.begin() + (long) ((r0 + w) / 16 * row));
                u1.assign(u0.begin() + (long) (r0 * 2), u0.begin() + (long) ((r0 + w) * 2));
                v1 = v0;
                m.k = w;
            }
            uint32_t* dt = nullptr;
            uint16_t *ds = nullptr, *dv = nullptr;
            if (!dalloc(d, dt, t1.size() / 4, err, acct) || !dalloc(d, ds, u1.size() / 2, err, acct) ||
                !dalloc(d, dv, v1.size() / 2, err, acct))
                return false;
            cudaSetDevice(d.id);
            CK(cudaMemcpy(dt, t1.data(), t1.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(ds, u1.data(), u1.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dv, v1.data(), v1.size(), cudaMemcpyHostToDevice));
            m.trellis = dt;
            m.suh = ds;
            m.svh = dv;
            return x3::supported(m) || (err = pre + ": a part not covered by the EXL3 kernels", false);
        }
        uint32_t* dt = nullptr;
        uint16_t *ds = nullptr, *dv = nullptr;
        if (!dalloc(d, dt, tr->bytes / 4, err, acct) || !dalloc(d, ds, su->bytes / 2, err, acct) ||
            !dalloc(d, dv, sv->bytes / 2, err, acct))
            return false;
        if (!upload(d, *tr, dt, err) || !upload(d, *su, ds, err) || !upload(d, *sv, dv, err)) return false;
        m.trellis = dt;
        m.suh = ds;
        m.svh = dv;
        if (!x3::supported(m)) {
            err = pre + ": k " + std::to_string(m.k) + " n " + std::to_string(m.n) + " K " + std::to_string(m.K) +
                  " not covered by the EXL3 kernels";
            return false;
        }
        return true;
    }
    // a float tensor (F32 / F16 / BF16) as F32 on the device
    bool f32(Dev& d, const std::string& name, float*& p, std::string& err, size_t* acct, size_t want = 0) {
        const Tensor* t = pk.get(name);
        if (t == nullptr) {
            err = name + ": not in the pack";
            return false;
        }
        std::vector<uint8_t> raw(t->bytes);
        if (!pk.read(*t, raw.data())) {
            err = name + ": read failed";
            return false;
        }
        size_t n = 0;
        std::vector<float> v;
        if (t->dtype == "F32") {
            n = t->bytes / 4;
            v.resize(n);
            std::memcpy(v.data(), raw.data(), t->bytes);
        } else if (t->dtype == "F16" || t->dtype == "BF16") {
            n = t->bytes / 2;
            v.resize(n);
            const uint16_t* s = (const uint16_t*) raw.data();
            for (size_t i = 0; i < n; ++i) v[i] = t->dtype == "F16" ? f16_to_f32(s[i]) : bf16_to_f32(s[i]);
        } else if (t->dtype == "Q8_0") {   // (a GGUF's router may come so)
            n = t->bytes / gg::kQ8Bytes * 32;
            v = q8_values(raw.data(), n);
        } else {
            err = name + ": dtype " + t->dtype + " is not a float";
            return false;
        }
        if (want != 0 && n != want) {
            err = name + ": " + std::to_string(n) + " values, expected " + std::to_string(want);
            return false;
        }
        if (!dalloc(d, p, n, err, acct)) return false;
        cudaSetDevice(d.id);
        CK(cudaMemcpy(p, v.data(), n * 4, cudaMemcpyHostToDevice));
        return true;
    }
    bool f16raw(Dev& d, const std::string& name, uint16_t*& p, std::string& err, size_t* acct) {
        const Tensor* t = pk.get(name);
        if (t != nullptr && (t->dtype == "F32" || t->dtype == "BF16" || t->dtype == "Q8_0")) {   // (a GGUF's)
            const size_t n = t->dtype == "Q8_0" ? t->bytes / gg::kQ8Bytes * 32 : t->bytes / (t->dtype == "F32" ? 4 : 2);
            std::vector<uint8_t> raw(t->bytes);
            if (!pk.read(*t, raw.data())) {
                err = name + ": read failed";
                return false;
            }
            const std::vector<float> q = t->dtype == "Q8_0" ? q8_values(raw.data(), n) : std::vector<float>();
            std::vector<uint16_t> h(n);
            for (size_t i = 0; i < n; ++i) {
                float v;
                if (t->dtype == "F32") std::memcpy(&v, raw.data() + i * 4, 4);
                else if (t->dtype == "Q8_0") v = q[i];
                else v = bf16_to_f32(((const uint16_t*) raw.data())[i]);
                h[i] = strata::kernels::f16_from_f32(v);
            }
            if (!dalloc(d, p, n, err, acct)) return false;
            cudaSetDevice(d.id);
            CK(cudaMemcpy(p, h.data(), n * 2, cudaMemcpyHostToDevice));
            return true;
        }
        if (t == nullptr || t->dtype != "F16") {
            err = name + ": not an F16 tensor in the pack";
            return false;
        }
        if (!dalloc(d, p, t->bytes / 2, err, acct)) return false;
        return upload(d, *t, p, err);
    }

    // The markov bias's head as Q8_0 blocks of 32 (34 B: F16 d + 32 int8) instead of F16 values: the same bias sums with the
    // quantisation's error, half the bytes the drafter's five markov steps read (66 MB -> 33 MB) and half the memory.  The
    // pack is untouched: the F32 / F16 / BF16 tensor is quantised as it is loaded.
    bool mk_head_q8(Dev& d, const std::string& name, uint8_t*& p, std::string& err, size_t* acct) {
        const Tensor* t = pk.get(name);
        if (t == nullptr || (t->dtype != "F32" && t->dtype != "F16" && t->dtype != "BF16")) {
            err = name + ": not an F32 / F16 / BF16 tensor";
            return false;
        }
        const size_t n = t->bytes / (t->dtype == "F32" ? 4 : 2);
        if (n == 0 || n % 32 != 0) {
            err = name + ": not a whole number of 32-value blocks";
            return false;
        }
        std::vector<uint8_t> raw(t->bytes);
        if (!pk.read(*t, raw.data())) {
            err = name + ": read failed";
            return false;
        }
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i) {
            if (t->dtype == "F32") std::memcpy(&v[i], raw.data() + i * 4, 4);
            else v[i] = t->dtype == "F16" ? f16_to_f32(((const uint16_t*) raw.data())[i])
                                          : bf16_to_f32(((const uint16_t*) raw.data())[i]);
        }
        const size_t nb = n / 32;
        std::vector<uint8_t> q(nb * 34);
        for (size_t b = 0; b < nb; ++b) {
            const float* s = v.data() + b * 32;
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(s[i]));
            const float dd = amax > 0.0f ? amax / 127.0f : 0.0f;
            const uint16_t dh = strata::kernels::f16_from_f32(dd);
            std::memcpy(q.data() + b * 34, &dh, 2);
            int8_t* o = (int8_t*) (q.data() + b * 34 + 2);
            for (int i = 0; i < 32; ++i) {
                const float x = dd > 0.0f ? s[i] / dd : 0.0f;
                o[i] = (int8_t) std::max(-127.0f, std::min(127.0f, std::round(x)));
            }
        }
        if (!dalloc(d, p, nb * 34, err, acct)) return false;
        cudaSetDevice(d.id);
        CK(cudaMemcpy(p, q.data(), nb * 34, cudaMemcpyHostToDevice));
        return true;
    }

    size_t mat_bytes(const std::string& pre) const {
        if (const Tensor* t = pk.get(pre + ".weight")) return align256(t->bytes);
        size_t b = 0;
        for (const char* s : {".trellis", ".suh", ".svh"})
            if (const Tensor* t = pk.get(pre + s)) b += align256(t->bytes);
        return b;
    }    size_t tensor_bytes(const std::string& name) const {
        const Tensor* t = pk.get(name);
        return t ? align256(t->bytes * (t->dtype == "F32" ? 1 : 2)) : 0;
    }

    // ---- expert regions and blob layout
    // a GGUF's MXFP4 experts: the blob is gate | up | down ([out][in / 32] blocks each, as the file holds them), every
    // expert's matrix r at the same stride from expert 0's in one file (the GGUF's [n_exp][out][in] tensor)
    // (the other block types: lay.fmt stays kFmtMXFP4 - read in place as these - and lay.K[r] is role r's ggml type)
    static int gg_type_of(const std::string& dtype) {
        for (int t : {gg::kTypeMXFP4, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 145, 146, 224, 225, 226, 227})
            if (dtype == gg::type_name(t)) return t;
        return -1;
    }
    bool expert_layout_mx(Layer& l, std::string& err) {
        const std::string P = l.name + "ffn.experts.";
        const char* parts[3] = {"w1", "w3", "w2"};   // gate, up, down
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            const int n = r < 2 ? g.ff : g.dim, k = r < 2 ? g.dim : g.ff;
            const Tensor* t0 = pk.get(P + "0." + parts[r] + ".weight");
            const Tensor* t1 = l.n_exp > 1 ? pk.get(P + "1." + parts[r] + ".weight") : t0;
            const int ty = t0 != nullptr ? gg_type_of(t0->dtype) : -1;
            const size_t bytes = ty > 0 ? (size_t) n * gg::type_row_bytes(ty, k) : 0;
            if (t0 == nullptr || t1 == nullptr || bytes == 0 || t0->bytes != bytes) {
                err = P + "0." + parts[r] + ": not an MXFP4 or low-bit [" + std::to_string(n) + "][" + std::to_string(k) +
                      "] matrix the engine reads (" + (t0 != nullptr ? t0->dtype : std::string("missing")) + ")";
                return false;
            }
            l.mx_file[r] = t0->file;
            l.mx_off[r] = t0->off;
            l.mx_stride[r] = l.n_exp > 1 ? t1->off - t0->off : bytes;
            for (int e = 0; e < l.n_exp; ++e) {
                const Tensor* t = pk.get(P + std::to_string(e) + "." + parts[r] + ".weight");
                if (t == nullptr || t->dtype != t0->dtype || t->bytes != bytes || t->file != l.mx_file[r] ||
                    t->off != l.mx_off[r] + (uint64_t) e * l.mx_stride[r]) {
                    err = P + std::to_string(e) + "." + parts[r] + ": missing, or not where the layer's others are";
                    return false;
                }
            }
            l.lay.trellis[r] = off;
            l.lay.suh[r] = l.lay.svh[r] = off;
            l.lay.K[r] = ty;
            off += align256(bytes) + (store_gb() > 0 ? ExpertStore::kSlack : 0);   // (the store's aligned reads spill into the gaps)
        }
        l.lay.fmt = gg::kFmtMXFP4;
        if (!gg::mx_experts(l.lay) && (gg::expert_type(l.lay, 0) != gg::expert_type(l.lay, 1) ||
                                       gg::expert_type(l.lay, 0) == gg::kTypeMXFP4 ||
                                       gg::expert_type(l.lay, 2) == gg::kTypeMXFP4)) {
            err = P + ": gate / up / down of types " + gg::type_name(l.lay.K[0]) + " / " + gg::type_name(l.lay.K[1]) +
                  " / " + gg::type_name(l.lay.K[2]) + " (gate and up must match; MXFP4 does not mix with the others)";
            return false;
        }
        l.blob = off;
        // a type the device kernels do not take (none of the i-quants now: gg::type_on_device lists them): the layer stays on the CPU lane, where
        // ggml-cpu's row dots read it - one warning, then the placement code keeps it out of VRAM
        for (int r = 0; r < 3; ++r)
            if (!gg::type_on_device(l.lay.K[r])) {
                l.cpu_only = true;
#if defined(STRATA_METAL)
                static bool said = false;   // (a mixed pack has many such layers: one line, not one a layer)
                if (!said) {
                    said = true;
                    std::fprintf(stderr,
                                 "ds4: this pack has %s experts: no GPU kernel for that type, every such layer's "
                                 "experts stay on the CPU lane (ggml-cpu's rows; VRAM tier: none)\n",
                                 gg::type_name(l.lay.K[r]));
                }
#else
                std::fprintf(stderr,
                             "ds4: layer %d %s is %s: no GPU kernel for it, its experts stay on the CPU lane "
                             "(ggml-cpu's rows; VRAM tier: none)\n", l.il, r == 2 ? "down" : "gate/up",
                             gg::type_name(l.lay.K[r]));
#endif
                break;
            }
        return true;
    }
    const uint8_t* mx_mat(const Layer& l, int e, int r) const {   // expert e's matrix r: its RAM-tier slot, else the file's
        if (store) {
            const int s = store->find(l.il, e);
            if (s >= 0) return store->mat(s, r);
        }
        if (!l.ram.empty() && l.ram[(size_t) e] >= 0)   // (a copy with NH > 1: part 0's rows only - whole: read_mx)
            return copy_mode ? ram_part[(size_t) l.ram[(size_t) e]][0] + cp_lay[(size_t) l.il][(size_t) r]
                             : ram_slots[(size_t) l.ram[(size_t) e]] + l.lay.trellis[r];
        return pk.at(l.mx_file[r], l.mx_off[r] + (uint64_t) e * l.mx_stride[r]);
    }
    size_t mx_part_bytes(const Layer& l, int r) const {   // (its type's)
        return r < 2 ? (size_t) g.ff * gg::type_row_bytes(gg::expert_type(l.lay, r), g.dim)
                     : (size_t) g.dim * gg::type_row_bytes(gg::expert_type(l.lay, 2), g.ff);
    }
    // (copy_mode) matrix r's rows, and the rows of one NUMA part of it
    int mx_n(int r) const { return r < 2 ? g.ff : g.dim; }
    size_t mx_row_b(const Layer& l, int r) const { return mx_part_bytes(l, r) / (size_t) mx_n(r); }
    // part X's rows [X n / NH, ..) of expert e's matrix r: its copy's part on node X, else where the file has them
    const uint8_t* mx_part_ptr(const Layer& l, int e, int r, int X) const {
        if (copy_mode && l.ram[(size_t) e] >= 0)
            return ram_part[(size_t) l.ram[(size_t) e]][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r];
        return mx_mat(l, e, r) + (size_t) X * (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
    }
    // a copy's NH parts -> the blob, and back (each matrix's rows of part X at X n / NH rows into the blob's matrix)
    void ram_to_blob(const Layer& l, int s, uint8_t* blob) const {
        for (int r = 0; r < 3; ++r) {
            const size_t nb = (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
            for (int X = 0; X < NH; ++X)
                std::memcpy(blob + l.lay.trellis[r] + (size_t) X * nb, ram_part[(size_t) s][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r], nb);
        }
    }
    void blob_to_ram(const Layer& l, const uint8_t* blob, int s) const {
        for (int r = 0; r < 3; ++r) {
            const size_t nb = (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
            for (int X = 0; X < NH; ++X)
                std::memcpy(ram_part[(size_t) s][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r], blob + l.lay.trellis[r] + (size_t) X * nb, nb);
        }
    }
    // expert e (from the file) -> its blob; and member m of nm's part of a blob (gate / up rows [m n_ff / nm, ..), down
    // inputs of the same range: in each row the blocks [m n_ff / nm / 32, ..)) and back
    bool read_mx(const Layer& l, int e, uint8_t* blob) const {   // (from its RAM-tier slot or copy when it has one)
        if (!l.ram.empty() && l.ram[(size_t) e] >= 0) {
            if (copy_mode) ram_to_blob(l, l.ram[(size_t) e], blob);
            else std::memcpy(blob, ram_slots[(size_t) l.ram[(size_t) e]], l.blob);
            return true;
        }
        return read_mx_file(l, e, blob);
    }
    bool read_mx_file(const Layer& l, int e, uint8_t* blob) const {   // (the movers: never a slot that may change)
        for (int r = 0; r < 3; ++r)
            if (!pk.read_at(l.mx_file[r], l.mx_off[r] + (uint64_t) e * l.mx_stride[r], mx_part_bytes(l, r),
                            blob + l.lay.trellis[r]))
                return false;
        return true;
    }
    void mx_blob_to_half(const Layer& l, const uint8_t* blob, int m, uint8_t* dst) const {
        const int ffh = g.ff / l.nm;
        const size_t gr = gg::type_row_bytes(gg::expert_type(l.lay, 0), g.dim),
                     dr = gg::type_row_bytes(gg::expert_type(l.lay, 2), g.ff),
                     dh = gg::type_row_bytes(gg::expert_type(l.lay, 2), ffh);
        for (int r = 0; r < 2; ++r)
            std::memcpy(dst + l.hlay.trellis[r], blob + l.lay.trellis[r] + (size_t) m * ffh * gr, (size_t) ffh * gr);
        for (size_t row = 0; row < (size_t) g.dim; ++row)
            std::memcpy(dst + l.hlay.trellis[2] + row * dh, blob + l.lay.trellis[2] + row * dr + (size_t) m * dh, dh);
    }
    void mx_half_to_blob(const Layer& l, const uint8_t* src, int m, uint8_t* blob) const {
        const int ffh = g.ff / l.nm;
        const size_t gr = gg::type_row_bytes(gg::expert_type(l.lay, 0), g.dim),
                     dr = gg::type_row_bytes(gg::expert_type(l.lay, 2), g.ff),
                     dh = gg::type_row_bytes(gg::expert_type(l.lay, 2), ffh);
        for (int r = 0; r < 2; ++r)
            std::memcpy(blob + l.lay.trellis[r] + (size_t) m * ffh * gr, src + l.hlay.trellis[r], (size_t) ffh * gr);
        for (size_t row = 0; row < (size_t) g.dim; ++row)
            std::memcpy(blob + l.lay.trellis[2] + row * dr + (size_t) m * dh, src + l.hlay.trellis[2] + row * dh, dh);
    }

    bool expert_layout(Layer& l, std::string& err) {
        if (const Tensor* t = pk.get(l.name + "ffn.experts.0.w1.weight"); t != nullptr && gg_type_of(t->dtype) > 0)
            return expert_layout_mx(l, err);
        const std::string P = l.name + "ffn.experts.";
        const char* parts[3] = {"w1", "w3", "w2"};   // gate, up, down
        const char* pieces[3] = {".trellis", ".suh", ".svh"};
        l.region_off.resize((size_t) l.n_exp);
        l.region_bytes.resize((size_t) l.n_exp);
        l.region_file.resize((size_t) l.n_exp);
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            const Tensor* tr = pk.get(P + "0." + parts[r] + ".trellis");
            if (tr == nullptr || pk.get(P + "0." + parts[r] + ".mul1") == nullptr) {
                err = P + "0." + parts[r] + ": not an EXL3 (mul1) matrix";
                return false;
            }
            l.lay.K[r] = (int) tr->shape[2] / 16;
            l.lay.trellis[r] = off;
            off += align256(tr->bytes);
            l.lay.suh[r] = off;
            off += align256(pk.get(P + "0." + parts[r] + ".suh")->bytes);
            l.lay.svh[r] = off;
            off += align256(pk.get(P + "0." + parts[r] + ".svh")->bytes);
        }
        l.lay.cb = 2;
        l.blob = off;
        for (int e = 0; e < l.n_exp; ++e) {
            const std::string E = P + std::to_string(e) + ".";
            uint64_t lo = UINT64_MAX, hi = 0;
            int file = -1;
            for (int r = 0; r < 3; ++r)
                for (int q = 0; q < 3; ++q) {
                    const Tensor* t = pk.get(E + parts[r] + pieces[q]);
                    if (t == nullptr) {
                        err = E + parts[r] + pieces[q] + ": missing";
                        return false;
                    }
                    if (file < 0) file = t->file;
                    if (t->file != file) {
                        err = E + ": split across files";
                        return false;
                    }
                    lo = std::min(lo, t->off);
                    hi = std::max(hi, t->off + t->bytes);
                    if (e == 0) l.piece_bytes[r][q] = t->bytes;
                    else if (t->bytes != l.piece_bytes[r][q]) {
                        err = E + parts[r] + ": not the shape of expert 0's";
                        return false;
                    }
                }
            l.region_off[(size_t) e] = lo;
            l.region_bytes[(size_t) e] = hi - lo;
            l.region_file[(size_t) e] = file;
            for (int r = 0; r < 3; ++r)
                for (int q = 0; q < 3; ++q) {
                    const size_t src = pk.get(E + parts[r] + pieces[q])->off - lo;
                    if (e == 0) l.piece_src[r][q] = src;
                    else if (src != l.piece_src[r][q]) {
                        err = E + ": the pieces lie differently from expert 0's";
                        return false;
                    }
                }
        }
        return true;
    }
    // the region of expert e (read from the file) -> its blob
    void repack(const Layer& l, const uint8_t* region, uint8_t* blob) const {
        for (int r = 0; r < 3; ++r) {
            const size_t dst[3] = {l.lay.trellis[r], l.lay.suh[r], l.lay.svh[r]};
            for (int q = 0; q < 3; ++q) std::memcpy(blob + dst[q], region + l.piece_src[r][q], l.piece_bytes[r][q]);
        }
    }

    // the region of expert e -> its CPU-lane part X (columns [X n / NH, (X + 1) n / NH) of each matrix)
    void repack_part(const Layer& l, const uint8_t* region, int X, uint8_t* dst) const {
        for (int r = 0; r < 3; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r];
            const int k = l.hl.k[r], nt = l.hl.n[r] * NH / 16, nth = l.hl.n[r] / 16;
            const uint8_t* tr = region + l.piece_src[r][0];
            for (int sl = 0; sl < k / 16; ++sl)
                std::memcpy(dst + l.hl.trellis[r] + (size_t) sl * nth * tile, tr + ((size_t) sl * nt + (size_t) X * nth) * tile,
                            (size_t) nth * tile);
            std::memcpy(dst + l.hl.suh[r], region + l.piece_src[r][1], (size_t) k * 2);
            std::memcpy(dst + l.hl.svh[r], region + l.piece_src[r][2] + (size_t) X * l.hl.n[r] * 2, (size_t) l.hl.n[r] * 2);
        }
    }

    // ---- loading
    bool load(const Options& opt, std::string& err) {
        o = opt;
        const double t0 = now_ms();
        if (!pk.load(o.pack, err)) return false;
        g.from(pk);
        if (g.dim == 0 || g.nl == 0 || (int) g.ratios.size() < g.nl) {
            err = o.pack + "/ds4.txt: incomplete geometry";
            return false;
        }
        if (g.hd != 512 || g.ix_hd != 128 || g.k_exp > 8 || g.hc > 8) {
            err = "geometry not covered (head_dim " + std::to_string(g.hd) + ", index_head_dim " +
                  std::to_string(g.ix_hd) + ", experts per token " + std::to_string(g.k_exp) + ")";
            return false;
        }
        freqs_win = rope_freqs(g.rd, 0, g.rope_theta, g.factor, g.beta_fast, g.beta_slow);
        freqs_comp = rope_freqs(g.rd, g.orig_ctx, g.comp_theta, g.factor, g.beta_fast, g.beta_slow);

        // devices
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
            err = "no CUDA device";
            return false;
        }
        std::vector<int> ids = o.gpus;
        if (ids.empty())
            for (int i = 0; i < ndev; ++i) ids.push_back(i);
        devs.resize(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            Dev& d = devs[i];
            d.id = ids[i];
            CK(cudaSetDevice(d.id));
            CK(cudaStreamCreateWithFlags(&d.st, cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&d.ev, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&d.ev2, cudaEventDisableTiming));
            if (!x3::prepare(d.st) || (pk.gguf && !gg::prepare(d.st))) {
                err = "exl3 scratch on CUDA" + std::to_string(d.id);
                return false;
            }
            CK(cudaMemGetInfo(&d.free0, &d.total));
            d.free0 = uma_free(d.id, d.free0);
        }
        // the groups' width: o.tp, less while the experts' blocks do not split in it (a member takes n_ff / tp of each
        // expert: the down's inputs in whole blocks - a GGUF's 256-value types split n_ff 2304 in 1, 3 or 9 parts)
        int tp = o.tp;
        if (pk.gguf && tp > 1) {
            int blk = 32;
            for (const auto& kv : pk.t) {
                const std::string& n = kv.first;
                if (n.find(".ffn.experts.") == std::string::npos || n.size() < 9 ||
                    n.compare(n.size() - 9, 9, "w2.weight") != 0)
                    continue;
                const int ty = gg_type_of(kv.second.dtype);
                if (ty >= 0) blk = std::max(blk, gg::type_block(ty));
            }
            while (tp > 1 && (g.ff % tp != 0 || (g.ff / tp) % blk != 0 || (g.ff / tp) % (std::getenv("DS4_TP_ALIGN") ? std::atoi(std::getenv("DS4_TP_ALIGN")) : 128) != 0 || g.heads % tp != 0 ||
                              g.o_groups % tp != 0))
                --tp;
            if (tp != o.tp)
                std::fprintf(stderr, "ds4: tensor parallel %d -> %d: the experts' %d-value blocks split n_ff %d no other way "
                                     "(each GPU its own layers)\n", o.tp, tp, blk, g.ff);
        }
        // the groups: runs of tp devices of one kind (their memory), a device unlike the next alone
        // - or, tp the number of GPUs, one group of all of them whatever their kind (its tightest member sets the
        // experts' room); threads mode for the windows (decode) where a group is 2 wide or more or --threads on, the
        // prompt path (its own thread a GPU) as asked for - also where the experts' blocks took the groups' width down
        // to 1 (a 256-value type's: Maya's quants), which left every prompt in windows of 16
        par = o.threads || tp > 1;
        // the chunked (layer-major) prompt path: with --tp / --threads as before; --prefill-chunk N (N > 0) or
        // kChunkedPromptDefault bring it to a single card's default run too (a host thread a GPU for the prompt only;
        // decode stays as asked).  --prefill-chunk 0 keeps the windows of 16 (DS4_WINDOWS=1 the same).  Off by default
        // until a prompt costs the decode nothing: its buffers hold VRAM the windowed path gives to experts
        prompt_par = par || o.tp > 1 ||
                     ((o.prefill_chunk > 0 || (o.prefill_chunk < 0 && kChunkedPromptDefault)) &&
                      std::getenv("DS4_WINDOWS") == nullptr);
#if defined(STRATA_METAL)
        if (std::getenv("DS4_NO_CHUNKS") == nullptr) prompt_par = true;   // the chunked prompt path on one device, the decode steps stay on the host-routed path
#endif
        const bool tp_all = tp > 1 && tp == (int) devs.size();
        for (size_t i = 0; i < devs.size();) {
            Group G;
            G.mem.push_back((int) i);
            size_t j = i + 1;
            while ((int) G.mem.size() < tp && j < devs.size() && (tp_all || devs[j].total == devs[i].total))
                G.mem.push_back((int) j++);
            if ((int) G.mem.size() < tp && G.mem.size() > 1) {   // a short run at the end: devices of their own
                for (int m : G.mem) {
                    Group one;
                    one.mem.push_back(m);
                    groups.push_back(one);
                }
            } else groups.push_back(G);
            i = j;
        }
        for (size_t gi = 0; gi < groups.size(); ++gi)
            for (size_t m = 0; m < groups[gi].mem.size(); ++m) {
                devs[(size_t) groups[gi].mem[m]].grp = (int) gi;
                devs[(size_t) groups[gi].mem[m]].mi = (int) m;
            }
        // whether the CPU lane's experts are read in place (--mmap-experts; auto: the pack's expert bytes beyond what the
        // GPUs can take - their free memory less the other weights and the prompt path's buffers - do not fit the RAM
        // this process can get beside kRamMargin), decided here because it sets the prompt chunk's default: in place
        // every chunk streams every expert VRAM does not hold from the files, so the larger the chunk the fewer times
        if (pk.gguf) file_experts = true;   // (a GGUF's: the files stay their source - copy_experts copies them into RAM
                                            // after VRAM's are in, by --mmap-experts)
        else if (o.mmap_experts >= 0) file_experts = o.mmap_experts > 0;
        else {
            double exp_b = 0, other_b = 0, room = 0;
            for (const auto& kt : pk.t) {
                const std::string& n = kt.first;
                if (n.find(".engram.embed.") != std::string::npos || n.rfind("vision.", 0) == 0 || n == "embed.weight" ||
                    (!o.dspark && n.rfind("mtp.", 0) == 0))
                    continue;
                (n.find(".ffn.experts.") != std::string::npos ? exp_b : other_b) += (double) kt.second.bytes;
            }
            for (const Dev& d : devs)
                room += std::max(0.0, (double) d.free0 - (double) ((size_t) o.reserve_mib << 20) -
                                          (prompt_par ? 1.5 : 0.4) * (1 << 30)) *
                        o.vram_experts;
            file_experts = exp_b - std::max(0.0, room - other_b) + (double) kRamMargin > (double) ram_avail();
        }
        // The default prompt chunk, one rule (ds4_experts_rule.hpp: prefill_chunk_default).  Measured: the chunked pass streams the experts the GPUs
        // do not hold once per chunk, so where they hold less than half of them a larger chunk is fewer passes (1 x R9700 + 251 GB, 8K prompt:
        // 1024 200, 2048 277, 4096 353, 8192 431 tok/s; RTX 5070 / 3060 12 GB in place: 8192 244 / 228 against 1024 186 / 175); where they hold
        // most (4 x R9700: 8192 214, 1024 370) the small one wins; with the experts copied into RAM (--mmap-experts off) 4096 beats 8192 by 7-9%
        // (H100 32 GiB: 2048 502, 4096 538, 8192 495).  A GGUF pack always has file_experts set (copied only by --mmap-experts off).
        const bool own_default = !(par || o.tp > 1);   // (a run with tp 1 and no --threads: `strata launch` writes --tp 1 for 4+ AMD cards, --tp 2 for 2-3 and nothing for one card = the engine's tp 2)
        const bool multi_tp = rdna4() && !own_default && !o.threads && devs.size() >= 2;   // (the 4-card default-tp run the 1024 chunk was also measured on; an explicit --prefill-chunk below wins over the rule)
        const bool gguf_copy = pk.gguf && (o.mmap_experts == 0 || (store_gb() <= 0.0 && auto_copy_all));
        uint64_t vmin_bytes = ~0ull;
        for (const Dev& dv : devs) vmin_bytes = std::min<uint64_t>(vmin_bytes, (uint64_t) dv.total);
        if (devs.empty()) vmin_bytes = 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        const bool hip_build = true;
#else
        const bool hip_build = false;
#endif
        const int rule_chunk = prefill_chunk_default(own_default, gpu_expert_share(), gguf_copy, file_experts, hip_build, vmin_bytes, kVlRows, multi_tp, defaults_026());
        chunk = prompt_par ? (o.prefill_chunk >= 0 ? o.prefill_chunk : rule_chunk) : 0;
        if (prompt_par && o.prefill_chunk < 0 && rule_chunk == 4096 && gguf_copy)
            std::fprintf(stderr, "ds4: prompts in chunks of 4096 tokens (the experts are copied into RAM, %.0f GiB of graphics memory; --prefill-chunk N changes it)\n",
                         (double) vmin_bytes / (double) (1ull << 30));
        const char* dc = std::getenv("DS4_CHUNK");   // (tests: down to 16, a short prompt in several chunks)
        if (dc != nullptr) chunk = prompt_par ? std::atoi(dc) : 0;
        if (const char* v = std::getenv("DS4_CHUNK_MIN")) chunk_min = std::max(1, std::atoi(v));
        if (chunk > 0) {
            chunk = std::min(kMaxChunk, std::max(dc != nullptr ? 16 : 256, chunk / 16 * 16));
            eg_cap = (size_t) chunk;
            tap_cap = std::max(kWin, 128);
        } else   // (only as asked for: --prefill-chunk 0, or --tp 1 with --threads off)
            std::fprintf(stderr,
                         "ds4: prompts run in windows of %d tokens (%s): long prompts prefill far slower than in\n"
                         "     chunks (a 7-GPU RTX 3090 rig, Maya-S read in place: 8K / 32K tokens 115 / 99 tokens/s,\n"
                         "     461 / 1010 in chunks of %d) - %s\n",
                         kWin, o.prefill_chunk == 0 ? "--prefill-chunk 0" : "--tp 1 with --threads off", kVlRows,
                         o.prefill_chunk == 0 ? "leave --prefill-chunk out for chunks"
                                              : "leave --tp 1 out, or pass --threads on, for chunks");
        // the vision tower on the first GPU, before the split: the layers get what it leaves
        if (o.vision && !load_vision(err)) {   // (a tower that cannot load leaves the model reading text: said loudly,
            if (o.vision_only) return false;   // and the server is told pictures are off - info_lines)
            std::fprintf(stderr, "ds4: WARNING: the vision tower did not load (%s): pictures are OFF, the model reads "
                                 "text only\n", err.c_str());
            vision_err = err.empty() ? "unknown reason" : err;
            err.clear();
            vt.release();
            img_tok = -1;
        }
        if (o.vision_only) return true;

        // layers: roles and the sources they read
#if defined(STRATA_METAL)
        // On a Mac the drafter is off unless it is asked for outright: it gave the same answers and no speed-up there, and the head of its
        // quantised Markov bias has no Metal kernel (the engine stopped with "kernel launch with no Metal dispatch" on the setup test).
        if (o.dspark && g.n_mtp > 0) {
            if (o.dspark_auto) {
                o.dspark = false;
                std::fprintf(stderr, "ds4: the DSpark drafter is not supported on a Mac yet, so it is not loaded\n");
            } else {   // --dspark on / off: loaded as asked
                std::fprintf(stderr, "ds4: WARNING: the DSpark drafter is not supported on a Mac yet; --dspark on will probably stop at the Markov bias step (kernel launch with no Metal dispatch)\n");
            }
        }
#endif
        if (o.dspark_auto && o.dspark && g.n_mtp > 0) {   // (see Options::dspark_auto)
            double exp_b = 0, room_b = 0;
            for (const auto& kt : pk.t)
                if (kt.first.rfind("layers.", 0) == 0 && kt.first.find(".ffn.experts.") != std::string::npos) exp_b += (double) kt.second.bytes;
            for (const Dev& d : devs) room_b += std::max(0.0, (double) d.free0 - (double) ((size_t) o.reserve_mib << 20));
            if (exp_b > 1.5 * room_b) {
                o.dspark = false;
                std::fprintf(stderr, "ds4: DSpark not loaded: the routed experts (%.0f GB) are far more than the GPU holds (%.0f GB), where its verify "
                                     "rounds read 3.3x a token's bytes and gain nothing (--dspark on loads it anyway)\n", exp_b / 1e9, room_b / 1e9);
            }
        }
        if (!o.dspark) g.n_mtp = 0;
        if (o.dspark_block > 0 && std::getenv("DS4_DSPARK_MAX") == nullptr) max_drafts = o.dspark_block;   // (--dspark-block N: at most N drafts a round)
        L.resize((size_t) (g.nl + g.n_mtp));
        int last_src = -1;
        int experts_used = 1 << 20;
        for (int il = 0; il < g.nl + g.n_mtp; ++il) {
            Layer& l = L[(size_t) il];
            l.il = il;
            l.mtp = il >= g.nl;
            l.name = l.mtp ? "mtp." + std::to_string(il - g.nl) + "." : "layers." + std::to_string(il) + ".";
            l.n_exp = l.mtp ? g.d_n_exp : g.n_exp;
            l.k_exp = l.mtp ? g.d_k : g.k_exp;
            if (!l.mtp) {   // experts a word takes: --experts N / the faster-mode rule (ds4_experts_rule.hpp), DS4_TOPK=N over both (a test)
                if (il == 0) {
                    uint64_t vram_all = 0;
                    bool integrated = false;   // (a GPU that shares the PC's RAM: Strix Halo)
                    for (const Dev& dv : devs) {
                        vram_all += (uint64_t) dv.total;
                        cudaDeviceProp ip{};
                        if (cudaGetDeviceProperties(&ip, dv.id) == cudaSuccess && ip.integrated) integrated = true;
                    }
                    // two or more RDNA4 cards holding at least half of the experts (the 4 x R9700 class): Faster by default too
                    const bool multi_rdna4 = rdna4() && devs.size() >= 2 && gpu_expert_share() >= 0.5;
                    const int n = experts_test_override(experts_per_word(o.experts, vram_all, l.k_exp, integrated, multi_rdna4, defaults_026()), l.k_exp);
                    experts_used = n;
                    if (n < l.k_exp)
                        std::fprintf(stderr, "ds4: experts per word: %d (%s; full quality: --experts %d)\n", n,
                                     o.experts >= 1 ? "as asked" : multi_rdna4 ? "faster mode: the cards hold most of the experts" : "faster mode", l.k_exp);
                    else if (o.experts >= 1)
                        std::fprintf(stderr, "ds4: experts per word: %d (full quality)\n", n);
                }
                l.k_exp = std::min(l.k_exp, experts_used);
            }
            l.ratio = g.ratios[(size_t) il];
            if (l.mtp) {   // window attention only: no compressor, indexer or Engram
                if (l.ratio != 0 || l.n_exp <= 0 || l.k_exp <= 0 || l.k_exp > kPicks) {
                    err = l.name + ": a DSpark block of compress ratio " + std::to_string(l.ratio) + ", " +
                          std::to_string(l.n_exp) + " experts / " + std::to_string(l.k_exp) + " is not covered";
                    return false;
                }
                if (!expert_layout(l, err)) return false;
                continue;
            }
            l.kv_src = std::find(g.kv_src.begin(), g.kv_src.end(), il) != g.kv_src.end();
            l.ix_src = std::find(g.ix_src.begin(), g.ix_src.end(), il) != g.ix_src.end();
            l.owns_k = l.kv_src;
            l.cand_src = il == g.cand_src;
            l.uses_cand = g.cand_src >= 0 && g.cand_src < il && l.ix_src;
            if (l.kv_src) {
                Src s;
                s.layer = il;
                s.ratio = l.ratio;
                s.cap = (o.max_context + l.ratio - 1) / l.ratio + 1;
                srcs.push_back(s);
                last_src = (int) srcs.size() - 1;
            }
            if (l.ratio > 0) {
                if (last_src < 0 || srcs[(size_t) last_src].ratio != l.ratio) {
                    err = "layer " + std::to_string(il) + ": compress ratio " + std::to_string(l.ratio) +
                          " without a KV source of that ratio before it";
                    return false;
                }
                l.src = last_src;
                l.ik_src = last_src;
            }
            auto eg = std::find(g.eg_layers.begin(), g.eg_layers.end(), il);
            if (eg != g.eg_layers.end()) l.eg = (int) (eg - g.eg_layers.begin());
            if (!expert_layout(l, err)) return false;
        }

        // A layer whose experts the chunk path's tile GEMM cannot take (IQ1_M on Metal today) would be streamed to the
        // GPU for every prompt and stop in "kernel launch with no Metal dispatch".  Prompts then run in windows of 16,
        // which is slower but correct; say so instead of failing at the first prompt.
        {
            bool chunk_unsupported = false;
            const char* why = "";
            for (const Layer& l : L) {
                for (int r = 0; r < 3; ++r)
                    if (!gg::type_on_chunk(gg::expert_type(l.lay, r))) { chunk_unsupported = true; why = gg::type_name(gg::expert_type(l.lay, r)); break; }
                if (chunk_unsupported) break;
            }
            if (chunk > 0 && chunk_unsupported) {
                chunk = 0;
                std::fprintf(stderr,
                             "ds4: the prompt chunk path has no GPU kernel for %s: prompts run in windows of %d instead "
                             "of chunks - slower prompts, and the only correct path on this build\n",
                             why, kWin);
            }
        }

        // the split: each layer's dense weights + its experts, the head on the last GPU
        for (Layer& l : L) {
            const std::string& P = l.name;
            size_t b = 0;
            if (l.mtp && l.il == g.nl) b += mat_bytes("mtp.0.main_proj");
            for (const char* m : {"attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b", "ffn.shared_experts.w1",
                                  "ffn.shared_experts.w2", "ffn.shared_experts.w3", "attn.compressor.wkv",
                                  "attn.compressor.wgate", "attn.indexer.wq_b", "attn.indexer.wk", "engram.wkv"})
                b += mat_bytes(P + m);
            for (int gi = 0; gi < g.o_groups; ++gi) b += mat_bytes(P + "attn.wo_a.slice." + std::to_string(gi));
            b += tensor_bytes(P + "hc_attn_fn") + tensor_bytes(P + "hc_ffn_fn") + tensor_bytes(P + "ffn.gate.weight");
            b += (size_t) g.win * win_rb() + (2u << 20);
            l.dense_bytes = b;
        }
        const size_t head_b = mat_bytes("head") + (size_t) g.vocab * 4 + (16u << 20);
        const size_t per_dev_fixed = (size_t) o.reserve_mib << 20;
        const size_t kv_per_src_dev = 0;   // counted after the split
        (void) kv_per_src_dev;
        std::vector<double> avail(groups.size(), 0.0);
        double avail_sum = 0;
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            for (int m : groups[gi].mem)
                avail[gi] += std::max(0.0, (double) devs[(size_t) m].free0 - (double) per_dev_fixed - 400.0 * (1 << 20));
            avail_sum += avail[gi];
        }
        double need_sum = (double) head_b;
        for (const Layer& l : L) need_sum += (double) l.dense_bytes + (double) l.blob * l.n_exp;
        {
            // contiguous layer ranges, each group's share of the total by its free memory
            size_t d = 0;
            double cum = 0, target = avail[0] / avail_sum * need_sum;
            std::vector<Layer*> order;   // the drafter's blocks first (the first GPU), then the trunk's layers
            for (Layer& l : L)
                if (l.mtp) order.push_back(&l);
            for (Layer& l : L)
                if (!l.mtp) order.push_back(&l);
            for (Layer* lp : order) {
                Layer& l = *lp;
                const double lb = (double) l.dense_bytes + (double) l.blob * l.n_exp;
                if (cum + lb / 2 > target && d + 1 < groups.size()) {
                    ++d;
                    target += avail[d] / avail_sum * need_sum;
                }
                l.grp = (int) d;
                cum += lb;
            }
            // with the chunked prompt path: the contiguous split that streams the least into its busiest member (its
            // VRAM short of its layers' experts, per member) - a chunk's groups run as a pipeline, its slowest stage
            // sets the pace; a token's decode passes every layer either way, the experts in VRAM the same in all
            if (chunk > 0 && groups.size() > 1 && groups.size() <= 6 && std::getenv("DS4_SPLIT") == nullptr) {
                const int G = (int) groups.size(), NL = g.nl;
                std::vector<double> cap(groups.size());
                for (int gi = 0; gi < G; ++gi) {
                    const int nm = (int) groups[(size_t) gi].mem.size();
                    double mn = 1e300;
                    for (int m : groups[(size_t) gi].mem)
                        mn = std::min(mn, std::max(0.0, (double) devs[(size_t) m].free0 - (double) per_dev_fixed -
                                                            400.0 * (1 << 20) - 1.1 * (1 << 30)));   // (the chunk's buffers)
                    cap[(size_t) gi] = mn * nm;
                }
                for (const Layer& l : L)
                    if (l.mtp) cap[0] -= (double) l.dense_bytes + (double) l.blob * l.n_exp;
                cap[(size_t) G - 1] -= (double) head_b;
                std::vector<double> pe(NL + 1, 0.0), pd(NL + 1, 0.0);   // prefix sums: experts, dense
                for (int il = 0; il < NL; ++il) {
                    pe[(size_t) il + 1] = pe[(size_t) il] + (double) L[(size_t) il].blob * L[(size_t) il].n_exp;
                    pd[(size_t) il + 1] = pd[(size_t) il] + (double) L[(size_t) il].dense_bytes;
                }
                auto cost = [&](int gi, int a, int b) {   // per-member streaming of layers [a, b) on group gi
                    const int nm = (int) groups[(size_t) gi].mem.size();
                    const double dense = (pd[(size_t) b] - pd[(size_t) a]) * (nm > 1 ? 1.3 : 1.0);
                    return std::max(0.0, pe[(size_t) b] - pe[(size_t) a] - (cap[(size_t) gi] - dense)) / nm;
                };
                // best[gi][b]: the least max cost of layers [0, b) over groups 0..gi (each at least one layer)
                std::vector<std::vector<double>> best((size_t) G, std::vector<double>((size_t) NL + 1, 1e300));
                std::vector<std::vector<int>> cut((size_t) G, std::vector<int>((size_t) NL + 1, 0));
                for (int b = 1; b <= NL; ++b) best[0][(size_t) b] = cost(0, 0, b);
                for (int gi = 1; gi < G; ++gi)
                    for (int b = gi + 1; b <= NL; ++b)
                        for (int a = gi; a < b; ++a) {
                            const double v = std::max(best[(size_t) gi - 1][(size_t) a], cost(gi, a, b));
                            if (v < best[(size_t) gi][(size_t) b]) {
                                best[(size_t) gi][(size_t) b] = v;
                                cut[(size_t) gi][(size_t) b] = a;
                            }
                        }
                int b = NL;
                for (int gi = G - 1; gi >= 0; --gi) {
                    const int a = gi > 0 ? cut[(size_t) gi][(size_t) b] : 0;
                    for (int il = a; il < b; ++il) L[(size_t) il].grp = gi;
                    b = a;
                }
            }
            // DS4_SPLIT=a,b,..: the trunk layers each group takes (the drafter stays with the first)
            if (const char* sp = std::getenv("DS4_SPLIT")) {
                std::vector<int> n;
                for (const char* q = sp; *q;) {
                    n.push_back(std::atoi(q));
                    while (*q && *q != ',') ++q;
                    if (*q == ',') ++q;
                }
                int il = 0;
                for (size_t gi = 0; gi < n.size() && gi < groups.size(); ++gi)
                    for (int j = 0; j < n[gi] && il < g.nl; ++j) L[(size_t) il++].grp = (int) gi;
                for (; il < g.nl; ++il) L[(size_t) il].grp = (int) groups.size() - 1;
            }
            for (Layer& l : L) {
                l.dev = groups[(size_t) l.grp].mem[0];
                l.nm = l.mtp ? 1 : (int) groups[(size_t) l.grp].mem.size();   // (the drafter's blocks: on the lead alone)
            }
            head_dev = L[(size_t) g.nl - 1].dev;
            d_dev = g.n_mtp > 0 ? L[(size_t) g.nl].dev : 0;
        }
        // the groups' members: copies of each layer's device state, a member's part of an expert
        for (Layer& l : L) {
            l.ff0 = 0;
            l.ffn = g.ff;
            l.ah0 = l.og0 = 0;
            l.ahn = g.heads;
            l.ogn = g.o_groups;
            if (l.nm == 1) continue;
            const int na = head_members(l.nm);
            if (g.heads % na != 0 || g.o_groups % na != 0 || g.ff % 128 != 0 || g.ff / 128 < l.nm) {
                err = "tensor parallel " + std::to_string(l.nm) + ": the heads, output groups or FFN do not split";
                return false;
            }
            for (int m = 1; m < l.nm; ++m) {
                Layer c = l;
                c.mates.clear();
                c.mi = m;
                c.dev = groups[(size_t) l.grp].mem[(size_t) m];
                l.mates.push_back(std::move(c));
            }
            // each member's FFN columns and its part of an expert's layout: an EXL3 expert's in whole 128-blocks, as even
            // as they go (split128, the odd blocks going round the members layer by layer); a GGUF's in even shares (its
            // groups are only as wide as its experts' blocks split evenly: mx_blob_to_half's n_ff / nm)
            const bool gguf_x = l.lay.fmt == gg::kFmtMXFP4;
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                const int ha = m - (l.nm - na);   // (its place among the members with heads)
                x.ahn = ha < 0 ? 0 : g.heads / na;
                x.ah0 = ha < 0 ? 0 : ha * x.ahn;
                x.ogn = ha < 0 ? 0 : g.o_groups / na;
                x.og0 = ha < 0 ? 0 : ha * x.ogn;
                if (gguf_x) {
                    x.ffn = g.ff / l.nm;
                    x.ff0 = m * x.ffn;
                } else
                    split128(g.ff, l.nm, m, x.ff0, x.ffn, l.il % l.nm);   // (rot: the odd blocks go round the members)
                size_t off = 0;
                for (int r = 0; r < 3; ++r) {
                    const int k = r < 2 ? g.dim : x.ffn, n = r < 2 ? x.ffn : g.dim;
                    x.hlay.K[r] = l.lay.K[r];
                    x.hlay.trellis[r] = off;
                    if (gguf_x) {   // gate / up rows, down inputs: [n][k / block] blocks
                        const int ty = gg::expert_type(l.lay, r);
                        if (k % gg::type_block(ty) != 0) {
                            err = "tensor parallel " + std::to_string(l.nm) + ": the experts' " + gg::type_name(ty) +
                                  " blocks do not split";
                            return false;
                        }
                        x.hlay.suh[r] = x.hlay.svh[r] = off;
                        off += align256((size_t) n * gg::type_row_bytes(ty, k));
                        continue;
                    }
                    off += align256(x3::trellis_bytes(k, n, l.lay.K[r]));
                    x.hlay.suh[r] = off;
                    off += align256((size_t) k * 2);
                    x.hlay.svh[r] = off;
                    off += align256((size_t) n * 2);
                }
                x.hlay.cb = l.lay.cb;
                x.hlay.fmt = l.lay.fmt;
                x.hblob = off;
            }
        }
        // which devices hold each source's compressed KV / index keys
        for (Src& s : srcs) {
            s.ckv.assign(devs.size(), nullptr);
            s.ik.assign(devs.size(), nullptr);
        }
        for (const Layer& l0 : L)
            for (int m = 0; m < l0.nm; ++m) {
                const Layer& l = member(l0, m);
                if (l.src >= 0) {
                    srcs[(size_t) l.src].ckv[(size_t) l.dev] = (uint8_t*) 1;
                    if (l0.grp != L[(size_t) srcs[(size_t) l.src].layer].grp) srcs[(size_t) l.src].remote = true;
                }
                if (l.ix_src && l.ik_src >= 0) {
                    srcs[(size_t) l.ik_src].ik[(size_t) l.dev] = (uint8_t*) 1;
                    if (l0.grp != L[(size_t) srcs[(size_t) l.ik_src].layer].grp) srcs[(size_t) l.ik_src].remote = true;
                }
            }

        // per device: the scratch, the caches, the dense weights
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            const int D = g.dim, H = g.hc;
            const size_t W = kWin, P = (size_t) kWin * kPicks, cap = (size_t) o.max_context + 64;
            bool ok = dalloc(d, d.h, W * H * D, err) && dalloc(d, d.h2, W * H * D, err) && dalloc(d, d.x, W * D, err) &&
                      dalloc(d, d.qr, W * g.q_lora, err) && dalloc(d, d.q, W * g.heads * g.hd, err) &&
                      dalloc(d, d.kvr, W * g.hd, err) && dalloc(d, d.ckr, W * g.hd, err) &&
                      dalloc(d, d.cgr, W * g.hd, err) && dalloc(d, d.latent, W * g.hd, err) &&
                      dalloc(d, d.ikr, W * g.ix_hd, err) && dalloc(d, d.iq, W * g.ix_heads * g.ix_hd, err) &&
                      dalloc(d, d.iw, W * 64, err) && dalloc(d, d.scores, W * cap, err) &&
                      dalloc(d, d.bm, W * bm_stride(), err) && dalloc(d, d.o, W * g.heads * g.hd, err) &&
                      dalloc(d, d.oa, W * g.o_groups * g.o_lora, err) && dalloc(d, d.ao, W * D, err) &&
                      dalloc(d, d.mix, W * 32, err) && dalloc(d, d.a_pre, W * 8, err) && dalloc(d, d.a_post, W * 8, err) &&
                      dalloc(d, d.a_comb, W * 64, err) && dalloc(d, d.f_pre, W * 8, err) &&
                      dalloc(d, d.f_post, W * 8, err) && dalloc(d, d.f_comb, W * 64, err) && dalloc(d, d.pre, W * 8, err) &&
                      dalloc(d, d.glog, W * 1024, err) && dalloc(d, d.glog_a, 2 * W * 1024, err) &&
                      dalloc(d, d.rw, W * kPicks, err) &&
                      dalloc(d, d.ridx, W * kPicks, err) && dalloc(d, d.ridx_a, 2 * kPicks, err) && dalloc(d, d.rw_a, 2 * kPicks, err) &&
                      dalloc(d, d.moe_out, W * D, err) &&
                      dalloc(d, d.sg, W * g.ff, err) && dalloc(d, d.su, W * g.ff, err) && dalloc(d, d.sh, W * g.ff, err) &&
                      dalloc(d, d.sh_out, W * D, err) && dalloc(d, d.cpu_part, W * D, err) &&
                      dalloc(d, d.freqs_win, 64, err) && dalloc(d, d.freqs_comp, 64, err) &&
                      dalloc(d, d.topk, W * topk_stride(), err) && dalloc(d, d.cand, W * cand_stride(), err) &&
                      dalloc(d, d.tok, 16, err) && dalloc(d, d.emb_rows, W * D, err) &&
                      dalloc(d, d.ext, ((size_t) g.win + W) * g.hd, err) && dalloc(d, d.xe, P * D, err) &&
                      dalloc(d, d.xg, P * g.ff, err) && dalloc(d, d.xu, P * g.ff, err) && dalloc(d, d.xhh, P * g.ff, err) &&
                      dalloc(d, d.ed, P * D, err) && dalloc(d, d.plan_ptr, P, err);
        if (ok && ds4_dbg("DS4_DBG_DEVH"))
            std::fprintf(stderr, "DBG devh: id=%d h=%p h2=%p\n", d.id, (void*) d.h, (void*) d.h2);
            if (!ok) return false;
            d.attn_scr.max_t = kAttnScratchT;
            d.attn_scr.cap = (int) topk_stride();
            if (!dalloc(d, d.attn_scr.rows, (size_t) kAttnScratchT * topk_stride() * g.hd, err)) return false;
            if (!g.eg_layers.empty()) {
                const int cols = (pk.eg_ngram - 1) * pk.eg_heads;
                ok = dalloc(d, d.eg_rows, W * cols * (pk.eg_hd + pk.eg_hd / 32), err) &&
                     dalloc(d, d.eg_x, W * cols * pk.eg_hd, err) && dalloc(d, d.eg_kv, W * (g.hc + 1) * D, err);
                if (!ok) return false;
            }
            if ((int) di == head_dev && !dalloc(d, d.logits, (size_t) kWin * g.vocab, err)) return false;
            if (!halloc(d.bounce, bounce_bytes(), err)) return false;
            if (!dalloc(d, d.xcnt, 4, err) || !dalloc(d, d.hccnt, kWin, err) || !hmap(d.hand, bounce_bytes(), err) || !hmap(d.hand_flag, 32, err) ||
                !dalloc(d, d.grp_ptr, P, err) || !dalloc(d, d.grp_start, P + 1, err) || !dalloc(d, d.grp_ent, P, err) ||
                !dalloc(d, d.n_groups, 4, err))
                return false;
            if (g.n_mtp > 0 && !dalloc(d, d.tap, (size_t) tap_cap * D, err)) return false;
            if (g.n_mtp > 0 && (int) di == head_dev &&
                (!dalloc(d, d_confv, kWin, err) || !dalloc(d, d_ints, 64, err) || !dalloc(d, d_specq, kWin, err)))
                return false;
            CK(cudaEventCreateWithFlags(&d.ev_out, cudaEventDisableTiming));
            cudaSetDevice(d.id);
            CK(cudaMemcpy(d.freqs_win, freqs_win.data(), freqs_win.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d.freqs_comp, freqs_comp.data(), freqs_comp.size() * 4, cudaMemcpyHostToDevice));
            for (Src& s : srcs) {
                if (s.h_ckv == nullptr) {   // (its events belong to the source layer's device: it records them)
                    if (!halloc(s.h_ckv, (size_t) kWin * ckv_rb(), err) || !halloc(s.h_ik, (size_t) kWin * ik_rb(), err))
                        return false;
                    CK(cudaSetDevice(devs[(size_t) L[(size_t) s.layer].dev].id));
                    CK(cudaEventCreateWithFlags(&s.ev_ckv, cudaEventDisableTiming));
                    CK(cudaEventCreateWithFlags(&s.ev_ik, cudaEventDisableTiming));
                    CK(cudaSetDevice(d.id));
                }
                if (s.ckv[di] != nullptr && !dalloc(d, s.ckv[di], (size_t) s.cap * ckv_rb(), err, &d.caches))
                    return false;
                if (s.ik[di] != nullptr && !dalloc(d, s.ik[di], (size_t) s.cap * ik_rb(), err, &d.caches))
                    return false;
            }
        }
        for (Src& sr : srcs) {
            if (sr.remote && (!hmap(sr.x_stage, x_cap() * (ckv_rb() + ik_rb()), err) || !hmap(sr.x_flag, 32, err)))
                return false;
            if (sr.remote && chunk > 0 &&
                (!hmap(sr.x_stage1, x_cap() * (ckv_rb() + ik_rb()), err) || !hmap(sr.x_flag1, 32, err) ||
                 !hmap(sr.x_ack, (size_t) 2 * devs.size() * 32, err)))
                return false;
            const int sg = L[(size_t) sr.layer].grp;
            for (size_t di = 0; di < devs.size(); ++di)
                if ((sr.ckv[di] != nullptr || sr.ik[di] != nullptr) && devs[di].grp != sg) sr.readers.push_back((int) di);
        }
        for (Group& G : groups) {
            G.ar.nm = (int) G.mem.size();
            G.ar.slots = 4;
            G.ar.stride = (size_t) kWin * g.dim;
            if (G.ar.nm > 1 && (!hmap(G.ar.data, (size_t) G.ar.nm * G.ar.slots * G.ar.stride, err) ||
                                !hmap(G.ar.flags, (size_t) G.ar.nm * G.ar.slots * 32, err)))
                return false;
            // three members or more: the reduce-scatter's sums and its per-block flags (allreduce)
            if (G.ar.nm > 2 && (!hmap(G.ar.sums, (size_t) G.ar.slots * G.ar.stride, err) ||
                                !hmap(G.ar.bflags, (size_t) 2 * G.ar.nm * G.ar.slots * kArBlocks * 8, err)))
                return false;
            if (G.ar.nm > 1 && &G == &groups[(size_t) devs[(size_t) head_dev].grp] &&
                (!hmap(G.lg_stage, (size_t) 2 * (G.ar.nm - 1) * kWin * ((g.vocab / 128 + G.ar.nm - 1) / G.ar.nm * 128), err) ||
                 !hmap(G.lg_flag, (size_t) 2 * (G.ar.nm - 1) * 32, err) || !hmap(G.lg_ack, (size_t) 2 * 32, err)))
                return false;
            G.arb = G.ar;   // (a chunk's: two slots of chunk rows)
            G.arb.slots = 2;
            G.arb.stride = (size_t) chunk * g.dim;
            if (chunk > 0 && G.arb.nm > 1 && (!hmap(G.arb.data, (size_t) G.arb.nm * G.arb.slots * G.arb.stride, err) ||
                                              !hmap(G.arb.flags, (size_t) G.arb.nm * G.arb.slots * 32, err)))
                return false;
            if (chunk > 0 && G.arb.nm > 2 &&
                (!hmap(G.arb.sums, (size_t) G.arb.slots * G.arb.stride, err) ||
                 !hmap(G.arb.bflags, (size_t) 2 * G.arb.nm * G.arb.slots * kArBlocks * 8, err)))
                return false;
        }
        if (g.n_mtp > 0) {
            if (!halloc(h_taps, (size_t) tap_cap * g.d_targets.size() * g.dim, err)) return false;
            CK(cudaSetDevice(devs[(size_t) d_dev].id));
            CK(cudaEventCreateWithFlags(&ev_seed, cudaEventDisableTiming));
            CK(cudaEventRecord(ev_seed, devs[(size_t) d_dev].st));
        }
        // the dense weights, each device's layers by a thread of its own
        {
            std::vector<std::string> errs(devs.size());
            std::vector<std::thread> th;
            for (size_t di = 0; di < devs.size(); ++di)
                th.emplace_back([this, di, &errs] {
                    std::string e;
                    if (!load_dense((int) di, e)) errs[di] = e;
                });
            for (auto& t : th) t.join();
            for (auto& e : errs)
                if (!e.empty()) {
                    err = e;
                    return false;
                }
        }
        if (o.verbose)
            std::fprintf(stderr, "ds4: dense weights loaded in %.1f s\n", (now_ms() - t0) / 1000.0);
        // the markov steps over a head group of two or more (DS4_MK_SPLIT=0: the lead's whole rows, as before): E
        // host-mapped, the steps' slots
        // (only where every member holds its rows of the markov head: a type load_dense does not take keeps the lead's)
        bool mk_rows = g.n_mtp > 0 && head_nm() > 1;
        for (int m : groups[(size_t) devs[(size_t) head_dev].grp].mem) mk_rows = mk_rows && devs[(size_t) m].mk_share != nullptr;
        if (mk_rows && !(std::getenv("DS4_MK_SPLIT") != nullptr && std::atoi(std::getenv("DS4_MK_SPLIT")) == 0)) {
            const Tensor* te = pk.get("mtp." + std::to_string(g.n_mtp - 1) + ".markov_head.embed.weight");
            if (te == nullptr || te->dtype != "BF16" || te->bytes != (uint64_t) g.vocab * g.d_rank * 2) {
                err = "the markov embed: not a BF16 [vocab][rank] table";
                return false;
            }
            if (!hmap(h_mk_embed, (size_t) g.vocab * g.d_rank, err) || !pk.read(*te, h_mk_embed) ||
                !hmap(mk_slots, (size_t) kWin * head_nm() * 4, err)) {
                if (err.empty()) err = "the markov embed: read failed";
                return false;
            }
            devs[(size_t) head_dev].mk_ints = d_ints;   // (the lead's drafted tokens: where the confidence head reads them)
            mk_split = true;
        }

        // the embedding table, pinned on the host
        {
            const Tensor* t = pk.get("embed.weight");
            const size_t nv = (size_t) g.vocab * g.dim;
            const bool q8 = t != nullptr && t->dtype == "Q8_0" && t->bytes == nv / 32 * gg::kQ8Bytes;
            // other block types (a GGUF whose dense tensors are Q4_K / Q5_K / Q6_K: the IST-DAS sets' Q5_K embed.weight):
            // each row through ggml's to_float, to the same BF16 the host table holds
            const int ety = t != nullptr ? gg_type_of(t->dtype) : -1;
            const size_t erb = ety > 0 ? gg::type_row_bytes(ety, g.dim) : 0;
            const bool qrow = t != nullptr && !(t->dtype == "BF16") && erb > 0 && t->bytes == (size_t) g.vocab * erb;
            if (t == nullptr || (!q8 && !qrow && (t->dtype != "BF16" || t->bytes != nv * 2))) {
                err = "embed.weight: not a BF16 (or Q8_0 / a ggml block type) [vocab][dim] table";
                return false;
            }
            if (!halloc(embed_pinned, nv, err)) return false;
            if (qrow) {   // (ggml's decode of each row, rounded to BF16 as the Q8_0 path rounds)
                std::vector<uint8_t> row(erb);
                for (size_t r = 0; r < (size_t) g.vocab; ++r) {
                    if (!pk.read_at(t->file, t->off + r * erb, erb, row.data())) {
                        err = "embed.weight: read failed";
                        return false;
                    }
                    if (!gg::gg_row_to_bf16(ety, row.data(), g.dim, embed_pinned + r * g.dim)) {
                        err = "embed.weight: ggml has no decode for " + t->dtype;
                        return false;
                    }
                }
            } else if (q8) {   // (a GGUF's: each value rounded to BF16, a row at a time)
                std::vector<uint8_t> row((size_t) g.dim / 32 * gg::kQ8Bytes);
                for (size_t r = 0; r < (size_t) g.vocab; ++r) {
                    if (!pk.read_at(t->file, t->off + r * row.size(), row.size(), row.data())) {
                        err = "embed.weight: read failed";
                        return false;
                    }
                    const std::vector<float> v = q8_values(row.data(), (size_t) g.dim);
                    for (size_t c = 0; c < (size_t) g.dim; ++c) {
                        uint32_t u;
                        std::memcpy(&u, &v[c], 4);
                        u += 0x7fffu + ((u >> 16) & 1u);   // round to nearest even
                        embed_pinned[r * g.dim + c] = (uint16_t) (u >> 16);
                    }
                }
            } else if (!pk.read(*t, embed_pinned)) {
                err = "embed.weight: read failed";
                return false;
            }
            if (ds4_dbg("DS4_DBG_EMB")) std::fprintf(stderr,
                    "DBG emb pinned: %04x %04x %04x %04x  (row1: %04x %04x)\n",
                    (unsigned) embed_pinned[0], (unsigned) embed_pinned[1],
                    (unsigned) embed_pinned[2], (unsigned) embed_pinned[3],
                    (unsigned) embed_pinned[(size_t) g.dim], (unsigned) embed_pinned[(size_t) g.dim + 1]);

        }
        // the host side of the CPU lane and the Engram rows
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads;
        if (!halloc(h_x, (size_t) kWin * g.dim, err) || !halloc(h_w, (size_t) kWin * kPicks, err) ||
            !halloc(h_part, (size_t) kWin * g.dim, err) || !halloc(h_idx, (size_t) kWin * kPicks, err) || !halloc(h_idx_a, (size_t) 2 * kPicks, err) ||
            !halloc(h_emb, (size_t) std::max(kWin, chunk) * g.dim, err) ||
            !halloc(h_eg, (size_t) std::max(1, pk.eg_layers) * eg_cap * cols * (pk.eg_hd + pk.eg_hd / 32) + 64, err))
            return false;
        if (pk.eg_layers > 0) {
            // the Engram rows' reader: threads enough to keep an SSD's queue busy (reads from the page cache need few);
            // the row cache a small share of the RAM free (DS4_ENGRAM_CACHE_MB: its size, 0 off; DS4_ENGRAM_THREADS)
            const size_t row = (size_t) pk.eg_hd + pk.eg_hd / 32;
#if defined(_WIN32)
            // (Windows: each thread's ReadFile is synchronous, so the threads are the queue depth - Project Maya: a 28K-token
            //  prompt in one chunk, 10 threads 17.1 s, 64 13.4 s on a 20-core PC.  Here (6 cores, shipped profile) 64 readers
            //  gave +4.5% on a 4K prompt but no decode gain in every pair: DS4_ENGRAM_THREADS=64 turns them on)
#endif
            const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
            int nth = (int) std::min(16u, std::max(4u, hw / 2));
#if defined(STRATA_METAL)
            // A Mac reads the Engram rows with two threads: with more, a decode of the same prompt gave slightly different words run to run
            // (3 to 9 runs of 16 differed from the serial reference, against none of 24 with one or two threads; the cause is not found yet).  DS4_ENGRAM_THREADS=N sets another number.
            nth = 2;
            if (std::getenv("DS4_ENGRAM_THREADS") == nullptr) std::fprintf(stderr, "ds4: Engram rows are read with 2 threads (DS4_ENGRAM_THREADS=N sets another number)\n");
#endif
            if (const char* e = std::getenv("DS4_ENGRAM_THREADS")) nth = std::max(1, std::atoi(e));
            const uint64_t avail = ram_avail();
            uint64_t mb = avail == ~uint64_t{0} ? 64 : std::min<uint64_t>(256, std::max<uint64_t>(16, (avail >> 20) / 64));
            if (const char* e = std::getenv("DS4_ENGRAM_CACHE_MB")) mb = std::strtoull(e, nullptr, 10);
            eg_cache.init((size_t) ((mb << 20) / row), row);
            std::vector<std::string> paths;
            for (const std::string& fn : pk.files) paths.push_back(pk.model_dir + "/" + fn);
            // Windows: the Engram tables' files that hold nothing else are unmapped and read unbuffered - an SSD's
            // queue kept by the reader threads instead of the page cache's (whose reads of one file from many threads
            // went ~255K rows a second here); unbuffered reads of a mapped file take a slow path, so only unmapped
            // ones (off by default: -8% on a 4K prompt here; DS4_EG_DIRECT=1 turns it on)
            std::vector<char> eg_direct(pk.files.size(), 0);
#if defined(_WIN32)
            if (pk.gguf && std::getenv("DS4_EG_DIRECT") != nullptr && std::atoi(std::getenv("DS4_EG_DIRECT")) != 0) {
                std::vector<char> eg_file(pk.files.size(), 0), other(pk.files.size(), 0);
                for (const auto& kt : pk.t)
                    if (kt.second.file >= 0 && (size_t) kt.second.file < pk.files.size())
                        (kt.first.find(".engram.embed.") != std::string::npos ? eg_file : other)[(size_t) kt.second.file] = 1;
                for (size_t i = 0; i < pk.files.size(); ++i)
                    if (eg_file[i] && !other[i]) {
                        if (i < pk.maps.size() && pk.maps[i].first != nullptr) {
                            os::unmap(pk.maps[i].first, pk.maps[i].second);
                            pk.maps[i] = {nullptr, 0};
                        }
                        eg_direct[i] = 1;
                        if (o.verbose) std::fprintf(stderr, "ds4: %s unmapped, its Engram rows read unbuffered\n",
                                                    pk.files[i].c_str());
                    }
            }
#endif
            eg_io.start(paths, nth, eg_direct);
            for (int li = 0; li < pk.eg_layers; ++li) {
                eg_win.emplace_back(new EgBatch());
                const std::string id = std::to_string(g.eg_layers[(size_t) li]);
                const Tensor* tw = pk.get("layers." + id + ".engram.embed.weight");
                // The row layout the reader and the dequant take.  A type not here (or no tensor at all) would be read
                // as some other format's bytes - the old code then read the scale tensor of a pack that has none and
                // crashed in eg_issue - so refuse the pack once, with the type's name, before any token is read.
                int fmt;
                if (tw == nullptr) {
                    std::fprintf(stderr, "ds4: Engram layer %s: engram.embed.weight is missing from the pack\n", id.c_str());
                    std::exit(2);
                } else if (tw->dtype == "MXFP8") {
                    fmt = 0;
                } else if (tw->dtype == "Q6_K") {
                    fmt = 1;
                } else if (tw->dtype == "IQ4_NL") {
                    fmt = 2;
                } else if (pk.get("layers." + id + ".engram.embed.scale") != nullptr) {
                    fmt = 3;   // (an old pack: FP8 codes in one tensor, their E8M0 scales in another)
                } else {
                    std::fprintf(stderr,
                                 "ds4: Engram layer %s: engram.embed.weight type %s is not one the Engram reader handles "
                                 "(MXFP8, Q6_K, IQ4_NL)\n",
                                 id.c_str(), tw->dtype.c_str());
                    std::exit(2);
                }
                if (fmt == 1 && pk.eg_hd % 256 != 0) {
                    err = "Engram layer " + id + ": Q6_K rows of " + std::to_string(pk.eg_hd) +
                          " values (not whole superblocks of 256)";
                    return false;
                }
                if (fmt == 2 && pk.eg_hd % 32 != 0) {
                    err = "Engram layer " + id + ": IQ4_NL rows of " + std::to_string(pk.eg_hd) +
                          " values (not whole blocks of 32)";
                    return false;
                }
                eg_fmt.push_back((uint8_t) fmt);
            }
            if (o.verbose)
                std::fprintf(stderr, "ds4: Engram rows read on %d threads, issued with the tokens, collected before "
                                     "their layer; row cache %llu MB (%zu rows)\n",
                             nth, (unsigned long long) mb, eg_cache.capacity());
        }
        prof.assign((size_t) (g.nl + g.n_mtp) * g.n_exp, 0u);
        if (!o.profile.empty()) {
            std::FILE* f = std::fopen(o.profile.c_str(), "rb");
            if (f != nullptr) {
                std::vector<uint32_t> p(prof.size());
                const size_t n = std::fread(p.data(), 4, p.size(), f);
                if (n == p.size() || n == (size_t) g.nl * g.n_exp) std::copy(p.begin(), p.begin() + (long) n, prof.begin());
                std::fclose(f);
            }
        }

        // the chunked prompt path's buffers (before the experts: they take what is left)
        if (chunk > 0) {
            // recommend, never force: a chunk the cards cannot hold in buffers is halved (down to 256 rows) and then the
            // prompt runs in windows of 16, said in one line - the load only fails for a reason that is not this
            std::vector<size_t> na0, sc0;
            for (Dev& d : devs) {
                na0.push_back(d.allocs.size());
                sc0.push_back(d.scratch);
            }
            const size_t nh = host_allocs.size();
            std::string aerr;
            while (chunk > 0 && !alloc_chunks(aerr)) {
                for (size_t di = 0; di < devs.size(); ++di) {
                    Dev& d = devs[di];
                    cudaSetDevice(d.id);
                    cudaDeviceSynchronize();
                    if (di < chunks.size() && chunks[di]) {
                        if (chunks[di]->blas) cublasDestroy(chunks[di]->blas);
                        if (chunks[di]->cs) cudaStreamDestroy(chunks[di]->cs);
                    }
                    while (d.allocs.size() > na0[di]) {
                        cudaFree(d.allocs.back());
                        d.allocs.pop_back();
                    }
                    d.scratch = sc0[di];
                }
                while (host_allocs.size() > nh) {
                    cudaFreeHost(host_allocs.back());
                    host_allocs.pop_back();
                }
                chunks.clear();
                if (slab_auto_grew_) {   // the automatic slab did not fit: the 0.2.6 slab first, the chunk only if that does not fit either
                    slab_auto_grew_ = false;
                    slab_force_base_ = true;
                    std::fprintf(stderr, "ds4: the automatic attention slab did not fit (%s): trying the 256-row slab\n", aerr.c_str());
                    aerr.clear();
                    continue;
                }
                const int was = chunk;
                chunk = chunk > 512 ? chunk / 2 / 16 * 16 : 0;
                if (chunk > 0) {
                    eg_cap = (size_t) chunk;
                    std::fprintf(stderr, "ds4: the prompt path's buffers for chunks of %d rows did not fit (%s): trying %d\n",
                                 was, aerr.c_str(), chunk);
                } else {
                    prompt_par = false;
                    eg_cap = (size_t) kWin;
                    tap_cap = kWin;
                    std::fprintf(stderr, "ds4: the prompt path's buffers did not fit (%s): prompts run in windows of %d "
                                         "tokens (slower on long prompts; more free GPU memory or a lower --reserve-mib "
                                         "brings the chunks back)\n", aerr.c_str(), kWin);
                }
                aerr.clear();
            }
            if (chunk > 0 && aerr.empty()) err.clear();
        }
        // the CPU lane's experts read in place (decided above): how much of them the load reads ahead, from what VRAM
        // now leaves over
        if (file_experts) {
            double room = 0, all = 0;   // the VRAM place_experts will find for experts; every expert's bytes
            for (Dev& d : devs) {
                cudaSetDevice(d.id);
                size_t fr = 0, tot = 0;
                CK(cudaMemGetInfo(&fr, &tot));
                fr = uma_free(d.id, fr);
                room += std::max(0.0, (double) fr - (double) ((size_t) o.reserve_mib << 20)) * o.vram_experts;
            }
            for (const Layer& l : L) all += (double) l.blob * l.n_exp;
            const double need = std::max(0.0, all - room);
            const uint64_t avail = ram_avail();
            warm_bytes = avail > kRamMargin ? std::min<uint64_t>(avail - kRamMargin, (uint64_t) need) : 0;
            if (avail == ~uint64_t{0}) warm_bytes = (uint64_t) need;
            if (store_gb() > 0) warm_bytes = 0;   // (the store is the RAM tier: nothing to warm in the OS cache)
            std::fprintf(stderr,
                         "ds4: the experts VRAM leaves over (~%.1f GB) are read in place from the model's files%s "
                         "(%.1f GB of RAM free to keep them: the most-routed %.1f GB read ahead; prompts in chunks of %d)\n",
                         need / 1e9, o.mmap_experts > 0 ? " (--mmap-experts on)"
                                     : pk.gguf ? " (--mmap-experts off copies them into RAM)" : ", not copied into RAM",
                         avail == ~uint64_t{0} ? 0.0 : avail / 1e9, warm_bytes / 1e9, chunk);
        }
        // the CPU lane: one pool of workers per NUMA node, each expert split in NH column parts (part X on node X); in
        // place, one part (the files' layout) and one pool over every node's cores (the pages lie where the cache put them)
        cores = node_cores();
        NH = file_experts ? 1 : (int) std::min<size_t>(2, cores.size());
        if (NH > 1 && ((g.ff / NH) % 128 != 0 || (g.dim / NH) % 128 != 0)) NH = 1;
        if (!pk.gguf) make_pools();   // (a GGUF's after its experts: copy_experts decides NH)
        {
            const size_t E = (size_t) g.dim, FF = (size_t) g.ff, GQ = E / 256, DQ = FF / 128, S = kSlots;
            c_xh.resize((size_t) NH * S * 2 * E);
            c_xsb.resize((size_t) NH * S * 2 * (E / 128));
            c_acc.resize((size_t) NH * S * 2 * GQ * (FF / NH));
            c_xhd.resize(S * FF);
            c_xsd.resize(S * (FF / 128));
            c_accd.resize((size_t) NH * S * DQ * (E / NH));
            c_yd.resize(S * E);
        }
        // the experts: VRAM slots in what each device has left, the rest on the host
        if (!place_experts(err)) return false;
        if (!load_experts(err)) return false;
        if (pk.gguf) make_pools();
        if (chunk > 0) {
            // the CPU lane's parts pinned (the chunks stream them to the GPUs by DMA), each layer's virtual slots
            const double tr = now_ms();
            for (auto& m : host_maps)
                if (cudaHostRegister(m.first, m.second, cudaHostRegisterPortable) != cudaSuccess) {
                    err = "pinning the CPU lane's experts (" + std::to_string(m.second >> 20) + " MiB) failed";
                    return false;
                }
            if (o.verbose) std::fprintf(stderr, "ds4: CPU lane's experts pinned in %.1f s\n", (now_ms() - tr) / 1000.0);
            if (file_experts) {   // in place: each device's staging for its ring, and its readers (DS4_READERS, 8)
                size_t mr = 0;
                for (const Layer& l : L)
                    if (!l.mtp) {
                        for (uint64_t r : l.region_bytes) mr = std::max<size_t>(mr, r);
                        if (l.lay.fmt == gg::kFmtMXFP4) {   // (a GGUF's: read as its blob)
                            mr = std::max(mr, l.blob);
                            for (int r = 0; r < 3; ++r) flow_part[r] = std::max(flow_part[r], mx_part_bytes(l, r));
                        }
                    }
                const char* nr = std::getenv("DS4_READERS");
                const int n_readers = std::max(1, nr != nullptr ? std::atoi(nr) : 8);
                const bool flow_on = std::getenv("DS4_FLOW") == nullptr || std::atoi(std::getenv("DS4_FLOW")) != 0;
                // (a tensor-parallel group's members on the flow only with DS4_FLOW_TP=1 for now: each reads the
                // whole down of every expert it streams, 25 MB an expert for a pair, 18.8 alone - 3x V100, the
                // 17-token prompt from the page cache: 2.5-2.7 s on the flow, 1.3-1.6 s in the group's windows)
                const bool flow_tp = std::getenv("DS4_FLOW_TP") != nullptr && std::atoi(std::getenv("DS4_FLOW_TP")) != 0;
                for (size_t di = 0; di < chunks.size(); ++di) {
                    auto& cp = chunks[di];
                    if (cp == nullptr || cp->ring == nullptr) continue;
                    cp->stage_stride = (mr + 4095) & ~(size_t) 4095;
                    // the flow: every trunk layer of the device a GGUF's (alone or a tensor-parallel group's member)
                    bool fl = flow_on;
                    int n_exp_max = 0;
                    for (const Layer& l : L)
                        if (!l.mtp)
                            for (int m = 0; m < l.nm; ++m)
                                if (member(l, m).dev == (int) di) {
                                    // (a group's members: each its share of the down's blocks in every row)
                                    fl = fl && l.lay.fmt == gg::kFmtMXFP4 &&
                                         (l.nm == 1 || (flow_tp && (g.ff / l.nm) %
                                                        std::max(1, gg::type_block(gg::expert_type(l.lay, 2))) == 0));
                                    n_exp_max = std::max(n_exp_max, l.n_exp);
                                }
                    if (fl)   // (each part read unbuffered from its offset rounded down: a page more each)
                        cp->stage_stride = std::max(cp->stage_stride, flow_stride());
                    cp->S = cp->ring_n;
                    if (!halloc(cp->stage, (size_t) cp->S * cp->stage_stride, err)) return false;
                    cp->readers = std::make_unique<Pool>(n_readers - 1, std::vector<int>{}, 50);
                    if (!fl) continue;
                    cudaSetDevice(devs[di].id);
                    cp->flow = true;
#if defined(STRATA_METAL)
                    cp->direct = std::getenv("DS4_FLOW_DIRECT") == nullptr || std::atoi(std::getenv("DS4_FLOW_DIRECT")) != 0;
#endif
                    cp->dev = (int) di;
                    cp->io = std::make_unique<IoQueue>(n_readers);
                    cp->stage_item.assign((size_t) cp->S, -1);
                    cp->stage_ev.resize((size_t) cp->S);
                    cp->ring_fill.resize((size_t) cp->ring_n);
                    cp->ring_free.resize((size_t) cp->ring_n);
                    for (auto* v : {&cp->stage_ev, &cp->ring_fill, &cp->ring_free})
                        for (cudaEvent_t& e : *v) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                    // (a region is written again only bc_n batches later - more than a layer's items, so after that
                    // layer's bounds wait, which every copy from it precedes)
                    cp->bc_n = n_exp_max + 2;
                    cp->bc_w = cp->ring_n + 1;
                    if (!halloc(cp->bc_pin, (size_t) cp->bc_n * cp->bc_w, err) ||
                        !dalloc(devs[di], cp->bc_dev, (size_t) cp->bc_n * cp->bc_w, err))
                        return false;
                }
                // every device with layers on the flow: a prompt reads only the experts it picks, once, so a short one
                // takes the chunked path too (one pass, where windows of 16 read the CPU lane's picks again for each
                // window, a page fault at a time) - DS4_CHUNK_MIN still overrides
                bool all_flow = true;
                for (const auto& cp : chunks) all_flow = all_flow && (cp == nullptr || cp->ring == nullptr || cp->flow);
                if (all_flow && std::getenv("DS4_CHUNK_MIN") == nullptr) chunk_min = kFlowChunkMin;
            }
            for (Layer& l : L) {
                if (l.mtp) continue;
                for (int m = 0; m < l.nm; ++m) {   // (each member its own: the flow's members decide them each)
                    Layer& x = member(l, m);
                    if (!halloc(x.vs_pin, (size_t) l.n_exp, err)) return false;
                    if (!dalloc(devs[(size_t) x.dev], x.vs_dev, (size_t) l.n_exp, err)) return false;
                }
            }
        }
        {
            void* m = nullptr;
            if (cudaHostAlloc(&m, sizeof(Req) * kRing, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                err = "host-mapped memory for the decode path's requests";
                return false;
            }
            host_allocs.push_back(m);
            reqs = (Req*) m;
            if (cudaHostAlloc(&m, sizeof(Resp) * kRing, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                err = "host-mapped memory for the decode path's answers";
                return false;
            }
            host_allocs.push_back(m);
            resps = (Resp*) m;
            std::memset((void*) reqs, 0, sizeof(Req) * kRing);
            std::memset((void*) resps, 0, sizeof(Resp) * kRing);
            for (auto& x : layer_of) x.store(-1);
            ahead_fl.assign(L.size() * 1024, 0);
            ahead_rank.assign(L.size() * 1024, 0);
            ahead_for.assign(L.size(), 0);
            const int core = !cores.empty() && !cores[0].empty() ? cores[0][0] : -1;
            service = std::thread([this, core] {
                if (core >= 0) pin_self(core);
                serve_lane();
            });
            if (ahead_mode > 0) pf_th = std::thread([this] { prefetch_loop(); });
        }
        if (o.expert_cache) {
            size_t maxb = 0;
            for (const Layer& l : L) maxb = std::max(maxb, l.blob);
            const int nt = std::max(1, std::min(c_inflight, std::getenv("DS4_CACHE_THREADS") ? std::atoi(std::getenv("DS4_CACHE_THREADS")) : 1));
            sw_bufs.resize((size_t) nt);
            for (SwapBufs& B : sw_bufs) {
                if (!halloc(B.a, maxb, err) || !halloc(B.b, maxb, err) || !halloc(B.h, maxb, err)) return false;
                for (Dev& d : devs) {
                    cudaSetDevice(d.id);
                    cudaStream_t cs = nullptr;
                    CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
                    B.st.push_back(cs);
                }
            }
            for (SwapBufs& B : sw_bufs) cache_ths.emplace_back([this, &B] { cache_loop(B); });
        }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            CK(cudaStreamSynchronize(d.st));
        }
        if (par || chunk > 0)   // (the windows' threads; the prompt path's, whatever mode the windows take)
            for (size_t di = 0; di < devs.size(); ++di) {
                workers.push_back(std::make_unique<Worker>());
                workers.back()->start();
            }
        calibrate_paths();
        init_store();      // (last: the store's arena is registered with the driver - page-locked memory the other pinned allocations must come before)
        preload_store();
        // On by default for a Windows HIP build on an APU (Strix Halo: measured +7.5% decode, no change in the prompt); DS4_NO_VIEW=1 forces it
        // on anywhere, DS4_NO_VIEW=0 turns it off
        bool no_view = std::getenv("DS4_NO_VIEW") != nullptr && std::atoi(std::getenv("DS4_NO_VIEW")) != 0;
#if defined(_WIN32) && defined(STRATA_USE_HIP)
        if (std::getenv("DS4_NO_VIEW") == nullptr && !devs.empty()) {
            cudaDeviceProp vp{};
            no_view = cudaGetDeviceProperties(&vp, devs[0].id) == cudaSuccess && vp.integrated != 0;
        }
#endif
        if (no_view && store && pk.gguf) {
            pk.drop_views();
            store->set_pfds(pk.pfds);
            views = false;
            std::fprintf(stderr, "ds4: no view of the model's files after the load: unbuffered reads at the drive's rate\n");
        }
        if (o.verbose) std::fprintf(stderr, "ds4: loaded in %.1f s\n%s", (now_ms() - t0) / 1000.0, describe().c_str());
        if (chunk > 0) (void) pick_flow_min();   // (the startup line)
        reset();
        start_warm();
        return true;
    }


    // the caches' row bytes, packed as DeepSeek keeps them (ds4_kernels.cuh): the window ring's (FP8), the compressed
    // KV's (FP4 + E4M3 scales), the index keys' (FP4 + power-of-two scales)
    size_t win_rb() const { return kv_row_bytes(kKvFp8, g.hd); }
    size_t ckv_rb() const { return kv_row_bytes(kKvFp4, g.hd); }
    size_t ik_rb() const { return kv_row_bytes(kKvFp4p, g.ix_hd); }

    // The chunked prompt's first GEMM of each kind pays the BLAS library's start (its handle's heuristics, the code objects of its
    // kernels loaded on first use): ~1 s of hipModuleLoad plus ~2.5 s of library time on the R9700 (rocprofv3), 1-2 s on CUDA, once a
    // process - 3.6 s on the first prompt that went through the flow, measured as 4.9 s for a 25-token pass against 1.0 s for the
    // next one.  Paid here instead, at load, on a thread beside the store's read: the chunk GEMMs on tiny inputs into the chunk's own
    // scratch buffers (the mHC mix GEMM in F32, the FP16 GEMM the projections and the indexer weights use) - no engine state is
    // touched (heat, profile, stats, positions, KV), so the answers are what they were.  DS4_WARM=0 turns it off.
    void warm_flow() {
        const int T = 16, D = g.dim, hcd = g.hc * g.dim, n_mix = (2 + g.hc) * g.hc;
        const float one = 1.0f, zero = 0.0f;
        for (size_t di = 0; di < chunks.size(); ++di) {
            if (!chunks[di] || chunks[di]->blas == nullptr) continue;
            Chunk& c = *chunks[di];
            if (c.C < T) continue;
            cudaSetDevice(devs[di].id);
            (void) cublasSgemm(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n_mix, T, hcd, &one, c.h, hcd, c.h, hcd, &zero, c.mix, 32);
            const int ns[2] = {256, 32};
            for (int n : ns)
                (void) cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, T, D, &one, c.w16, CUDA_R_16F, D, c.x16, CUDA_R_16F, D, &zero,
                                    c.y, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            (void) cudaStreamSynchronize(devs[di].st);
            (void) cudaGetLastError();
        }
    }
    // The warm-up runs once the model is loaded, on a thread of its own: HIP serialises the BLAS start with the store's page-locked
    // arena (measured: beside the store's read it finished only when the read did, 21 s, and load still grew by its 3.5 s), so
    // it cannot hide inside the load; after it, a person typing gives it the seconds it needs.  A chunked pass that starts first
    // waits for it (warm_mu) - never later than that pass would have been without it - and a warm-up that finds the flow already
    // used does nothing.  DS4_WARM=0 turns it off.
    std::mutex warm_mu;
    std::atomic<bool> flow_used{false};
    std::thread warm_bg;
    void start_warm() {
        if (chunk <= 0 || (std::getenv("DS4_WARM") && std::atoi(std::getenv("DS4_WARM")) == 0)) return;
        warm_bg = std::thread([this] {
            std::lock_guard<std::mutex> lk(warm_mu);
            if (flow_used.load() || quit.load()) return;
            const double tw = now_ms();
            warm_flow();
            if (o.verbose) std::fprintf(stderr, "ds4: prompt path warmed up in %.0f ms\n", now_ms() - tw);
        });
    }
    // ---- the chunked prompt path
    size_t x_cap() const { return (size_t) std::max<int>(kWin, chunk + 2); }   // a window's / chunk's new source rows
    size_t chunk_hand_bytes() const {
        return (size_t) chunk * ((size_t) g.hc * g.dim * 4 + 8 * 4 + topk_stride() * 4 + cand_stride() * 4);
    }

    // the staging ring's size (MB) for the experts a chunk streams from the CPU lane: 400 where a good part of them are
    // streamed; 16 where the cards hold most of the experts (they are within 30% of fitting) (a 4-card box) - the ring's VRAM is then worth more
    // as expert slots, and the decode after a prompt keeps the windowed path's speed (R9700 x4: 29.6 -> 31.7 tok/s)
    // The share of the routed experts the GPUs can hold, from what is known at layout (their free memory less the reserve, the
    // prompt path's buffers and the dense weights; the trunk's experts only when no drafter is loaded): 1 = all of them.
    double gpu_expert_share(double* exp_out = nullptr, double* room_out = nullptr, double* other_out = nullptr) const {
        double exp_b = 0, other_b = 0, room = 0;
        for (const auto& kt : pk.t) {
            const std::string& n = kt.first;
            if (n.find(".engram.embed.") != std::string::npos || n.rfind("vision.", 0) == 0 || n == "embed.weight" ||
                (!o.dspark && n.rfind("mtp.", 0) == 0))
                continue;
            (n.find(".ffn.experts.") != std::string::npos ? exp_b : other_b) += (double) kt.second.bytes;
        }
        for (const Dev& d : devs)
            room += std::max(0.0, (double) d.free0 - (double) ((size_t) o.reserve_mib << 20) - 1.5 * (1 << 30)) * o.vram_experts;
        if (exp_out) *exp_out = exp_b;
        if (room_out) *room_out = room;
        if (other_out) *other_out = other_b;
        return exp_b > 0 ? std::min(1.0, std::max(0.0, room - other_b) / exp_b) : 1.0;
    }
    size_t ring_mb_auto() const {
        double exp_b = 0, other_b = 0, room = 0;
        const double share = gpu_expert_share(&exp_b, &room, &other_b);
        // One rule for the staging ring (ds4_experts_rule.hpp: staging_ring_mb): 16 MB where the cards hold the experts, a small
        // ring from each device's room where they hold most of them (4 x R9700), the larger streamed ring from the summed room on a
        // unified-memory GPU (Strix Halo: 400 -> 1200 MB) and where the cards hold under half.  DS4_RING_MB overrides.
        bool unified = false;
        for (const Dev& d : devs) {
            cudaDeviceProp ip{};
            if (cudaGetDeviceProperties(&ip, d.id) == cudaSuccess && ip.integrated != 0) unified = true;
        }
        const size_t mb = (size_t) strata::ds4::staging_ring_mb(exp_b > 0 ? share : 0.0, room, (int) devs.size(), unified, rdna4(), defaults_026());
        if (o.verbose)
            std::fprintf(stderr, "ds4: staging ring: experts %.1f GB, VRAM room %.1f GB, dense %.1f GB -> %d MB\n", exp_b / 1e9,
                         room / 1e9, other_b / 1e9, (int) mb);
        return mb;
    }
    bool slab_auto_grew_ = false, slab_force_base_ = false;   // the automatic attention slab: grew this try / must not grow (it did not fit)
    bool alloc_chunks(std::string& err) {
        slab_auto_grew_ = false;
        chunks.resize(devs.size());
        const int D = g.dim, H = g.hc, HD = g.hd;
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            cudaSetDevice(d.id);
            chunks[di] = std::make_unique<Chunk>();
            Chunk& c = *chunks[di];
            const int nm = (int) groups[(size_t) d.grp].mem.size();
            const int NHL = g.heads / head_members(nm), HQ = NHL * HD, OG = g.o_groups / head_members(nm);
            int FF = 0, ff0_ = 0;
            split128(g.ff, nm, 0, ff0_, FF);   // (the widest member's share: member 0 takes the odd blocks)
            const size_t C = (size_t) chunk;
            c.C = chunk;
            // the base slab: STRATA_DS4_A_MB megabytes of q.  Every dense GEMM of a layer is issued once a slab, so the slab width divides
            // the prompt's dense weight traffic.  A unified-memory GPU (Strix Halo, main 3133160ed) takes 32 MB: measured 8 -> 32 MB took
            // its 32K prompt 325.89 -> 357.78 tok/s, 16K 315.01 -> 328.04, 8K +1.6%, 4K flat, decode unmoved.  A discrete card keeps 8 MB
            // (the 0.2.6 slab): its VRAM is the expert tier's, and on the RTX 5070 a bigger slab bought +2% prompt for -3% decode.
            // STRATA_DS4_A_MB overrides the base on any GPU (8 restores the old slab).
            bool unified = false;
            {
                cudaDeviceProp ip{};
                unified = cudaGetDeviceProperties(&ip, d.id) == cudaSuccess && ip.integrated != 0;
            }
            static const size_t a_mb_env = [] {
                const char* v = std::getenv("STRATA_DS4_A_MB");
                return (size_t) (v != nullptr ? std::max(1, std::atoi(v)) : 0);
            }();
            const size_t a_mb = strata::ds4::attn_slab_base_mb(unified, a_mb_env, defaults_026());
            c.A = std::max(64, (int) ((a_mb << 20) / (size_t) HQ) / 64 * 64);
            // DS4_ATTN_SLAB=N: the attention slab's rows.  The dense projections (q_b, wo_a/b) are dequantized and GEMM'd
            // once a slab, so a slab of 256 rows dequantizes the same weights 32 times for an 8K prompt (measured: the
            // dense dequant 916 ms of the 8K prompt, rocBLAS 867 ms); a larger slab costs VRAM for c.q/c.o and the oa
            // buffers (A * HQ x 4 bytes each).  0.2.7: unset = automatic (attn_slab_rows: up to 2048 rows where the buffers
            // take VRAM no expert would have used), 0 = the 0.2.6 rule above, N = that many rows.
            {
                int env = -1;
                if (const char* v = std::getenv("DS4_ATTN_SLAB")) env = std::max(0, std::atoi(v));
                if ((slab_force_base_ || unified || a_mb_env != 0) && env < 0) env = 0;   // (no automatic growth on a unified-memory GPU, or when STRATA_DS4_A_MB chose the base)
                // per row: c.q + c.o (float), c.o16 (half), c.oa (float), c.oa16 (half)
                const double bpr = (double) HQ * 10.0 + (double) OG * (double) g.o_lora * 6.0;
                double spare = 1e18;   // the expert tier off (--vram-experts 0): the buffers cost no expert
                double share = 0.0;    // the share of the routed experts the cards hold (0 with the tier off)
                if (o.vram_experts > 0.0) {
                    double exp_b = 0, room = 0, other_b = 0;
                    share = gpu_expert_share(&exp_b, &room, &other_b);
                    spare = (room - other_b - exp_b) / (double) std::max<size_t>(1, devs.size());
                }
                const int base = c.A;
                // one rule (ds4_experts_rule.hpp: attn_slab_rows): the chunk's whole slab (<= 1024 rows) where the cards hold half the
                // experts or more (4 x R9700, +9% prompt), up to 2048 rows where the VRAM is spare, the base otherwise
                c.A = strata::ds4::attn_slab_rows(env, chunk, base, spare, bpr, rdna4() ? share : 0.0, defaults_026());
                if (env < 0 && c.A > base) {
                    slab_auto_grew_ = true;
                    if (di == 0 && o.verbose) {
                        if (rdna4() && share >= 0.5)
                            std::fprintf(stderr, "ds4: attention slab %d rows (automatic: the cards hold %.0f%% of the experts; DS4_ATTN_SLAB=0 keeps %d)\n",
                                         c.A, share * 100.0, base);
                        else
                            std::fprintf(stderr, "ds4: attention slab %d rows (automatic: %.1f GB of VRAM to spare beyond the experts; DS4_ATTN_SLAB=0 keeps %d)\n",
                                         c.A, spare / 1e9, base);
                    }
                }
            }
            // (the indexer's slab: its scores rows are the context wide - about 64 MB of them)
            c.AI = std::max(32, std::min(128, (int) (((size_t) 64 << 20) / (((size_t) o.max_context + 64) * 4)) / 32 * 32));
            c.w16_elems = (size_t) 8 << 20;
            bool has_eg = false, has_first = false;
            size_t stride = 0;
            for (const Layer& l : L)
                for (int m = 0; m < l.nm; ++m)
                    if (member(l, m).dev == (int) di && !l.mtp) {
                        has_eg = has_eg || l.eg >= 0;
                        has_first = has_first || l.il == 0;
                        stride = std::max(stride, align256(unit(member(l, m))));
                    }
            const size_t A = (size_t) c.A, AI = (size_t) c.AI, ss = (size_t) o.max_context + 64;
            bool ok = dalloc(d, c.h, C * H * D, err) && dalloc(d, c.x, C * D, err) && dalloc(d, c.acc, C * D, err) &&
                      dalloc(d, c.x16, C * D, err) && dalloc(d, c.qr, C * g.q_lora, err) &&
                      dalloc(d, c.qr16, C * g.q_lora, err) && dalloc(d, c.kvr, C * HD, err) &&
                      dalloc(d, c.ckr, C * HD, err) && dalloc(d, c.cgr, C * HD, err) &&
                      dalloc(d, c.latent, (C + 2) * HD, err) && dalloc(d, c.lat16, (C + 2) * HD, err) &&
                      dalloc(d, c.ikr, (C + 2) * g.ix_hd, err) && dalloc(d, c.ext, ((size_t) g.win + C) * HD, err) &&
                      dalloc(d, c.q, A * HQ, err) && dalloc(d, c.o, A * HQ, err) && dalloc(d, c.o16, A * HQ, err) &&
                      dalloc(d, c.oa, A * OG * g.o_lora, err) && dalloc(d, c.oa16, A * OG * g.o_lora, err) &&
                      dalloc(d, c.iq, AI * g.ix_heads * g.ix_hd, err) && dalloc(d, c.iw, C * 64, err) &&
                      dalloc(d, c.scores, AI * ss, err) && dalloc(d, c.bm, AI * bm_stride(), err) &&
                      dalloc(d, c.topk, C * topk_stride(), err) && dalloc(d, c.cand, C * cand_stride(), err) &&
                      dalloc(d, c.mix, C * 32, err) && dalloc(d, c.a_pre, C * 8, err) && dalloc(d, c.a_post, C * 8, err) &&
                      dalloc(d, c.a_comb, C * 64, err) && dalloc(d, c.f_pre, C * 8, err) &&
                      dalloc(d, c.f_post, C * 8, err) && dalloc(d, c.f_comb, C * 64, err) && dalloc(d, c.pre, C * 8, err) &&
                      dalloc(d, c.glog, C * 1024, err) && dalloc(d, c.ridx, C * kPicks, err) &&
                      dalloc(d, c.rw, C * kPicks, err) && dalloc(d, c.bounds, 2048, err) &&
                      dalloc(d, c.row_tok, C * kPicks, err) && dalloc(d, c.pos, C * kPicks, err) &&
                      dalloc(d, c.sg, C * FF, err) && dalloc(d, c.su, C * FF, err) && dalloc(d, c.sh16, C * FF, err) &&
                      dalloc(d, c.h16, (C + kH16Pad) * FF, err) && dalloc(d, c.y, C * D, err) && dalloc(d, c.w16, c.w16_elems, err) &&
                      halloc(c.bounds_pin, 2048, err);
            if (!ok) return false;
            if (has_first && !dalloc(d, c.emb, C * D, err)) return false;
            if (has_eg) {
                const size_t ES = 256, per = (size_t) cols * (ehd + ehd / 32);
                if (!dalloc(d, c.eg_x, ES * cols * ehd, err) || !dalloc(d, c.eg_x16, ES * cols * ehd, err) ||
                    !dalloc(d, c.eg_kv, ES * (H + 1) * D, err) || !dalloc(d, c.eg_rows, C * per, err))
                    return false;
            }
            if (stride > 0) {   // DS4_RING_MB (400) of slots in kParts batches
                const size_t mb = std::getenv("DS4_RING_MB") ? (size_t) std::atoi(std::getenv("DS4_RING_MB")) : ring_mb_auto();
                c.ring_stride = stride;
                c.ring_n = std::max(Chunk::kParts, (int) ((mb << 20) / stride) / Chunk::kParts * Chunk::kParts);
                if (!dalloc(d, c.ring, (size_t) c.ring_n * stride, err)) return false;
            }
            CK(cudaStreamCreateWithFlags(&c.cs, cudaStreamNonBlocking));
            for (int i = 0; i < Chunk::kParts; ++i) {
                CK(cudaEventCreateWithFlags(&c.filled[i], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&c.freed[i], cudaEventDisableTiming));
            }
            if (cublasCreate(&c.blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(c.blas, d.st) != CUBLAS_STATUS_SUCCESS) {
                err = "cuBLAS on CUDA" + std::to_string(d.id);
                return false;
            }
            for (int sl = 0; sl < 2; ++sl) {
                if (d.mi == 0 && (!hmap(c.hand[sl], chunk_hand_bytes(), err) || !hmap(c.hand_flag[sl], 32, err))) return false;
                if (has_first && !halloc(c.emb_host[sl], C * D, err)) return false;
                if (has_eg && !halloc(c.eg_host[sl], C * pk.eg_layers * (size_t) cols * (ehd + ehd / 32), err)) return false;
                CK(cudaEventCreateWithFlags(&c.emb_ev[sl], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&c.eg_ev[sl], cudaEventDisableTiming));
            }
            if (d.mi == 0 && !hmap(c.hand_ack, (size_t) 2 * 8 * 32, err)) return false;
        }
        return true;
    }

    // y[r][c] = x16[r] . W[:, c] (+ beta y) for r < rows: W reconstructed to FP16 in whole 128-column blocks, then the product.
    // The product prefers the engine's own RDNA3 WMMA kernel (src/prefill/wmma_gemm.cu) over hipBLAS: on gfx1151 hipBLAS
    // measures 4.97 TFLOPS at this call convention while that kernel measures 20-40 (BM 128 x BN 256, grouped tile order), and
    // the chunked prompt's dense GEMM is 11.1 s of a 4K prompt at the library's rate.  Same operands and the same values (the
    // k order differs, so it is rounding-level: the quality gate and the acceptance gate cover it); STRATA_DENSE_PFGEMM=0 keeps
    // hipBLAS for an A/B.
    bool gemm(Dev& d, Chunk& c, const x3::Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta,
              std::string& err) {
        if (rows <= 0) return true;
#if defined(STRATA_USE_HIP)   // (wmma_gemm.cu is part of the AMD builds only: a CUDA build has no such kernel and keeps cuBLAS)
        static const bool pf = strata::ds4::dense_pfgemm_default(env_tri("STRATA_DENSE_PFGEMM"), defaults_026());
#endif
        const int cols = std::max(128, (int) std::min<size_t>((size_t) m.n, c.w16_elems / (size_t) m.k) / 128 * 128);
        for (int c0 = 0; c0 < m.n; c0 += cols) {
            const int nc = std::min(cols, m.n - c0);
            if (m.fmt != 0) gg::rows(nc == m.n ? m : gg::view(m, c0, nc), c.w16, false, d.st);   // (a GGUF's)
            else x3::reconstruct_rows(nc == m.n ? m : x3::view(m, c0, nc), c.w16, false, d.st);
#if defined(STRATA_USE_HIP)
            if (pf && rows >= 64 && ((uintptr_t) x16 & 15u) == 0 &&
                strata_pf_gemm_f16_ld(x16, ldx, c.w16, m.k, y + c0, rows, nc, m.k, ldy, beta, d.st))
                continue;
#endif
            const float one = 1.0f;
            const cublasStatus_t st = cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, nc, rows, m.k, &one, c.w16, CUDA_R_16F,
                                                   m.k, x16, CUDA_R_16F, ldx, &beta, y + c0, CUDA_R_32F, ldy,
                                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (st != CUBLAS_STATUS_SUCCESS) {
                err = "cublasGemmEx: status " + std::to_string((int) st);
                return false;
            }
        }
        return true;
    }
    // ---- the busy experts (run_set): an expert of many rows as FP16 GEMMs.  The tiles decode its weights again for
    // every 64 of its rows (a prompt's chunk gives a popular expert hundreds); decoded once, to the FP16 values the tiles
    // multiply (iq::f16_rows), the library's GEMM takes them at its own speed.  DS4_FAT_ROWS=N (default 0, off; N:
    // the experts of N rows or more) takes the experts of N rows or more; the same products, only the sums' order differ
    // (3x V100, Maya-XS: 64 rows +6-8% on 8K / 32K prompts; KL vs the reference 0.276, the stock path 0.285 - the same;
    //  RTX 5090, IST-DAS 2.25 bpw: a 13K-token prompt 1,400 -> 1,625 tok/s at 64 rows, decode unchanged)
    static int fat_rows() {
        static const int n = [] {
            const int v = std::getenv("DS4_FAT_ROWS") ? std::max(0, std::atoi(std::getenv("DS4_FAT_ROWS"))) : 0;
            return v > 0 ? std::max(v, 17) : 0;   // (the light kernels take the experts of 16 rows or fewer)
        }();
        return n;
    }
    bool fat_ready(Dev& d, Chunk& c, int D, int FF) {
#if defined(STRATA_USE_HIP) || defined(STRATA_METAL)   // (CUDA only: the Mac has no dispatch for these kernels, HIP is unmeasured)
        (void) d, (void) c, (void) D, (void) FF;
        return false;
#else
        if (fat_rows() <= 0 || c.fat_off) return false;
        if (c.fat_w != nullptr) return true;
        static const int R = std::getenv("DS4_FAT_PIECE") ? std::max(64, std::atoi(std::getenv("DS4_FAT_PIECE"))) : 1024;
        if (cudaSetDevice(d.id) != cudaSuccess || cudaMalloc(&c.fat_w, (size_t) 2 * FF * D * 2) != cudaSuccess ||
            cudaMalloc(&c.fat_x, (size_t) R * D * 2) != cudaSuccess ||
            cudaMalloc(&c.fat_gu, (size_t) R * 2 * FF * 4) != cudaSuccess) {
            cudaGetLastError();
            std::fprintf(stderr, "ds4: CUDA%d: no VRAM for the busy experts' path (%.0f MB): the tiles take them\n", d.id,
                         ((double) 2 * FF * D * 2 + (double) R * D * 2 + (double) R * 2 * FF * 4) / 1e6);
            c.fat_off = true;
            return false;
        }
        c.fat_r = R;
        std::fprintf(stderr, "ds4: CUDA%d busy experts (%d rows or more) as FP16 GEMMs, %d rows a piece (%.0f MB)\n", d.id,
                     fat_rows(), R, ((double) 2 * FF * D * 2 + (double) R * D * 2 + (double) R * 2 * FF * 4) / 1e6);
        return true;
#endif
    }
    // h16[lr0 + r] = FP16 clamped SwiGLU of the expert's gate and up over x16[row_tok[row + r]], r < n
    bool fat_gate_up(Chunk& c, const x3::SetArgs& sa, const uint8_t* blob, int row, int n, int a, int D, int FF,
                     cudaStream_t s, std::string& err) {
        const int tg = gg::expert_type(sa.lay, 0);
        if (!gg::iq::f16_rows(tg, blob + sa.lay.trellis[0], D, FF, c.fat_w, D, s) ||
            !gg::iq::f16_rows(tg, blob + sa.lay.trellis[1], D, FF, c.fat_w + (size_t) FF * D, D, s)) {
            err = "the busy experts' path: gate / up not decoded";
            return false;
        }
        const float one = 1.0f, zero = 0.0f;
        for (int p0 = 0; p0 < n; p0 += c.fat_r) {
            const int np = std::min(c.fat_r, n - p0);
            gg::iq::gather_f16(c.x16, D, c.row_tok + row + p0, np, D, c.fat_x, s);
            if (cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * FF, np, D, &one, c.fat_w, CUDA_R_16F, D, c.fat_x,
                             CUDA_R_16F, D, &zero, c.fat_gu, CUDA_R_32F, 2 * FF, CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) {
                err = "the busy experts' path: cublasGemmEx (gate / up) failed";
                return false;
            }
            gg::iq::swiglu_f16(c.fat_gu, 2 * FF, np, FF, sa.limit, c.h16 + (size_t) (row - a + p0) * FF, FF, s);
        }
        return true;
    }
    // y[lr0 + r] = down(h16[lr0 + r]), r < n
    bool fat_down(Chunk& c, const x3::SetArgs& sa, const uint8_t* blob, int lr0, int n, int D, int FF, cudaStream_t s,
                  std::string& err) {
        if (!gg::iq::f16_rows(gg::expert_type(sa.lay, 2), blob + sa.lay.trellis[2], FF, D, c.fat_w, FF, s)) {
            err = "the busy experts' path: down not decoded";
            return false;
        }
        const float one = 1.0f, zero = 0.0f;
        if (cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, D, n, FF, &one, c.fat_w, CUDA_R_16F, FF,
                         c.h16 + (size_t) lr0 * FF, CUDA_R_16F, FF, &zero, c.y + (size_t) lr0 * D, CUDA_R_32F, D,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) {
            err = "the busy experts' path: cublasGemmEx (down) failed";
            return false;
        }
        return true;
    }
    // y[r][c] = alpha x16[r] . W[c] for F16 weights W [n][k] (the indexer's weights_proj)
    bool gemm_f16w(Chunk& c, const uint16_t* W, int n, int k, const uint16_t* x16, int ldx, int rows, float alpha,
                   float* y, int ldy, std::string& err) {
        const float zero = 0.0f;
        const cublasStatus_t st = cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &alpha, W, CUDA_R_16F, k,
                                               x16, CUDA_R_16F, ldx, &zero, y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) {
            err = "cublasGemmEx (F16 weights): status " + std::to_string((int) st);
            return false;
        }
        return true;
    }
    // hc_mix over a chunk: the dots by an F32 GEMM, then the rms scale
    bool hc_mix_chunk(Dev& d, Chunk& c, const float* h, const float* fn, int T, float* mixes, std::string& err) {
        const int hcd = g.hc * g.dim, n_mix = (2 + g.hc) * g.hc;
#if defined(STRATA_METAL)
        if (!std::getenv("DS4_CHUNK_SGEMM")) {   // the window path's fused kernel: no F32 GEMM, no second pass
            hc_mix(h, fn, hcd, n_mix, T, g.eps, mixes, d.st);
            return true;
        }
#endif
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n_mix, T, hcd, &one, fn, hcd, h, hcd, &zero, mixes, 32) !=
            CUBLAS_STATUS_SUCCESS) {
            err = "cublasSgemm (mHC mixes)";
            return false;
        }
        hc_mix_scale(h, hcd, n_mix, T, g.eps, mixes, d.st);
        return true;
    }

    // expert e of l (on the CPU lane: its NH column parts) -> member m's VRAM copy at dst, in the GPU layout (the full
    // blob, or a group member's part), by DMA from the pinned parts on the copy stream (in place: from `src`, its file
    // region read into pinned staging)
    bool stream_expert(const Layer& l, int m, int e, uint8_t* dst, cudaStream_t cs, std::string& err,
                       const uint8_t* src = nullptr) {
        if (l.lay.fmt == gg::kFmtMXFP4) {   // the blob, or member m's rows / blocks: from `src` (the expert read into
            // pinned staging as its blob) or from the file's mapping
            auto mat_r = [&](int r) { return src != nullptr ? src + l.lay.trellis[r] : mx_mat(l, e, r); };
            bool ok = true;
            if (l.nm == 1) {
                if (src != nullptr)
                    ok = cudaMemcpyAsync(dst, src, l.blob, cudaMemcpyHostToDevice, cs) == cudaSuccess;
                else
                    for (int r = 0; r < 3; ++r)
                        ok = ok && cudaMemcpyAsync(dst + l.lay.trellis[r], mat_r(r), mx_part_bytes(l, r),
                                                   cudaMemcpyHostToDevice, cs) == cudaSuccess;
            } else {
                const int ffh = g.ff / l.nm;
                const size_t gr = gg::type_row_bytes(gg::expert_type(l.lay, 0), g.dim),
                             dr = gg::type_row_bytes(gg::expert_type(l.lay, 2), g.ff),
                             dh = gg::type_row_bytes(gg::expert_type(l.lay, 2), ffh);
                for (int r = 0; r < 2; ++r)
                    ok = ok && cudaMemcpyAsync(dst + l.hlay.trellis[r], mat_r(r) + (size_t) m * ffh * gr,
                                               (size_t) ffh * gr, cudaMemcpyHostToDevice, cs) == cudaSuccess;
                ok = ok && cudaMemcpy2DAsync(dst + l.hlay.trellis[2], dh, mat_r(2) + (size_t) m * dh, dr, dh,
                                             (size_t) g.dim, cudaMemcpyHostToDevice, cs) == cudaSuccess;
            }
            if (!ok) err = "streaming layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
            return ok;
        }
        const auto& parts = l.hh[(size_t) e];
        const int D = g.dim, FF = g.ff;
        auto cp2 = [&](uint8_t* d0, size_t dp, const uint8_t* s0, size_t sp, size_t w, size_t h) {
            return cudaMemcpy2DAsync(d0, dp, s0, sp, w, h, cudaMemcpyHostToDevice, cs) == cudaSuccess;
        };
        auto cp = [&](uint8_t* d0, const uint8_t* s0, size_t n) {
            return cudaMemcpyAsync(d0, s0, n, cudaMemcpyHostToDevice, cs) == cudaSuccess;
        };
        bool ok = true;
        if (NH == 1) {   // one part: the whole expert in exllamav3's layout, its pieces at hl's offsets
            const uint8_t* p = src != nullptr ? src : parts[0];
            if (l.nm == 1) {
                for (int r = 0; r < 3; ++r)
                    ok = ok && cp(dst + l.lay.trellis[r], p + l.hl.trellis[r], l.piece_bytes[r][0]) &&
                         cp(dst + l.lay.suh[r], p + l.hl.suh[r], (size_t) l.hl.k[r] * 2) &&
                         cp(dst + l.lay.svh[r], p + l.hl.svh[r], (size_t) l.hl.n[r] * 2);
            } else {   // member m's part (its FFN columns [f0, f0 + fn)), cut as blob_to_half cuts a blob
                const Layer& x = member(l, m);
                const int f0 = x.ff0, fn = x.ffn;
                for (int r = 0; r < 2; ++r) {
                    const size_t tile = (size_t) 32 * l.lay.K[r], nt = (size_t) FF / 16, nth = (size_t) fn / 16;
                    ok = ok && cp2(dst + x.hlay.trellis[r], nth * tile, p + l.hl.trellis[r] + (size_t) (f0 / 16) * tile,
                                   nt * tile, nth * tile, (size_t) D / 16) &&
                         cp(dst + x.hlay.suh[r], p + l.hl.suh[r], (size_t) D * 2) &&
                         cp(dst + x.hlay.svh[r], p + l.hl.svh[r] + (size_t) f0 * 2, (size_t) fn * 2);
                }
                const size_t row = (size_t) (D / 16) * 32 * l.lay.K[2];
                ok = ok && cp(dst + x.hlay.trellis[2], p + l.hl.trellis[2] + (size_t) (f0 / 16) * row,
                              (size_t) (fn / 16) * row) &&
                     cp(dst + x.hlay.suh[2], p + l.hl.suh[2] + (size_t) f0 * 2, (size_t) fn * 2) &&
                     cp(dst + x.hlay.svh[2], p + l.hl.svh[2], (size_t) D * 2);
            }
        } else if (l.nm == 1) {   // the full blob: each matrix's columns from the two parts side by side
            for (int r = 0; r < 3; ++r) {
                const size_t tile = (size_t) 32 * l.lay.K[r];
                const int k = l.hl.k[r], nh = l.hl.n[r];   // part: k x nh
                for (int X = 0; X < 2; ++X) {
                    ok = ok && cp2(dst + l.lay.trellis[r] + (size_t) X * (nh / 16) * tile, (size_t) (2 * nh / 16) * tile,
                                   parts[(size_t) X] + l.hl.trellis[r], (size_t) (nh / 16) * tile, (size_t) (nh / 16) * tile,
                                   (size_t) (k / 16));
                    ok = ok && cp(dst + l.lay.svh[r] + (size_t) X * nh * 2, parts[(size_t) X] + l.hl.svh[r], (size_t) nh * 2);
                }
                ok = ok && cp(dst + l.lay.suh[r], parts[0] + l.hl.suh[r], (size_t) k * 2);
            }
        } else {   // member m (FFN columns [f0, f0 + fn)): gate / up from the parts holding them; down rows [f0, ..) of
                   // both parts' columns
            const Layer& x = member(l, m);
            const int f0 = x.ff0, fn = x.ffn;
            for (int r = 0; r < 2; ++r) {
                const size_t tile = (size_t) 32 * l.lay.K[r];
                const int hn = l.hl.n[r];   // part X: columns [X hn, (X + 1) hn)
                for (int X = 0; X < 2; ++X) {
                    const int a = std::max(f0, X * hn), b = std::min(f0 + fn, (X + 1) * hn);
                    if (a >= b) continue;
                    ok = ok && cp2(dst + x.hlay.trellis[r] + (size_t) ((a - f0) / 16) * tile, (size_t) (fn / 16) * tile,
                                   parts[(size_t) X] + l.hl.trellis[r] + (size_t) ((a - X * hn) / 16) * tile,
                                   (size_t) (hn / 16) * tile, (size_t) ((b - a) / 16) * tile, (size_t) (D / 16)) &&
                         cp(dst + x.hlay.svh[r] + (size_t) (a - f0) * 2, parts[(size_t) X] + l.hl.svh[r] + (size_t) (a - X * hn) * 2,
                            (size_t) (b - a) * 2);
                }
                ok = ok && cp(dst + x.hlay.suh[r], parts[0] + l.hl.suh[r], (size_t) D * 2);
            }
            const size_t tile = (size_t) 32 * l.lay.K[2];
            const int nh = D / 2;
            for (int X = 0; X < 2; ++X) {
                ok = ok && cp2(dst + x.hlay.trellis[2] + (size_t) X * (nh / 16) * tile, (size_t) (D / 16) * tile,
                               parts[(size_t) X] + l.hl.trellis[2] + (size_t) (f0 / 16) * (nh / 16) * tile,
                               (size_t) (nh / 16) * tile, (size_t) (nh / 16) * tile, (size_t) (fn / 16));
                ok = ok && cp(dst + x.hlay.svh[2] + (size_t) X * nh * 2, parts[(size_t) X] + l.hl.svh[2], (size_t) nh * 2);
            }
            ok = ok && cp(dst + x.hlay.suh[2], parts[0] + l.hl.suh[2] + (size_t) f0 * 2, (size_t) fn * 2);
        }
        if (!ok) err = "streaming layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
        return ok;
    }

    // a device's streamed experts: batches of ring_n / kParts of each of its layers' CPU-lane experts, chunk by chunk
    // in layer order; the copies run at most kParts batches ahead of the compute
    struct Batch {
        int k = 0, il = 0, j0 = 0, n = 0;   // chunk, layer, its host_list range
    };
    struct Streamer {
        std::vector<Batch> b;
        size_t cp = 0, kp = 0;        // the next batch to copy, to compute
        std::atomic<size_t> done{0};  // cp, for read_ahead's thread
    };
    // in place (file_experts): the batches' file regions read ahead of their copies - POSIX_FADV_WILLNEED, the kernel
    // reading them in the background in large requests, where the copies alone fault the pages in a few at a time -
    // at most DS4_READAHEAD_MB (2048) beyond the batch being copied; each region by its layer's first member alone
    void read_ahead(const Streamer& S, int di, const std::atomic<bool>& stop) const {
        static const double cap = (std::getenv("DS4_READAHEAD_MB") ? std::atof(std::getenv("DS4_READAHEAD_MB")) : 2048.0) *
                                  1048576.0;
        std::deque<std::pair<size_t, double>> issued;   // (batch, bytes) not yet copied
        double ahead = 0;
        for (size_t q = 0; q < S.b.size() && !stop.load(std::memory_order_relaxed);) {
            const size_t done = S.done.load(std::memory_order_relaxed);
            while (!issued.empty() && issued.front().first < done) {
                ahead -= issued.front().second;
                issued.pop_front();
            }
            if (ahead > cap) {
                std::this_thread::sleep_for(std::chrono::microseconds(500));
                continue;
            }
            const Batch& bt = S.b[q];
            const Layer& l0 = L[(size_t) bt.il];
            double nb = 0;
            if (l0.dev == di)
                for (int i = 0; i < bt.n; ++i) {
                    const int e = l0.host_list[(size_t) (bt.j0 + i)];
                    if (l0.lay.fmt == gg::kFmtMXFP4) {
                        if (copy_mode && l0.ram[(size_t) e] >= 0) continue;   // (in RAM: nothing to read)
                        for (int r = 0; r < 3; ++r)
                            os::advise(pk.fds[(size_t) l0.mx_file[r]], (l0.mx_off[r] + (uint64_t) e * l0.mx_stride[r]), mx_part_bytes(l0, r), true);
                        nb += (double) l0.blob;
                        continue;
                    }
                    os::advise(pk.fds[(size_t) l0.region_file[(size_t) e]], l0.region_off[(size_t) e], l0.region_bytes[(size_t) e], true);
                    nb += (double) l0.region_bytes[(size_t) e];
                }
            issued.push_back({q++, nb});
            ahead += nb;
        }
    }
    bool copy_ahead(Dev& d, Chunk& c, Streamer& S, int di, std::string& err) {
        const int P = Chunk::kParts;
        while (S.cp < S.b.size() && S.cp < S.kp + P) {
            const Batch& bt = S.b[S.cp];
            const int half = (int) (S.cp % P);
            const Layer& l0 = L[(size_t) bt.il];
            int mi = 0;
            for (int m = 0; m < l0.nm; ++m)
                if (member(l0, m).dev == di) mi = m;
            if (S.cp >= (size_t) P) CK(cudaStreamWaitEvent(c.cs, c.freed[half], 0));
            uint8_t* sb = nullptr;   // in place: the batch's regions in the part's staging slots
            if (c.stage != nullptr) {
                const double t_evt = now_ms();
                if (S.cp >= (size_t) P) CK(cudaEventSynchronize(c.filled[half]));   // (its last batch's copies out)
                c.dbg_evt_ms += now_ms() - t_evt;
                sb = c.stage + (size_t) half * (c.ring_n / P) * c.stage_stride;
                std::atomic<bool> bad{false};
                const double t_st = now_ms();
                c.readers->run(bt.n, [&](int i) {
                    const int e = l0.host_list[(size_t) (bt.j0 + i)];
                    if (l0.lay.fmt == gg::kFmtMXFP4 ? !read_mx(l0, e, sb + (size_t) i * c.stage_stride)
                                                    : !pk.read_at(l0.region_file[(size_t) e], l0.region_off[(size_t) e],
                                                                  l0.region_bytes[(size_t) e],
                                                                  sb + (size_t) i * c.stage_stride))
                        bad.store(true);
                });
                c.dbg_stage_ms += now_ms() - t_st;
                if (bad.load()) {
                    err = "reading layer " + std::to_string(l0.il) + "'s experts from the model's files failed";
                    return false;
                }
            }
            const double t_dma = now_ms();
            for (int i = 0; i < bt.n; ++i) {
                uint8_t* dst = c.ring + ((size_t) half * (c.ring_n / P) + i) * c.ring_stride;
                if (!stream_expert(l0, mi, l0.host_list[(size_t) (bt.j0 + i)], dst, c.cs, err,
                                   sb != nullptr ? sb + (size_t) i * c.stage_stride : nullptr))
                    return false;
                c.stream_bytes += (double) unit(member(l0, mi));
            }
            c.dbg_dma_ms += now_ms() - t_dma;
            CK(cudaEventRecord(c.filled[half], c.cs));
            ++S.cp;
            S.done.store(S.cp, std::memory_order_relaxed);
        }
        return true;
    }

    // ---- the flow (see Chunk)
    // Windows (WDDM: HIP's runtime and CUDA's driver alike) submits queued work only when the host waits on the GPU:
    // the flow's copies and the compute it queued between its waits are submitted at once (win_submit); elsewhere
    // nothing.  (CUDA too since #303: two RTX 3080s hung before the first layer, one GPU spinning on the other's flow)
    static void kick(cudaStream_t s) { win_submit(s); }
    size_t flow_region(int r) const {   // a part's room in a staging slot (read from its offset rounded down)
        return ((flow_part[r] + os::kDirectAlign - 1) & ~(size_t) (os::kDirectAlign - 1)) + os::kDirectAlign;
    }
    size_t flow_stride() const { return flow_region(0) + flow_region(1) + flow_region(2); }
    static Chunk::Item& fit(Chunk& c, size_t i) { return c.items[i - c.base]; }
    // what member m of l's group reads of matrix r of an expert: gate / up its rows [m n_ff / nm, ..) - one run, from
    // this offset in the matrix, this long; the down the whole matrix (its part, the blocks [m n_ff / nm / 32, ..) of
    // every row, is DMA'd out of it row by row); alone (nm 1) every matrix whole
    size_t mx_read_off(const Layer& l, int m, int r) const {
        return r < 2 && l.nm > 1 ? (size_t) m * (size_t) (g.ff / l.nm) * gg::type_row_bytes(gg::expert_type(l.lay, r), g.dim)
                                 : 0;
    }
    size_t mx_read_len(const Layer& l, int r) const {
        return r < 2 && l.nm > 1 ? (size_t) (g.ff / l.nm) * gg::type_row_bytes(gg::expert_type(l.lay, r), g.dim)
                                 : mx_part_bytes(l, r);
    }
    // the device's member index in L0's group (0 alone)
    int flow_mi(const Chunk& c, const Layer& L0) const {
        for (int m = 0; m < L0.nm; ++m)
            if (member(L0, m).dev == c.dev) return m;
        return 0;
    }
    void flow_push(Chunk& c, const Layer& l, int mi, int k, int e, int v) {
        c.items.emplace_back();
        Chunk::Item& it = c.items.back();
        it.il = l.il;
        it.mi = mi;
        it.k = k;
        it.e = e;
        it.v = v;
        bool in_ram = !l.ram.empty() && l.ram[(size_t) e] >= 0;
        const uint8_t* store_blob = nullptr;   // (the RAM store holds it: copied into staging like a RAM-tier slot, no read)
        if (!in_ram && store && l.nm == 1) {
            const int ss = store->find(l.il, e);
            if (ss >= 0) {
                store_blob = store->blob(ss);
                in_ram = true;
            }
        }
        if (in_ram && ram_pinned && store_blob == nullptr) {   // (DMA'd from where it lies)
            const uint8_t* p = copy_mode ? ram_part[(size_t) l.ram[(size_t) e]][0]
                                         : ram_slots[(size_t) l.ram[(size_t) e]];
            for (int r = 0; r < 3; ++r)
                it.src[r] = p + (copy_mode ? cp_lay[(size_t) l.il][(size_t) r] : l.lay.trellis[r]) + mx_read_off(l, mi, r);
            ++c.it_ram;
            return;
        }
        if (in_ram)   // (copied into staging by io, then DMA'd; copy_mode: from its NH parts, flow_pump)
            it.ram = store_blob != nullptr ? store_blob
                     : copy_mode ? ram_part[(size_t) l.ram[(size_t) e]][0] : ram_slots[(size_t) l.ram[(size_t) e]];
        bool hit = false;
        if (!in_ram && c.lc != nullptr && l.nm == 1) {   // layer-major: from the layer cache, or kept for the chunks after
            const size_t i = c.base + c.items.size() - 1, stride = (l.blob + 63) & ~(size_t) 63;
            if (c.lc_il != l.il) {   // (a layer's first item: the slots the layer before used are its own again
                c.lc_il = l.il;      // once that layer's items are all read - `after` below)
                c.lc_used = 0;
                c.lc_n = (int) std::min<size_t>(c.lc_of.size(), c.lc_bytes / stride);
                c.lc_first = i;
                std::fill(c.lc_of.begin(), c.lc_of.end(), -1);
            }
            const int j = c.lc_of[(size_t) e];
            if (j >= 0) {   // (copied into staging by io once its reader is done, then DMA'd)
                it.ram = c.lc + (size_t) j * stride;
                it.after = c.lc_writer[(size_t) j];
                hit = true;
            } else if (k + 1 < (int) c.fl_T.size() && c.lc_used < c.lc_n) {   // (a later chunk may pick it)
                const int jn = c.lc_used++;
                c.lc_of[(size_t) e] = jn;
                c.lc_writer[(size_t) jn] = (int64_t) i;
                it.keep = c.lc + (size_t) jn * stride;
                it.after = (int64_t) c.lc_first - 1;
            }
        }
        if (c.direct && l.nm == 1 && !(in_ram && copy_mode)) {   // (the ring slot is the destination: filled by flow_pump when its turn comes)
            it.direct = true;
            it.stage = -1;
            it.left.store(3, std::memory_order_relaxed);
            ++(in_ram ? c.it_ram : hit ? c.it_lc : c.it_file);
            return;
        }
        it.stage = (int) (c.n_file++ % (uint64_t) c.S);
        uint8_t* sb = c.stage + (size_t) it.stage * c.stage_stride;
        for (int r = 0; r < 3; ++r) {
            const uint64_t off = l.mx_off[r] + (uint64_t) e * l.mx_stride[r] + mx_read_off(l, mi, r);
            it.src[r] = sb + (in_ram || hit ? 0 : off % os::kDirectAlign);
            sb += flow_region(r);
        }
        it.left.store(in_ram && copy_mode && store_blob == nullptr ? 3 * NH : 3, std::memory_order_relaxed);
        ++(in_ram ? c.it_ram : hit ? c.it_lc : c.it_file);
    }
    // the likely experts of (chunk k, layer il) for T tokens, first in its stream: the CPU lane's experts whose chance
    // to be picked at least once, 1 - (1 - share)^(picks), is at least DS4_SPEC_Q (0.9: the reads are what takes the
    // time, an unpicked one costs a whole read, and the routing is known a few ms after the layer before is done -
    // early reads pay where a chunk's long attention keeps the disk otherwise idle, and there nearly every expert is
    // that likely), likeliest first, at most what the staging and the ring hold (no more can be read before the layer
    // routes); the layer's virtual slots put them first, then its other CPU-lane experts in index order (the order the
    // rest of its picks are streamed in)
    // (a tensor-parallel group's members each decide their own stream and virtual slots - the same, from the same
    // profile and the same routing - and each reads only its part of every expert)
    void flow_spec(Chunk& c, int k, int il, int T) {
        static const double spec_q = std::getenv("DS4_SPEC_Q") ? std::atof(std::getenv("DS4_SPEC_Q")) : 0.9;
        Layer& l = L[(size_t) il];
        const int mi = flow_mi(c, l);
        int* vs = member(l, mi).vs_pin;
        c.spec_k = k;
        c.spec_il = il;
        c.spec_begin = c.base + c.items.size();
        const uint32_t* pr = prof.data() + (size_t) il * g.n_exp;
        double sum = 0;
        for (int e = 0; e < l.n_exp; ++e) sum += pr[e];
        std::vector<std::pair<double, int>> cand;
        const double picks = (double) l.k_exp * T;
        // (early reads only while they hit: once a run's likely ones went unpicked more often than DS4_SPEC_WASTE
        // (0.2) of the time, the rest of the run reads only what the routing picked - an unpicked one costs a whole
        // read, and how well the profile guesses depends on the PC's history and the prompt: V100, 64 tokens - 376
        // of 921 likely ones unpicked, 7 GB of 84 read for nothing; Radeon 8065S, 128 and 512 tokens - 44 of 945 and
        // 21 of 1568, the early reads 9-12% faster there)
        static const double waste = std::getenv("DS4_SPEC_WASTE") ? std::atof(std::getenv("DS4_SPEC_WASTE")) : 0.2;
        const bool early = c.it_spec < 2u * (uint64_t) c.ring_n || (double) c.it_spec_unused <= waste * (double) c.it_spec;
        for (int e : l.host_list) {
            if (!early) break;
            const double p = sum > 0 ? pr[e] / sum : 0.0;
            const double q = p >= 1.0 ? 1.0 : 1.0 - std::exp(picks * std::log1p(-p));
            if (q >= spec_q && spec_q <= 1.0) cand.push_back({q, e});
        }
        std::stable_sort(cand.begin(), cand.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first > b.first;
        });
        if (cand.size() > (size_t) (c.S + c.ring_n)) cand.resize((size_t) (c.S + c.ring_n));
        std::vector<char> is((size_t) l.n_exp, 0);
        int j = 0;
        for (const auto& x : cand) {
            vs[x.second] = l.n_slots + j++;
            is[(size_t) x.second] = 1;
        }
        for (int e : l.host_list)
            if (!is[(size_t) e]) vs[e] = l.n_slots + j++;
        for (const auto& x : cand) flow_push(c, l, mi, k, x.second, vs[x.second]);
        c.n_spec = (int) cand.size();
        c.it_spec += cand.size();
    }
    // (chunk k, layer il) routed (bounds b by virtual slot): the rest of its picks after its likely ones
    void flow_rest(Chunk& c, int k, int il, const std::vector<int>& b) {
        const Layer& l = L[(size_t) il];
        const int mi = flow_mi(c, l);
        const int* vs = member(l, mi).vs_pin;
        for (size_t i = c.spec_begin; i < c.spec_begin + (size_t) c.n_spec; ++i) {
            const int v = fit(c, i).v;
            if (b[(size_t) v + 1] == b[(size_t) v]) ++c.it_spec_unused;
        }
        for (int e : l.host_list) {   // (index order: their virtual slots ascending)
            const int v = vs[e];
            if (v >= l.n_slots + c.n_spec && b[(size_t) v + 1] > b[(size_t) v]) flow_push(c, l, mi, k, e, v);
        }
    }
    // the next (chunk, layer) of the device after (k, il) in the run's order that streams experts (false: none) - a
    // long prompt's skip leaves out its early chunks' layers from Run::yoco on (kv_only there: no experts)
    bool flow_next(const Chunk& c, int k, int il, int& k2, int& il2) const {
        while (flow_next1(c, k, il, k2, il2)) {
            if (c.fl_yoco < 0 || k2 >= c.fl_cut_k || il2 < c.fl_yoco) return true;
            k = k2;
            il = il2;
        }
        return false;
    }
    // the rows chunk k runs layer il on (a long prompt's skip: the cut chunk's from cut_r on, from the yoco layer on)
    static int fl_rows(const Chunk& c, int k, int il) {
        const int T = c.fl_T[(size_t) k];
        return c.fl_yoco >= 0 && k == c.fl_cut_k && il >= c.fl_yoco ? T - c.fl_cut_r : T;
    }
    bool flow_next1(const Chunk& c, int k, int il, int& k2, int& il2) const {
        const auto it = std::find(c.fl_layers.begin(), c.fl_layers.end(), il);
        const size_t i = (size_t) (it - c.fl_layers.begin());
        if (c.lm) {   // (layer-major: the layer's next chunk, then the next layer's first)
            if (k + 1 < (int) c.fl_T.size()) {
                k2 = k + 1;
                il2 = il;
                return true;
            }
            if (i + 1 >= c.fl_layers.size()) return false;
            k2 = 0;
            il2 = c.fl_layers[i + 1];
            return true;
        }
        if (i + 1 < c.fl_layers.size()) {
            k2 = k;
            il2 = c.fl_layers[i + 1];
            return true;
        }
        if (k + 1 < (int) c.fl_T.size() && !c.fl_layers.empty()) {
            k2 = k + 1;
            il2 = c.fl_layers[0];
            return true;
        }
        return false;
    }
    // the reads (in order, each into its staging slot once that slot's last DMA is done) and the DMAs (in order, each
    // into its ring slot once that slot's last use is queued: the copy stream waits for it) as far as they can go now
    bool flow_pump(Chunk& c, std::string& err) {
        const size_t end = c.base + c.items.size();
        IoQueue::Job jobs[6];
        while (c.n_read < end) {
            Chunk::Item& it = fit(c, c.n_read);
            if (it.after >= 0) {   // layer-major: a slot's expert copied out of it once its reader is done; a slot
                // written again once the layer before's items are all read (none still copies out of it)
                bool wait = false;
                if (it.keep == nullptr)
                    wait = (size_t) it.after >= c.base && fit(c, (size_t) it.after).left.load(std::memory_order_acquire) != 0;
                else {
                    while (c.lc_chk < c.n_read &&
                           (c.lc_chk < c.base || fit(c, c.lc_chk).left.load(std::memory_order_acquire) == 0))
                        ++c.lc_chk;
                    wait = c.lc_chk <= (size_t) it.after;
                }
                if (wait) break;
            }
            if (it.direct) {
                const size_t R = (size_t) c.ring_n, r = c.n_read % R;
                if (c.n_read >= R) {   // the slot's last use has to be computed - and finished: the CPU writes it
                    if (c.n_comp <= c.n_read - R) break;
                    CK(cudaEventSynchronize(c.ring_free[r]));
                }
                const Layer& l = L[(size_t) it.il];
#if defined(STRATA_METAL)
                uint8_t* dst = (uint8_t*) strata::gpu::host_view(c.ring + r * c.ring_stride);   // (device addresses are the GPU's: the CPU writes through the host view)
#else
                uint8_t* dst = c.ring + r * c.ring_stride;
#endif
                const x3::ExpertLayout& lay = l.lay;
                int nj = 0;
                for (int q = 0; q < 3; ++q) {
                    IoQueue::Job& j = jobs[nj++];
                    j = IoQueue::Job{};
                    const os::fd_t pf = pk.pfds.empty() ? os::kBadFd : pk.pfds[(size_t) l.mx_file[q]];
                    j.fd = pf != os::kBadFd ? pf : pk.fds[(size_t) l.mx_file[q]];
                    j.dfd = pk.dfds.empty() ? os::kBadFd : pk.dfds[(size_t) l.mx_file[q]];
                    j.off = l.mx_off[q] + (uint64_t) it.e * l.mx_stride[q];
                    j.len = mx_part_bytes(l, q);
                    j.dst = dst + lay.trellis[q];
                    j.exact = true;
                    j.left = &it.left;
                    j.bad = &it.bad;
                    j.mem = it.ram != nullptr ? it.ram + lay.trellis[q] : nullptr;
                    j.map = pk.at(l.mx_file[q], j.off);
                    j.keep = it.keep != nullptr ? it.keep + lay.trellis[q] : nullptr;
                }
                c.io->push(jobs, nj);
            } else if (it.stage >= 0) {
                const int64_t prev = c.stage_item[(size_t) it.stage];
                if (prev >= 0) {
                    if ((size_t) prev >= c.n_dma) break;
                    const cudaError_t q = cudaEventQuery(c.stage_ev[(size_t) it.stage]);
                    if (q == cudaErrorNotReady) break;
                    CK(q);
                }
                c.stage_item[(size_t) it.stage] = (int64_t) c.n_read;
                const Layer& l = L[(size_t) it.il];
                uint8_t* sb = c.stage + (size_t) it.stage * c.stage_stride;
                const int cs = copy_mode && it.ram != nullptr ? l.ram[(size_t) it.e] : -1;   // (a copy's slot)
                int nj = 0;
                for (int r = 0; r < 3; ++r) {
                    if (cs >= 0) {   // a copy: each matrix's NH parts of rows, side by side into staging
                        const size_t nb = (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
                        for (int X = 0; X < NH; ++X) {
                            IoQueue::Job& j = jobs[nj++];
                            j = IoQueue::Job{};
                            j.len = nb;
                            j.dst = sb + (size_t) X * nb;
                            j.left = &it.left;
                            j.bad = &it.bad;
                            j.mem = ram_part[(size_t) cs][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r];
                        }
                        sb += flow_region(r);
                        continue;
                    }
                    IoQueue::Job& j = jobs[nj++];
                    const os::fd_t pf = pk.pfds.empty() ? os::kBadFd : pk.pfds[(size_t) l.mx_file[r]];
                    j.fd = pf != os::kBadFd ? pf : pk.fds[(size_t) l.mx_file[r]];
                    j.dfd = pk.dfds.empty() ? os::kBadFd : pk.dfds[(size_t) l.mx_file[r]];
                    j.off = l.mx_off[r] + (uint64_t) it.e * l.mx_stride[r] + mx_read_off(l, it.mi, r);
                    j.len = mx_read_len(l, r);
                    j.dst = sb;
                    j.left = &it.left;
                    j.bad = &it.bad;
                    j.mem = it.ram != nullptr ? it.ram + l.lay.trellis[r] + mx_read_off(l, it.mi, r) : nullptr;
                    j.map = pk.at(l.mx_file[r], j.off);
                    j.keep = it.keep != nullptr ? it.keep + l.lay.trellis[r] : nullptr;
                    sb += flow_region(r);
                }
                c.io->push(jobs, nj);
            }
            ++c.n_read;
        }
        const size_t R = (size_t) c.ring_n, dma0 = c.n_dma;
        struct Kick {   // (every way out: what was queued is submitted)
            cudaStream_t s;
            const size_t &a, b;
            ~Kick() {
                if (a != b) kick(s);
            }
        } kq{c.cs, c.n_dma, dma0};
        while (c.n_dma < c.n_read && c.n_dma < c.n_comp + R) {
            Chunk::Item& it = fit(c, c.n_dma);
            if (it.left.load(std::memory_order_acquire) != 0) break;
            if (it.bad.load(std::memory_order_acquire)) {
                err = "reading layer " + std::to_string(it.il) + " expert " + std::to_string(it.e) +
                      " from the model's files failed";
                return false;
            }
            const size_t r = c.n_dma % R;
            if (c.n_dma >= R) CK(cudaStreamWaitEvent(c.cs, c.ring_free[r], 0));
            const Layer& l = L[(size_t) it.il];
            uint8_t* dst = c.ring + r * c.ring_stride;
            const x3::ExpertLayout& lay = l.nm > 1 ? l.hlay : l.lay;
            if (it.direct) {   // (already in the slot)
                CK(cudaEventRecord(c.ring_fill[r], c.cs));
                c.stream_bytes += (double) unit(l);
                ++c.n_dma;
                continue;
            }
            for (int p = 0; p < 2; ++p)
                CK(cudaMemcpyAsync(dst + lay.trellis[p], it.src[p], mx_read_len(l, p), cudaMemcpyHostToDevice, c.cs));
            if (l.nm > 1) {   // (the down's part: in each of its rows the member's blocks)
                const int t2 = gg::expert_type(l.lay, 2);
                const size_t dr = gg::type_row_bytes(t2, g.ff), dh = gg::type_row_bytes(t2, g.ff / l.nm);
                CK(cudaMemcpy2DAsync(dst + lay.trellis[2], dh, it.src[2] + (size_t) it.mi * dh, dr, dh, (size_t) g.dim,
                                     cudaMemcpyHostToDevice, c.cs));
            } else
                CK(cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2), cudaMemcpyHostToDevice, c.cs));
            CK(cudaEventRecord(c.ring_fill[r], c.cs));
            if (it.stage >= 0) CK(cudaEventRecord(c.stage_ev[(size_t) it.stage], c.cs));
            c.stream_bytes += (double) unit(l);
            ++c.n_dma;
        }
        return true;
    }
    // layer L0 of chunk k routed (bounds b, the device's copy d_bounds): its streamed experts as they land, the VRAM
    // ones in shares between them (the device works while the reads run)
    bool flow_layer(Dev& d, Chunk& c, Layer& L0, int mi, int k, int T, const std::vector<int>& b, std::string& err) {
        cudaStream_t s = d.st;
        const uint8_t* vbase = member(L0, mi).vram_base;   // (a member's: its part of each VRAM expert, L0's slots)
        if (c.spec_k != k || c.spec_il != L0.il) {
            err = "the prompt path's expert stream is out of step (layer " + std::to_string(L0.il) + ")";
            return false;
        }
        flow_rest(c, k, L0.il, b);
        const size_t lb = c.spec_begin, le = c.base + c.items.size(), n_lay = le - lb;
        int k2 = 0, il2 = 0;
        if (flow_next(c, k, L0.il, k2, il2)) flow_spec(c, k2, il2, fl_rows(c, k2, il2));
        const size_t R = (size_t) c.ring_n, bmax = std::max<size_t>(1, R / 2);
        int v_done = 0;
        while (c.n_comp < le) {
            // a batch's bounds are fixed by the stream alone (bmax items, never across the ring's end), never by what
            // has landed: the experts' outputs are summed batch after batch (the VRAM ones' shares between them), so
            // the sums come out the same on every run whatever the reads' timing
            const size_t i0 = c.n_comp, i1 = std::min({le, i0 + bmax, (i0 / R + 1) * R});
            if (!flow_pump(c, err)) return false;
            if (c.n_dma < i1) {   // not all of the batch landed: wait for a read, or for a staging slot's copy
                const double tw = now_ms();
                kick(s);
                if (c.n_read > c.n_dma) c.io->wait_any(500);
                else {
                    const Chunk::Item& it = fit(c, c.n_read);
                    if (it.stage >= 0) CK(cudaEventSynchronize(c.stage_ev[(size_t) it.stage]));
                }
                c.wait_ms += now_ms() - tw;
                continue;
            }
            const int n = (int) (i1 - i0);
            int* bc = c.bc_pin + (size_t) (c.n_batch % (uint64_t) c.bc_n) * c.bc_w;
            int* bcd = c.bc_dev + (size_t) (c.n_batch % (uint64_t) c.bc_n) * c.bc_w;
            ++c.n_batch;
            // (the items' virtual slots ascend and the slots between them have no rows: their bounds side by side)
            for (int m = 0; m < n; ++m) bc[m] = b[(size_t) fit(c, i0 + (size_t) m).v];
            bc[n] = b[(size_t) fit(c, i1 - 1).v + 1];
            if (bc[n] > bc[0]) {
                CK(cudaStreamWaitEvent(s, c.ring_fill[(i1 - 1) % R], 0));
                CK(cudaMemcpyAsync(bcd, bc, (size_t) (n + 1) * 4, cudaMemcpyHostToDevice, s));
                if (!run_set(d, c, L0, c.ring + (i0 % R) * c.ring_stride, 0, c.ring_stride, bcd, bc, 0, n, T, err))
                    return false;
            }
            for (size_t i = i0; i < i1; ++i) CK(cudaEventRecord(c.ring_free[i % R], s));
            kick(s);
            c.n_comp = i1;
            if (!flow_pump(c, err)) return false;
            const int v_to = (int) ((int64_t) L0.n_slots * (int64_t) (c.n_comp - lb) / (int64_t) std::max<size_t>(1, n_lay));
            if (!run_set(d, c, L0, vbase, 0, unit(L0), c.bounds, b.data(), v_done, v_to, T, err)) return false;
            v_done = v_to;
        }
        if (!run_set(d, c, L0, vbase, 0, unit(L0), c.bounds, b.data(), v_done, L0.n_slots, T, err)) return false;
        while (c.base < c.n_comp && c.items.size() > 4096) {   // (done with: read, copied and used)
            c.items.pop_front();
            ++c.base;
        }
        return true;
    }

    // the experts [i0, i1) of a sorted set (blobs at wbase + i * stride; bounds b on the host, d_bounds on the device)
    // over the chunk's rows, in pieces of at most C rows, each added into acc in pick order
    bool run_set(Dev& d, Chunk& c, const Layer& l, const uint8_t* wbase, int wb0, size_t stride, const int* d_bounds,
                 const int* b, int i0, int i1, int T, std::string& err) {
        const int D = g.dim, FF = l.ffn;
        while (i0 < i1) {
            int j = i0 + 1;
            while (j < i1 && b[j + 1] - b[i0] <= c.C) ++j;
            const int a = b[i0], nrows = b[j] - a;
            int max_rows = 0;
            for (int i = i0; i < j; ++i) max_rows = std::max(max_rows, b[i + 1] - b[i]);
            if (nrows > 0) {
                x3::SetArgs sa;
                sa.wbase = wbase + (size_t) (i0 - wb0) * stride;   // (wbase: set member wb0's blob)
                sa.stride = stride;
                sa.lay = l.nm > 1 ? l.hlay : l.lay;
                sa.n_exp = j - i0;
                sa.bounds = d_bounds + i0;
                sa.row0 = a;
                sa.max_rows = max_rows;
#if defined(STRATA_METAL)
                sa.hbounds = b + i0;
                sa.total_rows = nrows;
#endif
                sa.n_embd = D;
                sa.n_ff = FF;
                sa.limit = g.lim;
                // gate / up read the layer's F16 copy of the activations (the same rounding the tiles make for their
                // shared tile): with 18-36 column blocks an expert the F32 rows were read that many times
                sa.x16 = strata::ds4::gate_up_x16_default(defaults_026()) ? c.x16 : nullptr;
                // the busy experts (DS4_FAT_ROWS=N: those of N rows or more): their weights decoded to FP16 once and
                // multiplied by the library's GEMM - the tiles decode an expert's weights again for every 64 of its rows
                std::vector<int> fat;
                if (sa.lay.fmt == gg::kFmtMXFP4 && !gg::mx_experts(sa.lay) && D % 256 == 0 && FF % 256 == 0 &&
                    gg::expert_type(sa.lay, 1) == gg::expert_type(sa.lay, 0) &&
                    gg::iq::f16_rows_ok(gg::expert_type(sa.lay, 0)) && gg::iq::f16_rows_ok(gg::expert_type(sa.lay, 2)) &&
                    fat_ready(d, c, D, FF)) {
                    int rest = 0;
                    for (int i = i0; i < j; ++i) {
                        const int r = b[i + 1] - b[i];
                        if (r >= fat_rows()) fat.push_back(i);
                        else rest = std::max(rest, r);
                    }
                    if (!fat.empty()) {
                        sa.skip_ge = fat_rows();
                        sa.max_rows = rest;
                    }
                }
                EkMark ek{};
                const bool ekp = prof_chunk;
                if (ekp) {
                    std::lock_guard<std::mutex> lk(prof_mu);
                    ek.e0 = ek_ev();
                    ek.e1 = ek_ev();
                    ek.e2 = ek_ev();
                    ek.tg = gg::expert_type(sa.lay, 0);
                    ek.td = gg::expert_type(sa.lay, 2);
                    ek.nrows = nrows;
                    ek.nexp = j - i0;
                    ek.maxrows = max_rows;
                    ek.ff = FF;
                    cudaGetDevice(&ek.dev);
                    cudaEventRecord(ek.e0, d.st);
                }
                if (sa.lay.fmt == gg::kFmtMXFP4) {
                    gg::set_gate_up(sa, c.x, D, c.row_tok + a, c.h16, FF, d.st);
                    for (const int i : fat)
                        if (!fat_gate_up(c, sa, sa.wbase + (size_t) (i - i0) * stride, b[i], b[i + 1] - b[i], a, D, FF,
                                         d.st, err))
                            return false;
                    if (ekp) cudaEventRecord(ek.e1, d.st);
                    gg::set_down(sa, c.h16, FF, c.y, D, d.st);
                    for (const int i : fat)
                        if (!fat_down(c, sa, sa.wbase + (size_t) (i - i0) * stride, b[i] - a, b[i + 1] - b[i], D, FF,
                                      d.st, err))
                            return false;
                } else {
                    x3::set_gate_up(sa, c.x, D, c.row_tok + a, c.h16, FF, d.st);
                    if (ekp) cudaEventRecord(ek.e1, d.st);
                    x3::set_down(sa, c.h16, FF, c.y, D, d.st);
                }
                if (ekp) {
                    cudaEventRecord(ek.e2, d.st);
                    std::lock_guard<std::mutex> lk(prof_mu);
                    ek_marks.push_back(ek);
                }
                if (dump_y != nullptr && (dump_y_layer < 0 || l.il == dump_y_layer)) {
                    std::vector<int> bb(b + i0, b + j + 1);
                    std::vector<uint16_t> hv((size_t) nrows * FF);
                    std::vector<float> yv((size_t) nrows * D);
                    CK(cudaStreamSynchronize(d.st));
                    CK(cudaMemcpy(hv.data(), c.h16, hv.size() * 2, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(yv.data(), c.y, yv.size() * 4, cudaMemcpyDeviceToHost));
                    std::lock_guard<std::mutex> lk(dump_mu);
                    const int64_t hdr[6] = {l.il, a, nrows, j - i0, wbase == c.ring || (wbase > c.ring &&
                                            wbase < c.ring + (size_t) c.ring_n * c.ring_stride) ? 1 : 0, max_rows};
                    std::fwrite(hdr, 8, 6, dump_y);
                    std::fwrite(bb.data(), 4, bb.size(), dump_y);
                    std::fwrite(hv.data(), 2, hv.size(), dump_y);
                    std::fwrite(yv.data(), 4, yv.size(), dump_y);
                    std::fflush(dump_y);
                }
                moe_combine(c.acc, c.y, c.pos, c.rw, T, l.k_exp, a, a + nrows, D, d.st);
            }
            i0 = j;
        }
        (void) err;
        return true;
    }

    bool load_dense(int di, std::string& err) {
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        size_t* acct = &d.weights;
        for (Layer& l0 : L)
          for (int mi = 0; mi < l0.nm; ++mi) {
            Layer& l = member(l0, mi);
            if (l.dev != di) continue;
            // (a group's member: its share of the heads (wq_b columns, the sinks), of wo_a's groups and wo_b's rows,
            // of the shared expert's FFN; the rest whole)
            const int nm = l0.nm, na = g.heads / std::max(1, l.ahn), ha = l.ah0 / std::max(1, l.ahn);
            const std::string& P = l.name;
            const std::string A = P + "attn.";
            bool ok = mat(d, A + "wq_a", l.wq_a, err, acct) &&
                      (l.ahn == 0 || mat(d, A + "wq_b", l.wq_b, err, acct, ha, na, true)) &&
                      mat(d, A + "wkv", l.wkv, err, acct) &&
                      (l.ahn == 0 || mat(d, A + "wo_b", l.wo_b, err, acct, ha, na, false)) &&
                      f32(d, A + "q_norm.weight", l.q_norm, err, acct, (size_t) g.q_lora) &&
                      f32(d, A + "kv_norm.weight", l.kv_norm, err, acct, (size_t) g.hd) &&
                      f32(d, A + "attn_sink", l.sink, err, acct, (size_t) g.heads) &&
                      f32(d, P + "attn_norm.weight", l.attn_norm, err, acct, (size_t) g.dim) &&
                      f32(d, P + "ffn_norm.weight", l.ffn_norm, err, acct, (size_t) g.dim) &&
                      f32(d, P + "hc_attn_fn", l.hca_fn, err, acct) && f32(d, P + "hc_attn_scale", l.hca_scale, err, acct) &&
                      f32(d, P + "hc_attn_base", l.hca_base, err, acct) && f32(d, P + "hc_ffn_fn", l.hcf_fn, err, acct) &&
                      f32(d, P + "hc_ffn_scale", l.hcf_scale, err, acct) &&
                      f32(d, P + "hc_ffn_base", l.hcf_base, err, acct) &&
                      f32(d, P + "ffn.gate.weight", l.gate_w, err, acct, (size_t) l.n_exp * g.dim) &&
                      f32(d, P + "ffn.gate.bias", l.gate_bias, err, acct, (size_t) l.n_exp) &&
                      (l.mtp || img_tok < 0 || pk.get(P + "ffn.gate.bias_vl") == nullptr ||
                       f32(d, P + "ffn.gate.bias_vl", l.gate_bias_vl, err, acct, (size_t) l.n_exp)) &&
                      mat(d, P + "ffn.shared_experts.w1", l.s1, err, acct, mi, nm, true, l.il % nm) &&
                      mat(d, P + "ffn.shared_experts.w2", l.s2, err, acct, mi, nm, false, l.il % nm) &&
                      mat(d, P + "ffn.shared_experts.w3", l.s3, err, acct, mi, nm, true, l.il % nm);
            if (!ok) return false;
            l.sink += (size_t) l.ah0;
            for (int gi = 0; gi < l.ogn; ++gi)
                if (!mat(d, A + "wo_a.slice." + std::to_string(l.og0 + gi), l.wo_a[gi], err, acct)) return false;
            if (l.kv_src) {
                const std::string C = A + "compressor.";
                if (!mat(d, C + "wkv", l.c_wkv, err, acct) || !f32(d, C + "norm.weight", l.c_norm, err, acct)) return false;
                if (l.ratio > 1 && !mat(d, C + "wgate", l.c_wgate, err, acct)) return false;
                if (l.ratio > 1 && (!dalloc(d, l.c_kv, (size_t) l.ratio * g.hd, err) ||
                                    !dalloc(d, l.c_score, (size_t) l.ratio * g.hd, err)))
                    return false;
            }
            if (l.ix_src) {
                const std::string I = A + "indexer.";
                if (!mat(d, I + "wq_b", l.i_wq_b, err, acct) || !f16raw(d, I + "weights_proj.weight", l.i_wproj, err, acct))
                    return false;
                if (l.owns_k && (!mat(d, I + "wk", l.i_wk, err, acct) || !f32(d, I + "k_norm.weight", l.ik_norm, err, acct)))
                    return false;
            }
            if (l.eg >= 0) {
                if (!mat(d, P + "engram.wkv", l.eg_wkv, err, acct)) return false;
                if (pk.get(P + "engram.gate.weight") != nullptr) {   // a GGUF: q_weight * k_weight, folded already
                    if (!f32(d, P + "engram.gate.weight", l.eg_w, err, acct, (size_t) g.hc * g.dim)) return false;
                } else {
                    float *qw = nullptr, *kw = nullptr;
                    if (!f32(d, P + "engram.q_weight", qw, err, acct) || !f32(d, P + "engram.k_weight", kw, err, acct))
                        return false;
                    std::vector<float> a((size_t) g.hc * g.dim), b(a.size());
                    CK(cudaMemcpy(a.data(), qw, a.size() * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(b.data(), kw, b.size() * 4, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < a.size(); ++i) a[i] *= b[i];   // only ever used as a product
                    CK(cudaMemcpy(qw, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
                    l.eg_w = qw;
                }
            }
            if (!dalloc(d, l.win, (size_t) g.win * win_rb(), err, &d.caches)) return false;
            if (g.n_mtp > 0 && !l.mtp) {
                if (!dalloc(d, l.ring_bak, (size_t) kWin * win_rb(), err)) return false;
                if (l.kv_src && l.ratio > 1 && (!dalloc(d, l.c_snap, (size_t) 2 * l.ratio * g.hd, err) ||
                                                !dalloc(d, l.c_rows, (size_t) 2 * kWin * g.hd, err)))
                    return false;
            }
            if (!dalloc(d, l.vram_tab, (size_t) l.n_exp, err, acct)) return false;
          }
        if (d.grp == devs[(size_t) head_dev].grp) {   // the head: each member of its group a share of the columns
            const int hn = head_nm();
            if (!mat(d, "head", d.head, err, acct, d.mi, hn, true) || !f32(d, "norm.weight", d.norm_w, err, acct, (size_t) g.dim))
                return false;
            if (g.n_mtp > 0 && !f32(d, "mtp." + std::to_string(g.n_mtp - 1) + ".norm.weight", d.d_norm, err, acct, (size_t) g.dim))
                return false;
            if (!dalloc(d, d.logits_part, (size_t) kWin * ((g.vocab / 128 + hn - 1) / hn * 128), err)) return false;
            if (g.n_mtp > 0 && hn > 1) {   // the markov steps over the group (mk_split): this member's rows of the head
                const Tensor* th = pk.get("mtp." + std::to_string(g.n_mtp - 1) + ".markov_head.head.weight");
                const int R = g.d_rank;
                head_cols(hn, d.mi, d.mk_v0, d.mk_vn);   // (its share of the head's columns: the same rows)
                // F16 as the pack holds it, or a GGUF drafter's BF16 rounded to F16 as f16raw rounds d_mk_head (the same
                // values as the lead's whole rows); another type: no mk_split (the lead's whole rows)
                if (th != nullptr && (th->dtype == "F16" || th->dtype == "BF16") && th->bytes == (uint64_t) g.vocab * R * 2) {
                    std::vector<uint16_t> rows((size_t) d.mk_vn * R);
                    if (!pk.read_at(th->file, th->off + (uint64_t) d.mk_v0 * R * 2, rows.size() * 2, rows.data())) {
                        err = "the markov head: read failed";
                        return false;
                    }
                    if (th->dtype == "BF16")
                        for (uint16_t& h : rows) h = strata::kernels::f16_from_f32(bf16_to_f32(h));
                    if (!dalloc(d, d.mk_share, rows.size(), err, acct) || !dalloc(d, d.mk_erow, (size_t) R, err) ||
                        !dalloc(d, d.mk_ints, (size_t) kWin + 2, err))
                        return false;
                    CK(cudaMemcpy(d.mk_share, rows.data(), rows.size() * 2, cudaMemcpyHostToDevice));
                }
            }
        }
        if (di == head_dev) {
            head = d.head;
            norm_w = d.norm_w;
            d_norm = d.d_norm;
            if (g.n_mtp > 0) {
                const std::string Z = "mtp." + std::to_string(g.n_mtp - 1) + ".";
                if (!f32(d, Z + "confidence_head.proj.weight", d_conf, err, acct, (size_t) (g.dim + g.d_rank)))
                    return false;
                const Tensor* te = pk.get(Z + "markov_head.embed.weight");
                const Tensor* th = pk.get(Z + "markov_head.head.weight");
                if (te == nullptr || th == nullptr || te->dtype != "BF16" || (th->dtype != "F16" && th->dtype != "BF16")) {
                    err = Z + "markov_head: not the BF16 embed and F16 / BF16 head expected";
                    return false;
                }
                if (!dalloc(d, d_mk_embed, te->bytes / 2, err, acct) || !upload(d, *te, d_mk_embed, err) ||
                    (strata::ds4::markov_head_q8_default(defaults_026())
                         ? !mk_head_q8(d, Z + "markov_head.head.weight", d_mk_head_q8, err, acct)
                         : !f16raw(d, Z + "markov_head.head.weight", d_mk_head, err, acct)))   // (0.2.6: the head as F16)
                    return false;
            }
        }
        if (g.n_mtp > 0 && di == d_dev) {
            if (!mat(d, "mtp.0.main_proj", d_main_proj, err, acct) ||
                !f32(d, "mtp.0.main_norm.weight", d_main_norm, err, acct, (size_t) g.dim))
                return false;
        }
        return true;
    }

#if defined(STRATA_METAL)
    // The most the VRAM experts may add on a Mac: the engine's anonymous memory (Metal buffers + host memory, all counted) stays
    // under the cap metal/src/engine/arena_budget.h derived from the freezes of the 128 GB M5 Max (macOS keeps a file-cache floor and
    // compresses / swaps anonymous memory past it): max(15% of the RAM, 40% of the RAM, the floor + 12 GB) left to macOS, at most
    // the GPU working set; what the process holds now (task_info phys_footprint) is already inside it.
    double metal_expert_cap() const {
        uint64_t phys = 0;
        size_t len = sizeof phys;
        sysctlbyname("hw.memsize", &phys, &len, nullptr, 0);
        unsigned long long pages = 0;
        size_t pl = sizeof pages;
        double floor_b = 0;
        if (sysctlbyname("vm.vm_page_filecache_min", &pages, &pl, nullptr, 0) == 0) floor_b = (double) pages * (double) vm_page_size;
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        const double gb = 1e9, P = (double) phys;
        double keep = std::max(7.0 * gb, 0.15 * P);
        if (const char* e = std::getenv("STRATA_METAL_KEEP_FREE_GB")) keep = std::atof(e) * gb;
        double share = 0.40;
        if (const char* e = std::getenv("STRATA_METAL_LEAVE_SHARE")) share = std::atof(e);
        double leave = keep;
        if (P >= 64.0 * gb) leave = std::max({leave, share * P, floor_b > 0 ? floor_b + 12.0 * gb : 0.0});
        const double cap = std::min((double) tot, P - leave);
        task_vm_info_data_t ti;
        mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
        double foot = 0;
        if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &ti, &cnt) == KERN_SUCCESS) foot = (double) ti.phys_footprint;
        // Other programs' anonymous memory (a browser, another model's server).  The cap above was measured with a small server beside
        // the engine (NEIGHBOUR_ALLOWANCE); with more anonymous memory held elsewhere the same share leaves macOS too little and the Mac
        // pages (a 128 GB M5 Max froze twice with a 60 GB Qwen server running next to this engine).  The rest comes off the experts, so
        // they stream from the SSD a little more.  STRATA_METAL_NEIGHBOURS=0 turns it off.
        double neighbours = 0;
        const char* ne = std::getenv("STRATA_METAL_NEIGHBOURS");
        if (ne == nullptr || std::atoi(ne) != 0) {
            vm_statistics64_data_t vs;
            mach_msg_type_number_t vc = HOST_VM_INFO64_COUNT;
            if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vs, &vc) == KERN_SUCCESS) {
                const double anon = ((double) vs.internal_page_count + (double) vs.total_uncompressed_pages_in_compressor) * (double) vm_page_size;
                constexpr double NEIGHBOUR_ALLOWANCE = 8.0e9;
                neighbours = std::max(0.0, anon - foot - NEIGHBOUR_ALLOWANCE);
                if (neighbours > 0.5e9)
                    std::fprintf(stderr, "ds4: other programs hold %.1f GB of anonymous memory: %.1f GB less for the experts\n", anon / gb, neighbours / gb);
            }
        }
        return std::max(4.0 * gb, cap - foot - 1.0 * gb - neighbours);
    }
#endif
#if defined(STRATA_METAL)
    // Metal (unified memory): a CPU-lane expert's matrices through no-copy buffers over the model's own pages, so the GPU computes it
    // like a VRAM one (no CPU lane, no second pass).  false: not mappable / the cap is reached (the CPU lane has it).
    bool zc_try(Layer& l, int e) {
        static const bool on = std::getenv("DS4_ZC") != nullptr;   // (measured: no faster than the CPU lane - cold pages come from the SSD either way)
        if (!on || l.lay.fmt != gg::kFmtMXFP4 || l.nm != 1 || l.vm.empty()) return false;
        if (l.vm[(size_t) e][0].raw != nullptr) return true;
        uint64_t a[3];
        for (int r = 0; r < 3; ++r) {
            a[r] = ::strata::dsv41::metal_host::zc_map(mx_mat(l, e, r), mx_part_bytes(l, r));
            if (a[r] == 0) return false;
        }
        for (int r = 0; r < 3; ++r) {
            x3::Mat& m = l.vm[(size_t) e][(size_t) r];
            m = x3::Mat{};
            m.K = gg::expert_type(l.lay, r);
            m.fmt = m.K == gg::kTypeMXFP4 ? gg::kFmtMXFP4 : gg::kFmtGG;
            m.raw = (const void*) a[r];
            m.k = r < 2 ? g.dim : g.ff;
            m.n = r < 2 ? g.ff : g.dim;
        }
        return true;
    }
#endif
    void set_vm(Layer& l, int e, int slot) {
        uint8_t* b = l.vram_base + (size_t) slot * l.blob;
        if (l.lay.fmt == gg::kFmtMXFP4) {
            for (int r = 0; r < 3; ++r) {
                x3::Mat& m = l.vm[(size_t) e][(size_t) r];
                m = x3::Mat{};
                m.K = gg::expert_type(l.lay, r);   // (a low-bit type: its GEMV, ds4_iq's)
                m.fmt = m.K == gg::kTypeMXFP4 ? gg::kFmtMXFP4 : gg::kFmtGG;
                m.raw = b + l.lay.trellis[r];
                m.k = r < 2 ? g.dim : g.ff;
                m.n = r < 2 ? g.ff : g.dim;
            }
            return;
        }
        for (int r = 0; r < 3; ++r) {
            x3::Mat& m = l.vm[(size_t) e][(size_t) r];
            m.trellis = (const uint32_t*) (b + l.lay.trellis[r]);
            m.suh = (const uint16_t*) (b + l.lay.suh[r]);
            m.svh = (const uint16_t*) (b + l.lay.svh[r]);
            m.k = r < 2 ? g.dim : g.ff;
            m.n = r < 2 ? g.ff : g.dim;
            m.K = l.lay.K[r];
            m.cb = 2;
            m.ld = 0;
        }
    }
    // an expert's VRAM blob -> its CPU-lane part X, and its NH parts -> the blob (the cache's swaps)
    void blob_to_part(const Layer& l, const uint8_t* blob, int X, uint8_t* dst) const {
        for (int r = 0; r < 3; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r];
            const int k = l.hl.k[r], nt = l.hl.n[r] * NH / 16, nth = l.hl.n[r] / 16;
            const uint8_t* tr = blob + l.lay.trellis[r];
            for (int sl = 0; sl < k / 16; ++sl)
                std::memcpy(dst + l.hl.trellis[r] + (size_t) sl * nth * tile, tr + ((size_t) sl * nt + (size_t) X * nth) * tile,
                            (size_t) nth * tile);
            std::memcpy(dst + l.hl.suh[r], blob + l.lay.suh[r], (size_t) k * 2);
            std::memcpy(dst + l.hl.svh[r], blob + l.lay.svh[r] + (size_t) X * l.hl.n[r] * 2, (size_t) l.hl.n[r] * 2);
        }
    }
    void parts_to_blob(const Layer& l, const std::array<uint8_t*, 4>& parts, uint8_t* blob) const {
        for (int r = 0; r < 3; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r];
            const int k = l.hl.k[r], nt = l.hl.n[r] * NH / 16, nth = l.hl.n[r] / 16;
            uint8_t* tr = blob + l.lay.trellis[r];
            for (int X = 0; X < NH; ++X)
                for (int sl = 0; sl < k / 16; ++sl)
                    std::memcpy(tr + ((size_t) sl * nt + (size_t) X * nth) * tile,
                                parts[(size_t) X] + l.hl.trellis[r] + (size_t) sl * nth * tile, (size_t) nth * tile);
            std::memcpy(blob + l.lay.suh[r], parts[0] + l.hl.suh[r], (size_t) k * 2);
            for (int X = 0; X < NH; ++X)
                std::memcpy(blob + l.lay.svh[r] + (size_t) X * l.hl.n[r] * 2, parts[(size_t) X] + l.hl.svh[r],
                            (size_t) l.hl.n[r] * 2);
        }
    }

    // an expert's blob -> member m's part (gate / up columns [ff0, ff0 + ffn), down rows of the same), and back
    void blob_to_half(const Layer& l, const uint8_t* blob, int m, uint8_t* dst) const {
        if (l.lay.fmt == gg::kFmtMXFP4) return mx_blob_to_half(l, blob, m, dst);
        const Layer& x = member(l, m);
        const int f0 = x.ff0, fn = x.ffn;
        for (int r = 0; r < 2; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r], nt = (size_t) g.ff / 16, nth = (size_t) fn / 16;
            for (size_t sl = 0; sl < (size_t) g.dim / 16; ++sl)
                std::memcpy(dst + x.hlay.trellis[r] + sl * nth * tile,
                            blob + l.lay.trellis[r] + (sl * nt + (size_t) f0 / 16) * tile, nth * tile);
            std::memcpy(dst + x.hlay.suh[r], blob + l.lay.suh[r], (size_t) g.dim * 2);
            std::memcpy(dst + x.hlay.svh[r], blob + l.lay.svh[r] + (size_t) f0 * 2, (size_t) fn * 2);
        }
        const size_t row = (size_t) (g.dim / 16) * 32 * l.lay.K[2];
        std::memcpy(dst + x.hlay.trellis[2], blob + l.lay.trellis[2] + (size_t) (f0 / 16) * row, (size_t) (fn / 16) * row);
        std::memcpy(dst + x.hlay.suh[2], blob + l.lay.suh[2] + (size_t) f0 * 2, (size_t) fn * 2);
        std::memcpy(dst + x.hlay.svh[2], blob + l.lay.svh[2], (size_t) g.dim * 2);
    }
    void half_to_blob(const Layer& l, const uint8_t* src, int m, uint8_t* blob) const {
        if (l.lay.fmt == gg::kFmtMXFP4) return mx_half_to_blob(l, src, m, blob);
        const Layer& x = member(l, m);
        const int f0 = x.ff0, fn = x.ffn;
        for (int r = 0; r < 2; ++r) {
            const size_t tile = (size_t) 32 * l.lay.K[r], nt = (size_t) g.ff / 16, nth = (size_t) fn / 16;
            for (size_t sl = 0; sl < (size_t) g.dim / 16; ++sl)
                std::memcpy(blob + l.lay.trellis[r] + (sl * nt + (size_t) f0 / 16) * tile,
                            src + x.hlay.trellis[r] + sl * nth * tile, nth * tile);
            if (m == 0) std::memcpy(blob + l.lay.suh[r], src + x.hlay.suh[r], (size_t) g.dim * 2);
            std::memcpy(blob + l.lay.svh[r] + (size_t) f0 * 2, src + x.hlay.svh[r], (size_t) fn * 2);
        }
        const size_t row = (size_t) (g.dim / 16) * 32 * l.lay.K[2];
        std::memcpy(blob + l.lay.trellis[2] + (size_t) (f0 / 16) * row, src + x.hlay.trellis[2], (size_t) (fn / 16) * row);
        std::memcpy(blob + l.lay.suh[2] + (size_t) f0 * 2, src + x.hlay.suh[2], (size_t) fn * 2);
        if (m == 0) std::memcpy(blob + l.lay.svh[2], src + x.hlay.svh[2], (size_t) g.dim * 2);
    }

    bool place_experts(std::string& err) {
        // (Windows, an integrated GPU: what its experts may commit - commit_room - and what that kept off the GPU)
        double room = 1e300, kept = 0;
        for (const Dev& d : devs)
            if (commit_charged(d)) room = commit_room();
        const double room0 = room;
        // per group: what its tightest member has left (each member holds its part of every VRAM expert of the
        // group's layers, its unit bytes a slot - counted at the members' mean: an uneven split's odd blocks go round
        // the members layer by layer, so each member's total comes to about the mean's)
        auto ucount = [](const Layer& l) {
            double s = 0;
            for (int m = 0; m < l.nm; ++m) s += (double) unit(member(l, m));
            return s / l.nm;
        };
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            double budget = 1e300;
            bool charged = false;
            for (int m : groups[gi].mem) {
                cudaSetDevice(devs[(size_t) m].id);
                size_t fr = 0, tot = 0;
                CK(cudaMemGetInfo(&fr, &tot));
                fr = uma_free(devs[(size_t) m].id, fr);
                // (an integrated GPU, Windows: its memory is the system's.  What it allocates after the experts - the long
                // prompt's buffers, the GEMM workspaces - spills into RAM when the experts took every byte of the GPU's
                // address space, and on a 32K prompt that ran the machine out of RAM.  So the experts leave room for them,
                // growing with the context.  DS4_UMA_GPU_RESERVE_GB: that room, tests)
                double uma_extra = 0.0;
                if (commit_charged(devs[(size_t) m])) {
                    uma_extra = (4.0 + 2.0e-4 * std::max(0.0, (double) o.max_context - 16384.0)) * (double) (1ull << 30);
                    if (const char* v = std::getenv("DS4_UMA_GPU_RESERVE_GB")) uma_extra = std::atof(v) * (double) (1ull << 30);
                }
                budget = std::min(budget, std::max(0.0, ((double) fr - (double) ((size_t) o.reserve_mib << 20) - uma_extra)) *
                                              o.vram_experts);
#if defined(STRATA_METAL)
                if (o.vram_experts_auto && std::getenv("DS4_NO_AUTO_VX") == nullptr) {
                    const double cap = metal_expert_cap();
                    if (cap < budget) {
                        std::fprintf(stderr, "ds4: expert share: %.1f GB of VRAM experts (the RAM cap leaves %.1f GB; --vram-experts sets it)\n",
                                     cap / 1e9, cap / 1e9);
                        budget = cap;
                    }
                }
#endif
                charged = charged || commit_charged(devs[(size_t) m]);
            }
            if (charged && room < 1e300) {   // (each member's slots charged: the group's share of the room)
                double all = 0;
                for (const Layer& l : L)
                    if (l.grp == (int) gi) all += (double) unit(l) * l.n_exp;
                const double nm = (double) groups[gi].mem.size(), capped = std::min(budget, room / nm);
                kept += std::max(0.0, std::min(budget, all) - std::min(capped, all)) * nm;
                budget = capped;
                room = std::max(0.0, room - std::min(budget, all) * nm);
            }
            double budget_t = budget;   // the drafter's experts first, all of them (every draft uses its blocks)
            for (Layer& l : L)
                if (l.grp == (int) gi && l.mtp) {
                    l.n_vram = std::min(l.n_exp, (int) std::floor(budget_t / (double) ucount(l)));
                    budget_t -= (double) l.n_vram * ucount(l);
                }
            double want = 0;
            for (const Layer& l : L)
                if (l.grp == (int) gi && !l.mtp && !l.cpu_only) want += (double) ucount(l) * l.n_exp;
            const double frac = want > 0 ? std::min(1.0, std::max(0.0, budget_t) / want) : 0.0;
            for (Layer& l : L) {
                if (l.grp != (int) gi || l.mtp) continue;
                if (l.cpu_only) {   // (its type has no device kernel: the CPU lane reads every expert of it)
                    l.n_vram = 0;
                    continue;
                }
                l.n_vram = std::min(l.n_exp, (int) std::floor(frac * l.n_exp));
                // never more than fits: the floor of each layer's share can still round up the total
                while (l.n_vram > 0 && (double) ucount(l) * l.n_vram > budget) --l.n_vram;
            }
            double used = 0;
            for (Layer& l : L)
                if (l.grp == (int) gi) used += (double) ucount(l) * l.n_vram;
            // what the floors left: whole experts, layer by layer
            for (Layer& l : L) {
                if (l.grp != (int) gi || l.cpu_only) continue;
                while (l.n_vram < l.n_exp && used + (double) ucount(l) <= budget) {
                    ++l.n_vram;
                    used += (double) ucount(l);
                }
            }
        }
        if (kept > 0) {   // the commit kept experts off the GPU: say so, and the page file that keeps them on it
            const os::Commit c = os::commit();
            const double gib = (double) (1ull << 30);
            const double pf = c.limit != ~uint64_t{0} && c.ram != ~uint64_t{0} && c.limit > c.ram ? (double) (c.limit - c.ram) : 0.0;
            commit_kept = kept;
            pagefile_hint = std::ceil((pf + kept) / gib);
            std::fprintf(stderr,
                         "ds4: WARNING: %.1f GB of experts stay off the GPU - Windows charges the GPU's memory to its "
                         "commit (RAM + page file) and can commit only %.1f GB more for them (%.0f GB kept for the "
                         "other programs). A page file of %.0f GB (%.0f MB; now %.0f GB) keeps them on the GPU: System "
                         "> About > Advanced system settings > Performance > Advanced > Virtual memory, then restart\n",
                         kept / 1e9, room0 / 1e9, commit_margin() / 1e9, pagefile_hint, pagefile_hint * 1024.0, pf / gib);
        } else if (room0 < 1e300 && o.verbose) {
            std::fprintf(stderr, "ds4: Windows' commit has room for the GPU's experts (%.1f GB beside %.0f GB for the other "
                                 "programs)\n", room0 / 1e9, commit_margin() / 1e9);
        }
        for (Layer& l : L) {
            // the hottest experts of the profile first, then by index
            std::vector<int> order((size_t) l.n_exp);
            std::iota(order.begin(), order.end(), 0);
            const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });
            l.slot.assign((size_t) l.n_exp, -1);
            l.host.assign((size_t) l.n_exp, nullptr);
            // a layer with experts on both sides keeps one VRAM slot free for the cache's swaps
            l.n_slots = std::min(l.n_vram, l.n_exp);
            if (o.expert_cache && l.n_slots >= 2 && l.n_slots < l.n_exp) {   // (every expert fits: nothing to swap)
                l.n_vram = l.n_slots - 1;
                l.spare_vram = l.n_slots - 1;
            } else {
                l.n_vram = l.n_slots;
                l.spare_vram = -1;
            }
            for (int i = 0; i < l.n_vram; ++i) l.slot[(size_t) order[(size_t) i]] = i;
            for (int m = 0; m < l.nm && l.n_slots > 0; ++m) {
                Layer& x = member(l, m);
                Dev& d = devs[(size_t) x.dev];
                cudaSetDevice(d.id);
                void* v = nullptr;
                if (cudaMalloc(&v, (size_t) l.n_slots * unit(x)) != cudaSuccess) {
                    err = "cudaMalloc of layer " + std::to_string(l.il) + "'s " + std::to_string(l.n_slots) +
                          " experts on CUDA" + std::to_string(d.id) + " failed";
                    return false;
                }
                d.allocs.push_back(v);
                d.experts += (size_t) l.n_slots * unit(x);
                x.vram_base = (uint8_t*) v;
            }
            l.heat.assign((size_t) l.n_exp, 0.0f);
            {   // the profile as the starting heat: its counts scaled to ~512 hits a layer
                double sum = 0;
                for (int e = 0; e < l.n_exp; ++e) sum += p[e];
                const double sc = sum > 0 ? 512.0 / sum : 0.0;
                for (int e = 0; e < l.n_exp; ++e) l.heat[(size_t) e] = (float) (p[e] * sc);
            }
            const int n_host = l.n_exp - l.n_vram;
            {   // the CPU lane's parts of an expert (in place: the whole expert, its pieces where the file has them)
                size_t off = 0;
                for (int r = 0; r < 3; ++r) {
                    const int k = r < 2 ? g.dim : g.ff, nf = r < 2 ? g.ff : g.dim, n = nf / NH;
                    l.hl.k[r] = k;
                    l.hl.n[r] = n;
                    l.hl.trellis[r] = file_experts ? l.piece_src[r][0] : off;
                    off += align256((size_t) (k / 16) * (n / 16) * 32 * l.lay.K[r]);
                    l.hl.suh[r] = file_experts ? l.piece_src[r][1] : off;
                    off += align256((size_t) k * 2);
                    l.hl.svh[r] = file_experts ? l.piece_src[r][2] : off;
                    off += align256((size_t) n * 2);
                }
                l.hl.bytes = off;
            }
            l.hh.assign((size_t) l.n_exp, {nullptr, nullptr, nullptr, nullptr});
            if (l.lay.fmt == gg::kFmtMXFP4) {   // a GGUF's: read in place from the file's mapping
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) l.host[(size_t) e] = (uint8_t*) mx_mat(l, e, 0);
            } else if (file_experts) {   // every expert's file mapped (the cache's swaps hand VRAM experts back to the files)
                for (int e = 0; e < l.n_exp; ++e)
                    if (pk.map(l.region_file[(size_t) e]) == nullptr) {
                        err = "mapping " + pk.files[(size_t) l.region_file[(size_t) e]] + " failed";
                        return false;
                    }
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) {
                        l.hh[(size_t) e][0] = file_part(l, e);
                        l.host[(size_t) e] = l.hh[(size_t) e][0];
                    }
            } else if (n_host > 0) {
                const int n_parts = n_host + (l.spare_vram >= 0 ? 1 : 0);   // (+ the cache's spare)
                const size_t bytes = (size_t) n_parts * l.hl.bytes;
                for (int X = 0; X < NH; ++X) {
                    void* m = node_map(bytes, X, NH);
                    if (m == nullptr) {
                        err = "host memory for layer " + std::to_string(l.il) + "'s " + std::to_string(n_host) +
                              " CPU experts (" + std::to_string(bytes >> 20) + " MiB on node " + std::to_string(X) + ")";
                        return false;
                    }
                    host_maps.emplace_back(m, bytes);
                    int hi = 0;
                    for (int e = 0; e < l.n_exp; ++e)
                        if (l.slot[(size_t) e] < 0) l.hh[(size_t) e][(size_t) X] = (uint8_t*) m + (size_t) (hi++) * l.hl.bytes;
                    if (l.spare_vram >= 0) l.spare_host[(size_t) X] = (uint8_t*) m + (size_t) hi * l.hl.bytes;
                }
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) l.host[(size_t) e] = l.hh[(size_t) e][0];
            }
            l.vm.assign((size_t) l.n_exp, {});
            if (l.nm == 1)   // (the host-routed window path's views: a group's members take the device-routed path)
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] >= 0) set_vm(l, e, l.slot[(size_t) e]);
            if (!halloc(l.tab_pin, (size_t) 2 * l.nm, err)) return false;
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                std::vector<unsigned long long> tab((size_t) l.n_exp, 0ull);
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] >= 0)
                        tab[(size_t) e] = (unsigned long long) (x.vram_base + (size_t) l.slot[(size_t) e] * unit(x));
                cudaSetDevice(devs[(size_t) x.dev].id);
                CK(cudaMemcpy(x.vram_tab, tab.data(), tab.size() * 8, cudaMemcpyHostToDevice));
            }
        }
        return true;
    }

    // a GGUF's VRAM experts, in file order, by a few threads through pinned staging blobs (the CPU lane's stay in the
    // file: the mapping reads them on first use)
    bool load_experts_mx(std::string& err) {
        struct Job {
            int layer, e;
        };
        std::vector<Job> jobs;
        size_t maxb = 0;
        for (const Layer& l : L) {
            maxb = std::max(maxb, l.blob);
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] >= 0) jobs.push_back({l.il, e});
        }
        std::sort(jobs.begin(), jobs.end(), [&](const Job& a, const Job& b) {
            const Layer &la = L[(size_t) a.layer], &lb = L[(size_t) b.layer];
            if (la.mx_file[0] != lb.mx_file[0]) return la.mx_file[0] < lb.mx_file[0];
            return la.mx_off[0] + (uint64_t) a.e * la.mx_stride[0] < lb.mx_off[0] + (uint64_t) b.e * lb.mx_stride[0];
        });
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::mutex emu;
        std::string ferr;
        std::atomic<int64_t> done_bytes{0};
        const double t0 = now_ms();
        std::vector<std::thread> th;
        for (int t = 0; t < 12; ++t)
            th.emplace_back([&] {
                uint8_t *stage = nullptr, *stage2 = nullptr;
                cudaHostAlloc((void**) &stage, maxb, cudaHostAllocPortable);
                cudaHostAlloc((void**) &stage2, maxb, cudaHostAllocPortable);
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= jobs.size() || failed.load()) break;
                    const Layer& l = L[(size_t) jobs[i].layer];
                    const int e = jobs[i].e;
                    bool ok = stage != nullptr && stage2 != nullptr && read_mx(l, e, stage);
                    for (int m = 0; m < l.nm && ok; ++m) {   // (a group: each member its part)
                        const Layer& x = member(l, m);
                        cudaSetDevice(devs[(size_t) x.dev].id);
                        if (l.nm > 1) blob_to_half(l, stage, m, stage2);
                        ok = cudaMemcpy(x.vram_base + (size_t) l.slot[(size_t) e] * unit(l), l.nm > 1 ? stage2 : stage,
                                        unit(l), cudaMemcpyHostToDevice) == cudaSuccess;
                    }
                    if (!ok) {
                        std::lock_guard<std::mutex> lk(emu);
                        ferr = "loading layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
                        failed.store(true);
                        break;
                    }
                    const int64_t db = done_bytes.fetch_add((int64_t) l.blob) + (int64_t) l.blob;
                    if (o.verbose && (i % 1024) == 0)
                        std::fprintf(stderr, "ds4: experts %zu / %zu (%.1f GB, %.2f GB/s)\n", i, jobs.size(), db / 1e9,
                                     db / 1e6 / std::max(1.0, now_ms() - t0));
                }
                if (stage) cudaFreeHost(stage);
                if (stage2) cudaFreeHost(stage2);
            });
        for (auto& t : th) t.join();
        if (failed.load()) {
            err = ferr;
            return false;
        }
        // as load_experts in place: the VRAM experts' pages leave the cache, the most-routed of the others (the
        // profile's share of their layer, then by index) are read once, up to warm_bytes, so the cache starts with them
        struct Warm {
            int layer, e;
            double share;
        };
        std::vector<Warm> warm;
        for (const Layer& l : L) {
            const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
            double sum = 0;
            for (int e = 0; e < l.n_exp; ++e) sum += p[e];
            for (int e = 0; e < l.n_exp; ++e) {
                if (l.slot[(size_t) e] >= 0) {
                    for (int r = 0; r < 3; ++r)
                        os::advise(pk.fds[(size_t) l.mx_file[r]], (l.mx_off[r] + (uint64_t) e * l.mx_stride[r]), mx_part_bytes(l, r), false);
                } else {
                    warm.push_back({l.il, e, sum > 0 ? p[e] / sum : 0.0});
                }
            }
        }
        std::stable_sort(warm.begin(), warm.end(), [](const Warm& a, const Warm& b) { return a.share > b.share; });
        if (!ram_tier && !warm.empty()) {   // copied into RAM (copy_experts decides), the rest read ahead into what is left
            std::vector<std::pair<int, int>> hot;
            for (const Warm& w : warm) hot.push_back({w.layer, w.e});
            if (!copy_experts(hot, err)) return false;
            if (copy_mode) {
                warm.erase(std::remove_if(warm.begin(), warm.end(),
                                          [&](const Warm& w) { return L[(size_t) w.layer].ram[(size_t) w.e] >= 0; }),
                           warm.end());
                const uint64_t av = ram_avail();
                if (av != ~uint64_t{0}) warm_bytes = av > kRamMargin ? std::min<uint64_t>(warm_bytes, av - kRamMargin) : 0;
            }
        }
        if (ram_tier && warm_bytes > 0 && !warm.empty()) {
            // chunks of 32 slots up to warm_bytes (all the RAM the rest leaves, kRamMargin kept), one slot kept free for
            // a move in; the hottest experts read in, their pages then dropped from the page cache.  Plain memory, not
            // pinned: only the CPU reads it (what goes to a GPU passes the pinned staging), and pinned memory is capped on
            // Windows - the engine's own pinned buffers come after this
            const size_t want = std::min(warm.size(), (size_t) (warm_bytes / maxb));
            while (ram_slots.size() < want + 1) {
                const size_t n = std::min<size_t>(32, want + 1 - ram_slots.size());
                uint8_t* c = (uint8_t*) os::anon_alloc(n * maxb, -1, 1);
                if (c == nullptr) break;
                ram_chunks.push_back(c);
                ram_chunk_bytes.push_back(n * maxb);
                for (size_t i = 0; i < n; ++i) ram_slots.push_back(c + i * maxb);
            }
            ram_held = ram_slots.empty() ? 0 : (int) ram_slots.size() - 1;
            for (Layer& l : L)
                if (l.lay.fmt == gg::kFmtMXFP4) l.ram.assign((size_t) l.n_exp, -1);
            std::atomic<size_t> ri{0};
            std::atomic<bool> rfail{false};
            std::vector<std::thread> rt;
            for (int t = 0; t < 12 && ram_held > 0; ++t)
                rt.emplace_back([&] {
                    for (size_t i; (i = ri.fetch_add(1)) < (size_t) ram_held;) {
                        const Layer& l = L[(size_t) warm[i].layer];
                        if (!read_mx_file(l, warm[i].e, ram_slots[i])) rfail.store(true);
                        for (int r = 0; r < 3; ++r)
                            os::advise(pk.fds[(size_t) l.mx_file[r]], l.mx_off[r] + (uint64_t) warm[i].e * l.mx_stride[r],
                                       mx_part_bytes(l, r), false);
                    }
                });
            for (auto& t : rt) t.join();
            if (rfail.load()) {
                err = "reading the RAM tier's experts from the model's files failed";
                return false;
            }
            for (int i = 0; i < ram_held; ++i) L[(size_t) warm[(size_t) i].layer].ram[(size_t) warm[(size_t) i].e] = i;
            if (ram_held > 0) ram_free.push_back(ram_held);
            if (o.expert_cache && ram_held > 0) ram_th = std::thread([this] { ram_loop(); });
            if (o.verbose)
                std::fprintf(stderr, "ds4: RAM tier: %d experts (%.1f GB of RAM), the hottest after VRAM's; the other %zu "
                                     "read from the model's files (%.1f s)\n",
                             ram_held, ram_slots.size() * (double) maxb / 1e9, warm.size() - (size_t) ram_held,
                             (now_ms() - t0) / 1000.0);
            return true;
        }
        size_t nw = 0;
        for (double b = 0; nw < warm.size() && b + (double) L[(size_t) warm[nw].layer].blob <= (double) warm_bytes; ++nw)
            b += (double) L[(size_t) warm[nw].layer].blob;
        std::atomic<size_t> wi{0};
        std::vector<std::thread> wt;
        for (int t = 0; t < 12 && nw > 0; ++t)
            wt.emplace_back([&] {
                std::vector<uint8_t> buf(maxb);
                for (size_t i; (i = wi.fetch_add(1)) < nw;) read_mx(L[(size_t) warm[i].layer], warm[i].e, buf.data());
            });
        for (auto& t : wt) t.join();
        if (o.verbose && nw > 0)
            std::fprintf(stderr, "ds4: %zu CPU-lane experts read ahead (%.1f GB) in %.1f s\n", nw,
                         nw * (double) L[0].blob / 1e9, (now_ms() - t0) / 1000.0);
        return true;
    }

    // a GGUF's CPU-lane experts copied into RAM (copy_mode), decided once VRAM's are in - on a unified-memory APU the GPU
    // has taken its share of the RAM by then, so the copies only get what it leaves.  `hot`: the CPU lane's experts, the
    // most-routed first.  --mmap-experts off: all that fit beside kRamMargin (a warning if not all do); on and auto:
    // none - read in place: the page cache follows a conversation's experts where a static set (the profile's
    // hottest) does not (the RAM tier's lesson), and copies that all fit were no faster; DS4_COPY_GB=<n>: the
    // hottest n GB (the static hybrid, for tests).  Each copy in NH parts on the NUMA nodes (DS4_COPY_NUMA=1: one), read from the files in file
    // order by a few threads, anonymous memory on huge pages (Windows: plain), not pinned (only the CPU reads it; what
    // goes to a GPU passes the pinned staging) - a layer that cannot get its memory keeps its experts in place
    bool copy_experts(const std::vector<std::pair<int, int>>& hot, std::string& err) {
        if (o.mmap_experts > 0 || hot.empty()) return true;
        const double t0 = now_ms();
        auto spare_of = [&](const Layer& l) { return o.expert_cache && l.spare_vram >= 0 ? 1 : 0; };
        double all = 0;
        {
            std::vector<char> seen(L.size(), 0);
            for (const auto& h : hot) {
                const Layer& l = L[(size_t) h.first];
                all += (double) l.blob * (1 + (seen[(size_t) l.il]++ == 0 ? spare_of(l) : 0));
            }
        }
        const uint64_t avail = ram_avail();
        // (Windows: the copies are committed memory, within what its commit still has beside the other programs' margin)
        // DS4_COPY_HUGETLB: the arena comes from the kernel's huge-page pool, which MemAvailable does not count: add
        // its free bytes so the full copy is taken when the pool holds it.
        // 0.2.7: DS4_COPY_HUGETLB unset = ON when the reserved pool already holds the whole copy (HugePages_Free), else the
        // ordinary arena exactly as before; =0 is the off switch, =1 asks for it anyway (a smaller pool is used, the rest warns).
        const uint64_t pool_free = os::huge_free_bytes();
        const bool huge_arena = strata::ds4::copy_hugetlb_default(env_tri("DS4_COPY_HUGETLB"), (double) pool_free, all, defaults_026());
        if (huge_arena && std::getenv("DS4_COPY_HUGETLB") == nullptr)
            std::fprintf(stderr, "ds4: the reserved huge-page pool (%.1f GB free) holds the whole copy (%.1f GB): the arena on 2 MB pages "
                                 "(DS4_COPY_HUGETLB=0 keeps the ordinary arena)\n", (double) pool_free / 1e9, all / 1e9);
        bool arena_all_huge = huge_arena;   // (every chunk really from the pool: the pin below needs that)
        const double pool = huge_arena ? (double) pool_free : 0.0;
        const double can = std::min(avail == ~uint64_t{0} ? 1e300 : std::max(0.0, (double) avail - (double) kRamMargin) + pool,
                                    commit_room());
        // (auto: none - copies that all fit were measured no faster than the page cache: the CPU lane's rows are
        // compute-bound, ~46 GB/s from either on 2 Xeon sockets, NUMA parts included; a part copied cost decode speed
        // on 64 GB and on 32 GB of RAM)
        double budget = o.mmap_experts == 0 ? std::min(all, can) : auto_copy_all && all <= can ? all : 0.0;   // (auto: all or none)
        if (const char* v = std::getenv("DS4_COPY_GB")) budget = std::atof(v) * 1e9;
        if (o.mmap_experts < 0 && auto_copy_all && all > can)
            std::fprintf(stderr, "ds4: the CPU lane's experts (%.1f GB) do not all fit the RAM this process can get now (%.1f GB beside %.0f GB for "
                                 "the rest): read in place from the model's files (--mmap-experts off copies what fits)\n", all / 1e9, can / 1e9, kRamMargin / 1e9);
        if (o.mmap_experts == 0 && all > can)
            std::fprintf(stderr, "ds4: --mmap-experts off: the CPU lane's experts (%.1f GB) do not all fit the RAM this "
                                 "process can get (%.1f GB beside %.0f GB for the rest) - the most-routed copied, the "
                                 "others read in place\n", all / 1e9, can / 1e9, kRamMargin / 1e9);
        if (auto_copy_all && budget > 0)
            std::fprintf(stderr, "ds4: --mmap-experts auto: the CPU lane's experts (%.1f GB) fit the RAM this process can get (%.1f GB beside %.0f GB for "
                                 "the rest): copied in at load, no demand store (--mmap-experts on keeps them in place)\n", all / 1e9, can / 1e9, kRamMargin / 1e9);
        if (budget <= 0) {
            if (o.verbose)
                std::fprintf(stderr, "ds4: the CPU lane's experts (%.1f GB) stay in place: the RAM this process can get "
                                     "(%.1f GB beside %.0f GB for the rest) does not hold them all\n",
                             all / 1e9, can / 1e9, kRamMargin / 1e9);
            return true;
        }
        // NUMA parts: whole rows of each matrix, so any row count the nodes divide
        int nh = (int) std::min<size_t>(2, std::max<size_t>(1, cores.size()));
        if (const char* v = std::getenv("DS4_COPY_NUMA"))   // (0 / 1: one part; 2: two, even on one node - a test)
            nh = std::atoi(v) >= 2 ? 2 : 1;
        if (g.ff % nh != 0 || g.dim % nh != 0) nh = 1;
        NH = nh;
        cp_lay.assign(L.size(), {0, 0, 0, 0});
        for (const Layer& l : L) {
            size_t off = 0;
            for (int r = 0; r < 3; ++r) {
                cp_lay[(size_t) l.il][(size_t) r] = off;
                off += align256((size_t) (mx_n(r) / NH) * mx_row_b(l, r));
            }
            cp_lay[(size_t) l.il][3] = off;
        }
        // the hottest that fit the budget (a layer's first one brings its spare)
        std::vector<int> n_copy(L.size(), 0);
        double used = 0;
        size_t take = 0;
        for (; take < hot.size(); ++take) {
            const Layer& l = L[(size_t) hot[take].first];
            const double add = (double) l.blob * (1 + (n_copy[(size_t) l.il] == 0 ? spare_of(l) : 0));
            if (used + add > budget) break;
            used += add;
            ++n_copy[(size_t) l.il];
        }
        // each layer's slots: one block per node of (copies + spare) parts
        std::vector<int> first(L.size(), -1);
        copy_spare.assign(L.size(), -1);
        for (Layer& l : L) {
            l.ram.assign((size_t) l.n_exp, -1);
            const int ns = n_copy[(size_t) l.il] + (n_copy[(size_t) l.il] > 0 ? spare_of(l) : 0);
            if (ns == 0) continue;
            const size_t part = cp_lay[(size_t) l.il][3], bytes = (size_t) ns * part;
            std::array<uint8_t*, 2> base = {nullptr, nullptr};
            // DS4_COPY_HUGETLB=1: the arena on explicit 2 MB pages (the pool must be reserved), so a pin of it keeps
            // 2 MB pages and the prompt's DMA reads stay stable (a THP arena pinned by cudaHostRegister runs in 4 KB
            // pages: measured 11.9 s one prompt, 30.4 s the next for the same 140 GB)
            const bool huge = huge_arena;
            bool ok = true;
            for (int X = 0; X < NH && ok; ++X) {
                size_t got = bytes;
                base[(size_t) X] = (uint8_t*) (huge ? os::huge_alloc(bytes, &got) : node_map(bytes, X, NH));
                if (huge && base[(size_t) X] == nullptr) {   // no pool on this machine: the ordinary arena instead
                    static bool warned = false;
                    if (!warned) {
                        std::fprintf(stderr, "ds4: the 2 MB huge-page arena was asked for but the pool ran out (reserve more: sysctl "
                                             "vm.nr_hugepages=N); using the ordinary arena\n");
                        warned = true;
                    }
                    arena_all_huge = false;
                    got = bytes;
                    base[(size_t) X] = (uint8_t*) node_map(bytes, X, NH);
                }
                ok = base[(size_t) X] != nullptr;
                if (ok) {
                    ram_chunks.push_back(base[(size_t) X]);
                    ram_chunk_bytes.push_back(got);
                }
            }
            if (!ok) {
                std::fprintf(stderr, "ds4: no RAM for layer %d's %d copied experts (%.1f GB): they stay in place\n", l.il,
                             ns, (double) bytes * NH / 1e9);
                n_copy[(size_t) l.il] = 0;
                continue;
            }
            first[(size_t) l.il] = (int) ram_part.size();
            for (int i = 0; i < ns; ++i)
                ram_part.push_back({base[0] + (size_t) i * part, NH > 1 ? base[1] + (size_t) i * part : nullptr});
            if (spare_of(l)) copy_spare[(size_t) l.il] = first[(size_t) l.il] + ns - 1;
        }
        struct Job {
            int layer, e, slot;
        };
        std::vector<Job> jobs;
        {
            std::vector<int> next = first;
            for (size_t i = 0; i < take; ++i) {
                const int il = hot[i].first;
                if (n_copy[(size_t) il] == 0) continue;
                jobs.push_back({il, hot[i].second, next[(size_t) il]++});
            }
        }
        std::sort(jobs.begin(), jobs.end(), [&](const Job& a, const Job& b) {   // (file order: the disk reads ahead)
            const Layer &la = L[(size_t) a.layer], &lb = L[(size_t) b.layer];
            if (la.mx_file[0] != lb.mx_file[0]) return la.mx_file[0] < lb.mx_file[0];
            return la.mx_off[0] + (uint64_t) a.e * la.mx_stride[0] < lb.mx_off[0] + (uint64_t) b.e * lb.mx_stride[0];
        });
        std::atomic<size_t> ji{0};
        std::atomic<bool> bad{false};
        std::atomic<int64_t> done_b{0};
        std::vector<std::thread> th;
        for (int t = 0; t < 12 && !jobs.empty(); ++t)
            th.emplace_back([&] {
                for (size_t i; (i = ji.fetch_add(1)) < jobs.size() && !bad.load();) {
                    const Job& j = jobs[i];
                    const Layer& l = L[(size_t) j.layer];
                    for (int r = 0; r < 3; ++r) {
                        const size_t nb = (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
                        for (int X = 0; X < NH; ++X)
                            if (!pk.read_at(l.mx_file[r], l.mx_off[r] + (uint64_t) j.e * l.mx_stride[r] + (uint64_t) X * nb, nb,
                                            ram_part[(size_t) j.slot][(size_t) X] + cp_lay[(size_t) l.il][(size_t) r]))
                                bad.store(true);
                    }
                    const int64_t db = done_b.fetch_add((int64_t) l.blob) + (int64_t) l.blob;
                    if (o.verbose && i % 2048 == 0)
                        std::fprintf(stderr, "ds4: copying experts %zu / %zu (%.1f GB, %.2f GB/s)\n", i, jobs.size(), db / 1e9,
                                     db / 1e6 / std::max(1.0, now_ms() - t0));
                }
            });
        for (auto& t : th) t.join();
        if (bad.load()) {
            err = "copying the CPU lane's experts from the model's files failed";
            return false;
        }
        for (const Job& j : jobs) L[(size_t) j.layer].ram[(size_t) j.e] = j.slot;
        copy_mode = true;
        ram_held = (int) jobs.size();
        double cb = 0;
        for (const Job& j : jobs) cb += (double) L[(size_t) j.layer].blob;
        std::fprintf(stderr, "ds4: %zu of the CPU lane's %zu experts copied into RAM (%.1f GB, %d NUMA part%s) in %.1f s%s\n",
                     jobs.size(), hot.size(), cb / 1e9, NH, NH > 1 ? "s" : "", (now_ms() - t0) / 1000.0,
                     jobs.size() < hot.size() ? "; the others read in place" : "");
        // DS4_COPY_PIN: pin the copies, so the prompt path DMAs each expert from where it lies instead of copying it to
        // the pinned staging first (the pass streams 122 GB at 6.8 GB/s through the staging, where the RAM-to-staging
        // and DMA probes are 24 and 28 GB/s).  The CPU lane reads the same memory, and pinned 4 KB pages can cost its
        // TLB (an earlier registration measured -15% decode on a 12 GB card, where the CPU lane does nearly all of the
        // decode) - so AUTO (unset) pins when the cards hold at least half the routed experts: the CPU lane is then a
        // minority of the decode and the prompt stream is what hurts.  Measured on 4x R9700, 2.25 bpw (share 0.72),
        // three interleaved pairs: the second 8K prompt of a process 12.0/11.9/11.9 s against 28.8/13.8/12.3 s unpinned,
        // decode 35.42/35.50/34.9-36.0 against 35.43/35.45/34.9-36.0 tok/s (no cost).  DS4_COPY_PIN=0/1 forces it.
        // DS4_COPY_PIN_RO=1: cudaHostRegisterReadOnly (the GPU only reads it).
        // 0.2.7: DS4_COPY_PIN unset = ON when EITHER condition measured reliable holds: (a) the whole arena is on explicit 2 MB pages (a pin of a
        // THP arena registers 4 KB pages and the DMA rate was erratic on the 12 GB cell: 11.9 s one prompt, 30.4 s the next), or (b) the
        // cards hold at least half the routed experts on RDNA4 (4 x R9700: the pin was reliable in 4/4 runs there, see above).  =0 is the off
        // switch, =1 pins any arena.
        const bool pin_copies = strata::ds4::copy_pin_default(env_tri("DS4_COPY_PIN"), arena_all_huge, rdna4(), gpu_expert_share(),
                                                              !ram_chunks.empty(), defaults_026());
        if (pin_copies) {
            const bool ro = std::getenv("DS4_COPY_PIN_RO") != nullptr && std::atoi(std::getenv("DS4_COPY_PIN_RO")) != 0;
            const double tr = now_ms();
            bool ok = true;
            for (size_t i = 0; i < ram_chunks.size() && ok; ++i)
                ok = cudaHostRegister(ram_chunks[i], ram_chunk_bytes[i],
                                      ro ? cudaHostRegisterReadOnly : cudaHostRegisterPortable) == cudaSuccess;
            if (ok) {
                ram_pinned = true;
                std::fprintf(stderr, "ds4: the copied experts pinned in %.1f s (%s): the prompt path DMAs them in place\n",
                             (now_ms() - tr) / 1000.0, ro ? "read-only" : "read-write");
            } else {
                std::fprintf(stderr, "ds4: pinning the copied experts failed: the prompt path stages them\n");
            }
        }
        return true;
    }

    bool load_experts(std::string& err) {
        if (!L.empty() && L[0].lay.fmt == gg::kFmtMXFP4) return load_experts_mx(err);
        // every expert's region, in file order, by a few threads; VRAM ones through a pinned staging blob.  In place
        // (file_experts) the CPU lane's experts are not copied: the most-routed of them (the profile's share of their
        // layer) are read once, up to warm_bytes, so the page cache starts with them, and the VRAM ones' pages are
        // dropped from it again
        struct Job {
            int layer, e;
        };
        std::vector<Job> jobs;
        for (const Layer& l : L)
            for (int e = 0; e < l.n_exp; ++e)
                if (!file_experts || l.slot[(size_t) e] >= 0) jobs.push_back({l.il, e});
        uint64_t warm = 0;
        if (file_experts) {
            std::vector<std::pair<double, Job>> cand;
            for (const Layer& l : L) {
                const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
                double sum = 0;
                for (int e = 0; e < l.n_exp; ++e) sum += p[e];
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0) cand.push_back({sum > 0 ? p[e] / sum : 0.0, Job{l.il, e}});
            }
            std::stable_sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (const auto& c : cand) {
                const uint64_t b = L[(size_t) c.second.layer].region_bytes[(size_t) c.second.e];
                if (warm + b > warm_bytes) break;
                warm += b;
                jobs.push_back(c.second);
            }
        }
        std::sort(jobs.begin(), jobs.end(), [&](const Job& a, const Job& b) {
            const Layer &la = L[(size_t) a.layer], &lb = L[(size_t) b.layer];
            if (la.region_file[(size_t) a.e] != lb.region_file[(size_t) b.e])
                return la.region_file[(size_t) a.e] < lb.region_file[(size_t) b.e];
            return la.region_off[(size_t) a.e] < lb.region_off[(size_t) b.e];
        });
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::mutex emu;
        std::string ferr;
        const int nth = 12;
        size_t maxr = 0, maxb = 0;
        for (const Layer& l : L) {
            maxb = std::max(maxb, l.blob);
            for (uint64_t r : l.region_bytes) maxr = std::max<size_t>(maxr, r);
        }
        std::atomic<int64_t> done_bytes{0};
        const double t0 = now_ms();
        std::vector<std::thread> th;
        for (int t = 0; t < nth; ++t)
            th.emplace_back([&] {
                std::vector<uint8_t> region(maxr);
                uint8_t *stage = nullptr, *stage2 = nullptr;
                cudaHostAlloc((void**) &stage, maxb, cudaHostAllocPortable);
                cudaHostAlloc((void**) &stage2, maxb, cudaHostAllocPortable);
                int cur_dev = -1;
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= jobs.size() || failed.load()) break;
                    const Layer& l = L[(size_t) jobs[i].layer];
                    const int e = jobs[i].e;
                    if (!pk.read_at(l.region_file[(size_t) e], l.region_off[(size_t) e], l.region_bytes[(size_t) e],
                                    region.data())) {
                        std::lock_guard<std::mutex> lk(emu);
                        ferr = "reading layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
                        failed.store(true);
                        break;
                    }
                    if (l.slot[(size_t) e] >= 0) {
                        repack(l, region.data(), stage);
                        bool up = true;
                        for (int m = 0; m < l.nm && up; ++m) {   // (a group: each member its part)
                            const Layer& x = member(l, m);
                            const Dev& d = devs[(size_t) x.dev];
                            if (cur_dev != d.id) {
                                cudaSetDevice(d.id);
                                cur_dev = d.id;
                            }
                            if (l.nm > 1) blob_to_half(l, stage, m, stage2);
                            up = cudaMemcpy(x.vram_base + (size_t) l.slot[(size_t) e] * unit(x), l.nm > 1 ? stage2 : stage,
                                            unit(x), cudaMemcpyHostToDevice) == cudaSuccess;
                        }
                        if (!up) {
                            std::lock_guard<std::mutex> lk(emu);
                            ferr = "uploading layer " + std::to_string(l.il) + " expert " + std::to_string(e);
                            failed.store(true);
                            break;
                        }
                        if (file_experts)
                            os::advise(pk.fds[(size_t) l.region_file[(size_t) e]], l.region_off[(size_t) e], l.region_bytes[(size_t) e], false);
                    } else if (!file_experts) {
                        for (int X = 0; X < NH; ++X) repack_part(l, region.data(), X, l.hh[(size_t) e][(size_t) X]);
                    }
                    const int64_t db = done_bytes.fetch_add((int64_t) l.region_bytes[(size_t) e]) +
                                       (int64_t) l.region_bytes[(size_t) e];
                    if (o.verbose && (i % 1024) == 0)
                        std::fprintf(stderr, "ds4: experts %zu / %zu (%.1f GB, %.2f GB/s)\n", i, jobs.size(),
                                     db / 1e9, db / 1e6 / std::max(1.0, now_ms() - t0));
                }
                if (stage) cudaFreeHost(stage);
                if (stage2) cudaFreeHost(stage2);
            });
        for (auto& t : th) t.join();
        if (failed.load()) {
            err = ferr;
            return false;
        }
        if (file_experts)
            std::fprintf(stderr, "ds4: experts loaded in %.1f s (%.1f GB of the CPU lane's read ahead)\n",
                         (now_ms() - t0) / 1000.0, warm / 1e9);
        return true;
    }

    std::string describe() const {
        std::ostringstream s;
        for (size_t di = 0; di < devs.size(); ++di) {
            const Dev& d = devs[di];
            int a = -1, b = -1, ev = 0, et = 0;
            for (const Layer& l : L)
                for (int m = 0; m < l.nm; ++m)
                    if (member(l, m).dev == (int) di) {
                        if (a < 0) a = l.il;
                        b = l.il;
                        ev += l.n_vram;
                        et += l.n_exp;
                    }
            size_t fr = 0, tot = 0;
            cudaSetDevice(d.id);
            cudaMemGetInfo(&fr, &tot);
            const size_t gsz = groups.empty() ? 1 : groups[(size_t) d.grp].mem.size();
            char tp[64] = "";
            if (gsz > 1) std::snprintf(tp, sizeof tp, " (tensor parallel %zu, part %d)", gsz, d.mi);
            char line[384];
            std::snprintf(line, sizeof line,
                          "ds4: CUDA%d layers [%d, %d]%s%s: weights %.2f GB, experts %.2f GB (%d of %d in VRAM), caches "
                          "%.2f GB, %.0f MiB free\n",
                          d.id, a, b, tp, (int) di == head_dev ? " + head" : "", d.weights / 1e9, d.experts / 1e9, ev, et,
                          d.caches / 1e9, fr / 1048576.0);
            s << line;
        }
        int nh = 0;
        size_t hb_ = 0;
        for (const Layer& l : L) {
            nh += l.n_exp - l.n_vram;
            hb_ += (size_t) (l.n_exp - l.n_vram) * l.blob;
        }
        if (std::getenv("DS4_PLACE_HASH")) {   // debugging: which experts hold a VRAM slot, as one hash (is the placement the same from run to run?)
            uint64_t h = 1469598103934665603ull;
            for (const Layer& l : L)
                for (int e = 0; e < l.n_exp; ++e) { h ^= (uint64_t) (l.slot[(size_t) e] >= 0 ? 1 + l.il * 1000 + e : 0); h *= 1099511628211ull; }
            char hl[96];
            std::snprintf(hl, sizeof hl, "ds4: placement hash %016llx\n", (unsigned long long) h);
            s << hl;
        }
        char line[200];
        if (copy_mode)
            std::snprintf(line, sizeof line, "ds4: %d routed experts on the CPU lane (%.1f GB): %d copied into RAM (%d NUMA "
                                             "parts), the rest in place in the model's files; %d threads\n",
                          nh, hb_ / 1e9, ram_held, NH, cpu_threads());
        else if (file_experts)
            std::snprintf(line, sizeof line,
                          "ds4: %d routed experts on the CPU lane (%.1f GB, in place in the model's files), %d threads\n",
                          nh, hb_ / 1e9, cpu_threads());
        else
            std::snprintf(line, sizeof line, "ds4: %d routed experts on the CPU lane (%.1f GB of RAM, %d NUMA parts), %d threads\n", nh,
                          hb_ / 1e9, NH, cpu_threads());
        s << line;
        return s.str();
    }

    // The most the automatic store may take: all of the trunk's routed experts (the CPU lane's are a part of them; a store larger
    // than they are is never filled), never less than the old 48 GB.  A PC whose free RAM holds them all (the 251 GB R9700 box:
    // 115 GB of CPU-lane experts, 1 card, 2.00 set: 12.6 -> 21.4 tokens/s with the whole set resident, misses 15.4 -> 1.5 a token)
    // gets every one; the RAM that the room below leaves (free - 12 GB) is what bounds it elsewhere.
    // Whether the experts the GPUs will not hold (the CPU lane's: the pack's less the GPUs' share) all fit the RAM this process can get beside
    // kRamMargin, so that --mmap-experts auto copies them in (a copied expert is the CPU lane's own memory: no page cache to lose to the
    // Engram tables and the files' churn, no store indirection or miss path).  Measured, 1 x R9700 + 251 GB, 2.00 pack, warm: decode +10-15%.
    // --mmap-experts on/off and --ssd-cache N stand as given.
    bool cpu_lane_fits_ram() const {
        if (!pk.gguf || o.mmap_experts >= 0 || o.ssd_cache_gb >= 0.0 || std::getenv("DS4_STORE_GB") || std::getenv("DS4_COPY_GB")) return false;
        if (const char* v = std::getenv("DS4_COPY_ALL")) return std::atoi(v) != 0 && ram_avail() != ~uint64_t{0};
        // An integrated GPU's experts are already in this same RAM (and, above the RAM, in the page file it commits): copying
        // the CPU lane's in beside them takes the pages the GPU's experts live on.  Measured on Strix Halo (64 GB, 2.00 set):
        // 82 GB of experts on the GPU, then 53 GB copied in, and an 8K prompt did not finish in 11 minutes (the in-place path is
        // what the 8-hour soak ran).  DS4_COPY_ALL=1 still forces it; DS4_COPY_ALL=0 leaves it off everywhere.
        for (const Dev& d : devs)
            if (commit_charged(d)) return false;
        const uint64_t avail = ram_avail();
        if (avail == ~uint64_t{0}) return false;
        double exp_b = 0, room = 0, other_b = 0;
        const double share = gpu_expert_share(&exp_b, &room, &other_b);
        if (exp_b <= 0) return false;
        const double lane = exp_b * (1.0 - std::min(1.0, share));
        return copy_all_fits(lane, avail, (double) kRamMargin, commit_room());   // (Windows: the commit room too, not only the free RAM)
    }
    mutable bool auto_copy_all = false;   // (set by store_gb(): the auto size is the CPU lane's experts copied in, no demand store)
    double store_cap_gb() const {
        double b = 0;
        for (const auto& kt : pk.t)
            if (kt.first.find(".ffn.experts.") != std::string::npos && kt.first.rfind("mtp.", 0) != 0) b += (double) kt.second.bytes;
        return std::max(48.0, b / 1e9);
    }
    double store_gb() const {
        static double cached = -2.0;   // (decided once: the layout of the blobs depends on it)
        if (cached > -1.5) return cached;
        double gb = o.ssd_cache_gb;
        if (const char* v = std::getenv("DS4_STORE_GB")) gb = std::atof(v);
        if (!pk.gguf) gb = 0.0;
        else if (gb < 0.0 && cpu_lane_fits_ram()) {   // auto, and the RAM holds every CPU-lane expert beside kRamMargin: they are copied in at load
            auto_copy_all = true;                       // (copy_experts), the demand store would only be a second, smaller copy of them
            gb = 0.0;
        } else if (gb < 0.0) {   // auto: the RAM free now less 12 GB for the system and the owner's programs, all the experts at most
            const uint64_t avail = ram_avail();
            avail0_ = avail;
            const double room = avail == ~uint64_t{0} ? 0.0 : (double) avail / 1e9 - 12.0;
            gb = room >= 4.0 ? std::min(store_cap_gb(), room) : 0.0;
        }
        return cached = gb;
    }
    // The auto size is decided again here, after the weights are on the GPU: on a unified-memory machine (Strix Halo) the GPU's
    // allocations come out of the same RAM the store would use, and the first estimate (made at the layout, before any upload)
    // saw all of it free.  The explicit sizes (--ssd-cache, DS4_STORE_GB) stand.
    mutable uint64_t avail0_ = ~uint64_t{0};   // (free RAM when the first estimate was made, before the weights went to the GPU)
    double store_gb_now() const {
        const double gb = store_gb();
        if (gb <= 0.0 || o.ssd_cache_gb >= 0.0 || std::getenv("DS4_STORE_GB")) return gb;
        const uint64_t avail = ram_avail();
        if (avail == ~uint64_t{0}) return gb;
        // (the prompt buffers and the context's caches are touched after this point and, on such a machine, take RAM too: measured on
        // Strix Halo about 0.11 MB a token of context beyond 16384, a 40960 context ending 2.5 GB lower than a 16384 one)
        bool any_integrated = false;   // (only a device the driver calls integrated can share the RAM: a discrete card's pinned buffers also lower the free RAM, #4147)
        for (const Dev& d : devs) {
            int v = 0;
            if (cudaDeviceGetAttribute(&v, cudaDevAttrIntegrated, d.id) == cudaSuccess) any_integrated = any_integrated || v != 0;
            else (void) cudaGetLastError();
        }
        const bool uma = strata::ds4::store_gpu_shares_ram(any_integrated, avail0_, avail);
        const double ctx_gb = uma ? 1.1e-4 * std::max(0.0, (double) o.max_context - 16384.0) : 0.0;
        const double room = (double) avail / 1e9 - 12.0 - ctx_gb;
        const double now = room >= 4.0 ? std::min(store_cap_gb(), room) : 0.0;
        if (now < gb)
            std::fprintf(stderr, "ds4: RAM store: %.1f GB of RAM free after the GPU's weights: store sized %.1f GB, not %.1f\n",
                         (double) avail / 1e9, now, gb);
        return std::min(gb, now);
    }
    void init_store() {
        const double gb = store_gb_now();
        if (gb <= 0.0 || L.empty() || L[0].lay.fmt != gg::kFmtMXFP4 || !file_experts) return;
        if (o.ssd_cache_gb < 0.0 && !std::getenv("DS4_STORE_GB")) {   // (the auto size: say when it holds less than the CPU lane's experts)
            double exp_b = 0;
            const double share = gpu_expert_share(&exp_b), lane_gb = exp_b * (1.0 - std::min(1.0, share)) / 1e9;
            if (lane_gb > gb + 1.0 && o.verbose)
                std::fprintf(stderr, "ds4: RAM store: the auto size (%.1f GB) holds less than the CPU lane's experts (%.1f GB): the hottest are kept, the "
                                     "rest is read from the SSD as the prompt picks it (--ssd-cache N sets the size)\n", gb, lane_gb);
        }
        std::vector<ExpertStore::LayerGeo> geo((size_t) g.nl);   // (the trunk's layers: the drafter's bigger experts stay in place)
        for (const Layer& l : L) {
            if (l.il >= g.nl) continue;
            ExpertStore::LayerGeo& gl = geo[(size_t) l.il];
            gl.n_exp = l.n_exp;
            for (int r = 0; r < 3; ++r) {
                gl.file[r] = l.mx_file[r];
                gl.off[r] = l.mx_off[r];
                gl.stride[r] = l.mx_stride[r];
                gl.bytes[r] = mx_part_bytes(l, r);
                gl.tr[r] = l.lay.trellis[r];
            }
            gl.blob = l.blob;
        }
        ExpertStore::Cfg cfg;
        cfg.budget_bytes = (size_t) (gb * 1e9);
        cfg.threads = std::getenv("DS4_STORE_THREADS") ? std::atoi(std::getenv("DS4_STORE_THREADS")) : 8;
        cfg.floor_bytes = (size_t) (std::getenv("DS4_STORE_FLOOR_GB") ? std::atof(std::getenv("DS4_STORE_FLOOR_GB")) : 8.0) << 30;
        cfg.policy = (ExpertStore::Policy) (std::getenv("DS4_STORE_POLICY") ? std::atoi(std::getenv("DS4_STORE_POLICY")) : o.ssd_policy);
        cfg.ram_avail = [] { return ram_avail(); };
        cfg.reserve = (size_t) (std::getenv("DS4_STORE_RESERVE") ? std::atoi(std::getenv("DS4_STORE_RESERVE")) : 192);
        cfg.vis_percent = std::getenv("DS4_STORE_ZC_FRAC") ? std::atoi(std::getenv("DS4_STORE_ZC_FRAC")) : 50;
        // (the GPU computing the store's experts over PCIe is opt-in, DS4_STORE_ZC=1: measured slower than the CPU lane alone on a
        // 5070 / PCIe 5.0 - the staging copy sits in the layer's critical path - and its answers depend on the timing of which lane
        // took which expert, the CPU lane's are the mmap path's byte for byte)
        const bool want_zc = std::getenv("DS4_STORE_ZC") && std::atoi(std::getenv("DS4_STORE_ZC")) != 0 && devs.size() == 1 &&
                             L[0].nm == 1;
        if (want_zc) {   // the arena's chunks registered for the GPU to read in place (zero-copy over PCIe)
            const int dev0 = devs[0].id;
            {   // (a unified-memory GPU whose budget the weights already fill: the registered store is the same RAM, past the budget the GPU faults)
                cudaDeviceProp zp{};
                size_t zfree = 0, ztotal = 0;
                cudaSetDevice(dev0);
                if (cudaGetDeviceProperties(&zp, dev0) == cudaSuccess && zp.integrated && cudaMemGetInfo(&zfree, &ztotal) == cudaSuccess &&
                    (double) zfree < gb * 1e9)
                    std::fprintf(stderr,
                                 "ds4: DS4_STORE_ZC=1 on a GPU that shares the PC's RAM: the weights already fill its memory budget (%.1f GB left, the store would "
                                 "register up to %.1f GB more), so this will likely fail with a GPU fault. The store is the same RAM either way, zero-copy gains nothing here; "
                                 "leave DS4_STORE_ZC off.\n", (double) zfree / 1e9, gb);
            }
            cfg.map_chunk = [dev0](void* h, size_t n) -> void* {
                cudaSetDevice(dev0);
                unsigned reg_flags = cudaHostRegisterMapped | cudaHostRegisterPortable;
#if defined(STRATA_USE_HIP)
                // (test arm: DS4_ZC_COARSE=1, hipExtHostRegisterCoarseGrained, which ROCm Core SDK 10.1 honors on Windows; the store is read-only to the GPU)
                if (std::getenv("DS4_ZC_COARSE") && std::atoi(std::getenv("DS4_ZC_COARSE")) != 0) reg_flags |= 0x8u;
#endif
                if (cudaHostRegister(h, n, reg_flags) != cudaSuccess) {
                    (void) cudaGetLastError();
                    return nullptr;
                }
                void* d = nullptr;
                if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
                    (void) cudaGetLastError();
                    cudaHostUnregister(h);
                    return nullptr;
                }
                if (d != h) std::fprintf(stderr, "ds4: RAM store: chunk host %p -> device %p (not the identity)\n", h, d);
                return d;
            };
            cfg.unmap_chunk = [dev0](void* h) {
                cudaSetDevice(dev0);
                cudaHostUnregister(h);
            };
        }
        bool direct = true;
        for (os::fd_t f : pk.dfds) direct = direct && f != os::kBadFd;
        auto st = std::make_unique<ExpertStore>();
        if (!st->init(cfg, geo, pk.dfds, pk.pfds)) {
            std::fprintf(stderr, "ds4: --ssd-cache: the store could not start\n");
            return;
        }
        store = std::move(st);
        // (the router lookahead's guesses are not read ahead with the store: measured 4.5 against 4.7 tok/s, 62% of the guesses are right and
        // the SSD is the narrow part; DS4_AHEAD=1 turns them on)
        if (std::getenv("DS4_AHEAD") == nullptr) ahead_mode = 0;
        for (Layer& l : L)
            if (l.il < g.nl && want_zc) l.zc.assign((size_t) l.n_exp, 0);
        std::fprintf(stderr, "ds4: RAM store: up to %zu slots of %.1f MB (%.1f GB), policy %d, %d reader threads, %s reads\n",
                     store->slots_max(), store->slot_bytes() / 1e6, store->slots_max() * (double) store->slot_bytes() / 1e9,
                     (int) cfg.policy, cfg.threads, direct ? "unbuffered" : "BUFFERED (no direct handle)");
        if (want_zc) {
            if (!store->gpu_mapped()) {
                std::fprintf(stderr, "ds4: RAM store: the arena could not be mapped for the GPU: zero-copy off\n");
                for (Layer& l : L) l.zc.clear();
            } else {
                zc_stride = (store->max_blob() + 4095) & ~(size_t) 4095;
                cudaSetDevice(devs[0].id);
                if (cudaMalloc((void**) &zc_stage_buf, zc_stride * kPicks) != cudaSuccess) {
                    (void) cudaGetLastError();
                    zc_stage_buf = nullptr;
                    for (Layer& l : L) l.zc.clear();
                } else {
                    for (Layer& l : L)
                        if (!l.zc.empty() && cudaMalloc((void**) &l.zc_tab, (size_t) l.n_exp * 8) == cudaSuccess) cudaMemset(l.zc_tab, 0, (size_t) l.n_exp * 8);
                        else if (!l.zc.empty()) {
                            (void) cudaGetLastError();
                            l.zc.clear();
                        }
                }
            }
            std::fprintf(stderr, "ds4: RAM store: the GPU takes its slots over PCIe (%s), %zu slots kept evictable\n",
                         zc_stage_buf != nullptr ? "copied into a staging area at the PCIe rate first" : "off", cfg.reserve);
        }
    }
    // the profile's hottest experts (VRAM's left out) read into the store before the first token (DS4_STORE_PRELOAD=0: not)
    void preload_store() {
        if (!store) return;
        if (std::getenv("DS4_STORE_PRELOAD") && std::atoi(std::getenv("DS4_STORE_PRELOAD")) == 0) return;
        std::vector<std::pair<double, std::pair<int, int>>> cand;
        double tot = 0;
        for (const Layer& l : L) {
            if (l.il >= g.nl) continue;
            const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
            for (int e = 0; e < l.n_exp; ++e) tot += p[e];
        }
        if (tot <= 0) return;   // (no profile yet: the first tokens fill it)
        for (const Layer& l : L) {
            if (l.il >= g.nl) continue;
            const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] < 0 && p[e] > 0) cand.push_back({(double) p[e] / (double) l.blob, {l.il, e}});
        }
        std::stable_sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<std::pair<int, int>> list;
        list.reserve(cand.size());
        for (const auto& c : cand) list.push_back(c.second);
        // The slots the CPU lane's experts leave over: the experts the VRAM cache holds that the profile counts least (the cache gives
        // up its coldest to the CPU lane, and a store without them reads each from the SSD while the CPU lane waits - 1.8 misses a
        // token on the 251 GB R9700 box, 2.9 ms of a 49 ms token), then the lane's experts the profile never saw.  Last in the list: they
        // take only the slots the likely ones left.  DS4_STORE_SPARE=0: off.
        if (strata::ds4::store_feature_default(env_tri("DS4_STORE_SPARE"), defaults_026())) {
            std::vector<std::pair<double, std::pair<int, int>>> vr, zero;
            for (const Layer& l : L) {
                if (l.il >= g.nl) continue;
                const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
                for (int e = 0; e < l.n_exp; ++e) {
                    if (l.slot[(size_t) e] >= 0) vr.push_back({(double) p[e] / (double) l.blob, {l.il, e}});
                    else if (p[e] == 0) zero.push_back({0.0, {l.il, e}});
                }
            }
            std::stable_sort(vr.begin(), vr.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& c : vr) list.push_back(c.second);
            for (const auto& c : zero) list.push_back(c.second);
        }
        const double t0 = now_ms();
        // (with the spare experts in the list the reserve is theirs: last in the list, so the oldest ticks, so the first a demand read replaces)
        const bool spare_on = strata::ds4::store_feature_default(env_tri("DS4_STORE_SPARE"), defaults_026());
        const size_t leave = spare_on ? 0 : (size_t) (std::getenv("DS4_STORE_RESERVE") ? std::atoi(std::getenv("DS4_STORE_RESERVE")) : 192) + 64;
        // The first 48 GB (what a store held before it could take every expert) are read before the first token; the rest of the
        // hottest-first list is read after it, in pieces of 128 experts on a thread of its own, behind the model's own reads (a
        // guess's priority) - the same bytes.  Opt-in (DS4_STORE_BG=1): "ready" comes 9 s sooner on the 251 GB R9700 box (24 s against 34), but a prompt sent at
        // once then waits for the same SSD (200 tokens: 18.7 s against 9.0), so the first answer starts no sooner and decode after
        // it measured 3.7% lower; it only pays when the first prompt comes more than ~10 s after ready.
        const size_t first = !(std::getenv("DS4_STORE_BG") && std::atoi(std::getenv("DS4_STORE_BG")) == 1)
                                 ? list.size()
                                 : std::min(list.size(), (size_t) (48e9 / (double) std::max<size_t>(1, L[0].blob)));
        const std::vector<std::pair<int, int>> head(list.begin(), list.begin() + (long) first), tail(list.begin() + (long) first, list.end());
        const size_t n = store->preload(head, leave);
        std::fprintf(stderr, "ds4: RAM store: %zu of the profile's hottest experts read in (%.1f GB) in %.1f s%s\n", n,
                     (double) n * (double) L[0].blob / 1e9, (now_ms() - t0) / 1000.0,
                     tail.empty() ? "" : "; the rest is read in the background");
        if (tail.empty()) return;
        store_bg = std::thread([this, tail, leave] {
            const double tb = now_ms();
            size_t got = 0;
            std::vector<std::pair<int, int>> piece;
            for (size_t i = 0; i < tail.size() && !quit.load(std::memory_order_relaxed); i += 128) {
                piece.assign(tail.begin() + (long) i, tail.begin() + (long) std::min(tail.size(), i + 128));
                got += store->preload(piece, leave);
            }
            std::fprintf(stderr, "ds4: RAM store: %zu more experts (%.1f GB) read in the background in %.1f s\n", got,
                         (double) got * (double) L[0].blob / 1e9, (now_ms() - tb) / 1000.0);
        });
    }
    // at a window boundary: the store's ready slots become the device's (zero-copy) experts, taken back when it needs the room;
    // each touched layer's whole table is written again (VRAM slots, the store's, 0), in stream order before the next window
    void store_sync() {
        if (!store || !store->gpu_mapped()) return;
        std::vector<ExpertStore::Vis> ev;
        store->sync(ev, [&](int layer, int e) { return L[(size_t) layer].slot[(size_t) e] < 0; });
        if (ev.empty()) return;
        std::vector<char> dirty(L.size(), 0);
        for (const auto& v : ev) {
            Layer& l = L[(size_t) v.layer];
            if (l.zc.empty()) continue;
            l.zc[(size_t) v.e] = v.dev != 0 ? 1 : 0;
            dirty[(size_t) v.layer] = 1;
        }
        for (Layer& l : L) {
            if (!dirty[(size_t) l.il]) continue;
            if (l.zc_tab == nullptr) continue;
            std::vector<unsigned long long> tab((size_t) l.n_exp, 0ull);
            for (int e = 0; e < l.n_exp; ++e)
                if (l.zc[(size_t) e]) {
                    const int s = l.slot[(size_t) e] < 0 ? store->find(l.il, e) : -1;
                    tab[(size_t) e] = s >= 0 ? store->dev_blob(s) : 0ull;
                    if (s < 0) l.zc[(size_t) e] = 0;
                }
            cudaSetDevice(devs[0].id);
            cudaMemcpyAsync(l.zc_tab, tab.data(), tab.size() * 8, cudaMemcpyHostToDevice, devs[0].st);
        }
    }

    int cpu_threads() const {
        int n = 0;
        for (const auto& pl : pools) n += pl->size() - 1;
        return n + 1;
    }
    // the CPU lane's workers: a pool per NUMA node (NH of them) on its cores; in place with one part, one pool over
    // every node's cores (the pages lie where the cache put them)
    void make_pools() {
        std::vector<std::vector<int>> cs = cores;
        if ((int) cs.size() < NH && !cs.empty()) {   // (DS4_COPY_NUMA=2 on one node, a test: its cores split in NH)
            std::vector<int> all = cs[0];
            cs.assign((size_t) NH, {});
            for (size_t i = 0; i < all.size(); ++i) cs[i * (size_t) NH / all.size()].push_back(all[i]);
        }
        static const int lane_free = [] {
            const char* v = std::getenv("STRATA_DS4_LANE_FREE");
            return v != nullptr ? std::max(0, std::atoi(v)) : 2;
        }();
        size_t lane_cpus = 0;
        for (const auto& v : cs) lane_cpus += v.size();
        const bool lane_takes_all = lane_cpus >= (size_t) std::max(1u, std::thread::hardware_concurrency());
        for (int X = 0; X < (int) cs.size() && X < std::max(NH, 1); ++X) {
            std::vector<int> c = cs[(size_t) X];
            if (file_experts && NH == 1)
                for (size_t Y = 1; Y < cs.size(); ++Y) c.insert(c.end(), cs[Y].begin(), cs[Y].end());
            if (X == 0 && !c.empty()) c.erase(c.begin());   // the host thread's core (it helps node 0's pool)
            // cores of the first pool left free beside the host thread's, for the thread that launches the decode's kernels
            // (and waits on the device) and whatever else the PC runs: with a worker spinning on every other logical CPU
            // that thread waits for a core (Project Maya #359, a 20-core Windows 11 PC: decode 28.3 -> 55.5 tok/s).  Only
            // when the lane would take every logical CPU and --cpu-threads is not given.
            // STRATA_DS4_LANE_FREE: the cores left free (2 by default, 0: as before)
            if (X == 0 && o.cpu_threads <= 0 && lane_free > 0 && (int) c.size() > lane_free + 1 && lane_takes_all)
                c.resize(c.size() - (size_t) lane_free);
            int nw = (int) c.size();
            if (o.cpu_threads > 0) nw = std::max(X == 0 ? 0 : 1, std::min(nw, o.cpu_threads / NH - (X == 0 ? 1 : 0)));
            pools.push_back(std::make_unique<Pool>(nw, c, 20000));   // spin through a token's layers (20 ms)
        }
        (void) gg::mx_tier();   // (measured here, not on the first token)
        if (o.verbose) {
            int nthr = 0;
            for (auto& p : pools) nthr += p->size();
            std::fprintf(stderr, "ds4: CPU lane: %d threads, MXFP4 rows %s (%s)\n", nthr, gg::mx_tier_name(gg::mx_tier()),
                         gg::mx_tier_report());
        }
    }

    void reset() {
        pos = 0;
        toks.clear();
        ctoks.clear();
        slot_held = false;
        for (Check& c0 : ck) c0.pos = -1;
        for (Layer& l0 : L)
            for (int m = 0; m < l0.nm; ++m) {
                Layer& l = member(l0, m);
                Dev& d = devs[(size_t) l.dev];
                cudaSetDevice(d.id);
                cudaMemsetAsync(l.win, 0, (size_t) g.win * win_rb(), d.st);
                if (l.c_kv) cudaMemsetAsync(l.c_kv, 0, (size_t) l.ratio * g.hd * 4, d.st);
                if (l.c_score) cudaMemsetAsync(l.c_score, 0, (size_t) l.ratio * g.hd * 4, d.st);
            }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
    }

    // the state at pos into checkpoint `slot` (the window rings and the compressors' partial groups: the rest of the
    // state is indexed by position, and what lies past a restored position is written again)
    void checkpoint(int slot) {
        Check& c = ck[slot & (kChecks - 1)];
        c.pos = pos;
        c.win.resize(L.size());
        c.comp.resize(L.size() * 2);
        for (size_t i = 0; i < L.size(); ++i) {
            Layer& l = L[i];
            Dev& d = devs[(size_t) l.dev];
            cudaSetDevice(d.id);
            c.win[i].resize((size_t) g.win * win_rb());
            cudaMemcpyAsync(c.win[i].data(), l.win, c.win[i].size(), cudaMemcpyDeviceToHost, d.st);
            if (l.c_kv) {
                c.comp[2 * i].resize((size_t) l.ratio * g.hd);
                c.comp[2 * i + 1].resize((size_t) l.ratio * g.hd);
                cudaMemcpyAsync(c.comp[2 * i].data(), l.c_kv, c.comp[2 * i].size() * 4, cudaMemcpyDeviceToHost, d.st);
                cudaMemcpyAsync(c.comp[2 * i + 1].data(), l.c_score, c.comp[2 * i + 1].size() * 4,
                                cudaMemcpyDeviceToHost, d.st);
            }
        }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
    }

    // the checkpoint at or before position limit (the state's tokens agree with a new prompt up to there), or -1
    int best_check(int64_t limit) const {
        int b = -1;
        for (int i = 0; i < kChecks; ++i)
            if (ck[i].pos >= 0 && ck[i].pos <= limit && ck[i].pos <= pos && (b < 0 || ck[i].pos > ck[b].pos)) b = i;
        return b;
    }
    int64_t restorable(int64_t limit) const {
        const int b = best_check(limit);
        return b < 0 ? -1 : ck[b].pos;
    }

    // checkpoint 1 (the position before the last prompt's end) becomes the anchor too
    void pin() {
        if (ck[1].pos > 0) ck[2] = ck[1];
    }
    // the checkpoints the state can go back to (what a slot of it keeps), latest first
    std::vector<int64_t> check_positions() const {
        std::vector<int64_t> v;
        for (const Check& c : ck)
            if (c.pos > 0 && c.pos <= pos && c.win.size() == L.size() &&
                std::find(v.begin(), v.end(), c.pos) == v.end())
                v.push_back(c.pos);
        std::sort(v.rbegin(), v.rend());
        return v;
    }
    // the checkpoints a slot keeps (each with its role: 0, 1 or the anchor)
    std::vector<int> slot_checks() const {
        std::vector<int> v;
        for (int i = 0; i < kChecks; ++i)
            if (ck[i].pos > 0 && ck[i].pos <= pos && ck[i].win.size() == L.size()) v.push_back(i);
        return v;
    }
    size_t check_bytes() const {
        size_t n = 16;
        for (const Layer& l : L) n += (size_t) g.win * win_rb() + (l.c_kv ? 2 * (size_t) l.ratio * g.hd * 4 : 0);
        return n;
    }

    bool restore(int64_t limit) {
        const int b = best_check(limit);
        if (b < 0) return false;
        const Check& c = ck[b];
        for (size_t i = 0; i < L.size(); ++i)
            for (int m = 0; m < L[i].nm; ++m) {   // (a group's members hold the same state: the lead's copy)
                Layer& l = member(L[i], m);
                Dev& d = devs[(size_t) l.dev];
                cudaSetDevice(d.id);
                cudaMemcpyAsync(l.win, c.win[i].data(), c.win[i].size(), cudaMemcpyHostToDevice, d.st);
                if (l.c_kv) {
                    cudaMemcpyAsync(l.c_kv, c.comp[2 * i].data(), c.comp[2 * i].size() * 4, cudaMemcpyHostToDevice, d.st);
                    cudaMemcpyAsync(l.c_score, c.comp[2 * i + 1].data(), c.comp[2 * i + 1].size() * 4,
                                    cudaMemcpyHostToDevice, d.st);
                }
            }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
        pos = c.pos;
        toks.resize((size_t) pos);
        ctoks.resize((size_t) pos);
        slot_held = false;
        for (Check& o2 : ck)   // a later checkpoint saw tokens the state no longer has
            if (o2.pos > pos) o2.pos = -1;
        return true;
    }

    // ---- the kept conversations (a slot): the state out of the model and back, through a file or a host buffer.
    // A prompt that continues the state is reused in place (checkpoint / restore, the compressed KV left where it is);
    // a conversation that leaves the model for another one is a slot - the whole state: the position, its tokens,
    // every window ring (the DSpark blocks' included), the compressors' partial groups and each KV source's compressed
    // KV and index keys up to it (rows 0 .. pos / ratio, the groups its position completed; one the prompt has not
    // completed yet lives in the compressor's partial group).  A tensor parallel group's members hold the same state,
    // so one member's ring is saved and written back into all of them.  Text only: a conversation that holds pictures
    // is not a slot (the Engram history's rows are the tokens' again, no -1 among them).
    static constexpr uint64_t kSlotMagic = 0x31544f4c53345344ull;   // "DS4SLOT1" (little endian)

    int64_t src_rows(const Src& s) const { return pos / std::max(1, s.ratio); }

    // the slot's bytes: the header, the tokens, every layer's ring and partial group, every source's rows, the end
    size_t slot_bytes() const {
        size_t n = 8 + 8 + slot_identity().size() + 8 + (size_t) pos * 4;
        for (const Layer& l : L) n += 8 + (size_t) g.win * win_rb() + 8 + (l.c_kv ? 2 * (size_t) l.ratio * g.hd * 4 : 0);
        for (const Src& s : srcs) n += 8 + (size_t) src_rows(s) * (ckv_rb() + ik_rb());
        n += 8 + slot_checks().size() * check_bytes();   // (the checkpoints: what the slot can be taken back at)
        return n + 8;
    }

    uint64_t slot_save(std::vector<uint8_t>& out, std::string& err) {
        if (slot_held) {
            err = "it is already set aside";
            return 0;
        }
        if (pos <= 0) {
            err = "there is no conversation";
            return 0;
        }
        if ((int64_t) toks.size() != pos) {
            err = "the state's tokens do not match its position";
            return 0;
        }
        if (!imgs.empty()) {
            err = "it holds pictures";
            return 0;
        }
        const std::string id = slot_identity();
        out.assign(slot_bytes(), 0);
        uint8_t* p = out.data();
        size_t off = 0;
        const auto put = [&](const void* s, size_t b) {
            std::memcpy(p + off, s, b);
            off += b;
        };
        const uint64_t il = id.size(), n = (uint64_t) pos;
        put(&kSlotMagic, 8);
        put(&il, 8);
        put(id.data(), (size_t) il);
        put(&n, 8);
        put(toks.data(), (size_t) n * 4);
        for (Dev& d : devs) {   // (the devices' work ends before their state is read)
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
        for (size_t i = 0; i < L.size(); ++i) {
            const Layer& l = L[i];
            const Dev& d = devs[(size_t) l.dev];
            cudaSetDevice(d.id);
            const uint64_t wb = (uint64_t) g.win * win_rb();
            put(&wb, 8);
            if (cudaMemcpy(p + off, l.win, (size_t) wb, cudaMemcpyDeviceToHost) != cudaSuccess) {
                err = l.name + "window ring: " + cudaGetErrorString(cudaGetLastError());
                return 0;
            }
            off += (size_t) wb;
            const uint64_t cn = l.c_kv ? (uint64_t) l.ratio * (uint64_t) g.hd : 0;
            put(&cn, 8);
            if (cn > 0) {
                const size_t cb = (size_t) cn * 4;
                if (cudaMemcpy(p + off, l.c_kv, cb, cudaMemcpyDeviceToHost) != cudaSuccess ||
                    cudaMemcpy(p + off + cb, l.c_score, cb, cudaMemcpyDeviceToHost) != cudaSuccess) {
                    err = l.name + "compressor's group: " + cudaGetErrorString(cudaGetLastError());
                    return 0;
                }
                off += 2 * cb;
            }
        }
        for (const Src& s : srcs) {
            const uint64_t rows = (uint64_t) src_rows(s);
            put(&rows, 8);
            if (rows == 0) continue;
            const int di = L[(size_t) s.layer].dev;
            const uint8_t* ckv = s.ckv[(size_t) di];
            const uint8_t* ik = s.ik[(size_t) di];   // (a pack whose sources are not index sources: none, zeros)
            if (ckv == nullptr) {
                err = "source layer " + std::to_string(s.layer) + "'s rows are not on its device";
                return 0;
            }
            const size_t cb = (size_t) rows * ckv_rb(), ib = (size_t) rows * ik_rb();
            cudaSetDevice(devs[(size_t) di].id);
            if (cudaMemcpy(p + off, ckv, cb, cudaMemcpyDeviceToHost) != cudaSuccess ||
                (ik != nullptr && cudaMemcpy(p + off + cb, ik, ib, cudaMemcpyDeviceToHost) != cudaSuccess)) {
                err = "source layer " + std::to_string(s.layer) + "'s compressed KV: " + cudaGetErrorString(cudaGetLastError());
                return 0;
            }
            off += cb + ib;
        }
        {
            const std::vector<int> cs = slot_checks();
            const uint64_t nc = cs.size();
            put(&nc, 8);
            for (int k : cs) {
                const Check& c = ck[k];
                const uint64_t role = (uint64_t) k, cp = (uint64_t) c.pos;
                put(&role, 8);
                put(&cp, 8);
                for (size_t i = 0; i < L.size(); ++i) {
                    put(c.win[i].data(), (size_t) g.win * win_rb());
                    if (L[i].c_kv) {
                        put(c.comp[2 * i].data(), (size_t) L[i].ratio * g.hd * 4);
                        put(c.comp[2 * i + 1].data(), (size_t) L[i].ratio * g.hd * 4);
                    }
                }
            }
        }
        if (off + 8 != out.size()) {
            err = "the state's size does not add up";
            return 0;
        }
        put(&kSlotMagic, 8);
        slot_held = true;
        return (uint64_t) off;
    }

    bool slot_load(const uint8_t* data, size_t bytes, int64_t n_pos, std::string& err) {
        size_t off = 0;
        bool wrote = false, ok = false;
        // a load that stops after some of another conversation's state went into the layers leaves a state no
        // checkpoint can repair: it is reset (the prompt is read from token 0), never served from the mix
        struct FailClosed {
            Impl& p;
            const bool& wrote;
            const bool& ok;
            ~FailClosed() {
                if (!ok && wrote) p.reset();
            }
        } guard{*this, wrote, ok};
        const auto room = [&](size_t b) { return b <= bytes && off <= bytes - b; };
        uint64_t magic = 0, il = 0, n = 0;
        if (!room(16)) {
            err = "the slot is cut short";
            return false;
        }
        std::memcpy(&magic, data + off, 8);
        std::memcpy(&il, data + off + 8, 8);
        off += 16;
        if (magic != kSlotMagic || il == 0 || il > (1u << 20)) {
            err = "not a saved conversation";
            return false;
        }
        if (!room((size_t) il + 8)) {
            err = "the slot is cut short";
            return false;
        }
        {
            std::string id((size_t) il, '\0');
            std::memcpy(id.data(), data + off, (size_t) il);
            off += (size_t) il;
            std::memcpy(&n, data + off, 8);
            off += 8;
            if (id != slot_identity()) {
                err = "another model's, or another engine version's, conversation";
                return false;
            }
        }
        if (n != (uint64_t) n_pos || n_pos <= 0 || n_pos > o.max_context || !room((size_t) n * 4)) {
            err = "the conversation's position is not this prompt's";
            return false;
        }
        std::vector<int32_t> toks_in((size_t) n);
        std::memcpy(toks_in.data(), data + off, (size_t) n * 4);
        off += (size_t) n * 4;
        for (Dev& d : devs) {   // (their work ends before the state is written over)
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
        for (size_t i = 0; i < L.size(); ++i) {
            Layer& l = L[i];
            const Dev& d = devs[(size_t) l.dev];
            cudaSetDevice(d.id);
            uint64_t wb = 0, cn = 0;
            if (!room(8)) {
                err = "the slot is cut short";
                return false;
            }
            std::memcpy(&wb, data + off, 8);
            off += 8;
            if (wb != (uint64_t) g.win * win_rb() || !room((size_t) wb + 8)) {
                err = l.name + "window ring of another size";
                return false;
            }
            wrote = true;
            for (int m = 0; m < l.nm; ++m) {
                Layer& lm = member(l, m);
                cudaSetDevice(devs[(size_t) lm.dev].id);   // (a tensor-parallel group's members sit on different GPUs: #4350)
                if (cudaMemcpy(lm.win, data + off, (size_t) wb, cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = l.name + "window ring: " + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
            }
            off += (size_t) wb;
            std::memcpy(&cn, data + off, 8);
            off += 8;
            if (cn != (l.c_kv ? (uint64_t) l.ratio * (uint64_t) g.hd : 0)) {
                err = l.name + "compressor's group of another size";
                return false;
            }
            if (cn > 0) {
                const size_t cb = (size_t) cn * 4;
                if (!room(2 * cb)) {
                    err = "the slot is cut short";
                    return false;
                }
                for (int m = 0; m < l.nm; ++m) {
                    Layer& lm = member(l, m);
                    cudaSetDevice(devs[(size_t) lm.dev].id);
                    if (cudaMemcpy(lm.c_kv, data + off, cb, cudaMemcpyHostToDevice) != cudaSuccess ||
                        cudaMemcpy(lm.c_score, data + off + cb, cb, cudaMemcpyHostToDevice) != cudaSuccess) {
                        err = l.name + "compressor's group: " + cudaGetErrorString(cudaGetLastError());
                        return false;
                    }
                }
                off += 2 * cb;
            }
        }
        for (const Src& s : srcs) {
            uint64_t rows = 0;
            if (!room(8)) {
                err = "the slot is cut short";
                return false;
            }
            std::memcpy(&rows, data + off, 8);
            off += 8;
            if (rows != (uint64_t) (n_pos / std::max(1, s.ratio))) {
                err = "source layer " + std::to_string(s.layer) + "'s rows are not this conversation's";
                return false;
            }
            if (rows == 0) continue;
            const size_t cb = (size_t) rows * ckv_rb(), ib = (size_t) rows * ik_rb();
            if (!room(cb + ib)) {
                err = "the slot is cut short";
                return false;
            }
            for (size_t di = 0; di < devs.size(); ++di) {
                if (s.ckv[di] == nullptr && s.ik[di] == nullptr) continue;
                cudaSetDevice(devs[di].id);
                if (s.ckv[di] != nullptr && cudaMemcpy(s.ckv[di], data + off, cb, cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = "source layer " + std::to_string(s.layer) + "'s compressed KV: " + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
                if (s.ik[di] != nullptr &&
                    cudaMemcpy(s.ik[di], data + off + cb, ib, cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = "source layer " + std::to_string(s.layer) + "'s index keys: " + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
            }
            off += cb + ib;
        }
        Check got[kChecks];   // the slot's checkpoints: the positions it can be taken back at
        {
            uint64_t nc = 0;
            if (!room(8)) {
                err = "the slot is cut short";
                return false;
            }
            std::memcpy(&nc, data + off, 8);
            off += 8;
            if (nc > (uint64_t) kChecks) {
                err = "the slot's checkpoints are damaged";
                return false;
            }
            for (uint64_t k = 0; k < nc; ++k) {
                uint64_t role = 0, cp = 0;
                if (!room(16)) {
                    err = "the slot is cut short";
                    return false;
                }
                std::memcpy(&role, data + off, 8);
                std::memcpy(&cp, data + off + 8, 8);
                off += 16;
                if (role >= (uint64_t) kChecks || got[role].pos >= 0 || cp == 0 || cp > (uint64_t) n_pos) {
                    err = "the slot's checkpoints are damaged";
                    return false;
                }
                Check& c = got[role];
                c.pos = (int64_t) cp;
                c.win.resize(L.size());
                c.comp.resize(L.size() * 2);
                for (size_t i = 0; i < L.size(); ++i) {
                    const size_t wb = (size_t) g.win * win_rb(), cb = L[i].c_kv ? (size_t) L[i].ratio * g.hd * 4 : 0;
                    if (!room(wb + 2 * cb)) {
                        err = "the slot is cut short";
                        return false;
                    }
                    c.win[i].assign(data + off, data + off + wb);
                    off += wb;
                    if (cb > 0) {
                        c.comp[2 * i].resize(cb / 4);
                        c.comp[2 * i + 1].resize(cb / 4);
                        std::memcpy(c.comp[2 * i].data(), data + off, cb);
                        std::memcpy(c.comp[2 * i + 1].data(), data + off + cb, cb);
                        off += 2 * cb;
                    }
                }
            }
        }
        uint64_t end = 0;
        if (!room(8)) {
            err = "the slot is cut short";
            return false;
        }
        std::memcpy(&end, data + off, 8);
        off += 8;
        if (end != kSlotMagic) {
            err = "the slot does not end where it should";
            return false;
        }
        pos = n_pos;
        toks.swap(toks_in);
        ctoks.resize((size_t) pos);
        for (size_t i = 0; i < (size_t) pos; ++i)
            ctoks[i] = toks[i] < pk.eg_vocab ? pk.eg_map[(size_t) toks[i]] : 0;
        for (int k = 0; k < kChecks; ++k) ck[k] = std::move(got[k]);   // (the slot's own; the old ones held the state
                                                               // this one replaced)
        slot_held = false;
        ok = true;
        return true;
    }

    std::string slot_identity() const {
        const auto size_of = [](const std::string& p) {
            std::error_code ec;
            const auto s = std::filesystem::file_size(p, ec);
            return ec ? 0ull : (unsigned long long) s;
        };
        std::ostringstream id;
        id << "strata-ds4-session-3 engine=" << STRATA_VERSION << " mtp=" << g.n_mtp << " dim=" << g.dim << " hd=" << g.hd
           << " win=" << g.win << " ratios=";
        for (int r : g.ratios) id << r << ",";
        id << " kv=";
        for (int v : g.kv_src) id << v << ",";
        id << " ix=";
        for (int v : g.ix_src) id << v << ",";
        id << " ds4.txt:" << size_of(pk.dir + "/ds4.txt") << " engram.bin:" << size_of(pk.dir + "/engram.bin")
           << " tensors.txt:" << size_of(pk.dir + "/tensors.txt");
        for (size_t i = 0; i < pk.files.size(); ++i)
            id << " " << pk.files[i] << ":" << os::file_size(pk.fds[i]);
        return id.str();
    }

    size_t bounce_bytes() const {
        return (size_t) kWin * ((size_t) g.hc * g.dim * 4 + 8 * 4 + topk_stride() * 4 + cand_stride() * 4);
    }
    size_t topk_stride() const { return (size_t) std::max(g.ix_topk, 32); }
    size_t cand_stride() const { return (size_t) std::max(g.cand_topk, 32); }
    size_t bm_stride() const { return (size_t) (o.max_context + 64) / std::max(1, g.cand_blk) + 64; }
    static constexpr int kSlots = kWin * kPicks;   // the CPU lane's (token, expert) rows a window can need

    // ---- the vision tower (the checkpoint's vision.* / aligner.*) on the first GPU, its weights streamed from the
    // host; a prompt's pictures as their spans' rows (set_images)
    VisionTower vt;
    VisionGeo vg;
    int img_tok = -1;
    std::string vision_err;              // why the tower did not load (pictures off), or empty
    std::vector<uint16_t> img_mark[3];   // image_start, image_newline, image_end: BF16 [dim]
    std::vector<Image> imgs;             // the pictures of the prompt being fed, by start
    // a BF16 (or, rounded to nearest even, F32) tensor of n values into dst
    bool read_bf16(const Tensor& t, size_t n, void* dst) {
        if (t.dtype == "BF16") return t.bytes == n * 2 && pk.read(t, dst);
        if (t.dtype != "F32" || t.bytes != n * 4) return false;
        std::vector<float> f(n);
        if (!pk.read(t, f.data())) return false;
        uint16_t* o = (uint16_t*) dst;
        for (size_t i = 0; i < n; ++i) {
            uint32_t u;
            std::memcpy(&u, &f[i], 4);
            o[i] = (uint16_t) ((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
        }
        return true;
    }
    bool load_vision(std::string& err) {
        vg.layers = (int) pk.num("vision.num_hidden_layers", 0);
        const int tok = (int) pk.num("image_token_id", -1);
        if (vg.layers <= 0 || tok < 0 || pk.get("vision.patch_embed.proj.weight") == nullptr) return true;   // none
        vg.dim = (int) pk.num("vision.hidden_size", 1024), vg.heads = (int) pk.num("vision.num_attention_heads", 16);
        vg.inter = (int) pk.num("vision.intermediate_size", 2816), vg.patch = (int) pk.num("vision.patch_size", 14);
        vg.merge = (int) pk.num("vision.downsample_ratio", 3), vg.theta = (float) pk.num("vision.rope_theta", 1e4);
        vg.max_tokens = (int) pk.num("vision.max_image_tokens", 1024), vg.out = g.dim;
        const char* mark[3] = {"image_start", "image_newline", "image_end"};
        for (int i = 0; i < 3; ++i) {
            const Tensor* t = pk.get(mark[i]);
            img_mark[i].resize((size_t) g.dim);
            if (t == nullptr || !read_bf16(*t, (size_t) g.dim, img_mark[i].data())) {
                err = std::string(mark[i]) + ": not a BF16 (or F32) row of the model's width";
                return false;
            }
        }
        const double t0 = now_ms();
        Dev& d = devs[0];
        // the tower's tensors as BF16: the checkpoint's own, or a GGUF's mmproj (F32 norms / biases / marker rows, which
        // came from BF16, converted back exactly; mlp.w1 as the two files' gate and up, the halves of its rows)
        auto read = [this](const std::string& n, std::vector<int64_t>& shape, void* dst) {
            const Tensor* t = pk.get(n);
            if (t == nullptr && n.size() > 13 && n.compare(n.size() - 13, 13, "mlp.w1.weight") == 0) {
                const std::string pre = n.substr(0, n.size() - 13);
                const Tensor* tg = pk.get(pre + "mlp.gate.weight");
                const Tensor* tu = pk.get(pre + "mlp.up.weight");
                if (tg == nullptr || tu == nullptr || tg->shape.size() != 2 || tg->shape != tu->shape) return false;
                shape = {tg->shape[0] * 2, tg->shape[1]};
                const size_t half = (size_t) tg->shape[0] * (size_t) tg->shape[1];
                return dst == nullptr || (read_bf16(*tg, half, dst) && read_bf16(*tu, half, (uint16_t*) dst + half));
            }
            if (t == nullptr) return false;
            shape = t->shape;
            size_t n_el = 1;
            for (int64_t x : t->shape) n_el *= (size_t) x;
            return dst == nullptr || read_bf16(*t, n_el, dst);
        };
        if (!vt.load(vg, d.id, read, err)) return false;
        img_tok = tok;
        CK(cudaSetDevice(d.id));
        CK(cudaMemGetInfo(&d.free0, &d.total));   // (the split sees what the tower left)
        d.free0 = uma_free(d.id, d.free0);
        if (o.verbose)
            std::fprintf(stderr, "ds4: vision tower (%d blocks) on CUDA%d: %.0f MiB, up to %d patches, %.1f s\n",
                         vg.layers, d.id, (double) vt.device_bytes() / (1 << 20), vt.max_patches(),
                         (now_ms() - t0) / 1000.0);
        return true;
    }
    bool encode_image(const uint8_t* rgb, int w, int h, std::vector<uint16_t>& span, std::string& err) {
        if (img_tok < 0) {
            err = "this model has no vision tower (or it is off)";
            return false;
        }
        std::vector<uint16_t> rows;
        int nh = 0, nw = 0;
        if (!vt.encode(rgb, w, h, rows, nh, nw, err)) return false;
        const size_t D = (size_t) g.dim;
        span.resize(((size_t) nh * (nw + 1) + 2) * D);
        uint16_t* o = span.data();
        auto put = [&](const uint16_t* r) {
            std::memcpy(o, r, D * 2);
            o += D;
        };
        put(img_mark[0].data());
        for (int i = 0; i < nh; ++i) {
            for (int j = 0; j < nw; ++j) put(rows.data() + ((size_t) i * nw + j) * D);
            put(img_mark[1].data());
        }
        put(img_mark[2].data());
        return true;
    }
    // the row of a picture's span at position p (null: p is in none)
    const uint16_t* image_row(int64_t p) const {
        if (imgs.empty()) return nullptr;
        auto it = std::upper_bound(imgs.begin(), imgs.end(), p, [](int64_t v, const Image& im) { return v < im.start; });
        if (it == imgs.begin()) return nullptr;
        const Image& im = *(it - 1);
        const int64_t r = p - im.start;
        if (!im.rows || r >= (int64_t) (im.rows->size() / (size_t) g.dim)) return nullptr;
        return im.rows->data() + (size_t) r * g.dim;
    }
    // token tk's embedding row at position p: inside a picture's span, the picture's
    const uint16_t* emb_row(int64_t p, int tk) const {
        const uint16_t* r = tk == img_tok ? image_row(p) : nullptr;
        return r != nullptr ? r : embed_pinned + (size_t) tk * g.dim;
    }
    bool in_image(int64_t p) const { return img_tok >= 0 && toks[(size_t) p] == img_tok && image_row(p) != nullptr; }
    // the Engram's compressed id of token tk at p (-1: a picture's, which no n-gram spans)
    int32_t ctok(int64_t p, int tk) const {
        if (tk == img_tok && image_row(p) != nullptr) return -1;
        return tk < pk.eg_vocab ? pk.eg_map[(size_t) tk] : 0;
    }
    // a chunk's image rows (route()'s bits); false: none
    bool vl_bits(int64_t p0, int T, VlBits& b) const {
        if (imgs.empty()) return false;
        std::memset(&b, 0, sizeof b);
        bool any = false;
        for (int t = 0; t < T && t < kVlRows; ++t)
            if (in_image(p0 + t)) {
                b.w[t >> 5] |= 1u << (t & 31);
                any = true;
            }
        return any;
    }

    // ---- Engram: a window's rows of one table (Engram layer li), tokens p0 .. p0 + T - 1 into dst ([token] of [cols]
    // [hd] codes then the scales; nullptr: h_eg).  eg_issue serves the rows already fetched from the row cache (and a
    // row the window needs twice from one read) and hands the rest to the reader's threads; eg_finish waits for them,
    // copies the repeats and keeps the new rows in the cache.  The rows are the file's bytes either way.
    void eg_issue(int64_t p0, int T, int li, uint8_t* dst, EgBatch& b) {
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads, hd = pk.eg_hd;
        const size_t per = (size_t) cols * (hd + hd / 32);
        b.clear();
        b.reads.reserve((size_t) T * cols);
        const std::string P = "layers." + std::to_string(g.eg_layers[(size_t) li]) + ".engram.embed.";
        const Tensor* tw = pk.get(P + "weight");
        const Tensor* ts = pk.get(P + "scale");
        const int fmt = eg_fmt[(size_t) li];
        const bool mx8 = fmt == 0;   // (a GGUF: the scales between the codes, one per 32)
        const bool q6k = fmt == 1;   // (a GGUF: one Q6_K superblock a 256 values, its scales inside)
        const bool iq4 = fmt == 2;   // (a GGUF: 18-byte blocks a 32 values, d and 16 nibble bytes, no scales)
        std::unordered_map<uint64_t, int> first;   // a row this batch reads already -> its read
        int64_t hits = 0;
        {
            std::lock_guard<std::mutex> lk(eg_cache_mu);
            for (int t = 0; t < T; ++t) {
                const int64_t p = p0 + t;
                uint8_t* rows = dst != nullptr ? dst + (size_t) t * per : h_eg + ((size_t) li * eg_cap + t) * per;
                uint8_t* scales = rows + (size_t) cols * hd;
                if (ctoks[(size_t) p] < 0) {   // a picture's token: no rows (zeros: its gate adds nothing)
                    std::memset(rows, 0, per);
                    continue;
                }
                int64_t prod[8];
                bool blocked = false;
                for (int s = 0; s < pk.eg_ngram; ++s) {   // (the look-back stops at the start and at a picture)
                    blocked = blocked || p < s || ctoks[(size_t) (p - s)] < 0;
                    const int64_t tk = blocked ? pk.eg_pad : ctoks[(size_t) (p - s)];
                    prod[s] = tk * pk.eg_mult[(size_t) li * pk.eg_ngram + s];
                }
                int64_t rolling = prod[0];
                for (int n = 1; n < pk.eg_ngram; ++n) {
                    rolling ^= prod[n];
                    for (int h = 0; h < pk.eg_heads; ++h) {
                        const int c = (n - 1) * pk.eg_heads + h;
                        const int64_t id = rolling % pk.eg_primes[(size_t) li * cols + c] + pk.eg_off[(size_t) li * cols + c];
                        uint8_t* rd = rows + (size_t) c * hd;
                        uint8_t* sd = scales + (size_t) c * (hd / 32);
                        const uint64_t key = (uint64_t) id * 8 + (uint64_t) li;
                        if (const uint8_t* hit = eg_cache.find(key)) {
                            std::memcpy(rd, hit, (size_t) hd);
                            std::memcpy(sd, hit + hd, (size_t) hd / 32);
                            ++hits;
                            continue;
                        }
                        const auto fi = first.find(key);
                        if (fi != first.end()) {
                            b.dups.push_back({fi->second, {rd, sd}});
                            continue;
                        }
                        first.emplace(key, (int) b.reads.size());
                        EgRead r;
                        r.file = tw->file;
                        r.dst = rd;
                        r.scales = sd;
                        r.key = key;
                        if (mx8) {
                            r.mx8 = true;
                            r.n = (uint32_t) (hd / 32 * 33);
                            r.off = tw->off + (uint64_t) id * r.n;
                        } else if (q6k || iq4) {   // (into the row's code bytes; its scale bytes unused)
                            r.n = (uint32_t) (q6k ? hd / 256 * 210 : hd / 32 * 18);
                            r.off = tw->off + (uint64_t) id * r.n;
                        } else {
                            r.n = (uint32_t) hd;
                            r.off = tw->off + (uint64_t) id * hd;
                            r.file2 = ts->file;
                            r.n2 = (uint32_t) (hd / 32);
                            r.off2 = ts->off + (uint64_t) id * (hd / 32);
                        }
                        b.reads.push_back(r);
                    }
                }
            }
        }
        eg_rows_n += hits + (int64_t) b.reads.size() + (int64_t) b.dups.size();
        eg_hits_n += hits + (int64_t) b.dups.size();
        eg_reads_n += (int64_t) b.reads.size();
        eg_io.submit(b);
    }
    // b's rows all in place (any thread; a group's members may ask at once: one finishes, the others wait for it)
    void eg_finish(EgBatch& b) {
        if (b.post.load(std::memory_order_acquire) == 2) return;
        const int64_t t0 = now_ns();
        eg_io.wait(b);
        int want = 0;
        if (b.post.compare_exchange_strong(want, 1, std::memory_order_acq_rel)) {
            const int hd = pk.eg_hd;
            for (const auto& d : b.dups) {
                const EgRead& r = b.reads[(size_t) d.first];
                std::memcpy(d.second.first, r.dst, (size_t) hd);
                std::memcpy(d.second.second, r.scales, (size_t) hd / 32);
            }
            if (ds4_dbg("DS4_DBG_EG") && pk.eg_hd >= 32) {   // DBG: a row whose IQ4_NL block scale is not finite
                for (size_t i = 0; i < b.reads.size(); ++i) {
                    const EgRead& r = b.reads[i];
                    for (int k = 0; k < hd / 32; ++k) {
                        uint16_t hb;
                        std::memcpy(&hb, r.dst + (size_t) k * 18, 2);
                        if ((hb & 0x7c00) == 0x7c00)
                            std::fprintf(stderr, "DBG eg bad scale: read %zu/%zu file=%d off=%llu n=%u block=%d half=%04x ok=%d\n", i, b.reads.size(),
                                         r.file, (unsigned long long) r.off, r.n, k, hb, (int) b.ok[i]);
                    }
                }
            }
            {
                std::lock_guard<std::mutex> lk(eg_cache_mu);
                for (size_t i = 0; i < b.reads.size(); ++i) {
                    const EgRead& r = b.reads[i];
                    if (!b.ok[i]) {   // (a failed read: zeros, its gate adds nothing; not kept)
#if defined(STRATA_METAL)
                        std::fprintf(stderr, "ds4: WARNING: an Engram row could not be read (file %d, offset %llu, %u bytes): it counts as zeros\n", r.file,
                                     (unsigned long long) r.off, r.n);
#endif
                        std::memset(r.dst, 0, (size_t) hd);
                        std::memset(r.scales, 0, (size_t) hd / 32);
                        continue;
                    }
                    if (uint8_t* c = eg_cache.put(r.key)) {
                        std::memcpy(c, r.dst, (size_t) hd);
                        std::memcpy(c + hd, r.scales, (size_t) hd / 32);
                    }
                }
            }
            eg_io_ns += b.t1.load() - b.t0;
            b.live = false;
            b.post.store(2, std::memory_order_release);
        } else {
            while (b.post.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        }
        eg_wait_ns += now_ns() - t0;
    }
    void eg_stats() {
        st.eg_rows = eg_rows_n.load();
        st.eg_hits = eg_hits_n.load();
        st.eg_reads = eg_reads_n.load();
        st.eg_io_ms = (double) eg_io_ns.load() / 1e6;
        std::lock_guard<std::mutex> lk(eg_cache_mu);
        st.eg_cached = (int64_t) eg_cache.size();
        st.eg_cache_cap = (int64_t) eg_cache.capacity();
    }
    // the chunked prompt path: rows of Engram layer `only` into dst, now (a helper thread, a chunk ahead)
    void engram_rows(int64_t p0, int T, int only, uint8_t* dst) {
        EgBatch b;
        eg_issue(p0, T, only, dst, b);
        eg_finish(b);
    }

    // ---- the CPU lane: a window's routed experts that live on the host, as entries (an expert and up to 8 of its
    // tokens, in rows slot0 ..) computed together: each node its column part of every matrix, from memory on that node,
    // in four rounds (the gate / up split-K pieces, each preparing its two input blocks of every token; each band's
    // sums / codebook / Hadamard / swiglu and that block of the down input, which both nodes then read whole; the down
    // pieces; the down bands).  yd[slot] gets each row's expert output.
    static constexpr int kEntMax = kWin > kc::kExl3CpuMaxN ? kWin : kc::kExl3CpuMaxN;
    struct Ent {
        int e = 0, n = 0, slot0 = 0;
        int tok[kEntMax] = {};
    };
    // tokens an entry holds: the i-quant GGUF experts take a whole window's tokens of an expert in one call (the CPU
    // lane's exact multi-token rows, 7+ tokens an expert: decoded once for all), the others (MXFP4, EXL3) up to 8
    static int ent_cap(const Layer& l) {
        return l.lay.fmt == gg::kFmtMXFP4 && !gg::mx_experts(l.lay) ? kEntMax : kc::kExl3CpuMaxN;
    }
    void run_nodes(int n, const std::function<void(int)>* f) {
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->start(n, f[X]);
        pools[0]->help();
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->wait();
    }

    // a GGUF's MXFP4 experts on the CPU lane, read from the file's mapping or their copy in RAM (mx_part_ptr): each
    // node its rows of every matrix - the gate / up rows [X n_ff / NH, ..) of each entry (the swiglu into its h row),
    // then (h whole) the down rows [X dim / NH, ..) into yd.  A row's sum is the same whatever the split: the same bits
    // in a copy as in place.  The device's arithmetic (gg::mx_rows: int8 activations rounded as the device rounds
    // them, exact integer sums a block; every CPU tier the same bits): each token's input quantized once, whatever
    // number of entries take it, and each job its band of h straight into the down's int8 input.
    // (ord: the order the entries are computed in - their slots, so the sums, stay the same; null: as they are)
    void cpu_experts_mx(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx, const int* ord) {
        if (!gg::mx_experts(l.lay)) return cpu_experts_iq(l, x, ents, ldx, ord);
        constexpr int RB = 64;   // rows a job (whole blocks of h: RB % 32 == 0)
        const int E = g.dim, FF = g.ff, FFh = FF / NH, Eh = E / NH;
        const int GB = (FFh + RB - 1) / RB, DB = (Eh + RB - 1) / RB, ne = (int) ents.size();
        const size_t gr = gg::mx_row_bytes(E), dr = gg::mx_row_bytes(FF);
        const size_t xa = gg::mx_act_bytes(E), ha = gg::mx_act_bytes(FF);
        const bool band_q = FFh % 32 == 0;   // (each job's rows whole blocks of h)
        const float lim = g.lim;
        static_assert(kc::kExl3CpuMaxN <= gg::kMxCpuMaxN, "a CPU-lane entry's tokens in one call");
        {   // the experts read from the files (not in the RAM tier) that are not resident: all their pages asked for at
            // once, so the SSD reads them with large requests side by side while the first entries compute
            // (os::prefetch_cold: the resident ones are left alone - the hint walks every page of a range, which cost
            // more than computing a warm expert, 0.34 ms a 6.3 MB matrix on Windows)
            std::vector<std::pair<const void*, size_t>> want;
            for (const Ent& en : ents)
                if (!store && (l.ram.empty() || l.ram[(size_t) en.e] < 0))
                    for (int r = 0; r < 3; ++r) want.push_back({mx_mat(l, en.e, r), mx_part_bytes(l, r)});
            os::prefetch_cold(want);
        }
        // the int8 inputs in c_xh's bytes: token t's (t < kWin) at t xa, then the slots' h at (kWin + slot) ha; a token
        // index past the window (none on the decode or chunk paths) gets a row of its own after them
        uint8_t* xq = (uint8_t*) c_xh.data();
        uint8_t* hq = xq + (size_t) kWin * xa;
        uint8_t* xq_extra = hq + (size_t) kSlots * ha;
        auto x_act = [&](const Ent& en, int i) -> const void* {
            const int t = en.tok[i];
            return t < kWin ? xq + (size_t) t * xa : xq_extra + (size_t) (en.slot0 + i) * xa;
        };
        {
            bool done[kWin] = {};
            for (const Ent& en : ents)
                for (int i = 0; i < en.n; ++i) {
                    const int t = en.tok[i];
                    if (t < kWin && done[t]) continue;
                    if (t < kWin) done[t] = true;
                    gg::mx_quant(x + (size_t) t * ldx, E, (void*) x_act(en, i));
                }
        }
        std::function<void(int)> f[4];
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / GB] : j / GB)];
                const int r0 = X * FFh + (j % GB) * RB, r1 = std::min(X * FFh + FFh, r0 + RB);
                const void* ap[gg::kMxCpuMaxN];
                float yg[gg::kMxCpuMaxN][RB], yu[gg::kMxCpuMaxN][RB];
                float *gp[gg::kMxCpuMaxN], *up[gg::kMxCpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    ap[i] = x_act(en, i);
                    gp[i] = yg[i];
                    up[i] = yu[i];
                }
                // (part X's rows, from where they are: a copy's part on node X, or the file's)
                gg::mx_rows(mx_part_ptr(l, en.e, 0, X), gr, E, r0 - X * FFh, r1 - X * FFh, ap, en.n, gp);
                gg::mx_rows(mx_part_ptr(l, en.e, 1, X), gr, E, r0 - X * FFh, r1 - X * FFh, ap, en.n, up);
                for (int i = 0; i < en.n; ++i) {
                    float* h = band_q ? yg[i] : c_xhd.data() + (size_t) (en.slot0 + i) * FF + r0;
                    for (int r = r0; r < r1; ++r) {
                        float gv = yg[i][r - r0], uv = yu[i][r - r0];
                        if (lim > 0.0f) {
                            gv = std::min(gv, lim);
                            uv = std::min(std::max(uv, -lim), lim);
                        }
                        h[r - r0] = gv / (1.0f + std::exp(-gv)) * uv;
                    }
                    if (band_q) gg::mx_quant_blocks(h, r0 / 32, r1 / 32, hq + (size_t) (en.slot0 + i) * ha, FF);
                }
            };
        run_nodes(ne * GB, f);
        if (!band_q)
            for (const Ent& en : ents)
                for (int i = 0; i < en.n; ++i)
                    gg::mx_quant(c_xhd.data() + (size_t) (en.slot0 + i) * FF, FF, hq + (size_t) (en.slot0 + i) * ha);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / DB] : j / DB)];
                const int r0 = X * Eh + (j % DB) * RB, r1 = std::min(X * Eh + Eh, r0 + RB);
                const void* ap[gg::kMxCpuMaxN];
                float* yp[gg::kMxCpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    ap[i] = hq + (size_t) (en.slot0 + i) * ha;
                    yp[i] = c_yd.data() + (size_t) (en.slot0 + i) * E + r0;
                }
                gg::mx_rows(mx_part_ptr(l, en.e, 2, X), dr, FF, r0 - X * Eh, r1 - X * Eh, ap, en.n, yp);
            };
        run_nodes(ne * DB, f);
    }
    // ... the low-bit ones (ds4_iq's types): as above, through ggml-cpu's dots (gg::iq_cpu_*) - each token's input
    // quantized once to the weight type's activation (Q8_K) in its c_xh row's bytes, then (h whole) its h the same
#if defined(STRATA_METAL)
    // The CPU lane's cold reads were 78% of its time (a madvise(WILLNEED) of a range that is not in the page cache blocks while the SSD
    // reads it).  Stale-router lookahead (no training): the GPU applies layer L+a's gate (+ bias, the router's own rule) to layer L's FFN
    // input (a gemv and a route a layer, queued behind the shared expert) and the host reads the guessed picks with the real ones; the
    // guessed CPU-lane experts whose pages are not in the page cache are read in on helper threads while this layer computes.  Bytes per
    // layer capped (DS4_LA_MAX experts, default 4); DS4_LOOKAHEAD layers ahead (default 2; 0 off).
    static bool la_enabled() { return la_depth() > 0; }
    static int la_depth() {
        static const int d = std::getenv("DS4_LOOKAHEAD") ? std::atoi(std::getenv("DS4_LOOKAHEAD")) : 0;   // (off by default until it is measured faster)
        return std::min(d, 2);
    }
    struct LaPool {
        std::mutex m;
        std::condition_variable cv;
        std::deque<std::pair<const void*, size_t>> q;
        LaPool() {
            const int nth = std::getenv("DS4_LA_THREADS") ? std::atoi(std::getenv("DS4_LA_THREADS")) : 3;
            for (int i = 0; i < nth; ++i)
                std::thread([this] {
                    for (;;) {
                        std::pair<const void*, size_t> r;
                        {
                            std::unique_lock<std::mutex> lk(m);
                            cv.wait(lk, [&] { return !q.empty(); });
                            r = q.front();
                            q.pop_front();
                            while (q.size() > 64) q.pop_front();   // (stale reads are worth nothing)
                        }
                        const uintptr_t a = (uintptr_t) r.first & ~(uintptr_t) 16383;
                        const size_t len = r.second + ((uintptr_t) r.first - a);
                        std::vector<unsigned char> vec((len + 16383) / 16384);
                        bool cold = true;
                        if (mincore((void*) a, len, (char*) vec.data()) == 0) {
                            cold = false;
                            for (unsigned char c : vec) if (!(c & 1)) { cold = true; break; }
                        }
                        if (cold) os::prefetch({r});
                    }
                }).detach();
        }
        void push(const void* p, size_t n) {
            { std::lock_guard<std::mutex> lk(m); q.push_back({p, n}); }
            cv.notify_one();
        }
    };
    std::mutex la_mu;
    std::vector<std::vector<int>> la_guess;   // per layer: the CPU-lane experts its last lookahead guessed
    long la_hit = 0, la_all = 0, la_issued = 0;
    void lookahead_prefetch(Layer& l, cudaEvent_t ev2, int la_n, const std::vector<Ent>& ents) {
        if (l.nm != 1) return;
        static LaPool* pool = nullptr;
        static bool reg = false;
        if (pool == nullptr) {
            la_guess.assign(L.size(), {});
            pool = new LaPool();
        }
        if (ds4_dbg("DS4_CPU_LOG") && !reg) {
            reg = true;
            static Impl* self = this;
            std::atexit([] { std::fprintf(stderr, "lookahead: %ld of %ld CPU-lane experts were guessed; %ld matrices queued\n", self->la_hit, self->la_all, self->la_issued); });
        }
        for (const Ent& en : ents) {
            ++la_all;
            const auto& gs = la_guess[(size_t) l.il];
            if (std::find(gs.begin(), gs.end(), en.e) != gs.end()) ++la_hit;
        }
        if (la_n <= 0) return;
        cudaEventSynchronize(ev2);
        static const int cap = std::getenv("DS4_LA_MAX") ? std::atoi(std::getenv("DS4_LA_MAX")) : 4;
        for (int a = 1; a <= la_n; ++a) {
            Layer& nx = L[(size_t) (l.il + a)];
            std::vector<int> cpu;
            for (int r = 0; r < nx.k_exp; ++r) {
                const int e = h_idx_a[(a - 1) * kPicks + r];
                if (e >= 0 && e < nx.n_exp && nx.slot[(size_t) e] < 0 && (int) cpu.size() < cap) cpu.push_back(e);
            }
            la_guess[nx.il] = cpu;
            for (int e : cpu)
                for (int r = 0; r < 3; ++r) { pool->push(mx_mat(nx, e, r), mx_part_bytes(nx, r)); ++la_issued; }
        }
    }
#endif
    void cpu_experts_iq(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx, const int* ord) {
        constexpr int RB = 64;   // rows a job
        const int E = g.dim, FF = g.ff, FFh = FF / NH, Eh = E / NH;
        const int GB = (FFh + RB - 1) / RB, DB = (Eh + RB - 1) / RB, ne = (int) ents.size();
        gg::IqCpu fm;
        std::string ferr;
        if (!gg::iq_cpu_fmt(gg::expert_type(l.lay, 0), gg::expert_type(l.lay, 2), E, FF, fm, ferr) ||
            fm.act_bytes > (size_t) E * 4 || fm.h_bytes > (size_t) E * 4) {
            static std::atomic<bool> said{false};
            if (!said.exchange(true)) std::fprintf(stderr, "ds4: layer %d's CPU-lane experts: %s\n", l.il, ferr.c_str());
            for (const Ent& en : ents) std::memset(c_yd.data() + (size_t) en.slot0 * E, 0, (size_t) en.n * E * 4);
            return;
        }
        static double t_pre = 0, t_q1 = 0, t_r1 = 0, t_q2 = 0, t_r2 = 0; static long n_calls = 0; static bool reg = false;
        if (ds4_dbg("DS4_CPU_LOG") && !reg) { reg = true; std::atexit([] { std::fprintf(stderr, "cpu_experts_iq phases (ms over %ld calls): prefetch %.1f, quant1 %.1f, gate_up %.1f, quant2 %.1f, down %.1f\n", n_calls, t_pre, t_q1, t_r1, t_q2, t_r2); }); }
        const bool cpulog = ds4_dbg("DS4_CPU_LOG");   // (the Mac engine's counters; false elsewhere)
        if (cpulog) ++n_calls;
        double tp0 = now_ms();
        const float lim = g.lim;
        {   // (os::prefetch_cold, as above: the resident ones are left alone - on Windows the hint walks every page of a
            // range, ~0.13 ms an IQ2_XXS expert, on the lane's critical path before the first entry computes.
            // off by default (the mmap path, owner PC: not faster in every pair); DS4_IQ_PF_COLD=1 turns it on)
            std::vector<std::pair<const void*, size_t>> want;
            for (const Ent& en : ents)
                if (!store && (l.ram.empty() || l.ram[(size_t) en.e] < 0))
                    for (int r = 0; r < 3; ++r) want.push_back({mx_mat(l, en.e, r), mx_part_bytes(l, r)});
            static const bool pf_all = std::getenv("DS4_IQ_PF_COLD") == nullptr || std::atoi(std::getenv("DS4_IQ_PF_COLD")) == 0;
#if defined(__APPLE__)   // (macOS: the cold reads of a call's matrices in parallel, and a page-cache census under DS4_CPU_LOG)
            if (pf_all && ds4_dbg("DS4_CPU_LOG")) {   // how many of the experts' matrices are in the page cache (mincore, the first 64 pages)
                static long cold_n = 0, warm_n = 0; static double cold_ms = 0, warm_ms = 0; static bool reg2 = false;
                if (!reg2) { reg2 = true; std::atexit([] { std::fprintf(stderr, "prefetch: %ld matrices cold (%.1f ms), %ld warm (%.1f ms)\n", cold_n, cold_ms, warm_n, warm_ms); }); }
                for (const auto& w : want) {
                    const uintptr_t a = (uintptr_t) w.first & ~(uintptr_t) 16383;
                    unsigned char vec[64];
                    const size_t len = std::min<size_t>(w.second, 64 * 16384);
                    bool cold = false;
                    if (mincore((void*) a, len, (char*) vec) == 0) for (size_t i = 0; i < (len + 16383) / 16384; ++i) if (!(vec[i] & 1)) cold = true;
                    const double t0 = now_ms();
                    os::prefetch({w});
                    (cold ? cold_ms : warm_ms) += now_ms() - t0;
                    (cold ? cold_n : warm_n)++;
                }
            } else if (pf_all && want.size() > 1 && std::getenv("DS4_SERIAL_PREFETCH") == nullptr) {
                // (a madvise(WILLNEED) of a cold range blocks while the SSD reads it: queue depth 1 reads 3.9 GB/s; the matrices of
                // all the call's experts at once keep the SSD busy)
                std::function<void(int)> pf[4];
                for (int X = 0; X < NH; ++X) pf[X] = [&](int j) { os::prefetch({want[(size_t) j]}); };
                run_nodes((int) want.size(), pf);
            } else
            if (pf_all) os::prefetch(want);
            else os::prefetch_cold(want);
#else
            if (pf_all) os::prefetch(want);
            else os::prefetch_cold(want);
#endif
        }
        double tp1 = now_ms(); if (cpulog) t_pre += tp1 - tp0;
        // the activations in c_xh's bytes, E 4 apart: token t's (t < kWin) at row t - quantized once, whatever number of
        // entries take it - then a row a slot (kWin + slot): a token past the window's input (none on the decode or
        // chunk paths), then (x no longer needed) the slot's h
        uint8_t* aq = (uint8_t*) c_xh.data();
        auto x_act = [&](const Ent& en, int i) -> uint8_t* {
            const int t = en.tok[i];
            return aq + (size_t) (t < kWin ? t : kWin + en.slot0 + i) * E * 4;
        };
        {
            bool done[kWin] = {};
            for (const Ent& en : ents)
                for (int i = 0; i < en.n; ++i) {
                    const int t = en.tok[i];
                    if (t < kWin && done[t]) continue;
                    if (t < kWin) done[t] = true;
                    gg::iq_cpu_quant(fm.gu_act, x + (size_t) t * ldx, E, x_act(en, i));
                }
        }
        double tq1 = now_ms(); if (cpulog) t_q1 += tq1 - tp1;
        std::function<void(int)> f[4];
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / GB] : j / GB)];
                const int r0 = X * FFh + (j % GB) * RB, r1 = std::min(X * FFh + FFh, r0 + RB);
                const void* ap[kEntMax];
                float* hp[kEntMax];
                for (int i = 0; i < en.n; ++i) {
                    ap[i] = x_act(en, i);
                    hp[i] = c_xhd.data() + (size_t) (en.slot0 + i) * FF + (size_t) X * FFh;   // (part X's rows of h)
                }
                // (part X's rows, from where they are: a copy's part on node X, or the file's - whole rows, so any
                // block size of the type)
                gg::iq_cpu_gate_up(fm, mx_part_ptr(l, en.e, 0, X), mx_part_ptr(l, en.e, 1, X), ap, en.n, hp,
                                   r0 - X * FFh, r1 - X * FFh, lim);
            };
        run_nodes(ne * GB, f);
        double tr1 = now_ms(); if (cpulog) t_r1 += tr1 - tq1;
        for (const Ent& en : ents)
            for (int i = 0; i < en.n; ++i)
                gg::iq_cpu_quant(fm.d_act, c_xhd.data() + (size_t) (en.slot0 + i) * FF, FF,
                                 aq + (size_t) (kWin + en.slot0 + i) * E * 4);
        double tq2 = now_ms(); if (cpulog) t_q2 += tq2 - tr1;
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / DB] : j / DB)];
                const int r0 = X * Eh + (j % DB) * RB, r1 = std::min(X * Eh + Eh, r0 + RB);
                const void* hp[kEntMax];
                float* yp[kEntMax];
                for (int i = 0; i < en.n; ++i) {
                    hp[i] = aq + (size_t) (kWin + en.slot0 + i) * E * 4;
                    yp[i] = c_yd.data() + (size_t) (en.slot0 + i) * E + (size_t) X * Eh;   // (part X's rows of it)
                }
                gg::iq_cpu_down(fm, mx_part_ptr(l, en.e, 2, X), hp, en.n, yp, r0 - X * Eh, r1 - X * Eh);
            };
        run_nodes(ne * DB, f);
        if (cpulog) t_r2 += now_ms() - tq2;
        {   // DS4_PROF: where the CPU lane's time goes (act prep + prefetch, gate/up, h quant, down), summed
            static const bool on = std::getenv("DS4_PROF") != nullptr;
            if (on) {
                static double acc[6] = {};
                const double te = now_ms();
                acc[0] += tq1 - tp0; acc[1] += tr1 - tq1; acc[2] += tq2 - tr1; acc[3] += te - tq2; acc[4] += 1; acc[5] += ne;
                if (((long long) acc[4]) % 1000 == 0)
                    std::fprintf(stderr, "ds4 prof CPU lane (IQ rows) per call: n %.0f, entries/call %.2f, prep %.3f ms, gate+up %.3f ms, h-quant %.3f ms, down %.3f ms\n", acc[4], acc[5] / acc[4], acc[0] / acc[4], acc[1] / acc[4], acc[2] / acc[4], acc[3] / acc[4]);
            }
        }
    }

    // the window's CPU-lane experts made ready (and pinned) in the store for the length of the call
    struct StoreHold {
        Impl* m = nullptr;
        int layer = 0;
        std::vector<int> es, sl;
        std::vector<char> ok;
        // lazy: the reads are started here and waited for by wait_all() / wait_expert(); the destructor releases
        StoreHold(Impl* m_, const Layer& l, const std::vector<Ent>& ents) {
            if (!m_->store) return;
            m = m_;
            layer = l.il;
            for (const Ent& en : ents)
                if ((l.ram.empty() || l.ram[(size_t) en.e] < 0) && std::find(es.begin(), es.end(), en.e) == es.end()) es.push_back(en.e);
            m->store->begin(layer, es.data(), (int) es.size(), ok, sl);
        }
        int idx(int e) const {
            for (size_t i = 0; i < es.size(); ++i)
                if (es[i] == e) return (int) i;
            return -1;
        }
        // expert e needs no waiting: not a store expert (RAM copy) or its reads are done
        bool ready(int e) const {
            if (!m) return true;
            const int i = idx(e);
            return i < 0 || m->store->loaded(sl[(size_t) i]);
        }
        // (no view of the files is left to fall back on: an expert without a slot is read again, one at a time, until it has one)
        void rescue(int a, int b) {
            if (!m || m->views) return;
            for (int i = a; i < b; ++i) {
                for (int tries = 0; !ok[(size_t) i]; ++tries) {
                    std::vector<char> o2;
                    m->store->ensure(layer, &es[(size_t) i], 1, o2);
                    if (!o2.empty() && o2[0]) {
                        ok[(size_t) i] = 1;
                        break;
                    }
                    if (tries > 2000) {
                        std::fprintf(stderr, "ds4: layer %d expert %d: no store slot and no view of the files\n", layer, es[(size_t) i]);
                        std::exit(3);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        }
        void wait_expert(int e) {
            if (!m) return;
            const int i = idx(e);
            if (i >= 0) {
                m->store->wait_slots(sl, ok, i, i + 1);
                rescue(i, i + 1);
            }
        }
        void wait_all() {
            if (m) {
                m->store->wait_slots(sl, ok, 0, (int) es.size());
                rescue(0, (int) es.size());
            }
        }
        ~StoreHold() {
            if (m) m->store->release(layer, es.data(), (int) es.size(), ok);
        }
    };
    // DS4_MISS_SPLIT=0: the old order (wait for every read, then compute); default: the entries whose reads are done
    // computed first, the others each as its read lands (the same bits: every entry's rows are independent)
    void cpu_experts(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx = 0,
                     const int* ord = nullptr) {
        if (ldx == 0) ldx = (size_t) g.dim;
        if (l.lay.fmt == gg::kFmtMXFP4) {
            StoreHold hold(this, l, ents);
            static const bool split = std::getenv("DS4_MISS_SPLIT") == nullptr || std::atoi(std::getenv("DS4_MISS_SPLIT")) != 0;
            if (!split || ord != nullptr || !hold.m || ents.size() < 2) {
                hold.wait_all();
                return cpu_experts_mx(l, x, ents, ldx, ord);
            }
            std::vector<Ent> hot, cold;
            for (const Ent& en : ents) (hold.ready(en.e) ? hot : cold).push_back(en);
            if (cold.empty()) {
                hold.wait_all();
                return cpu_experts_mx(l, x, ents, ldx, ord);
            }
            for (const Ent& en : hot) hold.wait_expert(en.e);   // (done reading: only the bookkeeping)
            if (!hot.empty()) cpu_experts_mx(l, x, hot, ldx, nullptr);
            // the ones that came in meanwhile first, then the rest in the order they land
            std::vector<char> done(cold.size(), 0);
            for (size_t left = cold.size(); left > 0;) {
                size_t pick = cold.size();
                for (size_t i = 0; i < cold.size(); ++i)
                    if (!done[i] && hold.ready(cold[i].e)) {
                        pick = i;
                        break;
                    }
                if (pick == cold.size())
                    for (size_t i = 0; i < cold.size(); ++i)
                        if (!done[i]) {
                            pick = i;
                            break;
                        }
                hold.wait_expert(cold[pick].e);
                std::vector<Ent> one(1, cold[pick]);
                cpu_experts_mx(l, x, one, ldx, nullptr);
                done[pick] = 1;
                --left;
            }
            return;
        }
        const int E = g.dim, FF = g.ff, FFh = FF / NH, Eh = E / NH, GQ = E / 256, DQ = FF / 128;
        const int GB = FFh / 128, DB = Eh / 128, EB = E / 128, FB = FF / 128;
        const int ne = (int) ents.size();
        const float lim = g.lim;
        auto part = [&](int e, int X, int r) {
            const uint8_t* b = l.hh[(size_t) e][(size_t) X];
            kc::Exl3Part p;
            p.trellis = (const uint32_t*) (b + l.hl.trellis[r]);
            p.suh = (const uint16_t*) (b + l.hl.suh[r]);
            p.svh = (const uint16_t*) (b + l.hl.svh[r]);
            p.k = l.hl.k[r];
            p.n = l.hl.n[r];
            p.K = l.lay.K[r];
            return p;
        };
        auto xh = [&](int X, int sl, int r) { return c_xh.data() + ((size_t) (X * kSlots + sl) * 2 + r) * E; };
        auto xsb = [&](int X, int sl, int r) { return c_xsb.data() + ((size_t) (X * kSlots + sl) * 2 + r) * EB; };
        auto acc = [&](int X, int sl, int r, int q) {
            return c_acc.data() + (((size_t) (X * kSlots + sl) * 2 + r) * GQ + q) * FFh;
        };
        auto xhd = [&](int sl) { return c_xhd.data() + (size_t) sl * FF; };
        auto xsd = [&](int sl) { return c_xsd.data() + (size_t) sl * FB; };
        auto accd = [&](int X, int sl, int q) { return c_accd.data() + ((size_t) (X * kSlots + sl) * DQ + q) * Eh; };
        std::function<void(int)> f[4];
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / (2 * GQ))];
                const int r = (j / GQ) % 2, q = j % GQ;
                const kc::Exl3Part p = part(en.e, X, r);
                const float* xp[kc::kExl3CpuMaxN];
                float* ap[kc::kExl3CpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    const int sl = en.slot0 + i;
                    const float* xt = x + (size_t) en.tok[i] * ldx;
                    for (int b = 2 * q; b < 2 * q + 2; ++b)
                        xsb(X, sl, r)[b] = kc::exl3_cpu_prepare_block(p.suh + b * 128, xt + b * 128, xh(X, sl, r) + b * 128);
                    xp[i] = xh(X, sl, r);
                    ap[i] = acc(X, sl, r, q);
                }
                if (en.n == 1) kc::exl3_cpu_rows_acc(p, xp[0], q * 16, (q + 1) * 16, ap[0]);
                else kc::exl3_cpu_rows_acc_n(p, xp, en.n, q * 16, (q + 1) * 16, ap);
            };
        run_nodes(ne * 2 * GQ, f);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / GB)];
                const int b = j % GB;
                const kc::Exl3Part pg = part(en.e, X, 0), pu = part(en.e, X, 1), pd = part(en.e, X, 2);
                for (int i = 0; i < en.n; ++i) {
                    const int sl = en.slot0 + i;
                    float y[2][128], a[128], hb[128];
                    for (int r = 0; r < 2; ++r) {
                        for (int c = 0; c < 128; ++c) a[c] = 0.0f;
                        for (int q = 0; q < GQ; ++q) {
                            const float* aq = acc(X, sl, r, q) + b * 128;
                            for (int c = 0; c < 128; ++c) a[c] += aq[c];
                        }
                        double xs = 0.0;
                        for (int k = 0; k < EB; ++k) xs += xsb(X, sl, r)[k];
                        kc::exl3_cpu_band_out(r == 0 ? pg : pu, a, (float) xs, b * 128, y[r]);
                    }
                    for (int c = 0; c < 128; ++c) {
                        float gv = y[0][c], uv = y[1][c];
                        if (lim > 0.0f) {
                            gv = std::min(gv, lim);
                            uv = std::min(std::max(uv, -lim), lim);
                        }
                        hb[c] = gv / (1.0f + std::exp(-gv)) * uv;
                    }
                    const int gb = X * GB + b;
                    xsd(sl)[gb] = kc::exl3_cpu_prepare_block(pd.suh + gb * 128, hb, xhd(sl) + gb * 128);
                }
            };
        run_nodes(ne * GB, f);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / DQ)];
                const int q = j % DQ;
                const kc::Exl3Part p = part(en.e, X, 2);
                const float* xp[kc::kExl3CpuMaxN];
                float* ap[kc::kExl3CpuMaxN];
                for (int i = 0; i < en.n; ++i) {
                    xp[i] = xhd(en.slot0 + i);
                    ap[i] = accd(X, en.slot0 + i, q);
                }
                if (en.n == 1) kc::exl3_cpu_rows_acc(p, xp[0], q * 8, (q + 1) * 8, ap[0]);
                else kc::exl3_cpu_rows_acc_n(p, xp, en.n, q * 8, (q + 1) * 8, ap);
            };
        run_nodes(ne * DQ, f);
        for (int X = 0; X < NH; ++X)
            f[X] = [&, X](int j) {
                const Ent& en = ents[(size_t) (j / DB)];
                const int b = j % DB;
                const kc::Exl3Part pd = part(en.e, X, 2);
                for (int i = 0; i < en.n; ++i) {
                    const int sl = en.slot0 + i;
                    float a[128];
                    for (int c = 0; c < 128; ++c) a[c] = 0.0f;
                    for (int q = 0; q < DQ; ++q) {
                        const float* aq = accd(X, sl, q) + b * 128;
                        for (int c = 0; c < 128; ++c) a[c] += aq[c];
                    }
                    double xs = 0.0;
                    for (int k = 0; k < FB; ++k) xs += xsd(sl)[k];
                    kc::exl3_cpu_band_out(pd, a, (float) xs, b * 128, c_yd.data() + (size_t) sl * E + (size_t) X * Eh + b * 128);
                }
            };
        run_nodes(ne * DB, f);
    }

    // ---- the expert cache.  A swap moves a CPU-lane expert (`in`) into the layer's spare VRAM slot and the VRAM expert it
    // replaces (`out`) into the spare host parts - both copies of weights that do not change, made while the model runs
    // (the copy stream, the host's repacking) - and cache_tick switches the tables between windows, so the device's
    // table and the host's view of a layer always agree within a window.  In place (file_experts) `out` needs no copy:
    // the CPU lane reads it from the file again.
    void ram_loop() {   // the RAM tier's moves in: an expert read from the file into a free slot
        for (;;) {
            RamSwap rs;
            {
                std::unique_lock<std::mutex> lk(sw_mu);
                ram_cv.wait(lk, [&] { return quit.load() || !ram_queue.empty(); });
                if (quit.load()) return;
                rs = ram_queue.front();
                ram_queue.pop_front();
            }
            const Layer& l = L[(size_t) rs.layer];
            if (read_mx_file(l, rs.in, ram_slots[(size_t) rs.slot])) {
                for (int r = 0; r < 3; ++r)   // (one copy of it in RAM)
                    os::advise(pk.fds[(size_t) l.mx_file[r]], l.mx_off[r] + (uint64_t) rs.in * l.mx_stride[r],
                               mx_part_bytes(l, r), false);
            } else {
                rs.in = -1;   // (failed: the slot goes back)
            }
            std::lock_guard<std::mutex> lk(sw_mu);
            ram_ready.push_back(rs);
        }
    }

    void cache_loop(SwapBufs& B) {
        for (;;) {
            Swap sw;
            {
                std::unique_lock<std::mutex> lk(sw_mu);
                sw_cv.wait(lk, [&] { return quit.load() || !sw_queue.empty(); });
                if (quit.load()) return;
                sw = sw_queue.front();
                sw_queue.pop_front();
            }
            Layer& l = L[(size_t) sw.layer];
            if (l.lay.fmt == gg::kFmtMXFP4) {   // in: from the file; out: nothing to keep (the file has it)
                swap_in_mx(l, sw, B);
                std::lock_guard<std::mutex> lk(sw_mu);
                sw_ready.push_back(sw);
                continue;
            }
            parts_to_blob(l, l.hh[(size_t) sw.in], B.a);
            if (l.nm == 1) {
                const Dev& d = devs[(size_t) l.dev];
                cudaSetDevice(d.id);
                cudaStream_t cs = B.st[(size_t) l.dev];
                cudaMemcpyAsync(l.vram_base + (size_t) l.spare_vram * l.blob, B.a, l.blob, cudaMemcpyHostToDevice, cs);
                if (!file_experts)
                    cudaMemcpyAsync(B.b, l.vram_base + (size_t) l.slot[(size_t) sw.out] * l.blob, l.blob,
                                    cudaMemcpyDeviceToHost, cs);
                cudaStreamSynchronize(cs);
            } else {   // a group: each member's part in, each member's part out
                for (int m = 0; m < l.nm; ++m) {
                    const Layer& x = member(l, m);
                    cudaSetDevice(devs[(size_t) x.dev].id);
                    cudaStream_t cs = B.st[(size_t) x.dev];
                    blob_to_half(l, B.a, m, B.h);
                    cudaMemcpyAsync(x.vram_base + (size_t) l.spare_vram * x.hblob, B.h, x.hblob, cudaMemcpyHostToDevice, cs);
                    cudaStreamSynchronize(cs);
                    if (file_experts) continue;
                    cudaMemcpyAsync(B.h, x.vram_base + (size_t) l.slot[(size_t) sw.out] * x.hblob, x.hblob,
                                    cudaMemcpyDeviceToHost, cs);
                    cudaStreamSynchronize(cs);
                    half_to_blob(l, B.h, m, B.b);
                }
            }
            if (!file_experts)
                for (int X = 0; X < NH; ++X) blob_to_part(l, B.b, X, l.spare_host[(size_t) X]);
            if (check_swaps) {   // DS4_CHECK_SWAPS: both moves against the model file, byte for byte
                std::vector<uint8_t> reg, blob(l.blob), part(l.hl.bytes);
                bool ok = true;
                for (int pass = 0; pass < (file_experts ? 1 : 2); ++pass) {   // (in place: no host copy of `out`)
                    const int e = pass == 0 ? sw.in : sw.out;
                    reg.resize(l.region_bytes[(size_t) e]);
                    pk.read_at(l.region_file[(size_t) e], l.region_off[(size_t) e], reg.size(), reg.data());
                    if (pass == 0) {
                        repack(l, reg.data(), blob.data());
                        std::vector<uint8_t> dev(l.blob);
                        if (l.nm == 1)
                            cudaMemcpy(dev.data(), l.vram_base + (size_t) l.spare_vram * l.blob, l.blob, cudaMemcpyDeviceToHost);
                        else
                            for (int m = 0; m < l.nm; ++m) {   // (the members' parts, put together again)
                                std::vector<uint8_t> h(member(l, m).hblob);
                                cudaSetDevice(devs[(size_t) member(l, m).dev].id);
                                cudaMemcpy(h.data(), member(l, m).vram_base + (size_t) l.spare_vram * member(l, m).hblob, member(l, m).hblob,
                                           cudaMemcpyDeviceToHost);
                                half_to_blob(l, h.data(), m, dev.data());
                            }
                        for (int r = 0; r < 3; ++r)
                            ok = ok && std::memcmp(dev.data() + l.lay.trellis[r], blob.data() + l.lay.trellis[r], l.piece_bytes[r][0]) == 0 &&
                                 std::memcmp(dev.data() + l.lay.suh[r], blob.data() + l.lay.suh[r], l.piece_bytes[r][1]) == 0 &&
                                 std::memcmp(dev.data() + l.lay.svh[r], blob.data() + l.lay.svh[r], l.piece_bytes[r][2]) == 0;
                    } else {
                        for (int X = 0; X < NH; ++X) {
                            repack_part(l, reg.data(), X, part.data());
                            for (int r = 0; r < 3; ++r) {
                                const size_t tb = (size_t) (l.hl.k[r] / 16) * (l.hl.n[r] / 16) * 32 * l.lay.K[r];
                                ok = ok && std::memcmp(part.data() + l.hl.trellis[r], l.spare_host[(size_t) X] + l.hl.trellis[r], tb) == 0 &&
                                     std::memcmp(part.data() + l.hl.suh[r], l.spare_host[(size_t) X] + l.hl.suh[r], (size_t) l.hl.k[r] * 2) == 0 &&
                                     std::memcmp(part.data() + l.hl.svh[r], l.spare_host[(size_t) X] + l.hl.svh[r], (size_t) l.hl.n[r] * 2) == 0;
                            }
                        }
                    }
                }
                std::fprintf(stderr, "ds4 cache: layer %d expert %d in, %d out: %s\n", l.il, sw.in, sw.out,
                             ok ? "both match the model file" : "MISMATCH");
            }
            std::lock_guard<std::mutex> lk(sw_mu);
            sw_ready.push_back(sw);
        }
    }

    // a GGUF's swap: expert sw.in from the file (copy_mode: its copy) into the spare VRAM slot (each member its part);
    // the expert it replaces goes back to the CPU lane, which reads it from the file's mapping - or (copy_mode, sw.ram)
    // from the layer's spare RAM slot, which it is copied into from VRAM here
    void swap_in_mx(Layer& l, Swap& sw, SwapBufs& B) {
        uint8_t *sw_a = B.a, *sw_b = B.b, *sw_h = B.h;   // (this thread's)
        const std::vector<cudaStream_t>& copy_st = B.st;
        bool ok = copy_mode ? read_mx(l, sw.in, sw_a) : read_mx_file(l, sw.in, sw_a);   // (a copy holds its slot till
                                                                                          // this swap's switch)
        for (int m = 0; m < l.nm && ok; ++m) {
            const Layer& x = member(l, m);
            cudaSetDevice(devs[(size_t) x.dev].id);
            cudaStream_t cs = copy_st[(size_t) x.dev];
            if (l.nm > 1) blob_to_half(l, sw_a, m, sw_h);
            ok = cudaMemcpyAsync(x.vram_base + (size_t) l.spare_vram * unit(l), l.nm > 1 ? sw_h : sw_a, unit(l),
                                 cudaMemcpyHostToDevice, cs) == cudaSuccess &&
                 cudaStreamSynchronize(cs) == cudaSuccess;
        }
        if (sw.ram >= 0) {   // `out` from its VRAM slot (each member its part) into the spare RAM slot
            bool got = ok;
            for (int m = 0; m < l.nm && got; ++m) {
                const Layer& x = member(l, m);
                cudaSetDevice(devs[(size_t) x.dev].id);
                cudaStream_t cs = copy_st[(size_t) x.dev];
                got = cudaMemcpyAsync(l.nm > 1 ? sw_h : sw_b, x.vram_base + (size_t) l.slot[(size_t) sw.out] * unit(l),
                                      unit(l), cudaMemcpyDeviceToHost, cs) == cudaSuccess &&
                      cudaStreamSynchronize(cs) == cudaSuccess;
                if (got && l.nm > 1) half_to_blob(l, sw_h, m, sw_b);
            }
            if (got) blob_to_ram(l, sw_b, sw.ram);
            else sw.ram = -1;   // (back to the file's pages then)
        } else if (ok && store && l.nm == 1 && l.lay.fmt == gg::kFmtMXFP4 && strata::ds4::store_feature_default(env_tri("DS4_STORE_PUT"), defaults_026())) {
            // `out` goes back to the CPU lane: its blob is in VRAM, so it is copied into a store slot now - the lane would otherwise read it from
            // the SSD and wait for it (1-2 misses a token on the 251 GB R9700 box).  DS4_STORE_PUT=0: off.
            uint8_t* dst = store->put_begin(l.il, sw.out);
            if (dst != nullptr) {
                const Layer& x = member(l, 0);
                cudaSetDevice(devs[(size_t) x.dev].id);
                cudaStream_t cs = copy_st[(size_t) x.dev];
                const bool got = cudaMemcpyAsync(dst, x.vram_base + (size_t) l.slot[(size_t) sw.out] * unit(l), l.blob, cudaMemcpyDeviceToHost, cs) ==
                                     cudaSuccess &&
                                 cudaStreamSynchronize(cs) == cudaSuccess;
                store->put_done(l.il, sw.out, got);
            }
        }
        if (check_swaps) {   // DS4_CHECK_SWAPS: the slot (and `out`'s RAM copy) against the model file, byte for byte
            std::vector<uint8_t> dev(l.blob), h(unit(l)), file(l.blob);
            for (int m = 0; m < l.nm && ok; ++m) {
                const Layer& x = member(l, m);
                cudaSetDevice(devs[(size_t) x.dev].id);
                cudaMemcpy(l.nm > 1 ? h.data() : dev.data(), x.vram_base + (size_t) l.spare_vram * unit(l), unit(l),
                           cudaMemcpyDeviceToHost);
                if (l.nm > 1) half_to_blob(l, h.data(), m, dev.data());
            }
            ok = ok && read_mx_file(l, sw.in, file.data());
            for (int r = 0; r < 3 && ok; ++r)
                ok = std::memcmp(dev.data() + l.lay.trellis[r], file.data() + l.lay.trellis[r], mx_part_bytes(l, r)) == 0;
            bool ok_out = true;
            if (sw.ram >= 0) {
                ok_out = read_mx_file(l, sw.out, file.data());
                ram_to_blob(l, sw.ram, dev.data());
                for (int r = 0; r < 3 && ok_out; ++r)
                    ok_out = std::memcmp(dev.data() + l.lay.trellis[r], file.data() + l.lay.trellis[r], mx_part_bytes(l, r)) == 0;
            }
            std::fprintf(stderr, "ds4 cache: layer %d expert %d in, %d out%s: %s\n", l.il, sw.in, sw.out,
                         sw.ram >= 0 ? " (into RAM)" : "", ok && ok_out ? "matches the model file" : "MISMATCH");
        } else if (!ok) {
            std::fprintf(stderr, "ds4 cache: layer %d expert %d: the move into VRAM failed\n", l.il, sw.in);
        }
    }

    // after a chunked prompt: the expert cache catches up with what the prompt picked at once - a window prompt gets
    // its swaps one tick at a time over its hundreds of windows, a chunked one has a single tick - so the decode that
    // follows finds the hot experts in VRAM.  Rounds of the ordinary swap rule (the hottest CPU-lane pick of each layer
    // in, its coldest VRAM expert out) until nothing is worth moving or DS4_SETTLE_MS is spent.
    // The budget was 6,000 ms, and it sits inside the prompt's own time ("prompt: N tokens in X ms"): measured on Strix Halo
    // 2026-10-28, the same p4k fixture took 26.6 s with it and 20.4 s without, +30.5% on the prompt (153.9 -> 200.8 tok/s).
    // What the swaps buy is decode: about 10% on a 256-token answer, i.e. 6.2 s of prompt against 1.6 s of decode on that
    // answer and nothing at all on a 50-300-token tool-call answer.  800 ms keeps the hottest of them and gives the prompt
    // back about 5 s (the plan's own table: prompt + 256 tokens, 27.2 s at 6,000 ms, 23.9 s at 800, 23.4 s at 0).
    void cache_settle() {
        if (!o.expert_cache || cache_ths.empty()) return;
        static const double budget = env_f("DS4_SETTLE_MS", 800.0f);
        if (budget <= 0.0) return;
        const double t0 = now_ms();
        const int64_t s0 = st.swaps;
        while (now_ms() - t0 < budget) {
            for (Layer& l : L) {
                if (l.mtp || l.spare_vram < 0 || l.swap_pending) continue;
                l.cpu_hits.clear();
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0 && l.heat[(size_t) e] >= 2.0f) l.cpu_hits.push_back(e);
            }
            cache_tick_swaps(0);
            if (sw_inflight == 0) break;   // (nothing queued: every layer's best CPU-lane expert is not hotter than its coldest VRAM one)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // The RAM store's preload after a chunked prompt is off by default: measured on Strix Halo (code fixture, prompt +
        // 256 tokens, 3 runs a side) it costs the prompt 4.7 s and the decode that follows gains nothing (13.9/14.4 s with it
        // against 14.2 s without).  The swaps above are what the decode actually uses; DS4_SETTLE_STORE=1 restores it.
        if (store && env_f("DS4_SETTLE_STORE", strata::ds4::settle_store_default(defaults_026())) != 0.0f) {   // the RAM store too: a window prompt fills it through the CPU
            // lane's picks, a chunked one streams past it - the hottest CPU-lane experts of the prompt into its free slots
            // (copy_mode: an expert already in the CPU lane's RAM copy gains nothing from a second copy of the same bytes -
            // measured on the R9700 12 GB cell: 4152 such experts, 40 GB read at 6.4 GB/s, 6.2 s of an 8K prompt, 0.0 store
            // hits in the decode after it; the VRAM swaps above already read them from the copy)
            std::vector<std::pair<float, std::pair<int, int>>> hot;
            for (const Layer& l : L) {
                if (l.mtp) continue;
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0 && !(strata::ds4::settle_skips_copy_experts(defaults_026()) && copy_mode && !l.ram.empty() && l.ram[(size_t) e] >= 0) &&
                        l.heat[(size_t) e] >= 4.0f) hot.push_back({l.heat[(size_t) e], {l.il, e}});
            }
            std::sort(hot.begin(), hot.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            std::vector<std::pair<int, int>> list;
            for (const auto& h : hot) {
                if (list.size() >= store->slots_max()) break;
                list.push_back(h.second);
            }
            const double ts = now_ms();
            const size_t n = store->preload(list, 0);
            if (o.verbose) std::fprintf(stderr, "ds4: RAM store: %zu hot experts read after the prompt (%.0f ms)\n", n, now_ms() - ts);
        }
        if (o.verbose && st.swaps > s0)
            std::fprintf(stderr, "ds4: expert cache: %lld swaps after the prompt (%.0f ms)\n", (long long) (st.swaps - s0),
                         now_ms() - t0);
    }
    // WP-8's option (1): the same swap rounds, run on a thread while the server has no request in flight, stopped by `cancel`.
    // The prompt path keeps its 800 ms (a run with no server has no idle to use), so this is pure addition: an agent's turn
    // ends, the swaps go in while it runs its tool, and the next turn's answer finds them.  Interrupting costs one round (a
    // swap of ~9 MB, a millisecond or two) because cancel is tested between rounds, and cache_tick_swaps is the same call the
    // prompt path makes, so the state it leaves is the state a decode window already handles.
    void cache_settle_idle(const std::atomic<bool>& cancel) {
        if (!o.expert_cache || cache_ths.empty()) return;
        static const double budget = env_f("DS4_SETTLE_IDLE_MS", 6000.0f);
        const double t0 = now_ms();
        const int64_t s0 = st.swaps;
        while (!cancel.load(std::memory_order_acquire) && now_ms() - t0 < budget) {
            for (Layer& l : L) {
                if (l.mtp || l.spare_vram < 0 || l.swap_pending) continue;
                l.cpu_hits.clear();
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] < 0 && l.heat[(size_t) e] >= 2.0f) l.cpu_hits.push_back(e);
            }
            cache_tick_swaps(0);
            if (sw_inflight == 0) break;   // (nothing queued: no layer's best CPU-lane expert is hotter than its coldest VRAM one)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (o.verbose)
            std::fprintf(stderr, "ds4: expert cache: %lld swaps in the idle (%.0f ms, cancelled %d)\n",
                         (long long) (st.swaps - s0), now_ms() - t0, cancel.load() ? 1 : 0);
    }
    void cache_tick(int T) {
        cache_tick_swaps(T);
        store_sync();
        store_watch();
    }
    // the store gives RAM back when the PC runs short (the owner opened something): below 4 GB free, a 0.7 GB chunk at a time
    // until 5 GB are free again (checked every 16 windows); it grows again by itself once 8 GB beside its chunk are free
    void store_watch() {
        if (!store) return;
        static int tick = 0;
        if (++tick % 16 != 0) return;
        const uint64_t avail = ram_avail();
        if (avail == ~uint64_t{0} || avail >= (4ull << 30)) return;
        const size_t chunks = 1 + (size_t) (((5ull << 30) - avail) / (700ull << 20));
        const size_t n = store->shrink_chunks(chunks);
        if (n > 0) std::fprintf(stderr, "ds4: RAM store: %.1f GB free: gave back %zu chunks (%zu slots left)\n", avail / 1e9, n, store->slots_allocated());
    }
    void cache_tick_swaps(int T) {
        if (!o.expert_cache || cache_ths.empty()) return;
        std::vector<Swap> ready;
        std::vector<RamSwap> rready;
        {
            std::lock_guard<std::mutex> lk(sw_mu);
            ready.swap(sw_ready);
            rready.swap(ram_ready);
        }
        for (const RamSwap& rs : rready) {   // (nothing reads the tier between windows)
            --ram_inflight;
            Layer& l = L[(size_t) rs.layer];
            if (rs.in < 0 || l.ram[(size_t) rs.in] >= 0 || l.slot[(size_t) rs.in] >= 0) {
                ram_free.push_back(rs.slot);
                continue;
            }
            l.ram[(size_t) rs.in] = rs.slot;
            if (rs.out >= 0 && l.ram[(size_t) rs.out] >= 0) {
                ram_free.push_back(l.ram[(size_t) rs.out]);
                l.ram[(size_t) rs.out] = -1;
            }
            ++st.ram_moves;
        }
        for (const Swap& sw : ready) {
            Layer& l = L[(size_t) sw.layer];
            const int old_slot = l.slot[(size_t) sw.out];
            const std::array<uint8_t*, 4> old_parts = l.hh[(size_t) sw.in];
            if (copy_mode) {   // (`out` takes the RAM place `in` leaves: through the spare, which `in`'s slot becomes)
                const int s_in = l.ram[(size_t) sw.in];
                l.ram[(size_t) sw.in] = -1;
                if (sw.ram >= 0) {
                    l.ram[(size_t) sw.out] = sw.ram;
                    copy_spare[(size_t) l.il] = s_in;
                } else if (s_in >= 0) {   // (the move into RAM failed: `out` back to the file, one copy fewer)
                    --ram_held;
                    if (copy_spare[(size_t) l.il] < 0) copy_spare[(size_t) l.il] = s_in;
                }
            } else if (!l.ram.empty() && l.ram[(size_t) sw.in] >= 0) {   // (in VRAM now: its RAM slot for the next hottest)
                ram_free.push_back(l.ram[(size_t) sw.in]);
                l.ram[(size_t) sw.in] = -1;
            }
            l.slot[(size_t) sw.in] = l.spare_vram;
            if (!l.zc.empty() && l.zc[(size_t) sw.in]) {   // (VRAM's now: not the store's to compute any more)
                l.zc[(size_t) sw.in] = 0;
                if (l.zc_tab != nullptr) {   // (the device's table too: its entry pointed into the slot the store now reuses - when `in` left VRAM later,
                                             // zc_stage copied whatever lay there, another expert's blob, into the plan)
                    cudaSetDevice(devs[0].id);
                    cudaMemsetAsync(l.zc_tab + sw.in, 0, 8, devs[0].st);
                }
                store->unmark(l.il, sw.in);
            }
            if (l.nm == 1) set_vm(l, sw.in, l.spare_vram);
            l.hh[(size_t) sw.in] = {nullptr, nullptr, nullptr, nullptr};
            l.host[(size_t) sw.in] = nullptr;
            l.slot[(size_t) sw.out] = -1;
            if (l.lay.fmt == gg::kFmtMXFP4) {   // (the file has it)
                l.host[(size_t) sw.out] = (uint8_t*) mx_mat(l, sw.out, 0);
            } else if (file_experts) {   // (back to its pages in the file)
                l.hh[(size_t) sw.out] = {file_part(l, sw.out), nullptr, nullptr, nullptr};
                l.host[(size_t) sw.out] = l.hh[(size_t) sw.out][0];
            } else {
                l.hh[(size_t) sw.out] = l.spare_host;
                l.host[(size_t) sw.out] = l.spare_host[0];
                l.spare_host = old_parts;
            }
            l.spare_vram = old_slot;
            for (int m = 0; m < l.nm; ++m) {
                Layer& x = member(l, m);
                const Dev& dm = devs[(size_t) x.dev];
                l.tab_pin[2 * m] = (unsigned long long) (x.vram_base + (size_t) l.slot[(size_t) sw.in] * unit(x));
                l.tab_pin[2 * m + 1] = 0ull;
                cudaSetDevice(dm.id);   // (stream order: before the next window)
                cudaMemcpyAsync(x.vram_tab + sw.in, &l.tab_pin[2 * m], 8, cudaMemcpyHostToDevice, dm.st);
                cudaMemcpyAsync(x.vram_tab + sw.out, &l.tab_pin[2 * m + 1], 8, cudaMemcpyHostToDevice, dm.st);
            }
            l.swap_pending = false;
            --sw_inflight;
            ++st.swaps;
        }
        decay_tokens += T;
        if (decay_tokens >= 256) {   // older use counts less: x0.9 every 256 tokens
            decay_tokens = 0;
            for (Layer& l : L)
                for (float& h : l.heat) h *= c_decay;
        }
        bool queued = false, rqueued = false;
        for (Layer& l : L) {
            if (l.cpu_hits.empty()) continue;
            std::vector<int> hits;
            hits.swap(l.cpu_hits);
            int best = -1, vin = -1;
            for (int e : hits)
                if (l.slot[(size_t) e] < 0 && (best < 0 || l.heat[(size_t) e] > l.heat[(size_t) best])) best = e;
            if (best >= 0 && l.spare_vram >= 0 && !l.swap_pending && sw_inflight < c_inflight) {   // VRAM: the hottest in
                int cold = -1;
                for (int e = 0; e < l.n_exp; ++e)
                    if (l.slot[(size_t) e] >= 0 && (cold < 0 || l.heat[(size_t) e] < l.heat[(size_t) cold])) cold = e;
                if (cold >= 0 && l.heat[(size_t) best] > l.heat[(size_t) cold] * c_margin + c_bonus) {
                    l.swap_pending = true;
                    ++sw_inflight;
                    vin = best;
                    Swap sw{l.il, best, cold};
                    if (copy_mode && l.ram[(size_t) best] >= 0) sw.ram = copy_spare[(size_t) l.il];   // (-1: none)
                    std::lock_guard<std::mutex> lk(sw_mu);
                    sw_queue.push_back(sw);
                    queued = true;
                }
            }
            if (l.ram.empty() || ram_free.empty() || ram_inflight >= kMaxSwaps) continue;
            int rb = -1;   // RAM: the hottest read from the file, in place of the layer's coldest held (or a free slot)
            for (int e : hits)
                if (e != vin && l.slot[(size_t) e] < 0 && l.ram[(size_t) e] < 0 &&
                    (rb < 0 || l.heat[(size_t) e] > l.heat[(size_t) rb]))
                    rb = e;
            if (rb < 0) continue;
            int rc = -1;
            for (int e = 0; e < l.n_exp; ++e)
                if (l.ram[(size_t) e] >= 0 && (rc < 0 || l.heat[(size_t) e] < l.heat[(size_t) rc])) rc = e;
            const bool room = ram_free.size() > 1;   // (slots VRAM freed: filled at once)
            if (!room && !(rc >= 0 && l.heat[(size_t) rb] > l.heat[(size_t) rc] * 1.25f + 1.0f)) continue;
            RamSwap rs;
            rs.layer = l.il;
            rs.in = rb;
            rs.out = room ? -1 : rc;
            rs.slot = ram_free.back();
            ram_free.pop_back();
            ++ram_inflight;
            std::lock_guard<std::mutex> lk(sw_mu);
            ram_queue.push_back(rs);
            rqueued = true;
        }
        if (queued) sw_cv.notify_all();
        if (rqueued) ram_cv.notify_one();
    }

    // lookahead's thread: the ranges the CPU lane asked for, read in (os::prefetch) off the lane, which computes
    void prefetch_loop() {
        std::vector<std::pair<const void*, size_t>> b;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(pf_mu);
                pf_cv.wait(lk, [&] { return quit.load(std::memory_order_relaxed) || !pf_q.empty(); });
                if (quit.load(std::memory_order_relaxed)) return;
                b.swap(pf_q);
            }
            os::prefetch_cold(b);   // (the resident ones would only cost the page walk)
            b.clear();
        }
    }
    // an expert the CPU lane reads from the files (neither in VRAM nor in the RAM tier)
    static bool file_read(const Layer& l, int e) {
        return l.slot[(size_t) e] < 0 && (l.ram.empty() || l.ram[(size_t) e] < 0);
    }
    // the layer whose picks layer l0 guesses a layers ahead (1, 2): the next ones of its stack on its device, MXFP4
    // experts read in place, some of them from the files now (none: nothing to read early - null)
    const Layer* ahead_layer(const Layer& l0, int a) const {
        const int j = l0.il + a, end = l0.mtp ? g.nl + g.n_mtp : g.nl;
        if (j >= end || j >= (int) L.size()) return nullptr;
        const Layer& n = L[(size_t) j];
        if (n.dev != l0.dev || n.mtp != l0.mtp || n.lay.fmt != gg::kFmtMXFP4 || n.gate_w == nullptr || n.n_exp > 1024 ||
            (int) n.slot.size() < n.n_exp)
            return nullptr;
        for (int e = 0; e < n.n_exp; ++e)
            if (file_read(n, e)) return &n;
        return nullptr;
    }
    // lookahead on the CPU lane: request r (layer li, seq) carries the device's guesses for the next layers - the
    // guessed experts read from the files asked for now (each once a window), the guesses kept for the hit counts
    std::vector<std::pair<const void*, size_t>> pf_want;
    void ahead_ask(int li, const Req& r, unsigned seq, int T) {
        pf_want.clear();
        for (int a = 1; a <= 2; ++a) {
            const int m = std::min(r.na[a - 1], a == 1 ? kAhead1 : kPicks), j = li + a;
            if (m <= 0 || j >= (int) L.size()) continue;
            const Layer& n = L[(size_t) j];
            uint8_t* fl = ahead_fl.data() + (size_t) j * 1024;
            uint8_t* rk = ahead_rank.data() + (size_t) j * 1024;
            if (ahead_for[(size_t) j] != seq + (unsigned) a) {
                std::memset(fl, 0, 1024);
                std::memset(rk, 0, 1024);
                ahead_for[(size_t) j] = seq + (unsigned) a;
            }
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < m; ++i) {
                    const int e = r.ahead[t][(a - 1) * kAhead1 + i];
                    if (e < 0 || e >= n.n_exp || e >= (int) n.slot.size()) continue;
                    fl[e] |= a == 2 ? 4 : (i < n.k_exp ? 3 : 2);
                    if (a == 1 && (rk[e] == 0 || rk[e] > i + 1)) rk[e] = (uint8_t) (i + 1);
                    if ((fl[e] & 8) || !file_read(n, e)) continue;
                    fl[e] |= 8;
                    if (store) store->prefetch(j, e);
                    else
                        for (int q = 0; q < 3; ++q) pf_want.push_back({mx_mat(n, e, q), mx_part_bytes(n, q)});
                    ++st.ahead_pf;
                }
        }
        if (pf_want.empty()) return;
        {
            std::lock_guard<std::mutex> lk(pf_mu);
            pf_q.insert(pf_q.end(), pf_want.begin(), pf_want.end());
        }
        pf_cv.notify_one();
    }

    // the decode path's host side: each published routing in turn - its CPU lane experts (if any) into the answer's
    // part, then the answer's seq.  Spins while requests come (a token's 40 layers), naps after 2 ms without one.
    void serve_lane() {
        unsigned next = 1;
        auto last = std::chrono::steady_clock::now();
        std::vector<Ent> ents;
        while (!quit.load(std::memory_order_relaxed)) {
            Req& r = reqs[next % kRing];
            if (*(volatile unsigned*) &r.seq != next) {
                cpu_relax();
                // (a decode layer takes ~2.8 ms on a Strix Halo, longer than the old 2 ms spin window, so the thread slept
                // before every request and Windows' sleep_for(20 us) wakes after ~1 ms or more: the lane's round trip.  Now it
                // spins for DS4_LANE_SPIN_MS (default 250) after its last answer, i.e. through a whole decode token.)
                static const long spin_ms = std::getenv("DS4_LANE_SPIN_MS") ? std::atol(std::getenv("DS4_LANE_SPIN_MS")) : 250;
                if (std::chrono::steady_clock::now() - last > std::chrono::milliseconds(spin_ms)) {
#if defined(_WIN32)
                    // DS4_LANE_NAP=sleep turns it off: a high-resolution waitable timer (Windows 10 1803+) instead of sleep_for's timer tick
                    static const bool hires = !(std::getenv("DS4_LANE_NAP") != nullptr && std::string(std::getenv("DS4_LANE_NAP")) == "sleep");
                    static HANDLE tmr = hires ? CreateWaitableTimerExW(nullptr, nullptr, 0x2 /* CREATE_WAITABLE_TIMER_HIGH_RESOLUTION */, TIMER_ALL_ACCESS) : nullptr;
                    if (tmr != nullptr) {
                        LARGE_INTEGER due;
                        due.QuadPart = -200;   // 20 us in 100 ns units
                        SetWaitableTimer(tmr, &due, 0, nullptr, nullptr, FALSE);
                        WaitForSingleObject(tmr, 5);
                    } else
#endif
                    std::this_thread::sleep_for(std::chrono::microseconds(20));
                }
                continue;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            const int li = layer_of[next % kRing].load(std::memory_order_acquire);
            Resp& rp = resps[next % kRing];
            const int T = std::max(1, std::min(kWin, r.T));
            for (int t = 0; t < T; ++t) rp.cpu[t] = 0;
            if (li >= 0 && li < (int) L.size()) {
                Layer& l = L[(size_t) li];
                uint32_t* pr = prof.data() + (size_t) li * g.n_exp;
                ents.clear();
                if (store && li > 0 && li < g.nl) store->cancel_stale(li - 1);   // (guesses for layers the token has passed: dropped unread)
                // lookahead: the guesses made for this layer (when they are for this request), counted on decode
                const uint8_t* fl = T == 1 && !ahead_for.empty() && ahead_for[(size_t) li] == next
                                        ? ahead_fl.data() + (size_t) li * 1024 : nullptr;
                // the window's CPU-lane picks by expert (up to 8 tokens an entry), their weights by entry row
                float wv[kWin * kPicks];
                int tok_of[kWin * kPicks];
                int nslots = 0;
                std::vector<std::pair<int, int>> picks;   // (expert, t * kPicks + i), CPU-lane only
                if (trace_f != nullptr && !l.mtp) {   // DS4_TRACE=file: one record a route request: u8 layer, u8 T, T x k int16 expert ids
                    // DS4_TRACE_AHEAD=1: T has bit 7 set and the guesses follow: u8 na1, u8 na2, T x (na1 + na2) int16 (the
                    // next layer's likely picks by rank, then the one after's), as the lookahead made them for THIS layer's
                    // predecessors (the request carries the guesses for the layers after it)
                    static const bool with_ahead = std::getenv("DS4_TRACE_AHEAD") != nullptr;
                    uint8_t hd[2] = {(uint8_t) li, (uint8_t) (T | (with_ahead ? 0x80 : 0))};
                    std::fwrite(hd, 1, 2, trace_f);
                    for (int t = 0; t < T; ++t)
                        for (int i = 0; i < l.k_exp; ++i) {
                            const int16_t v = (int16_t) r.ids[t][i];
                            std::fwrite(&v, 2, 1, trace_f);
                        }
                    if (with_ahead) {
                        const uint8_t na[2] = {(uint8_t) std::max(0, std::min(r.na[0], kAhead1)), (uint8_t) std::max(0, std::min(r.na[1], kPicks))};
                        std::fwrite(na, 1, 2, trace_f);
                        for (int t = 0; t < T; ++t) {
                            for (int i = 0; i < na[0]; ++i) {
                                const int16_t v = (int16_t) r.ahead[t][i];
                                std::fwrite(&v, 2, 1, trace_f);
                            }
                            for (int i = 0; i < na[1]; ++i) {
                                const int16_t v = (int16_t) r.ahead[t][kAhead1 + i];
                                std::fwrite(&v, 2, 1, trace_f);
                            }
                        }
                    }
                }
                const bool dsk = disk_skip_w > 0.0f && !l.mtp && li >= disk_skip_from && li < g.nl;
                for (int t = 0; t < T; ++t) {
                    float wsum = 0.0f;   // (DISK_SKIP_W: the token's route weights)
                    if (dsk)
                        for (int i = 0; i < l.k_exp; ++i)
                            if (r.ids[t][i] >= 0 && r.ids[t][i] < l.n_exp) wsum += r.w[t][i];
                    for (int i = 0; i < l.k_exp; ++i) {
                        const int e = r.ids[t][i];
                        if (e < 0 || e >= l.n_exp) continue;
                        ++pr[e];
                        l.heat[(size_t) e] += 1.0f;
                        if (fl != nullptr) {
                            ++st.ahead_picks;
                            st.ahead_hit_k += fl[e] & 1;
                            st.ahead_hit += (fl[e] & 6) != 0;
                            st.ahead_hit2 += (fl[e] >> 2) & 1;
                        }
                        if (l.slot[(size_t) e] < 0) {
                            l.cpu_hits.push_back(e);
                            if (T == 1 && !l.zc.empty() && l.zc[(size_t) e]) {   // the device computed it from the store's slot
                                store->touch(li, e);
                                ++st.store_zc;
                                continue;
                            }
                            if (dsk && r.w[t][i] < disk_skip_w * wsum && !in_ram(l, e)) {
                                ++st.skipped;
                                continue;
                            }
                            if (!l.ram.empty() && l.ram[(size_t) e] >= 0) ++st.ram_experts;
                            else if (fl != nullptr) {
                                ++st.ahead_cold;
                                st.ahead_cold_hit += (fl[e] >> 3) & 1;
                            }
                            picks.push_back({e, t * kPicks + i});
                        }
                    }
                }
                if (fl != nullptr) {   // the guesses by rank: how many were picked
                    const uint8_t* rk = ahead_rank.data() + (size_t) li * 1024;
                    for (int e = 0; e < l.n_exp && e < 1024; ++e) {
                        if (rk[e] == 0 || rk[e] > 16) continue;
                        bool hit = false;
                        for (int i = 0; i < l.k_exp; ++i) hit = hit || r.ids[0][i] == e;
                        ++st.ahead_rank_n[rk[e] - 1];
                        st.ahead_rank_hit[rk[e] - 1] += hit;
                    }
                }
                if (ahead_mode > 0 && !ahead_for.empty()) ahead_ask(li, r, next, T);
                std::sort(picks.begin(), picks.end());
                for (size_t a = 0; a < picks.size();) {
                    Ent en;
                    en.e = picks[a].first;
                    en.slot0 = nslots;
                    en.n = 0;
                    while (a < picks.size() && picks[a].first == en.e && en.n < ent_cap(l)) {
                        const int t = picks[a].second / kPicks, i = picks[a].second % kPicks;
                        en.tok[en.n++] = t;
                        tok_of[nslots] = t;
                        wv[nslots] = r.w[t][i];
                        ++nslots;
                        ++a;
                    }
                    ents.push_back(en);
                }
                // lookahead: the entries asked for early (or in the RAM tier) computed first, so the ones asked for
                // only now get the others' time to come in from the SSD (their slots, so the sums, stay the same)
                int ord[kWin * kPicks];
                const int* po = nullptr;
                if (ahead_order && !ahead_for.empty() && ahead_for[(size_t) li] == next && ents.size() > 1 &&
                    ents.size() <= (size_t) (kWin * kPicks)) {
                    const uint8_t* fa = ahead_fl.data() + (size_t) li * 1024;
                    int n = 0;
                    for (int pass = 0; pass < 2; ++pass)
                        for (size_t q = 0; q < ents.size(); ++q) {
                            const int e = ents[q].e;
                            const bool early = !file_read(l, e) || (fa[e] & 8) != 0;
                            if (early == (pass == 0)) ord[n++] = (int) q;
                        }
                    po = ord;
                }
                static const bool lane_skip = std::getenv("DS4_LANE_SKIP") != nullptr;   // (timing oracle: the lane's compute left out; rp.cpu stays 0)
                if (!ents.empty() && lane_skip) { for (int t = 0; t < T; ++t) rp.cpu[t] = 0; }
                else if (!ents.empty()) {
                    const double tc = now_ms();
                    cpu_experts(l, &r.x[0][0], ents, (size_t) kMaxDim, po);
                    const int D = g.dim;
                    for (int t = 0; t < T; ++t) {
                        bool any = false;
                        for (int sl = 0; sl < nslots; ++sl) any = any || tok_of[sl] == t;
                        if (!any) continue;
                        float* o = rp.part[t];
                        for (int e = 0; e < D; ++e) o[e] = 0.0f;
                        for (int sl = 0; sl < nslots; ++sl) {
                            if (tok_of[sl] != t) continue;
                            const float* y = c_yd.data() + (size_t) sl * D;
                            for (int e = 0; e < D; ++e) o[e] += wv[sl] * y[e];
                        }
                        rp.cpu[t] = 1;
                    }
                    st.cpu_ms += now_ms() - tc;
                    st.cpu_experts += nslots;
                }
            }
            std::atomic_thread_fence(std::memory_order_release);
            *(volatile unsigned*) &rp.seq = next;
            ++next;
            last = std::chrono::steady_clock::now();
        }
    }

    // a GGUF's matrices (BF16 / MXFP4) to its own kernels, EXL3 ones to x3's
    static bool mv_any(const x3::MvJob* j, int n, cudaStream_t s) {
        bool g = false;
        for (int i = 0; i < n; ++i) g = g || j[i].m->fmt != 0;
        return g ? gg::mv(j, n, s) : x3::mv(j, n, s);
    }
    bool mv(Dev& d, std::initializer_list<x3::MvJob> jobs, std::string& err) {
        x3::MvJob j[x3::kMaxMvJobs];
        int n = 0;
        for (const x3::MvJob& x : jobs) j[n++] = x;
        if (!mv_any(j, n, d.st)) {
            err = "exl3 mv failed";
            return false;
        }
        return true;
    }
    bool mv_n(Dev& d, x3::MvJob* j, int n, std::string& err) {
        for (int a = 0; a < n; a += x3::kMaxMvJobs)
            if (!mv_any(j + a, std::min(x3::kMaxMvJobs, n - a), d.st)) {
                err = "exl3 mv failed";
                return false;
            }
        return true;
    }
    // y[c] = x[c] W for the T activations x + c * x_cs (floats), into y + c * y_cs
    static x3::MvJob job(const x3::Mat& m, const float* x, float* y, int T = 1, size_t x_cs = 0, size_t y_cs = 0,
                         float alpha = 1.0f) {
        x3::MvJob j;
        j.m = &m;
        j.xf = x;
        j.y = y;
        j.alpha = alpha;
        j.ncol = T;
        j.xf_cs = x_cs;
        j.y_cs = y_cs;
        return j;
    }

    // ---- the DSpark drafter (DeepSeek's DSparkBlock / forward_spec).  Its blocks' rings hold the trunk's main KV:
    // seed() turns a window's taps (the streams' mean entering layers 37..39) into it, through main_proj / main_norm
    // and each block's wkv / kv_norm / rope / FP8, for the positions the trunk kept.  draft() runs the block [y, noise x
    // (block - 1)] at positions pos .. through the blocks (each query attending to the ring and to the whole block),
    // the trunk's head, the markov bias token by token (greedy) and the confidence head.
    bool seed_drafter(int64_t p0, int n, int row0, std::string& err) {
        if (g.n_mtp == 0 || n <= 0) return true;
        if (n > kWin * kPicks) {
            err = "seeding the drafter with " + std::to_string(n) + " rows";
            return false;
        }
        Dev& d = devs[(size_t) d_dev];
        cudaSetDevice(d.id);
        const int nt = (int) g.d_targets.size(), D = g.dim, HD = g.hd;
        if (dump_d != nullptr) {   // DS4_DSPARK_DUMP: 'S', the first position, the rows, then the taps
            for (Dev& e : devs) {
                cudaSetDevice(e.id);
                CK(cudaStreamSynchronize(e.st));
            }
            cudaSetDevice(d.id);
            const int64_t hdr[4] = {'S', p0, n, (int64_t) nt * D};
            std::fwrite(hdr, 8, 4, dump_d);
            std::fwrite(h_taps + (size_t) row0 * nt * D, 4, (size_t) n * nt * D, dump_d);
            std::fflush(dump_d);
        }
        for (int a = 0; a < n; a += kWin) {   // (a chunk's last 128 rows: 16 a GEMV)
            const int na = std::min(kWin, n - a);
            CK(cudaMemcpyAsync(d.xe, h_taps + (size_t) (row0 + a) * nt * D, (size_t) na * nt * D * 4,
                               cudaMemcpyHostToDevice, d.st));
            if (!mv(d, {job(d_main_proj, d.xe, d.ed + (size_t) a * D, na, (size_t) nt * D, D)}, err)) return false;
        }
        CK(cudaEventRecord(ev_seed, d.st));
        Finish fm;
        fm.w = d_main_norm;
        fm.eps = g.eps;
        finish(d.ed, n, D, D, fm, d.st);
        for (int i = 0; i < g.n_mtp; ++i) {
            Layer& l = L[(size_t) (g.nl + i)];
            if (l.dev != d_dev) {
                err = "the DSpark blocks are split across devices";
                return false;
            }
            for (int a = 0; a < n; a += kWin) {
                const int na = std::min(kWin, n - a);
                if (!mv(d, {job(l.wkv, d.ed + (size_t) a * D, d.kvr, na, D, HD)}, err)) return false;
                Finish fk;
                fk.w = l.kv_norm;
                fk.eps = g.eps;
                fk.rd = g.rd;
                fk.pos = (int) (p0 + a);
                fk.freqs = d.freqs_win;
                fk.quant = 1;
                fk.dst = d.ext + (size_t) g.win * HD;
                fk.dst_stride = HD;
                finish(d.kvr, na, HD, HD, fk, d.st);
                ext_to_ring(d.ext, l.win, p0 + a, g.win, na, HD, d.st);
            }
        }
        return true;
    }

    bool draft(int y, std::vector<int>& out, std::vector<float>& conf, std::string& err, float temperature, float top_p,
               int top_k, float min_p, uint64_t seed, uint64_t counter) {
        const double t0 = now_ms();
        drafts_q = temperature > 0.0f && sample_drafts;
        dq_temp = temperature;
        dq_top_p = top_p;
        dq_top_k = top_k;
        dq_min_p = min_p;
        dq_seed = seed;
        dq_counter = counter;
        const bool ok = draft_(y, out, conf, err);
        if (!ok) drafts_q = false;
        if (max_drafts > 0 && (int) out.size() > max_drafts) {
            out.resize((size_t) max_drafts);
            conf.resize((size_t) max_drafts);
        }
        st.draft_ms += now_ms() - t0;
        ++st.drafts_rounds;
        return ok;
    }
    bool draft_(int y, std::vector<int>& out, std::vector<float>& conf, std::string& err) {
        out.clear();
        conf.clear();
        const int T = g.d_block, D = g.dim, H = g.hc, V = g.vocab, R = g.d_rank;
        const int64_t p0 = pos;
        if (g.n_mtp == 0 || T < 1 || T > kWin || p0 + T > o.max_context) return true;
        int tk[kWin];
        tk[0] = y;
        for (int t = 1; t < T; ++t) tk[t] = g.d_noise;
        Dev& d = devs[(size_t) d_dev];
        cudaSetDevice(d.id);
        for (int t = 0; t < T; ++t) std::memcpy(h_emb + (size_t) t * D, embed_pinned + (size_t) tk[t] * D, (size_t) D * 2);
        CK(emb_upload(d.emb_rows, h_emb, (size_t) T * D * 2, d.st));
        embed(d.emb_rows, T, d.h, D, H, d.st);
        if (ds4_dbg("DS4_DBG_EMB")) {
            ds4_metal_flush();
            float *hp = d.h;
            float hh[4] = {0,0,0,0};
            cudaMemcpy(hh, hp, sizeof(hh), cudaMemcpyDeviceToHost);
            std::fprintf(stderr, "DBG at embed site %d: h[0..3] = %g %g %g %g\n", __LINE__, hh[0], hh[1], hh[2], hh[3]);
        }
        if (ds4_dbg("DS4_DBG_EMB")) {
            ds4_metal_flush();
            float hh[4] = {0, 0, 0, 0};
            cudaMemcpy(hh, d.h, sizeof(hh), cudaMemcpyDeviceToHost);
            std::fprintf(stderr, "DBG after embed: h[0..3] = %g %g %g %g\n", hh[0], hh[1], hh[2], hh[3]);
        }
        init_pre(d.pre, T, d.st);
        WinPlan dp;   // (window attention alone: the mailbox seqs)
        dp.async = !sync_decode && T <= async_max_t;
        for (int i = 0; i < g.n_mtp && dp.async; ++i) {
            const int il = g.nl + i;
            dp.seq[il] = ++seq_issued;
            layer_of[dp.seq[il] % kRing].store(il, std::memory_order_release);
        }
        const size_t hs = (size_t) T * H * D, rs = (size_t) g.win * g.hd;
        if (dump_d != nullptr && h_dump == nullptr &&
            (!halloc(h_dump, (size_t) g.n_mtp * kWin * H * D, err) || !halloc(h_dump_ring, (size_t) g.n_mtp * rs, err)))
            return false;
        for (int i = 0; i < g.n_mtp; ++i) {
            Layer& l = L[(size_t) (g.nl + i)];
            if (dump_d != nullptr) {   // (stream order: the ring the block reads - as BF16 rows - the stream it leaves)
                ring_to_ext(l.win, d.ext, g.win, g.win, g.hd, d.st);
                CK(cudaMemcpyAsync(h_dump_ring + (size_t) i * rs, d.ext, rs * 2, cudaMemcpyDeviceToHost, d.st));
            }
            if (!layer(l, 0, p0, T, dp, err)) return false;
            if (dump_d != nullptr) {
                cudaSetDevice(d.id);
                CK(cudaMemcpyAsync(h_dump + (size_t) i * hs, d.h, hs * 4, cudaMemcpyDeviceToHost, d.st));
            }
        }
        cudaSetDevice(d.id);
        hc_pre_norm(d.h, d.pre, nullptr, D, H, T, g.eps, d.x, d.st);   // (the confidence head reads it before the norm)
        Dev& hd = devs[(size_t) head_dev];
        // (through the host to the head's device, and to the head group's other members - the drafter's device can be
        // the head group's lead itself, a group of every GPU)
        if (d_dev != head_dev || groups[(size_t) hd.grp].mem.size() > 1) {
            CK(cudaMemcpyAsync(d.bounce, d.x, (size_t) T * D * 4, cudaMemcpyDeviceToHost, d.st));
            CK(cudaEventRecord(d.ev_out, d.st));
        }
        if (d_dev != head_dev) {
            cudaSetDevice(hd.id);
            CK(cudaStreamWaitEvent(hd.st, d.ev_out, 0));
            CK(cudaMemcpyAsync(hd.x, d.bounce, (size_t) T * D * 4, cudaMemcpyHostToDevice, hd.st));
        }
        // the drafter's final norm and the trunk's head, on every member of the head group (the others' shares first)
        const std::vector<int>& hg = groups[(size_t) hd.grp].mem;
        const bool mk = mk_split && !drafts_q;   // (sampled drafts: the lead's whole rows, gathered)
        for (int m = (int) hg.size() - 1; m >= 0; --m) {
            Dev& e = devs[(size_t) hg[(size_t) m]];
            cudaSetDevice(e.id);
            if (m > 0) {
                CK(cudaStreamWaitEvent(e.st, d.ev_out, 0));
                CK(cudaMemcpyAsync(e.ao, d.bounce, (size_t) T * D * 4, cudaMemcpyHostToDevice, e.st));
            } else
                CK(cudaMemcpyAsync(e.ao, e.x, (size_t) T * D * 4, cudaMemcpyDeviceToDevice, e.st));
            Finish fn;
            fn.w = e.d_norm;
            fn.eps = g.eps;
            finish(e.ao, T, D, D, fn, e.st);
            if (!head_share(e, m, e.ao, T, err, !mk)) return false;
        }
        h_idx[0] = y;
        if (mk) {
            // the markov steps over the group: each member its rows of the bias and its share's argmax, every member
            // then the global one (the sampler's tie rule) and its E row - the same drafts as the lead's whole rows,
            // each step ~1/nm of their 66 MB and no gather of the shares
            const int nm = (int) hg.size();
            const unsigned seq = ++mk_seq;
            for (int m = 0; m < nm; ++m) {
                Dev& e = devs[(size_t) hg[(size_t) m]];
                cudaSetDevice(e.id);
                CK(cudaMemcpyAsync(e.mk_ints, h_idx, 4, cudaMemcpyHostToDevice, e.st));
                CK(cudaMemcpyAsync(e.mk_erow, h_mk_embed + (size_t) y * R, (size_t) R * 2, cudaMemcpyHostToDevice, e.st));
            }
            for (int i = 0; i < T; ++i)
                for (int m = 0; m < nm; ++m) {
                    Dev& e = devs[(size_t) hg[(size_t) m]];
                    cudaSetDevice(e.id);
                    float* row = m == 0 ? hd.logits + (size_t) i * V : e.logits_part + (size_t) i * e.mk_vn;
                    markov_share(row, e.mk_share, e.mk_erow, e.mk_vn, R, e.st);
                    argmax_share(row, e.mk_vn, e.mk_v0, mk_slots + ((size_t) i * nm + m) * 4, seq, e.st);
                    markov_pick(mk_slots + (size_t) i * nm * 4, nm, seq, e.mk_ints + i + 1, h_mk_embed, e.mk_erow, R,
                                e.st);
                }
            cudaSetDevice(hd.id);
        } else {
            CK(cudaMemcpyAsync(d_ints, h_idx, 4, cudaMemcpyHostToDevice, hd.st));
            strata::dsv41k::SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            if (drafts_q) {   // the request's own chain: draft i drawn as the model's token i would be (verify's counter + i)
                sp.greedy = false;
                sp.temperature = dq_temp * (q_temp_scale > 0.0f ? q_temp_scale : 1.0f);
                sp.top_p = dq_top_p;
                sp.top_k = dq_top_k;
                sp.min_p = dq_min_p;
                sp.seed = dq_seed;
            }
            for (int i = 0; i < T; ++i) {   // the markov bias of the token before, then that position's draft
                if (d_mk_head_q8 != nullptr) markov_add_q8(hd.logits + (size_t) i * V, d_mk_embed, d_mk_head_q8, d_ints + i, V, R, hd.st);
                else markov_add(hd.logits + (size_t) i * V, d_mk_embed, d_mk_head, d_ints + i, V, R, hd.st);
                if (drafts_q) {
                    sp.counter = dq_counter + (uint64_t) i;
                    strata::dsv41k::sample_draft(hd.logits + (size_t) i * V, 1, V, sp, d_specq + i, d_ints + i + 1, hd.st);
                } else
                    strata::dsv41k::sample_tokens(hd.logits + (size_t) i * V, 1, V, nullptr, 0, sp, d_ints + i + 1, hd.st);
            }
        }
        confidence(hd.x, d_mk_embed, d_ints, d_conf, T, D, R, d_confv, hd.st);
        int tok[kWin + 1];
        float cf[kWin];
#if defined(STRATA_METAL)
        ds4_metal_flush();   // (a host read of a device result: commit the batched dispatches first)
#endif
        CK(cudaMemcpyAsync(h_idx, d_ints, (size_t) (T + 1) * 4, cudaMemcpyDeviceToHost, hd.st));
        CK(cudaMemcpyAsync(h_w, d_confv, (size_t) T * 4, cudaMemcpyDeviceToHost, hd.st));
        CK(cudaStreamSynchronize(hd.st));
        for (int i = 0; i <= T; ++i) tok[i] = h_idx[i];
        for (int i = 0; i < T; ++i) cf[i] = h_w[i];
        if (dump_d != nullptr) {   // 'D', the block's position, y, T, then the drafts, their confidences, the hidden
            std::vector<float> xs((size_t) T * D);   // rows before the final norm; then each block's ring and stream
            CK(cudaMemcpy(xs.data(), hd.x, xs.size() * 4, cudaMemcpyDeviceToHost));
            cudaSetDevice(d.id);
            CK(cudaStreamSynchronize(d.st));
            const int64_t hdr[4] = {'D', p0, y, T};
            std::fwrite(hdr, 8, 4, dump_d);
            std::fwrite(tok + 1, 4, (size_t) T, dump_d);
            std::fwrite(cf, 4, (size_t) T, dump_d);
            std::fwrite(xs.data(), 4, xs.size(), dump_d);
            for (int i = 0; i < g.n_mtp; ++i) {
                std::fwrite(h_dump_ring + (size_t) i * rs, 2, rs, dump_d);
                std::fwrite(h_dump + (size_t) i * hs, 4, hs, dump_d);
            }
            std::fflush(dump_d);
            cudaSetDevice(hd.id);
        }
        for (int i = 0; i < T; ++i) {
            out.push_back(tok[i + 1]);
            conf.push_back(cf[i]);
        }
        return true;
    }

    int sample_row(int row, float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter,
                   const float* bias) {
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        strata::dsv41k::SamplerParams sp;
        sp.temperature = temperature;
        sp.greedy = temperature <= 0.0f;
        sp.top_p = top_p;
        sp.top_k = top_k;
        sp.min_p = min_p;
        sp.seed = seed;
        sp.counter = counter;
        sp.logit_bias = bias;
        strata::dsv41k::sample_tokens(d.logits + (size_t) row * g.vocab, 1, g.vocab, nullptr, 0, sp, d.tok, d.st);
        int t = -1;
#if defined(STRATA_METAL)
        ds4_metal_flush();   // (a host read of a device result: commit the batched dispatches first)
#endif
        cudaMemcpyAsync(&t, d.tok, 4, cudaMemcpyDeviceToHost, d.st);
        cudaStreamSynchronize(d.st);
        return t;
    }

    // [y, drafts] at positions pos ..: the trunk's verdict, its next token, and the state back to what it kept.
    // Greedy: the drafts the trunk's argmax agrees with.  Sampled: the speculative-sampling rule over the request's own
    // chain (top_k, top_p, min_p, temperature): draft d stands with probability min(1, p(d) / q(d)), q the distribution
    // it was drawn from (the drafter's argmax: a point at d, so p(d)), a rejected one is replaced by a draw from the
    // residual max(0, p - q) - so each kept token is distributed as the trunk's own draw.
    bool verify(int y, const std::vector<int>& drafts, float temperature, float top_p, int top_k, float min_p,
                uint64_t seed, uint64_t counter, int& accepted, int& next, std::string& err) {
        const int n = (int) std::min<size_t>(drafts.size(), (size_t) kWin - 1);
        const int T = n + 1, V = g.vocab;
        const int64_t p0 = pos;
        int tk[kWin];
        tk[0] = y;
        for (int i = 0; i < n; ++i) tk[i + 1] = drafts[(size_t) i];
        spec_win = true;
        const double tv = now_ms();
        const bool ok = forward(tk, T, T, err, 0);
        st.verify_ms += now_ms() - tv;
        spec_win = false;
        if (!ok) return false;
        Dev& hd = devs[(size_t) head_dev];
        cudaSetDevice(hd.id);
        int m = 0;
        if (temperature <= 0.0f) {
            strata::dsv41k::SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            strata::dsv41k::sample_tokens(hd.logits, T, V, nullptr, 0, sp, d_ints + 16, hd.st);   // (the T rows at once)
#if defined(STRATA_METAL)
        ds4_metal_flush();   // (a host read of a device result: commit the batched dispatches first)
#endif
            CK(cudaMemcpyAsync(h_idx, d_ints + 16, (size_t) T * 4, cudaMemcpyDeviceToHost, hd.st));
            CK(cudaStreamSynchronize(hd.st));
            while (m < n && h_idx[m] == drafts[(size_t) m]) ++m;
            next = h_idx[m];
        } else {
            // row i keeps draft i with probability min(1, p(d) / q(d)), else draws from the residual max(0, p - q)
            // (never d itself): each row a token distributed as the model's own draw at that position, p through the
            // request's chain - the top_k / top_p / min_p cut included (the drafter's argmax can lie outside it: p(d) 0,
            // never kept); past the first rejection the rows do not count
            const bool own = drafts_q && dq_temp == temperature && dq_top_p == top_p && dq_top_k == top_k &&
                             dq_min_p == min_p && dq_seed == seed && dq_counter == counter;
            if (!own) {   // the drafter's argmax: q a point at each draft
                for (int i = 0; i < n; ++i) {
                    h_specq[(size_t) i].n = 1;
                    h_specq[(size_t) i].ids[0] = drafts[(size_t) i];
                    h_specq[(size_t) i].p[0] = 1.0;
                }
                CK(cudaMemcpyAsync(d_specq, h_specq.data(), (size_t) n * sizeof(strata::dsv41k::SpecList),
                                   cudaMemcpyHostToDevice, hd.st));
            }
            for (int i = 0; i < n; ++i) h_idx[i] = drafts[(size_t) i];
            CK(cudaMemcpyAsync(d_ints + 32, h_idx, (size_t) n * 4, cudaMemcpyHostToDevice, hd.st));
            strata::dsv41k::SamplerParams sp;
            sp.greedy = false;
            sp.temperature = temperature;
            sp.top_p = top_p;
            sp.top_k = top_k;
            sp.min_p = min_p;
            sp.seed = seed;
            sp.counter = counter;
            strata::dsv41k::sample_verify(hd.logits, n, V, sp, d_specq, d_ints + 32, d_ints + 16, hd.st);
#if defined(STRATA_METAL)
        ds4_metal_flush();   // (a host read of a device result: commit the batched dispatches first)
#endif
            CK(cudaMemcpyAsync(h_idx, d_ints + 16, (size_t) n * 4, cudaMemcpyDeviceToHost, hd.st));
            CK(cudaStreamSynchronize(hd.st));
            while (m < n && h_idx[m] == drafts[(size_t) m]) ++m;
            next = m < n ? h_idx[m] : sample_row(n, temperature, top_p, top_k, min_p, seed, counter + (uint64_t) n, nullptr);
        }
        drafts_q = false;
        rollback(p0, T, m + 1);
        if (!seed_drafter(p0, m + 1, 0, err)) return false;
        accepted = m;
        st.drafts_offered += n;
        st.drafts_accepted += m;
        for (int i = 0; i < n && i <= m && i < 8; ++i) {
            ++st.draft_reached[i];
            if (i < m) ++st.draft_kept[i];
        }
        return true;
    }

    // a verified window keeps its first `keep` positions: the ring rows and the compressors' state of the rest undone
    // (the compressed KV and the index keys past them are written again when those positions come)
    void rollback(int64_t p0, int T, int keep) {
        if (keep >= T) return;
        // the ring rows: one launch a device for all its layers (a launch a layer and member was ~300 on 8-way tensor
        // parallel, a millisecond of the host's before the next draft)
        std::vector<RingBatch> rb(devs.size());
        for (int il = 0; il < g.nl; ++il)
            for (int m = 0; m < L[(size_t) il].nm; ++m) {
                Layer& l = member(L[(size_t) il], m);
                RingBatch& b = rb[(size_t) l.dev];
                if (b.n == kMaxRingBatch) {   // (more layers on a device than a batch takes: this one alone)
                    cudaSetDevice(devs[(size_t) l.dev].id);
                    ring_restore(l.win, l.ring_bak, p0, g.win, keep, T, g.hd, devs[(size_t) l.dev].st);
                    continue;
                }
                b.ring[b.n] = l.win;
                b.bak[b.n++] = l.ring_bak;
            }
        for (size_t di = 0; di < devs.size(); ++di)
            if (rb[di].n > 0) {
                cudaSetDevice(devs[di].id);
                ring_restore_batch(rb[di], p0, g.win, keep, T, g.hd, devs[di].st);
            }
        for (int il = 0; il < g.nl; ++il)
            for (int m = 0; m < L[(size_t) il].nm; ++m) {
                Layer& l = member(L[(size_t) il], m);
                Dev& d = devs[(size_t) l.dev];
                if (l.c_snap != nullptr) {
                    cudaSetDevice(d.id);
                    const size_t ns = (size_t) l.ratio * g.hd * 4;
                    cudaMemcpyAsync(l.c_kv, l.c_snap, ns, cudaMemcpyDeviceToDevice, d.st);
                    cudaMemcpyAsync(l.c_score, l.c_snap + (size_t) l.ratio * g.hd, ns, cudaMemcpyDeviceToDevice, d.st);
                    compress_win(l.c_rows, l.c_rows + (size_t) kWin * g.hd, l.c_kv, l.c_score, l.ratio, (int) p0, keep,
                                 g.hd, l.c_norm, g.eps, d.latent, d.st);
                }
            }
        pos = p0 + keep;
        toks.resize((size_t) pos);
        ctoks.resize((size_t) pos);
        slot_held = false;
    }

    // ---- a window: T tokens at positions pos .. pos + T - 1 through every layer; the logits of its last `logits` rows
    bool step(int token, bool want_logits, std::string& err) { return forward(&token, 1, want_logits ? 1 : 0, err); }


    // layer L0 for a chunk of T tokens at p0 as group member mi: layer()'s arithmetic over the chunk's rows (the
    // projections as GEMMs, the attention and the indexer in slabs, the routed experts as sorted rows: the VRAM ones in
    // place, the CPU lane's streamed in batches).  kv_only (a long prompt's first pass, Run::stop): only what the layers
    // after it read of these rows - its compressed KV and index keys - and nothing of its own attention or experts
    bool layer_chunk(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, Streamer& S, std::string& err,
                     bool kv_only = false) {
        Layer& l = member(L0, mi);
        Dev& d = devs[(size_t) l.dev];
        Chunk& c = *chunks[(size_t) l.dev];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        prof_mark(d, 0);
        const int nm = L0.nm;
        const int D = g.dim, H = g.hc, HD = g.hd, NHL = l.ahn, HQ = NHL * HD, OG = l.ogn, FF = l.ffn;
        const ArGroup& ab = groups[(size_t) L0.grp].arb;
        const float* freqs = l.ratio ? d.freqs_comp : d.freqs_win;
        // Engram, in slabs of 256 tokens
        if (l.eg >= 0) {
            const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
            const size_t per = (size_t) cols * (ehd + ehd / 32);
            const int slot = wp.eg_slot >= 0 ? wp.eg_slot : wp.k % 2;   // (read ahead by the run: two in flight)
            uint8_t* host = c.eg_host[slot] + (size_t) l.eg * c.C * per;
            CK(cudaMemcpyAsync(c.eg_rows, host, (size_t) T * per, cudaMemcpyHostToDevice, s));
            CK(cudaEventRecord(c.eg_ev[slot], s));
            for (int s0 = 0; s0 < T; s0 += 256) {
                const int ns = std::min(256, T - s0);
                engram_dequant_tok(c.eg_rows + (size_t) s0 * per, per, cols, ns, ehd, c.eg_x, s, (int) eg_fmt[(size_t) l.eg]);
                x3::rows_f16(c.eg_x, cols * ehd, nullptr, ns, cols * ehd, c.eg_x16, s);
                if (!gemm(d, c, l.eg_wkv, c.eg_x16, cols * ehd, ns, c.eg_kv, (H + 1) * D, 0.0f, err)) return false;
                engram_gate(c.h + (size_t) s0 * H * D, c.eg_kv, l.eg_w, H, D, ns, g.eps, s);
            }
        }
        prof_mark(d, 4);
        // attention
#if defined(STRATA_METAL)
        // the mix and the split in one dispatch (hc_mix + hc_split were two, back to back on the same token)
        hc_mix_split(c.h, l.hca_fn, g.hc * g.dim, (2 + g.hc) * g.hc, T, g.eps, c.mix, l.hca_scale, l.hca_base, H,
                     g.sinkhorn, g.hc_eps, c.a_pre, c.a_post, c.a_comb, s);
#else
        if (!hc_mix_chunk(d, c, c.h, l.hca_fn, T, c.mix, err)) return false;
        hc_split(c.mix, l.hca_scale, l.hca_base, H, g.sinkhorn, g.hc_eps, T, c.a_pre, c.a_post, c.a_comb, s);
#endif
        dump_sub(d, "mix0", l.il, p0, T, c.mix, (size_t) T * 32 * 4);
        hc_pre_norm(c.h, c.pre, l.attn_norm, D, H, T, g.eps, c.x, s);
        dump_sub(d, "a_mix", l.il, p0, T, c.mix, (size_t) T * 32 * 4);
        dump_sub(d, "a_pre", l.il, p0, T, c.a_pre, (size_t) T * 8 * 4);
        dump_sub(d, "a_post", l.il, p0, T, c.a_post, (size_t) T * 8 * 4);
        dump_sub(d, "a_comb", l.il, p0, T, c.a_comb, (size_t) T * 64 * 4);
        dump_sub(d, "attn_x", l.il, p0, T, c.x, (size_t) T * D * 4);
        x3::rows_f16(c.x, D, nullptr, T, D, c.x16, s);
        prof_mark(d, 5);
        if (!kv_only && (!gemm(d, c, l.wq_a, c.x16, D, T, c.qr, g.q_lora, 0.0f, err) ||
                         !gemm(d, c, l.wkv, c.x16, D, T, c.kvr, HD, 0.0f, err)))
            return false;
        if (l.kv_src && !gemm(d, c, l.c_wkv, c.x16, D, T, c.ckr, HD, 0.0f, err)) return false;
        if (l.kv_src && l.ratio > 1 && !gemm(d, c, l.c_wgate, c.x16, D, T, c.cgr, HD, 0.0f, err)) return false;
        prof_mark(d, 6);
        const bool snap = !kv_only && wp.snap_t >= 0 && mi == 0 && L0.snap_win != nullptr;
        if (!kv_only) {
            dump_sub(d, "qr_raw", l.il, p0, T, c.qr, (size_t) T * g.q_lora * 4);
            dump_sub(d, "kvr_raw", l.il, p0, T, c.kvr, (size_t) T * HD * 4);
            Finish fq;
            fq.w = l.q_norm;
            fq.eps = g.eps;
            finish(c.qr, T, g.q_lora, g.q_lora, fq, s);
            x3::rows_f16(c.qr, g.q_lora, nullptr, T, g.q_lora, c.qr16, s);
            // the window rows: the ring's (positions p0 - win ..) then the chunk's
            ring_to_ext(l.win, c.ext, p0, g.win, HD, s);
            Finish fk;
            fk.w = l.kv_norm;
            fk.eps = g.eps;
            fk.rd = g.rd;
            fk.pos = (int) p0;
            fk.freqs = freqs;
            fk.quant = 1;
            fk.dst = c.ext + (size_t) g.win * HD;
            fk.dst_stride = HD;
            finish(c.kvr, T, HD, HD, fk, s);
            // feed_mark's checkpoint: the ring as it was before row snap_t - the old ring, then the chunk's rows before it
            if (snap) {
                CK(cudaMemcpyAsync(L0.snap_win, l.win, (size_t) g.win * win_rb(), cudaMemcpyDeviceToDevice, s));
                if (wp.snap_t > 0) ext_to_ring(c.ext, L0.snap_win, p0, g.win, wp.snap_t, HD, s, nullptr);
            }
            ext_to_ring(c.ext, l.win, p0, g.win, T, HD, s, nullptr);
        }
        const uint8_t* ckv = nullptr;
        const long long ss = (long long) o.max_context + 64;
        if (l.ratio > 0) {
            Src& sr = srcs[(size_t) l.src];
            const int R = l.ratio;
            const int64_t g_first = p0 / R, g_end = (p0 + T) / R;
            const int ng = (int) std::max<int64_t>(0, g_end - g_first);
            const float* latent = c.latent;
            if (l.kv_src) {
                if (R == 1) {
                    Finish fc;
                    fc.w = l.c_norm;
                    fc.eps = g.eps;
                    finish(c.ckr, T, HD, HD, fc, s);
                    latent = c.ckr;
                } else {
                    if (snap && L0.snap_kv != nullptr) {   // (each slot: the latest row before snap_t that holds it)
                        const int64_t ps = p0 + wp.snap_t;
                        for (int sl = 0; sl < R; ++sl) {
                            const int64_t q = ps - 1 - (((ps - 1 - sl) % R) + R) % R;
                            const float* kv = q >= p0 ? c.ckr + (size_t) (q - p0) * HD : l.c_kv + (size_t) sl * HD;
                            const float* sc = q >= p0 ? c.cgr + (size_t) (q - p0) * HD : l.c_score + (size_t) sl * HD;
                            CK(cudaMemcpyAsync(L0.snap_kv + (size_t) sl * HD, kv, (size_t) HD * 4, cudaMemcpyDeviceToDevice, s));
                            CK(cudaMemcpyAsync(L0.snap_sc + (size_t) sl * HD, sc, (size_t) HD * 4, cudaMemcpyDeviceToDevice, s));
                        }
                    }
                    compress_win(c.ckr, c.cgr, l.c_kv, l.c_score, R, (int) p0, T, HD, l.c_norm, g.eps, c.latent, s);
                }
            }
            const int clen_max = (int) ((p0 + T) / R);
            if (l.ix_src) {
                Src& ks = srcs[(size_t) l.ik_src];
                if (l.owns_k && ng > 0) {
                    x3::rows_f16(latent, HD, nullptr, ng, HD, c.lat16, s);
                    if (!gemm(d, c, l.i_wk, c.lat16, HD, ng, c.ikr, g.ix_hd, 0.0f, err)) return false;
                    Finish fi;
                    fi.w = l.ik_norm;
                    fi.eps = g.eps;
                    fi.rd = g.rd;
                    fi.pos = (int) (g_first * R);
                    fi.pos_step = R;
                    fi.freqs = freqs;
                    fi.quant = 3;
                    fi.pdst = ks.ik[(size_t) l.dev] + (size_t) g_first * ik_rb();
                    fi.pdst_stride = (long long) ik_rb();
                    finish(c.ikr, ng, g.ix_hd, g.ix_hd, fi, s);
                }
                if (clen_max > 0 && !kv_only) {
                    if (!gemm_f16w(c, l.i_wproj, g.ix_heads, D, c.x16, D, T,
                                   1.0f / std::sqrt((float) g.ix_hd) / std::sqrt((float) g.ix_heads), c.iw, 64, err))
                        return false;
                    for (int a0 = 0; a0 < T; a0 += c.AI) {
                        const int na = std::min(c.AI, T - a0);
                        const int64_t pa = p0 + a0;
                        if (!gemm(d, c, l.i_wq_b, c.qr16 + (size_t) a0 * g.q_lora, g.q_lora, na, c.iq,
                                  g.ix_heads * g.ix_hd, 0.0f, err))
                            return false;
                        Finish fi;
                        fi.rd = g.rd;
                        fi.pos = (int) pa;
                        fi.vpp = g.ix_heads;
                        fi.freqs = freqs;
                        fi.quant = 3;
                        finish(c.iq, na * g.ix_heads, g.ix_hd, g.ix_hd, fi, s);
                        Rows lim;
                        lim.gen = 1;
                        lim.p0 = pa;
                        lim.R = R;
                        lim.kcap = g.ix_topk;
                        lim.ccap = g.cand_topk;
                        lim.blk = std::max(1, g.cand_blk);
                        const int nlast = (int) ((pa + na) / R);   // the slab's widest query
                        int* tk = c.topk + (size_t) a0 * topk_stride();
                        int* cd = c.cand + (size_t) a0 * cand_stride();
                        const float* iw = c.iw + (size_t) a0 * 64;
                        if (nlast <= 0) continue;
                        if (l.uses_cand && (nlast + g.cand_blk - 1) / g.cand_blk > g.cand_topk) {
                            Rows cl = lim;
                            cl.gen = 2;
                            index_score(c.iq, iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, na, g.cand_topk * g.cand_blk,
                                        cd, (int) cand_stride(), g.cand_blk, cl, c.scores, ss, s);
                            Rows tr = lim;
                            tr.gen = 3;
                            topk(c.scores, ss, na, tr, cd, (int) cand_stride(), g.cand_blk, tk, (int) topk_stride(), s);
                        } else {
                            index_score(c.iq, iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, na, nlast, nullptr, 0, 1,
                                        lim, c.scores, ss, s);
                            if (l.cand_src) {
                                Rows cr = lim;
                                cr.kcap = g.cand_topk;
                                cand_blocks(c.scores, ss, na, cr, g.cand_blk, c.bm, (long long) bm_stride(), cd,
                                            (int) cand_stride(), s);
                            }
                            topk(c.scores, ss, na, lim, nullptr, 0, 1, tk, (int) topk_stride(), s);
                        }
                    }
                }
            }
            if (l.kv_src && ng > 0) {
                Finish fc;
                fc.rd = g.rd;
                fc.pos = (int) (g_first * R);
                fc.pos_step = R;
                fc.freqs = freqs;
                fc.quant = 2;
                fc.pdst = sr.ckv[(size_t) l.dev] + (size_t) g_first * ckv_rb();
                fc.pdst_stride = (long long) ckv_rb();
                finish(const_cast<float*>(latent), ng, HD, HD, fc, s);
                if (mi == 0 && sr.remote) {   // the new rows for the other groups' readers (slot k % 2, once
                    const int slot = wp.k % 2;  // they took the chunk before last's)
                    if (wp.k >= 2)
                        for (int r : sr.readers) {
                            host_wait_at_least(sr.x_ack + ((size_t) slot * devs.size() + r) * 32, wp.win - 2);
                            wait_flags(sr.x_ack + ((size_t) slot * devs.size() + r) * 32, 1, 32, wp.win - 2, s);
                        }
                    Xfer x;
                    x.src[0] = fc.pdst;
                    x.dst[0] = sr.stage(slot);
                    x.bytes[0] = (size_t) ng * ckv_rb();
                    x.n = 1;
                    if (l.owns_k && srcs[(size_t) l.ik_src].ik[(size_t) l.dev] != nullptr) {
                        x.src[1] = srcs[(size_t) l.ik_src].ik[(size_t) l.dev] + (size_t) g_first * ik_rb();
                        x.dst[1] = sr.stage(slot) + x_cap() * ckv_rb();
                        x.bytes[1] = (size_t) ng * ik_rb();
                        x.n = 2;
                    }
                    xfer_put(x, sr.flag(slot), wp.win, d.xcnt, s);
                }
            }
            ckv = sr.ckv[(size_t) l.dev];
        }
        if (kv_only) return true;
        prof_mark(d, 7);
        // the attention in slabs: q, the sparse attention, wo_a, wo_b (this member's share) into acc
        if (NHL == 0) CK(cudaMemsetAsync(c.acc, 0, (size_t) T * D * sizeof(*c.acc), s));   // (no heads)
        for (int a0 = 0; a0 < T && NHL > 0; a0 += c.A) {
            const int na = std::min(c.A, T - a0);
            if (!gemm(d, c, l.wq_b, c.qr16 + (size_t) a0 * g.q_lora, g.q_lora, na, c.q, HQ, 0.0f, err)) return false;
            Finish fr;
            fr.rd = g.rd;
            fr.pos = (int) (p0 + a0);
            fr.vpp = NHL;
            fr.freqs = freqs;
            finish(c.q, na * NHL, HD, HD, fr, s);
            prof_mark(d, 8);
            if (a0 == 0 && na == T) dump_sub(d, "q", l.il, p0, T, c.q, (size_t) T * HQ * 4);
            AttnRows ar;
            ar.gen = 1;
            ar.p0 = p0;
            ar.t0 = a0;
            ar.win = g.win;
            ar.R = l.ratio;
            ar.kcap = g.ix_topk;
            sparse_attn(c.q, c.ext, ckv, c.topk + (size_t) a0 * topk_stride(), (int) topk_stride(), ar, l.sink, NHL, HD,
                        na, 1.0f / std::sqrt((float) HD), c.o, s, kKvFp4, 0, &d.attn_scr);   // (a short last slab)
            prof_mark(d, 9);
            if (a0 == 0 && na == T) dump_sub(d, "o", l.il, p0, T, c.o, (size_t) T * HQ * 4);
            Finish fo;
            fo.rd = g.rd;
            fo.pos = (int) (p0 + a0);
            fo.vpp = NHL;
            fo.freqs = freqs;
            fo.inverse = 1;
            fo.quant = 1;
            finish(c.o, na * NHL, HD, HD, fo, s);
            if (a0 == 0 && na == T) dump_sub(d, "o_fin", l.il, p0, T, c.o, (size_t) T * HQ * 4);
            x3::rows_f16(c.o, HQ, nullptr, na, HQ, c.o16, s);
            const int per = HQ / OG;
            for (int gi = 0; gi < OG; ++gi)
                if (!gemm(d, c, l.wo_a[gi], c.o16 + (size_t) gi * per, HQ, na, c.oa + (size_t) gi * g.o_lora,
                          OG * g.o_lora, 0.0f, err))
                    return false;
            x3::rows_f16(c.oa, OG * g.o_lora, nullptr, na, OG * g.o_lora, c.oa16, s);
            if (!gemm(d, c, l.wo_b, c.oa16, OG * g.o_lora, na, c.acc + (size_t) a0 * D, D, 0.0f, err)) return false;
            if (a0 == 0 && na == T) dump_sub(d, "ao", l.il, p0, T, c.acc, (size_t) T * D * 4);
        }
        if (nm > 1) allreduce(c.acc, T * D, ab, mi, ++c.arb_seq, d.xcnt, s);
        dump_sub(d, "h_in", l.il, p0, T, c.h, (size_t) T * H * D * 4);
        hc_post_inplace(c.acc, c.h, c.a_post, c.a_comb, D, H, T, s);
        if (dump_l != nullptr && mi == 0) dump_layer(d, c.h, 1000 + l.il, p0, T);   // (after the attention, as the window path)
        prof_mark(d, 1);
        // the MoE
#if defined(STRATA_METAL)
        // the FFN half of the same fusion: mix and split in one dispatch
        hc_mix_split(c.h, l.hcf_fn, g.hc * g.dim, (2 + g.hc) * g.hc, T, g.eps, c.mix, l.hcf_scale, l.hcf_base, H,
                     g.sinkhorn, g.hc_eps, c.f_pre, c.f_post, c.f_comb, s);
#else
        if (!hc_mix_chunk(d, c, c.h, l.hcf_fn, T, c.mix, err)) return false;
        hc_split(c.mix, l.hcf_scale, l.hcf_base, H, g.sinkhorn, g.hc_eps, T, c.f_pre, c.f_post, c.f_comb, s);
#endif
        hc_pre_norm(c.h, c.a_pre, l.ffn_norm, D, H, T, g.eps, c.x, s);
        x3::rows_f16(c.x, D, nullptr, T, D, c.x16, s);
        gemv_f32(l.gate_w, c.x, l.n_exp, D, T, c.glog, 1024, s);   // (F32 x: the token path's bits)
        for (int r0 = 0; r0 < T; r0 += kVlRows) {   // (route()'s image bits cover kVlRows rows a call)
            const int nr = std::min(kVlRows, T - r0);
            VlBits vb;
            const bool vl = l.gate_bias_vl != nullptr && vl_bits(p0 + r0, nr, vb);
            route(c.glog + (size_t) r0 * 1024, l.gate_bias, l.n_exp, l.k_exp, nr, g.route_scale,
                  c.ridx + (size_t) r0 * kPicks, c.rw + (size_t) r0 * kPicks, s, l.gate_bias_vl, vl ? &vb : nullptr);
        }
        const int n_vs = L0.n_slots + (int) L0.host_list.size();
        CK(cudaMemcpyAsync(l.vs_dev, l.vs_pin, (size_t) l.n_exp * 4, cudaMemcpyHostToDevice, s));   // (this member's)
        moe_sort(c.ridx, T, l.k_exp, l.vs_dev, n_vs, c.bounds, c.row_tok, c.pos, s);
        CK(cudaMemcpyAsync(c.bounds_pin, c.bounds, (size_t) (n_vs + 1) * 4, cudaMemcpyDeviceToHost, s));
        prof_mark(d, 10);
        // the shared expert into acc while the host waits for the bounds
        if (!gemm(d, c, l.s1, c.x16, D, T, c.sg, FF, 0.0f, err) || !gemm(d, c, l.s3, c.x16, D, T, c.su, FF, 0.0f, err))
            return false;
        swiglu(c.sg, c.su, T * FF, g.lim, c.sg, s);
        x3::rows_f16(c.sg, FF, nullptr, T, FF, c.sh16, s);
        if (!gemm(d, c, l.s2, c.sh16, FF, T, c.acc, D, 0.0f, err)) return false;
        prof_mark(d, 2);
        // the expert stream's copies keep going while the device runs this layer's attention, routing and shared expert:
        // the next items (this layer's likely experts, queued when the layer before routed) go out as they are ready and
        // ring slots free, not only once the bounds are back (Project Maya #360; DS4_FLOW_PUMP_WAIT=0: wait as before)
        static const bool pump_wait = std::getenv("DS4_FLOW_PUMP_WAIT") == nullptr ||
                                      std::atoi(std::getenv("DS4_FLOW_PUMP_WAIT")) != 0;
        if (c.flow && pump_wait)
            while (cudaStreamQuery(s) == cudaErrorNotReady) {
                if (!flow_pump(c, err)) return false;
                std::this_thread::yield();
            }
        CK(cudaStreamSynchronize(s));
        std::vector<int> b(c.bounds_pin, c.bounds_pin + n_vs + 1);
        if (dump_r != nullptr && mi == 0) {
            const size_t nk = (size_t) T * l.k_exp, np = (size_t) T * kPicks;
            std::vector<int> ri(np), rt(nk);
            std::vector<float> gl((size_t) T * l.n_exp), gb((size_t) l.n_exp);
            CK(cudaMemcpy(ri.data(), c.ridx, np * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(rt.data(), c.row_tok, nk * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy2D(gl.data(), (size_t) l.n_exp * 4, c.glog, 1024 * 4, (size_t) l.n_exp * 4, (size_t) T,
                            cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(gb.data(), l.gate_bias, gb.size() * 4, cudaMemcpyDeviceToHost));
            std::lock_guard<std::mutex> lk(dump_mu);
            const int64_t hdr[8] = {l.il, p0, T, l.k_exp, l.n_exp, n_vs, L0.n_slots, kPicks};
            std::fwrite(hdr, 8, 8, dump_r);
            std::fwrite(ri.data(), 4, np, dump_r);
            std::fwrite(L0.vs_pin, 4, (size_t) l.n_exp, dump_r);
            std::fwrite(b.data(), 4, b.size(), dump_r);
            std::fwrite(rt.data(), 4, nk, dump_r);
            std::fwrite(gl.data(), 4, gl.size(), dump_r);
            std::fwrite(gb.data(), 4, gb.size(), dump_r);
            std::fflush(dump_r);
        }
        if (mi == 0 && !l.mtp && (pred_f != nullptr || xin_f != nullptr)) {
            std::lock_guard<std::mutex> lk(dump_mu);
            if (pred_f != nullptr) {
                float* dp = nullptr;
                CK(cudaMalloc(&dp, (size_t) T * 256 * 4));
                strata_ds4_launch(pred_proj_k, T, 256, 0, s, c.x, (int) D, dp);
                std::vector<float> hp((size_t) T * 256), hl((size_t) T * l.n_exp);
                CK(cudaMemcpyAsync(hp.data(), dp, hp.size() * 4, cudaMemcpyDeviceToHost, s));
                CK(cudaMemcpy2DAsync(hl.data(), (size_t) l.n_exp * 4, c.glog, 1024 * 4, (size_t) l.n_exp * 4, (size_t) T,
                                     cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                CK(cudaFree(dp));
                std::vector<uint16_t> o(hp.size() + hl.size());
                for (size_t i = 0; i < hl.size(); ++i) o[i] = strata::kernels::f16_from_f32(hl[i]);
                for (size_t i = 0; i < hp.size(); ++i) o[hl.size() + i] = strata::kernels::f16_from_f32(hp[i]);
                const uint8_t hd[3] = {(uint8_t) l.il, (uint8_t) (T & 255), (uint8_t) (T >> 8)};
                std::fwrite(hd, 1, 3, pred_f);
                std::fwrite(o.data(), 2, o.size(), pred_f);
            }
            if (xin_f != nullptr && xin_got[(size_t) l.il] < xin_n) {   // up to 32 evenly spaced tokens of the chunk
                const int tk = std::min(std::min(T, 32), xin_n - xin_got[(size_t) l.il]);
                const int stride = std::max(1, T / tk);
                std::vector<float> hx((size_t) T * D);
                CK(cudaMemcpyAsync(hx.data(), c.x, hx.size() * 4, cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                const uint8_t hd[3] = {(uint8_t) l.il, (uint8_t) (tk & 255), (uint8_t) (tk >> 8)};
                std::fwrite(hd, 1, 3, xin_f);
                for (int i = 0; i < tk; ++i) std::fwrite(hx.data() + (size_t) std::min(T - 1, i * stride) * D, 4, (size_t) D, xin_f);
                xin_got[(size_t) l.il] += tk;
            }
        }
        if (mi == 0 && trace_f != nullptr && !l.mtp) {   // DS4_TRACE: a chunk's picks, u8 (layer | 0x80), u16 T, T x k int16 ids
            std::vector<int> ri((size_t) T * kPicks);
            CK(cudaMemcpy(ri.data(), c.ridx, ri.size() * 4, cudaMemcpyDeviceToHost));
            std::lock_guard<std::mutex> lk(dump_mu);
            const uint8_t hd[3] = {(uint8_t) (l.il | 0x80), (uint8_t) (T & 255), (uint8_t) (T >> 8)};
            std::fwrite(hd, 1, 3, trace_f);
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < l.k_exp; ++i) {
                    const int16_t v = (int16_t) ri[(size_t) t * kPicks + i];
                    std::fwrite(&v, 2, 1, trace_f);
                }
        }
        if (mi == 0) {   // the profile (the CPU lane's experts: by the plan's order)
            uint32_t* pr = prof.data() + (size_t) l.il * g.n_exp;
            for (int e = 0; e < l.n_exp; ++e) {
                const int v = L0.vs_pin[e];
                const int cnt = b[(size_t) v + 1] - b[(size_t) v];
                pr[e] += (uint32_t) cnt;
                if (cnt > 0) {   // (the expert cache learns from a chunked prompt as from windows: heat, and the CPU lane's hits)
                    l.heat[(size_t) e] += (float) cnt;
                    if (l.slot[(size_t) e] < 0) l.cpu_hits.push_back(e);
                }
            }
            st.experts += (int64_t) T * l.k_exp;
        }
        if (c.flow) {
            if (!flow_layer(d, c, L0, mi, wp.k, T, b, err)) return false;
            if (nm > 1) allreduce(c.acc, T * D, ab, mi, ++c.arb_seq, d.xcnt, s);
            hc_post_inplace(c.acc, c.h, c.f_post, c.f_comb, D, H, T, s);
            CK(cudaMemcpyAsync(c.pre, c.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
            prof_mark(d, 3);
            return true;
        }
        // the streamed experts as their batches arrive (each batch's slots freed for the copies at once), the VRAM ones
        // in shares between them (the device works while the copies run)
        const int di = l.dev, P = Chunk::kParts;
        int nb = 0;
        for (size_t q = S.kp; q < S.b.size() && S.b[q].il == L0.il && S.b[q].k == wp.k; ++q) ++nb;
        int v_done = 0;
        for (int bi = 0; bi < nb; ++bi) {
            const Batch& bt = S.b[S.kp];
            const int half = (int) (S.kp % P);
            CK(cudaStreamWaitEvent(s, c.filled[half], 0));
            const int v0 = L0.n_slots + bt.j0;
            if (!run_set(d, c, l, c.ring + (size_t) half * (c.ring_n / P) * c.ring_stride, v0, c.ring_stride, c.bounds,
                         b.data(), v0, v0 + bt.n, T, err))
                return false;
            CK(cudaEventRecord(c.freed[half], s));
            ++S.kp;
            if (!copy_ahead(d, c, S, di, err)) return false;
            const int v_to = (int) ((int64_t) L0.n_slots * (bi + 1) / nb);
            if (!run_set(d, c, l, l.vram_base, 0, unit(l), c.bounds, b.data(), v_done, v_to, T, err)) return false;
            v_done = v_to;
        }
        if (!run_set(d, c, l, l.vram_base, 0, unit(l), c.bounds, b.data(), v_done, L0.n_slots, T, err)) return false;
        if (nm > 1) allreduce(c.acc, T * D, ab, mi, ++c.arb_seq, d.xcnt, s);
        hc_post_inplace(c.acc, c.h, c.f_post, c.f_comb, D, H, T, s);
        CK(cudaMemcpyAsync(c.pre, c.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
        prof_mark(d, 3);
        return true;
    }

    // the CPU lane's experts of every layer, in index order, and each expert's virtual slot (before a chunk)
    void plan_streams() {
        for (Layer& l : L) {
            if (l.mtp || l.vs_pin == nullptr) continue;
            l.host_list.clear();
            for (int e = 0; e < l.n_exp; ++e)
                if (l.slot[(size_t) e] < 0) l.host_list.push_back(e);
            for (int m = 0; m < l.nm; ++m) {   // (a group's members: each its own copy)
                Layer& x = member(l, m);
                int j = 0;
                for (int e = 0; e < l.n_exp; ++e) x.vs_pin[e] = l.slot[(size_t) e] >= 0 ? l.slot[(size_t) e] : l.n_slots + j++;
            }
        }
    }

    // a chunk's hand-off between groups: h | pre | topk | cand of a chunk of T rows
    Xfer chunk_xfer(uint8_t* host, Chunk& c, int T, bool put) const {
        const size_t nh = (size_t) T * g.hc * g.dim * 4, np = (size_t) T * 8 * 4, nk = (size_t) T * topk_stride() * 4,
                     nc = (size_t) T * cand_stride() * 4;
        void* dv[4] = {c.h, c.pre, c.topk, c.cand};
        uint8_t* hv[4] = {host, host + nh, host + nh + np, host + nh + np + nk};
        const size_t nb[4] = {nh, np, nk, nc};
        Xfer x;
        x.n = 4;
        for (int i = 0; i < 4; ++i) {
            x.src[i] = put ? (const void*) dv[i] : (const void*) hv[i];
            x.dst[i] = put ? (void*) hv[i] : dv[i];
            x.bytes[i] = nb[i];
        }
        return x;
    }

    // a run of chunks (start positions p0[k], sizes T[k], seqs seq[k]) on device di: its layers chunk after chunk, the
    // groups pipelined (a group takes chunk k + 1 while the next takes chunk k: the hand-offs and the sources' rows
    // two deep, each slot reused once its readers acknowledged it), the streamed experts' copies running ahead across
    // layers and chunks
    struct Run {
        std::vector<int64_t> p0;
        std::vector<int> T;
        std::vector<unsigned> seq;
        int64_t snap_pos = -1;   // feed_mark: the position whose state before it the layers keep (-1: none)
        // a long prompt's skip (forward_chunks): the cut at row cut_r of chunk cut_k - the chunks before it go through
        // the layers before `yoco`, that layer as layer_chunk's kv_only and none after it; chunk cut_k's rows before
        // cut_r the same, its rows from cut_r on and the chunks after it through every layer (yoco -1: no skip)
        int yoco = -1, cut_k = 0, cut_r = 0;
        bool runs(int k, int il) const { return yoco < 0 || k >= cut_k || il <= yoco; }
        bool kv_only(int k, int il) const { return yoco >= 0 && k < cut_k && il == yoco; }
        int r0(int k, int il) const { return yoco >= 0 && k == cut_k && il >= yoco ? cut_r : 0; }   // the first row
    };
    // layer L0 over chunk k per a long prompt's skip (Run): its kv_only, or its rows from Run::r0 on - the chunk's
    // buffers seen from that row (the yoco layer: its kv_only over the rows before it first)
    bool run_layer(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, Streamer& S, std::string& err,
                   const Run& run, int k, int il) {
        if (run.kv_only(k, il)) return layer_chunk(L0, mi, p0, T, wp, S, err, true);
        const int r0 = run.r0(k, il);
        if (r0 <= 0) return layer_chunk(L0, mi, p0, T, wp, S, err);
        if (il == run.yoco) {
            WinPlan w0 = wp;
            w0.snap_t = -1;
            if (!layer_chunk(L0, mi, p0, r0, w0, S, err, true)) return false;
        }
        Chunk& c = *chunks[(size_t) member(L0, mi).dev];
        float* const h0 = c.h;
        float* const pre0 = c.pre;
        c.h += (size_t) r0 * g.hc * g.dim;
        c.pre += (size_t) r0 * 8;
        WinPlan w1 = wp;
        w1.snap_t = wp.snap_t >= r0 ? wp.snap_t - r0 : -1;
        const bool ok = layer_chunk(L0, mi, p0 + r0, T - r0, w1, S, err);
        c.h = h0;
        c.pre = pre0;
        return ok;
    }

    // layer-major (c.lm: a run of several chunks, device di every trunk layer, the flow set up by run_dev_chunks):
    // layer after layer, each through all the chunks in order - a chunk's state to host memory after each of its
    // layers and back before the next - so a layer's experts are read from the files once for the whole run (the
    // layer cache keeps what a chunk read for the chunks after it), where chunk after chunk reads them once a chunk
    bool run_dev_layers(int di, const Run& run, std::string& err) {
        Dev& d = devs[(size_t) di];
        Chunk& c = *chunks[(size_t) di];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int D = g.dim, H = g.hc, nk = (int) run.T.size();
        Streamer S;   // (none: the flow streams)
        const size_t hb = chunk_hand_bytes();
        // the Engram rows: (layer, chunk) after (layer, chunk) in the run's order, each read on a helper thread while
        // the one before runs (slot u % 2, once unit u - 2's copy of it is done)
        std::vector<std::pair<int, int>> egu;   // (Engram layer, chunk)
        for (int il : c.fl_layers)
            if (L[(size_t) il].eg >= 0)
                for (int k = 0; k < nk; ++k)
                    if (run.runs(k, il)) egu.push_back({L[(size_t) il].eg, k});
        const size_t eg_per = (size_t) (pk.eg_ngram - 1) * pk.eg_heads * (pk.eg_hd + pk.eg_hd / 32);
        auto read_eg = [this, &c, &run, &egu, eg_per](size_t u) {
            if (u >= 2) cudaEventSynchronize(c.eg_ev[u % 2]);
            const int li = egu[u].first, k = egu[u].second;
            engram_rows(run.p0[(size_t) k], run.T[(size_t) k], li, c.eg_host[u % 2] + (size_t) li * c.C * eg_per);
        };
        size_t eg_u = 0;
        std::future<void> eg_next;
        if (!egu.empty()) eg_next = std::async(std::launch::async, read_eg, (size_t) 0);
        for (size_t li = 0; li < c.fl_layers.size(); ++li) {
            const int il = c.fl_layers[li];
            Layer& L0 = L[(size_t) il];
            const bool last_l = li + 1 == c.fl_layers.size();
            for (int k = 0; k < nk; ++k) {
                if (!run.runs(k, il)) continue;   // (a skipped early chunk: nothing after the yoco layer)
                if (halt_now()) return halt_end(di, err);   // (a Stop: one GPU, this one)
                const int64_t p0 = run.p0[(size_t) k];
                const int T = run.T[(size_t) k], slot = k % 2;
                const bool last_k = k + 1 == nk;
                WinPlan wp;
                wp.par = true;
                wp.async = true;
                wp.win = run.seq[(size_t) k];
                wp.k = k;
                wp.snap_t = run.snap_pos >= p0 && run.snap_pos < p0 + T ? (int) (run.snap_pos - p0) : -1;
                if (L0.eg >= 0) {   // this unit's rows ready, the next one's on their way
                    const double te = now_ms();
                    eg_next.get();
                    st.engram_ms += now_ms() - te;
                    wp.eg_slot = (int) (eg_u % 2);
                    if (++eg_u < egu.size()) eg_next = std::async(std::launch::async, read_eg, eg_u);
                }
                if (li == 0) {   // the chunk's input: its embedding rows
                    if (k >= 2) CK(cudaEventSynchronize(c.emb_ev[slot]));
                    for (int t = 0; t < T; ++t)
                        std::memcpy(c.emb_host[slot] + (size_t) t * D, emb_row(p0 + t, toks[(size_t) (p0 + t)]),
                                    (size_t) D * 2);
                    CK(cudaMemcpyAsync(c.emb, c.emb_host[slot], (size_t) T * D * 2, cudaMemcpyHostToDevice, s));
                    CK(cudaEventRecord(c.emb_ev[slot], s));
                    embed(c.emb, T, c.h, D, H, s);
                    if (ds4_dbg("DS4_DBG_EMB")) {
                        ds4_metal_flush();
                        float *hp = c.h;
                        float hh[4] = {0,0,0,0};
                        cudaMemcpy(hh, hp, sizeof(hh), cudaMemcpyDeviceToHost);
                        std::fprintf(stderr, "DBG at embed site %d: h[0..3] = %g %g %g %g\n", __LINE__, hh[0], hh[1], hh[2], hh[3]);
                    }
                    init_pre(c.pre, T, s);
                } else {   // or its state after the layer before
                    const Xfer x = chunk_xfer(c.lm_state + (size_t) k * hb, c, T, false);
                    for (int i = 0; i < x.n; ++i)
                        CK(cudaMemcpyAsync(x.dst[i], x.src[i], x.bytes[i], cudaMemcpyHostToDevice, s));
                }
                if (last_k && g.n_mtp > 0) {   // the drafter's taps: the last chunk's last rows
                    const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
                    if (it != g.d_targets.end()) {
                        const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
                        const int nr = std::min(T, tap_cap);
                        hc_mean(c.h + (size_t) (T - nr) * H * D, nr, H, D, d.tap, D, s);
                        CK(cudaStreamWaitEvent(s, ev_seed, 0));
                        CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4,
                                             (size_t) nr, cudaMemcpyDeviceToHost, s));
                    }
                }
                const bool kvo = run.kv_only(k, il);
                if (!run_layer(L0, 0, p0, T, wp, S, err, run, k, il)) return false;
                if (dump_l != nullptr) dump_layer(d, c.h, il, p0, T);
                if (kvo) continue;   // (no layer after it reads this chunk's state)
                if (!last_l) {   // the chunk's state kept for the next layer
                    const Xfer x = chunk_xfer(c.lm_state + (size_t) k * hb, c, T, true);
                    for (int i = 0; i < x.n; ++i)
                        CK(cudaMemcpyAsync(x.dst[i], x.src[i], x.bytes[i], cudaMemcpyDeviceToHost, s));
                    continue;
                }
                // through every layer: the chunk's end, and the logits
                if ((size_t) k < chunk_done.size()) {
                    CK(cudaEventRecord(chunk_done[(size_t) k], s));
                    chunk_rec[(size_t) k].store(1, std::memory_order_release);
                }
                if (logits_all) {   // every row's logits (tests): the head over the chunk's rows, 16 at a time
                    const size_t base = (size_t) (p0 - run.p0[0]) * g.vocab;
                    for (int r0 = 0; r0 < T; r0 += kWin) {
                        const int nr = std::min(kWin, T - r0);
                        hc_pre_norm(c.h + (size_t) r0 * H * D, c.pre + (size_t) r0 * 8, d.norm_w, D, H, nr, g.eps, d.x, s);
                        if (!head_share(d, d.mi, d.x, nr, err)) return false;
                        CK(cudaMemcpyAsync(chunk_logits.data() + base + (size_t) r0 * g.vocab, d.logits,
                                           (size_t) nr * g.vocab * 4, cudaMemcpyDeviceToHost, s));
                        CK(cudaStreamSynchronize(s));
                    }
                } else if (last_k && n_logits > 0) {   // (the last rows' logits, into the window path's buffers)
                    const int t0 = T - n_logits;
                    hc_pre_norm(c.h + (size_t) t0 * H * D, c.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, s);
                    if (!head_share(d, d.mi, d.x, n_logits, err)) return false;
                }
            }
            lm_layers.store((int) li + 1, std::memory_order_release);
        }
        return true;
    }
    bool run_dev_chunks(int di, const Run& run, std::string& err) {
        Dev& d = devs[(size_t) di];
        Chunk& c = *chunks[(size_t) di];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int D = g.dim, H = g.hc, nk = (int) run.T.size();
        Streamer S;
        const int B = std::max(1, c.ring_n / Chunk::kParts);
        if (c.flow) {   // (the flow decides its streams as the layers route; nothing planned here)
            CK(cudaStreamSynchronize(s));
            CK(cudaStreamSynchronize(c.cs));
            c.items.clear();
            c.base = c.n_read = c.n_dma = c.n_comp = 0;
            c.n_file = 0;
            std::fill(c.stage_item.begin(), c.stage_item.end(), -1);
            c.fl_layers.clear();
            for (int il = 0; il < g.nl; ++il)   // (a group's layers: on each member)
                for (int m = 0; m < L[(size_t) il].nm; ++m)
                    if (member(L[(size_t) il], m).dev == di) c.fl_layers.push_back(il);
            c.fl_T = run.T;
            c.fl_yoco = run.yoco;
            c.fl_cut_k = run.cut_k;
            c.fl_cut_r = run.cut_r;
            c.spec_k = c.spec_il = -1;
            c.wait_ms = 0;
            c.it_file = c.it_ram = c.it_spec = c.it_spec_unused = c.it_lc = 0;
            c.lc_il = -1;
            c.lc_used = 0;
            c.lc_first = c.lc_chk = 0;
            c.io_d0 = c.io->direct_bytes();
            c.io_b0 = c.io->buffered_bytes();
            c.io_busy0 = c.io->busy_ms();
            if (!c.fl_layers.empty()) {
                flow_spec(c, 0, c.fl_layers[0], run.T[0]);
                if (!flow_pump(c, err)) return false;
            }
            if (c.lm) return run_dev_layers(di, run, err);
        }
        for (int k = 0; k < nk && !c.flow; ++k)
            for (int il = 0; il < g.nl; ++il) {
                if (!run.runs(k, il) || run.kv_only(k, il)) continue;   // (a long prompt's skip: no experts there)
                const Layer& L0 = L[(size_t) il];
                bool here = false;
                for (int m = 0; m < L0.nm; ++m) here = here || member(L0, m).dev == di;
                if (!here) continue;
                for (int j0 = 0; j0 < (int) L0.host_list.size(); j0 += B)
                    S.b.push_back({k, il, j0, std::min(B, (int) L0.host_list.size() - j0)});
            }
        if (!S.b.empty() && c.ring == nullptr) {
            err = "no slots to stream experts into on CUDA" + std::to_string(d.id);
            return false;
        }
        std::atomic<bool> ra_stop{false};
        std::thread ra;
        if (file_experts && !S.b.empty()) ra = std::thread([this, &S, di, &ra_stop] { read_ahead(S, di, ra_stop); });
        struct Join {   // (every way out of the run: the read-ahead ends before S does)
            std::atomic<bool>& stop;
            std::thread& th;
            ~Join() {
                stop.store(true);
                if (th.joinable()) th.join();
            }
        } ra_join{ra_stop, ra};
        if (!copy_ahead(d, c, S, di, err)) return false;
        // this device's Engram layers: chunk k's rows read on a helper thread while chunk k - 1 runs (slot k % 2, once
        // chunk k - 2's copy of it is done)
        std::vector<int> egs;
        for (const Layer& l0 : L)
            for (int m = 0; m < l0.nm; ++m)
                if (member(l0, m).dev == di && !l0.mtp && l0.eg >= 0) egs.push_back(l0.eg);
        const size_t eg_per = (size_t) (pk.eg_ngram - 1) * pk.eg_heads * (pk.eg_hd + pk.eg_hd / 32);
        auto read_eg = [this, &c, &run, &egs, eg_per](int k) {
            if (k >= 2) cudaEventSynchronize(c.eg_ev[k % 2]);
            for (int li : egs)
                engram_rows(run.p0[(size_t) k], run.T[(size_t) k], li, c.eg_host[k % 2] + (size_t) li * c.C * eg_per);
        };
        std::future<void> eg_next;
        if (!egs.empty()) eg_next = std::async(std::launch::async, read_eg, 0);
        for (int k = 0; k < nk; ++k) {
            if (!chunk_go(k)) return halt_end(di, err);   // (a Stop: the chunks every device reaches are done)
            const int64_t p0 = run.p0[(size_t) k];
            const int T = run.T[(size_t) k], slot = k % 2;
            const bool last_k = k + 1 == nk;
            if (!egs.empty()) {   // chunk k's rows ready, chunk k + 1's on their way
                const double te = now_ms();
                eg_next.get();
                if (d.mi == 0) st.engram_ms += now_ms() - te;
                if (k + 1 < nk) eg_next = std::async(std::launch::async, read_eg, k + 1);
            }
            WinPlan wp;
            wp.par = true;
            wp.async = true;
            wp.win = run.seq[(size_t) k];
            wp.k = k;
            wp.snap_t = run.snap_pos >= p0 && run.snap_pos < p0 + T ? (int) (run.snap_pos - p0) : -1;
            bool started = false;
            std::vector<char> got(srcs.size(), 0);
            for (int il = 0; il < g.nl && run.runs(k, il); ++il) {
                if (halt_now()) return halt_end(di, err);   // (one GPU: after the layer it is on)
                Layer& L0 = L[(size_t) il];
                int mi = -1;
                for (int m = 0; m < L0.nm; ++m)
                    if (member(L0, m).dev == di) mi = m;
                if (mi < 0) continue;
                if (!started) {   // the chunk's input: its embedding rows, or the previous group's hand-off
                    started = true;
                    if (il == 0) {
                        if (k >= 2) CK(cudaEventSynchronize(c.emb_ev[slot]));
                        for (int t = 0; t < T; ++t)
                            std::memcpy(c.emb_host[slot] + (size_t) t * D, emb_row(p0 + t, toks[(size_t) (p0 + t)]),
                                        (size_t) D * 2);
                        CK(cudaMemcpyAsync(c.emb, c.emb_host[slot], (size_t) T * D * 2, cudaMemcpyHostToDevice, s));
                        CK(cudaEventRecord(c.emb_ev[slot], s));
                        embed(c.emb, T, c.h, D, H, s);
                        if (ds4_dbg("DS4_DBG_EMB")) {
                            ds4_metal_flush();
                            float *hp = c.h;
                            float hh[4] = {0,0,0,0};
                            cudaMemcpy(hh, hp, sizeof(hh), cudaMemcpyDeviceToHost);
                            std::fprintf(stderr, "DBG at embed site %d: h[0..3] = %g %g %g %g\n", __LINE__, hh[0], hh[1], hh[2], hh[3]);
                        }
                        init_pre(c.pre, T, s);
                    } else {
                        Chunk& a = *chunks[(size_t) L[(size_t) il - 1].dev];
                        host_wait_at_least(a.hand_flag[slot], wp.win);
                        xfer_get(chunk_xfer(a.hand[slot], c, T, false), a.hand_flag[slot], wp.win, s);
                        set_flag(a.hand_ack + ((size_t) slot * 8 + mi) * 32, wp.win, s);
                    }
                }
                for (int sidx : {L0.src, L0.ix_src ? L0.ik_src : -1}) {
                    if (sidx < 0 || got[(size_t) sidx]) continue;
                    got[(size_t) sidx] = 1;
                    Src& sr = srcs[(size_t) sidx];
                    if (L[(size_t) sr.layer].grp == L0.grp) continue;
                    const int64_t g_first = p0 / sr.ratio;
                    const int ng = (int) std::max<int64_t>(0, (p0 + T) / sr.ratio - g_first);
                    Xfer x;
                    if (ng > 0 && sr.ckv[(size_t) di] != nullptr) {
                        x.src[x.n] = sr.stage(slot);
                        x.dst[x.n] = sr.ckv[(size_t) di] + (size_t) g_first * ckv_rb();
                        x.bytes[x.n++] = (size_t) ng * ckv_rb();
                    }
                    if (ng > 0 && sr.ik[(size_t) di] != nullptr) {
                        x.src[x.n] = sr.stage(slot) + x_cap() * ckv_rb();
                        x.dst[x.n] = sr.ik[(size_t) di] + (size_t) g_first * ik_rb();
                        x.bytes[x.n++] = (size_t) ng * ik_rb();
                    }
                    if (x.n > 0) {
                        host_wait_at_least(sr.flag(slot), wp.win);
                        xfer_get(x, sr.flag(slot), wp.win, s);
                    }
                    set_flag(sr.x_ack + ((size_t) slot * devs.size() + di) * 32, wp.win, s);   // (taken, or none)
                }
                if (last_k && mi == 0 && g.n_mtp > 0) {   // the drafter's taps: the last chunk's last rows
                    const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
                    if (it != g.d_targets.end()) {
                        const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
                        const int nr = std::min(T, tap_cap);
                        hc_mean(c.h + (size_t) (T - nr) * H * D, nr, H, D, d.tap, D, s);
                        CK(cudaStreamWaitEvent(s, ev_seed, 0));
                        CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4,
                                             (size_t) nr, cudaMemcpyDeviceToHost, s));
                    }
                }
                if (!run_layer(L0, mi, p0, T, wp, S, err, run, k, il)) return false;
                if (dump_l != nullptr && mi == 0) dump_layer(d, c.h, il, p0, T);
                const bool last = il + 1 == g.nl || L[(size_t) il + 1].grp != L0.grp;
                if (last && mi == 0 && il + 1 < g.nl) {   // the hand-off, once the next group took the chunk before last
                    const int nnext = L[(size_t) il + 1].nm;
                    if (k >= 2) {
                        for (int m = 0; m < nnext; ++m)
                            host_wait_at_least(c.hand_ack + ((size_t) slot * 8 + m) * 32, wp.win - 2);
                        wait_flags(c.hand_ack + (size_t) slot * 8 * 32, nnext, 32, wp.win - 2, s);
                    }
                    xfer_put(chunk_xfer(c.hand[slot], c, T, true), c.hand_flag[slot], wp.win, d.xcnt, s);
                }
            }
            if (di == head_dev && (size_t) k < chunk_done.size()) {
                CK(cudaEventRecord(chunk_done[(size_t) k], s));
                chunk_rec[(size_t) k].store(1, std::memory_order_release);
            }
            if (d.grp == devs[(size_t) head_dev].grp) {   // the head group: each member its share of the logits
                if (logits_all) {   // every row's logits (tests): the head over the chunk's rows, 16 at a time
                    const size_t base = (size_t) (p0 - run.p0[0]) * g.vocab;
                    for (int r0 = 0; r0 < T; r0 += kWin) {
                        const int nr = std::min(kWin, T - r0);
                        hc_pre_norm(c.h + (size_t) r0 * H * D, c.pre + (size_t) r0 * 8, d.norm_w, D, H, nr, g.eps, d.x, s);
                        if (!head_share(d, d.mi, d.x, nr, err)) return false;
                        if (di != head_dev) continue;
                        CK(cudaMemcpyAsync(chunk_logits.data() + base + (size_t) r0 * g.vocab, d.logits,
                                           (size_t) nr * g.vocab * 4, cudaMemcpyDeviceToHost, s));
                        CK(cudaStreamSynchronize(s));
                    }
                } else if (last_k && n_logits > 0) {   // (the last rows' logits, into the window path's buffers)
                    const int t0 = T - n_logits;
                    hc_pre_norm(c.h + (size_t) t0 * H * D, c.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, s);
                    if (!head_share(d, d.mi, d.x, n_logits, err)) return false;
                }
            }
        }
        return true;
    }

    // ---- a prompt's path: the flow or windows of 16
    // this PC's speeds for pick_path (every device on the flow, the experts in place): the DMA from pinned staging to a
    // GPU; the readers copying RAM into staging; the disk read by them (experts not in RAM); the CPU lane on experts in
    // RAM and on ones it reads from the disk first - each timed once, a few experts' worth (DS4_PATH_PICK=0: always the
    // flow, as before)
    void calibrate_paths() {
        if (chunk <= 0 || chunk_min != kFlowChunkMin || std::getenv("DS4_CHUNK_MIN") != nullptr) return;
        if (const char* v = std::getenv("DS4_PATH_PICK"))
            if (std::atoi(v) == 0) return;
        int di = -1;
        for (size_t i = 0; i < chunks.size() && di < 0; ++i)
            if (chunks[i] != nullptr && chunks[i]->flow && chunks[i]->S >= 2) di = (int) i;
        if (di < 0) return;
        Chunk& c = *chunks[(size_t) di];
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        const double t0 = now_ms();
        // the DMA: staging -> the ring (once to warm up, once timed)
        const size_t nb = std::min({(size_t) 256 << 20, (size_t) c.S * c.stage_stride, (size_t) c.ring_n * c.ring_stride});
        double ms = 0;
        for (int rep = 0; rep < 2; ++rep) {
            const double a = now_ms();
            if (cudaMemcpyAsync(c.ring, c.stage, nb, cudaMemcpyHostToDevice, c.cs) != cudaSuccess ||
                cudaStreamSynchronize(c.cs) != cudaSuccess) {
                cudaGetLastError();
                return;
            }
            ms = now_ms() - a;
        }
        pm.dma = (double) nb / std::max(1e-3, ms / 1000.0);
        // the readers: RAM -> staging (a buffer of its own, touched first), in as many jobs as they are
        auto run_jobs = [&](std::vector<IoQueue::Job>& jobs, std::atomic<int>& left) {
            const double a = now_ms();
            left.store((int) jobs.size());
            c.io->push(jobs.data(), (int) jobs.size());
            while (left.load(std::memory_order_acquire) != 0) c.io->wait_any(500);
            return now_ms() - a;
        };
        {
            const size_t part = std::min(nb, (size_t) 64 << 20) / 8 / 4096 * 4096;
            std::vector<uint8_t> src(part * 8, 1);
            std::vector<IoQueue::Job> jobs(8);
            std::atomic<int> left{0};
            std::atomic<bool> bad{false};
            for (int i = 0; i < 8; ++i) {
                jobs[(size_t) i].mem = src.data() + (size_t) i * part;
                jobs[(size_t) i].dst = c.stage + (size_t) i * part;
                jobs[(size_t) i].len = part;
                jobs[(size_t) i].left = &left;
                jobs[(size_t) i].bad = &bad;
            }
            run_jobs(jobs, left);
            const double t1 = run_jobs(jobs, left), t2 = run_jobs(jobs, left);
            pm.copy = (double) (part * 8) / std::max(1e-3, std::min(t1, t2) / 1000.0);
        }
        // experts not in RAM (none of their sampled pages resident), of the trunk layer with the most CPU-lane ones:
        // up to min(8, staging slots) read by the readers as the flow reads them, then 8 others through the CPU lane
        // three times (from the disk, then in RAM: the faster of two)
        int best = -1, most = 0;
        for (int il = 0; il < g.nl; ++il) {
            int n = 0;
            for (int e = 0; e < L[(size_t) il].n_exp; ++e) n += L[(size_t) il].slot[(size_t) e] < 0;
            if (n > most) most = n, best = il;
        }
        if (best < 0) return;
        const Layer& l = L[(size_t) best];
        std::vector<int> cold;
        const int n_read = std::min(8, c.S), n_cpu = 8;
        for (int e = 0; e < l.n_exp && (int) cold.size() < n_read + n_cpu; ++e) {
            if (l.slot[(size_t) e] >= 0 || (!l.ram.empty() && l.ram[(size_t) e] >= 0)) continue;
            bool res = false;
            for (int r = 0; r < 3; ++r) res = res || os::resident_sampled(mx_mat(l, e, r), mx_part_bytes(l, r), 4);
            if (!res) cold.push_back(e);
        }
        if ((int) cold.size() >= n_read + n_cpu) {
            std::vector<IoQueue::Job> jobs;
            std::atomic<int> left{0};
            std::atomic<bool> bad{false};
            for (int i = 0; i < n_read; ++i) {
                uint8_t* sb = c.stage + (size_t) i * c.stage_stride;
                for (int r = 0; r < 3; ++r) {
                    IoQueue::Job j;
                    const os::fd_t pf = pk.pfds.empty() ? os::kBadFd : pk.pfds[(size_t) l.mx_file[r]];
                    j.fd = pf != os::kBadFd ? pf : pk.fds[(size_t) l.mx_file[r]];
                    j.dfd = pk.dfds.empty() ? os::kBadFd : pk.dfds[(size_t) l.mx_file[r]];
                    j.off = l.mx_off[r] + (uint64_t) cold[(size_t) i] * l.mx_stride[r];
                    j.len = mx_part_bytes(l, r);
                    j.dst = sb;
                    j.left = &left;
                    j.bad = &bad;
                    j.map = pk.at(l.mx_file[r], j.off);
                    jobs.push_back(j);
                    sb += flow_region(r);
                }
            }
            const double rd = run_jobs(jobs, left);
            if (!bad.load()) pm.disk = (double) n_read * (double) l.blob / std::max(1e-3, rd / 1000.0);
        }
        {   // the CPU lane: entries of one token (as most of a window's are), from the disk (if any were cold), then RAM
            const size_t first = (int) cold.size() >= n_read + n_cpu ? (size_t) n_read : 0;
            std::vector<Ent> ents;
            for (int i = 0; i < n_cpu; ++i) {
                Ent en;
                en.e = first + (size_t) i < cold.size() ? cold[first + (size_t) i] : -1;
                if (en.e < 0)   // (no cold ones: the first CPU-lane experts, wherever they are)
                    for (int e = 0, k = 0; e < l.n_exp && en.e < 0; ++e)
                        if (l.slot[(size_t) e] < 0 && k++ == i) en.e = e;
                if (en.e < 0) break;
                en.n = 1;
                en.slot0 = i;
                en.tok[0] = 0;
                ents.push_back(en);
            }
            if (ents.empty()) return;
            std::vector<float> x((size_t) g.dim);
            for (size_t i = 0; i < x.size(); ++i) x[i] = 0.02f * std::sin(0.37f * (float) i);
            const double bytes = (double) ents.size() * (double) l.blob;
            double a = now_ms();
            cpu_experts(l, x.data(), ents, (size_t) g.dim);
            const double t_first = now_ms() - a;
            double t_warm = 1e30;
            for (int rep = 0; rep < 2; ++rep) {
                a = now_ms();
                cpu_experts(l, x.data(), ents, (size_t) g.dim);
                t_warm = std::min(t_warm, now_ms() - a);
            }
            pm.cpu = bytes / std::max(1e-3, t_warm / 1000.0);
            if (first > 0 || (int) cold.size() >= n_cpu) pm.cpu_cold = bytes / std::max(1e-3, t_first / 1000.0);
        }
        pm.on = pm.dma > 0 && pm.copy > 0 && pm.cpu > 0;
        if (o.verbose) {
            char dk[32] = "not measured", cc[32] = "not measured";
            if (pm.disk > 0) std::snprintf(dk, sizeof dk, "%.2f GB/s", pm.disk / 1e9);
            if (pm.cpu_cold > 0) std::snprintf(cc, sizeof cc, "%.2f GB/s", pm.cpu_cold / 1e9);
            std::fprintf(stderr,
                         "ds4: prompts through the flow or windows by this PC's speeds: DMA %.1f GB/s, RAM to staging "
                         "%.1f GB/s, the disk %s; the CPU lane %.1f GB/s in RAM, %s from the disk (%.0f ms)\n",
                         pm.dma / 1e9, pm.copy / 1e9, dk, pm.cpu / 1e9, cc, now_ms() - t0);
        }
    }
    // layer l's CPU-lane experts (VRAM holds the rest) and the chance a token picks a given one of them: the profile's
    // share of the layer's picks they took, spread evenly over them, times pm.m (what the flow's runs found)
    double pm_rate(const Layer& l, int& n_cold) const {
        const uint32_t* pr = prof.data() + (size_t) l.il * g.n_exp;
        double sum = 0, cold = 0;
        n_cold = 0;
        for (int e = 0; e < l.n_exp; ++e) {
            sum += pr[e];
            if (l.slot[(size_t) e] < 0) cold += pr[e], ++n_cold;
        }
        if (n_cold == 0) return 0.0;
        const double share = sum > 0 && cold > 0 ? cold / sum : (double) n_cold / l.n_exp;
        return pm.m * (double) l.k_exp * share / n_cold;
    }
    // the CPU-lane experts n tokens pick in the trunk's layers, expected (each layer's n_cold (1 - e^(-rate n)))
    double pm_picks(int n) const {
        double d = 0;
        for (int il = 0; il < g.nl; ++il) {
            int nc = 0;
            const double a = pm_rate(L[(size_t) il], nc);
            d += nc * (1.0 - std::exp(-a * n));
        }
        return d;
    }
    // a run of one chunk of n tokens on the flow picked `picked` CPU-lane experts: pm.m to what makes the expectation
    // that (bisection over its log; a running mean of the log)
    void pm_learn(int n, double picked) {
        if (!pm.on || n < 2 || picked < 1) return;
        const double m0 = pm.m;
        double lo = std::log(1e-3), hi = std::log(1e3);
        for (int it = 0; it < 40; ++it) {
            pm.m = std::exp(0.5 * (lo + hi));
            (pm_picks(n) < picked ? lo : hi) = 0.5 * (lo + hi);
        }
        const double obs = 0.5 * (lo + hi);
        pm.m = std::exp(pm.n_m == 0 ? obs : 0.7 * std::log(m0) + 0.3 * obs);
        ++pm.n_m;
    }
    // a prompt of n tokens that the flow could take (n >= chunk_min): true the flow, false windows of 16 - each path's
    // time predicted from the bytes of CPU-lane experts it would move (the flow: each picked one once, n_cold (1 -
    // e^(-rate n)) a layer, pm_rate) or compute (windows: each window's own picks), those in RAM (each layer's share:
    // sampled pages of up to 32 of its CPU-lane experts, mincore / QueryWorkingSetEx; the RAM tier's) and those not at
    // this PC's speeds - the flow's times what it really took against its predictions so far for prompts of that size
    // (pm_k), windows' once some ran what a window took times the windows; the flow unless windows are predicted
    // faster (by half, with less than kPickRes in RAM); said with --verbose
    // The choice is made from the prompt's length alone (pick_flow_min): the same prompt takes the same path - and so
    // gives the same bits, the two paths round differently - in every session and on every restart.  The timing prediction below
    // (DS4_PICK=adaptive) used measured speeds and sampled page residency, so it flipped between runs: r9700, one 16-token feed
    // through the flow in one run, another feed in the next, three different logits checksums in six runs.  Measured on 4 and 1 R9700
    // (first prompt of a run, windows against the flow, ms): 28 tokens 594 / 4966 (4 cards), 1094 / 5588 (1 card); 100: 1164 / 5375,
    // 3279 / 7299; 300: 2831 / 5874, 9018 / 9803; 600: 5304 / 6808, 16969 / 11327; 1500: 13325 / 8012, 41547 / 18187 - the flow pays a
    // fixed ~5 s the first time (its buffers and GEMM libraries), then wins from about 400 tokens up.
    // The length from which the flow is taken: one number for every box (ds4_pick_rule.hpp says why 96), DS4_PICK_MIN=N overrides.
    int pick_flow_min() {
        if (pick_min_ > 0) return pick_min_;
        pick_min_ = prompt_flow_min(std::getenv("DS4_PICK_MIN"));
        std::fprintf(stderr, "ds4: prompt path: the chunked pass from %d tokens, windows of 16 below\n", pick_min_);
        return pick_min_;
    }
    int pick_min_ = 0;
    bool pick_path(int n) {
        static const bool adaptive = [] {
            const char* e = std::getenv("DS4_PICK");
            return e != nullptr && std::strcmp(e, "adaptive") == 0;
        }();
        pm.last = -1;
        if (!adaptive) return n >= pick_flow_min();
        if (!pm.on || n >= kPickMax) return true;
        double f_res = 0, f_cold = 0, w = 0;
        const int nw = n / kWin, rem = n % kWin;
        for (int il = 0; il < g.nl; ++il) {
            const Layer& l = L[(size_t) il];
            int nc = 0;
            const double a = pm_rate(l, nc);
            if (nc == 0) continue;
            int sampled = 0, res = 0, seen = 0;
            const int step = std::max(1, nc / 32);
            for (int e = 0; e < l.n_exp; ++e) {
                if (l.slot[(size_t) e] >= 0 || seen++ % step != 0) continue;
                bool r = !l.ram.empty() && l.ram[(size_t) e] >= 0;
                if (!r) {
                    r = true;
                    for (int p = 0; p < 3 && r; ++p) r = os::resident_sampled(mx_mat(l, e, p), mx_part_bytes(l, p), 2);
                }
                ++sampled;
                res += r;
            }
            const double share = sampled > 0 ? (double) res / sampled : 0.0, b = (double) l.blob;
            auto u = [&](int t) { return nc * (1.0 - std::exp(-a * t)); };
            f_res += share * u(n) * b;
            f_cold += (1.0 - share) * u(n) * b;
            w += ((double) nw * u(kWin) + (rem > 0 ? u(rem) : 0.0)) * b;
        }
        const double r_fr = std::min(pm.copy, pm.dma), r_fc = pm.disk > 0 ? std::min(pm.disk, pm.dma) : r_fr;
        const double r_wc = pm.cpu_cold > 0 ? pm.cpu_cold : 1.0 / (1.0 / r_fc + 1.0 / pm.cpu);
        const double t_flow = f_res / r_fr + f_cold / r_fc;
        const double t_win = std::max(0.0, w - f_cold) / pm.cpu + f_cold / r_wc;
        const int cls = pm_cls(n), n_win = (n + kWin - 1) / kWin;
        // (windows: once some ran, what a window took on average times the windows the prompt needs - each pays the
        // fixed costs of every layer on every GPU again, which a pass of the flow pays once)
        const double pf = pm_k(cls) * t_flow, pw = pm.win_n > 0 ? pm.win_s * n_win : t_win;
        // (the flow unless windows are predicted faster - in half its time at least, unless kPickRes of the CPU-lane
        // experts' bytes are in RAM: the flow wins on every PC whose experts come from an SSD, where a wrong pick would
        // cost the most; windows win where they are all in RAM)
        const double res_share = f_res / std::max(1.0, f_res + f_cold);
        const bool flow = !(pw < pf && (res_share >= kPickRes || 2.0 * pw <= pf));
        pm.last = flow ? 1 : 0;
        pm.last_cls = cls;
        pm.last_nw = n_win;
        pm.last_pred = flow ? t_flow : t_win;
        if (o.verbose)
            std::fprintf(stderr,
                         "ds4: a prompt of %d tokens through %s: predicted %.2f s, %s %.2f s (%.0f%% of its CPU-lane "
                         "experts' %.1f GB in RAM)\n",
                         n, flow ? "the flow" : "windows", flow ? pf : pw, flow ? "windows" : "the flow", flow ? pw : pf,
                         100.0 * res_share, (f_res + f_cold) / 1e9);
        return flow;
    }
    // a prompt's size class for pm.k: 2-31, 32-127, 128-511, 512 tokens and more
    static int pm_cls(int n) { return n < 32 ? 0 : n < 128 ? 1 : n < 512 ? 2 : 3; }
    // the flow's time factor for size class c: its own, else its nearest class's, else 1 (never the windows': theirs
    // is mostly the fixed costs a window pays again and again)
    double pm_k(int c) const {
        for (int dc = 0; dc < 4; ++dc)
            for (int cc : {c - dc, c + dc})
                if (cc >= 0 && cc < 4 && pm.kn[0][cc] > 0) return pm.k[0][cc];
        return 1.0;
    }
    // n tokens through windows of 16, the last one's logits_rows rows of logits
    bool feed_windows(const int* tokens, int n, int logits_rows, std::string& err) {
        logits_all = false;
        for (int a = 0; a < n; a += kWin) {
            const int T = std::min(kWin, n - a);
            if (!forward(tokens + a, T, a + T == n ? logits_rows : 0, err)) return false;
        }
        return true;
    }
    // the picked path took ms: the flow's predictions for the prompt's size class scaled by what it really takes (taken
    // / predicted), windows' time a window - running means
    void path_done(double ms) {
        if (pm.last < 0 || pm.last_pred <= 0) return;
        if (pm.last == 0) {
            const double w = ms / 1000.0 / std::max(1, pm.last_nw);
            pm.win_s = pm.win_n == 0 ? w : 0.7 * pm.win_s + 0.3 * w;
            ++pm.win_n;
        } else {
            const double ratio = std::min(8.0, std::max(0.25, ms / 1000.0 / pm.last_pred));
            double& k = pm.k[0][pm.last_cls];
            int64_t& n = pm.kn[0][pm.last_cls];
            k = n == 0 ? ratio : 0.7 * k + 0.3 * ratio;
            ++n;
        }
        pm.last = -1;
    }
    // a run layer-major? - one device holding every layer on the flow, several chunks, and what the chunks after the
    // first would read from the files again more than the chunks' states moved to host memory and back at every layer
    // (DS4_LAYER_MAJOR=0: never); then its host memory: the chunks' states (pinned, else pageable) and the layer cache,
    // the largest layer's experts read from the files if RAM allows (ram_avail() less kRamMargin; DS4_LAYER_CACHE_MB
    // sets it) - at least one expert, or not layer-major
    bool lm_begin(const Run& run) {
        static const char* env = std::getenv("DS4_LAYER_MAJOR");
        if (devs.size() != 1 || run.T.size() < 2 || (env != nullptr && std::atoi(env) == 0)) return false;
        Chunk& c = *chunks[0];
        if (!c.flow) return false;
        size_t need = 0, n_exp = 0, blob_max = 0;
        double file_b = 0;
        for (int il = 0; il < g.nl; ++il) {
            const Layer& l = L[(size_t) il];
            size_t ne = 0;
            for (int e : l.host_list)
                if (l.ram.empty() || l.ram[(size_t) e] < 0) ++ne;
            const size_t stride = (l.blob + 63) & ~(size_t) 63;
            need = std::max(need, ne * stride);
            n_exp = std::max(n_exp, (size_t) l.n_exp);
            blob_max = std::max(blob_max, stride);
            file_b += (double) ne * (double) l.blob;
        }
        int64_t rows = 0;
        for (int T : run.T) rows += T;
        const size_t hb = chunk_hand_bytes(), state_b = run.T.size() * hb;
        const double moved = 2.0 * (double) rows * ((double) hb / (double) chunk) * (double) g.nl;
        if (need == 0 || (double) (run.T.size() - 1) * file_b <= moved) return false;
        size_t lc_bytes = need;
        if (const char* mb = std::getenv("DS4_LAYER_CACHE_MB")) lc_bytes = std::min(need, (size_t) std::atoll(mb) << 20);
        else {
            const uint64_t avail = ram_avail();
            if (avail != ~uint64_t{0})
                lc_bytes = std::min<uint64_t>(need, avail > kRamMargin + state_b ? avail - kRamMargin - state_b : 0);
        }
        if (lc_bytes < blob_max) return false;
        void* st_p = nullptr;
        c.lm_pinned = cudaHostAlloc(&st_p, state_b, cudaHostAllocPortable) == cudaSuccess;
        if (!c.lm_pinned) {
            cudaGetLastError();
            st_p = os::anon_alloc(state_b, -1, 1);
            if (st_p == nullptr) return false;
        }
        c.lc = (uint8_t*) os::anon_alloc(lc_bytes, -1, 1);
        if (c.lc == nullptr) {
            if (c.lm_pinned) cudaFreeHost(st_p);
            else os::anon_free(st_p, state_b);
            return false;
        }
        c.lm_state = (uint8_t*) st_p;
        c.lm_state_bytes = state_b;
        c.lc_bytes = lc_bytes;
        c.lc_of.assign(n_exp, -1);
        c.lc_writer.assign(n_exp, -1);
        c.lm = true;
        lm_layers.store(0);
        return true;
    }
    void lm_end() {
        for (size_t di = 0; di < chunks.size(); ++di) {
            Chunk& c = *chunks[di];
            if (!c.lm) continue;
            cudaSetDevice(devs[di].id);
            cudaStreamSynchronize(devs[di].st);   // (nothing of the run's still reads or writes them: the device's
            cudaStreamSynchronize(c.cs);          // copies done, every read the run issued done - a failed run's too)
            for (size_t i = c.base; i < c.n_read; ++i)
                while (fit(c, i).left.load(std::memory_order_acquire) != 0) c.io->wait_any(500);
            if (c.lm_pinned) cudaFreeHost(c.lm_state);
            else os::anon_free(c.lm_state, c.lm_state_bytes);
            os::anon_free(c.lc, c.lc_bytes);
            c.lm_state = c.lc = nullptr;
            c.lm_state_bytes = c.lc_bytes = 0;
            c.lm = c.lm_pinned = false;
        }
    }

    // the first layer of a long prompt's second half (forward_chunks' two passes), or -1: the first with compress ratio 1
    // (V4.1: 20), every one after it ratio 1 too and owning no compressed KV or index keys of its own (they read its),
    // every trunk layer on one GPU.  The settings' MAYA_DS4_PROMPT_SKIP (STRATA_DS4_PROMPT_SKIP; DS4_YOCO for tests):
    // 1 (the default) the skip, 2 DeepSeek's own serving's (the last 128 rows only: faster, the answers shift a
    // little), 0 off
    static int yoco_mode() {
        static const int m = [] {
            const char* v = std::getenv("DS4_YOCO");
            if (v == nullptr) v = std::getenv("STRATA_DS4_PROMPT_SKIP");
            return v != nullptr ? std::atoi(v) : 1;
        }();
        return m;
    }
    int yoco_layer() const {
        if (yoco_mode() == 0 || chunk <= 0 || g.nl < 2 || (int) g.ratios.size() < g.nl) return -1;
        int yl = -1;
        for (int il = 0; il < g.nl && yl < 0; ++il)
            if (g.ratios[(size_t) il] == 1) yl = il;
        if (yl <= 0 || !L[(size_t) yl].kv_src) return -1;
        for (int il = 0; il < g.nl; ++il) {
            const Layer& l = L[(size_t) il];
            if (l.nm != 1 || l.dev != L[0].dev) return -1;
            if (il > yl && (g.ratios[(size_t) il] != 1 || l.kv_src)) return -1;
        }
        return yl;
    }

    // n tokens at positions pos .. in chunks of equal size (at most `chunk`), pipelined over the groups; the logits of
    // the last min(logits_rows, 16) rows (logits_rows > 16: a run of one chunk, every row's into chunk_logits)
    bool forward_chunks(const int* tk, int n, int logits_rows, std::string& err, int snap_slot = -1) {
        if (n < 1) return true;
        flow_used.store(true);
        std::lock_guard<std::mutex> warm_lk(warm_mu);   // (the warm-up, if it is running, finishes first)
        if (pos + n > o.max_context) {
            err = "the context is full";
            return false;
        }
        for (int t = 0; t < n; ++t)
            if (tk[t] < 0 || tk[t] >= g.vocab) {
                err = "token " + std::to_string(tk[t]) + " out of range";
                return false;
            }
        // a long prompt's skip: the layers from yoco_layer() on (V4.1: 20-39) read the earlier positions only through
        // that layer's compressed KV and index keys, and each looks back one window (win - 1 positions) more than the
        // layer before - so they need their own work only on the last keep = (nl - yoco) (win - 1) + win rows for the
        // run's last rows and the rings the decode reads to be what the full work gives (up to the experts' summing
        // order).  The chunks stay as they are: the ones before the cut go through the layers before yoco and its
        // kv_only, the cut's chunk through those over all its rows and through the rest from the cut on (Run::r0, a
        // view of its buffers), the chunks after it through every layer - each layer's experts read as without the
        // skip.  One GPU (DS4_YOCO=1; 0: off); not with every row's logits (tests)
        const int yl = logits_rows > kWin ? -1 : yoco_layer();
        // (DS4_YOCO_KEEP=<rows>: fewer - DeepSeek's own serving replays the last 128 tokens through those layers
        //  (DeepInfer's "SWA bounded replay"), which shifts the answers a little; at least one window)
        static const int keep_env = std::getenv("DS4_YOCO_KEEP") ? std::atoi(std::getenv("DS4_YOCO_KEEP")) : 0;
        const int keep = yl <= 0 ? 0
                         : keep_env > 0 ? std::max(keep_env, g.win)
                         : yoco_mode() == 2 ? g.win : (g.nl - yl) * (g.win - 1) + g.win;
        const int a = yl > 0 && n - keep >= kYocoMin ? n - keep : 0;
        const int nc = (n + chunk - 1) / chunk, per = (n + nc - 1) / nc;
        if (ds4_dbg("DS4_DBG_CHUNK")) std::fprintf(stderr, "forward_chunks n %d nc %d per %d\n", n, nc, per);
        Run run;
        for (int x = 0; x < n; x += per) {
            run.p0.push_back(pos + x);
            run.T.push_back(std::min(per, n - x));
            run.seq.push_back(++win_seq);
        }
        if (a > 0) {
            run.yoco = yl;
            run.cut_k = a / per;
            run.cut_r = a - run.cut_k * per;
        }
        logits_all = logits_rows > kWin;
        if (logits_all) chunk_logits.resize((size_t) n * g.vocab);
        for (int t = 0; t < n; ++t) {
            toks.push_back(tk[t]);
            ctoks.push_back(ctok(pos + t, tk[t]));
        }
        plan_streams();
        if (snap_slot >= 0) {   // feed_mark: the state before the run's last row, kept by its layers on the way
            run.snap_pos = pos + n - 1;
            for (int il = 0; il < g.nl; ++il) {
                Layer& l = L[(size_t) il];
                Dev& d = devs[(size_t) l.dev];
                if (l.snap_win == nullptr && !dalloc(d, l.snap_win, (size_t) g.win * win_rb(), err)) return false;
                if (l.c_kv != nullptr && l.snap_kv == nullptr &&
                    (!dalloc(d, l.snap_kv, (size_t) l.ratio * g.hd, err) || !dalloc(d, l.snap_sc, (size_t) l.ratio * g.hd, err)))
                    return false;
            }
        }
        const int T_last = run.T.back();
        n_logits = logits_all ? 0 : std::min(logits_rows, T_last);
        const double tl = now_ms();
        {   // each chunk's end on the head device, for the progress reports
            cudaSetDevice(devs[(size_t) head_dev].id);
            while (chunk_done.size() < run.T.size()) {
                cudaEvent_t e = nullptr;
                CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                chunk_done.push_back(e);
            }
            chunk_rec.reset(new std::atomic<int>[chunk_done.size()]);
            for (size_t i = 0; i < chunk_done.size(); ++i) chunk_rec[i].store(0);
        }
        prof_chunk = prof_on && !prof_sync_on;   // (DS4_PROF_SYNC: the decode profile only; the prompt events hang the driver)
        const bool lm = lm_begin(run);
        struct LmEnd {   // (every way out: its host memory freed once the device is done with it)
            Impl* p;
            bool on;
            ~LmEnd() {
                if (on) p->lm_end();
            }
        } lm_end_{this, lm};
        std::vector<std::string> errs(devs.size());
        halted = false;
        halt.store(false);
        {
            std::lock_guard<std::mutex> lk(halt_mu);
            halt_k = INT_MAX;
            halt_seen = -1;
        }
        bool halting = false;
        for (size_t di = 0; di < devs.size(); ++di)
            workers[di]->submit([this, di, &run, &errs] {
                std::string e;
                if (!run_dev_chunks((int) di, run, e)) errs[di] = e.empty() ? "a device's thread failed" : e;
            });
        size_t reported = 0;
        for (;;) {
            bool all = true;
            for (auto& w : workers) all = all && w->idle();
            if (!all && !halting && cancel && cancel()) {   // a Stop: the devices end the run early
                halting = true;
                std::lock_guard<std::mutex> lk(halt_mu);
                halt_k = halt_seen + 1;
                halt.store(true, std::memory_order_release);
            }
            if (progress && lm) {   // (layer-major: the share of the layers every chunk went through)
                const int nd = lm_layers.load(std::memory_order_acquire);
                if (nd > 0 && nd < g.nl) progress((int64_t) n * nd / g.nl);
            } else if (progress) {   // chunks through every layer (the last layer's device past their event)
                while (reported < run.T.size() && chunk_rec[reported].load(std::memory_order_acquire) &&
                       cudaEventQuery(chunk_done[reported]) == cudaSuccess)
                    ++reported;
                if (reported > 0 && reported < run.T.size())
                    progress(run.p0[reported - 1] + run.T[reported - 1] - run.p0[0]);
            }
            if (all) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        for (const std::string& e : errs)   // (a device's failure first: a Stop's early end is not one)
            if (!e.empty() && e != kHalted) {
                err = e;
                return false;
            }
        for (const std::string& e : errs)
            if (e == kHalted) {
                halted = true;
                err = "stopped while reading the prompt";
                return false;
            }
        st.layer_ms += now_ms() - tl;
        eg_stats();
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        const double th = now_ms();
        CK(cudaStreamSynchronize(d.st));
        st.head_ms += now_ms() - th;
        if (prof_chunk) {   // DS4_PROF: the run's GPU time by stage, every device's summed
            for (Dev& dv : devs) {
                cudaSetDevice(dv.id);
                CK(cudaStreamSynchronize(dv.st));
            }
            cudaSetDevice(d.id);
            prof_collect();
            std::fprintf(stderr,
                         "ds4 prompt GPU time, %d tokens: attention %.0f ms, routing + shared expert %.0f ms, experts %.0f ms "
                         "(with the waits for streamed ones), between layers %.0f ms\n",
                         n, cprof_ms[1], cprof_ms[2], cprof_ms[3], cprof_ms[0]);
            {   // the finer table (ms, % of the sum)
                static const char* nm_[16] = {"between layers", "attn rest (q_b, wo_a/b, hc_post)", "shared expert", "experts (incl waits)",
                                              "engram", "attn mHC (mix/split/norm)", "attn projections", "compressor + indexer",
                                              "q_b + rope", "sparse attention", "ffn mHC + router + sort", "", "", "", "", ""};
                double sum = 0;
                for (double v : cprof_ms) sum += v;
                for (int i = 0; i < 11; ++i)
                    std::fprintf(stderr, "ds4 prof  %-34s %9.1f ms %5.1f%%\n", nm_[i], cprof_ms[i], sum > 0 ? 100.0 * cprof_ms[i] / sum : 0.0);
                struct Agg {
                    double ms = 0, flops = 0;
                    long calls = 0, rows = 0, experts = 0;
                };
                std::map<std::pair<int, int>, Agg> agg;   // (type, role 0 gate+up, 1 down)
                {
                    std::lock_guard<std::mutex> lk(prof_mu);
                    for (const EkMark& m : ek_marks) {
                        float a = 0, b = 0;
                        cudaSetDevice(m.dev);
                        cudaEventElapsedTime(&a, m.e0, m.e1);
                        cudaEventElapsedTime(&b, m.e1, m.e2);
                        Agg& x = agg[{m.tg, 0}];
                        x.ms += a;
                        x.flops += 2.0 * 2.0 * m.nrows * (double) g.dim * (double) m.ff;
                        x.calls++, x.rows += m.nrows, x.experts += m.nexp;
                        Agg& y = agg[{m.td, 1}];
                        y.ms += b;
                        y.flops += 2.0 * m.nrows * (double) g.dim * (double) m.ff;
                        y.calls++, y.rows += m.nrows, y.experts += m.nexp;
                        ek_pool[m.dev].push_back(m.e0);
                        ek_pool[m.dev].push_back(m.e1);
                        ek_pool[m.dev].push_back(m.e2);
                    }
                    ek_marks.clear();
                }
                double tot = 0;
                for (auto& kv : agg) tot += kv.second.ms;
                for (auto& kv : agg)
                    std::fprintf(stderr,
                                 "ds4 prof  expert kernels type %-8s %-7s %9.1f ms  calls %5ld  experts %6ld (%.1f us each, %.1f a call)  rows/expert %6.1f  ~%5.1f TFLOPS\n",
                                 gg::iq::name(kv.first.first), kv.first.second ? "down" : "gate+up", kv.second.ms, kv.second.calls,
                                 kv.second.experts, kv.second.experts > 0 ? 1e3 * kv.second.ms / (double) kv.second.experts : 0.0,
                                 kv.second.calls > 0 ? (double) kv.second.experts / (double) kv.second.calls : 0.0,
                                 kv.second.experts > 0 ? (double) kv.second.rows / (double) kv.second.experts : 0.0,
                                 kv.second.ms > 0 ? kv.second.flops / kv.second.ms / 1e9 : 0.0);
                std::fprintf(stderr, "ds4 prof  expert kernels total %.1f ms\n", tot);
            }
            for (double& v : cprof_ms) v = 0;
            prof_chunk = false;
        }
        if (o.verbose || std::getenv("DS4_FLOW_STATS") != nullptr) {
            const double ms = now_ms() - tl;
            for (size_t di = 0; di < chunks.size(); ++di) {
                const Chunk& c = *chunks[di];
                if (!c.flow) continue;
                const double db = (double) (c.io->direct_bytes() - c.io_d0) / 1e9;
                const double bb = (double) (c.io->buffered_bytes() - c.io_b0) / 1e9;
                const double fb = db + bb, rb = (double) c.it_ram * (double) L[0].blob / 1e9;   // (a member: its parts)
                char lcs[160] = "";
                if (c.lm)
                    std::snprintf(lcs, sizeof lcs, ", %llu again from the layer cache (layer-major, %d chunks, %.1f GB "
                                  "of RAM)", (unsigned long long) c.it_lc, (int) run.T.size(), (double) c.lc_bytes / 1e9);
                char dbgs[200] = "";
                if (pm.on)
                    std::snprintf(dbgs, sizeof dbgs,
                                  "; io threads busy %.0f ms, copy_ahead staging %.0f ms, DMA issues %.0f ms, slot waits %.0f ms",
                                  c.io->busy_ms() - c.io_busy0, c.dbg_stage_ms, c.dbg_dma_ms, c.dbg_evt_ms);
                std::fprintf(stderr,
                             "ds4 prompt stream, CUDA%d: %d tokens in %.0f ms; %llu experts streamed: %llu from the files "
                             "(%.1f GB, %.2f GB/s: %.1f GB from the disk unbuffered, %.1f GB through the page cache), %llu "
                             "from RAM (%.1f GB)%s; %llu likely ones queued early, %llu of them unpicked; the compute waited "
                             "%.0f ms for reads%s\n",
                             devs[di].id, n, ms, (unsigned long long) (c.it_file + c.it_ram + c.it_lc),
                             (unsigned long long) c.it_file, fb, fb / std::max(1e-3, ms / 1000.0), db, bb,
                             (unsigned long long) c.it_ram, rb, lcs, (unsigned long long) c.it_spec,
                             (unsigned long long) c.it_spec_unused, c.wait_ms, dbgs);
            }
        }
        if (pm.on && run.T.size() == 1) {   // (the picks' spread for pick_path: the CPU-lane experts this run picked)
            bool tp = false;
            for (const Layer& l : L) tp = tp || l.nm > 1;
            double picked = 0;
            for (const auto& cp : chunks)
                if (cp != nullptr && cp->flow) picked += (double) (cp->it_file + cp->it_ram) - (double) cp->it_spec_unused;
            if (!tp) pm_learn(n, picked);
        }
        if (logits_all) n_logits = (T_last - 1) % kWin + 1;   // (sample() takes the last row of d.logits)
        pos += n;
        st.tokens += n;
        cache_tick(n);
        if (lm_end_.on) {   // (the layer cache's RAM back before the store and the swaps want it)
            lm_end();
            lm_end_.on = false;
        }
        cache_settle();
        const int nr = std::min(T_last, tap_cap);
        if (!seed_drafter(pos - nr, nr, 0, err)) return false;
        if (snap_slot >= 0) {   // checkpoint `snap_slot` from what the layers kept (the drafter's blocks: as they are -
            // the next feed seeds them again from its own rows, past this position)
            Check& ckp = ck[snap_slot & 1];
            ckp.pos = run.snap_pos;
            ckp.win.resize(L.size());
            ckp.comp.resize(L.size() * 2);
            for (size_t i = 0; i < L.size(); ++i) {
                Layer& l = L[i];
                Dev& dv = devs[(size_t) l.dev];
                cudaSetDevice(dv.id);
                const bool kept = !l.mtp && l.snap_win != nullptr;
                ckp.win[i].resize((size_t) g.win * win_rb());
                CK(cudaMemcpyAsync(ckp.win[i].data(), kept ? l.snap_win : l.win, ckp.win[i].size(),
                                   cudaMemcpyDeviceToHost, dv.st));
                if (l.c_kv) {
                    ckp.comp[2 * i].resize((size_t) l.ratio * g.hd);
                    ckp.comp[2 * i + 1].resize((size_t) l.ratio * g.hd);
                    CK(cudaMemcpyAsync(ckp.comp[2 * i].data(), kept && l.snap_kv ? l.snap_kv : l.c_kv,
                                       ckp.comp[2 * i].size() * 4, cudaMemcpyDeviceToHost, dv.st));
                    CK(cudaMemcpyAsync(ckp.comp[2 * i + 1].data(), kept && l.snap_sc ? l.snap_sc : l.c_score,
                                       ckp.comp[2 * i + 1].size() * 4, cudaMemcpyDeviceToHost, dv.st));
                }
            }
            for (Dev& dv : devs) {
                cudaSetDevice(dv.id);
                CK(cudaStreamSynchronize(dv.st));
            }
        }
        return true;
    }

    // the window's plan, in window order (the indexer's and the candidate source's sizes carry from window to window)
    void plan_window(int64_t p0, int T, WinPlan& wp) {
        wp.par = par;
        wp.async = par || (!sync_decode && T <= async_max_t);
        wp.vl = 0;
        for (int t = 0; t < T && !imgs.empty(); ++t)
            if (in_image(p0 + t)) wp.vl |= 1u << t;
        for (int il = 0; il < g.nl; ++il) {
            const Layer& l = L[(size_t) il];
            if (l.ratio > 0) {
                const int R = l.ratio;
                int n[kWin], k[kWin], clen_max = 0;
                for (int t = 0; t < T; ++t) {
                    n[t] = (int) ((p0 + t + 1) / R);
                    k[t] = std::min(g.ix_topk, n[t]);
                    clen_max = std::max(clen_max, n[t]);
                }
                if (l.ix_src) {
                    for (int t = 0; t < T; ++t) wp.cand[il][t] = cur_cand_n[t];
                    if (clen_max > 0 && l.cand_src)
                        for (int t = 0; t < T; ++t) cur_cand_n[t] = std::min(g.cand_topk, (n[t] + g.cand_blk - 1) / g.cand_blk);
                    for (int t = 0; t < T; ++t) cur_topk_n[t] = k[t];
                }
                for (int t = 0; t < T; ++t) wp.topk[il][t] = cur_topk_n[t];
            }
            if (wp.async) {
                wp.seq[il] = ++seq_issued;
                layer_of[wp.seq[il] % kRing].store(il, std::memory_order_release);
            }
        }
    }

    int head_nm() const { return (int) groups[(size_t) devs[(size_t) head_dev].grp].mem.size(); }
    // the head's columns on member mi of its group of nm: an EXL3 head's split128 of the vocabulary (mat's), a GGUF's
    // BF16 / Q8_0 head's even shares (mat_bf16's) - head_share's, its gather's and the markov steps' rows
    void head_cols(int nm, int mi, int& v0, int& vn) const {
        const Tensor* tw = pk.get("head.weight");
        if (tw != nullptr && (tw->dtype == "BF16" || tw->dtype == "Q8_0")) {
            vn = g.vocab / nm;
            v0 = mi * vn;
        } else
            split128(g.vocab, nm, mi, v0, vn);
    }
    // the head over `rows` (<= 16) normed rows xn on member mi of the head group (device d): its share of each row's
    // logits (columns [v0, v0 + vn), head_cols); the lead gathers the others' into d.logits (rows V apart) - two
    // slots by seq (Vmax apart, the widest share's), a member reusing one once the lead acknowledged it.  gather
    // false: each member keeps its share (the drafter's markov steps over the group, mk_split)
    bool head_share(Dev& d, int mi, const float* xn, int rows, std::string& err, bool gather = true) {
        const int V = g.vocab, D = g.dim, nm = head_nm();
        int v0 = 0, vn = 0, Vmax = 0, z = 0;
        head_cols(nm, mi, v0, vn);
        head_cols(nm, 0, z, Vmax);
        Group& G = groups[(size_t) d.grp];
        float* out = mi == 0 ? d.logits : d.logits_part;
        const size_t ld = mi == 0 ? (size_t) V : (size_t) vn;
        cudaSetDevice(d.id);
        // (an EXL3 head: as many activations at a time as its split-K partials fit the scratch - 8 for a shared head, 4
        // for a whole one, each pass reading the whole share; a GGUF's: 8, so that every window the engine can build (rows
        // <= kWin = 8) walks the head once instead of twice - measured on the drafter's fixed T=5: 4.55 -> 2.83 ms, its
        // round 14.49 -> 13.46; the verify window's head 3.48 -> 2.68 ms, the same logits.  DS4_HEAD_PER overrides.)
        const int per = d.head.fmt != 0 ? (std::getenv("DS4_HEAD_PER") ? std::atoi(std::getenv("DS4_HEAD_PER")) :
#if defined(STRATA_METAL)
                                                                                   4   // (8 columns a pass on Metal moved the KL by +0.002: the mvn8 kernel sums in another order)
#else
                                                                                   strata::ds4::head_per_default(defaults_026())
#endif
                                                                                   )
                                        : std::max(1, x3::mv_cols(d.head));
        for (int c = 0; c < rows; c += per)
            if (!mv(d, {job(d.head, xn + (size_t) c * D, out + (size_t) c * ld, std::min(per, rows - c), D, ld)}, err))
                return false;
        if (nm == 1 || !gather) return true;
        const unsigned seq = ++d.head_seq;
        const int slot = (int) (seq % 2);
        auto stage = [&](int m) { return G.lg_stage + ((size_t) slot * (nm - 1) + (m - 1)) * kWin * Vmax; };
        auto flag = [&](int m) { return G.lg_flag + ((size_t) slot * (nm - 1) + (m - 1)) * 32; };
        if (mi > 0) {
            if (seq > 2) wait_flags(G.lg_ack + (size_t) slot * 32, 1, 32, seq - 2, d.st);
            Xfer x;
            x.n = 1;
            x.src[0] = d.logits_part;
            x.dst[0] = stage(mi);
            x.bytes[0] = (size_t) rows * vn * 4;
            xfer_put(x, flag(mi), seq, d.xcnt, d.st);
            return true;
        }
        for (int m = 1; m < nm; ++m) {
            int m0 = 0, mn = 0;
            head_cols(nm, m, m0, mn);
            Xfer x;
            x.n = 1;
            x.src[0] = stage(m);
            x.dst[0] = d.logits_part;
            x.bytes[0] = (size_t) rows * mn * 4;
            xfer_get(x, flag(m), seq, d.st);
            CK(cudaMemcpy2DAsync(d.logits + (size_t) m0, (size_t) V * 4, d.logits_part, (size_t) mn * 4,
                                 (size_t) mn * 4, (size_t) rows, cudaMemcpyDeviceToDevice, d.st));
        }
        set_flag(G.lg_ack + (size_t) slot * 32, seq, d.st);
        return true;
    }

    // the head's rows (the last n_logits of the window) on head group member di (its h / pre: the group's)
    bool head_rows_dev(int di, int T, std::string& err) {
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        const int D = g.dim, H = g.hc;
        if (n_logits <= 0) return true;
        const int t0 = T - n_logits;
        hc_pre_norm(d.h + (size_t) t0 * H * D, d.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, d.st);
        return head_share(d, d.mi, d.x, n_logits, err);
    }

    // the DSpark drafter's taps: the streams' mean entering target layer il, on device d (after the last seeding read
    // h_taps)
    bool tap(Dev& d, int il, int T, std::string& err) {
        const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
        if (g.n_mtp == 0 || it == g.d_targets.end()) return true;
        const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
        const int D = g.dim;
        cudaSetDevice(d.id);
        hc_mean(d.h, T, g.hc, D, d.tap, D, d.st);
        CK(cudaStreamWaitEvent(d.st, ev_seed, 0));
        CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4, (size_t) T,
                             cudaMemcpyDeviceToHost, d.st));
        return true;
    }

    bool forward(const int* tk, int T, int logits_rows, std::string& err, int seed_rows = -1) {
#if defined(STRATA_METAL)
        struct BusyAll {   // DS4_BUSY_ALL: the GPU is held busy for the whole step (the host gaps between its bursts too)
            bool on;
            BusyAll() : on(std::getenv("DS4_BUSY_ALL") != nullptr) { if (on) ds4_busy_begin(); }
            ~BusyAll() { if (on) ds4_busy_end(); }
        } busy_all;
#endif
        if (T < 1 || T > kWin) {
            err = "a window of " + std::to_string(T) + " tokens";
            return false;
        }
        if (pos + T > o.max_context) {
            err = "the context is full";
            return false;
        }
        for (int t = 0; t < T; ++t)
            if (tk[t] < 0 || tk[t] >= g.vocab) {
                err = "token " + std::to_string(tk[t]) + " out of range";
                return false;
            }
        const int64_t p0 = pos;
        slot_held = false;   // (the state grows now: what slot_save took is no longer the conversation)
        for (int t = 0; t < T; ++t) {
            toks.push_back(tk[t]);
            ctoks.push_back(ctok(pos + t, tk[t]));
        }
        // the Engram rows asked for now (the first Engram layer's first), collected right before their layer: what is
        // left of their reads by then is the window's Engram time
        const double te = now_ms();
        const int64_t eg_w0 = eg_wait_ns.load();
        for (int li = 0; li < pk.eg_layers; ++li) {
            EgBatch& b = *eg_win[(size_t) li];
            if (b.live) eg_finish(b);   // (a window that failed midway: its reads end before h_eg is reused)
            eg_issue(p0, T, li, nullptr, b);
        }
        st.engram_ms += now_ms() - te;
        const int D = g.dim;
        for (int t = 0; t < T; ++t)   // (h_emb is written again by the next window, after this one's last sync)
            std::memcpy(h_emb + (size_t) t * D, emb_row(p0 + t, tk[t]), (size_t) D * 2);
        plan_window(p0, T, wplan);
        prof_window = prof_on && T == 1;
        n_logits = std::min(logits_rows, T);
        const double tl = now_ms();
        if (par) {
            if (!forward_par(p0, T, err)) return false;
        } else if (!forward_serial(p0, T, err)) return false;
        st.layer_ms += now_ms() - tl;
        for (auto& b : eg_win)
            if (b->live) eg_finish(*b);
        st.engram_ms += (double) (eg_wait_ns.load() - eg_w0) / 1e6;
        eg_stats();
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        const double th = now_ms();
        CK(cudaStreamSynchronize(d.st));
        st.head_ms += now_ms() - th;
        prof_collect();
        pos += T;
        st.tokens += T;
        cache_tick(T);
        if (seed_rows != 0 && !seed_drafter(p0, seed_rows < 0 ? T : seed_rows, 0, err)) return false;
        return true;
    }

    // one thread: the layers device after device, the hand-offs through pinned memory and events
    void dbg_nf(const char* tag, int il, const float* dev, size_t n) {   // DBG: non-finite count of a device buffer
        ds4_metal_flush();
        const float* hp = (const float*) strata::gpu::host_view(dev);
        size_t nf = 0, first = 0; float mx = 0.0f;
        for (size_t i = 0; i < n; ++i) { const float v = hp[i]; if (!(v == v)) { if (!nf) first = i; ++nf; } else if (std::fabs(v) > mx) mx = std::fabs(v); }
        std::fprintf(stderr, "DBG nf il=%d %s n=%zu nonfinite=%zu first=%zu max=%.4g\n", il, tag, n, nf, first, mx);
    }
    void dbg_h(const char* tag, Dev& d, int T) {   // DBG: max |h| of the window's residual stream, after the queue drains
        ds4_metal_flush();
        const float* hp = (const float*) strata::gpu::host_view(d.h);
        float mx = 0.0f; size_t nf = 0;
        const size_t n = (size_t) T * g.hc * g.dim;
        for (size_t i = 0; i < n; ++i) { const float v = hp[i]; if (!(v == v)) ++nf; else if (std::fabs(v) > mx) mx = std::fabs(v); }
        std::fprintf(stderr, "DBG hmax %s T=%d max=%.4g nonfinite=%zu h0=%g\n", tag, T, mx, nf, hp[0]);
    }
    bool forward_serial(int64_t p0, int T, std::string& err) {
        const int D = g.dim, H = g.hc;
        int cur = L[0].dev;
        if (ds4_dbg("DS4_SYNC_WIN")) cudaDeviceSynchronize();
        {
            Dev& d = devs[(size_t) cur];
            cudaSetDevice(d.id);
            CK(emb_upload(d.emb_rows, h_emb, (size_t) T * D * 2, d.st));
            embed(d.emb_rows, T, d.h, D, H, d.st);
            if (ds4_dbg("DS4_DBG_EMB")) {
                ds4_metal_flush();
                float *hp = d.h;
                float hh[4] = {0,0,0,0};
                cudaMemcpy(hh, hp, sizeof(hh), cudaMemcpyDeviceToHost);
                std::fprintf(stderr, "DBG at embed site %d: h[0..3] = %g %g %g %g\n", __LINE__, hh[0], hh[1], hh[2], hh[3]);
            }
            init_pre(d.pre, T, d.st);
            if (ds4_dbg("DS4_DBG_EMB")) dbg_h("after init_pre", d, T);
        }
        std::vector<float> dump;
        if (dump_h != nullptr) dump.resize((size_t) T * g.nl * H * D);
        for (int il = 0; il < g.nl; ++il) {
            Layer& l = L[(size_t) il];
            if (l.dev != cur) {
                Dev& a = devs[(size_t) cur];
                Dev& b = devs[(size_t) l.dev];
                cudaSetDevice(a.id);
                const size_t nh = (size_t) T * H * D * 4, np = (size_t) T * 8 * 4, nk = (size_t) T * topk_stride() * 4,
                             nc = (size_t) T * cand_stride() * 4;
                uint8_t* bb = a.bounce;
                CK(cudaMemcpyAsync(bb, a.h, nh, cudaMemcpyDeviceToHost, a.st));
                CK(cudaMemcpyAsync(bb + nh, a.pre, np, cudaMemcpyDeviceToHost, a.st));
                CK(cudaMemcpyAsync(bb + nh + np, a.topk, nk, cudaMemcpyDeviceToHost, a.st));
                CK(cudaMemcpyAsync(bb + nh + np + nk, a.cand, nc, cudaMemcpyDeviceToHost, a.st));
                CK(cudaEventRecord(a.ev_out, a.st));
                cudaSetDevice(b.id);
                CK(cudaStreamWaitEvent(b.st, a.ev_out, 0));
                CK(cudaMemcpyAsync(b.h, bb, nh, cudaMemcpyHostToDevice, b.st));
                CK(cudaMemcpyAsync(b.pre, bb + nh, np, cudaMemcpyHostToDevice, b.st));
                CK(cudaMemcpyAsync(b.topk, bb + nh + np, nk, cudaMemcpyHostToDevice, b.st));
                CK(cudaMemcpyAsync(b.cand, bb + nh + np + nk, nc, cudaMemcpyHostToDevice, b.st));
                cur = l.dev;
            }
            if (!tap(devs[(size_t) l.dev], il, T, err)) return false;
            if (il == 0 && ds4_dbg("DS4_DBG_EMB")) dbg_h("after tap", devs[(size_t) l.dev], T);
            if (!layer(l, 0, p0, T, wplan, err)) return false;
            if (dump_h != nullptr) {   // DS4_DUMP_H: every layer's residual stream, for tools/ds4/compare.py --layers
                Dev& dd = devs[(size_t) l.dev];
                std::vector<float> hv((size_t) T * H * D);
                cudaSetDevice(dd.id);
                CK(cudaMemcpyAsync(hv.data(), dd.h, hv.size() * 4, cudaMemcpyDeviceToHost, dd.st));
                CK(cudaStreamSynchronize(dd.st));
                for (int t = 0; t < T; ++t)
                    std::memcpy(dump.data() + ((size_t) t * g.nl + l.il) * H * D, hv.data() + (size_t) t * H * D,
                                (size_t) H * D * 4);
            }
        }
        if (dump_h != nullptr) std::fwrite(dump.data(), 4, dump.size(), dump_h);
        if (cur != head_dev) {
            err = "the head is not on the last layer's GPU";
            return false;
        }
        return head_rows_dev(head_dev, T, err);
    }

    // threads: each device's thread enqueues its layers (a group's members each their share) at once, the
    // devices waiting on each other through host-mapped flags (the window's seq)
    bool forward_par(int64_t p0, int T, std::string& err) {
        wplan.win = ++win_seq;
        std::vector<std::string> errs(devs.size());
        for (size_t di = 0; di < devs.size(); ++di)
            workers[di]->submit([this, di, p0, T, &errs] {
                std::string e;
                if (!run_dev((int) di, p0, T, wplan, e)) errs[di] = e.empty() ? "a device's thread failed" : e;
            });
        for (auto& w : workers) w->wait();
        for (const std::string& e : errs)
            if (!e.empty()) {
                err = e;
                return false;
            }
        return true;
    }

    // the hand-off between groups: h | pre | topk | cand of device d <-> the host-mapped buffer
    Xfer hand_xfer(uint8_t* host, Dev& d, int T, bool put) const {
        const size_t nh = (size_t) T * g.hc * g.dim * 4, np = (size_t) T * 8 * 4, nk = (size_t) T * topk_stride() * 4,
                     nc = (size_t) T * cand_stride() * 4;
        void* dv[4] = {d.h, d.pre, d.topk, d.cand};
        uint8_t* hv[4] = {host, host + nh, host + nh + np, host + nh + np + nk};
        const size_t nb[4] = {nh, np, nk, nc};
        Xfer x;
        x.n = 4;
        for (int i = 0; i < 4; ++i) {
            x.src[i] = put ? (const void*) dv[i] : (const void*) hv[i];
            x.dst[i] = put ? (void*) hv[i] : dv[i];
            x.bytes[i] = nb[i];
        }
        return x;
    }

    bool run_dev(int di, int64_t p0, int T, const WinPlan& wp, std::string& err) {
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int D = g.dim, H = g.hc;
        bool started = false;
        std::vector<char> got(srcs.size(), 0);
        for (int il = 0; il < g.nl; ++il) {
            Layer& L0 = L[(size_t) il];
            int mi = -1;
            for (int m = 0; m < L0.nm; ++m)
                if (member(L0, m).dev == di) mi = m;
            if (mi < 0) continue;
            if (!started) {   // the window's input: the embedding rows, or the previous group's hand-off
                started = true;
                if (il == 0) {
                    CK(emb_upload(d.emb_rows, h_emb, (size_t) T * D * 2, s));
                    embed(d.emb_rows, T, d.h, D, H, s);
                    if (ds4_dbg("DS4_DBG_EMB")) {
                        ds4_metal_flush();
                        float *hp = d.h;
                        float hh[4] = {0,0,0,0};
                        cudaMemcpy(hh, hp, sizeof(hh), cudaMemcpyDeviceToHost);
                        std::fprintf(stderr, "DBG at embed site %d: h[0..3] = %g %g %g %g\n", __LINE__, hh[0], hh[1], hh[2], hh[3]);
                    }
                    init_pre(d.pre, T, s);
                } else {
                    const Dev& a = devs[(size_t) L[(size_t) il - 1].dev];
                    xfer_get(hand_xfer(a.hand, d, T, false), a.hand_flag, wp.win, s);
                }
            }
            // the compressed KV / index keys another group's source layer wrote this window
            for (int sidx : {L0.src, L0.ix_src ? L0.ik_src : -1}) {
                if (sidx < 0 || got[(size_t) sidx]) continue;
                got[(size_t) sidx] = 1;
                Src& sr = srcs[(size_t) sidx];
                if (L[(size_t) sr.layer].grp == L0.grp) continue;
                const int64_t g_first = p0 / sr.ratio;
                const int ng = (int) std::max<int64_t>(0, (p0 + T) / sr.ratio - g_first);
                if (ng == 0) continue;
                Xfer x;
                if (sr.ckv[(size_t) di] != nullptr) {
                    x.src[x.n] = sr.x_stage;
                    x.dst[x.n] = sr.ckv[(size_t) di] + (size_t) g_first * ckv_rb();
                    x.bytes[x.n++] = (size_t) ng * ckv_rb();
                }
                if (sr.ik[(size_t) di] != nullptr) {
                    x.src[x.n] = sr.x_stage + x_cap() * ckv_rb();
                    x.dst[x.n] = sr.ik[(size_t) di] + (size_t) g_first * ik_rb();
                    x.bytes[x.n++] = (size_t) ng * ik_rb();
                }
                if (x.n > 0) xfer_get(x, sr.x_flag, wp.win, s);
            }
            if (mi == 0 && !tap(d, il, T, err)) return false;
            if (!layer(L0, mi, p0, T, wp, err)) return false;
            if (dump_l != nullptr && mi == 0) dump_layer(d, d.h, il, p0, T);
            const bool last = il + 1 == g.nl || L[(size_t) il + 1].grp != L0.grp;
            if (last && mi == 0 && il + 1 < g.nl) xfer_put(hand_xfer(d.hand, d, T, true), d.hand_flag, wp.win, d.xcnt, s);
        }
        if (d.grp == devs[(size_t) head_dev].grp) return head_rows_dev(di, T, err);
        return true;
    }

    // a source's new cache rows (src on device `from`, bytes of them) to the same place of its other devices' copies,
    // through pinned memory: the copy out on the source's stream, each copy in on its device's stream after it
    bool replicate(std::vector<uint8_t*>& copies, uint8_t* stage, cudaEvent_t ev, const uint8_t* src, size_t at,
                   size_t bytes, int from, std::string& err) {
        bool any = false;
        for (size_t od = 0; od < devs.size(); ++od) any = any || (copies[od] != nullptr && (int) od != from);
        if (!any) return true;
        Dev& d = devs[(size_t) from];
        CK(cudaMemcpyAsync(stage, src, bytes, cudaMemcpyDeviceToHost, d.st));
        CK(cudaEventRecord(ev, d.st));
        for (size_t od = 0; od < devs.size(); ++od) {
            if (copies[od] == nullptr || (int) od == from) continue;
            Dev& o2 = devs[od];
            cudaSetDevice(o2.id);
            CK(cudaStreamWaitEvent(o2.st, ev, 0));
            CK(cudaMemcpyAsync(copies[od] + at, stage, bytes, cudaMemcpyHostToDevice, o2.st));
        }
        cudaSetDevice(d.id);
        return true;
    }

    // layer L0 for the window, as group member mi (its share of the heads, of the output projection and of the FFNs;
    // the partial sums all-reduced across the group)
    bool layer(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, std::string& err) {
        Layer& l = member(L0, mi);
        Dev& d = devs[(size_t) l.dev];
        cudaSetDevice(d.id);
        cudaStream_t s = d.st;
        const int nm = L0.nm;
        const int D = g.dim, H = g.hc, HD = g.hd, NHL = l.ahn, HQ = NHL * HD, OG = l.ogn, FF = l.ffn;
        const ArGroup& ag = groups[(size_t) L0.grp].ar;
        const float* freqs = l.ratio ? d.freqs_comp : d.freqs_win;
        if (ds4_dbg("DS4_DBG_SLOTS") && l.il < 2 && l.lay.fmt == gg::kFmtMXFP4) {   // DBG: a resident expert's bytes against its file's
            ds4_metal_flush();
            int shown = 0;
            for (int e = 0; e < l.n_exp && shown < 3; ++e) {
                if (l.slot[(size_t) e] < 0) continue;
                std::vector<uint8_t> want(l.blob, 0);
                read_mx_file(l, e, want.data());
                const uint8_t* have = (const uint8_t*) strata::gpu::host_view(l.vram_base + (size_t) l.slot[(size_t) e] * unit(l));
                size_t bad = 0, first = 0;
                for (size_t i = 0; i < l.blob; ++i)
                    if (want[i] != have[i]) { if (!bad) first = i; ++bad; }
                std::fprintf(stderr, "DBG slot il=%d e=%d slot=%d blob=%zu mismatching bytes=%zu first=%zu\n", l.il, e, l.slot[(size_t) e], l.blob, bad, first);
                ++shown;
            }
        }
        if (ds4_dbg("DS4_TRACE_DISP")) std::fprintf(stderr, "TRACE layer %d p0=%lld T=%d\n", l.il, (long long) p0, T);
        if (ds4_dbg("DS4_DBG_H") && mi == 0 && (l.il < 2 || ds4_dbg("DS4_DBG_ALL"))) {   // DBG
            std::vector<float> hv((size_t) T * H * D);
            cudaSetDevice(d.id);
            ds4_metal_flush();
            {
                const float* hvp = (const float*) strata::gpu::host_view(d.h);
                if (hvp) std::memcpy(hv.data(), hvp, hv.size() * 4);
            }
            int nf = 0, nzv = 0;
            float mx = 0.0f;
            size_t firstnan = 0, lastnan = 0;
            for (size_t i = 0; i < hv.size(); ++i) {
                const float v = hv[i];
                if (!(v == v)) { if (nf == 0) firstnan = i; lastnan = i; ++nf; continue; }
                if (v != 0.0f) ++nzv;
                if (std::fabs(v) > mx) mx = std::fabs(v);
            }
            std::fprintf(stderr, "DBG h: il=%d dev=%d h=%p p0=%lld T=%d p0tok=%d zeros=%zu ninf=%d max=%.4g first=%.5g %.5g nan_range=%zu..%zu\n",
                         l.il, (int) l.dev, (void*) d.h, (long long) p0, T, toks.empty() ? -1 : toks[(size_t) p0],
                         hv.size() - (size_t) nzv - (size_t) nf, nf, mx, hv[0], hv.size() > 1 ? hv[1] : 0.0f, firstnan, lastnan);
        }
        prof_mark(d, 0);
        // Engram
        if (l.eg >= 0) {
            const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
            const size_t per = (size_t) cols * (ehd + ehd / 32);
            eg_finish(*eg_win[(size_t) l.eg]);   // (asked for when the window began)
            uint8_t* src = h_eg + (size_t) l.eg * eg_cap * per;
            for (int t = 0; t < T; ++t)   // [token][rows | scales] -> [all rows][all scales]
                CK(cudaMemcpyAsync(d.eg_rows + (size_t) t * cols * ehd, src + (size_t) t * per, (size_t) cols * ehd,
                                   cudaMemcpyHostToDevice, s));
            uint8_t* sc = d.eg_rows + (size_t) T * cols * ehd;
            for (int t = 0; t < T; ++t)
                CK(cudaMemcpyAsync(sc + (size_t) t * cols * (ehd / 32), src + (size_t) t * per + (size_t) cols * ehd,
                                   (size_t) cols * (ehd / 32), cudaMemcpyHostToDevice, s));
            engram_dequant(d.eg_rows, sc, T * cols, ehd, d.eg_x, s, (int) eg_fmt[(size_t) l.eg]);
            dump_sub(d, "eg_rows", l.il, p0, T, d.eg_rows, (size_t) T * cols * ehd);
            dump_sub(d, "eg_x", l.il, p0, T, d.eg_x, (size_t) T * cols * ehd * 4);
            if (ds4_dbg("DS4_DBG_EG")) {
                dbg_nf("eg_x", l.il, d.eg_x, (size_t) T * cols * ehd);
                const float* xv = (const float*) strata::gpu::host_view(d.eg_x);
                for (size_t i = 0; i < (size_t) T * cols * ehd; ++i)
                    if (!std::isfinite(xv[i])) {
                        const size_t t = i / ((size_t) cols * ehd), c = (i / ehd) % cols;
                        const uint8_t* dv = (const uint8_t*) strata::gpu::host_view(d.eg_rows) + (t * cols + c) * ehd;
                        const uint8_t* hv = src + t * per + c * ehd;
                        std::fprintf(stderr, "DBG eg bad x: t=%zu c=%zu e=%zu | dev/host row bytes:", t, c, i % ehd);
                        for (int b = 0; b < 80; ++b) std::fprintf(stderr, "%s%02x/%02x", b % 18 == 0 ? " |" : " ", dv[b], hv[b]);
                        std::fprintf(stderr, "\n");
                        break;
                    }
            }
            if (!mv(d, {job(l.eg_wkv, d.eg_x, d.eg_kv, T, (size_t) cols * ehd, (size_t) (H + 1) * D)}, err)) return false;
            if (ds4_dbg("DS4_DBG_EG")) dbg_nf("eg_kv", l.il, d.eg_kv, (size_t) T * (H + 1) * D);
            dump_sub(d, "eg_kv", l.il, p0, T, d.eg_kv, (size_t) T * (H + 1) * D * 4);
            dump_sub(d, "eg_h0", l.il, p0, T, d.h, (size_t) T * H * D * 4);
            dump_sub(d, "eg_w", l.il, p0, 1, l.eg_w, (size_t) H * D * 4);
            engram_gate(d.h, d.eg_kv, l.eg_w, H, D, T, g.eps, s);
            dump_sub(d, "eg_h", l.il, p0, T, d.h, (size_t) T * H * D * 4);
            if (ds4_dbg("DS4_DBG_EG")) dbg_nf("h_after_gate", l.il, d.h, (size_t) T * H * D);
        }
        dump_sub(d, "h_in", l.il, p0, T, d.h, (size_t) T * H * D * 4);
        dump_sub(d, "ring_in", l.il, p0, T, l.win, (size_t) g.win * kv_row_bytes(kKvFp8, HD));
        // attention
#if defined(STRATA_METAL)
        hc_mix_split(d.h, l.hca_fn, H * D, (2 + H) * H, T, g.eps, d.mix, l.hca_scale, l.hca_base, H,
                     g.sinkhorn, g.hc_eps, d.a_pre, d.a_post, d.a_comb, s);
#else
        hc_mix_split(d.h, l.hca_fn, H * D, (2 + H) * H, T, g.eps, d.mix, d.hccnt, l.hca_scale, l.hca_base, H, g.sinkhorn, g.hc_eps,
                     d.a_pre, d.a_post, d.a_comb, s);
#endif
        dump_sub(d, "a_mix", l.il, p0, T, d.mix, (size_t) T * 32 * 4);
        dump_sub(d, "a_pre", l.il, p0, T, d.a_pre, (size_t) T * 8 * 4);
        dump_sub(d, "a_post", l.il, p0, T, d.a_post, (size_t) T * 8 * 4);
        dump_sub(d, "a_comb", l.il, p0, T, d.a_comb, (size_t) T * 64 * 4);
        hc_pre_norm(d.h, d.pre, l.attn_norm, D, H, T, g.eps, d.x, s);
        dump_sub(d, "pre_in", l.il, p0, T, d.pre, (size_t) T * 8 * 4);
        dump_sub(d, "attn_x", l.il, p0, T, d.x, (size_t) T * D * 4);
        {
            x3::MvJob j[4];
            int nj = 0;
            j[nj++] = job(l.wq_a, d.x, d.qr, T, D, g.q_lora);
            j[nj++] = job(l.wkv, d.x, d.kvr, T, D, HD);
            if (l.kv_src) j[nj++] = job(l.c_wkv, d.x, d.ckr, T, D, HD);
            if (l.kv_src && l.ratio > 1) j[nj++] = job(l.c_wgate, d.x, d.cgr, T, D, HD);
            if (!mv_n(d, j, nj, err)) return false;
        }
        dump_sub(d, "qr_raw", l.il, p0, T, d.qr, (size_t) T * g.q_lora * 4);
        dump_sub(d, "kvr_raw", l.il, p0, T, d.kvr, (size_t) T * HD * 4);
        Finish fq;
        fq.w = l.q_norm;
        fq.eps = g.eps;
        finish(d.qr, T, g.q_lora, g.q_lora, fq, s);
        dump_sub(d, "qr", l.il, p0, T, d.qr, (size_t) T * g.q_lora * 4);
        if (NHL > 0) {
            if (!mv(d, {job(l.wq_b, d.qr, d.q, T, g.q_lora, HQ)}, err)) return false;
            Finish fr;
            fr.rd = g.rd;
            fr.pos = (int) p0;
            fr.vpp = NHL;
            fr.freqs = freqs;
            finish(d.q, T * NHL, HD, HD, fr, s);
        }
        if (NHL > 0) dump_sub(d, "q", l.il, p0, T, d.q, (size_t) T * HQ * 4);
        // the window rows: the ring's (positions p0 - win ..) then this window's, each query its own 128.  A DSpark
        // block's ring holds the trunk's main KV (positions <= p0 - 1, seeded); its block of drafts attends to all of
        // that ring and to the whole block, and leaves the ring as it was
        ring_to_ext(l.win, d.ext, l.mtp ? (int64_t) g.win : p0, g.win, HD, s);   // (a block's ring: its rows as they lie)
        Finish fk;
        fk.w = l.kv_norm;
        fk.eps = g.eps;
        fk.rd = g.rd;
        fk.pos = (int) p0;
        fk.freqs = freqs;
        fk.quant = 1;
        fk.dst = d.ext + (size_t) g.win * HD;
        fk.dst_stride = HD;
        finish(d.kvr, T, HD, HD, fk, s);
        if (!l.mtp) ext_to_ring(d.ext, l.win, p0, g.win, T, HD, s, spec_win ? l.ring_bak : nullptr);
        dump_sub(d, "kv_win", l.il, p0, T, d.ext + (size_t) g.win * HD, (size_t) T * HD * 4);
        dump_sub(d, "ext", l.il, p0, T, d.ext, ((size_t) g.win + T) * HD * 2);   // (BF16 rows: the ring's, then this window's)
        AttnRows ar;
        for (int t = 0; t < T; ++t) {
            if (l.mtp) {
                ar.win0[t] = 0;
                ar.nwin[t] = (int) std::min<int64_t>(g.win, p0);
                ar.nsel[t] = T;
                continue;
            }
            const int first = (int) std::max<int64_t>(g.win - p0, t + 1);   // ext row of position max(0, p - win + 1)
            ar.win0[t] = first;
            ar.nwin[t] = g.win + t - first + 1;
            ar.nsel[t] = 0;
        }
        const void* ckv = nullptr;
        int ckv_fmt = kKvFp4;
        if (l.ratio > 0) {
            Src& sr = srcs[(size_t) l.src];
            const int R = l.ratio;
            // the groups this window completes: g_first .. g_first + ng - 1
            const int64_t g_first = p0 / R;              // the first group whose last position is >= p0
            const int64_t g_end = (p0 + T) / R;          // groups completed by p0 + T - 1
            const int ng = (int) std::max<int64_t>(0, g_end - g_first);
            if (l.kv_src) {
                if (R == 1) {
                    Finish fc;
                    fc.w = l.c_norm;
                    fc.eps = g.eps;
                    finish(d.ckr, T, HD, HD, fc, s);
                    CK(cudaMemcpyAsync(d.latent, d.ckr, (size_t) T * HD * 4, cudaMemcpyDeviceToDevice, s));
                } else {
                    if (spec_win && l.c_snap != nullptr) {   // to replay the accepted rows if drafts are rejected
                        const size_t ns = (size_t) R * HD * 4, nr = (size_t) T * HD * 4;
                        CK(cudaMemcpyAsync(l.c_snap, l.c_kv, ns, cudaMemcpyDeviceToDevice, s));
                        CK(cudaMemcpyAsync(l.c_snap + (size_t) R * HD, l.c_score, ns, cudaMemcpyDeviceToDevice, s));
                        CK(cudaMemcpyAsync(l.c_rows, d.ckr, nr, cudaMemcpyDeviceToDevice, s));
                        CK(cudaMemcpyAsync(l.c_rows + (size_t) kWin * HD, d.cgr, nr, cudaMemcpyDeviceToDevice, s));
                    }
                    compress_win(d.ckr, d.cgr, l.c_kv, l.c_score, R, (int) p0, T, HD, l.c_norm, g.eps, d.latent, s);
                    dump_sub(d, "ckr", l.il, p0, T, d.ckr, (size_t) T * HD * 4);
                    dump_sub(d, "cgr", l.il, p0, T, d.cgr, (size_t) T * HD * 4);
                }
            }
            Rows lim;
            int clen_max = 0;
            for (int t = 0; t < T; ++t) {
                lim.n[t] = (int) ((p0 + t + 1) / R);
                lim.k[t] = std::min(g.ix_topk, lim.n[t]);
                clen_max = std::max(clen_max, lim.n[t]);
            }
            if (l.ix_src) {
                Src& ks = srcs[(size_t) l.ik_src];
                if (l.owns_k && ng > 0) {
                    if (!mv(d, {job(l.i_wk, d.latent, d.ikr, ng, HD, g.ix_hd)}, err)) return false;
                    Finish fi;
                    fi.w = l.ik_norm;
                    fi.eps = g.eps;
                    fi.rd = g.rd;
                    fi.pos = (int) (g_first * R);
                    fi.pos_step = R;
                    fi.freqs = freqs;
                    fi.quant = 3;
                    fi.pdst = ks.ik[(size_t) l.dev] + (size_t) g_first * ik_rb();
                    fi.pdst_stride = (long long) ik_rb();
                    finish(d.ikr, ng, g.ix_hd, g.ix_hd, fi, s);
                    if (!wp.par && !replicate(ks.ik, ks.h_ik, ks.ev_ik, fi.pdst, (size_t) g_first * ik_rb(),
                                              (size_t) ng * ik_rb(), l.dev, err))
                        return false;
                }
                if (clen_max > 0) {
                    if (!mv(d, {job(l.i_wq_b, d.qr, d.iq, T, g.q_lora, (size_t) g.ix_heads * g.ix_hd)}, err)) return false;
                    Finish fi;
                    fi.rd = g.rd;
                    fi.pos = (int) p0;
                    fi.vpp = g.ix_heads;
                    fi.freqs = freqs;
                    fi.quant = 3;
                    finish(d.iq, T * g.ix_heads, g.ix_hd, g.ix_hd, fi, s);
                    // (iw rows are 64 apart)
                    gemv_f16_rows(l.i_wproj, d.x, (size_t) D, g.ix_heads, D, T,
                                  1.0f / std::sqrt((float) g.ix_hd) / std::sqrt((float) g.ix_heads), d.iw, 64, s);
                    const long long ss = (long long) o.max_context + 64;
                    bool restrict_ = false;
                    for (int t = 0; t < T; ++t)
                        restrict_ = restrict_ || (lim.n[t] + g.cand_blk - 1) / g.cand_blk > g.cand_topk;
                    if (l.uses_cand && restrict_) {
                        Rows cl;   // the candidate positions of each query: its blocks x blk
                        int nmax = 0;
                        for (int t = 0; t < T; ++t) {
                            cl.n[t] = lim.n[t];
                            cl.k[t] = wp.cand[l.il][t];
                            nmax = std::max(nmax, wp.cand[l.il][t] * g.cand_blk);
                        }
                        index_score(d.iq, d.iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, T, nmax, d.cand,
                                    (int) cand_stride(), g.cand_blk, cl, d.scores, ss, s);
                        Rows tr;
                        for (int t = 0; t < T; ++t) {
                            tr.n[t] = wp.cand[l.il][t] * g.cand_blk;
                            tr.k[t] = lim.k[t];
                        }
                        topk(d.scores, ss, T, tr, d.cand, (int) cand_stride(), g.cand_blk, d.topk, (int) topk_stride(), s);
                    } else {
                        index_score(d.iq, d.iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, T, clen_max, nullptr, 0, 1,
                                    lim, d.scores, ss, s);
                        if (l.cand_src) {
                            Rows cr;
                            for (int t = 0; t < T; ++t) {
                                cr.n[t] = lim.n[t];
                                cr.k[t] = g.cand_topk;
                            }
                            cand_blocks(d.scores, ss, T, cr, g.cand_blk, d.bm, (long long) bm_stride(), d.cand,
                                        (int) cand_stride(), s);
                        }
                        topk(d.scores, ss, T, lim, nullptr, 0, 1, d.topk, (int) topk_stride(), s);
                    }
                }
            }
            if (l.kv_src && ng > 0) {
                Finish fc;
                fc.rd = g.rd;
                fc.pos = (int) (g_first * R);
                fc.pos_step = R;
                fc.freqs = freqs;
                fc.quant = 2;
                fc.pdst = sr.ckv[(size_t) l.dev] + (size_t) g_first * ckv_rb();
                fc.pdst_stride = (long long) ckv_rb();
                finish(d.latent, ng, HD, HD, fc, s);
                if (!wp.par && !replicate(sr.ckv, sr.h_ckv, sr.ev_ckv, fc.pdst, (size_t) g_first * ckv_rb(),
                                          (size_t) ng * ckv_rb(), l.dev, err))
                    return false;
                if (wp.par && mi == 0 && sr.remote) {   // the new rows for the other groups' readers
                    Xfer x;
                    x.src[0] = fc.pdst;
                    x.dst[0] = sr.x_stage;
                    x.bytes[0] = (size_t) ng * ckv_rb();
                    x.n = 1;
                    if (l.owns_k && srcs[(size_t) l.ik_src].ik[(size_t) l.dev] != nullptr) {
                        x.src[1] = srcs[(size_t) l.ik_src].ik[(size_t) l.dev] + (size_t) g_first * ik_rb();
                        x.dst[1] = sr.x_stage + x_cap() * ckv_rb();
                        x.bytes[1] = (size_t) ng * ik_rb();
                        x.n = 2;
                    }
                    xfer_put(x, sr.x_flag, wp.win, d.xcnt, s);
                }
            }
            if (ng > 0) dump_sub(d, "latent", l.il, p0, ng, d.latent, (size_t) ng * HD * 4);
            dump_sub(d, "topk", l.il, p0, T, d.topk, (size_t) T * topk_stride() * 4);
            for (int t = 0; t < T; ++t) ar.nsel[t] = lim.n[t] > 0 ? wp.topk[l.il][t] : 0;
            ckv = sr.ckv[(size_t) l.dev];
        }
        if (l.mtp) {   // the block's rows (BF16), after the ring's in ext, as every query's selection
            ckv = d.ext + (size_t) g.win * HD;
            ckv_fmt = kKvBf16;
            iota_rows(d.topk, T, T, (int) topk_stride(), s);
        }
        if (NHL > 0) {
            sparse_attn(d.q, d.ext, ckv, d.topk, (int) topk_stride(), ar, l.sink, NHL, HD, T,
                        1.0f / std::sqrt((float) HD), d.o, s, ckv_fmt, 0, &d.attn_scr);
            dump_sub(d, "o", l.il, p0, T, d.o, (size_t) T * HQ * 4);
            Finish fo;
            fo.rd = g.rd;
            fo.pos = (int) p0;
            fo.vpp = NHL;
            fo.freqs = freqs;
            fo.inverse = 1;
            fo.quant = 1;
            finish(d.o, T * NHL, HD, HD, fo, s);
            dump_sub(d, "o_fin", l.il, p0, T, d.o, (size_t) T * HQ * 4);
            x3::MvJob j[16];
            const int per = HQ / OG;
            for (int gi = 0; gi < OG; ++gi)
                j[gi] = job(l.wo_a[gi], d.o + (size_t) gi * per, d.oa + (size_t) gi * g.o_lora, T, HQ, (size_t) OG * g.o_lora);
            if (!mv_n(d, j, OG, err)) return false;
            if (!mv(d, {job(l.wo_b, d.oa, d.ao, T, (size_t) OG * g.o_lora, D)}, err)) return false;
        } else CK(cudaMemsetAsync(d.ao, 0, (size_t) T * D * sizeof(*d.ao), s));   // (no heads: nothing to add)
        dump_sub(d, "ao", l.il, p0, T, d.ao, (size_t) T * D * 4);
        if (nm > 1) allreduce(d.ao, T * D, ag, mi, ++d.ar_seq, d.xcnt, s);
        hc_post(d.ao, d.h, d.a_post, d.a_comb, D, H, T, d.h2, s);
        std::swap(d.h, d.h2);
        if (dump_l != nullptr && mi == 0) dump_layer(d, d.h, 1000 + l.il, p0, T);   // (DS4_DUMP_L: after the attention)
        dump_sub(d, "h_attn", l.il, p0, T, d.h, (size_t) T * H * D * 4);
        prof_mark(d, 1);
        // the MoE
#if defined(STRATA_METAL)
        hc_mix_split(d.h, l.hcf_fn, H * D, (2 + H) * H, T, g.eps, d.mix, l.hcf_scale, l.hcf_base, H,
                     g.sinkhorn, g.hc_eps, d.f_pre, d.f_post, d.f_comb, s);
#else
        hc_mix_split(d.h, l.hcf_fn, H * D, (2 + H) * H, T, g.eps, d.mix, d.hccnt, l.hcf_scale, l.hcf_base, H, g.sinkhorn, g.hc_eps,
                     d.f_pre, d.f_post, d.f_comb, s);
#endif
        hc_pre_norm(d.h, d.a_pre, l.ffn_norm, D, H, T, g.eps, d.x, s);
        dump_sub(d, "f_pre", l.il, p0, T, d.f_pre, (size_t) T * 8 * 4);
        dump_sub(d, "f_post", l.il, p0, T, d.f_post, (size_t) T * 8 * 4);
        dump_sub(d, "f_comb", l.il, p0, T, d.f_comb, (size_t) T * 64 * 4);
        dump_sub(d, "ffn_x", l.il, p0, T, d.x, (size_t) T * D * 4);
        if (wp.async) {
            // the routing goes to the service thread through the mailbox (a group's lead publishes it, the members
            // make the same plan) while the device runs the shared expert and the VRAM experts (from the plan the
            // router made), and waits for the answer only before the experts' sum
            gemv_f32(l.gate_w, d.x, l.n_exp, D, T, d.glog, 1024, s);
            const unsigned sq = wp.seq[l.il];
            const int slot = (int) (sq % kRing);
            const bool lead = mi == 0;
            int na[2] = {0, 0};   // lookahead: the next layers' guesses (their gate on this FFN input), published below
            for (int a = 1; lead && a <= std::min(2, ahead_mode); ++a) {
                const Layer* nx = ahead_layer(L0, a);
                if (nx == nullptr) break;
                const int m = a == 1 ? std::max(1, std::min(kAhead1, nx->k_exp + ahead_extra)) : std::min(kPicks, nx->k_exp);
                float* gl = d.glog_a + (size_t) (a - 1) * kWin * 1024;
                gemv_f32(nx->gate_w, d.x, nx->n_exp, D, T, gl, 1024, s);   // (normed with the next layer's FFN
                                                                         // weights instead: the same hit rate)
                route_ahead(gl, nx->gate_bias, nx->n_exp, m, T, reqs + slot, (a - 1) * kAhead1, s);
                na[a - 1] = m;
            }
            route_publish(d.glog, l.gate_bias, l.n_exp, l.k_exp, T, g.route_scale, l.vram_tab, d.x, D, d.ridx, d.rw,
                          d.plan_ptr, lead ? reqs + slot : nullptr, sq, s, l.gate_bias_vl, l.mtp ? 0u : wp.vl, na[0],
                          na[1]);
            if (!mv(d, {job(l.s1, d.x, d.sg, T, D, FF), job(l.s3, d.x, d.su, T, D, FF)}, err)) return false;
            swiglu(d.sg, d.su, T * FF, g.lim, d.sh, s);
            if (!mv(d, {job(l.s2, d.sh, d.sh_out, T, FF, D)}, err)) return false;
            prof_mark(d, 2);
            const x3::ExpertLayout& lay = nm > 1 ? l.hlay : l.lay;
            if (T == 1 && moe_token_path) {
                x3::MoeArgs ma;
                ma.plan_ptr = d.plan_ptr;
                ma.plan_w = d.rw;
                ma.cpu_part = lead ? resps[slot].part[0] : nullptr;
                ma.cpu_flag = lead ? &resps[slot].cpu[0] : nullptr;
                ma.lay = lay;
                ma.k = l.k_exp;
                ma.n_embd = D;
                ma.n_ff = FF;
                ma.limit = g.lim;
                const bool mx = lay.fmt == gg::kFmtMXFP4;
                if (zc_stage_buf != nullptr && l.zc_tab != nullptr)   // (the store's experts copied to VRAM at the PCIe rate first)
                    zc_stage(d.plan_ptr, d.ridx, l.zc_tab, l.k_exp, zc_stage_buf, zc_stride, l.blob, s);
                // DS4_TLOG2: the expert stage's three host spans (gate/up issue, the CPU-lane wait, down issue)
                static const bool tlog2 = ds4_dbg("DS4_TLOG2");
                static double t_moe_gu = 0, t_moe_wait = 0, t_moe_dn = 0; static long t_moe_n = 0; static bool t_moe_reg = false;
                if (tlog2 && !t_moe_reg) {
                    t_moe_reg = true;
                    std::atexit([] {
                        std::fprintf(stderr, "tlog2: %ld stages | host ms a stage: gate_up %.2f  cpu_wait %.2f  down %.2f\n",
                                     t_moe_n, t_moe_n ? t_moe_gu / (double) t_moe_n : 0.0,
                                     t_moe_n ? t_moe_wait / (double) t_moe_n : 0.0,
                                     t_moe_n ? t_moe_dn / (double) t_moe_n : 0.0);
                    });
                }
                const double m0 = tlog2 ? now_ms() : 0.0;
                if (mx) gg::moe_gate_up(ma, d.x, d.xg, s);
                else x3::moe_gate_up(ma, d.x, d.xg, s);
                const double m1 = tlog2 ? now_ms() : 0.0;
                prof_mark(d, 3);
                if (lead) wait_seq(&resps[slot].seq, sq, s);
                const double m2 = tlog2 ? now_ms() : 0.0;
                prof_mark(d, 4);
                if (mx) gg::moe_down(ma, d.xg, d.sh_out, d.moe_out, s);
                else x3::moe_down(ma, d.xg, d.sh_out, d.moe_out, s);
                if (tlog2) { t_moe_gu += m1 - m0; t_moe_wait += m2 - m1; t_moe_dn += now_ms() - m2; ++t_moe_n; }
            } else {   // each distinct expert read once for the window's tokens
                gf::MoeWin mw;
                mw.plan_ptr = d.plan_ptr;
                mw.T = T;
                mw.k = l.k_exp;
                mw.grp_ptr = d.grp_ptr;
                mw.grp_start = d.grp_start;
                mw.grp_ent = d.grp_ent;
                mw.n_groups = d.n_groups;
                gf::moe_win_group(mw, s);
                x3::WinArgs wa;
                wa.grp_ptr = d.grp_ptr;
                wa.grp_start = d.grp_start;
                wa.grp_ent = d.grp_ent;
                wa.n_groups = d.n_groups;
                wa.plan_ptr = d.plan_ptr;
                wa.plan_w = d.rw;
                wa.cpu_part = lead ? &resps[slot].part[0][0] : nullptr;
                wa.cpu_stride = kMaxDim;
                wa.cpu_flag = lead ? resps[slot].cpu : nullptr;
                wa.wait_ptr = lead ? &resps[slot].seq : nullptr;
                wa.wait_val = sq;
                wa.x = d.x;
                wa.x_stride = (size_t) D;
                wa.h = d.xg;
                wa.part = d.ed;
                wa.sh_out = d.sh_out;
                wa.out = d.moe_out;
                wa.out_stride = (size_t) D;
                wa.lay = lay;
                wa.T = T;
                wa.k = l.k_exp;
                wa.n_embd = D;
                wa.n_ff = FF;
                wa.limit = g.lim;
                if (!(lay.fmt == gg::kFmtMXFP4 ? gg::win_experts(wa, s) : x3::win_experts(wa, s))) {
                    err = "exl3 win_experts failed";
                    return false;
                }
            }
            if (lead) st.experts += (int64_t) T * l.k_exp;
            if (nm > 1) allreduce(d.moe_out, T * D, ag, mi, ++d.ar_seq, d.xcnt, s);
            hc_post(d.moe_out, d.h, d.f_post, d.f_comb, D, H, T, d.h2, s);
            std::swap(d.h, d.h2);
            CK(cudaMemcpyAsync(d.pre, d.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
            prof_mark(d, 5);
            return true;
        }
        if (nm > 1) {
            err = "a tensor-parallel layer on the host-routed path";
            return false;
        }
        // (glog rows are 1024 apart: route reads them so)
        static const bool phases = ds4_dbg("DS4_PHASES");   // debugging: where a decode layer's host time goes
        static double ph[8] = {0}, ph_prev = 0; static long ph_n = 0;
        const double ph0 = phases ? now_ms() : 0.0;
        gemv_f32(l.gate_w, d.x, l.n_exp, D, T, d.glog, 1024, s);
        VlBits vb = {};
        vb.w[0] = l.mtp ? 0u : wp.vl;
        route(d.glog, l.gate_bias, l.n_exp, l.k_exp, T, g.route_scale, d.ridx, d.rw, s, l.gate_bias_vl,
              vb.w[0] != 0 ? &vb : nullptr);
        CK(cudaMemcpyAsync(h_idx, d.ridx, (size_t) T * kPicks * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(h_w, d.rw, (size_t) T * kPicks * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(h_x, d.x, (size_t) T * D * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaEventRecord(d.ev, s));
        // the shared expert while the host looks at the picks
        if (!mv(d, {job(l.s1, d.x, d.sg, T, D, g.ff), job(l.s3, d.x, d.su, T, D, g.ff)}, err)) return false;
#if defined(STRATA_METAL)
        int la_n = 0;   // the next layers' guessed picks (their gate on this FFN input), on the GPU: a gemv and a route a layer
        if (T == 1 && la_enabled()) {
            for (int a = 1; a <= la_depth(); ++a) {
                const int il = l.il + a;
                if (il >= g.nl) break;
                const Layer& nx = L[(size_t) il];
                if (nx.nm != 1 || nx.lay.fmt != gg::kFmtMXFP4 || nx.gate_bias == nullptr || nx.gate_w == nullptr) break;
                gemv_f32(nx.gate_w, d.x, nx.n_exp, D, 1, d.glog_a + (size_t) (a - 1) * kWin * 1024, 1024, s);
                route(d.glog_a + (size_t) (a - 1) * kWin * 1024, nx.gate_bias, nx.n_exp, nx.k_exp, 1, g.route_scale, d.ridx_a + (a - 1) * kPicks,
                      d.rw_a + (a - 1) * kPicks, s, nullptr, nullptr);
                CK(cudaMemcpyAsync(h_idx_a + (a - 1) * kPicks, d.ridx_a + (a - 1) * kPicks, (size_t) kPicks * 4, cudaMemcpyDeviceToHost, s));
                la_n = a;
            }
            if (la_n > 0) CK(cudaEventRecord(d.ev2, s));
        }
#endif
        if (ds4_dbg("DS4_DBG_MOE")) { dbg_nf("x", l.il, d.x, (size_t) T * D); dbg_nf("sg", l.il, d.sg, (size_t) T * g.ff); dbg_nf("su", l.il, d.su, (size_t) T * g.ff);
            std::fprintf(stderr, "DBG s1 fmt=%d K=%d k=%d n=%d ld=%d raw=%p\n", l.s1.fmt, l.s1.K, l.s1.k, l.s1.n, l.s1.ld, (void*) l.s1.raw); }
        swiglu(d.sg, d.su, T * g.ff, g.lim, d.sh, s);
        if (ds4_dbg("DS4_DBG_MOE")) dbg_nf("sh", l.il, d.sh, (size_t) T * g.ff);
        if (!mv(d, {job(l.s2, d.sh, d.sh_out, T, g.ff, D)}, err)) return false;
        const double tw = now_ms();
        CK(cudaEventSynchronize(d.ev));
        st.gpu_wait_ms += now_ms() - tw;
        const double ph2 = phases ? now_ms() : 0.0;
        dump_sub(d, "glog", l.il, p0, T, d.glog, (size_t) T * 1024 * 4);
        dump_sub_host("ridx", l.il, p0, T, h_idx, (size_t) T * kPicks * 4);
        dump_sub_host("rw", l.il, p0, T, h_w, (size_t) T * kPicks * 4);
        dump_sub(d, "sh_out", l.il, p0, T, d.sh_out, (size_t) T * D * 4);
        // the picks by expert: the VRAM ones' rows (gathered inputs, one GEMV per matrix for all their tokens), the
        // CPU lane's entries
        Combine cm;
        Gather ga;
        uint32_t* pr = prof.data() + (size_t) l.il * g.n_exp;
        int vex[kWin * kPicks], vrow0[kWin * kPicks], vn[kWin * kPicks];
        int nvex = 0, nrows = 0;
        std::vector<Ent> ents;
        std::vector<std::pair<int, float>> cpu_rows;   // slot -> (token, weight)
        std::vector<std::vector<std::pair<int, int>>> per((size_t) l.n_exp);   // expert -> (token, pick)
        for (int t = 0; t < T; ++t) {
            cm.n[t] = 0;
            cm.cpu[t] = 0;
            for (int i = 0; i < l.k_exp; ++i) {
                const int e = h_idx[t * kPicks + i];
                if (e < 0 || e >= l.n_exp) {
                    err = "router picked expert " + std::to_string(e) + " (layer " + std::to_string(l.il) + ", token " + std::to_string(t) + ", pick " + std::to_string(i) + ", of " + std::to_string(l.k_exp) + ")";
                    return false;
                }
                ++pr[e];
                l.heat[(size_t) e] += 1.0f;
                if (l.slot[(size_t) e] < 0) l.cpu_hits.push_back(e);
                per[(size_t) e].push_back({t, i});
            }
        }
        int nslots = 0;
        for (int e = 0; e < l.n_exp; ++e) {
            const auto& pe = per[(size_t) e];
            if (pe.empty()) continue;
#if defined(STRATA_METAL)
            const bool on_gpu = l.slot[(size_t) e] >= 0 || (T == 1 && zc_try(l, e));
#else
            const bool on_gpu = l.slot[(size_t) e] >= 0;
#endif
            if (on_gpu) {
                vex[nvex] = e;
                vrow0[nvex] = nrows;
                vn[nvex] = (int) pe.size();
                for (const auto& tp : pe) {
                    ga.src[nrows] = tp.first;
                    const int t = tp.first;
                    cm.row[t][cm.n[t]] = nrows;
                    cm.w[t][cm.n[t]] = h_w[t * kPicks + tp.second];
                    ++cm.n[t];
                    ++nrows;
                }
                ++nvex;
            } else {
                for (size_t a = 0; a < pe.size(); a += (size_t) ent_cap(l)) {
                    Ent en;
                    en.e = e;
                    en.slot0 = nslots;
                    en.n = (int) std::min<size_t>((size_t) ent_cap(l), pe.size() - a);
                    for (int i = 0; i < en.n; ++i) {
                        en.tok[i] = pe[a + (size_t) i].first;
                        cpu_rows.push_back({pe[a + (size_t) i].first, h_w[pe[a + (size_t) i].first * kPicks + pe[a + (size_t) i].second]});
                        cm.cpu[en.tok[i]] = 1;
                    }
                    nslots += en.n;
                    ents.push_back(en);
                }
            }
        }
        st.experts += (int64_t) T * l.k_exp;
        st.cpu_experts += nslots;
#if defined(STRATA_METAL)
        if (T == 1) lookahead_prefetch(l, d.ev2, la_n, ents);
#endif
        if (nrows > 0) {
            gather_rows(d.x, ga, nrows, D, d.xe, s);
            dump_sub(d, "xe", l.il, p0, nrows, d.xe, (size_t) nrows * D * 4);
            if (dump_s != nullptr) { std::vector<int> vmap; for (int i = 0; i < nvex; ++i) { vmap.push_back(vex[i]); vmap.push_back(vrow0[i]); vmap.push_back(vn[i]); }
              dump_sub_host("vmap", l.il, p0, nvex, vmap.data(), vmap.size() * 4); }
            x3::MvJob j[2 * kWin * kPicks];
            int nj = 0;
            for (int i = 0; i < nvex; ++i) {
                const auto& vm = l.vm[(size_t) vex[i]];
                j[nj++] = job(vm[0], d.xe + (size_t) vrow0[i] * D, d.xg + (size_t) vrow0[i] * g.ff, vn[i], D, g.ff);
                j[nj++] = job(vm[1], d.xe + (size_t) vrow0[i] * D, d.xu + (size_t) vrow0[i] * g.ff, vn[i], D, g.ff);
            }
            if (!mv_n(d, j, nj, err)) return false;
            dump_sub(d, "xg", l.il, p0, nrows, d.xg, (size_t) nrows * g.ff * 4);
            dump_sub(d, "xu", l.il, p0, nrows, d.xu, (size_t) nrows * g.ff * 4);
            swiglu(d.xg, d.xu, nrows * g.ff, g.lim, d.xhh, s);
            dump_sub(d, "xhh", l.il, p0, nrows, d.xhh, (size_t) nrows * g.ff * 4);
            nj = 0;
            for (int i = 0; i < nvex; ++i)
                j[nj++] = job(l.vm[(size_t) vex[i]][2], d.xhh + (size_t) vrow0[i] * g.ff, d.ed + (size_t) vrow0[i] * D,
                              vn[i], g.ff, D);
            if (!mv_n(d, j, nj, err)) return false;
            dump_sub(d, "ed", l.il, p0, nrows, d.ed, (size_t) nrows * D * 4);
        }
        const double ph3 = phases ? now_ms() : 0.0;
        if (!ents.empty()) {
            const double tc = now_ms();
#if defined(STRATA_METAL)
            ds4_metal_kick();   // (the VRAM experts run on the GPU while the CPU lane works)
            ds4_busy_begin();   // (and the GPU is kept out of its idle clocks until the lane is done)
#endif
            static const bool skip_cpu = ds4_dbg("DS4_SKIP_CPU");   // (timing only: the CPU lane's experts left out)
            if (!skip_cpu) cpu_experts(l, h_x, ents);
            else if (ds4_dbg("DS4_FAKE_CPU_US")) std::this_thread::sleep_for(std::chrono::microseconds(std::atoi(std::getenv("DS4_FAKE_CPU_US"))));   // (timing only)
#if defined(STRATA_METAL)
            ds4_busy_end();
#endif
            std::fill(h_part, h_part + (size_t) T * D, 0.0f);
            for (int sl = 0; sl < nslots; ++sl) {
                float* o = h_part + (size_t) cpu_rows[(size_t) sl].first * D;
                const float wv = cpu_rows[(size_t) sl].second;
                const float* y = c_yd.data() + (size_t) sl * D;
                for (int e = 0; e < D; ++e) o[e] += wv * y[e];
            }
            if (ds4_dbg("DS4_CPU_LOG")) { static double ms = 0; static long calls = 0, exps = 0; static bool reg = false;
              ms += now_ms() - tc; ++calls; exps += (long) ents.size();
              if (ds4_dbg("DS4_CPU_LOG") && !reg) { reg = true; std::atexit([] { std::fprintf(stderr, "cpu lane calls: %ld, experts %ld, %.1f ms (%.3f ms a call, %.3f ms an expert)%c", calls, exps, ms, ms / (double) calls, ms / (double) exps, 10); }); }
              (void) 0; (void) ms; }
            st.cpu_ms += now_ms() - tc;
            dump_sub_host("cpu_part", l.il, p0, T, h_part, (size_t) T * D * 4);
            CK(cudaMemcpyAsync(d.cpu_part, h_part, (size_t) T * D * 4, cudaMemcpyHostToDevice, s));
        }
        combine(d.ed, ents.empty() ? nullptr : d.cpu_part, d.sh_out, cm, T, D, d.moe_out, s);
        dump_sub(d, "moe_out", l.il, p0, T, d.moe_out, (size_t) T * D * 4);
        if (ds4_dbg("DS4_DBG_MOE")) {
            float hp = 0.0f;
            for (size_t i = 0; i < (size_t) T * D; ++i) hp = std::max(hp, std::fabs(h_part[i]));
            std::fprintf(stderr, "DBG moe il=%d T=%d vram_rows=%d cpu_slots=%d max|h_part|=%.4g", l.il, T, nrows, nslots, (double) hp);
            dbg_nf("moe_out", l.il, d.moe_out, (size_t) T * D);
            dbg_nf("sh_out", l.il, d.sh_out, (size_t) T * D);
        }
        hc_post(d.moe_out, d.h, d.f_post, d.f_comb, D, H, T, d.h2, s);
        std::swap(d.h, d.h2);
        dump_sub(d, "h_out", l.il, p0, T, d.h, (size_t) T * H * D * 4);
        if (dump_l != nullptr && mi == 0) dump_layer(d, d.h, l.il, p0, T);   // (post-layer residual, as the chunk path writes it)
        CK(cudaMemcpyAsync(d.pre, d.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
        if (phases && T == 1) {
            const double ph4 = now_ms();
            ph[0] += ph2 - ph0; ph[1] += ph3 - ph2; ph[2] += ph4 - ph3; if (ph_prev > 0 && ph0 - ph_prev < 5.0) ph[3] += ph0 - ph_prev; ph_prev = ph4; ++ph_n;
            static bool reg = false;
            if (!reg) { reg = true; std::atexit([] { std::fprintf(stderr, "phases per layer call (%ld): route enqueue + shared expert + wait %.3f ms, routing analysis + GPU experts enqueue %.3f ms, CPU lane + combine + hc_post %.3f ms, attention enqueue %.3f ms%s", ph_n, ph[0] / ph_n, ph[1] / ph_n, ph[2] / ph_n, ph[3] / ph_n, "\n"); }); }
        }
        // (h_part / h_x are written again only after the next layer's route has synchronized, which follows these)
        return true;
    }

    int sample(float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter) {
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        strata::dsv41k::SamplerParams sp;
        sp.temperature = temperature;
        sp.greedy = temperature <= 0.0f;
        sp.top_p = top_p;
        sp.top_k = top_k;
        sp.min_p = min_p;
        sp.seed = seed;
        sp.counter = counter;
        strata::dsv41k::sample_tokens(d.logits + (size_t) std::max(0, n_logits - 1) * g.vocab, 1, g.vocab, nullptr, 0,
                                       sp, d.tok, d.st);
        int t = -1;
#if defined(STRATA_METAL)
        ds4_metal_flush();   // (a host read of a device result: commit the batched dispatches first)
#endif
        cudaMemcpyAsync(&t, d.tok, 4, cudaMemcpyDeviceToHost, d.st);
        cudaStreamSynchronize(d.st);
        return t;
    }

    bool logits(std::vector<float>& out) {
        if (logits_all) {
            out = chunk_logits;
            return true;
        }
        Dev& d = devs[(size_t) head_dev];
        cudaSetDevice(d.id);
        out.resize((size_t) std::max(1, n_logits) * g.vocab);
#if defined(STRATA_METAL)
        ds4_metal_flush();   // (a host read of a device result: commit the batched dispatches first)
#endif
        return cudaMemcpy(out.data(), d.logits, out.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    void save_profile() const {
        if (o.profile.empty() || o.profile_readonly) return;   // (--profile-readonly: a test keeps the file it read)
        const std::string tmp = o.profile + ".tmp";
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        if (f == nullptr) return;
        const bool ok = std::fwrite(prof.data(), 4, prof.size(), f) == prof.size();
        std::fclose(f);
        // (the old profile replaced: std::rename does not replace an existing file on Windows, so there the profile
        // never moved on and only .tmp files were left)
        if (!ok || !os::replace_file(tmp, o.profile)) std::remove(tmp.c_str());
    }

    std::vector<std::string> info_lines() const {
        std::vector<std::string> v;
        int nh = 0;
        for (const Layer& l : L) nh += l.n_exp - l.n_vram;
        char b[512];
        // kv=fp4: DeepSeek's own cache layout (the compressed KV and the index keys FP4, the window FP8)
        std::snprintf(b, sizeof b, "INFO context=%lld kv=fp4 kv_resident=%lld expert_slots=%d engine_kind=ds4 "
                                   "cpu_experts=%d ram_experts=%d engine=ds4-0.1",
                      (long long) o.max_context, (long long) o.max_context, g.nl * g.n_exp - nh, nh, ram_held);
        v.push_back(b);
        if (commit_kept > 0) {   // (Windows, an integrated GPU: the commit kept experts off the GPU - the server says so)
            std::snprintf(b, sizeof b,
                          "NOTE experts: %.0f GB stay off the GPU - Windows' page file is too small to hold them there; "
                          "a page file of %.0f GB keeps them on the GPU (System > About > Advanced system settings > "
                          "Performance > Advanced > Virtual memory), then restart Maya", commit_kept / 1e9, pagefile_hint);
            v.push_back(b);
            std::snprintf(b, sizeof b, "INFO gpu_commit_capped_gb=%.0f pagefile_hint_gb=%.0f", std::ceil(commit_kept / 1e9),
                          pagefile_hint);
            v.push_back(b);
        }
        if (pk.eg_layers > 0) {   // the n-gram memory's (Engram's) table rows a token reads, for the Monitor
            std::snprintf(b, sizeof b, "INFO engram_layers=%d engram_lookups=%d", pk.eg_layers,
                          pk.eg_layers * (pk.eg_ngram - 1) * pk.eg_heads);
            v.push_back(b);
        }
        if (img_tok >= 0) {   // the server's image resize (DeepSeek's image_processor.py)
            std::snprintf(b, sizeof b,
                          "INFO vision=ds4 vision_patch=%d vision_merge=%d vision_min_pixels=%lld vision_max_tokens=%d "
                          "vision_max_wh_ratio=%lld image_token=%d vision_gpu=%d vision_mib=%lld",
                          vg.patch, vg.merge, (long long) pk.num("vision.min_pixels", 0), vg.max_tokens,
                          (long long) pk.num("vision.max_wh_ratio", 0), img_tok, vt.device(),
                          (long long) (vt.device_bytes() >> 20));
            v.push_back(b);
        } else if (!vision_err.empty()) {   // (the server offers no pictures and says why)
            std::string why = vision_err;
            for (char& ch : why)
                if (ch == '\n' || ch == '\r') ch = ' ';
            v.push_back("NOTE images: OFF - the model's vision tower did not load (" + why + ")");
            v.push_back("INFO vision=off");
        }
        return v;
    }
};

Model::Model() : p_(new Impl) {}
Model::~Model() = default;
bool Model::load(const Options& o, std::string& err) { return p_->load(o, err); }
bool Model::step(int token, bool logits, std::string& err) { return p_->step(token, logits, err); }
bool Model::feed(const int* tokens, int n, int logits_rows, std::string& err) {
    Impl& p = *p_;
    p.pm.last = -1;
    p.halted = false;
    if (ds4_dbg("DS4_DBG_CHUNK")) std::fprintf(stderr, "feed n %d chunk %d chunk_min %d\n", n, p.chunk, p.chunk_min);
    if (p.chunk > 0 && n >= p.chunk_min) {   // a long run - or one the flow could take, if it is the faster path here
        const double t0 = now_ms();   // (every row's logits (tests): the chunked path, which keeps them all)
        const bool ok = logits_rows > kWin || p.pick_path(n) ? p.forward_chunks(tokens, n, logits_rows, err)
                                                             : p.feed_windows(tokens, n, logits_rows, err);
        if (ok) p.path_done(now_ms() - t0);
        return ok;
    }
    return p.feed_windows(tokens, n, logits_rows, err);
}
bool Model::feed_mark(const int* tokens, int n, int logits_rows, int slot, std::string& err) {
    Impl& p = *p_;
    p.pm.last = -1;
    p.halted = false;
    const double t0 = now_ms();
    if (n >= 2 && p.chunk > 0 && n >= p.chunk_min && p.pick_path(n)) {   // (the last token rides along; its layers
        const bool ok = p.forward_chunks(tokens, n, logits_rows, err, slot);   // keep the state before it)
        if (ok) p.path_done(now_ms() - t0);
        return ok;
    }
    if (n > 1 && !p.feed_windows(tokens, n - 1, 0, err)) return false;
    p.checkpoint(slot);
    const bool ok = n < 1 || p.feed_windows(tokens + n - 1, 1, logits_rows, err);
    if (ok) p.path_done(now_ms() - t0);   // (windows, when pick_path chose them)
    return ok;
}
int Model::feed_unit() const { return p_->chunk > 0 ? std::max(p_->chunk, (int) p_->o.max_context) : kWin; }
void Model::set_progress(std::function<void(int64_t)> f) { p_->progress = std::move(f); }
void Model::set_cancel(std::function<bool()> f) { p_->cancel = std::move(f); }
bool Model::cancelled() const { return p_->halted; }
int Model::sample(float t, float tp, int tk, float mp, uint64_t seed, uint64_t counter) {
    return p_->sample(t, tp, tk, mp, seed, counter);
}
bool Model::logits(std::vector<float>& out) { return p_->logits(out); }
int Model::image_token() const { return p_->img_tok; }
int Model::dim() const { return p_->g.dim; }
bool Model::encode_image(const uint8_t* rgb, int width, int height, std::vector<uint16_t>& span, std::string& err) {
    return p_->encode_image(rgb, width, height, span, err);
}
void Model::set_images(std::vector<Image> images) {
    std::sort(images.begin(), images.end(), [](const Image& a, const Image& b) { return a.start < b.start; });
    p_->imgs = std::move(images);
}
int Model::draft_block() const { return p_->g.n_mtp > 0 ? p_->g.d_block : 0; }
bool Model::gpus_are_gfx12() const { return p_->rdna4(); }
bool Model::draft(int y, std::vector<int>& d, std::vector<float>& c, std::string& err, float t, float tp, int tk,
                  float mp, uint64_t seed, uint64_t counter) {
    return p_->draft(y, d, c, err, t, tp, tk, mp, seed, counter);
}
bool Model::verify(int y, const std::vector<int>& d, float t, float tp, int tk, float mp, uint64_t seed, uint64_t counter,
                   int& accepted, int& next, std::string& err) {
    return p_->verify(y, d, t, tp, tk, mp, seed, counter, accepted, next, err);
}
int64_t Model::pos() const { return p_->pos; }
int64_t Model::max_context() const { return p_->o.max_context; }
int Model::vocab() const { return p_->g.vocab; }
const std::vector<int32_t>& Model::tokens() const { return p_->toks; }
void Model::reset() { p_->reset(); }
void Model::checkpoint(int slot) { p_->checkpoint(slot); }
bool Model::restore(int64_t limit) { return p_->restore(limit); }
void Model::pin() { p_->pin(); }
std::vector<int64_t> Model::check_positions() const { return p_->check_positions(); }
int64_t Model::restorable(int64_t limit) const { return p_->restorable(limit); }
uint64_t Model::slot_save(std::vector<uint8_t>& out, std::string& err) { return p_->slot_save(out, err); }
bool Model::slot_load(const uint8_t* data, size_t bytes, int64_t n_pos, std::string& err) {
    return p_->slot_load(data, bytes, n_pos, err);
}
std::string Model::slot_identity() const { return p_->slot_identity(); }
void Model::settle_idle(const std::atomic<bool>& cancel) { p_->cache_settle_idle(cancel); }
void Model::save_profile() const { p_->save_profile(); }
const Stats& Model::stats() const {
    Stats& st = p_->st;
    if (p_->store) {   // (the store keeps its own counters: copied in when asked)
        const ExpertStore& x = *p_->store;
        st.store_hits = x.stats().hits.load();
        st.store_misses = x.stats().misses.load();
        st.store_prefetched = x.stats().prefetched.load();
        st.store_pf_used = x.stats().prefetch_used.load();
        st.store_pf_wasted = x.stats().prefetch_wasted.load();
        st.store_evictions = x.stats().evictions.load();
        st.store_bytes = x.stats().bytes_read.load();
        st.store_fails = x.stats().fails.load();
        st.store_wait_ms = x.stats().wait_ns.load() / 1e6;
        st.store_busy_ms = x.busy_ms();
        st.store_zc_visible = (int64_t) x.visible_count();
        st.store_slots = (int64_t) x.slots_allocated();
        st.store_slots_max = (int64_t) x.slots_max();
    }
    return st;
}
std::string Model::describe() const { return p_->describe(); }
std::vector<std::string> Model::info_lines() const { return p_->info_lines(); }

}  // namespace strata::ds4
