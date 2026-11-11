// src/kernels/cpu/expert_layout.cpp - plan v0.3 P6: the per-layer expert table.  See the header.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#if defined(__aarch64__) || defined(__ARM_ARCH) || defined(STRATA_CPU_ARM64)
// arm64: no x86 intrinsics to include (the cpuid probes below are compiled out; on Windows ARM64 clang-cl also
// defines _MSC_VER, which is why the _MSC_VER branch is not enough)
#elif defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#include <fstream>
#include <sstream>

namespace strata::kernels::cpu {
namespace {
ExpertLayout g_layout;
}

const ExpertLayout& expert_layout() { return g_layout; }

int cpu_isa_cap() {
    // STRATA_FORCE_ISA (tests, the experimental older-CPU builds): the engine's own dispatch acts as if this CPU
    // stopped at that level.  ggml-cpu is not affected: it runs what the build compiled it for.
    static const int cap = [] {
        const char* f = std::getenv("STRATA_FORCE_ISA");
        if (f == nullptr || f[0] == '\0') return 3;
        const std::string v(f);
        if (v == "avx2") return 2;
        if (v == "avx") return 1;
        if (v == "sse" || v == "sse4.2" || v == "none") return 0;
        std::fprintf(stderr, "strata: STRATA_FORCE_ISA=%s is not avx2, avx or sse; ignored\n", f);
        return 3;
    }();
    return cap;
}

#if !defined(__x86_64__) && !defined(__i386__) && !defined(_M_X64) && !defined(_M_IX86)
bool cpu_avx512_ok() { return false; }
bool cpu_avx512bw_ok() { return false; }
bool cpu_avx2_ok() { return false; }
bool cpu_avx1_ok() { return false; }
bool cpu_sse42_ok() { return false; }
bool cpu_gather_fast() { return false; }
bool cpu_gather_fast_here() { return false; }
bool cpu_avxvnni_ok() { return false; }
std::string cpu_name() {
#if defined(__APPLE__)
    return "unknown";   // as before: the Metal build's cpuid shim read all zeros
#else
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("model name", 0) == 0 || line.rfind("Model", 0) == 0) {
            const size_t c = line.find(':');
            if (c != std::string::npos) {
                const size_t b = line.find_first_not_of(" \t", c + 1);
                if (b != std::string::npos) return line.substr(b);
            }
        }
    }
#if defined(__aarch64__)
    return "aarch64";
#else
    return "unknown";
#endif
#endif  // __APPLE__
}
#else
bool cpu_avx512_ok() {
    static const bool ok = [] {
        if (const char* f = std::getenv("STRATA_FORCE_AVX2"); f != nullptr && f[0] == '1') return false;
        if (cpu_isa_cap() < 3) return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        if (!((r[2] >> 27) & 1u)) return false;             // OSXSAVE
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
#if !defined(__aarch64__)   // arm64 (the Metal build): no XCR0; cpuid reads zero there, so this is never reached
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
#endif
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0xE6) != 0xE6) return false;          // the OS saves the AVX-512 state
        cpuid(7, 0);
        const unsigned ebx = r[1], ecx = r[2];
        return ((ebx >> 16) & 1u) && ((ebx >> 30) & 1u) && ((ebx >> 31) & 1u) && ((ecx >> 11) & 1u) && ((ecx >> 1) & 1u);
    }();
    return ok;
}

bool cpu_avx512bw_ok() {
    static const bool ok = [] {
        if (const char* f = std::getenv("STRATA_FORCE_AVX2"); f != nullptr && f[0] == '1') return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        if (!((r[2] >> 27) & 1u)) return false;             // OSXSAVE
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
#if !defined(__aarch64__)   // arm64 (the Metal build): no XCR0; cpuid reads zero there, so this is never reached
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
#endif
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0xE6) != 0xE6) return false;          // the OS saves the AVX-512 state
        cpuid(7, 0);
        const unsigned ebx = r[1];
        return ((ebx >> 16) & 1u) && ((ebx >> 17) & 1u) && ((ebx >> 30) & 1u) && ((ebx >> 31) & 1u);
    }();
    return ok;
}

bool cpu_avx2_ok() {
    static const bool ok = [] {
        if (cpu_isa_cap() < 2) return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        const unsigned ecx1 = r[2];
        // FMA (12), OSXSAVE (27), AVX (28), F16C (29)
        if (!((ecx1 >> 12) & 1u) || !((ecx1 >> 27) & 1u) || !((ecx1 >> 28) & 1u) || !((ecx1 >> 29) & 1u)) return false;
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
#if !defined(__aarch64__)   // arm64 (the Metal build): no XCR0; cpuid reads zero there, so this is never reached
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
#endif
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0x6) != 0x6) return false;            // the OS saves the SSE and AVX state
        cpuid(7, 0);
        return ((r[1] >> 5) & 1u) != 0;                    // AVX2
    }();
    return ok;
}

bool cpu_avx1_ok() {
    // AVX (Sandy Bridge, 2011): AVX + OSXSAVE with the OS saving the YMM state; FMA, F16C and AVX2 not needed.
    // From the Strata_Dirigo fork (rwkeyes).
    static const bool ok = [] {
        if (cpu_isa_cap() < 1) return false;
        unsigned r[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
        int x[4];
        __cpuidex(x, 1, 0);
        for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
        __cpuid_count(1, 0, r[0], r[1], r[2], r[3]);
#endif
        if (!((r[2] >> 27) & 1u) || !((r[2] >> 28) & 1u)) return false;   // OSXSAVE, AVX
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
#if !defined(__aarch64__)   // arm64 (the Metal build): no XCR0; cpuid reads zero there, so this is never reached
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
#endif
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        return (xcr0 & 0x6) == 0x6;
    }();
    return ok;
}

bool cpu_sse42_ok() {
    static const bool ok = [] {
        unsigned r[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
        int x[4];
        __cpuidex(x, 1, 0);
        for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
        __cpuid_count(1, 0, r[0], r[1], r[2], r[3]);
#endif
        return ((r[2] >> 20) & 1u) && ((r[2] >> 23) & 1u);   // SSE4.2, POPCNT
    }();
    return ok;
}

#endif  // x86 cpu probes

const char* isa_floor_build() {
#if defined(STRATA_ISA_FLOOR_AVX)
    return "avx";
#elif defined(STRATA_ISA_FLOOR_NONE)
    return "sse4.2";
#else
    return "";
#endif
}

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
namespace {
void cpuid_regs(unsigned leaf, unsigned sub, unsigned r[4]) {
#if defined(_MSC_VER)
    int x[4];
    __cpuidex(x, (int) leaf, (int) sub);
    for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}
}  // namespace
#endif  // x86 cpuid helper

int iq256_gather_setting() {
    static const int s = [] {
        const char* v = std::getenv("STRATA_IQ256_GATHER");
        if (v == nullptr || v[0] == '\0') return -1;
        return std::atoi(v) != 0 ? 1 : 0;
    }();
    return s;
}

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
bool cpu_gather_fast() {
    static const bool ok = [] {
        if (cpu_isa_cap() < 3) return false;   // STRATA_FORCE_ISA: as on a CPU that stops at AVX2
        unsigned r[4];
        cpuid_regs(0, 0, r);
        const unsigned max_leaf = r[0];
        if (!(r[1] == 0x756e6547u && r[3] == 0x49656e69u && r[2] == 0x6c65746eu)) return false;   // "GenuineIntel"
        if (max_leaf < 7) return false;
        cpuid_regs(7, 0, r);
        if (r[0] < 1) return false;            // no sub-leaf 1
        cpuid_regs(7, 1, r);
        if (!((r[0] >> 4) & 1u)) return false; // AVX-VNNI: Alder Lake, Sapphire Rapids and later
        // The E-core-only parts with AVX-VNNI are not hybrid, so the core type below may not tell them apart: Alder
        // Lake-N / Twin Lake (6/BEh), Grand Ridge (6/B6h), Sierra Forest (6/AFh), Clearwater Forest (6/DDh).
        cpuid_regs(1, 0, r);
        const unsigned family = (r[0] >> 8) & 0xf, model = ((r[0] >> 4) & 0xf) | (((r[0] >> 16) & 0xf) << 4);
        if (family == 6 && (model == 0xBE || model == 0xB6 || model == 0xAF || model == 0xDD)) return false;
        return true;
    }();
    return ok;
}

bool cpu_gather_fast_here() {
    if (!cpu_gather_fast()) return false;
    static const unsigned max_leaf = [] { unsigned r[4]; cpuid_regs(0, 0, r); return r[0]; }();
    static const bool hybrid = [] { unsigned r[4]; cpuid_regs(7, 0, r); return ((r[3] >> 15) & 1u) != 0; }();
    // A CPUID can cost a microsecond under a hypervisor (Windows with VBS), so once per thread.
    thread_local int here = -1;
    if (here < 0) {
        unsigned type = 0;
        if (max_leaf >= 0x1A) {
            unsigned r[4];
            cpuid_regs(0x1A, 0, r);
            type = r[0] >> 24;
        }
        // 40h a performance core, 20h an E-core; no core type is a non-hybrid part (P-cores, see cpu_gather_fast)
        here = type == 0x40 || (type == 0 && !hybrid) ? 1 : 0;
    }
    return here == 1;
}

bool cpu_avxvnni_ok() {
    static const bool ok = [] {
        if (const char* v = std::getenv("STRATA_NO_AVXVNNI"); v != nullptr && std::atoi(v) != 0) return false;
        if (isa_floor_build()[0] != '\0' || cpu_isa_cap() < 3 || !cpu_avx2_ok()) return false;
        unsigned r[4];
        cpuid_regs(0, 0, r);
        if (r[0] < 7) return false;
        cpuid_regs(7, 0, r);
        if (r[0] < 1) return false;            // no sub-leaf 1
        cpuid_regs(7, 1, r);
        return ((r[0] >> 4) & 1u) != 0;        // AVX-VNNI; VEX-encoded, so the OS state is AVX's (cpu_avx2_ok)
    }();
    return ok;
}

std::string cpu_name() {
    unsigned r[12] = {};
#if defined(_MSC_VER)
    int x[4];
    __cpuid(x, (int) 0x80000000u);
    if ((unsigned) x[0] < 0x80000004u) return "unknown";
    for (unsigned i = 0; i < 3; ++i) {
        __cpuid(x, (int) (0x80000002u + i));
        for (int j = 0; j < 4; ++j) r[i * 4 + j] = (unsigned) x[j];
    }
#else
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid(0x80000000u, a, b, c, d);
    if (a < 0x80000004u) return "unknown";
    for (unsigned i = 0; i < 3; ++i) __cpuid(0x80000002u + i, r[i * 4], r[i * 4 + 1], r[i * 4 + 2], r[i * 4 + 3]);
#endif
    char s[49] = {};
    std::memcpy(s, r, 48);
    std::string name(s);
    const size_t b0 = name.find_first_not_of(' '), b1 = name.find_last_not_of(' ');
    return b0 == std::string::npos ? std::string("unknown") : name.substr(b0, b1 - b0 + 1);
}
#endif  // x86 cpuid probes

// ---- the AVX-512 probe and the oracle flag (#795) ----
//
// expert.cpp and iq_avx512.cpp are compiled for AVX-512, and a TU compiled that way may use AVX-512 in ANY of its
// code.  The code that decides whether this CPU has AVX-512 - and the startup flag every CPU, AVX2-only ones
// included, runs before that decision - therefore lives here, in a file compiled for the x86-64 baseline.
namespace {
std::atomic<bool> g_oracle_q8_0{false};
}

void expert_set_oracle_q8_0(bool enabled) { g_oracle_q8_0.store(enabled, std::memory_order_relaxed); }

bool expert_oracle_q8_0_enabled() { return g_oracle_q8_0.load(std::memory_order_relaxed); }

const char* CpuFeatures::reason() const {
    if (usable()) return "ok";
    // Named individually: "AVX-512 not supported" sends a user looking for a new CPU when the machine may have
    // AVX-512F and be missing only VNNI, which is a much narrower and more explicable gap.
    static char buf[160];
    std::snprintf(buf, sizeof buf, "missing %s%s%s%s%s", avx512f ? "" : "AVX512F ",
                  avx512bw ? "" : "AVX512BW ", avx512vl ? "" : "AVX512VL ",
                  avx512_vnni ? "" : "AVX512-VNNI ", avx512_vbmi ? "" : "AVX512-VBMI");
    return buf;
}

CpuFeatures cpu_features() {
    CpuFeatures f;
#if !defined(__x86_64__) && !defined(__i386__) && !defined(_M_X64) && !defined(_M_IX86)
    return f;   // the AVX-512 kernels are x86-only: no AVX-512 on this architecture
#else
    int reg[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
    __cpuid(reg, 0);
    if (reg[0] < 7) return f;
    __cpuidex(reg, 7, 0);
#else
    unsigned r[4] = {0, 0, 0, 0};
    __cpuid_count(0, 0, r[0], r[1], r[2], r[3]);
    if (r[0] < 7) return f;
    __cpuid_count(7, 0, r[0], r[1], r[2], r[3]);
    for (int i = 0; i < 4; ++i) reg[i] = (int) r[i];
#endif
    const unsigned ebx = (unsigned) reg[1], ecx = (unsigned) reg[2];
    f.avx512f = (ebx >> 16) & 1u;
    f.avx512bw = (ebx >> 30) & 1u;
    f.avx512vl = (ebx >> 31) & 1u;
    f.avx512_vnni = (ecx >> 11) & 1u;
    f.avx512_vbmi = (ecx >> 1) & 1u;
    return f;
#endif
}

void cpu_require_expert_support() {
    const CpuFeatures f = cpu_features();
    if (f.usable()) return;
    std::fprintf(stderr,
                 "strata: this CPU cannot run the expert kernel: %s.\n"
                 "        The pack in --pack is the AVX-512 Q2_0 pack (its folder has no native_experts.txt): its CPU kernel needs\n"
                 "        AVX512-VNNI and AVX512-VBMI (Intel Ice Lake / AMD Zen 4 or newer).\n"
                 "        An i-quant (IQ3_XXS, IQ3_S, ...), NVFP4 or Unsloth pack has native_experts.txt in its folder and runs on an\n"
                 "        AVX2 CPU as well: build that pack (setup does it on a CPU without AVX-512) and give its folder as --pack.\n"
                 "        The scalar fallback exists for tests only and is far too slow to decode with.\n",
                 f.reason());
    std::exit(1);
}

void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1) {
    if (cpu_avx512_ok()) q2_0_gguf_rows_multi(w, row_bytes, nblocks, a, nt, out, r0, r1);
    else if (cpu_avx2_ok()) q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a, nt, out, r0, r1);
    else q2_0_gguf_rows_multi_avx1(w, row_bytes, nblocks, a, nt, out, r0, r1);   // issue #1699: the third rung
}

void q2_rows_any_legacy(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                        int r0, int r1) {
    if (cpu_avx512_ok()) q2_0_gguf_rows_multi(w, row_bytes, nblocks, a, nt, out, r0, r1);
    else if (cpu_avx2_ok()) q2_0_gguf_rows_multi_avx2_legacy(w, row_bytes, nblocks, a, nt, out, r0, r1);
    else q2_0_gguf_rows_multi_avx1_legacy(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

void act_quant_any(const float* x, int n, ActQ& a) {
    if (cpu_avx512_ok()) act_quant_q8_1(x, n, a);
    else if (cpu_avx2_ok()) act_quant_q8_1_avx2(x, n, a);
    else act_quant_q8_1_avx1(x, n, a);   // issue #1699
}

#if !defined(STRATA_NATIVE_EXPERTS)
// Without ggml-cpu no native pack loads (expert_layout_load refuses), so these are never reached.
bool native_experts_available() noexcept { return false; }
bool native_fmt(int, int, int64_t, int64_t, NativeFmt&, std::string& err) { err = "built without native experts"; return false; }
void native_quant_act(const NativeFmt&, const float*, void*) { std::abort(); }
void native_quant_h(const NativeFmt&, const float*, void*) { std::abort(); }
int native_gu_mt_min(int) { return 2; }
void native_gu_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) { std::abort(); }
void native_down_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) { std::abort(); }
void native_gu_rows_invariant(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) { std::abort(); }
void native_down_rows_invariant(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) { std::abort(); }
#endif

bool expert_layout_load(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    ExpertLayout L;
    L.n_layers = n_layers;
    L.n_expert = n_expert;
    std::ifstream in(pack_dir + "/native_experts.txt");
    if (!in) {
        L.total = (uint64_t) n_layers * (uint64_t) n_expert * (uint64_t) BLOB;
        hadamard_set({});
        g_layout = L;
        return true;
    }
#if !defined(STRATA_NATIVE_EXPERTS)
    err = "this pack has native (IQ) experts but the engine was built without STRATA_NATIVE_EXPERTS";
    return false;
#else
    L.native = true;
    int64_t expert_embd = H, expert_ff = FF;
    L.fmt.resize((size_t) n_layers);
    L.offset.assign((size_t) n_layers, ~0ull);
    L.bytes.assign((size_t) n_layers, 0);
    L.max_blob = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            if (!line.empty() && line[0] == '#') {
                // "# strata native experts vN: ...".  A version this engine does not know may mean columns it
                // would misread, so it is refused rather than parsed as far as the known columns go.
                static const char tag[] = "# strata native experts v";
                if (line.compare(0, sizeof tag - 1, tag) == 0) {
                    L.version = std::atoi(line.c_str() + sizeof tag - 1);
                    if (L.version > kExpertLayoutVersion) {
                        err = "native_experts.txt is v" + std::to_string(L.version) + "; this engine reads up to v" +
                              std::to_string(kExpertLayoutVersion) + " (the pack was written by a newer tools/"
                              "iq_pack.py: update the engine, or repack with this one)";
                        return false;
                    }
                }
                // v3 packs record their expert count in the header; a pruned model (GSQ-RCO Coder) ships
                // fewer experts than the canonical geometry the caller passes, which is a compile-time
                // default, so the header wins.
                const size_t at = line.find("(n_expert ");
                if (at != std::string::npos) L.n_expert = std::atoll(line.c_str() + at + 10);
                const size_t eb = line.find("expert_embd ");
                if (eb != std::string::npos) expert_embd = std::atoll(line.c_str() + eb + 12);
                const size_t ef = line.find("expert_ff ");
                if (ef != std::string::npos) expert_ff = std::atoll(line.c_str() + ef + 10);
            }
            continue;
        }
        std::istringstream ss(line);
        long long l = -1, gt = -1, dt = -1;
        unsigned long long off = 0, blob = 0, go = 0, uo = 0, dox = 0;
        if (!(ss >> l >> gt >> dt >> off >> blob) || l < 0 || l >= n_layers) {
            err = "native_experts.txt: a malformed line: " + line;
            return false;
        }
        NativeFmt f;
        if (!native_fmt((int) gt, (int) dt, expert_embd, expert_ff, f, err)) return false;
        if (f.bytes != blob) {
            err = "native_experts.txt: layer " + std::to_string(l) + " blob is " + std::to_string(blob) +
                  " B but its formats make " + std::to_string(f.bytes);
            return false;
        }
        if (ss >> go >> uo >> dox) {   // v2 lines: the GGUF offsets
            if (L.gguf_off.empty()) L.gguf_off.assign((size_t) (3 * n_layers), 0);
            L.gguf_off[(size_t) (3 * l)] = go;
            L.gguf_off[(size_t) (3 * l + 1)] = uo;
            L.gguf_off[(size_t) (3 * l + 2)] = dox;
            std::string file;             // v3: the shard that holds this layer (a file name beside --native)
            if (ss >> file) {
                if (L.gguf_file.empty()) L.gguf_file.assign((size_t) (3 * n_layers), std::string());
                // One name covers all three roles.  When a shard boundary falls inside a layer the packer writes
                // "gate,up,down" instead (v4), an empty field meaning the --native shard.  A GGUF file name has no
                // comma, so the split is unambiguous; an engine older than v4 fails to open such a "file" loudly.
                // The comma form is from #255 (gopinath87607).
                std::vector<std::string> parts;
                for (size_t from = 0;;) {
                    const size_t comma = file.find(',', from);
                    parts.push_back(file.substr(from, comma == std::string::npos ? comma : comma - from));
                    if (comma == std::string::npos) break;
                    from = comma + 1;
                }
                if (parts.size() == 1) {
                    const std::string one = parts[0];   // not parts.assign(3, parts[0]): that aliases the element
                    parts.assign(3, one);                //   the assignment is about to overwrite
                }
                std::string extra;
                if (parts.size() != 3 || (ss >> extra)) {
                    err = "native_experts.txt: layer " + std::to_string(l) + ": the shard column is one name or "
                          "gate,up,down, not: " + line;
                    return false;
                }
                for (size_t r = 0; r < 3; ++r) L.gguf_file[(size_t) (3 * l) + r] = parts[r];
            }
        }
        L.fmt[(size_t) l] = f;
        L.offset[(size_t) l] = off;
        L.bytes[(size_t) l] = blob;
        if (blob > L.max_blob) L.max_blob = blob;
    }
    uint64_t at = 0;
    for (int64_t l = 0; l < n_layers; ++l) {
        if (L.offset[(size_t) l] != at) {
            err = "native_experts.txt: layer " + std::to_string(l) + " is missing or not contiguous";
            return false;
        }
        at += L.bytes[(size_t) l] * (uint64_t) L.n_expert;
    }
    L.total = at;
    // v5: Agention's Hadamard-folded experts (prism.hadamard, hadamard.hpp).  The pack must say which layers are
    // rotated and with what signs; a v5 pack without hadamard.txt (or the reverse) is refused, never run unrotated.
    {
        std::ifstream probe(pack_dir + "/hadamard.txt");
        const bool have = (bool) probe;
        if (L.version == 5 && !have) {   // v6 latent A-only packs carry no hadamard.txt (their expert rows are latent, not Hadamard-folded)
            err = "native_experts.txt v" + std::to_string(L.version) + " declares Hadamard-folded experts, but " + pack_dir + "/hadamard.txt is missing";
            return false;
        }
        if (have && L.version < 5) {
            err = pack_dir + "/hadamard.txt exists but native_experts.txt is v" + std::to_string(L.version) +
                  ": the pack is inconsistent (repack with tools/iq_pack.py)";
            return false;
        }
        for (int64_t l = 0; l < n_layers; ++l) L.fmt[(size_t) l].layer = (int) l;
        HadamardSpec h;
        if (have && !hadamard_parse(pack_dir + "/hadamard.txt", n_layers, h, err)) return false;
        hadamard_set(std::move(h));
        const HadamardSpec& hs = hadamard_spec();
        for (int64_t l = 0; hs.any() && l < n_layers; ++l) {
            NativeFmt& f = L.fmt[(size_t) l];
            f.had_block = hs.block;
            f.had_gu = hs.gu(l);
            f.had_d = hs.down(l);
            if ((f.had_gu && f.n_embd % hs.block) || (f.had_d && f.n_ff % hs.block)) {
                err = "hadamard.txt: block " + std::to_string(hs.block) + " does not divide layer " +
                      std::to_string(l) + "'s expert widths";
                return false;
            }
            // the Q2_0 rows quantize their activations with Strata's own act_quant_any, which does not rotate
            if ((f.had_gu && f.gu_type == 42) || (f.had_d && f.d_type == 42)) {
                err = "hadamard.txt: layer " + std::to_string(l) + " folds a Q2_0 expert, which this engine cannot rotate";
                return false;
            }
            f.gu_signs = f.had_gu ? hadamard_signs((int) f.n_embd) : nullptr;
            f.d_signs = f.had_d ? hadamard_signs((int) f.n_ff) : nullptr;
            // explicit signs are per width: a width without a vector would silently mean identity signs
            if (!hs.signs.empty() && ((f.had_gu && !f.gu_signs) || (f.had_d && !f.d_signs))) {
                err = "hadamard.txt: no sign vector for layer " + std::to_string(l) + "'s expert width";
                return false;
            }
        }
    }
    // Q3_K + shared column scales (make_latent_a_q3k.py): scales/l{l}.f32 beside the manifest holds s, the gate/up
    // rows are A*diag(s), and the D stage's xl rows are multiplied by 1/s where they are produced (verify.cpp /
    // prefill.cpp, lat_colscale_rows) so every consumer - the GPU experts' q8_1, act0, the CPU pool's f32
    // doorbell - sees scaled rows; native_quant_act does NOT multiply. All or none: a half-scaled pack is refused
    // rather than run wrong.
    {
        std::ifstream probe(pack_dir + "/scales/l0.f32");
        if (probe) {
            static std::vector<float> inv;   // outlives g_layout (its copied NativeFmt points into this)
            inv.clear();
            inv.reserve((size_t) n_layers * (size_t) expert_embd);
            for (int64_t l = 0; l < n_layers; ++l) {
                const int64_t n = L.fmt[(size_t) l].n_embd;
                if (n != expert_embd) {   // the reserve above assumes one width; a mixed pack would realloc
                    err = "scales: layer " + std::to_string(l) + " has n_embd " + std::to_string(n) +
                          ", the manifest says " + std::to_string(expert_embd);
                    return false;
                }
                const std::string p = pack_dir + "/scales/l" + std::to_string(l) + ".f32";
                std::ifstream f(p, std::ios::binary);
                if (!f) { err = "scales: " + p + " is missing (all-or-none)"; return false; }
                std::vector<float> s((size_t) n);
                f.read((char*) s.data(), (std::streamsize) (s.size() * sizeof(float)));
                if (f.gcount() != (std::streamsize) (s.size() * sizeof(float))) {
                    err = "scales: " + p + " is short";
                    return false;
                }
                L.fmt[(size_t) l].gu_colscale = inv.data() + inv.size();
                for (float v : s) inv.push_back(v > 0.0f ? 1.0f / v : 1.0f);
            }
        }
    }
    g_layout = L;
    return true;
#endif
}

}  // namespace strata::kernels::cpu

namespace strata::kernels {   // lat_gu_colscale_host: the GPU-side colscale multiplier's data source (iq_kernels.cu)

const float* lat_gu_colscale_host(int layer, int* width) {
    const cpu::ExpertLayout& L = cpu::expert_layout();
    if (width != nullptr) *width = 0;
    if (layer < 0 || (size_t) layer >= L.fmt.size()) return nullptr;
    const float* s = L.fmt[(size_t) layer].gu_colscale;
    if (s != nullptr && width != nullptr) *width = (int) L.fmt[(size_t) layer].n_embd;
    return s;
}

}  // namespace strata::kernels
