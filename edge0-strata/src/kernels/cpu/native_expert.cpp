// Native (GGUF-form) experts: ggml activation quantizers and format-specific CPU row kernels.
// Default keeps the existing dispatch. WindowInvariant preserves that dispatch's single-token arithmetic
// while retaining shared weight decoding where the grouped kernel gives the same result.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace strata::kernels::cpu {
namespace {

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

// Do the trellis rows take Strata's own kernels (tq_cpu.hpp)?  Always, unless STRATA_NO_TQ_KERNELS=1 asks for ggml-cpu's
// vec_dot AND this ggml has the type (the agentionai fork; an upstream ggml does not, and must never be asked about it).
bool tq_rows_own(int type) {
    if (!tq_supported(type)) return false;
    if (tq_kernels_enabled()) return true;
    return type >= (int) GGML_TYPE_COUNT || traits(type) == nullptr || traits(type)->vec_dot == nullptr;
}

// Window-invariant arithmetic (NativeArithmetic::WindowInvariant): the reference rows below reproduce ggml-cpu's one-token dot bit for bit
// only if that ggml was built to sum the same way (contraction of the scalar tail into an FMA, for one, depends on the build). So each
// reference is checked once against the linked ggml on synthetic blocks, odd block counts included, and a mismatch sends the invariant
// path to ggml's own one-token dot per token (slower, never different).
bool iq4nl_reference_ok() {
    static const bool ok = [] {
        const ggml_type_traits_cpu* tr = traits(GGML_TYPE_IQ4_NL);
        const ggml_type_traits_cpu* t8 = traits(GGML_TYPE_Q8_0);
        if (tr == nullptr || tr->vec_dot == nullptr || t8 == nullptr || t8->from_float == nullptr) return false;
        uint32_t seed = 4242u;
        auto rnd = [&seed] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
        for (const int n : {64, 96, 160, 640}) {   // even and odd numbers of blocks: the odd one ends in the scalar tail
            constexpr int rows = 16;
            const size_t rb = ggml_row_size(GGML_TYPE_IQ4_NL, (int64_t) n), nb = (size_t) n / 32, bb = rb / nb;
            std::vector<uint8_t> w((size_t) rows * rb), act(ggml_row_size(GGML_TYPE_Q8_0, (int64_t) n));
            for (auto& b : w) b = (uint8_t) rnd();
            for (int r = 0; r < rows; ++r)
                for (size_t k = 0; k < nb; ++k) { uint8_t* d = &w[(size_t) r * rb + k * bb]; d[0] = 0x66; d[1] = 0x2e; }
            std::vector<float> x((size_t) n);
            for (auto& v : x) v = ((float) (rnd() % 2001) - 1000.f) * 0.001f;
            t8->from_float(x.data(), act.data(), n);
            std::vector<float> got((size_t) rows), want((size_t) rows);
            const void* hp[1] = {act.data()};
            float* op[1] = {got.data()};
            iq4nl_reference_rows(w.data(), rb, n, hp, 1, op, 0, rows);
            for (int r = 0; r < rows; ++r) tr->vec_dot(n, &want[(size_t) r], 0, w.data() + (size_t) r * rb, 0, act.data(), 0, 1);
            if (std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) != 0) {
                std::fprintf(stderr, "strata: the window-invariant IQ4_NL rows differ from this ggml-cpu's dot (n=%d): its dot runs per token instead\n", n);
                return false;
            }
        }
        return true;
    }();
    return ok;
}

// The same check for the AVX-512 accumulation of IQ3_XXS / IQ3_S / IQ2_S gate/up rows (types 18, 21, 22), SwiGLU included.
bool iq512_reference_ok(int type) {
    static std::mutex mu;
    static std::map<int, bool> memo;
    std::lock_guard<std::mutex> lk(mu);
    if (const auto it = memo.find(type); it != memo.end()) return it->second;
    bool ok = false;
    const ggml_type_traits_cpu* tr = traits(type);
    const ggml_type_traits_cpu* t8 = traits(GGML_TYPE_Q8_K);
    if (tr != nullptr && tr->vec_dot != nullptr && t8 != nullptr && t8->from_float != nullptr) {
        constexpr int n = 2560, rows = 24;
        const size_t rb = ggml_row_size((ggml_type) type, n), nb = (size_t) n / 256, bb = rb / nb;
        std::vector<uint8_t> w((size_t) 2 * rows * rb), act(ggml_row_size(GGML_TYPE_Q8_K, n));
        uint32_t seed = 777u;
        auto rnd = [&seed] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
        for (auto& b : w) b = (uint8_t) rnd();
        for (int r = 0; r < 2 * rows; ++r)
            for (size_t k = 0; k < nb; ++k) { uint8_t* d = &w[(size_t) r * rb + k * bb]; d[0] = 0x66; d[1] = 0x2e; }
        std::vector<float> x(n);
        for (auto& v : x) v = ((float) (rnd() % 2001) - 1000.f) * 0.001f;
        t8->from_float(x.data(), act.data(), n);
        std::vector<float> got((size_t) rows), want((size_t) rows);
        const void* ap[1] = {act.data()};
        float* fp[1] = {got.data()};
        iq512_reference_gu_rows(type, w.data(), rb, (size_t) rows * rb, n, ap, 1, fp, 0, rows);
        for (int r = 0; r < rows; ++r) {
            float g = 0.f, u = 0.f;
            tr->vec_dot(n, &g, 0, w.data() + (size_t) r * rb, 0, act.data(), 0, 1);
            tr->vec_dot(n, &u, 0, w.data() + (size_t) rows * rb + (size_t) r * rb, 0, act.data(), 0, 1);
            want[(size_t) r] = (g / (1.f + std::exp(-g))) * u;
        }
        ok = std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0;
        if (!ok) std::fprintf(stderr, "strata: the window-invariant 512-bit rows of type %d differ from this ggml-cpu's dot: its dot runs per token instead\n", type);
    }
    memo[type] = ok;
    return ok;
}

// 0.2.5: the 512-bit exact rows (iq512x_rows) take one token of an IQ3_XXS / IQ3_S expert only if this build's ggml-cpu dot gives
// the same bits on synthetic blocks of that type (a ggml built for AVX-512 may carry another summation order): checked once a type.
bool x512_one_ok(int type) {
    static std::mutex mu;
    static std::map<int, bool> memo;
    std::lock_guard<std::mutex> lk(mu);
    if (const auto it = memo.find(type); it != memo.end()) return it->second;
    bool ok = false;
    const ggml_type_traits_cpu* tr = traits(type);
    const ggml_type_traits_cpu* t8 = traits(GGML_TYPE_Q8_K);
    if (tr != nullptr && tr->vec_dot != nullptr && t8 != nullptr && t8->from_float != nullptr && iq512x_supported(type)) {
        constexpr int n = 2560, rows = 24;
        const size_t rb = ggml_row_size((ggml_type) type, n), nb = (size_t) n / 256, bb = rb / nb;
        std::vector<uint8_t> w((size_t) rows * rb), act(ggml_row_size(GGML_TYPE_Q8_K, n));
        uint32_t seed = 12345u;
        auto rnd = [&seed] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
        for (auto& b : w) b = (uint8_t) rnd();
        for (int r = 0; r < rows; ++r)
            for (size_t k = 0; k < nb; ++k) { uint8_t* d = &w[(size_t) r * rb + k * bb]; d[0] = 0x66; d[1] = 0x2e; }   // d = ~0.1 (f16)
        std::vector<float> x(n);
        for (auto& v : x) v = ((float) (rnd() % 2001) - 1000.f) * 0.001f;
        t8->from_float(x.data(), act.data(), n);
        std::vector<float> got((size_t) rows), want((size_t) rows);
        const void* ap[1] = {act.data()};
        float* op[1] = {got.data()};
        iq512x_rows(type, w.data(), rb, n, ap, 1, op, 0, rows);
        for (int r = 0; r < rows; ++r) tr->vec_dot(n, &want[(size_t) r], 0, w.data() + (size_t) r * rb, 0, act.data(), 0, 1);
        ok = std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0;
        if (!ok) std::fprintf(stderr, "strata: the 512-bit exact one-token rows of type %d differ from this ggml-cpu's dot: not used\n", type);
    }
    memo[type] = ok;
    return ok;
}

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

}  // namespace

bool native_experts_available() noexcept { return true; }

// The "is this format handled by that kernel" questions, answered here and not next to the kernels: iq_avx512.cpp is
// compiled for AVX-512 and iq_avx2.cpp / kq_avx2.cpp for AVX2, and native_gu_rows asks these on every CPU before it
// has checked what the CPU can run.  A function in a wide-ISA file may use that ISA anywhere in its body, so an
// answer that is only a comparison must not come from one (#795).
bool iq512_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22;
}

bool iq256_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 23;
}

bool kq256_supported(int type) noexcept { return type == 12 || type == 7 || type == 8; }

bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err) {
    init_once();
    // The trellis types (Gyro, ggml types 144-146) are not in a ggml built from upstream (and an upstream ggml indexes its
    // type table by the id, so asking it about 144 reads past the table): their geometry and activation format (Q8_0)
    // are known here, and their rows run on tq_cpu.hpp.
    const bool tq_g = tq_supported(gu_type), tq_d = tq_supported(d_type);
    const ggml_type_traits_cpu* tg = tq_g ? nullptr : traits(gu_type);
    const ggml_type_traits_cpu* td = tq_d ? nullptr : traits(d_type);
    if ((!tq_g && (tg == nullptr || tg->vec_dot == nullptr)) || (!tq_d && (td == nullptr || td->vec_dot == nullptr))) {
        err = "native experts: ggml-cpu has no dot product for type " +
              std::to_string(!tq_g && !(tg && tg->vec_dot) ? gu_type : d_type);
        return false;
    }
    const ggml_type vdt_g = tq_g ? GGML_TYPE_Q8_0 : tg->vec_dot_type;
    const ggml_type vdt_d = tq_d ? GGML_TYPE_Q8_0 : td->vec_dot_type;
    const ggml_type_traits_cpu* ag = traits(vdt_g);
    const ggml_type_traits_cpu* ad = traits(vdt_d);
    if (ag == nullptr || ag->from_float == nullptr || ad == nullptr || ad->from_float == nullptr) {
        err = "native experts: ggml-cpu cannot quantize an activation for this layer";
        return false;
    }
    if ((tq_g ? n_embd % kTqBlock : n_embd % ggml_blck_size((ggml_type) gu_type)) ||
        (tq_d ? n_ff % kTqBlock : n_ff % ggml_blck_size((ggml_type) d_type)) ||
        n_embd % ggml_blck_size(vdt_g) || n_ff % ggml_blck_size(vdt_d)) {
        err = "native experts: expert geometry is not whole blocks";
        return false;
    }
    f.gu_type = gu_type;
    f.d_type = d_type;
    f.gu_act = (int) vdt_g;
    f.d_act = (int) vdt_d;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_row = tq_g ? (size_t) (n_embd / kTqBlock) * tq_block_bytes(gu_type) : ggml_row_size((ggml_type) gu_type, n_embd);
    f.d_row = tq_d ? (size_t) (n_ff / kTqBlock) * tq_block_bytes(d_type) : ggml_row_size((ggml_type) d_type, n_ff);
    f.up_off = f.gu_row * (size_t) n_ff;
    f.down_off = 2 * f.up_off;
    f.bytes = f.down_off + f.d_row * (size_t) n_embd;
    // NVFP4 in any projection: one scale tail per blob; a mixed expert (e.g. NVFP4 gate/up + Q8_0 down) has 1.0 for
    // the other projection's scale (tools/iq_pack.py), so every path below multiplies by the tail unconditionally
    if (gu_type == kNvfp4Type || d_type == kNvfp4Type) {
        f.tail_off = f.bytes;
        f.bytes += kNvfp4Tail;
    }
    f.act_bytes = ggml_row_size(vdt_g, n_embd);
    f.h_bytes = ggml_row_size(vdt_d, n_ff);
    if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes) {
        err = "native experts: activation larger than the pool's buffers";
        return false;
    }
    return true;
}

namespace {
// Q8_K (the activations of every i-quant row): ggml-cpu's x86 quantizer is the scalar reference, ~3 us per token
// and layer on the host before the pool can start; q8k_quant_avx2 writes the same bytes.  cpu_avx2_ok() too:
// iq_avx2.cpp is compiled for AVX2 (an AVX-only CPU, or STRATA_FORCE_ISA=avx, keeps ggml's).  STRATA_NO_Q8K_AVX2=1:
// ggml's on any CPU.
bool q8k_avx2(int type) {
    static const bool on = cpu_avx2_ok() && std::getenv("STRATA_NO_Q8K_AVX2") == nullptr;
    return on && type == (int) GGML_TYPE_Q8_K;
}
// the older CPUs' copy (iq_avx1.cpp): byte-identical like the AVX-2 one, and only where AVX2 is absent (same
// policy as STRATA_NO_IQ128 above; STRATA_NO_Q8K_AVX1=1 keeps ggml's scalar reference on those CPUs).
bool q8k_avx1(int type) {
    static const bool on = !cpu_avx2_ok() && cpu_avx1_ok() && std::getenv("STRATA_NO_Q8K_AVX1") == nullptr;
    return on && type == (int) GGML_TYPE_Q8_K;
}
}  // namespace

// STRATA_ACT_DUMP=<prefix> (research, off by default): the second moments of the vectors that enter the CPU experts, in the
// basis the experts use (after the Hadamard rotation of a folded layer): per layer the gate/up input (the token's hidden
// state, once per token) and the down input (the SwiGLU output, once per routed expert).  STRATA_ACT_DUMP_LAYERS=10,20 picks
// the layers (default none), STRATA_ACT_DUMP_COV=1 adds the full covariance.  Files <prefix>_L<l>_<gu|dn>.bin: int64 n,
// int64 count, double sum x^2 [n], (double sum x x^T [n*n]); rewritten every 2000 records and at exit.
namespace {
struct ActAcc {
    int64_t n = 0, count = 0;
    std::vector<double> sq, cov;
};
struct ActDump {
    bool on = false, cov = false;
    std::string prefix;
    std::vector<int> layers;
    std::map<std::pair<int, int>, ActAcc> acc;
    std::mutex mu;
    int64_t records = 0;
    ActDump() {
        const char* p = std::getenv("STRATA_ACT_DUMP");
        if (p == nullptr || *p == 0) return;
        prefix = p;
        cov = std::getenv("STRATA_ACT_DUMP_COV") != nullptr && std::atoi(std::getenv("STRATA_ACT_DUMP_COV")) != 0;
        if (const char* l = std::getenv("STRATA_ACT_DUMP_LAYERS")) {
            std::string s(l);
            size_t at = 0;
            while (at < s.size()) {
                layers.push_back(std::atoi(s.c_str() + at));
                const size_t c = s.find(',', at);
                if (c == std::string::npos) break;
                at = c + 1;
            }
        }
        on = !layers.empty();
        if (on) std::atexit([] { act_dump_flush(); });
    }
    static void act_dump_flush();
};
ActDump& act_dump() {
    static ActDump* d = new ActDump;   // never destroyed: the atexit flush must still find it
    return *d;
}
void ActDump::act_dump_flush() {
    ActDump& d = act_dump();
    std::lock_guard<std::mutex> lk(d.mu);
    for (auto& kv : d.acc) {
        const std::string path = d.prefix + "_L" + std::to_string(kv.first.first) + (kv.first.second ? "_dn.bin" : "_gu.bin");
        FILE* f = std::fopen((path + ".tmp").c_str(), "wb");
        if (!f) continue;
        std::fwrite(&kv.second.n, 8, 1, f);
        std::fwrite(&kv.second.count, 8, 1, f);
        std::fwrite(kv.second.sq.data(), 8, kv.second.sq.size(), f);
        if (!kv.second.cov.empty()) std::fwrite(kv.second.cov.data(), 8, kv.second.cov.size(), f);
        std::fclose(f);
        std::rename((path + ".tmp").c_str(), path.c_str());
    }
}
void act_dump_record(int layer, int kind, const float* v, int64_t n) {
    ActDump& d = act_dump();
    if (std::find(d.layers.begin(), d.layers.end(), layer) == d.layers.end()) return;
    std::lock_guard<std::mutex> lk(d.mu);
    ActAcc& a = d.acc[{layer, kind}];
    if (a.n == 0) {
        a.n = n;
        a.sq.assign((size_t) n, 0.0);
        if (d.cov) a.cov.assign((size_t) n * (size_t) n, 0.0);
    }
    for (int64_t i = 0; i < n; ++i) a.sq[(size_t) i] += (double) v[i] * v[i];
    if (d.cov)
        for (int64_t i = 0; i < n; ++i) {
            double* row = a.cov.data() + (size_t) i * (size_t) n;
            const double vi = v[i];
            for (int64_t j = 0; j < n; ++j) row[j] += vi * v[j];
        }
    ++a.count;
    if (++d.records % (d.cov ? 400000 : 2000) == 0) {
        d.mu.unlock();
        ActDump::act_dump_flush();
        d.mu.lock();
    }
}
}  // namespace

void native_quant_act(const NativeFmt& f, const float* x, void* dst) {
    if (f.had_gu) {   // Hadamard-folded gate/up: quantize H (s * x), never x itself (hadamard.hpp).
        thread_local std::vector<float> tmp;   // Column-scaled packs need no multiply here: the D stage's rows
        // arrive already scaled (lat_colscale_rows in verify.cpp / prefill.cpp multiplies x by 1/s at the source).
        tmp.assign(x, x + f.n_embd);
        hadamard_rotate(tmp.data(), (int) f.n_embd, f.had_block, f.gu_signs);
        x = tmp.data();
    }
    if (act_dump().on) act_dump_record(f.layer, 0, x, f.n_embd);
    if (q8k_avx2(f.gu_act)) { q8k_quant_avx2(x, dst, f.n_embd); return; }
    if (q8k_avx1(f.gu_act)) { q8k_quant_avx1(x, dst, f.n_embd); return; }
    traits(f.gu_act)->from_float(x, dst, f.n_embd);
}

void native_quant_h(const NativeFmt& f, const float* h, void* dst) {
    if (f.had_d) {
        thread_local std::vector<float> tmp;
        tmp.assign(h, h + f.n_ff);
        hadamard_rotate(tmp.data(), (int) f.n_ff, f.had_block, f.d_signs);
        h = tmp.data();
    }
    if (act_dump().on) act_dump_record(f.layer, 1, h, f.n_ff);
    if (q8k_avx2(f.d_act)) { q8k_quant_avx2(h, dst, f.n_ff); return; }
    if (q8k_avx1(f.d_act)) { q8k_quant_avx1(h, dst, f.n_ff); return; }
    traits(f.d_act)->from_float(h, dst, f.n_ff);
}

int native_gu_mt_min(int gu_type) {
    // #152: from how many tokens the multi-token kernels run (ggml's vec_dot below that).  The default 2 is the
    // measured-fastest rule, but a token's expert rows then round differently alone than in a group, so greedy output
    // can depend on how many drafts a verify window held.  STRATA_IQ_MT_MIN=1 (opt-in, 0.1.30) uses the multi-token
    // kernels for every group: output independent of the drafting, at a measured -1..-3% decode on IQ3_S (AVX-512).
    static const char* env = std::getenv("STRATA_IQ_MT_MIN");
    static const int mt_min = env ? std::atoi(env) : 2;
    // IQ3_S where the AVX-2 kernel gathers its grid (cpu_gather_fast, unless STRATA_IQ256_GATHER=0) on a CPU without
    // AVX-512: there that kernel beats ggml's dot for ONE token too (14900KF P-core, an expert's gate/up rows: 0.34 ->
    // 0.19 ms; its E-cores, which keep the scalar decode: 0.71 -> 0.68), and in decode most CPU experts serve one token
    // of the window.  So IQ3_S takes it for every group, and those rows no longer depend on the drafting.  A machine-
    // wide rule, not a per-core one: every core rounds an expert the same.  Every probe here is in expert_layout.cpp
    // and cpu_avx2_ok() comes first, so no AVX2 code runs before the check.  STRATA_IQ_MT_MIN set keeps its rule.
    // Opt-in (STRATA_IQ3S_MT1=1): it changes a lone token's IQ3_S rounding on those CPUs, so the default stays 0.1.39's.
    static const bool iq3s_one = env == nullptr && std::getenv("STRATA_IQ3S_MT1") != nullptr && cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr &&
                                 !cpu_avx512_ok() && iq256_gather_setting() != 0 && cpu_gather_fast();
    return iq3s_one && gu_type == 21 ? 1 : mt_min;
}

template <NativeArithmetic Arithmetic>
void native_gu_rows_impl(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1) {
    // the multi-token kernels decode the weights once for all tokens: 2.0-2.4x ggml-cpu at three tokens, no faster
    // at one (all are bound by the codebook lookups, ~5 GB/s per core), measured by native_expert_parity.  AVX-512
    // first, then the AVX-2 one (Zen 2/3, Intel 12th-14th gen).  STRATA_NO_IQ512 drops an AVX-512 CPU to the
    // AVX-2 kernel, STRATA_NO_IQ256 drops the AVX-2 kernel; ggml-cpu's single-token vec_dot is reached only with
    // both set (and on a CPU without AVX-512, STRATA_NO_IQ512 changes nothing).
    static const bool avx512 = cpu_avx512_ok() && std::getenv("STRATA_NO_IQ512") == nullptr;
    // STRATA_NO_IQ256 drops the AVX-2 kernel.  cpu_avx2_ok() is ALSO required: iq_avx2.cpp and kq_avx2.cpp
    // are compiled /arch:AVX2 and use AVX2 and FMA3, so calling them on a CPU without either is an
    // illegal instruction, not a slow path.  The gate sat next to `avx512` above, which does test
    // cpu_avx512_ok(), and that asymmetry is what let a pre-Haswell CPU reach iq256_gu_rows.
    // Without it this falls through to ggml-cpu's vec_dot below, which is compiled for whatever
    // baseline the build selected (AVX1 here) and covers the same types - IQ2_XXS and IQ2_S among
    // them.  That path loops over tokens itself, so it is correct for any `nt`, not just one.
    static const bool avx2 = cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr;
    // The older-CPU tier (STRATA_ISA_FLOOR=avx builds): without it every expert on such a CPU runs on ggml-cpu's
    // scalar _generic dots.  Only where AVX2 is absent, so a modern CPU's rounding never moves (and
    // STRATA_NO_IQ256=1 there still means ggml, as before).  iq_avx1.cpp is compiled for AVX and gated on
    // cpu_avx1_ok().  Measured on an E5-2470 v2 (iq_avx1_parity --bench): 1.2-3.1x ggml's dot from two tokens on,
    // at one token ggml's auto-vectorized generic loop matches the 128-bit kernel - so the same #152 mt_min rule
    // as the AVX-2 kernels, and STRATA_IQ_MT_MIN=1 takes the kernel for one token too (its rows are then
    // independent of the drafting: iq_avx1_parity checks every width bit for bit).
    static const bool avx1 = !cpu_avx2_ok() && cpu_avx1_ok() && std::getenv("STRATA_NO_IQ128") == nullptr;
    // The exact rows are ggml-cpu's AVX2 single-token dot bit for bit, so they need ggml-cpu itself built for AVX2 and
    // not AVX-512 (ggml_cpu_has_*): a STRATA_ISA_FLOOR build runs ggml's AVX1/SSE dots, a native build on an AVX-512 CPU
    // run with STRATA_FORCE_AVX2=1 runs its AVX-512 one, and both round differently (#2215's guard).
    static const bool exact_one = [] {
        const char* v = std::getenv("STRATA_CPU_IQ_ALL_EXACT1");
        return v && v[0] == '1' && v[1] == 0 && ggml_cpu_has_avx2() && !ggml_cpu_has_avx512();
    }();
    // only where ggml's dot runs today (below the #152 multi-token rule): with STRATA_IQ_MT_MIN=1 a lone token keeps the
    // multi-token kernel, so a token's rows still do not depend on how many tokens share its expert
    if (exact_one && nt == 1 && nt < native_gu_mt_min(f.gu_type) && avx2 && !cpu_avx512_ok() &&
        (f.gu_type == 16 || f.gu_type == 17 || f.gu_type == 18 || f.gu_type == 21 || f.gu_type == 22)) {
        static std::once_flag announced;
        std::call_once(announced, [] { std::fprintf(stderr, "strata ab: GATE=STRATA_CPU_IQ_ALL_EXACT1 active=1\n"); });
        iq256_gu_rows_exact_one(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act[0], ff[0], r0, r1);
        return;
    }
    // 0.2.5 (AVX-512 CPUs where a gather is cheap: Intel Sapphire Rapids and later): ONE token of an IQ3_XXS / IQ3_S expert on the
    // 512-bit exact rows, ggml-cpu's single-token AVX2 dot bit for bit with the grid words gathered: 1.3x ggml's dot on a Xeon 8468
    // (x512_bench), the same output.  Where the multi-token rule (#152) already sends one token to the multi-token kernel
    // (STRATA_IQ_MT_MIN=1) that stays; STRATA_NO_IQ512X1=1 turns this off; the first use checks the bits against this ggml.
    static const bool x512_one = avx512 && iq512_gather_on() && std::getenv("STRATA_NO_IQ512X1") == nullptr;
    if (x512_one && nt == 1 && nt < native_gu_mt_min(f.gu_type) && f.tail_off == 0 && (f.gu_type == 18 || f.gu_type == 21) &&
        x512_one_ok(f.gu_type)) {
        const int n1 = (int) f.n_embd;
        float g[64], u[64];
        float* gp[1] = {g};
        float* up[1] = {u};
        for (int c0 = r0; c0 < r1; c0 += 64) {
            const int c1 = std::min(r1, c0 + 64);
            iq512x_rows(f.gu_type, blob + (size_t) c0 * f.gu_row, f.gu_row, n1, act, 1, gp, 0, c1 - c0);
            iq512x_rows(f.gu_type, blob + f.up_off + (size_t) c0 * f.gu_row, f.gu_row, n1, act, 1, up, 0, c1 - c0);
            for (int r = c0; r < c1; ++r) ff[0][r] = (g[r - c0] / (1.f + std::exp(-g[r - c0]))) * u[r - c0];
        }
        return;
    }
    const int mt_min = native_gu_mt_min(f.gu_type);   // #152
    const bool singleton_iq = Arithmetic == NativeArithmetic::WindowInvariant && mt_min > 1;
    // Unsloth UD-Q4_K_XL's Q4_K gate/up: the multi-token kernel is bit-exact against ggml's per-token dot (any group
    // size, no #152 rule).  Opt-in, STRATA_KQ256=1: measured no faster in the engine (a window's expert groups hold
    // ~1.4 tokens and the weights stay in L1 across ggml's per-token calls; 1.01-1.13x in native_expert_parity).
    static const bool kq = [] { const char* v = std::getenv("STRATA_KQ256"); return cpu_avx2_ok() && v != nullptr && std::atoi(v) != 0; }();
    // KQ matches the ggml AVX2/FMA dot; other builds keep their own singleton arithmetic.
    if (kq && f.gu_type == 12 && nt >= 2 &&
        (Arithmetic == NativeArithmetic::Default || (ggml_cpu_has_avx2() && ggml_cpu_has_fma()))) {   // one token: ggml's own dot below (the same bits, less overhead)
        kq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        return;
    }
    // The trellis types (TQ2_T / TQK6 / TQK7): Strata's own kernels (tq_cpu.hpp), for every group size - they decode
    // a block once for all the tokens and give each token the same bits whatever the group (no #152 rule needed).
    if (tq_rows_own(f.gu_type)) {
        tq_gu_rows(tq_isa(), f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        return;
    }
    // A format with only an AVX-2 kernel (IQ4_XS, #415) takes it on AVX-2 CPUs only: an AVX-512 CPU keeps ggml-cpu for
    // it, as before (its rows would round differently).  Each kernel only for the formats it implements: falling
    // through an empty switch would leave ff unwritten instead of falling back to ggml-cpu.
    static const bool cpu512 = cpu_avx512_ok();
    if (nt >= mt_min && (iq512_supported(f.gu_type) || (!cpu512 && iq256_supported(f.gu_type)))) {
        if (singleton_iq) {
#if defined(__x86_64__) || defined(_M_X64)
            if (avx512 && ggml_cpu_has_avx2() && ggml_cpu_has_fma() &&
                (f.gu_type == 18 || f.gu_type == 21 || f.gu_type == 22) && iq512_reference_ok(f.gu_type)) {
                iq512_reference_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
                return;
            }
#endif
            // Otherwise use the singleton dot below, keeping its row-major weight reuse across tokens.
        } else {
            if (avx512 && iq512_supported(f.gu_type)) {
                iq512_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
                return;
            }
            if (avx2 && iq256_supported(f.gu_type)) {
                iq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
                return;
            }
        }
    }
    if (!singleton_iq && avx1 && nt >= mt_min && iq128_supported(f.gu_type)) {
        iq128_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        return;
    }
    const int n = (int) f.n_embd;
    // NVFP4: the per-expert global scales from the blob's tail (1 and 1 otherwise). s_down goes on the down output:
    // folded into up it left the hidden near 1e-5, whose Q8_0 block scale is an FP16 subnormal (expert error 2-12%)
    float sg = 1.f, su = 1.f;
    if (f.tail_off) {
        float tail[4];
        std::memcpy(tail, blob + f.tail_off, sizeof tail);
        sg = tail[0];
        su = tail[1];
    }
    // NVFP4 in 512-bit lanes: ggml's arithmetic, blocks decoded once per window; 1.8x at one token, 3.7x at seven
    // (nvfp4_avx512_parity). STRATA_NO_NVFP4_512 falls back to ggml-cpu.
    static const bool nvfp4_512 = cpu_avx512_ok() && std::getenv("STRATA_NO_NVFP4_512") == nullptr;
    if (f.gu_type == kNvfp4Type && nvfp4_512 && nvfp4_512_fits(n)) {
        nvfp4_512_gu_rows(blob, f.gu_row, f.up_off, n, act, nt, ff, r0, r1, sg, su);
        return;
    }
    // Without AVX-512: the same rows in 256-bit lanes, bit-equal to ggml-cpu's AVX2 dot below; 2.0-2.4x it at 4-8
    // tokens, 2x under DRAM load with 6 threads (nvfp4_avx512_parity). STRATA_NO_NVFP4_256 falls back to ggml-cpu.
    static const bool nvfp4_256 = cpu_avx2_ok() && std::getenv("STRATA_NO_NVFP4_256") == nullptr;
    if (f.gu_type == kNvfp4Type && nvfp4_256 && nvfp4_256_fits(n)) {
        nvfp4_256_gu_rows(blob, f.gu_row, f.up_off, n, act, nt, ff, r0, r1, sg, su);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.gu_type)->vec_dot;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * f.gu_row;
        const uint8_t* ur = blob + f.up_off + (size_t) r * f.gu_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, gr, 0, act[t], 0, 1);
            dot(n, &u, 0, ur, 0, act[t], 0, 1);
            g *= sg;
            u *= su;
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}

template <NativeArithmetic Arithmetic>
void native_down_rows_impl(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1) {
    // IQ4_NL down rows: the AVX-2 multi-token kernel decodes the nibbles and absolutises the weights once per
    // block instead of once per token; ggml-cpu's dot is single-token.  STRATA_NO_IQ4NL falls back to it.
    static const bool iq4nl_mt = std::getenv("STRATA_NO_IQ4NL") == nullptr;
    static const int mt_min = [] { const char* e = std::getenv("STRATA_IQ_MT_MIN"); return e ? std::atoi(e) : 2; }();
    static const bool kq = [] { const char* v = std::getenv("STRATA_KQ256"); return v != nullptr && std::atoi(v) != 0; }();
    // Both multi-token kernels below are /arch:AVX2 translation units (kq_avx2.cpp and iq_avx2.cpp),
    // so a CPU without AVX2 has to reach ggml-cpu's vec_dot instead - same reasoning as the gate/up
    // rows above, where `avx512` tested cpu_avx512_ok() and `avx2` did not.
    if (tq_rows_own(f.d_type)) {   // the trellis types, as in native_gu_rows
        tq_rows(tq_isa(), f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    if (cpu_avx2_ok() && kq && nt >= 2 && (f.d_type == 7 || f.d_type == 8) &&
        (Arithmetic == NativeArithmetic::Default || (ggml_cpu_has_avx2() && ggml_cpu_has_fma()))) {   // Q5_1 / Q8_0 down: bit-exact, any group size
        kq256_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    if (cpu_avx2_ok() && nt >= mt_min && f.d_type == 20 && iq4nl_mt) {   // #152: the same rule as the gate/up rows
        if (Arithmetic == NativeArithmetic::Default || mt_min <= 1) {
            iq4nl256_down_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
            return;
        }
#if defined(__x86_64__) || defined(_M_X64)
        if (ggml_cpu_has_avx2() && ggml_cpu_has_fma() && iq4nl_reference_ok()) {
            iq4nl_reference_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
            return;
        }
#endif
    }
    // the older-CPU tier: IQ4_NL and Q2_0 down rows in 128-bit (ggml-cpu's dots for both are scalar here)
    static const bool avx1 = !cpu_avx2_ok() && cpu_avx1_ok() && std::getenv("STRATA_NO_IQ128") == nullptr;
    if (avx1 && iq128_down_supported(f.d_type)) {
        iq128_down_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    const int n = (int) f.n_ff;
    float sd = 1.f;                                   // NVFP4: the expert's s_down (see native_gu_rows)
    if (f.tail_off) std::memcpy(&sd, blob + f.tail_off + 2 * sizeof(float), sizeof sd);
    static const bool nvfp4_512 = cpu_avx512_ok() && std::getenv("STRATA_NO_NVFP4_512") == nullptr;
    if (f.d_type == kNvfp4Type && nvfp4_512 && nvfp4_512_fits(n)) {
        nvfp4_512_rows(blob + f.down_off, f.d_row, n, hq, nt, out, r0, r1, sd);
        return;
    }
    static const bool nvfp4_256 = cpu_avx2_ok() && std::getenv("STRATA_NO_NVFP4_256") == nullptr;   // see above
    if (f.d_type == kNvfp4Type && nvfp4_256 && nvfp4_256_fits(n)) {
        nvfp4_256_rows(blob + f.down_off, f.d_row, n, hq, nt, out, r0, r1, sd);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.d_type)->vec_dot;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = blob + f.down_off + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = s * sd;
        }
    }
}

// Keep one dispatch implementation: only width-sensitive IQ kernels need a different arithmetic path.
// Already invariant kernels (including NVFP4) retain their shared decoding and scale handling.
void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt,
                    float* const* ff, int r0, int r1) {
    native_gu_rows_impl<NativeArithmetic::Default>(f, blob, act, nt, ff, r0, r1);
}

void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt,
                      float* const* out, int r0, int r1) {
    native_down_rows_impl<NativeArithmetic::Default>(f, blob, hq, nt, out, r0, r1);
}

void native_gu_rows_invariant(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt,
                              float* const* ff, int r0, int r1) {
    native_gu_rows_impl<NativeArithmetic::WindowInvariant>(f, blob, act, nt, ff, r0, r1);
}

void native_down_rows_invariant(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt,
                                float* const* out, int r0, int r1) {
    native_down_rows_impl<NativeArithmetic::WindowInvariant>(f, blob, hq, nt, out, r0, r1);
}

}  // namespace strata::kernels::cpu
