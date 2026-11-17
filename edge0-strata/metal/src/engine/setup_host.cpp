// metal/src/engine/setup_host.cpp - the host-side state and sizes of the CUDA kernel files that the shared host code
// sets and reads while it starts the engine (path switches, scratch sizes, the RoPE table and its config).  Copied
// from gr.cu, shared_expert.cu, native_*.cu, ple.cu, qsa_decode_attn.cu, rope.cu, native_rope.cu,
// s2_expert_grouped.cu; keep in step.  On Metal none of these steers a kernel: the decode window runs mac-kern's
// chain (metal/src/engine/window.cpp); they keep the host side's allocations and checks as on the PC.
#include "strata/kernels/gr.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/prefill/gemm.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace strata::kernels {

namespace {
bool g_fp32_activations = false, g_native_mmvf = false, g_sh_native_bf16 = false, g_ple_native_bf16 = false,
     g_ple_native_postops = false;
std::atomic<bool> g_moe_combine{false}, g_gdn{false}, g_router{false}, g_qsa{false}, g_qsa_indexer{false}, g_rope{false};
RopeScaling g_rope_scaling;
}  // namespace

// ---- path switches (each .cu file's own flag)
void gr_set_fp32_activations(bool enabled) { g_fp32_activations = enabled; }
void gr_set_native_mmvf(bool enabled) { g_native_mmvf = enabled; }
bool gr_native_mmvf_enabled() { return g_native_mmvf; }
void shared_expert_set_native_bf16(bool enabled) { g_sh_native_bf16 = enabled; }
bool shared_expert_native_bf16_enabled() { return g_sh_native_bf16; }
void ple_set_native_bf16(bool enabled) { g_ple_native_bf16 = enabled; }
bool ple_native_bf16_enabled() { return g_ple_native_bf16; }
void ple_set_native_postops(bool enabled) { g_ple_native_postops = enabled; }
bool ple_native_postops_enabled() { return g_ple_native_postops; }
void native_moe_combine_set_enabled(bool v) { g_moe_combine.store(v, std::memory_order_relaxed); }
bool native_moe_combine_enabled() { return g_moe_combine.load(std::memory_order_relaxed); }
void native_gdn_set_enabled(bool v) { g_gdn.store(v, std::memory_order_relaxed); }
bool native_gdn_enabled() { return g_gdn.load(std::memory_order_relaxed); }
void native_router_set_enabled(bool v) { g_router.store(v, std::memory_order_relaxed); }
bool native_router_enabled() { return g_router.load(std::memory_order_relaxed); }
void native_qsa_set_enabled(bool v) { g_qsa.store(v, std::memory_order_relaxed); }
bool native_qsa_enabled() { return g_qsa.load(std::memory_order_relaxed); }
void native_qsa_indexer_set_enabled(bool v) { g_qsa_indexer.store(v, std::memory_order_relaxed); }
bool native_qsa_indexer_enabled() { return g_qsa_indexer.load(std::memory_order_relaxed); }
void native_rope_set_enabled(bool v) { g_rope.store(v, std::memory_order_relaxed); }
bool native_rope_enabled() { return g_rope.load(std::memory_order_relaxed); }
void fused_gr_check() {}   // the CUDA read variants' self-test: no variants on Metal

// ---- scratch sizes
uint64_t ple_block_scratch_bytes() {
    const size_t f = (size_t) (5 * NG_HC_DIM + NG_N_EMBD + NG_HC) * sizeof(float);
    const size_t q = (size_t) (NG_N_EMBD / 32) * 34;
    const size_t e = (size_t) NG_N_EMBD * sizeof(uint16_t);
    return ((f + 15) & ~(size_t) 15) + ((q + 15) & ~(size_t) 15) + e + 256;
}
uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    constexpr int HD = 256, CHUNK = 64;
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}
uint64_t shared_expert_scratch_bytes(int64_t n_ff) {
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    return a * 2 + q0 + qk + 32;
}
uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) + ((xh + 15) & ~15ull);
}
size_t gr_workspace_init(const GrShapes& s, void* base, GrWorkspace& out) {
    const size_t hc_dim = (size_t) s.hc * (size_t) s.n_embd;
    const size_t sz[5] = {hc_dim * sizeof(float), hc_dim * sizeof(uint16_t), (size_t) s.hc_lr * sizeof(uint16_t),
                          hc_dim * sizeof(float), (size_t) s.hc_lr * sizeof(float)};
    size_t al[5], bytes = 0;
    for (int k = 0; k < 5; ++k) {
        al[k] = (sz[k] + 15) & ~(size_t) 15;
        bytes += al[k];
    }
    out.bytes = bytes;
    if (base != nullptr) {
        unsigned char* p = (unsigned char*) base;
        void* ptr[5];
        for (int k = 0; k < 5; ++k) {
            ptr[k] = p;
            p += al[k];
        }
        out.xn = (float*) ptr[0];
        out.xq = (uint16_t*) ptr[1];
        out.lq = (uint16_t*) ptr[2];
        out.gated = (float*) ptr[3];
        out.lo = (float*) ptr[4];
    }
    return bytes;
}

// ---- RoPE (rope.cu, native_rope.cu): the config and the host-built angle table
void rope_scaling_set(const RopeScaling& scaling) { g_rope_scaling = scaling; }
const RopeScaling& rope_scaling() { return g_rope_scaling; }
void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab) {
    const int half = n_rot / 2;
    for (int p = 0; p < max_pos; ++p)
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            cos_tab[(size_t) p * half + i] = (float) std::cos(ang);
            sin_tab[(size_t) p * half + i] = (float) std::sin(ang);
        }
}
void build_rope_table(int n_rot, const RopeScaling& sc, int max_pos, float* cos_tab, float* sin_tab) {
    if (sc.type == RopeScalingType::None) {
        build_rope_table(n_rot, sc.freq_base, max_pos, cos_tab, sin_tab);
        return;
    }
    const int half = n_rot / 2;
    const double fs = sc.freq_scale();
    const double ms = sc.mscale();
    double cd[2];
    sc.corr_dims(n_rot, cd);
    const bool correct = sc.ext_factor != 0;
    for (int p = 0; p < max_pos; ++p)
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(sc.freq_base, -2.0 * (double) i / (double) n_rot);
            const double extrap = (double) p * inv;
            const double interp = fs * extrap;
            double ang = interp;
            if (correct) {
                const double ramp = (double) rope_yarn_ramp((float) cd[0], (float) cd[1], i) * sc.ext_factor;
                ang = interp * (1.0 - ramp) + extrap * ramp;
            }
            cos_tab[(size_t) p * half + i] = (float) (std::cos(ang) * ms);
            sin_tab[(size_t) p * half + i] = (float) (std::sin(ang) * ms);
        }
}
void rope_table_set(const float*, const float*, int, const RopeScaling&) {}   // the window keeps its own table
void rope_table_release(const float*) {}
void native_q6_k_unpack(const void*) {}   // nothing packed on Metal

}  // namespace strata::kernels

namespace strata::prefill {
Gemm::~Gemm() {}   // nothing created on Metal (no prompt GEMMs yet)
bool prompt_f16() { return false; }   // the CUDA/HIP prompt path's FP16 image mode: not used on Metal
}  // namespace strata::prefill
