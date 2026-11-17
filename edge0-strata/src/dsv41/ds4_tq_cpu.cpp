// Strata. The trellis routed experts (ds4_iq.cuh's is_tq types) on the CPU lane of the DeepSeek-V4.1 engine path: Strata's
// own decode-and-dot kernels (include/strata/kernels/cpu/tq_cpu.hpp: AVX-512 / AVX2 / scalar, one result whatever the ISA)
// over the file's rows read in place.  The activations are the device's (ds4_gguf.cu's q8_quant: int8 with an F32 scale a 32
// values), so an expert gives the same sums on either lane up to the order of the blocks' float sums.
#include "ds4_gguf_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#if defined(STRATA_DSV41_TQ_CPU)
#include "strata/kernels/cpu/tq_cpu.hpp"
#endif

namespace strata::ds4::gguf {

#if defined(STRATA_DSV41_TQ_CPU)
namespace kc = strata::kernels::cpu;
namespace {

int kbits_of(int type) {
    switch (type) {
        case 145: return 6;
        case 146: return 7;
        case 224: case 226: return 5;
        case 225: case 227: return 4;
        default: return 0;
    }
}
constexpr int kActTq = 100;   // IqCpu::gu_act / d_act of these types: the layout below

// an activation of k values: k / 32 F32 scales, then k int8
size_t act_bytes_of(int k) { return (size_t) k / 32 * 4 + (size_t) k; }

void unpack(const void* act, int k, kc::tq_detail::TqAct& a) {
    const uint8_t* p = (const uint8_t*) act;
    std::memcpy(a.d, p, (size_t) k / 32 * 4);
    const int8_t* q = (const int8_t*) (p + (size_t) k / 32 * 4);
    for (int i = 0; i < k; ++i) a.q[i] = q[i];
}

// dots of rows [r0, r1) of one matrix against nt activations: out[t][r]
void dots(int type, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
    static thread_local std::vector<kc::tq_detail::TqAct> unp(kc::kTqMaxTokens);
    const kc::tq_detail::TqAct* ap[kc::kTqMaxTokens];
    for (int t = 0; t < nt; ++t) {
        unpack(act[t], n, unp[(size_t) t]);
        ap[t] = &unp[(size_t) t];
    }
    const int kb = kc::tq_kbits(type);
    const int16_t* tab = kc::tq_state_table_for(type);
    switch (kc::tq_isa()) {
#if defined(__x86_64__) || defined(_M_X64)   // the AVX units exist on x86-64 builds only (arm64: the scalar rows)
        case kc::TqIsa::avx512: kc::tq_detail::tq_rows_avx512(kb, tab, w, rb, n, ap, nt, out, r0, r1); break;
        case kc::TqIsa::avx2: kc::tq_detail::tq_rows_avx2(kb, tab, w, rb, n, ap, nt, out, r0, r1); break;
#endif
        default: kc::tq_detail::tq_rows_scalar(kb, tab, w, rb, n, ap, nt, out, r0, r1); break;
    }
}
}  // namespace

bool tq_cpu_fmt(int tg, int td, int n_embd, int n_ff, IqCpu& f, std::string& err) {
    if (kbits_of(tg) == 0 || kbits_of(td) == 0) {
        err = "not trellis types";
        return false;
    }
    if (n_embd % 128 != 0 || n_ff % 128 != 0 || n_embd > kc::kTqMaxN || n_ff > kc::kTqMaxN) {
        err = "the trellis CPU kernels take widths of 128 blocks up to " + std::to_string(kc::kTqMaxN);
        return false;
    }
    f.tg = tg;
    f.td = td;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_act = f.d_act = kActTq;
    f.gu_row = (size_t) (n_embd / 128) * (2 + 4 * kbits_of(tg));
    f.d_row = (size_t) (n_ff / 128) * (2 + 4 * kbits_of(td));
    f.act_bytes = act_bytes_of(n_embd);
    f.h_bytes = act_bytes_of(n_ff);
    return true;
}

// x (k values) -> the layout above (the device's q8_quant: sc = max|x| / 127, q = round-half-even(x / sc))
void tq_cpu_quant(const float* x, int k, void* dst) {
    float* d = (float*) dst;
    int8_t* q = (int8_t*) ((uint8_t*) dst + (size_t) k / 32 * 4);
    for (int b = 0; b < k / 32; ++b) {
        const float* xb = x + b * 32;
        float am = 0.0f;
        for (int j = 0; j < 32; ++j) am = std::max(am, std::fabs(xb[j]));
        const float sc = am / 127.0f;
        d[b] = sc;
        for (int j = 0; j < 32; ++j) q[b * 32 + j] = (int8_t) (am > 0.0f ? (int) std::nearbyint(xb[j] / sc) : 0);
    }
}

void tq_cpu_gate_up(const IqCpu& f, const uint8_t* gate, const uint8_t* up, const void* const* act, int nt, float* const* ff,
                    int r0, int r1, float limit) {
    static thread_local std::vector<float> tmp[kc::kTqMaxTokens];
    float* up_out[kc::kTqMaxTokens];
    for (int t = 0; t < nt; ++t) {
        tmp[t].resize((size_t) r1);
        up_out[t] = tmp[t].data();
    }
    dots(f.tg, gate, f.gu_row, f.n_embd, act, nt, ff, r0, r1);
    dots(f.tg, up, f.gu_row, f.n_embd, act, nt, up_out, r0, r1);
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r) {
            float g = ff[t][r], u = up_out[t][r];
            if (limit > 0.0f) {
                g = std::min(g, limit);
                u = std::min(std::max(u, -limit), limit);
            }
            ff[t][r] = g / (1.0f + std::exp(-g)) * u;
        }
}

void tq_cpu_down(const IqCpu& f, const uint8_t* down, const void* const* hq, int nt, float* const* out, int r0, int r1) {
    dots(f.td, down, f.d_row, f.n_ff, hq, nt, out, r0, r1);
}
bool tq_cpu_type(int type) { return kbits_of(type) > 0; }
int tq_cpu_act() { return kActTq; }
#else
bool tq_cpu_fmt(int, int, int, int, IqCpu&, std::string& err) {
    err = "this build has no CPU trellis kernels (STRATA_NATIVE_EXPERTS=OFF)";
    return false;
}
void tq_cpu_quant(const float*, int, void*) {}
void tq_cpu_gate_up(const IqCpu&, const uint8_t*, const uint8_t*, const void* const*, int, float* const*, int, int, float) {}
void tq_cpu_down(const IqCpu&, const uint8_t*, const void* const*, int, float* const*, int, int) {}
bool tq_cpu_type(int) { return false; }
int tq_cpu_act() { return 100; }
#endif

}  // namespace strata::ds4::gguf
