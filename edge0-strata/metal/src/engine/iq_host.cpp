// metal/src/engine/iq_host.cpp - the host-side helpers of src/kernels/cuda/iq_kernels.cu the shared host code reads
// (format tables, row sizes, the native expert blob layout).  The kernels themselves are not ported: on Metal the
// decode window runs mac-kern's chain (metal/src/engine/window.cpp).  Copied from iq_kernels.cu; keep in step.
#include "strata/kernels/iq_kernels.hpp"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-common-tq.h"   // block_tq2_t / block_tqk6 / block_tqk7

#include <cstddef>
#include <cstdint>

namespace strata::kernels {
namespace {
// agentionai's trellis types (Gyro): TQ2_T (144), TQK6 (145), TQK7 (146); 128 weights a block
bool is_tq(int t) { return t == 144 || t == 145 || t == 146; }
bool is_iq(int t) {
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11 ||
           t == 12 || t == 13 || t == 14 || t == 7 || t == 6 || t == 2 || t == 3 || t == 8;   // 14: Q6_K (Unsloth's Q5/Q6 gate/up)
}
// values per block of the types the grouped expert kernels take (0 = none): STRATA_GU_FMTS / STRATA_D_FMTS
int gu_qk(int t) {
    switch (t) {
        case 16: case 17: case 18: case 21: case 22: case 23: case 29: case 12: case 13: case 14: return 256;
        case 42: return 64;
        case 144: case 145: return 128;   // TQ2_T / TQK6 gate/up (the CUDA engine's STRATA_GU_FMTS)
        case 6: case 2: case 3: case 8: return 32;
        default: return 0;
    }
}
int d_qk(int t) {
    switch (t) {
        case 23: return 256;
        case 42: return 64;
        case 144: case 146: return 128;   // TQ2_T / TQK7 down
        case 20: case 7: case 6: case 2: case 3: case 8: return 32;
        default: return 0;
    }
}
}  // namespace

bool iq_supported(int t) noexcept { return is_iq(t); }
bool embed_type_supported(int t) noexcept { return is_iq(t) || t == 30; }

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        case 12: return (size_t) (n / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n / 256) * sizeof(block_q5_K);
        case 14: return (size_t) (n / 256) * sizeof(block_q6_K);
        case 7: return (size_t) (n / 32) * sizeof(block_q5_1);
        case 6: return (size_t) (n / 32) * sizeof(block_q5_0);
        case 2: return (size_t) (n / 32) * sizeof(block_q4_0);
        case 3: return (size_t) (n / 32) * sizeof(block_q4_1);
        case 8: return (size_t) (n / 32) * sizeof(block_q8_0);
        case 144: return (size_t) (n / 128) * sizeof(block_tq2_t);
        case 145: return (size_t) (n / 128) * sizeof(block_tqk6);
        case 146: return (size_t) (n / 128) * sizeof(block_tqk7);
        case 30: return (size_t) n * 2;
        default: return 0;
    }
}

bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    const int qg = gu_qk(gu_type), qd = d_qk(d_type);
    return qg > 0 && qd > 0 && (is_iq(gu_type) || is_tq(gu_type)) && (is_iq(d_type) || is_tq(d_type)) && n_embd % qg == 0 && n_ff % qd == 0 &&
           n_embd % 256 == 0 && (n_ff * n_embd) % 256 == 0;   // (the trellis rows: 128-value blocks, 2560 = 20, 640 = 5)
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
}

}  // namespace strata::kernels
