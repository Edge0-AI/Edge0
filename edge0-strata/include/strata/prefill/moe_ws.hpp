// include/strata/prefill/moe_ws.hpp - GP100 (sm_60, the STRATA_EXPERIMENTAL_SM60 build): the prompt path's expert
// products of a layer whose experts are all resident in VRAM, read in place there and weight-stationary: ONE launch
// for the layer's gate/up and ONE for its down instead of a gate/up, swiglu and down MMQ launch per 16-expert group.
// A CTA holds 128 (or 256) weight rows of one expert and runs them against up to 64 of the expert's activation rows, a
// thread per weight row: each weight block is decoded once per pass instead of once per MMQ tile, and an expert's
// rows are its own count, not the group's largest.  Every product keeps MMQ's int8 values, scales and per-32-value
// epilogue in its K order (the down K loop stops at 640), so GU, H and Dm are the per-group MMQ's bits.  Swift
// Flash's shapes and formats only: gate/up [1280 x 2560] IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS or IQ3_S, down [2560 x 640]
// Q2_0 or IQ4_NL.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::ws {

/// The products of a layer: 1 with STRATA_MMQ_WS=1 (opt-in) gate/up and down, else 0 (the per-group MMQ).  0 also where
/// the build has no kernels (HIP, no ggml sources, not the STRATA_EXPERIMENTAL_SM60 build) or a visible GPU is not
/// cc 6.0 (the kernels were tuned and checked on GP100 only).
int mode();
/// mode() != 0 and both formats have kernels.
bool supported(int gu_type, int d_type);
/// STRATA_WS_FAST=1 (opt-in, read once): the IQ4_NL down's 32-token tier (u >= 20 rows per routed expert) runs a
/// kernel that issues fewer instructions per pass - a switch into the unrolled token pairs, dy transposed - with every
/// token's operations and their order unchanged, so Dm is the same bits.  0 (unset): the kernels as before.
int fast();
/// Benches and tests: 0 / 1 overrides fast(), -1 goes back to the environment.
void set_fast(int v);
/// Tests: 0 / 1 overrides mode() (1 runs the kernels on a card that is not cc 6.0), -1 goes back to the environment.
void set_mode(int v);
/// The weight-stationary product of `ggml_type` beats the per-group MMQ at u rows per routed expert (the measured
/// ratios, moe_ws.cu; u is clamped to the measured 2.5 .. 160).
bool faster(int ggml_type, double u);
/// Every gate/up and down matrix starts this aligned (the IQ4_NL down reads 8-byte pieces; the expert cache's slots are
/// 256-byte aligned and the Swift gate/up sizes multiples of 8).
constexpr uint64_t kAlign = 8;

/// The CTAs' work: rows [row0, row0 + rows) of Xq / GU / Hq / Dm, all of the expert at position j of the layer's order
/// (MMQ's: its xoff entry), at most kChunkRows.  A CTA takes one chunk and one 128- or 256-row piece of the weights.
struct alignas(16) Chunk {
    int32_t j, row0, rows, pad;
};
constexpr int kChunkRows = 64;
/// The most chunks a layer of `rows` routed rows over at most `n_expert` experts makes.
inline size_t max_chunks(int64_t rows, int64_t n_expert) { return (size_t) (rows / kChunkRows + n_expert); }
/// The chunks of the n experts of `order` (expert e has cnt[e] rows from bounds[j], j its position), the largest
/// expert first (the long CTAs start first); returns their number (<= max_chunks).
size_t chunks(const int32_t* order, size_t n, const int32_t* cnt, const int32_t* bounds, Chunk* out);

/// One layer: its n routed experts' matrices at w + xoff[j] (gate/up, rows 0-639 gate then up) and w + xoff[j] +
/// down_off (down), xoff on the device in the layer's order (Product::xoff's table, all n entries); `rows` = T*K.
struct Layer {
    int gu_type = -1, d_type = -1;
    const void* w = nullptr;
    const int64_t* xoff = nullptr;
    int64_t down_off = 0;
    const Chunk* chunks = nullptr;   ///< on the device
    int64_t nchunks = 0;
    int64_t rows = 0;
    int n = 0;
    const void* xq = nullptr;        ///< the gate/up's q8_1 rows (mmq::quantize_scatter's, rows of them)
    float* gu = nullptr;             ///< [rows][1280], the rows MMQ writes through identity ids
    const void* hq = nullptr;        ///< H's q8_1 rows: ONE mmq::swiglu_quant over all `rows` rows of gu
    float* dm = nullptr;             ///< [rows][2560]
};
/// The layer's gate/up into gu, one launch; false (nothing launched) where the per-group MMQ is to run instead (mode 0,
/// or not faster at this u) - the same GU rows either way.
bool gate_up(const Layer& L, void* stream);
/// The layer's down from hq into dm, one launch.
void down(const Layer& L, void* stream);

}  // namespace strata::prefill::ws
