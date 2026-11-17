// include/strata/kernels/ple.hpp - P2.S4: the PLE block's GPU half (`build_ple`, qwen4exp.cpp L1206-1293).
//
// One token, sixteen gathered rows of 160 -> a residual-stack update:
//
//     key       = grouped_norm(ple_key   @ emb)          hc_dim, per-stream RMS then a [n_embd,hc] gamma
//     query     = grouped_norm(hidden,   ple_norm_query)
//     value     = ple_value @ emb                        n_embd
//     s[c]      = sum_d(key[c]*query[c]) / sqrt(n_embd)
//     gate[c]   = sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))     signed square root, then sigmoid
//     gated[c]  = value * gate[c]                        value broadcast over the hc streams
//     normalized= grouped_norm(gated, ple_norm_conv)     <-- THE CONV INPUT
//     conv      = sum_k kW[k][c] * padded[hist + t - (kern-1-k)*dil][c]
//     out       = hidden + gated + silu(conv)
//
// **`normalized` IS THE CONV INPUT, AND `ref/ngram.py` SAYS OTHERWISE.**  Its `ple_block` docstring calls
// `terms` "the padded gated values", but the source pads `normalized` - the grouped_norm of the gated values,
// not the gated values themselves - and `ple_layer_xcheck.cpp` L107-112 builds it that way from ggml.  The
// reference computes `normalized` and then ignores it, passing `terms` in from the caller, so its own code
// does not contradict its docstring and nothing catches it.  This header takes the HISTORY OF NORMALIZED
// ROWS, which is what the state actually is.
//
// WHERE THE WEIGHTS LIVE, and the two different layouts in play.  `ple_conv1d.weight` has manifest shape
// [4, 10240] with ne0 = 4 fast, so ggml's flat index is `k + 4*c`; `ref/ngram.py`'s `kernel` is a (4, 10240)
// numpy array whose flat index is `k*10240 + c`.  They are transposes, and the source's own view
// (`ggml_view_2d(tWc, 1, hc_dim, nb[1], k*nb[0])`) confirms which is meant.  `ple_key` and `ple_value` are
// [2560, n] with ne0 = 2560 fast, which IS the reference's row-major orientation.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Opt in to pinned CUDA BF16/F32 MMVF for the PLE value projection only (default false).
/// Configure before session capture; captured graphs retain their selected projection kernels.
void ple_set_native_bf16(bool enabled);
bool ple_native_bf16_enabled();

/// Opt in to pinned CUDA postprojection norms, gate, convolution and residual
/// arithmetic (default false). Set before capture; existing graphs keep their
/// selected kernels. Independent of the key/value projection options.
void ple_set_native_postops(bool enabled);
bool ple_native_postops_enabled();

/// Weights of the single PLE layer (`ple.layers = [1]`).  `blk.1.*` in the pack.
///
///   key_codes / key_scales   `ple_key.weight`   S2, [n_embd, hc_dim], code_bias -1, group 64.
///                            The pack stores `scales_fp16: true`, and every S-form kernel in this project
///                            (including `s2_gemv_q8`) takes scales as F32 - so the LOADER widens them once,
///                            as it already must for the experts.  Recorded because "the manifest says fp16"
///                            and "the kernel wants float" is a silent 2x-read if nobody says which wins.
///   value_bf16               `ple_value.weight` BF16 as raw bits, [n_embd, n_embd]
///   norm_key/query/conv      F32 [hc_dim], NO (1+w) folding - `ref/gdn.py` records the same for `ssm_norm`
///   conv1d_f16               `ple_conv1d.weight` F16, GGML-NATIVE `kW[k + kern*c]`
struct PleWeights {
    const uint8_t* key_codes = nullptr;    // hc_dim * (n_embd/4) bytes
    const float* key_scales = nullptr;     // hc_dim * (n_embd/64) floats
    const uint16_t* value_bf16 = nullptr;  // n_embd * n_embd
    const float* norm_key = nullptr;       // hc_dim
    const float* norm_query = nullptr;     // hc_dim
    const float* norm_conv = nullptr;      // hc_dim
    const uint16_t* conv1d_f16 = nullptr;  // PLE_CONV_KERNEL * hc_dim, flat index k + 4*c

    /// Optional unchanged GGUF Q2_0 [2560,10240] key projection. Nonnull data
    /// requires type 42 and caller-owned native_q8_1_bytes(2560) device scratch,
    /// disjoint from the PLE workspace, inputs, weights and exported outputs.
    /// Use an explicit ordered stream and retain both buffers through graph
    /// execution. Null data preserves the canonical key projection.
    const void* key_native_data = nullptr;
    int key_native_type = -1;
    void* key_native_q8_1 = nullptr;
    /// Plan v0.3 P6: the IQ model files keep `ple_key` in BF16 ([n_embd, hc_dim], `w[o*n_embd+i]`); when set it
    /// replaces both key paths above.
    const uint16_t* key_bf16 = nullptr;
    /// S25 (STRATA_PLE_BATCH=1): the key (hc_dim) and value (n_embd) projections of `emb`, already computed by the
    /// caller for a window's rows at once (the same kernels' multi-row forms); ple_block then copies them instead of
    /// running its own GEMVs
    const float* pre_key = nullptr;
    const float* pre_value = nullptr;
};

/// Everything the block produces, in the order `ple_layer_xcheck`'s oracle writes it.  Any pointer may be
/// null; the caller that only wants the result passes nothing else.  Having the intermediates is what makes
/// the parity test able to say WHICH stage diverged instead of only that the sum did.
/// Every nonnull output region, including result, must be disjoint from the block's entire scratch region.
/// result may alias hidden. The block rejects scratch/output overlap before launching any work.
struct PleOut {
    float* key = nullptr;         // hc_dim
    float* value = nullptr;       // n_embd
    float* gate = nullptr;        // hc
    float* gated = nullptr;       // hc_dim
    float* normalized = nullptr;  // hc_dim
    float* conv = nullptr;        // hc_dim
    float* result = nullptr;      // hc_dim
};

/// One token.  `emb` is the 2560-wide gathered n-gram embedding, `hidden` the hc_dim residual stack, and
/// `hist_rows` the NG_HIST previous NORMALIZED rows, oldest first (the caller owns that cache).
///
/// `hist_rows` IS ROW-FASTEST: `hist_rows[row + NG_HIST*channel]`.  That is not a choice - it is
/// `ggml_reshape_3d(state, d_conv-1, conv_channels, n_seqs)`, so ne0 = the history row varies fastest, and
/// `ple_layer_xcheck.cpp` L79-83 records the transposed reading failing ggml's own assert.
///
/// The block does NOT write back to the history: the caller keeps ownership of its state and this stays a
/// pure function of its inputs, which is what makes the chunked-versus-single-shot property testable.
/// Bytes the caller must hand `ple_block` as its workspace: five `hc_dim` buffers (key, query, norm,
/// gated, conv), one `n_embd` (value) and one `hc` (gate), plus the Q8 activation image and the BF16
/// embedding copy.  **THE WORKSPACE IS THE CALLER'S BECAUSE THIS FUNCTION IS CAPTURED**: it used to
/// `cudaMalloc` three times and `cudaStreamSynchronize` once per call, and inside a graph both are errors -
/// `ple_block: scratch: operation not permitted when stream is capturing`.  Same defect, same fix as
/// `shared_expert`.
uint64_t ple_block_scratch_bytes();

void ple_block(const float* emb, const float* hidden, const float* hist_rows, const PleWeights& w,
               PleOut& out, void* scratch, void* stream);

void ple_block_projected(const float* projected_key, const float* projected_value, const float* hidden,
                         const float* hist_rows, const PleWeights& w, PleOut& out, void* scratch, void* stream);

/// Advance the row-fastest normalized history by one token: hist[r,c] = old_hist[r+1,c], then append
/// normalized[c] at row NG_HIST-1. One thread owns each channel, making the in-place shift well-defined.
/// hist and normalized must be nonnull, disjoint device regions of NG_HIST*NG_HC_DIM and NG_HC_DIM floats.
/// Async on stream, allocation-free and capture-safe. Throws on null or overlapping regions.
void ple_history_advance(float* hist, const float* normalized, void* stream);

/// Whether the CUDA path is available.  The host hash and the table read work without it; the block does not.
bool ple_block_available();

/// STRATA_PLE_COMPACT: `rows` raw IQ4_NL PLE rows (PleTable::gather_raw_iq4nl, PLE_ROW_BYTES each) to 160 floats each,
/// the floats iq4nl_dequant_row writes (d times the code's value, one multiply), on `stream`.
void ple_iq4nl_dequant(const uint8_t* raw, int64_t rows, float* out, void* stream);

}  // namespace strata::kernels
