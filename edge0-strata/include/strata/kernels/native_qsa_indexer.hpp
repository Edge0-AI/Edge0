#pragma once
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/rope_scaling.hpp"

namespace strata::kernels {

// Configure before session construction/capture. Disabled independently of the
// native QSA norm/gate switch; existing captures retain their selected kernels.
void native_qsa_indexer_set_enabled(bool enabled);
bool native_qsa_indexer_enabled();

// Pinned F16 raw-key cache / F32 pooling / weighted RMSNorm / text IMRoPE.
// Fixed single-sequence geometry: idx_dim=128, idx_block=4, n_rot=64.
// Uses the existing buffers: tail[3,128] stores half-rounded raw values expanded
// to F32; dead[128] is the repeated-cell0 spare key; pooled[max_cells/4+1,128]
// contains completed blocks followed by the spare; block_pos[1] changes only on
// completion and records that block's first absolute position.
//
// relative_pos_device is a cell index, advancing 0..max_cells-1 (use kStepPos,
// not the absolute RoPE position vector). The sequence is text,
// contiguous, with nonnegative pos_base divisible by four. Clear state before
// starting a new sequence. Completed block b rotates at pos_base+4*b; the spare
// always rotates at zero, including nonzero sequence bases, as in the oracle.
// Under rope scaling (rope_scaling.hpp) the zero angle keeps the spare's
// rotation an identity; YaRN's mscale rides in through cos(0) on both the pooled
// keys and the indexer queries, so selection is unchanged.
//
// Explicit nonnull stream; no allocation or synchronization. All spans must be
// aligned and disjoint, remain valid through replay, and contain finite values;
// raw values must round to finite F16 and epsilon must be finite and positive.
// max_cells is a positive fixed capacity; pos_base+max_cells must fit int32.
// Out-of-range device positions are ignored defensively, not host-validated.
// This adapter does not alter score accumulation, top-k, or finite 1e9 tail bias.
// perf-review C-2: the appends of cells [p0, p0 + n) at once, leaving the buffers exactly as n calls of the
// single append in order would (the prompt path appends a chunk before any query reads the state). raw [n, 128].
// Host-side positions (not for a captured graph).
void native_qsa_indexer_append_batch(const float* raw, int64_t n, int64_t p0, int32_t pos_base, const float* gamma,
                                     float epsilon, const QsaIndexerBuffers& b, const QsaShapes& s, int64_t max_cells,
                                     const RopeScaling& scaling, void* stream);

void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device,
                               int32_t pos_base, const float* gamma, float epsilon,
                               const QsaIndexerBuffers& buffers, const QsaShapes& shapes,
                               int64_t max_cells, const RopeScaling& scaling, void* stream);

void native_qsa_indexer_append_steps(const float* raw, const int32_t* relative_pos_device,
                                     int pos_stride, int n_steps, int32_t pos_base,
                                     const float* gamma, float epsilon,
                                     const QsaIndexerBuffers& buffers, const QsaShapes& shapes,
                                     int64_t max_cells, const RopeScaling& scaling, void* stream);

// engine-latency: the commit's append of the same window rows into several QSA layers in one launch.  Layer i reads
// raw0 + i * raw_stride and its own gamma/buffers; with tail_src[i] set its tail is first restored from there (the
// commit's snapshot, mapped memory), as copy_from_mapped did before the per-layer append.  Bitwise the per-layer calls.
struct QsaIndexerLayers {
    static constexpr int kMax = 16;
    const float* gamma[kMax] = {};
    const float* tail_src[kMax] = {};
    float* tail[kMax] = {};
    float* dead[kMax] = {};
    float* pooled[kMax] = {};
    int32_t* block_pos[kMax] = {};
};
void native_qsa_indexer_append_steps_layers(const float* raw0, int64_t raw_stride, const int32_t* relative_pos_device,
                                            int pos_stride, int n_steps, int32_t pos_base, float epsilon,
                                            const QsaIndexerLayers& layers, int n_layers, const QsaShapes& shapes,
                                            int64_t max_cells, const RopeScaling& scaling, void* stream);

} // namespace strata::kernels
