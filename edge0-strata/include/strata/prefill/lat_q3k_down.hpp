// include/strata/prefill/lat_q3k_down.hpp - M1 of the q3k latent A stage (q3k_fused/BRIEF.md): the down half.
#pragma once

#include <cstdint>

namespace strata::prefill {

/// Bytes of the staged activation rows `lat_q3k_down` reads (one entry): 10 chunks of 64 values, 80 bytes each -
/// 64 int8 codes, then {d0, d1} (the two 32-value blocks' fp16 scales), then 8 pad.
constexpr int64_t lat_q3k_hq_row = 800;

/// The latent A stage's down half on moe_fused.cu's latent kernel structure: IQ4_NL weight rows (K = 640) decoded in
/// place out of each expert blob's own bytes (no copy, no gather), 1280 output rows, several experts in one launch.
/// The staged weight rows (64 int8 codes + 4 fp16 sub-scales per 64 values), the staged activation rows, the
/// mma.sync m16n8k16 products and the epilogue are the NVFP4 latent down kernel's; only the block decode is IQ4_NL's.
///   blob   : group g's expert blob, the native layout native_expert_layout(11, 20, 1280, 640) (down at 704000,
///            360 bytes a row); a null pointer skips the group
///   starts : device array, n_groups + 1 entries: group g's entries start at starts[g]; its end is starts[g + 1]
///            unless `counts` says otherwise.  A group may hold more than one 64-row tile.
///   counts : nullptr (the old behaviour) = group g's row count is starts[g + 1] - starts[g]; non-null = it is
///            counts[g] and starts[] need not be increasing, adjacent or a partition at all - the engine's flushes
///            write each group's rows as one run, but the runs are not in group order (off[] follows the ring slots)
///   ent_dst: nullptr = entry e writes row e of `out`; non-null = row ent_dst[e] (native_expert_grouped's ent_dst).
///            hq is always indexed by the ENTRY (the gate/up pass quantizes H per entry)
///   hq     : the staged H rows, entry-major, lat_q3k_hq_row bytes an entry
///   out    : [entry][ld] fp32, entry-major (out[e * ld + r], r = 0 .. 1279)
void lat_q3k_down(const uint8_t* const* blob, const int32_t* starts, int n_groups, const void* hq, float* out,
                  int64_t ld, void* stream, const int32_t* counts = nullptr, const int32_t* ent_dst = nullptr);

}  // namespace strata::prefill
