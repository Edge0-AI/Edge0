// include/strata/prefill/lat_q3k_gu.hpp - M2a of the q3k latent A stage: the gate/up half (Q3_K, K = 1280).
#pragma once

#include <cstdint>

namespace strata::prefill {

/// One entry's gate/up input row: the xl row's q8_1 blocks, n_embd / 32 = 40 blocks of 36 B.
constexpr int64_t lat_q3k_gu_xq_row = 1440;
/// The blob's up section (native_expert_layout(11, 20, 1280, 640)); gate rows start at 0.
constexpr int64_t lat_q3k_gu_up_off = 352000;
/// A staged weight row: 4 x 16 int8 codes, then four fp32 sub-scales (Q3_K's 6-bit scale x the super-block d).
/// fp32, not the NVFP4 latent kernel's fp16: d * (sc - 32) is not an fp16 value, and the decode gate asks for 1e-6.
constexpr int lat_q3k_gu_staged_row = 80;
/// 64-value chunks along K = 1280 (5 Q3_K super-blocks x 4).
constexpr int lat_q3k_gu_chunks = 20;

/// gate/up (Q3_K, 640 rows each) on the latent kernel's staged rows: 128 features of gate and their up rows per work
/// item, several experts in one launch, the weights decoded in place out of each blob (no copy, no gather).
///   blob   : group g's expert blob (gate rows at 0, up rows at lat_q3k_gu_up_off, 550 B a row); null skips the group
///   starts : device array, n_groups + 1 entries: group g's entries start at starts[g]
///   counts : nullptr = group g's row count is starts[g + 1] - starts[g]; non-null = counts[g] (see lat_q3k_down)
///   ent_tok: nullptr = entry e reads input row e; non-null = row ent_tok[e] (native_expert_grouped's ent_tok).
///            gate/up are still written per ENTRY: gate[e * ld + f]
///   xq     : entry e's q8_1 input row at xq + e * lat_q3k_gu_xq_row
///   gate/up: [entry][ld] fp32, entry-major (the engine's native layout: gate[e * 640 + f])
void lat_q3k_gu(const uint8_t* const* blob, const int32_t* starts, int n_groups, const void* xq, float* gate,
                float* up, int64_t ld, void* stream, const int32_t* counts = nullptr,
                const int32_t* ent_tok = nullptr);

/// The decode gate only: the staged weight rows a gate row decodes to (lat_q3k_gu_chunks x lat_q3k_gu_staged_row
/// bytes), from the same device code path the kernel uses.
void lat_q3k_gu_dump(const void* blob, int row, void* out, void* stream);

}  // namespace strata::prefill
