// include/strata/prefill/lat_q3k_fused.hpp - M3 of the q3k latent A stage: the whole A stage in one call.
#pragma once

#include <cstdint>

namespace strata::prefill {

/// An entry's staged H row (the down half's input): n_ff / 64 = 10 chunks of 80 B = n_ff / 32 = 20 q8_1 blocks.
constexpr int64_t lat_q3k_fused_hbuf_row = 800;

/// The q3k latent A stage in one call, two launches (experts_latent's shape): gate/up (Q3_K) with SwiGLU and the q8_1
/// of H in its epilogue into `hbuf`, then the shipped lat_q3k_down (IQ4_NL) on those H rows.  Nothing between the two
/// goes through the host, and gate/up never reach DRAM.
///   blob   : group g's expert blob (native_expert_layout(11, 20, 1280, 640)); null skips the group
///   starts : device array, n_groups + 1 entries: group g's entries start at starts[g]
///   counts : nullptr = group g's row count is starts[g + 1] - starts[g]; non-null = counts[g] (see lat_q3k_down)
///   ent_tok: nullptr = entry e reads input row e; non-null = row ent_tok[e] (native_expert_grouped's ent_tok)
///   ent_dst: nullptr = entry e writes row e of `out`; non-null = row ent_dst[e] (its ent_dst)
///   xq     : entry e's gate/up input row at xq + e * 1440 B (n_embd / 32 = 40 q8_1 blocks)
///   hbuf   : scratch, n_entries x lat_q3k_fused_hbuf_row bytes
///   out    : [entry][ld] fp32, entry-major (the A stage's yl)
/// The result is bit for bit lat_q3k_gu -> lat_q3k_swiglu_q8 -> lat_q3k_down on the same inputs (the test memcmps it).
void lat_q3k_fused(const uint8_t* const* blob, const int32_t* starts, int n_groups, const void* xq, void* hbuf,
                   float* out, int64_t ld, void* stream, const int32_t* counts = nullptr,
                   const int32_t* ent_tok = nullptr, const int32_t* ent_dst = nullptr);

}  // namespace strata::prefill
