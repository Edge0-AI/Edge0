// include/strata/kernels/s2_expert_grouped.hpp - R4's GPU expert tier.
//
// **WHAT THIS IS FOR.** The CPU expert pool is the engine's largest single cost - 663.6 MB of expert bytes per
// token at ~40 GB/s = 16.2 ms on a ~53 ms token - and `test_expert_pool` measures the pipeline hiding
// **1.055 ms of 19.035**. The CPU work is 96% exposed because the residual chain is strictly serial, so the
// only way past it is to stop reading those bytes on the CPU: put experts in VRAM and compute them on the GPU.
//
// **WHY GROUPED, AND WHY THAT WORD IS THE WHOLE DESIGN.** A layer routes ten experts. The obvious
// implementation is a loop over hits calling the per-matrix kernel, and it is wrong by arithmetic: five
// launches per expert at ~6.45 hits and 48 layers is **~1,550 launches per token**, which at a real-stream
// launch cost of ~3.6 us is 5.6 ms - most of the 10.4 ms the tier is supposed to save. So the hits are grouped
// into ONE launch per projection:
//
//     1. `gu_kernel`    n_hits x 2 x FF rows   - gate and up, together
//     2. `swiglu_kernel` n_hits x FF           - silu(gate) * up
//     3. `quantize_q8_0` (the existing one)    - the down weight's own activation contract
//     4. `down_kernel`   n_hits x H rows
//
// **four launches per layer, not five per expert.**
//
// `h_layer` held-out is **0.0456** - only 4.6% of (layer, token) pairs have all ten experts resident - so this
// cannot be a per-layer grouped kernel with a fixed grid, and the hit list is passed **as data** so the caller
// can keep one captured graph per layer with the maximum grid and a device-side count.
#pragma once

#include "strata/kernels/window_limits.hpp"

#include <cstdint>

namespace strata::kernels {

/// Bytes of caller-owned scratch `moe_hit_grouped_s2` needs for `n_hits` experts.
///
/// Two regions: the gate/up activations (`n_hits x 2 x FF` floats) and the quantized intermediate
/// (`n_hits x (FF/32) x 34` bytes).  Carved from the caller's arena rather than allocated - a `cudaMalloc`
/// on the token path is forbidden by P2.T10 and illegal during stream capture, and this is called from both.
uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff);

/// **ONE GROUPED EXPERT EVALUATION.**  For each hit `h` in `[0, n_hits)`, evaluates the expert held in
/// `blob_base + slot_index[h] * blob_bytes` against `x_q8_0` and writes `n_embd` floats to
/// `out + dst_index[h] * n_embd`.
///
/// * `blob_base`    the device slot arena; slot `s` starts at `blob_base + s * blob_bytes`
/// * `slot_index`   `n_hits` `int32_t` slot indices on the DEVICE, so the caller can change which experts are
///                  resident without recapturing anything
/// * **`dst_index`   where each hit's answer goes, also on the device and also `n_hits` long.**
///
///                  **THIS IS NOT A CONVENIENCE - IT IS WHAT LETS THE TWO HALVES MEET.**  A layer routes ten
///                  experts and only some are resident, so the CPU produces the misses at their ROUTED indices
///                  (the router's order is what `moe_combine` weights against) and the GPU produces the hits in
///                  whatever order the cache happens to hold them.  Writing the hits sequentially would land
///                  them on the misses' rows; `dst_index` puts hit `h`'s answer on the row its expert was routed
///                  to, so the CPU can zero the hit rows, the GPU can fill them, and `moe_combine` needs no
///                  change at all.
/// * `x_q8_0`       `(n_embd/32)` `block_q8_0`, 34 bytes each, from `quantize_q8_0`
/// * `scratch`      `moe_hit_grouped_scratch_bytes(n_hits, n_embd, n_ff)`, caller-owned
/// * `out`          `(k, n_embd)` floats - the SAME buffer the misses were written into
/// * **`x_scales`    R4.2h, OPTIONAL, and null keeps the previous behaviour exactly.**  When non-null it is
///                  `n_embd/32` **fp32** activation scales from `quantize_q8_0_scaled`, and they REPLACE the
///                  block's fp16 `d` as the activation multiplier - for the gate/up projection and for the
///                  intermediate's own quantization in step 3.
///
///                  **THIS EXISTS BECAUSE A HIT AND A MISS MUST PRODUCE THE SAME NUMBER.**  The CPU pool
///                  multiplies by the fp32 `ActQ::scale` (`cpu/expert.cpp:92`); this kernel read fp16 `d` out
///                  of the `block_q8_0`.  `bench/micro/act_quant_parity.cu` measured the gap with both real
///                  implementations linked: **80 of 80 chunks differ, max relative 4.761e-04** - 2^-11, pure
///                  fp16 rounding - which is why enabling the cache changed the generated tokens.  Pass null
///                  and this is bit-for-bit the kernel `moe_hit_parity` has always checked at 6.279e-08.
///
/// **THE BLOB LAYOUT IS `expert.hpp`'s AND IS NOT RESTATED HERE.**  Gate/up codes are interleaved by row
/// (`2r` = gate, `2r+1` = up), the down codes follow, and each plane's scales are fp16 - which is the ONE
/// difference from `s_gemv_q8`'s S-forms, whose scales are fp32.  That difference is why this is a separate
/// kernel rather than a call into the dense path.
void moe_hit_grouped_s2(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                        int64_t n_hits, int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch, float* out,
                        void* stream, const float* x_scales = nullptr);

/// Experimental CPU-order implementation. Eight CUDA lanes reproduce the CPU VNNI accumulator lanes,
/// fused multiply-add order, separate correction, and final horizontal reduction. The input must use
/// the CPU quantization contract and fp32 scales. SiLU uses accurate CUDA expf, which can differ from
/// the CPU math library by last bits. End-to-end correctness must be checked against the C1 oracle.
/// Validate both intermediate quantization and final outputs.
/// Optional gate_up_trace receives [all gates | all ups] before SiLU, for parity diagnostics only.
/// Plan v0.3 P4 token graph: the hit path with NO host step.  `moe_hit_select` builds this layer's hit list
/// from the routed `ids` and the static residency row (`res_row[e]` = slot or -1) into `slot`/`dst`/`count`
/// (device); `moe_hit_grouped_s2_dev` is `moe_hit_grouped_s2` sized for `cap` hits that reads the real count
/// from `d_count`; `moe_hit_add` adds each hit's row of `hit_out` into `parts` (rows the CPU left at zero).
void moe_hit_select(const int32_t* ids, const int32_t* res_row, int k, int n_expert, int32_t* slot, int32_t* dst,
                    int32_t* count, void* stream);
void moe_hit_grouped_s2_dev(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                            const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                            void* scratch, float* out, void* stream, const float* x_scales);
void moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream);
/// knife-0: the same walk with the u0 path's write semantics - parts[dst[i]] = hit_out[dst[i]] (an assignment; the
/// latent branch leaves those rows untouched, so an add would keep the previous window's values).
void moe_hit_set(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream);
/// LATENT-CUT-A: the fused wait+copy+add for the latent CPU-share merge (one launch; the
/// mapped rows read volatile - bit-identical to copy_rows_from_mapped + moe_hit_add).
void moe_hit_add_mapped(float* parts, const float* hit_out, const volatile float* ymiss, const int32_t* dst,
                        const int32_t* count, int64_t cap, int64_t n_embd, const volatile uint32_t* wait_flag,
                        uint32_t wait_value, void* stream);
/// Plan v0.3 P6 verify window: `moe_hit_select` over `n` <= 128 routed entries (T tokens x k, flattened), and the
/// hit kernel with one activation PER TOKEN - entry `dst` reads token `dst / k_per_token`'s rows of `x_q8_0`
/// ((n_embd/32)*34 bytes each) and `x_scales` (n_embd/32 floats each).  Per hit, bitwise `moe_hit_grouped_s2_dev`.
/// Plan v0.3 P6: routed experts GROUPED.  Group g's blob is at device address grp_ptr[g] (a VRAM slot, or a mapped
/// host blob the kernel reads over PCIe); its entries are [grp_start[g], grp_start[g+1]) with ent_tok (the token
/// whose activation row of x_q8_0/x_scales it reads) and ent_dst (its row of `out`).  Each row of a blob is read
/// once for all of its entries.  Counts are read from device memory; every entry is bitwise the per-entry kernel.
void moe_grouped_s2(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                    const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                    const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, void* stream);
#if STRATA_VERIFY_MAX_T > 8
/// The same for a window of 9..16 rows (CUDA / HIP engines built with a wide window, window_limits.hpp): `max_group` is the
/// most entries one group can hold = the window's rows (a token routes to an expert once).  Up to 8 it is
/// `moe_grouped_s2` (one pass of each kernel); 9..16 adds a second pass over the entries from the 9th on.
void moe_grouped_s2_wide(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                         const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                         const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, void* stream,
                         int max_group);
#endif
/// What a verify window calls: `max_group` is the window's rows.  A build with narrow windows (the default, and every SYCL / Metal
/// build, whose kernel libraries have only `moe_grouped_s2`) runs `moe_grouped_s2`: its windows hold at most 8 rows.
inline void moe_grouped_s2_window(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                                  const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                                  const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, void* stream,
                                  int max_group) {
#if STRATA_VERIFY_MAX_T > 8
    moe_grouped_s2_wide(grp_ptr, grp_start, n_groups, ent_dst, ent_tok, cap_groups, cap_entries, x_q8_0, x_scales, scratch, out,
                        stream, max_group);
#else
    (void) max_group;
    moe_grouped_s2(grp_ptr, grp_start, n_groups, ent_dst, ent_tok, cap_groups, cap_entries, x_q8_0, x_scales, scratch, out, stream);
#endif
}
/// Plan v0.3 P6: the groups for `moe_grouped_s2` when every expert id is resident at base + id * blob (the MTP
/// layer's 512 experts): counts[0] groups, counts[1] entries; entry dst = routing index, tok = index / k_per_tok.
void moe_group_resident(const int32_t* ids, int n, int k_per_tok, const uint8_t* base, int64_t blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* counts, int32_t* ent_dst,
                        int32_t* ent_tok, void* stream);
void moe_hit_select_multi(const int32_t* ids, const int32_t* res_row, int n, int n_expert, int32_t* slot, int32_t* dst,
                          int32_t* count, void* stream);
void moe_hit_grouped_s2_multi(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                              const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                              const float* x_scales, int k_per_token, void* scratch, float* out, void* stream);
/// The per-hit kernels (`moe_hit_grouped_s2`, `_dev`, `_multi`) and the grouped ones (`moe_grouped_s2`)
/// were rewritten with bitwise-identical outputs: conflict-free staged activations, `hx` once per chunk, wide loads,
/// two rows per warp.  The previous kernels stay for A/B: `STRATA_OLD_GROUPED=1` in the environment selects them,
/// and so does `moe_grouped_select_old(1)`; `0` forces the new ones, `-1` returns to the environment.  The environment
/// variable is read once, on first use; the kernel choice is made at each launch (so a captured graph keeps the
/// kernels chosen at capture).  Use `moe_grouped_select_old` to switch at runtime.  `STRATA_GROUPED_PAIR_MIN_HITS=N`
/// (read once too) keeps the previous per-hit kernels below N hits of capacity; a forced choice overrides it.  The
/// new kernels also need 4-byte aligned activations and scratch, and the per-hit ones an 8-byte aligned arena and
/// slot size; otherwise the previous kernels run.
void moe_grouped_select_old(int old);
/// Which kernels the last call of the entry points above launched: 1 = new, 0 = previous, -1 = none since the
/// previous query (the query clears it).  For the parity test, which must know the path it compared was taken.
int moe_grouped_last_path();
void moe_hit_grouped_s2_cpu_order(const uint8_t* blob_base, const int32_t* slot_index,
                                 const int32_t* dst_index, int64_t n_hits, int64_t blob_bytes,
                                 const uint8_t* x_q8_0, void* scratch, float* out, void* stream,
                                 const float* x_scales, float* gate_up_trace = nullptr);

}  // namespace strata::kernels
