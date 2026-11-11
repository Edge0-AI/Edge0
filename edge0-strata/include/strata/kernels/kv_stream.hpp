// include/strata/kernels/kv_stream.hpp - KV streaming for the QSA layers (docs/kv-streaming-design.md).
//
// A streamed QSA layer keeps its AUTHORITATIVE K/V in pinned, device-mapped host memory, laid out exactly as a
// fully resident pool with the identity page table would be. VRAM holds a pool of `n_slots` pages (one page =
// `page_size` = 4 cells = one indexer block, all KV heads, K and V: 4,224 B in int8), and the page table becomes a
// RESIDENCY MAP: `page_table[block]` is the slot holding that block, or -1.
//
// Every reader (`qsa_decode_attn_*`, `kv_gather_*`) is unchanged: it goes through the page table as before. What
// changes is that after the selection and before the attention, `kv_stream_resolve` makes every block the
// selection names resident - on device, inside the captured graphs:
//   1. one block of 1,024 threads walks the selected ids: a resident block gets this call's epoch and its clock
//      reference bit, a missing block is claimed once (page_table -1 -> -2) into a miss list;
//   2. the same block picks one victim slot per miss with a CLOCK sweep (second chance) that never takes a slot
//      this call uses, re-points the page table (old block -> -1, new block -> slot);
//   3. a copy kernel reads the missed blocks from the host copy (zero-copy, over PCIe) into their slots.
// Writers (`kv_append_*`) write the host copy always, and the slot only if the block is resident, so the slots
// never go stale and nothing needs invalidating - not after a rejected draft, not on a conversation-cache resume.
//
// A RING (the MTP drafter, which only ever reads its last `window` cells) uses the same host copy with a static
// table `block -> block % n_slots` and no resolve: the ring is refilled from the host copy on a resume.
#pragma once

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

/// The host copy of one layer's K/V: device-mapped pointers (UVA) into pinned memory, identity layout
/// `[block][kv_head][page_size][head_dim]`. All null when the layer is fully resident.
struct KvHostPools {
    uint16_t* k_pool = nullptr;   ///< fp16 mode
    uint16_t* v_pool = nullptr;
    int8_t* k_q = nullptr;        ///< int8 mode: codes
    int8_t* v_q = nullptr;
    uint16_t* k_scale = nullptr;  ///< int8 mode: fp16 scale per 64 values
    uint16_t* v_scale = nullptr;
    uint8_t* k_q4 = nullptr;      ///< q4_0 mode (kv_q4.hpp): block_q4_0 codes, 144 B per cell and head
    uint8_t* v_q4 = nullptr;
    bool present() const { return k_pool != nullptr || k_q != nullptr || k_q4 != nullptr; }
};

/// The KV storage format, for the functions below that move whole blocks (`fmt`): fp16, int8 (+ scales), q4_0, and
/// K8V4 (`--kv k8v4`: K as int8 codes + scales, V as rotated q4_0 - three runs, `k_q`, `k_scale`, `v_q4`).
/// (A bool `int8` argument still reads as kKvF16 / kKvInt8.)  kKvHybrid is 3, the number conversation snapshots
/// already give K8V4.
enum KvFormat : int { kKvF16 = 0, kKvInt8 = 1, kKvQ4 = 2, kKvHybrid = 3 };

/// K8V4's halves as the single-format append kernels see them, so a hybrid layer's host copy (and the prompt path's
/// staging pool) is written by the same calls that write its VRAM pools: K is "int8 whose V is K", V is "q4_0 whose
/// K is V" - the folded duplicate the hybrid appends already make (layer.cpp).  Both fields of a half point at the
/// same array on purpose: the kernels test the K-side pointer for presence and pick K or V per lane.
inline KvHostPools kv_hybrid_k_half(const KvHostPools& h) {
    KvHostPools r;
    r.k_q = h.k_q; r.v_q = h.k_q; r.k_scale = h.k_scale; r.v_scale = h.k_scale;
    return r;
}
inline KvHostPools kv_hybrid_v_half(const KvHostPools& h) {
    KvHostPools r;
    r.k_q4 = h.v_q4; r.v_q4 = h.v_q4;
    return r;
}

/// The residency map of a streamed layer, all device memory at fixed addresses (the graphs bake them in).
struct KvStreamMap {
    int32_t* page_table = nullptr;  ///< (n_blocks,) block -> slot, or -1
    int32_t* slot_block = nullptr;  ///< (n_slots,) slot -> block, or -1
    int32_t* slot_stamp = nullptr;  ///< (n_slots,) the resolve epoch that last used the slot
    int32_t* slot_ref = nullptr;    ///< (n_slots,) clock reference bit
    int32_t* ctl = nullptr;         ///< kKvCtlInts: epoch, hand, misses of the last call, overflow, u64 counters
    int32_t* miss_block = nullptr;  ///< (n_slots,)
    int32_t* miss_slot = nullptr;   ///< (n_slots,)
    int64_t n_blocks = 0;
    int64_t n_slots = 0;
};

inline constexpr int kKvCtlInts = 16;   ///< [0] epoch [1] hand [2] misses [3] overflow; u64 at [4] misses, [6] lookups, [8] calls
/// Bytes of the map's arrays besides the page table (which the state already has): 5 per-slot ints + ctl.
inline uint64_t kv_stream_map_bytes(int64_t n_slots) { return (uint64_t) n_slots * 4 * 5 + kKvCtlInts * 4; }

/// Every block evicted: page table all -1, slots free, counters zero.
void kv_stream_reset(const KvStreamMap& m, void* stream);

/// Make every block named by the selections of `n_q` queries resident (ids [n_q][cap], width from
/// steps[q * kStepCount + kStepWidth]). Capturable. `n_slots` must hold the distinct blocks of one call
/// (n_q x (cap / page_size + 2)); a call that cannot sets ctl[3] (see `kv_stream_counters`).
void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
                       const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s,
                       void* stream);

/// The static ring table `block -> block % n_slots`.
void kv_ring_table(int32_t* page_table, int64_t n_blocks, int64_t n_slots, void* stream);

/// Copy blocks [b0, b1) of the host copy into the pool pages `page_table` names (host-side table `phys(b)`
/// computed as `b % n_slots`: the ring restore). Not capturable.
void kv_ring_restore(const QsaAttnPools& slots, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                     int64_t n_slots, const QsaShapes& s, void* stream);

/// Copy the first `n_blocks` blocks of the host copy into a fully resident (identity-layout) pool: the prompt
/// path's staging of one layer. Not capturable (DMA).
void kv_stage_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t n_blocks,
                        const QsaShapes& s, void* stream);

/// #579 #613 (HIP A/B, STRATA_KV_HOST_DMA=1): the reverse of kv_stage_from_host for blocks [b0, b1) - the staging
/// pool's blocks into the host copy, by DMA, so the prompt path's append need not write host memory from a kernel.
/// Not capturable.
void kv_unstage_to_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                        const QsaShapes& s, void* stream);

struct KvStreamCounters {
    uint64_t misses = 0, lookups = 0, calls = 0;
    bool overflow = false;
};
/// Synchronous read of the counters (debug and the end-of-request summary).
KvStreamCounters kv_stream_counters(const KvStreamMap& m);

/// Bytes of one block (page) of K and V together.
uint64_t kv_block_bytes(const QsaShapes& s, int fmt);

}  // namespace strata::kernels
