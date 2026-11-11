#pragma once

#include <cstdint>
#include <cstdlib>
#include <optional>

#include "strata/core/conversation_memory.hpp"

namespace strata::core {

/// The 0.2.7 residency policy for the PLE (n-gram) table: keep the whole table in RAM, or leave it on direct reads?
///
/// The table is not the only thing that wants RAM.  The engine copies every routed expert into a pinned host arena at
/// start (about 51 GiB for IQ3_S), after the table has been opened, and an unbuffered read into that arena needs the
/// pages locked.  A locked table takes its share of the lockable memory first, so on a 64 GB PC (Windows, RTX 5070,
/// 2026-10-26) the policy that only looked at "table + 8 GiB available" admitted the table and the expert load then
/// failed with ERROR_WORKING_SET_QUOTA (1453).  The table is therefore admitted only when RAM holds the table AND the
/// arena the engine will still fill AND the headroom, all measured against what is available BEFORE either is
/// allocated.  Unknown memory telemetry, an unknown table size or an unknown arena size admit nothing (the direct
/// reads of 0.2.6 stay).
///
/// `experts_ram_bytes`: the native expert layout's total (every layer's blob x experts), 0 when it is not known.
inline bool ple_resident_admit(std::optional<uint64_t> available, uint64_t table_bytes, uint64_t experts_ram_bytes,
                               uint64_t headroom_bytes) {
    if (!available || table_bytes == 0 || experts_ram_bytes == 0) return false;
    if (table_bytes > UINT64_MAX - experts_ram_bytes) return false;
    return conversation_memory_admit(available, table_bytes + experts_ram_bytes, headroom_bytes);
}

/// May the residency policy run at all?  STRATA_PLE_AUTO_RAM when it is set (0 off, anything else on: an explicit value wins), otherwise on
/// except under STRATA_DEFAULTS=026 (`v026`: v0.2.6 had no such policy, the direct reads stayed).
inline bool ple_auto_ram_allowed(const char* env_value, bool v026) {
    if (env_value != nullptr) return std::atoi(env_value) != 0;
    return !v026;
}

/// #4518: the table was locked at start on the policy's word (table + arena + headroom fit), but what the rest of the engine takes after
/// that (the KV pools, the pinned staging, the GPUs' host buffers) is not known then.  On a 128 GB two-GPU Windows PC it left less than the
/// 5 GiB the automatic conversation cache needs, and the cache turned itself off.  The cache is worth more than the lock (parking a 100 K
/// conversation is 1.1 s against 30 s to read it again, the lock was measured at 2-3% of prompt speed), so once the cache is to be sized:
/// does giving the lock back make the automatic cache bigger?  `available` is the RAM the cache sees now (the table locked), the table's
/// pages count as available again once unlocked (clean file pages).  Unknown telemetry: no.
inline bool ple_lock_blocks_cache(std::optional<uint64_t> available, uint64_t table_bytes, int64_t min_free_mib) {
    if (!available || table_bytes == 0 || *available > UINT64_MAX - table_bytes) return false;
    return auto_conversation_cache_mib(*available + table_bytes, min_free_mib) > auto_conversation_cache_mib(available, min_free_mib);
}

} // namespace strata::core
