// include/strata/core/auto_cache.hpp - the slot count of the automatic expert cache (--expert-cache auto), as a rule with a test.
//
// The auto cache is sized from the VRAM free after the weights, the session and the drafter, minus --vram-reserve-mib (the reserve is what
// the operator wants LEFT once everything is loaded), the draft head that binds later (priced), the prompt path's own buffers when it does
// not borrow cache slots (priced) and the second verifier of --pipeline-windows (priced).  What was never priced is what the load allocates
// AFTER the cache whatever the settings: the CUDA libraries the prompt path and the verifier warm up (cuBLAS handles and workspaces, the
// lazily loaded modules) and the verifier's small buffers.  Measured with STRATA_TRACE=1 on the same build, config and flags
// (--vram-reserve-mib 700, 256K context, MTP, 4 drafts):
//   RTX A4000 16 GB (z440, IQ3_XXS):  883 MiB free after the cache -> 719 after the head and the prompt path (-164) -> 461 after the
//                                    verifier and the drafter's binding (-258 = 178 MiB draft head, priced, + 80)   = 244 MiB unpriced
//   RTX A6000 48 GB (boston, IQ3_S):  919 -> 755 (-164) -> 459 (-296 = 213 MiB draft head, priced, + 83)            = 247 MiB unpriced
//   --prefill 512 / 2048 / auto on the A4000: the same 164 + 80.  v0.2.6 ends at 461 MiB on the same card: not new, but every 16 GB card with
//   the default 700 MiB ends under the 512 MiB line the engine itself warns about (#2065, #4474), and the warning cannot be cured by the
//   operator raising the reserve by the advice's figure without losing the cache it was meant to keep.
// So the reserve is now honest: the cache leaves `reserve + kLateAllocMib`, and the card ends with about the reserve.
#pragma once

#include <cstdint>

namespace strata::core {

/// MiB the load allocates after the cache that no setting prices (see above; measured 244-247, a little margin).
constexpr int64_t kLateAllocMib = 256;

struct AutoCacheSlots {
    int64_t slots = 0;       ///< the cache's slot count (of the largest blob), >= 0
    bool late_priced = false; ///< kLateAllocMib was left out of the cache
};

/// `free_b`: VRAM free now; `reserve_b`: the reserve and everything else that is priced (bytes); `blob`: the largest slot; `min_slots`: the least a working
/// cache needs (#496).  `late_mib` 0 turns the late allocation off (STRATA_DEFAULTS=026).  The late allocation is only left out when the cache still has
/// `min_slots` after it: a small card keeps the sizing it had (its reserve is lowered further elsewhere, #496), never a smaller cache than it can work with.
inline AutoCacheSlots auto_cache_slots(int64_t free_b, int64_t reserve_b, int64_t blob, int64_t min_slots, int64_t late_mib = kLateAllocMib) {
    AutoCacheSlots r;
    if (blob <= 0) return r;
    const int64_t plain = (free_b - reserve_b) / blob;
    const int64_t late = late_mib > 0 ? (free_b - reserve_b - (late_mib << 20)) / blob : plain;
    if (late_mib > 0 && late >= min_slots && late > 0) {
        r.slots = late;
        r.late_priced = true;
    } else {
        r.slots = plain > 0 ? plain : 0;
    }
    return r;
}

}  // namespace strata::core
