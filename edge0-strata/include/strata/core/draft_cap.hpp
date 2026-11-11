// include/strata/core/draft_cap.hpp - the widest verify window --draft dflash2 uses on each card class (measured A/Bs, notes-dflash2-p1.md; recommend, never force).
#pragma once

#include <cstddef>

namespace strata::core {

/// 0 = no cap (the --spec given stays).  `uma`: Strix Halo / iGPU (gfx1151); `cc_major`: the CUDA compute capability's major (0 on HIP).
///  - Strix Halo / UMA: 5 (agent-safe; 6 ties on agent prompts)
///  - Blackwell (sm_120, cc 12): 6 (+1.9% at 6 on an RTX PRO 5000, 8 was worse)
///  - everything else: unmeasured, --spec as given
inline int dflash2_auto_cap(bool uma, int cc_major) {
    if (uma) return 5;
    if (cc_major == 12) return 6;
    return 0;
}

/// A 12 GB NVIDIA card (< 13 GiB, CUDA): the MTP drafter was faster in most measurements, which the engine says once.
inline bool dflash2_small_nvidia(bool cuda, size_t total_bytes) { return cuda && total_bytes < ((size_t) 13 << 30); }

}  // namespace strata::core
