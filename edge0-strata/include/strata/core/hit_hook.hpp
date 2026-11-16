// include/strata/core/hit_hook.hpp - R4.2c: the seam between the host loop and the VRAM expert tier.
//
// **IT IS ITS OWN HEADER BECAUSE BOTH SIDES NEED IT AND NEITHER SHOULD OWN IT.**  `session.hpp` declares the
// loop that calls the hook; `expert_source.hpp` declares the implementation that answers it.  Putting the type
// in either one makes the other include a header it has no other use for, and the loop knowing what an expert
// cache is (or the adapter knowing what a graph is) is exactly the coupling this seam exists to avoid.
#pragma once

#include <cstdint>

namespace strata::core {

/// **TWO PHASES, AND THE SPLIT IS THE WHOLE POINT.**  `Launch` runs BEFORE the CPU pool and `Combine` runs
/// AFTER the misses have been copied into `parts`.  That ordering is what lets the GPU's resident experts run
/// at the same time as the CPU's misses instead of after them:
///
///     ring -> hits(Launch)   [GPU: quantize + grouped experts into `hit_out`]
///          -> pool()         [CPU: the misses, CONCURRENT with the GPU]
///          -> H2D misses into `parts`
///          -> hits(Combine)  [GPU: parts += hit_out]
///          -> post[l]
///
/// **MEASURED, AND WHY IT IS NOT THE OBVIOUS ORDER.**  The first version called the hit path once, after the
/// pool, writing straight into `parts`.  The drain duly fell **18.2 -> 10.2 ms** - the CPU really was doing
/// half the experts - and **the token did not move at all**, because the GPU's half had simply been moved onto
/// the critical path behind the CPU's instead of beside it.  A second buffer and one `add_inplace` buy the
/// overlap back.
enum class HitPhase { Launch, Combine };

/// `ids` AND `k` ARE THE ROUTER'S OWN OUTPUT, AND THEY ARE WHY THIS HOOK TAKES ARGUMENTS AT ALL.
///
/// The first version took only `(user, stream, phase)` and left the hit/miss decision inside the pool
/// callback - which runs AFTER `Launch`.  So `Launch` read the hit list from the PREVIOUS layer and computed
/// the previous layer's experts into the current layer's rows.  It produced a token, the timings looked
/// plausible, and **C1 moved from mean KL 9.69e-02 to 1.03e+00 and top-1 from 0.867 to 0.333.**  The decision
/// has to happen in `Launch`, on the ids the doorbell published for THIS layer, and the pool then consumes it.
///
/// A null `HitFn` means "no VRAM tier" - the default, and what every caller had before R4.
using HitFn = void (*)(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k);

}  // namespace strata::core
