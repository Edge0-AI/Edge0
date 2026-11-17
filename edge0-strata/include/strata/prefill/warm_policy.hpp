// include/strata/prefill/warm_policy.hpp - whether the HIP start-up warm-up of hipBLAS (prefill.cpp warm_gemm_libraries)
// may run.  Pure, so it is unit tested without a GPU (warm_policy_test).
//
// #3204: on a layer split across a gfx1151 and a gfx1201 card, prompts read ~25% slower with the warm-up (1,250 against
// 1,710 tok/s at 64K; STRATA_PREFILL_WARM=0 restores it; hc read of the gfx1151 stage 2.7 -> 9.0 s a 32K block).  The
// same warm-up on two cards of ONE architecture (2x R9700) costs nothing (021-iD, interleaved pairs).  Cards of different
// architectures share one process-wide hipBLAS / hipBLASLt library state, and the warm-up of the second architecture runs
// before the first card's own tuned products resolve their solutions.  So: no warm-up when the process sees more than one
// architecture (the first prompt pays the one-time 1-2.5 s load, as before 0.2.0), unless STRATA_PREFILL_WARM=1 asks.
#pragma once

#include <string>
#include <vector>

namespace strata::prefill {

/// `gcn_archs`: gcnArchName of every visible HIP device (a ":sramecc+:xnack-" feature suffix is ignored);
/// `env`: the value of STRATA_PREFILL_WARM or nullptr.  "0" turns the warm-up off, "1" on whatever the cards are.
inline bool warm_gemm_wanted(const std::vector<std::string>& gcn_archs, const char* env) {
    if (env != nullptr && env[0] == '0') return false;
    if (env != nullptr && env[0] == '1') return true;
    if (gcn_archs.empty()) return true;
    const auto base = [](const std::string& a) { return a.substr(0, a.find(':')); };
    const std::string first = base(gcn_archs.front());
    for (const std::string& a : gcn_archs)
        if (base(a) != first) return false;
    return true;
}

}  // namespace strata::prefill
