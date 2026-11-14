// metal/src/engine/stubs_023.cpp - the two research hooks of the shared Verifier::record_window that only the CUDA / HIP backends implement
// (src/kernels/cuda/verify_kernels.cu): hc_mean_streams (STRATA_TAP_FILE) and route_marginal_stats (STRATA_ROUTE_STATS).  Verifier::record_window
// does not run on Metal (init hands over to the Metal window, metal/src/engine/window.cpp), so neither is reached; if one ever is, it stops
// the engine with its name instead of continuing on outputs nobody wrote.  They exist so the shared host file links.

#include <cstdint>

#include "strata/kernels/verify_kernels.hpp"

namespace strata::metal_engine { void missing(const char* name); }

namespace strata::kernels {
void hc_mean_streams(const float*, float*, int64_t, int, void*) { strata::metal_engine::missing("strata::kernels::hc_mean_streams"); }
void route_marginal_stats(const int32_t*, int, int, unsigned long long*, void*) { strata::metal_engine::missing("strata::kernels::route_marginal_stats"); }
}  // namespace strata::kernels
