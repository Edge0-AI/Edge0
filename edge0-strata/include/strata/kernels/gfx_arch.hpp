// include/strata/kernels/gfx_arch.hpp - which AMD architecture names the gfx11 matrix-core code (and the gfx12 native
// prompt-expert kernels) was built for.
//
// The WMMA kernels select their intrinsic with `#if defined(__gfx1100__) || ... || defined(__gfx1151__)`: a gfx11 part
// outside that list (gfx1152, ...) gets an empty or trapping body.  The runtime gates therefore match the SAME
// list, not the "gfx11" prefix, so such a part falls back to hipBLASLt / the portable kernels instead of launching code
// that is not there.  Pure string matching (no HIP headers), usable from host code of any build.
#pragma once

#include <cstddef>
#include <cstring>

namespace strata::kernels {

/// `gcn` is hipDeviceProp_t::gcnArchName ("gfx1151", or "gfx1151:sramecc-:xnack-"): true when it is exactly `name`.
inline bool gfx_arch_is(const char* gcn, const char* name) {
    if (gcn == nullptr) return false;
    const std::size_t n = std::strlen(name);
    return std::strncmp(gcn, name, n) == 0 && (gcn[n] == '\0' || gcn[n] == ':');
}

/// The gfx11 targets the matrix-core kernels (prompt GEMM / attention / scorer / fused experts) are compiled for.
inline bool gfx_arch_is_gfx11_wmma(const char* gcn) {
    return gfx_arch_is(gcn, "gfx1100") || gfx_arch_is(gcn, "gfx1101") || gfx_arch_is(gcn, "gfx1102") || gfx_arch_is(gcn, "gfx1103") ||
           gfx_arch_is(gcn, "gfx1150") || gfx_arch_is(gcn, "gfx1151");
}

/// The gfx12 (RDNA4) targets the gfx12 matrix-core kernels are compiled for (`#if defined(__gfx1200__) ||
/// defined(__gfx1201__)`): the R9700 / 9070 (XT) are gfx1201, the 9060 (XT) gfx1200.
inline bool gfx_arch_is_gfx12_wmma(const char* gcn) { return gfx_arch_is(gcn, "gfx1200") || gfx_arch_is(gcn, "gfx1201"); }

/// Strix Halo (Ryzen AI Max, RDNA3.5): the part the gfx1151 defaults (docs/STRIX_HALO.md) are measured on.
inline bool gfx_arch_is_gfx1151(const char* gcn) { return gfx_arch_is(gcn, "gfx1151"); }

/// RDNA4 (Radeon AI PRO R9700 = gfx1201, RX 9070 / 9070 XT = gfx1201, RX 9060 = gfx1200).
inline bool gfx_arch_is_gfx12(const char* gcn) { return gfx_arch_is(gcn, "gfx1200") || gfx_arch_is(gcn, "gfx1201"); }

}  // namespace strata::kernels
