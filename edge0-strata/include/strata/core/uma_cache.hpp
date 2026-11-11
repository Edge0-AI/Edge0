// include/strata/core/uma_cache.hpp - sizing the GPU expert cache on a device whose memory IS the system RAM.
//
// On a Strix Halo (or any APU) "free VRAM" is the GPU-addressable pool: the BIOS carve-out plus the GTT, up to nearly all of the RAM when
// the kernel was told ttm.pages_limit (docs/STRIX_HALO.md recommends 112 GiB of a 128 GiB box).  The expert arena (every expert, in RAM) is
// already counted in what the OS has used, but the cache copies experts INTO that same RAM, so a cache sized from the pool alone can ask for
// more bytes than the OS has left - the OOM killer then takes the engine, or the desktop.  The auto cache is therefore sized from
// min(what the GPU says is free, the RAM the OS says is available - a reserve for the desktop).  Only the AUTO size: --expert-cache N is
// kept as asked (the engine warns).  STRATA_UMA_RESERVE_GIB=N sets the reserve (0 = no reserve: the old sizing).
#pragma once

#include <cstdint>
#include <cstdlib>

namespace strata::core {

/// The RAM an integrated GPU's expert cache leaves to the OS and the desktop: STRATA_UMA_RESERVE_GIB when set (0 = none), else the larger of
/// 8 GiB and 8% of the installed RAM (a desktop session with a browser needs ~6 GiB; 8% is ~10 GiB on 128 GB).
inline uint64_t uma_desktop_reserve_bytes(uint64_t total_ram, const char* env_gib = std::getenv("STRATA_UMA_RESERVE_GIB")) {
    constexpr uint64_t GiB = 1ull << 30;
    if (env_gib != nullptr && env_gib[0] != '\0') {
        char* end = nullptr;
        const double v = std::strtod(env_gib, &end);
        if (end != env_gib && v >= 0.0) return (uint64_t) (v * (double) GiB);
    }
    const uint64_t pct = total_ram / 100 * 8;
    return pct > 8 * GiB ? pct : 8 * GiB;
}

/// The bytes the auto cache may use: the RAM available now, less the reserve (0 when that is already gone).
inline uint64_t uma_cache_room_bytes(uint64_t available, uint64_t reserve) {
    return available > reserve ? available - reserve : 0;
}

}  // namespace strata::core
