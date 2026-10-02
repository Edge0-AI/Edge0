// Process memory sample for e0_stats: resident (incl. file-backed) vs phys_footprint.
#pragma once
#include <cstdint>

namespace e0n {

struct MemSample {
  uint64_t resident_peak_bytes = 0;  // peak resident_size, including file-backed pages
  uint64_t resident_bytes = 0;       // current resident_size (not in ABI stats)
  uint64_t footprint_bytes = 0;      // current phys_footprint (Activity Monitor "Memory")
};

// Current process; zeros if rusage fails (caller must tell "unmeasured" from "measured 0").
MemSample mem_sample();

}  // namespace e0n
