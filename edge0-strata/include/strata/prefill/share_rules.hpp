// include/strata/prefill/share_rules.hpp - when the default CPU prefill share is worth arming (no GPU needed to check).
//
// The share hands the idle CPU pool experts the GPU would otherwise stream over PCIe.  With every expert resident in
// the GPU's cache nothing is streamed, so there is nothing to take; arming the default only moves the staged-chunk
// limit and costs ~5% at 1,000 tokens (#1595: V100, 3 rounds x 3 runs).  An explicit STRATA_PREFILL_CPU_SHARE is
// honoured by Prefill::arm_cpu_share whatever this says.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::prefill::share_rules {

/// resident_slots: the expert cache's live slots; total_pairs: layers x routed experts per layer (<= 0: unknown, arm).
constexpr bool default_share_has_work(int64_t resident_slots, int64_t total_pairs) {
    return total_pairs <= 0 || resident_slots < total_pairs;
}


// ---- memory (0.1.41 report, RTX 5090, tight RAM: chats failed and the engine restarted until STRATA_PREFILL_CPU_SHARE=0) ----
// The default share reads experts with the CPU and keeps two pinned buffers on the host.  When the host has little
// memory left, or the experts it would read are file pages the RAM cannot keep (each is a page-in on a worker thread),
// the default is not armed.  An explicit STRATA_PREFILL_CPU_SHARE is the user's: it is armed and the engine only warns.
constexpr uint64_t kShareRamMargin = 3ull << 30;     // RAM kept free beside the share's own buffers and the file pages
constexpr uint64_t kShareCommitFloor = 4ull << 30;   // Windows commit kept free (the #1607 floor)
constexpr uint64_t kUnknown = ~uint64_t{0};

struct ShareMemory {
    uint64_t avail_ram = kUnknown;       ///< RAM this process can get (cgroup-aware); kUnknown: not known
    uint64_t avail_commit = kUnknown;    ///< Windows commit left (RAM + page file); kUnknown elsewhere
    uint64_t buffer_bytes = 0;           ///< host bytes the share itself allocates (pinned buffers, scratch)
    uint64_t inplace_bytes = 0;          ///< expert bytes the CPU would read as file pages (not in the GPU cache, not in RAM)
};

struct ShareVerdict {
    bool arm = true;
    std::string why;   ///< when !arm: what the engine says, in one line
};

inline ShareVerdict default_share_memory_verdict(const ShareMemory& m) {
    ShareVerdict v;
    const double gib = 1073741824.0;
    char buf[300];
    if (m.avail_ram != kUnknown && m.avail_ram < m.buffer_bytes + kShareRamMargin) {
        std::snprintf(buf, sizeof buf, "only %.1f GiB of RAM is free (the share needs %.2f GiB and %.1f GiB left over)",
                      (double) m.avail_ram / gib, (double) m.buffer_bytes / gib, (double) kShareRamMargin / gib);
        v.arm = false;
    } else if (m.avail_commit != kUnknown && m.avail_commit < m.buffer_bytes + kShareCommitFloor) {
        std::snprintf(buf, sizeof buf, "only %.1f GiB of the Windows commit limit is left (the share needs %.2f GiB and %.1f GiB left over)",
                      (double) m.avail_commit / gib, (double) m.buffer_bytes / gib, (double) kShareCommitFloor / gib);
        v.arm = false;
    } else if (m.inplace_bytes > 0 && m.avail_ram != kUnknown && m.inplace_bytes + kShareRamMargin > m.avail_ram) {
        std::snprintf(buf, sizeof buf, "the experts it would read are %.1f GiB of file pages and only %.1f GiB of RAM is free to keep them",
                      (double) m.inplace_bytes / gib, (double) m.avail_ram / gib);
        v.arm = false;
    }
    if (!v.arm) v.why = buf;
    return v;
}

}  // namespace strata::prefill::share_rules
