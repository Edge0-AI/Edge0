// include/strata/platform/host_limits.hpp - host-side limits that need no GPU to check (#1630, #1607).
//
// Windows HIP, RX 7900 XTX (#1630): the first long prompt at --prefill 8192 / auto made the system commit charge jump
// ~70 GB inside a second and the engine died (0xC0000409); 7500 and below stayed flat (reporter's table, 26,902- and
// 66,235-token prompts), so a bare `auto` stops at 6144 on that platform (645-708 tok/s there, 7500 gave 748-782).
// A chunk the user asks for is theirs: it stays, with a warning.
//
// Windows commit limit (#1607): RAM + page file; the engine died silently when the charge reached it with GBs of
// physical RAM free. The startup check compares what the engine plans to allocate on the host with the commit that
// is left.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::platform::host_limits {

constexpr int64_t kWinHipAutoChunkCap = 6144;
constexpr int64_t kWinHipCrashChunk = 8192;   // the smallest chunk reported to crash

/// The ceiling a bare `--prefill auto` takes on this build; 0 = none (keep the usual 8192).
/// win_hip: a Windows HIP build; bare_auto: "auto" without ":N" and no STRATA_PREFILL_AUTO_MAX.
constexpr int64_t auto_chunk_cap(bool win_hip, bool bare_auto) {
    return win_hip && bare_auto ? kWinHipAutoChunkCap : 0;
}

/// Whether to warn about a chunk the user chose: a fixed N, or auto:N, of 8192 or more on a Windows HIP build.
constexpr bool warn_user_chunk(bool win_hip, bool bare_auto, int64_t chunk) {
    return win_hip && !bare_auto && chunk >= kWinHipCrashChunk;
}

struct CommitVerdict {
    bool warn = false;
    std::string text;
};

/// avail: commit left (ullAvailPageFile); limit: the commit limit (ullTotalPageFile); planned: host bytes the engine
/// plans to allocate from here on (conversation cache budget, checkpoints, staging); floor: kept free for Windows and
/// other programs. Warns when planned + floor exceeds what is left.
inline CommitVerdict commit_verdict(uint64_t avail, uint64_t limit, uint64_t planned, uint64_t floor_bytes) {
    CommitVerdict v;
    if (limit == 0) return v;   // unknown (not Windows)
    if (planned + floor_bytes <= avail) return v;
    v.warn = true;
    const double gib = 1073741824.0;
    char buf[400];
    std::snprintf(buf, sizeof buf,
                  "WARNING the Windows commit limit is %.1f GiB (RAM + page file) and %.1f GiB of it is left, but this run "
                  "plans %.1f GiB of host allocations and keeps %.1f GiB free: it can end at the limit with no message "
                  "(exit code 0xC0000409, #1607) while RAM is still free. Raise the page file (System managed, or "
                  "RAM-sized) or lower --conversation-cache-mib / --max-context / --prefill",
                  (double) limit / gib, (double) avail / gib, (double) planned / gib, (double) floor_bytes / gib);
    v.text = buf;
    return v;
}

}  // namespace strata::platform::host_limits
