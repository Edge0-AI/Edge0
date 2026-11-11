// include/strata/platform/memory.hpp - plan v0.3 P0.1/P1: keep a large host region resident.
//
// `cudaHostRegister` refuses the 31.6 GiB expert arena on Windows, and an unlocked arena is trimmed under memory
// pressure (the CPU pool then swung 2x between runs). With the n-gram table out of RAM there is headroom to lock
// it instead: raise the process's minimum working set by the region's size, then VirtualLock it (Windows needs
// only SeIncreaseWorkingSetPrivilege, which ordinary accounts hold). Linux: mlock.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::platform {

struct LockResult {
    bool ok = false;
    uint64_t locked_bytes = 0;   ///< may be less than requested; the rest stays pageable
    std::string note;            ///< what was done or why it failed, for the startup print
};

/// Lock [p, p + bytes) into physical memory. Partial success is reported, not hidden.
LockResult lock_resident(void* p, uint64_t bytes);

/// Pin-by-need: lock several disjoint regions (the holes between the registered ranges of an arena).  The Windows
/// working-set minimum is raised ONCE for the total (lock_resident per region would add its margin every time);
/// stops at the first refusal and reports the bytes locked so far.  Undo each region with unlock_resident.
LockResult lock_resident_ranges(const std::vector<std::pair<void*, uint64_t>>& regions);

/// Undo lock_resident for the same region (best effort).
void unlock_resident(void* p, uint64_t bytes);

/// #243, Windows: the GPU's shared (non-local) memory budget and this process's use of it, from DXGI
/// (IDXGIAdapter3::QueryVideoMemoryInfo, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL) for the adapter whose LUID is the
/// 8 bytes at `luid` (cudaDeviceProp::luid).  Page-locked host memory the GPU maps is charged there.  False (and
/// `why` says so) when the query is not possible - always elsewhere than Windows.
bool gpu_shared_memory_budget(const void* luid, uint64_t& budget, uint64_t& usage, std::string& why);

/// The machine's physical RAM in bytes (0 when unknown).
uint64_t total_physical_memory();

/// #357/#577: whether the OS file cache could keep the `read_bytes` the expert files are read for, beside
/// `arena_bytes` of RAM held by the engine's own copy of the experts and `margin` for everything else, with `avail`
/// bytes of RAM available.  The file tier passes only the expert bytes it really reads from the files (the experts
/// outside its resident RAM copy) and the RAM that copy really holds - not every shard's bytes and the requested
/// budget, which on a 96 GB PC (#577) made the file tier read unbuffered when the cache could keep its reads.
/// #1194: the file tier's reads are a skewed set (the GPU cache and the RAM copy took the hottest experts; the same
/// few of the rest come back token after token), so the cache does not have to hold all `read_bytes` to serve most of
/// them.  With `hot_fraction` < 1 it is enough that the room holds that share of them, but never less than
/// `floor_bytes` (below it the cache holds nothing worth having) and never more than every read.  The defaults ask for
/// every byte, as the Windows rule does.  Measured on Linux (#1194, a Tesla P100 box under 20 to 32 GB cgroups): through
/// the cache beat the unbuffered reads at every room tried, down to 7% of the read bytes (decode +7% to +118%, a third to
/// a sixth of the drive traffic).
inline bool file_cache_keeps(uint64_t avail, uint64_t arena_bytes, uint64_t read_bytes,
                             uint64_t margin = 4ull << 30, double hot_fraction = 1.0, uint64_t floor_bytes = 0) {
    const uint64_t room = avail > arena_bytes + margin ? avail - arena_bytes - margin : 0;
    uint64_t need = (uint64_t) ((double) read_bytes * hot_fraction);
    if (need < floor_bytes) need = floor_bytes;
    if (need > read_bytes) need = read_bytes;
    return room >= need;
}

/// #3237: with the room rule saying the file cache could keep the experts that come back, but holding only
/// `held_fraction` (< 0.5) of the read set: whether the unbuffered reads win, from a start-up timing of the drive both
/// ways (`direct_bps`: O_DIRECT preads on the I/O threads, `cached_bps`: first touches of a mapping's pages).
/// On a PCIe 5 NVMe (8.2 GB/s direct against 4.9 through the cache, the cache holding 13%) the cached path read prompts at
/// half the speed and decoded 3-4x slower than unbuffered reads (#3237).  On a 1.7-2.5 GB/s NVMe (ext4, notes-021-iD) the
/// cache decoded 1.1-1.6x faster at 10-60% held while direct reads read prompts 1.1-1.4x faster, so a timing ratio alone
/// (2.0x there, 1.66x on the reporter's) cannot separate the two: a miss costs 1/bandwidth, and only on a fast drive is
/// a miss cheap enough that the cache's faults and read-ahead cost more than the hits save.  Hence an absolute floor: the
/// direct path must run at 4 GB/s or more.  Then a cache that holds under 30% needs the direct path 1.3x faster; up to
/// half, 1.5x.
inline bool file_tier_prefers_direct(double held_fraction, double direct_bps, double cached_bps) {
    if (!(direct_bps > 0.0) || !(cached_bps > 0.0)) return false;
    if (held_fraction >= 0.5 || direct_bps < 4.0e9) return false;
    const double ratio = direct_bps / cached_bps;
    return held_fraction < 0.3 ? ratio >= 1.3 : ratio >= 1.5;
}

/// Whether `advise_willneed` asks the OS for anything: not on Windows, nor with STRATA_READ_AHEAD=0.
bool read_ahead_enabled();
/// Asks the OS to start reading [p, p + bytes) of a file mapping, without waiting.  Linux reads at most one
/// readahead window per request, so the range is asked for in 128 KiB steps.
void advise_willneed(const void* p, uint64_t bytes);
/// The same for [offset, offset + bytes) of an open file.
void advise_willneed(int fd, uint64_t offset, uint64_t bytes);

/// What the OS says this process read from storage: `read_bytes` (/proc/self/io: bytes fetched from the block layer on
/// its behalf, page faults on a mapping included, page-cache hits not) and `major_faults` (/proc/self/stat).  Linux
/// only; `valid` is false elsewhere.  Take one at the start of a request and one at its end.
struct ProcIo {
    bool valid = false;
    uint64_t read_bytes = 0, major_faults = 0;
};
ProcIo proc_io_sample();

}  // namespace strata::platform
