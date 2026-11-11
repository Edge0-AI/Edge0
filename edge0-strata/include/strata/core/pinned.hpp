// include/strata/core/pinned.hpp - P2.S1: the host arena the expert weights live in.
//
// 33.97 GB of expert weights cannot fit in a 12 GB card, so they stay in host memory and are streamed.  That
// makes this arena the engine's real working set: it must be PAGE-LOCKED for the copy engine to reach full
// bandwidth, and it must say which backing it got, because the two options differ by more than twice in TLB
// reach:
//
//   * LARGE PAGES (2 MB) - `mmap(MAP_HUGETLB)` / hugetlbfs on Linux, `VirtualAlloc(MEM_LARGE_PAGES)` on
//     Windows.  33.97 GB at 4 KB pages is 8.3 million TLB entries, which does not fit in any TLB, so every
//     block of every expert matvec takes TLB misses.
//   * NORMAL PAGES - the fallback.  Correct, slower, and it must be REPORTED rather than silently accepted:
//     "the engine adapts to the machine it is on" is only true if the engine says what it got.  On Windows
//     large pages additionally need SeLockMemoryPrivilege, which a normal user account does not have, so this
//     fallback is the common case and not an error path.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

enum class PageBacking { LargePages, NormalPages, PinnedByCuda };

/// #243: STRATA_ARENA_PIN_GIB, the cap on the expert arena's CUDA registration in GiB.  -1 when unset (the engine
/// decides, as in 0.1.30), 0 = no cap (the whole arena, or as many slices as the driver takes), N > 0 = at most N GiB,
/// -2 for "auto" (Windows: the sliced pin stays below the GPU's shared-memory budget).
int arena_pin_cap_gib();

/// 153-igpu, UMA mode: the engine sets this on an integrated GPU whose firmware cannot survive a KFD queue eviction (the Radeon
/// 780M, gfx1103).  The expert arena is then NOT registered with the GPU driver: a registered range is a userptr that the
/// kernel's page migration (compaction, reclaim) invalidates, and every invalidation evicts the process's GPU queues (MES
/// REMOVE_QUEUE), which resets that GPU while a dispatch sits in a barrier.  The arena stays resident through the working-set
/// lock (no experts are reachable over "PCIe": there is none), and the GPU gets what it needs by copy.
inline std::atomic<bool>& arena_uma_no_register_flag() {
    static std::atomic<bool> f{false};
    return f;
}
/// True when the arena is to stay unregistered: STRATA_ARENA_PIN_GIB=none always, otherwise the flag above unless the user set
/// STRATA_ARENA_PIN_GIB to a number (STRATA_ARENA_PIN_GIB=0 registers the whole arena as before).
bool arena_no_register();

struct PinnedArena {
    void* base = nullptr;
    uint64_t capacity = 0;
    PageBacking backing = PageBacking::NormalPages;
    std::string note;              // why the backing is what it is, for the startup print
    uint64_t locked_bytes = 0;     // resident via the working-set lock when CUDA could not pin it
    /// Plan v0.3 P5: when the whole arena cannot be registered, it is registered in `slice`-byte pieces from the
    /// start; this is the pinned prefix (a copy that stays inside one slice can then DMA straight from the arena).
    uint64_t registered_bytes = 0;
    uint64_t slice_bytes = 0;
    int registered_slices = 0;

    PinnedArena() = default;
    /// `slice`: the piece size for the per-slice registration fallback (0 = none).
    explicit PinnedArena(uint64_t bytes, uint64_t slice = 0);
    /// Plan v0.3 P6: slices of different sizes (one per layer of a native pack), given as their start offsets
    /// followed by the end of the last one.  `slice_starts` holds the registered ones.
    /// `max_pinned_bytes`: optional cap on CUDA registration. 0 preserves the normal unrestricted path.
    /// `shared_file`: on Linux, use a file-backed MAP_SHARED mapping instead of anonymous memory.
    /// `shared_pack_hash` identifies the pack that is allowed to populate that backing.  The file carries a
    /// small header and is refused when its stored hash does not match.  Empty `shared_file` preserves the
    /// existing allocation path.  Population/coordination and backing-file lifetime remain the caller's job.
    PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds, uint64_t max_pinned_bytes = 0,
                const std::string& shared_file = {}, uint64_t shared_pack_hash = 0);
    /// Pin by need (STRATA_PIN_BY_NEED=1): register only the given byte ranges (offset, length), in the given order
    /// (the caller's priority: the hottest experts the GPU does not hold first), each rounded out to 4 KiB pages and
    /// trimmed against the ranges already registered; stops at the first refusal of the driver (the WDDM budget is
    /// not raisable: we choose WHAT to pin), or when `max_pinned_bytes` (or STRATA_ARENA_PIN_GIB) would be passed.
    /// Everything not registered - the holes - is locked resident instead.  Registered before any page is touched,
    /// as the other constructors do.  `registered_bytes` is then the TOTAL registered, not a prefix: use
    /// `pin_ranges`.
    struct ByNeed {};
    PinnedArena(uint64_t bytes, ByNeed, const std::vector<std::pair<uint64_t, uint64_t>>& ranges,
                uint64_t max_pinned_bytes = 0, const std::string& shared_file = {}, uint64_t shared_pack_hash = 0);
    bool by_need = false;
    std::vector<std::pair<uint64_t, uint64_t>> pin_ranges;    ///< registered (offset, length), sorted by offset, page-rounded
    std::vector<std::pair<uint64_t, uint64_t>> locked_ranges; ///< the holes locked resident (offset, length)
    /// #285: reserve only; the caller registers the slices with register_slices(), on a thread, while its readers
    /// fill the ones already registered (one registration of the whole arena cost ~2.5 s at 33 GiB of 4 KB pages).
    struct Deferred {};
    PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds, Deferred);
    /// Registers the deferred slices in order, publishing how many are done in `ready`, and INT_MAX once it
    /// returns (a slice CUDA refuses ends the registration, and the rest is kept resident by the working-set lock).
    void register_slices(std::atomic<int>& ready);
    std::vector<uint64_t> bounds_;
    std::vector<uint64_t> slice_starts;
    void* mapping_base = nullptr;     ///< actual mapping start; differs from base when a shared-file header exists
    uint64_t mapping_bytes = 0;       ///< bytes to release from mapping_base
    /// shared-file arena only: the load state word in the file's header (0: nobody has loaded it, a pid: that engine is
    /// loading it, ~0: it is complete).  Shared by every engine that maps the file; null for a private arena.
    std::atomic<uint64_t>* shared_state = nullptr;
    ~PinnedArena();
    PinnedArena(const PinnedArena&) = delete;
    PinnedArena& operator=(const PinnedArena&) = delete;

    bool valid() const { return base != nullptr; }
    uint8_t* data() const { return (uint8_t*) base; }
};

struct LoadStats {
    double seconds = 0.0;
    // Loader fix: the two halves of `seconds`, both SUMMED OVER THE READER THREADS rather than wall clock, so
    // either can exceed `seconds` when the threads overlap.  They exist to separate the read from the copy and
    // the hash: an 8 MiB chunk that reaches the disk as ~2048 4 KiB reads spends its time in the first one.
    double read_seconds = 0.0;                  // inside the read call only: no seek, no copy, no hash
    double copy_seconds = 0.0;                  // memcpy + FNV-1a
    // A short read is not a slow load, it is a WRONG one: the caller must refuse the pack rather than run
    // on an arena whose tail was never written.  `seconds` alone cannot say that (it is -1.0 on failure, but
    // the caller's own check is on the REQUESTED byte count, which a refused read does not change).
    bool ok = true;                             // false: the load failed, see `error`
    std::string error;                          // why it failed, for the caller's message
    uint64_t bytes = 0;
    uint64_t layers = 0;
    std::vector<uint64_t> layer_checksums;      // one FNV-1a per layer
    double gib_per_second() const { return seconds > 0 ? (double) bytes / (1024.0 * 1024 * 1024) / seconds : 0.0; }
    // Aggregate rate of the readers WHILE THEY WERE INSIDE THE READ CALL: `bytes` over the mean per-thread
    // read time.  `gib_per_second()` above divides the same bytes by the wall clock of the whole loop, which
    // also contains the copy and the hash, so the two differ by how much of the loop was not I/O.
    double read_gib_per_second(int threads) const {
        if (read_seconds <= 0.0 || threads < 1) return 0.0;
        return (double) bytes / (1024.0 * 1024 * 1024) / (read_seconds / (double) threads);
    }
};

// Load `layers` layers of the expert arena into `dst` with `threads` readers, `chunk` bytes at a time.
// Each thread opens its OWN handle and seeks, which is the portable form of parallel pread: a shared handle
// needs a lock around the seek and defeats the parallelism on Windows.
//
// The reads go through `fread()` on a per-thread `FILE*`, NOT through `std::ifstream`.  MSVC's
// `basic_filebuf::xsgetn` splits every request larger than `_INTERNAL_BUFSIZ - 1` into 4095-byte `fread()`
// calls, so an 8 MiB `chunk` reached the disk as ~2048 4 KiB operations and the chunking did nothing at all
// (measured on a 42.9 GB pack: 1222 s, 4096 bytes per operation, 0.03 GiB/s).  `fread()` hands a request
// larger than the stream buffer straight to `_read()`/`ReadFile()`, which is what `chunk` is for.  The
// control flow, the copy, the per-layer FNV-1a and therefore the checksums are unchanged, so a pack loaded
// by the old and the new reader must produce identical `layer_checksums`.
LoadStats load_experts(const std::string& path, uint8_t* dst, uint64_t blob_bytes, uint64_t blobs_per_layer,
                       uint64_t layers, int threads, uint64_t chunk);
/// Pin by need: the arena's page-aligned blob stride.  Layer L's blob e sits at `dst_off[L] + e * stride[L]` (stride =
/// blob rounded up to 4 KiB, the pad is never read), instead of the packed `layer_off[L] + e * blob[L]`; the FILE
/// offsets stay the packed ones.  Every blob then starts on a page, so any set of blobs registers exactly.
struct PadMap {
    std::vector<uint64_t> dst_off, blob, stride;   ///< per layer
    uint64_t total = 0;                            ///< arena bytes of the padded layout (without the spare blob)
    uint64_t blob_off(int64_t layer, int64_t expert) const {
        return dst_off[(size_t) layer] + (uint64_t) expert * stride[(size_t) layer];
    }
};
/// Plan v0.3 P6: the same with one byte range per layer (`layer_off[L]`, `layer_bytes[L]`).
/// `pad` (optional): the destination is the padded layout above (`layer_off` stays the file offsets).
/// `ready` (optional): layer L is written only once *ready > L + 1 - its slice and the next one registered (a
/// PinnedArena registering its slices meanwhile: writing a page while cudaHostRegister runs on it corrupted the
/// arena under WDDM, even on large pages; a layer boundary inside a page puts that page in both registrations).
LoadStats load_experts_ranges(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
                              const std::atomic<int>* ready = nullptr);
/// (an overload, not a default argument: the SYCL build has its own pinned.dp.cpp that defines the signature above only)
LoadStats load_experts_ranges(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
                              const std::atomic<int>* ready, const PadMap* pad);

/// The same ranges read UNBUFFERED straight into `dst` (Windows): no staging buffer and no file-cache copy - the
/// drive's DMA lands where the experts live. Every range, `dst` and `chunk` must be 4 KiB aligned; returns ok =
/// false with an empty `error` when they are not (or off Windows), and the caller falls back to load_experts_ranges.
LoadStats load_experts_direct(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
                              const std::atomic<int>* ready = nullptr);

/// Whether the expert files are better read unbuffered (Windows; false elsewhere): a timed probe of random 64 KiB
/// reads says they are not in the OS file cache (a cached read takes ~10 us, the drive ~80), and the RAM left
/// beside the arena (`arena_bytes`) could not keep them cached for the next start either - so a warm restart after
/// an idle unload never gets slower, and only a start that reads the drive anyway skips the cache's copy.
/// STRATA_UNBUFFERED_LOAD=1 / 0 forces it. `why` says what decided.
/// `cache_counts` false (the file tier with a RAM budget): only whether the files could be kept decides - their mapped
/// pages land in the process's working set, so a partly cached file would still take the RAM the budget was sized for.
/// `read_bytes` (#577): the bytes the files are read for, when that is less than the files (the file tier reads only
/// the experts outside its RAM copy); `kAllFileBytes` = every byte of `files`.  See platform::file_cache_keeps.
constexpr uint64_t kAllFileBytes = ~0ull;
bool experts_unbuffered(const std::vector<std::string>& files, uint64_t arena_bytes, std::string& why,
                        bool cache_counts = true, uint64_t read_bytes = kAllFileBytes);

// FNV-1a 64.  Per layer, so a corrupt or short read names WHICH layer rather than just failing a whole-file
// comparison - the same reason the Phase 1 tools report the first differing element.
uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed = 1469598103934665603ull);

// ---- the expert stream -----------------------------------------------------
//
// THIS IS THE NUMBER THE OBJECTIVE DEPENDS ON.  With one expert per stream, the design's own arithmetic is:
//
//     per token: E = 663.6 MB of expert weights, of which only the MISSES cross the bus
//     miss fraction (1 - h) x 663.6 MB / measured bandwidth = milliseconds per token
//
// So `h` (the VRAM hit rate, ~0.167 by design) is only worth anything if a per-expert transfer can actually
// reach the bus's bandwidth.  A 1,382,400-byte copy is SMALL, and if small copies run at a fraction of large
// ones then the architecture's central constant is wrong and no kernel optimisation can fix it.  That is what
// this measures: the same total bytes at several granularities, so the difference is visible.
struct StreamStats {
    double seconds = 0.0;
    uint64_t bytes = 0;
    uint64_t chunk = 0;
    double gib_per_second() const { return seconds > 0 ? (double) bytes / (1024.0 * 1024 * 1024) / seconds : 0.0; }
};

// Copy `bytes` from pinned host memory at `src` to a device buffer of `chunk` bytes, in `chunk`-sized
// transfers, `iters` times.  The destination is REUSED, so this measures the bus and not allocation.
StreamStats stream_bandwidth(const uint8_t* src, uint64_t bytes, uint64_t chunk, int iters);

}  // namespace strata::core
