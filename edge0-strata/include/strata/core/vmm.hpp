#pragma once
// CUDA virtual memory (the elastic K/V, --kv-grow; the output head under --prefill-head-alias): an address range
// reserved once, whose physical memory is mapped and unmapped in chunks of the device's granularity (2 MiB).  The K/V
// pools grow with the context by taking chunks the expert cache gives up, and neither moves: every captured graph and
// stored pointer stays valid.  The driver calls are reached through the runtime's entry-point query, so nothing links
// the driver library.  HIP has the same calls (ROCm 6+, see vmm.cpp); Metal has none.

#include <cstdint>
#include <vector>

namespace strata::core {

/// A physical chunk: the driver's allocation handle (CUmemGenericAllocationHandle); 0 = none.
using VmmChunk = unsigned long long;

/// The current device supports it and the entry points resolved.
bool vmm_available();
/// The chunk size (0 when not available).
uint64_t vmm_granularity();
/// A new physical chunk on the current device (0: out of memory), and its release.
VmmChunk vmm_chunk_new();
void vmm_chunk_free(VmmChunk h);

class VmmRange {
public:
    VmmRange() = default;
    ~VmmRange() { release(); }
    VmmRange(const VmmRange&) = delete;
    VmmRange& operator=(const VmmRange&) = delete;

    /// Reserves `bytes` (rounded up to whole chunks) of addresses, nothing mapped.
    bool reserve(uint64_t bytes);
    /// Unmaps and frees every chunk, then the addresses.
    void release();
    uint8_t* base() const { return (uint8_t*) (uintptr_t) base_; }
    int64_t chunks() const { return (int64_t) h_.size(); }
    bool mapped(int64_t i) const { return i >= 0 && i < chunks() && h_[(std::size_t) i] != 0; }
    int64_t mapped_count() const;
    /// Maps chunks [lo, hi) that are not mapped yet, each to `take()` or, when that returns 0, to a new chunk.
    /// Readable and writable when it returns; false: out of memory (what was mapped stays mapped).
    template <class Take> bool map_range(int64_t lo, int64_t hi, Take take);
    /// Unmaps chunk i and hands back its physical chunk (0 when it was not mapped).
    VmmChunk unmap(int64_t i);

private:
    bool map_one(int64_t i, VmmChunk h);
    bool set_access(int64_t lo, int64_t hi);
    /// Readable and writable [lo, hi) (all newly mapped); when that fails they are unmapped and freed again, so no
    /// chunk counts as mapped without access.
    bool commit_run(int64_t lo, int64_t hi);
    unsigned long long base_ = 0;
    std::vector<VmmChunk> h_;   // per chunk: its physical chunk, 0 = unmapped
};

/// A buffer whose addresses are fixed for its whole life: `ensure` reserves them once, `map` takes physical memory
/// there, and `unmap` gives the physical memory back while the addresses stay reserved, so the next `map` returns the
/// same pointer.  A captured graph or a stored pointer survives an unmap/map pair.
///
/// That is what --prefill-head-alias needs: the head's VRAM is lent to the prompt path while a prompt is read, and the
/// verify windows' graphs hold the head's addresses from their capture.  With cudaMalloc the head may come back at
/// another address (any allocation in between, such as an elastic K/V growth, can take the freed block), and the
/// engine then refuses to run those graphs.  `ensure` returns false when the device has no virtual memory, or when its
/// reservation cannot be made: the caller then falls back to cudaMalloc, which keeps no address.
class VmmBuffer {
public:
    VmmBuffer() = default;
    ~VmmBuffer() { release(); }
    VmmBuffer(const VmmBuffer&) = delete;
    VmmBuffer& operator=(const VmmBuffer&) = delete;

    /// Reserves whole chunks of addresses for `bytes`: on the first call, and again only when `bytes` no longer fits.
    /// The addresses already reserved are released first when a larger range is needed.
    bool ensure(uint64_t bytes) {
        const uint64_t gran = vmm_granularity();
        if (gran == 0 || bytes == 0) return false;
        const uint64_t chunks = (bytes + gran - 1) / gran;
        if ((uint64_t) range_.chunks() >= chunks) return true;
        return range_.reserve(bytes);
    }
    /// Maps every chunk to new physical memory; false when it is not available (nothing stays mapped).
    bool map() {
        if (range_.chunks() == 0) return false;
        if (!range_.map_range(0, range_.chunks(), [] { return VmmChunk(0); })) {
            unmap();
            return false;
        }
        return true;
    }
    /// Frees the physical memory and keeps the addresses; a no-op on an unmapped buffer.
    void unmap() {
        for (int64_t i = 0; i < range_.chunks(); ++i) vmm_chunk_free(range_.unmap(i));
    }
    /// Frees the physical memory and the addresses.
    void release() { range_.release(); }
    uint8_t* base() const { return range_.base(); }
    bool reserved() const { return range_.chunks() > 0; }
    bool mapped() const { return range_.chunks() > 0 && range_.mapped_count() == range_.chunks(); }
    /// The reserved size in bytes (whole chunks; 0 when nothing is reserved).
    uint64_t bytes() const { return (uint64_t) range_.chunks() * vmm_granularity(); }

private:
    VmmRange range_;
};

template <class Take> bool VmmRange::map_range(int64_t lo, int64_t hi, Take take) {
    lo = lo < 0 ? 0 : lo;
    hi = hi > chunks() ? chunks() : hi;
    int64_t run = -1;   // access is set once per run of newly mapped chunks (it costs as much as the map)
    for (int64_t i = lo; i < hi; ++i) {
        if (mapped(i)) {
            if (run >= 0 && !commit_run(run, i)) return false;
            run = -1;
            continue;
        }
        VmmChunk h = take();
        if (h == 0) h = vmm_chunk_new();
        if (h == 0 || !map_one(i, h)) {
            if (h != 0) vmm_chunk_free(h);
            if (run >= 0) commit_run(run, i);
            return false;
        }
        if (run < 0) run = i;
    }
    return run < 0 || commit_run(run, hi);
}

}  // namespace strata::core
