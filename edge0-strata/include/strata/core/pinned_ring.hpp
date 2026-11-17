// #1996 (0.1.43.1): host-to-device uploads whose host side is pageable go up through pinned bounce slots.
//
// A cudaMemcpyAsync from pageable memory is staged by the runtime, and the staging is not ordered the way a pinned copy
// is.  On gfx1100 (ROCm 7.2.4) a text prompt after a picture one read stale bytes that way (#1712, fixed for the step
// records and the token ids in 0.1.43); a 4-hour multi-slot load (#1996) still faulted after 31 minutes, with the
// image rows, the draft tokens and bounds and the non-mapped routing tables still going up from caller memory.  The
// analysis of which copies those are, and the idea of a ring of pinned slots with one event each, are
// mantovaniluca91's (patch C on #1712); this is our own form of it.
//
// A PinnedRing owns a few pinned slots, each with an event.  h2d() copies the bytes into the next slot (after the copy
// that last read that slot is done), queues the DMA from the slot and records the slot's event.  So the caller may
// overwrite or free its source the moment the call returns, as a pageable copy allowed.  With no pinned RAM (or
// STRATA_TEST_PAGEABLE=1) the copy is made from the caller's memory and the stream is waited for, so the source is
// free on return all the same.  A ring belongs to ONE device (the one current when it is first used) and is used by one
// thread at a time.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

class PinnedRing {
public:
    static constexpr int kSlots = 4;

    PinnedRing() = default;
    PinnedRing(const PinnedRing&) = delete;
    PinnedRing& operator=(const PinnedRing&) = delete;
    ~PinnedRing() { release(); }

    // dst (device) <- bytes from src (host, any kind of memory); src is free to change when this returns
    cudaError_t h2d(void* dst, const void* src, size_t bytes, cudaStream_t s) {
        if (bytes == 0) return cudaSuccess;
        void* stage = acquire(bytes);
        if (stage == nullptr) return direct(dst, src, bytes, s);
        std::memcpy(stage, src, bytes);
        return submit(dst, stage, bytes, s);
    }

    // For a caller that gathers many small pieces into one upload: acquire() hands out a pinned slot of at least
    // `bytes` (nullptr: none could be had - use direct()), the caller fills it, then submit() queues the copy and
    // guards the slot.  An acquired slot must be submitted before the next acquire().
    void* acquire(size_t bytes) {
        if (bytes == 0 || pageable_forced()) return nullptr;
        int cur = -1;
        if (cudaGetDevice(&cur) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        if (dev_ < 0) dev_ = cur;
        if (cur != dev_) return nullptr;   // a ring is for one device
        Slot& sl = slots_[next_];
        next_ = (next_ + 1) % kSlots;
        if (sl.pending) {                  // the copy that last read this slot is done before it is refilled
            sl.pending = false;
            if (cudaEventSynchronize(sl.ev) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        }
        if (sl.cap < bytes) {
            if (sl.p) { cudaFreeHost(sl.p); sl.p = nullptr; sl.cap = 0; }   // idle: its last copy was waited for above
            const size_t cap = bytes < kMinCap ? kMinCap : bytes;
            if (cudaHostAlloc(&sl.p, cap, cudaHostAllocPortable) != cudaSuccess) {
                cudaGetLastError();
                sl.p = nullptr;
                return nullptr;
            }
            sl.cap = cap;
        }
        if (sl.ev == nullptr && cudaEventCreateWithFlags(&sl.ev, cudaEventDisableTiming) != cudaSuccess) {
            cudaGetLastError();
            sl.ev = nullptr;
            return nullptr;
        }
        acquired_ = &sl;
        return sl.p;
    }

    cudaError_t submit(void* dst, const void* stage, size_t bytes, cudaStream_t s) {
        Slot* sl = acquired_;
        acquired_ = nullptr;
        cudaError_t e = cudaMemcpyAsync(dst, stage, bytes, cudaMemcpyHostToDevice, s);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "strata: pinned upload of %zu bytes failed: %s\n", bytes, cudaGetErrorString(e));
            return e;
        }
        if (sl != nullptr && cudaEventRecord(sl->ev, s) == cudaSuccess) {
            sl->pending = true;
            return cudaSuccess;
        }
        cudaGetLastError();                // no event: the copy is waited for here instead
        return cudaStreamSynchronize(s);
    }

    // the same copy without a pinned slot: from the caller's memory, the stream waited for
    static cudaError_t direct(void* dst, const void* src, size_t bytes, cudaStream_t s) {
        if (bytes == 0) return cudaSuccess;
        cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s);
        if (e == cudaSuccess) e = cudaStreamSynchronize(s);
        if (e != cudaSuccess) std::fprintf(stderr, "strata: upload of %zu bytes failed: %s\n", bytes, cudaGetErrorString(e));
        return e;
    }

    // dst (pageable host) <- device bytes; the stream is drained and the bytes are in dst on return
    cudaError_t d2h_sync(void* dst, const void* src, size_t bytes, cudaStream_t s) {
        if (bytes == 0) return cudaSuccess;
        void* stage = acquire(bytes);
        cudaError_t e = cudaMemcpyAsync(stage != nullptr ? stage : dst, src, bytes, cudaMemcpyDeviceToHost, s);
        if (e == cudaSuccess) e = cudaStreamSynchronize(s);
        if (e == cudaSuccess && stage != nullptr) std::memcpy(dst, stage, bytes);
        if (stage != nullptr) acquired_ = nullptr;   // no event: the stream was drained
        if (e != cudaSuccess) std::fprintf(stderr, "strata: download of %zu bytes failed: %s\n", bytes, cudaGetErrorString(e));
        return e;
    }

    // Every slot holds at least `bytes` from here on: grown now, by a caller that must not allocate later (a pinned
    // allocation or free takes the driver's lock and can wait for the device: STRATA_PIPED_CONCURRENT grows the rings
    // before a read whose windows run beside it).  On the device that is current, as acquire().  False: a slot could
    // not be had (its uploads then grow it, or go up from the caller's memory, as before).
    bool reserve(size_t bytes) {
        if (bytes == 0 || pageable_forced()) return true;
        int cur = -1;
        if (cudaGetDevice(&cur) != cudaSuccess) { cudaGetLastError(); return false; }
        if (dev_ < 0) dev_ = cur;
        if (cur != dev_) return false;
        for (Slot& sl : slots_) {
            if (sl.cap >= bytes && sl.ev != nullptr) continue;
            if (sl.pending) {
                sl.pending = false;
                if (cudaEventSynchronize(sl.ev) != cudaSuccess) { cudaGetLastError(); return false; }
            }
            if (sl.cap < bytes) {
                if (sl.p) { cudaFreeHost(sl.p); sl.p = nullptr; sl.cap = 0; }
                const size_t cap = bytes < kMinCap ? kMinCap : bytes;
                if (cudaHostAlloc(&sl.p, cap, cudaHostAllocPortable) != cudaSuccess) {
                    cudaGetLastError();
                    sl.p = nullptr;
                    return false;
                }
                sl.cap = cap;
            }
            if (sl.ev == nullptr && cudaEventCreateWithFlags(&sl.ev, cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError();
                sl.ev = nullptr;
                return false;
            }
        }
        return true;
    }

    void release() {
        for (Slot& sl : slots_) {
            if (sl.pending && sl.ev) cudaEventSynchronize(sl.ev);
            if (sl.ev) cudaEventDestroy(sl.ev);
            if (sl.p) cudaFreeHost(sl.p);
            sl = Slot{};
        }
        cudaGetLastError();
        acquired_ = nullptr;
        dev_ = -1;
        next_ = 0;
    }

private:
    static constexpr size_t kMinCap = 64 * 1024;
    struct Slot {
        void* p = nullptr;
        size_t cap = 0;
        cudaEvent_t ev = nullptr;
        bool pending = false;
    };
    static bool pageable_forced() {
        static const bool v = [] { const char* e = std::getenv("STRATA_TEST_PAGEABLE"); return e != nullptr && e[0] == '1'; }();
        return v;
    }
    Slot slots_[kSlots];
    Slot* acquired_ = nullptr;
    int next_ = 0;
    int dev_ = -1;
};

}  // namespace strata::core
