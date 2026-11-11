// include/strata/core/pin_budget.hpp - pinned (page-locked) host memory is a budget, and the expert arena must not spend all of it.
//
// Windows (WDDM) lets a process page-lock about half the RAM for the GPU (the shared-memory segment).  The expert arena
// is registered slice by slice until the driver refuses one, which leaves NOTHING for the pinned/mapped buffers the
// engine allocates after it: the verify window's staging, the session's pool staging, the KV copy's tables.  Those were
// only a few MiB, and their failure ended the load ("verify: mapped staging allocation failed", exit 1) on a PC with
// 64 GB of free RAM.
//
// So the engine takes two reserves BEFORE the arena pins anything and gives them back after it:
//   * the SOFT reserve is released right after the arena is placed: the prompt path's staging and everything else
//     that already degrades to pageable memory then finds room, as it always did;
//   * the HARD reserve (the buffers whose failure ended the load) stays held until the first such allocation fails or
//     the engine is up; the allocation then releases it and retries once (host_alloc_guarded).
// A reserve that cannot be had is shrunk, then skipped: this never fails a start.
//
// Defaults: Windows CUDA 640 + 160 MiB (not the HIP build); Linux off (the NVIDIA driver there pins past RLIMIT_MEMLOCK and has no such
// ceiling; checked with a 4 GiB cudaHostRegister under `ulimit -l 1 GiB`).  STRATA_PIN_RESERVE_MIB=N sets the soft
// reserve (0 = off) on any OS; the hard one is a quarter of it, at least 64 MiB.
//
// STRATA_TEST_PIN_BUDGET_MIB=N is a TEST knob: it makes the engine's own pinned allocations and registrations that go
// through these wrappers (the arena's, the verify window's, the session's, the KV copy's) fail once N MiB are
// outstanding, which is what the WDDM ceiling does on Windows.  It lets Linux reproduce the exhaustion.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

namespace strata::core {

namespace pin_detail {

struct State {
    std::mutex mu;
    uint64_t sim_budget = 0;      // bytes; 0 = no simulated ceiling
    uint64_t sim_used = 0;
    uint64_t sim_peak = 0;
    std::map<const void*, uint64_t> sim_ptrs;
    void* soft = nullptr;
    uint64_t soft_bytes = 0;
    void* hard = nullptr;
    uint64_t hard_bytes = 0;
    bool taken = false;
    uint64_t arena_pinned = 0;    // what the arena registered (for the hint)
    int hard_retries = 0;
    State() {
        if (const char* e = std::getenv("STRATA_TEST_PIN_BUDGET_MIB"); e != nullptr && *e != '\0')
            sim_budget = (uint64_t) std::strtoull(e, nullptr, 10) << 20;
    }
};
inline State& st() {
    static State s;
    return s;
}
// caller holds the lock
inline bool sim_charge(State& s, const void* p, uint64_t bytes) {
    if (s.sim_budget == 0) return true;
    if (s.sim_used + bytes > s.sim_budget) return false;
    s.sim_used += bytes;
    if (s.sim_used > s.sim_peak) s.sim_peak = s.sim_used;
    if (p != nullptr) s.sim_ptrs[p] = bytes;
    return true;
}
inline void sim_uncharge(State& s, const void* p) {
    if (s.sim_budget == 0) return;
    auto it = s.sim_ptrs.find(p);
    if (it == s.sim_ptrs.end()) return;
    s.sim_used -= it->second;
    s.sim_ptrs.erase(it);
}

inline cudaError_t alloc_locked(State& s, void** p, size_t bytes, unsigned flags) {
    if (s.sim_budget != 0 && s.sim_used + bytes > s.sim_budget) return cudaErrorMemoryAllocation;
    const cudaError_t e = cudaHostAlloc(p, bytes, flags);
    if (e == cudaSuccess) sim_charge(s, *p, bytes);
    return e;
}
inline void free_locked(State& s, void* p) {
    if (p == nullptr) return;
    (void) cudaFreeHost(p);
    sim_uncharge(s, p);
}

inline uint64_t soft_reserve_default_mib() {
    if (const char* e = std::getenv("STRATA_PIN_RESERVE_MIB"); e != nullptr && *e != '\0')
        return (uint64_t) std::strtoull(e, nullptr, 10);
#if defined(_WIN32) && !defined(STRATA_USE_HIP)   // CUDA on Windows (WDDM); the AMD build keeps its pinning as it was
    return 640;
#else
    return st().sim_budget != 0 ? 640 : 0;   // Linux: only when the test ceiling is on
#endif
}
}  // namespace pin_detail

/// cudaHostRegister through the (test) budget.
inline cudaError_t pin_register(void* p, size_t bytes, unsigned flags) {
    auto& s = pin_detail::st();
    if (s.sim_budget == 0) return cudaHostRegister(p, bytes, flags);
    std::lock_guard<std::mutex> g(s.mu);
    if (s.sim_used + bytes > s.sim_budget) return cudaErrorMemoryAllocation;
    const cudaError_t e = cudaHostRegister(p, bytes, flags);
    if (e == cudaSuccess) pin_detail::sim_charge(s, p, bytes);
    return e;
}
inline cudaError_t pin_unregister(void* p) {
    auto& s = pin_detail::st();
    if (s.sim_budget != 0) {
        std::lock_guard<std::mutex> g(s.mu);
        pin_detail::sim_uncharge(s, p);
    }
    return cudaHostUnregister(p);
}

/// Take the reserves (call before the expert arena pins).  Never fails.
inline void pin_reserve_take() {
    auto& s = pin_detail::st();
    std::lock_guard<std::mutex> g(s.mu);
    if (s.taken) return;
    s.taken = true;
    const uint64_t soft_mib = pin_detail::soft_reserve_default_mib();
    if (soft_mib == 0) return;
    const uint64_t hard_mib = soft_mib / 4 > 64 ? soft_mib / 4 : 64;
    // the hard one first: it is the one whose absence ends a load.  Shrink by halves down to 16 MiB.
    auto take = [&](void*& p, uint64_t& got, uint64_t mib) {
        for (; mib >= 16; mib /= 2) {
            void* q = nullptr;
            if (pin_detail::alloc_locked(s, &q, (size_t) (mib << 20), cudaHostAllocPortable) == cudaSuccess) {
                p = q;
                got = mib << 20;
                return;
            }
            (void) cudaGetLastError();
        }
    };
    take(s.hard, s.hard_bytes, hard_mib);
    take(s.soft, s.soft_bytes, soft_mib);
    std::fprintf(stderr, "strata: pinned-memory headroom: holding %llu MiB for the buffers allocated after the expert arena "
                         "(released after it: %llu MiB; kept until needed: %llu MiB; STRATA_PIN_RESERVE_MIB changes it)\n",
                 (unsigned long long) ((s.soft_bytes + s.hard_bytes) >> 20), (unsigned long long) (s.soft_bytes >> 20),
                 (unsigned long long) (s.hard_bytes >> 20));
    if (s.soft_bytes + s.hard_bytes < ((soft_mib + hard_mib) << 20))
        std::fprintf(stderr, "strata: WARNING: pinned-memory budget is already nearly spent before the expert arena "
                             "(got %llu of %llu MiB of headroom)\n",
                     (unsigned long long) ((s.soft_bytes + s.hard_bytes) >> 20), (unsigned long long) (soft_mib + hard_mib));
}
/// The arena is placed: hand the soft reserve back.
inline void pin_reserve_release_soft() {
    auto& s = pin_detail::st();
    std::lock_guard<std::mutex> g(s.mu);
    if (s.soft == nullptr) return;
    pin_detail::free_locked(s, s.soft);
    s.soft = nullptr;
    std::fprintf(stderr, "strata: pinned-memory headroom: %llu MiB handed back after the expert arena\n",
                 (unsigned long long) (s.soft_bytes >> 20));
    s.soft_bytes = 0;
}
/// The engine is up (or the hard allocation needs it): hand the hard reserve back.
inline void pin_reserve_release_hard() {
    auto& s = pin_detail::st();
    std::lock_guard<std::mutex> g(s.mu);
    if (s.hard == nullptr) return;
    pin_detail::free_locked(s, s.hard);
    s.hard = nullptr;
    s.hard_bytes = 0;
}
inline void pin_note_arena(uint64_t pinned_bytes) {
    auto& s = pin_detail::st();
    std::lock_guard<std::mutex> g(s.mu);
    s.arena_pinned = pinned_bytes;
    // TEST knob: the driver took everything but N KiB (what WDDM did on the 64 GB PC: the arena pinned to the ceiling)
    if (s.sim_budget != 0)
        if (const char* e = std::getenv("STRATA_TEST_PIN_LEFT_KIB"); e != nullptr && *e != 0)
            s.sim_budget = s.sim_used + ((uint64_t) std::strtoull(e, nullptr, 10) << 10);
}
inline uint64_t pin_sim_peak() { return pin_detail::st().sim_peak; }
/// Test knob only: one line with the simulated ceiling, what is pinned now and the peak.
inline void pin_sim_report(const char* when) {
    auto& s = pin_detail::st();
    std::lock_guard<std::mutex> g(s.mu);
    if (s.sim_budget == 0) return;
    std::fprintf(stderr, "strata: test pinned budget %llu MiB: %llu MiB charged, peak %llu MiB (%s)\n",
                 (unsigned long long) (s.sim_budget >> 20), (unsigned long long) (s.sim_used >> 20),
                 (unsigned long long) (s.sim_peak >> 20), when);
}

/// cudaHostAlloc for a buffer the engine cannot run without.  On a refusal it hands the reserves back and tries once
/// more; the caller still reports a second refusal.
inline cudaError_t host_alloc_guarded(void** p, size_t bytes, unsigned flags) {
    auto& s = pin_detail::st();
    {
        std::lock_guard<std::mutex> g(s.mu);
        const cudaError_t e = pin_detail::alloc_locked(s, p, bytes, flags);
        if (e == cudaSuccess) return e;
        (void) cudaGetLastError();
        if (s.hard == nullptr && s.soft == nullptr) return e;
        if (s.soft != nullptr) { pin_detail::free_locked(s, s.soft); s.soft = nullptr; s.soft_bytes = 0; }
        if (s.hard != nullptr) { pin_detail::free_locked(s, s.hard); s.hard = nullptr; s.hard_bytes = 0; }
        ++s.hard_retries;
        std::fprintf(stderr, "strata: pinned-memory headroom: a %llu KiB pinned buffer was refused; the held headroom is handed back "
                             "and the allocation retried\n", (unsigned long long) (bytes >> 10));
        const cudaError_t e2 = pin_detail::alloc_locked(s, p, bytes, flags);
        if (e2 != cudaSuccess) (void) cudaGetLastError();
        return e2;
    }
}

/// The sentence every hard failure ends with, so the app can recognise it in the log.
inline std::string pin_exhausted_hint() {
    auto& s = pin_detail::st();
    const uint64_t gib = s.arena_pinned >> 30;
    std::string t = "[pinned-memory-exhausted] the OS gave no more page-locked memory";
    if (gib > 2) t += "; start again with STRATA_ARENA_PIN_GIB=" + std::to_string(gib - 2) + " (pin less of the expert arena) or with the low-RAM option";
    return t;
}

}  // namespace strata::core
