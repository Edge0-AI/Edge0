// STRATA_MMQ_DIRECT: the expert cache's slot guard (set_slot_guard), slot_room and slot_tail on the uniform cache
// (open: off_ empty, the stride is the largest blob) and on the sized one (open_sized: off_ holds each slot's start).
// No guard asked (the default) must give today's sized layout byte for byte.  Needs a CUDA device (exits 77 without).
#include "strata/core/expert_cache.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }
    using strata::core::ExpertCache;
    std::string err;

    // slot_stride: the blob plus the guard, 256-byte aligned; guard 0 is the old (b + 255) / 256 * 256
    for (int64_t b : {1, 255, 256, 257, 1000, 2457600, 2949120, 3047424}) {
        check(ExpertCache::slot_stride(b) == (b + 255) / 256 * 256, "slot_stride without a guard is today's stride");
        check(ExpertCache::slot_stride(b, 256) == (b + 256 + 255) / 256 * 256, "slot_stride with a 256 B guard");
        check(ExpertCache::slot_stride(b, 256) - b >= 256, "a guarded slot has at least the guard after its blob");
    }

    // 1. uniform: off_ is empty; the guard is asked for but not applied (every slot is the largest blob)
    {
        ExpertCache c;
        c.set_slot_guard(256);
        if (!c.open(6, 3, 4, 1024, err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
        check(c.slot_guard() == 0, "uniform: no guard applied");
        for (int32_t s = 0; s < 6; ++s) {
            check(c.slot_room(s) == 1024, "uniform: slot_room is the blob size");
            check(c.slot_offset(s) == (uint64_t) s * 1024, "uniform: slot_offset");
        }
        // the last slot reads slot_offset(slots) - the end - without indexing an empty off_
        check(c.slot_tail(5, 1000, 24) == c.device_slot(5) + 1000, "uniform: the last slot's slack covers 24 B");
        check(c.slot_tail(5, 1000, 25) == nullptr, "uniform: the last slot has no room for 25 B past 1000");
        check(c.slot_tail(0, 1024, 0) == c.device_slot(0) + 1024, "uniform: need 0 at the slot's end");
        check(c.slot_tail(0, 1024, 1) == nullptr, "uniform: a full slot has no room");
        check(c.slot_tail(3, 512, 136) == c.device_slot(3) + 512, "uniform: a smaller blob's slack");
        check(c.slot_tail(-1, 10, 10) == nullptr && c.slot_tail(6, 10, 10) == nullptr, "uniform: a bad slot is null");
        c.close();
        check(c.slot_guard() == 0 && c.slot_tail(0, 10, 10) == nullptr, "uniform: closed");
    }

    // 2. sized without a guard (the default): today's layout, and slot_tail sees only the 256-byte rounding
    const std::vector<int64_t> blobs = {1000, 2457600, 700, 3047424, 256, 2949120};
    std::vector<uint64_t> plain(blobs.size() + 1, 0);
    for (size_t i = 0; i < blobs.size(); ++i) plain[i + 1] = plain[i] + ((uint64_t) blobs[i] + 255) / 256 * 256;
    {
        ExpertCache c;
        if (!c.open_sized(blobs, 3, 4, err)) { std::fprintf(stderr, "open_sized: %s\n", err.c_str()); return 1; }
        check(c.slot_guard() == 0, "sized, no guard: slot_guard 0");
        for (size_t s = 0; s <= blobs.size(); ++s) check(c.slot_offset((int64_t) s) == plain[s], "sized, no guard: today's offsets");
        check(c.slot_tail(0, 1000, 24) == c.device_slot(0) + 1000, "sized, no guard: the rounding's 24 B");
        check(c.slot_tail(0, 1000, 25) == nullptr, "sized, no guard: no more than the rounding");
        check(c.slot_tail(4, 256, 1) == nullptr, "sized, no guard: a blob of 256 has no room");
        check(c.slot_tail(5, 2949120, 1) == nullptr, "sized, no guard: the last slot (offset of the end)");
    }

    // 3. sized with the 256 B guard: every slot has >= 256 B after its blob; close() drops the applied guard and a
    //    reopen applies the kept request again
    {
        ExpertCache c;
        c.set_slot_guard(256);
        for (int round = 0; round < 2; ++round) {
            if (!c.open_sized(blobs, 3, 4, err)) { std::fprintf(stderr, "open_sized: %s\n", err.c_str()); return 1; }
            check(c.slot_guard() == 256, "sized, guard: slot_guard 256");
            uint64_t off = 0;
            for (size_t s = 0; s < blobs.size(); ++s) {
                const int32_t si = (int32_t) s;
                check(c.slot_offset(si) == off, "sized, guard: offsets");
                off += (uint64_t) ExpertCache::slot_stride(blobs[s], 256);
                check(c.slot_room(si) == (uint64_t) ExpertCache::slot_stride(blobs[s], 256), "sized, guard: slot_room");
                check(c.device_slot(si) == c.device_slot(0) + c.slot_offset(si), "sized, guard: device_slot");
                uint8_t* t = c.slot_tail(si, (uint64_t) blobs[s], 256);
                check(t == c.device_slot(si) + blobs[s], "sized, guard: slot_tail at the blob's end");
                check(c.slot_tail(si, (uint64_t) blobs[s], c.slot_room(si) - (uint64_t) blobs[s] + 1) == nullptr,
                      "sized, guard: no room past the slot");
            }
            check(c.slot_offset((int64_t) blobs.size()) == off, "sized, guard: the end");
            // the guard is writable device memory inside the arena (the prompt path zeroes it)
            uint8_t* t = c.slot_tail((int32_t) blobs.size() - 1, (uint64_t) blobs.back(), 256);
            check(t != nullptr && cudaMemset(t, 0, 256) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess,
                  "sized, guard: the last slot's guard can be zeroed");
            c.close();
            check(c.slot_guard() == 0 && c.slot_tail(0, 10, 10) == nullptr, "sized, guard: closed");
        }
        // asked for 0 again: today's layout
        c.set_slot_guard(0);
        if (!c.open_sized(blobs, 3, 4, err)) { std::fprintf(stderr, "open_sized: %s\n", err.c_str()); return 1; }
        check(c.slot_guard() == 0 && c.slot_offset((int64_t) blobs.size()) == plain.back(), "sized: guard reset to 0");
    }

    if (g_fail != 0) {
        std::fprintf(stderr, "expert cache slot guard: %d checks failed\n", g_fail);
        return 1;
    }
    std::puts("expert cache slot guard: uniform and sized caches, with and without the guard, passed");
    return 0;
}
