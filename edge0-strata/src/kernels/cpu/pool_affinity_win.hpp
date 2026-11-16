#pragma once

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <cstdio>
#include <cstddef>
#include <vector>

namespace strata::kernels::cpu::detail {

// Hard affinity is only for pool-owned threads, which terminate with the pool. A caller's implicit
// all-group affinity cannot be restored from PreviousGroupAffinity, so the host uses CPU Sets below.
inline bool set_thread_group_affinity(int core, int worker) {
    if (core < 0) return false;
    GROUP_AFFINITY target{};
    target.Group = (WORD) (core / 64);
    target.Mask = KAFFINITY(1) << (core & 63);
    if (SetThreadGroupAffinity(GetCurrentThread(), &target, nullptr)) return true;
    std::fprintf(stderr, "strata cpu pool: SetThreadGroupAffinity for worker %d (group %u, mask 0x%llx) failed: %lu; previous affinity kept\n",
                 worker, (unsigned) target.Group, (unsigned long long) target.Mask, (unsigned long) GetLastError());
    return false;
}

inline bool get_thread_cpu_sets(std::vector<ULONG>& ids) {
    ULONG count = 0;
    if (!GetThreadSelectedCpuSets(GetCurrentThread(), nullptr, 0, &count) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) return false;
    ids.resize(count);
    if (count == 0) return true;
    if (!GetThreadSelectedCpuSets(GetCurrentThread(), ids.data(), count, &count)) return false;
    ids.resize(count);
    return true;
}

// CPU Set IDs are opaque: find the entry by both group and group-relative processor number.
inline bool cpu_set_for_core(int core, ULONG& id) {
    ULONG bytes = 0;
    if (!GetSystemCpuSetInformation(nullptr, 0, &bytes, GetCurrentProcess(), 0) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) return false;
    std::vector<unsigned char> buffer(bytes);
    if (bytes && !GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()),
                                            bytes, &bytes, GetCurrentProcess(), 0)) return false;
    constexpr size_t header_size = offsetof(SYSTEM_CPU_SET_INFORMATION, CpuSet);
    for (size_t offset = 0; offset + header_size <= bytes;) {
        const auto* info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
        if (info->Size < header_size || info->Size > bytes - offset) break;
        if (info->Type == CpuSetInformation && info->Size >= sizeof(SYSTEM_CPU_SET_INFORMATION) && info->CpuSet.Group == core / 64 &&
            info->CpuSet.LogicalProcessorIndex == (core & 63) &&
            (!info->CpuSet.Allocated || info->CpuSet.AllocatedToTargetProcess)) {
            id = info->CpuSet.Id;
            return true;
        }
        offset += info->Size;
    }
    SetLastError(ERROR_NOT_FOUND);
    return false;
}

}  // namespace strata::kernels::cpu::detail
#endif
