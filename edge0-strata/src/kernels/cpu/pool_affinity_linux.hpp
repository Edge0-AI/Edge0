#pragma once

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>

#include <cerrno>
#include <climits>
#include <cstring>
#include <vector>

namespace strata::kernels::cpu::detail {

// CPU_ALLOC and the sized macros also handle kernels whose CPU mask exceeds CPU_SETSIZE.
// Keep the native mask opaque: copy its bytes for storage and use CPU_*_S to inspect it.
class LinuxCpuSet {
public:
    explicit LinuxCpuSet(int cpus) : bytes(CPU_ALLOC_SIZE(cpus)), set(CPU_ALLOC(cpus)) {
        if (set) CPU_ZERO_S(bytes, set);
    }
    ~LinuxCpuSet() { CPU_FREE(set); }
    LinuxCpuSet(const LinuxCpuSet&) = delete;
    LinuxCpuSet& operator=(const LinuxCpuSet&) = delete;

    const size_t bytes;
    cpu_set_t* const set;
};

// pthread affinity functions return an error number directly; they do not set errno.
// A too-small query buffer returns EINVAL, even when the number of online CPUs is small.
inline int get_thread_affinity(std::vector<unsigned long>& mask, std::vector<int>* cpus = nullptr) {
    for (int count = CPU_SETSIZE;;) {
        LinuxCpuSet current(count);
        if (!current.set) return ENOMEM;
        const int error = pthread_getaffinity_np(pthread_self(), current.bytes, current.set);
        if (error == 0) {
            mask.resize(current.bytes / sizeof(unsigned long));
            std::memcpy(mask.data(), current.set, current.bytes);
            if (cpus) {
                cpus->clear();
                for (int cpu = 0; cpu < count; ++cpu)
                    if (CPU_ISSET_S(cpu, current.bytes, current.set)) cpus->push_back(cpu);
            }
            return 0;
        }
        if (error != EINVAL || count > INT_MAX / 2) return error;
        count *= 2;
    }
}

inline int set_thread_affinity(const std::vector<unsigned long>& mask) {
    constexpr size_t word_bits = sizeof(unsigned long) * CHAR_BIT;
    if (mask.empty() || mask.size() > INT_MAX / word_bits) return EINVAL;
    LinuxCpuSet target(static_cast<int>(mask.size() * word_bits));
    if (!target.set) return ENOMEM;
    std::memcpy(target.set, mask.data(), target.bytes);
    return pthread_setaffinity_np(pthread_self(), target.bytes, target.set);
}

inline int pin_thread_to_cpu(int core) {
    if (core < 0 || core == INT_MAX) return EINVAL;
    LinuxCpuSet target(core < CPU_SETSIZE ? CPU_SETSIZE : core + 1);
    if (!target.set) return ENOMEM;
    CPU_SET_S(core, target.bytes, target.set);
    return pthread_setaffinity_np(pthread_self(), target.bytes, target.set);
}

}  // namespace strata::kernels::cpu::detail
#elif defined(__APPLE__)
#include <thread>
#include <vector>

// macOS has no thread-to-core pinning (the scheduler places threads; affinity tags are hints at most): every CPU is
// allowed and a pin is a no-op, so the pool keeps its threads where the OS puts them.
namespace strata::kernels::cpu::detail {
inline int get_thread_affinity(std::vector<unsigned long>& mask, std::vector<int>* cpus = nullptr) {
    mask.assign(1, ~0ul);
    if (cpus) {
        cpus->clear();
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) cpus->push_back((int) i);
    }
    return 0;
}
inline int set_thread_affinity(const std::vector<unsigned long>&) { return 0; }
inline int pin_thread_to_cpu(int) { return 0; }   // nothing to do: the OS places the thread
}  // namespace strata::kernels::cpu::detail
#endif
