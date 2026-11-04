#include "platform/memory.h"
#include <fstream>
#include <limits>
#include <sstream>
#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#elif defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#endif
namespace edge0::platform {
namespace {
std::optional<std::uint64_t> kilobytes(const std::string& text, const std::string& key, std::string& reason) {
    std::istringstream lines(text);
    std::string line;
    std::optional<std::uint64_t> result;
    bool found = false;
    while (std::getline(lines, line)) {
        if (line.compare(0, key.size() + 1, key + ":") != 0) continue;
        if (found) { reason = "Duplicate field: " + key; return {}; }
        found = true;
        std::istringstream value(line.substr(key.size() + 1));
        std::string number, unit, extra;
        if (!(value >> number >> unit) || unit != "kB" || (value >> extra) || number.empty() ||
            number.find_first_not_of("0123456789") != std::string::npos) {
            reason = "Invalid field: " + key; return {};
        }
        std::uint64_t count = 0;
        for (char digit : number) {
            if (count > (std::numeric_limits<std::uint64_t>::max() / 1024 - unsigned(digit - '0')) / 10) {
                reason = "Overflow in field: " + key; return {};
            }
            count = count * 10 + unsigned(digit - '0');
        }
        result = count * 1024;
    }
    if (!found) reason = "Missing field: " + key;
    return result;
}
}
ProcessMemory parse_process_memory(const std::string& status, const std::string& smaps_rollup) {
    ProcessMemory result;
    result.rss.source = "proc_status";
    result.physical.source = "pss";
    MemoryValues rss, physical;
    if (status.empty()) {
        rss.current_reason = rss.peak_reason = "Cannot read /proc/self/status";
    } else {
        rss.current_bytes = kilobytes(status, "VmRSS", rss.current_reason);
        rss.peak_bytes = kilobytes(status, "VmHWM", rss.peak_reason);
    }
    if (smaps_rollup.empty()) physical.current_reason = "Cannot read /proc/self/smaps_rollup";
    else physical.current_bytes = kilobytes(smaps_rollup, "Pss", physical.current_reason);
    physical.peak_reason = "The kernel does not provide a PSS peak";
    result.rss.values = rss;
    result.physical.values = physical;
    return result;
}
ProcessMemory process_memory() {
    ProcessMemory result;
#if defined(__APPLE__)
    result.rss.source = "task_info.resident_size/ru_maxrss";
    result.physical.source = "phys_footprint";
    MemoryValues rss, physical;
    mach_task_basic_info_data_t basic{};
    mach_msg_type_number_t basic_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&basic), &basic_count) == KERN_SUCCESS)
        rss.current_bytes = basic.resident_size;
    else rss.current_reason = "RSS task_info failed";
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss >= 0)
        rss.peak_bytes = static_cast<std::uint64_t>(usage.ru_maxrss);
    else rss.peak_reason = "RSS getrusage failed";
    task_vm_info_data_t vm{};
    mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
    const auto vm_status = task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vm), &vm_count);
    if (vm_status == KERN_SUCCESS && vm_count >= TASK_VM_INFO_REV1_COUNT) physical.current_bytes = vm.phys_footprint;
    else physical.current_reason = "Physical footprint task_info unavailable";
    if (vm_status == KERN_SUCCESS && vm_count >= TASK_VM_INFO_REV3_COUNT && vm.ledger_phys_footprint_peak >= 0)
        physical.peak_bytes = static_cast<std::uint64_t>(vm.ledger_phys_footprint_peak);
    else physical.peak_reason = "Physical footprint peak unavailable";
    result.rss.values = rss;
    result.physical.values = physical;
#elif defined(__linux__)
    auto read = [](const char* path) {
        std::ifstream input(path);
        if (!input) return std::string{};
        std::string text(std::istreambuf_iterator<char>(input), {});
        return input.bad() ? std::string{} : text;
    };
    result = parse_process_memory(read("/proc/self/status"), read("/proc/self/smaps_rollup"));
#elif defined(_WIN32)
    // Keep Windows names: working set is the resident bucket, private bytes the
    // physical bucket. Windows has no private-bytes peak counter.
    result.rss.source = "working_set";
    result.physical.source = "private_usage";
    MemoryValues rss, physical;
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters))) {
        rss.current_bytes = static_cast<std::uint64_t>(counters.WorkingSetSize);
        rss.peak_bytes = static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
        physical.current_bytes = static_cast<std::uint64_t>(counters.PrivateUsage);
    } else {
        rss.current_reason = rss.peak_reason = "GetProcessMemoryInfo failed";
        physical.current_reason = "GetProcessMemoryInfo failed";
    }
    physical.peak_reason = "Windows provides no private-bytes peak counter";
    result.rss.values = rss;
    result.physical.values = physical;
#else
    result.rss.reason = "Process RSS query unsupported on this platform";
    result.physical.reason = "Physical footprint query unsupported on this platform";
#endif
    return result;
}
}
