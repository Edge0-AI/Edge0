#include "platform/system.h"
#include "platform/memory.h"
#include <fstream>
#include <limits>
#include <sstream>
#include <map>
#include <mutex>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <dlfcn.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <objc/message.h>
#include <objc/runtime.h>
#if defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__)
#include <os/proc.h>
#endif
#elif defined(__ANDROID__)
#include <dlfcn.h>
#elif defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
namespace edge0::platform {
MemoryLocation classify_device_memory(std::optional<enum ggml_backend_dev_type> type, bool unified_memory) {
    return type == GGML_BACKEND_DEVICE_TYPE_GPU && !unified_memory ?
        MemoryLocation::dedicated : MemoryLocation::system;
}
namespace {
bool metal_unified_memory() {
#if defined(__APPLE__)
    // ggml 7ab4ee7 uses the system default MTLDevice for every Metal device.
    auto library = dlopen("/System/Library/Frameworks/Metal.framework/Metal", RTLD_NOW | RTLD_LOCAL);
    if (!library) return false;
    auto create = reinterpret_cast<id (*)()>(dlsym(library, "MTLCreateSystemDefaultDevice"));
    const auto device = create ? create() : nullptr;
    bool unified = false;
    if (device) {
        unified = reinterpret_cast<bool (*)(id, SEL)>(objc_msgSend)(device, sel_registerName("hasUnifiedMemory"));
        reinterpret_cast<void (*)(id, SEL)>(objc_msgSend)(device, sel_registerName("release"));
    }
    dlclose(library);
    return unified;
#else
    return false;
#endif
}
}
MemoryLocation device_memory(const std::string& name) {
    // Called only by explicit snapshots. Cache stable device properties, not
    // current memory readings; generation and allocation never query here.
    static std::mutex mutex;
    static std::map<std::string, MemoryLocation> cache;
    std::lock_guard<std::mutex> lock(mutex);
    const auto existing = cache.find(name);
    if (existing != cache.end()) return existing->second;
    auto device = name == "host" ? nullptr : ggml_backend_dev_by_name(name.c_str());
    std::optional<enum ggml_backend_dev_type> type;
    bool unified = false;
    if (device) {
        type = ggml_backend_dev_type(device);
        const std::string backend = ggml_backend_reg_name(ggml_backend_dev_backend_reg(device));
        if (backend == "Metal" || backend == "MTL") unified = metal_unified_memory();
    }
    const auto memory = classify_device_memory(type, unified);
    cache.emplace(name, memory);
    return memory;
}

SystemMemory parse_system_memory(const std::string& text) {
    SystemMemory result;
    auto parse = [&](const std::string& key) {
        SystemValue value{{}, "/proc/meminfo." + key, "Missing field: " + key};
        std::istringstream input(text);
        std::string line;
        bool found = false;
        while (std::getline(input, line)) {
            if (line.compare(0, key.size() + 1, key + ":") != 0) continue;
            if (found) { value.bytes.reset(); value.reason = "Duplicate field: " + key; break; }
            found = true;
            std::istringstream fields(line.substr(key.size() + 1));
            std::string digits, unit, extra;
            if (!(fields >> digits >> unit) || unit != "kB" || (fields >> extra) ||
                digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos) {
                value.reason = "Invalid field: " + key; continue;
            }
            std::uint64_t count = 0;
            bool overflow = false;
            for (char digit : digits) {
                if (count > (std::numeric_limits<std::uint64_t>::max() / 1024 - unsigned(digit - '0')) / 10) { overflow = true; break; }
                count = count * 10 + unsigned(digit - '0');
            }
            if (overflow) value.reason = "Overflow in field: " + key;
            else { value.bytes = count * 1024; value.reason.clear(); }
        }
        return value;
    };
    result.total = parse("MemTotal"); result.available = parse("MemAvailable");
    return result;
}
SystemMemory windows_system_memory(bool success, std::uint64_t total, std::uint64_t available) {
    return {{success ? std::optional<std::uint64_t>(total) : std::nullopt,
                "GlobalMemoryStatusEx.ullTotalPhys", success ? "" : "GlobalMemoryStatusEx failed"},
            {success ? std::optional<std::uint64_t>(available) : std::nullopt,
                "GlobalMemoryStatusEx.ullAvailPhys", success ? "" : "GlobalMemoryStatusEx failed"}};
}
ThermalStatus apple_thermal_status(long state) {
    const char* names[]{"nominal", "fair", "serious", "critical"};
    if (state >= 0 && state < 4) return {names[state], "NSProcessInfo.thermalState", {}};
    return {{}, "NSProcessInfo.thermalState", "Unknown thermal state"};
}
ThermalStatus android_thermal_status(int state) {
    const char* names[]{"none", "light", "moderate", "severe", "critical", "emergency", "shutdown"};
    if (state >= 0 && state < 7) return {names[state], "AThermal_getCurrentThermalStatus", {}};
    return {{}, "AThermal_getCurrentThermalStatus", "Thermal query failed or returned an unknown status"};
}
ThermalStatus linux_thermal_unavailable(bool android) {
    if (android) return {{}, "AThermal_getCurrentThermalStatus", "Thermal API requires Android 11 (API 30) or newer"};
    return {{}, {}, "Linux has no corresponding public thermal-state API"};
}
SystemStatus system_status() {
    SystemStatus result;
    result.app_memory_limit.reason = "App memory limit is only available on iOS";
#if defined(__APPLE__)
    result.memory.total.source = "sysctl.hw.memsize";
    std::uint64_t total = 0; std::size_t size = sizeof(total);
    if (sysctlbyname("hw.memsize", &total, &size, nullptr, 0) == 0) result.memory.total.bytes = total;
    else result.memory.total.reason = "hw.memsize query failed";
    result.memory.available.source = "host_statistics64.free_count+inactive_count+purgeable_count+speculative_count";
    const auto host = mach_host_self();
    vm_statistics64_data_t vm{}; mach_msg_type_number_t count = HOST_VM_INFO64_COUNT; vm_size_t page = 0;
    if (host_page_size(host, &page) == KERN_SUCCESS &&
        host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) == KERN_SUCCESS)
        result.memory.available.bytes = (std::uint64_t(vm.free_count) + vm.inactive_count + vm.purgeable_count + vm.speculative_count) * page;
    else result.memory.available.reason = "host_statistics64 query failed";
    mach_port_deallocate(mach_task_self(), host);
    const auto cls = objc_getClass("NSProcessInfo");
    if (cls) {
        const auto process = reinterpret_cast<id (*)(id, SEL)>(objc_msgSend)(reinterpret_cast<id>(cls), sel_registerName("processInfo"));
        result.thermal = apple_thermal_status(reinterpret_cast<long (*)(id, SEL)>(objc_msgSend)(process, sel_registerName("thermalState")));
    } else result.thermal = {{}, "NSProcessInfo.thermalState", "NSProcessInfo unavailable"};
#if defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__)
    result.app_memory_limit.source = "phys_footprint+os_proc_available_memory";
    const auto memory = process_memory();
    if (memory.physical.values && memory.physical.values->current_bytes) {
        result.app_memory_limit.current_physical_bytes = memory.physical.values->current_bytes;
        result.app_memory_limit.bytes = *memory.physical.values->current_bytes + os_proc_available_memory();
        result.app_memory_limit.reason.clear();
    } else result.app_memory_limit.reason = "phys_footprint unavailable";
#endif
#elif defined(__linux__)
    std::ifstream input("/proc/meminfo"); std::ostringstream text; text << input.rdbuf();
    result.memory = parse_system_memory(text.str());
#if defined(__ANDROID__)
    result.thermal = linux_thermal_unavailable(true);
    if (auto library = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL)) {
        struct Manager;
        auto acquire = reinterpret_cast<Manager* (*)()>(dlsym(library, "AThermal_acquireManager"));
        auto release = reinterpret_cast<void (*)(Manager*)>(dlsym(library, "AThermal_releaseManager"));
        auto query = reinterpret_cast<int (*)(Manager*)>(dlsym(library, "AThermal_getCurrentThermalStatus"));
        if (acquire && release && query) {
            if (auto manager = acquire()) { result.thermal = android_thermal_status(query(manager)); release(manager); }
            else result.thermal.reason = "AThermal_acquireManager failed";
        }
        dlclose(library);
    }
#else
    result.thermal = linux_thermal_unavailable(false);
#endif
#elif defined(_WIN32)
    MEMORYSTATUSEX memory{}; memory.dwLength = sizeof(memory);
    const bool success = GlobalMemoryStatusEx(&memory) != 0;
    result.memory = windows_system_memory(success, memory.ullTotalPhys, memory.ullAvailPhys);
    result.thermal = {{}, {}, "Windows has no corresponding public thermal-state API"};
#else
    result.memory.total.reason = result.memory.available.reason = "System memory query is unsupported";
    result.thermal.reason = "Thermal-state query is unsupported";
#endif
    return result;
}
}
