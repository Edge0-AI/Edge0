#include "platform/system.h"
#include "tests/support/check.h"
int main() {
    using namespace edge0::platform;
    using edge0::MemoryLocation;
    CHECK(classify_device_memory(GGML_BACKEND_DEVICE_TYPE_CPU, false) == MemoryLocation::system);
    CHECK(classify_device_memory(std::nullopt, false) == MemoryLocation::system);
    CHECK(classify_device_memory(GGML_BACKEND_DEVICE_TYPE_GPU, true) == MemoryLocation::system);
    CHECK(classify_device_memory(GGML_BACKEND_DEVICE_TYPE_IGPU, false) == MemoryLocation::system);
    CHECK(classify_device_memory(GGML_BACKEND_DEVICE_TYPE_GPU, false) == MemoryLocation::dedicated);
    CHECK(device_memory("host") == MemoryLocation::system);
    CHECK(device_memory("host") == MemoryLocation::system);
    auto memory = parse_system_memory("MemTotal: 100 kB\nMemAvailable: 40 kB\n");
    CHECK(memory.total.bytes == 102400 && memory.available.bytes == 40960);
    CHECK(!parse_system_memory("MemTotal: 5 kB\n").available.bytes);
    CHECK(!parse_system_memory("MemTotal: -5 kB\n").total.bytes);
    CHECK(!parse_system_memory("MemTotal: 18446744073709551615 kB\n").total.bytes);
    CHECK(!parse_system_memory("MemTotal: 5 kB\nMemTotal: 4 kB\n").total.bytes);
    const char* android_names[]{"none", "light", "moderate", "severe", "critical", "emergency", "shutdown"};
    for (int state = 0; state < 7; ++state) CHECK(android_thermal_status(state).state == android_names[state]);
    CHECK(!android_thermal_status(-1).state && !android_thermal_status(7).state);
    {
        const auto android = linux_thermal_unavailable(true), linux = linux_thermal_unavailable(false);
        CHECK(!android.state && android.source == "AThermal_getCurrentThermalStatus" && android.reason.find("Android 11") != std::string::npos);
        CHECK(!linux.state && linux.source.empty() && !linux.reason.empty() && linux.reason.find("Android") == std::string::npos);
    }
    const char* apple_names[]{"nominal", "fair", "serious", "critical"};
    for (long state = 0; state < 4; ++state) CHECK(apple_thermal_status(state).state == apple_names[state]);
    CHECK(!apple_thermal_status(4).state);
    CHECK(windows_system_memory(true, 100, 40).available.bytes == 40);
    CHECK(!windows_system_memory(false, 100, 40).total.bytes);
    const auto system = system_status();
#if defined(__APPLE__)
    CHECK(system.memory.total.bytes && *system.memory.total.bytes > 0);
    CHECK(system.memory.available.bytes && *system.memory.available.bytes <= *system.memory.total.bytes);
    CHECK(system.thermal.state && system.thermal.reason.empty());
    CHECK(!system.app_memory_limit.bytes && !system.app_memory_limit.reason.empty());
#endif
    std::cout << "system: memory parsing, unavailable values, platform mappings and live system query passed\n";
}
