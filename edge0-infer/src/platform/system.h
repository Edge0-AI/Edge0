#pragma once
#include "edge0/diagnostics.h"
#include "ggml-backend.h"
namespace edge0::platform {
SystemMemory parse_system_memory(const std::string&);
SystemMemory windows_system_memory(bool success, std::uint64_t total, std::uint64_t available);
ThermalStatus apple_thermal_status(long);
ThermalStatus android_thermal_status(int);
ThermalStatus linux_thermal_unavailable(bool android);
SystemStatus system_status();
MemoryLocation classify_device_memory(std::optional<enum ggml_backend_dev_type> type, bool unified_memory);
MemoryLocation device_memory(const std::string& name);
}
