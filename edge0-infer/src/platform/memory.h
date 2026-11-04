#pragma once
#include <cstdint>
#include <optional>
#include <string>
namespace edge0::platform {
struct MemoryValues {
    std::optional<std::uint64_t> current_bytes, peak_bytes;
    std::string current_reason = {}, peak_reason = {};
};
struct MemoryMetric {
    std::optional<MemoryValues> values;
    std::string reason;
    std::string source = {};
};
struct ProcessMemory { MemoryMetric rss, physical; };
ProcessMemory process_memory();
// Empty input represents an unreadable or empty proc file.
ProcessMemory parse_process_memory(const std::string& status, const std::string& smaps_rollup);
}
