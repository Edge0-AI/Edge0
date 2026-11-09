#pragma once
#include "edge0/engine.h"
#include <chrono>
#include <map>
#include <cstdint>
#include <optional>
namespace edge0 {
enum class MemoryScope { runtime_lifetime, process_lifetime };
struct MemoryValues {
    std::optional<std::uint64_t> current_bytes, peak_bytes;
    std::string current_reason = {}, peak_reason = {};
};
struct MemoryMetric {
    std::optional<MemoryValues> values;
    std::string reason;
    MemoryScope scope = MemoryScope::process_lifetime;
    std::string source = {};
};
enum class MemoryPurpose { resident_weights, experts, session_state, workspace, transfer, staging, metadata };
const char* memory_purpose_name(MemoryPurpose);
enum class MemoryLocation { system, dedicated };
struct DeviceMemory {
    std::size_t current_bytes = 0, peak_bytes = 0;
    MemoryLocation memory = MemoryLocation::system;
};
struct MemoryLedger {
    std::size_t limit_bytes = 0, current_bytes = 0, peak_bytes = 0;
    MemoryScope scope = MemoryScope::runtime_lifetime;
    std::map<MemoryPurpose, std::size_t> by_purpose = {};
    std::map<std::string, DeviceMemory> by_device = {};
};
struct MemorySnapshot { MemoryLedger ledger; MemoryMetric rss, physical; };
enum class GenerationPhase { idle, prefill, decode };
struct GenerationProgress {
    GenerationPhase phase = GenerationPhase::idle;
    std::size_t input_tokens = 0, reused_tokens = 0, computed_tokens = 0, pending_tokens = 0, generated_tokens = 0;
    std::chrono::steady_clock::time_point started = {};
    std::optional<std::chrono::steady_clock::time_point> first_token = {};
};
Result<GenerationProgress> query_generation_progress(const RuntimeReference&);
struct SystemValue {
    std::optional<std::uint64_t> bytes = {};
    std::string source, reason;
};
struct SystemMemory { SystemValue total, available; };
struct ThermalStatus { std::optional<std::string> state = {}; std::string source, reason; };
struct AppMemoryLimit : SystemValue {
    // The same phys_footprint reading used to compute the limit, on iOS only.
    std::optional<std::uint64_t> current_physical_bytes = {};
};
struct SystemStatus { SystemMemory memory; ThermalStatus thermal; AppMemoryLimit app_memory_limit; };
SystemStatus query_system_status();
// Explicit, synchronous OS query. Process peaks cannot be reset per request.
Result<MemorySnapshot> memory_snapshot(const RuntimeReference&);
}
