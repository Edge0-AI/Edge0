#include "edge0/diagnostics.h"
#include "app/runtime.h"
#include "platform/system.h"
namespace edge0 {
Result<MemorySnapshot> memory_snapshot(const RuntimeReference& runtime) {
    auto acquired = RuntimeAccess::acquire(runtime);
    if (!acquired) return acquired.error();
    const auto& budget = acquired.value().budget;
    const auto process = acquired.value().registry->memory_query();
    auto metric = [](const platform::MemoryMetric& source) {
        MemoryMetric result;
        result.reason = source.reason;
        result.source = source.source;
        if (source.values) result.values = MemoryValues{source.values->current_bytes, source.values->peak_bytes, source.values->current_reason, source.values->peak_reason};
        return result;
    };
    const auto snapshot = budget.snapshot();
    MemorySnapshot result{{snapshot.limit, snapshot.current, snapshot.peak}, metric(process.rss), metric(process.physical)};
    for (const auto& entry : snapshot.by_purpose)
        result.ledger.by_purpose.emplace(static_cast<MemoryPurpose>(entry.first), entry.second);
    for (const auto& entry : snapshot.by_device)
        result.ledger.by_device.emplace(entry.first, DeviceMemory{entry.second.current, entry.second.peak, platform::device_memory(entry.first)});
    return result;
}
}

namespace edge0 {
const char* memory_purpose_name(MemoryPurpose purpose) {
    switch (purpose) {
    case MemoryPurpose::resident_weights: return "resident_weights";
    case MemoryPurpose::experts: return "experts";
    case MemoryPurpose::session_state: return "session_state";
    case MemoryPurpose::workspace: return "workspace";
    case MemoryPurpose::transfer: return "transfer";
    case MemoryPurpose::staging: return "staging";
    case MemoryPurpose::metadata: return "metadata";
    }
    return "unknown";
}
Result<GenerationProgress> query_generation_progress(const RuntimeReference& runtime) {
    auto resources = RuntimeAccess::acquire(runtime);
    if (!resources) return resources.error();
    const auto& registry = resources.value().registry;
    std::lock_guard<std::mutex> lock(registry->progress_mutex);
    return registry->progress;
}
}

namespace edge0 {
SystemStatus query_system_status() { return platform::system_status(); }
}
