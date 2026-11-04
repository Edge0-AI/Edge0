#pragma once
#include <edge0/diagnostics.h>
#include <iomanip>
#include <sstream>
#include <string>

namespace android_status {
inline std::string quoted(const std::string& input) {
    std::ostringstream out; out << '"';
    for (unsigned char c : input) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
        else out << c;
    }
    out << '"'; return out.str();
}
struct JsonObject {
    std::ostringstream out; bool first = true;
    JsonObject() { out << std::setprecision(17) << '{'; }
    template<class T> void field(const std::string& name, const T& value) { if (!first) out << ','; first = false; out << quoted(name) << ':' << value; }
    template<class T> void optional(const std::string& name, const std::optional<T>& value) { if (value) field(name, *value); }
    std::string str() { return out.str() + '}'; }
};
inline const char* purpose_name(edge0::MemoryPurpose purpose) {
    switch(purpose) {
        case edge0::MemoryPurpose::resident_weights: return "resident_weights";
        case edge0::MemoryPurpose::experts: return "experts";
        case edge0::MemoryPurpose::session_state: return "session_state";
        case edge0::MemoryPurpose::workspace: return "workspace";
        case edge0::MemoryPurpose::transfer: return "transfer";
        case edge0::MemoryPurpose::staging: return "staging";
        case edge0::MemoryPurpose::metadata: return "metadata";
    }
    return "unknown";
}
inline std::string progress_json(const edge0::GenerationProgress& progress, std::chrono::steady_clock::time_point now) {
    JsonObject json;
    const auto phase = progress.phase == edge0::GenerationPhase::prefill ? "prefill" : progress.phase == edge0::GenerationPhase::decode ? "decode" : "idle";
    json.field("phase", quoted(phase));
    json.field("inputTokens", progress.input_tokens); json.field("reusedTokens", progress.reused_tokens);
    json.field("computedTokens", progress.computed_tokens); json.field("pendingTokens", progress.pending_tokens); json.field("generatedTokens", progress.generated_tokens);
    if (progress.phase != edge0::GenerationPhase::idle) {
        json.field("elapsedSeconds", std::chrono::duration<double>(now - progress.started).count());
        if (progress.first_token) {
            json.field("firstTokenSeconds", std::chrono::duration<double>(*progress.first_token - progress.started).count());
            json.field("decodeElapsedSeconds", std::chrono::duration<double>(now - *progress.first_token).count());
        }
    }
    return json.str();
}
inline std::string memory_json(const edge0::MemorySnapshot& memory) {
    JsonObject json;
    json.field("limitBytes", memory.ledger.limit_bytes); json.field("currentBytes", memory.ledger.current_bytes); json.field("peakBytes", memory.ledger.peak_bytes);
    JsonObject purposes; for (const auto& pair : memory.ledger.by_purpose) purposes.field(purpose_name(pair.first), pair.second);
    json.field("byPurpose", purposes.str());
    JsonObject devices;
    for (const auto& pair : memory.ledger.by_device) {
        JsonObject device; device.field("currentBytes", pair.second.current_bytes); device.field("peakBytes", pair.second.peak_bytes);
        device.field("memory", quoted(pair.second.memory == edge0::MemoryLocation::dedicated ? "dedicated" : "system")); devices.field(pair.first, device.str());
    }
    json.field("byDevice", devices.str());
    if (memory.physical.values) json.optional("physicalBytes", memory.physical.values->current_bytes);
    if (memory.rss.values) json.optional("residentBytes", memory.rss.values->current_bytes);
    return json.str();
}
inline std::string system_json(const edge0::SystemStatus& system) {
    JsonObject json; json.optional("totalBytes", system.memory.total.bytes); json.optional("availableBytes", system.memory.available.bytes);
    json.optional("appLimitBytes", system.app_memory_limit.bytes);
    if (system.thermal.state) json.field("thermal", quoted(*system.thermal.state)); return json.str();
}
}
