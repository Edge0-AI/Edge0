#pragma once
#include "edge0/engine.h"
#include "edge0/generation.h"
#include <limits>
#include <sstream>
#include <stdexcept>
namespace cli {
inline bool runtime_option(const std::string& key, const std::string& value, edge0::RuntimeOptions& options) {
    if (key != "--threads" && key != "--prefill-threads" && key != "--memory-budget") return false;
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("invalid_argument: expected a positive integer for " + key);
    std::size_t used = 0;
    const auto count = std::stoull(value, &used);
    if (used != value.size() || count == 0) throw std::runtime_error("invalid_argument: expected a positive integer for " + key);
    if (key == "--memory-budget") {
        if (count > std::numeric_limits<std::size_t>::max()) throw std::runtime_error("invalid_argument: memory budget overflow");
        options.memory_budget_bytes = static_cast<std::size_t>(count);
    } else {
        if (count > static_cast<unsigned>(std::numeric_limits<int>::max())) throw std::runtime_error("invalid_argument: thread count overflow");
        (key == "--threads" ? options.cpu_threads : options.cpu_prefill_threads) = static_cast<int>(count);
    }
    return true;
}
inline bool device_family(const std::string& value) {
    return value == "cpu" || value == "metal" || value == "vulkan";
}
inline bool stage_option(const std::string& key, const std::string& value, edge0::LoadOverrides& loading) {
    if (key == "--stage-policy") { loading.stage_policy = value; return true; }
    if (key != "--prefill-device") return false;
    if (!device_family(value)) throw std::runtime_error("invalid_argument: prefill device must be cpu, metal, or vulkan");
    loading.prefill_device = value; return true;
}
inline edge0::Result<edge0::RuntimeReference> open_runtime(edge0::RuntimeOptions options, const std::string& device, edge0::LogReceiver logger,
    edge0::LoadOverrides* loading = nullptr) {
    options.devices = {device}; options.log_receiver = std::move(logger);
    if (loading && loading->prefill_device) {
        std::string prefill;
        for (const auto& candidate : edge0::query_capabilities(false, options.log_receiver).devices)
            if (candidate.backend == *loading->prefill_device) { prefill = candidate.name; break; }
        if (prefill.empty()) return edge0::Error{edge0::ErrorCategory::unsupported, "unsupported: prefill device is unavailable"};
        loading->prefill_device = prefill;
        if (prefill != device) { options.devices.push_back(prefill); loading->device = device; }
    }
    return edge0::open_runtime(options);
}
inline const char* thread_source(edge0::CpuThreadSource source) {
    return source == edge0::CpuThreadSource::explicit_value ? "explicit_value" : "platform_default";
}
inline void runtime_json(std::ostream& out, const edge0::RuntimeReference& runtime) {
    const auto value = edge0::query_runtime_cpu_threads(runtime);
    if (!value) throw std::runtime_error(value.error().message);
    const auto& threads = value.value();
    out << "{\"cpu_threads\":" << threads.cpu_threads << ",\"cpu_prefill_threads\":" << threads.cpu_prefill_threads
        << ",\"cpu_threads_source\":\"" << thread_source(threads.cpu_threads_source)
        << "\",\"cpu_prefill_threads_source\":\"" << thread_source(threads.cpu_prefill_threads_source) << "\"}";
}
inline std::string with_runtime(std::string summary, const edge0::RuntimeReference& runtime) {
    const auto end = summary.rfind('}');
    if (end == std::string::npos) throw std::runtime_error("Invalid summary JSON");
    std::ostringstream field; field << ",\"runtime\":"; runtime_json(field, runtime);
    summary.insert(end, field.str()); return summary;
}
inline void runtime_line(std::ostream& out, const edge0::RuntimeReference& runtime) {
    out << "runtime: "; runtime_json(out, runtime); out << '\n';
}
}
