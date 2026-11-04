#pragma once
#include "edge0/result.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace edge0 {
enum class TriState { yes, no, unknown };
enum class LogLevel { none, debug, info, warning, error, continuation };
using LogReceiver = std::function<void(LogLevel, const std::string&)>;
struct RuntimeOptions {
    std::vector<std::string> devices;
    LogReceiver log_receiver;
    std::optional<std::size_t> memory_budget_bytes = std::nullopt;
    std::optional<int> cpu_threads = std::nullopt;
    std::optional<int> cpu_prefill_threads = std::nullopt;
};

enum class BudgetSource { device_default, explicit_value };
struct RuntimeBudget {
    std::size_t limit_bytes;
    BudgetSource source;
};
enum class CpuThreadSource { platform_default, explicit_value };
struct RuntimeCpuThreads {
    int cpu_threads, cpu_prefill_threads;
    CpuThreadSource cpu_threads_source, cpu_prefill_threads_source;
};
struct RuntimeAccess;

// A generation-tagged reference; copying it does not create another runtime.
class RuntimeReference {
public:
    const std::vector<std::string>& available_devices() const noexcept { return devices_; }
private:
    RuntimeReference(std::uint64_t generation, std::vector<std::string> devices)
        : generation_(generation), devices_(std::move(devices)) {}
    std::uint64_t generation_;
    std::vector<std::string> devices_;
    friend struct RuntimeAccess;
    friend Result<RuntimeCpuThreads> query_runtime_cpu_threads(const RuntimeReference&);
    friend Result<RuntimeBudget> query_runtime_budget(const RuntimeReference&);
    friend Result<RuntimeReference> open_runtime(const RuntimeOptions&);
    friend Result<std::string> finish_runtime_use(const RuntimeReference&);
};

struct BackendCapability {
    std::string name;
    TriState compiled;
    TriState initializable;
    TriState verified;
    std::string reason;
};
struct DeviceCapability {
    std::string backend;
    std::string name;
    std::string description;
    std::size_t memory_total_bytes;
    std::size_t memory_free_bytes;
};
struct Capabilities {
    RuntimeCpuThreads cpu_threads{};
    std::string engine_version;
    std::string ggml_origin;
    std::string ggml_version;
    std::vector<BackendCapability> backends;
    std::vector<DeviceCapability> devices;
    // Identifiers of the models compiled into the library, written like the
    // `architecture` field of an asset directory's model.json.
    std::vector<std::string> models;
    // Build identity, refreshed on every build. The commit is empty when it could not be
    // determined; `source_modified` is unknown in that case and `source_reason` says why.
    std::string source_commit;
    TriState source_modified = TriState::unknown;
    std::string source_reason;
};

// Exactly one runtime may be open in a process. Devices must be explicit,
// nonempty and unique. No fallback is performed. GGML logging is process-wide;
// other GGML users must coordinate changes. Receivers must not throw or reenter
// lifecycle operations or capability queries, and must remain valid until
// synchronous finish returns.
Result<RuntimeReference> open_runtime(const RuntimeOptions& options = {});
Result<std::string> finish_runtime_use(const RuntimeReference& runtime);
Result<RuntimeBudget> query_runtime_budget(const RuntimeReference& runtime);
Result<RuntimeCpuThreads> query_runtime_cpu_threads(const RuntimeReference& runtime);
// Probe is opt-in and creates/releases temporary facilities, even with a runtime open.
// A receiver applies only during this query when no runtime is open. The prior
// GGML callback and user data are restored before returning, including on errors.
// Queries and runtime lifecycle calls serialize logging changes. Receivers must
// not throw or reenter queries or lifecycle operations. An open runtime keeps
// its existing logging configuration, ignoring the query receiver.
Capabilities query_capabilities(bool probe = false, const LogReceiver& log_receiver = {});
}
