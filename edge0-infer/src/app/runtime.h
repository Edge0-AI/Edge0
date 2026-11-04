#pragma once
#include "edge0/engine.h"
#include "edge0/diagnostics.h"
#include "platform/memory.h"
#include "exec/execution.h"
#include <thread>
namespace edge0 {
namespace app { struct ModelHandle; struct SessionHandle; struct RuntimeHooks; }
struct RuntimeRegistry {
    std::shared_ptr<app::RuntimeHooks> hooks;
    platform::ProcessMemory (*memory_query)() = platform::process_memory;
    std::mutex progress_mutex;
    GenerationProgress progress;
    std::mutex mutex;
    bool active = false, ended = false;
    std::thread::id callback_thread;
    std::vector<std::shared_ptr<app::ModelHandle>> models;
    std::vector<std::shared_ptr<app::SessionHandle>> sessions;
};
struct RuntimeResources {
    exec::Budget budget;
    std::map<std::string, std::shared_ptr<exec::Backend>> backends;
    std::shared_ptr<RuntimeRegistry> registry;
};
inline void select_cpu_threads(const RuntimeResources& resources, exec::Phase phase) {
    for (const auto& entry : resources.backends)
        entry.second->facility->select_cpu_use(phase == exec::Phase::prefill ? platform::CpuUse::multi_token : platform::CpuUse::single_token);
}
class RuntimeActivity {
public:
    explicit RuntimeActivity(std::shared_ptr<RuntimeRegistry> registry) : registry_(std::move(registry)) {}
    exec::Status enter() {
        std::lock_guard<std::mutex> lock(registry_->mutex);
        if (registry_->ended) return Error{ErrorCategory::runtime_ended, "Runtime has ended"};
        if (registry_->active || registry_->callback_thread != std::thread::id{})
            return Error{ErrorCategory::busy, "Runtime has an active call or callback"};
        registry_->active = true; entered_ = true; return {};
    }
    ~RuntimeActivity() { if (entered_) { std::lock_guard<std::mutex> lock(registry_->mutex); registry_->active = false; } }
    RuntimeActivity(const RuntimeActivity&) = delete;
    RuntimeActivity& operator=(const RuntimeActivity&) = delete;
private:
    std::shared_ptr<RuntimeRegistry> registry_;
    bool entered_ = false;
};
struct RuntimeAccess {
    static Result<RuntimeResources> acquire(const RuntimeReference& runtime);
};
}
