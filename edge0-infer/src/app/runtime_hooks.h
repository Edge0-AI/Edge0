#pragma once
#include "app/runtime.h"
namespace edge0::app {
struct RuntimeHooks {
    std::map<std::string, std::shared_ptr<exec::Backend>> aliases;
    exec::ConsumptionQuery consumption;
    exec::TransferHook transfer;
    std::function<bool(ggml_backend_t, ggml_backend_buffer_t)> direct;
    std::map<std::string, exec::StagePolicy> policies;
    std::size_t hand_off_calls = 0;
    bool skip_handoff = false;
};
Result<std::shared_ptr<RuntimeHooks>> runtime_hooks(const RuntimeReference&);
exec::Status add_backend_alias(const RuntimeReference&, const std::string& alias, const std::string& native_device);
}
