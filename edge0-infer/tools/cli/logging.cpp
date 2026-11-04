#include "edge0/engine.h"
#include <iostream>
#include <memory>
#include <mutex>

edge0::LogReceiver cli_log_receiver(bool verbose) {
    struct State {
        std::mutex mutex;
        bool continuing = false;
    };
    auto state = std::make_shared<State>();
    return [state, verbose](edge0::LogLevel level, const std::string& message) {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (level != edge0::LogLevel::continuation)
            state->continuing = level == edge0::LogLevel::warning || level == edge0::LogLevel::error;
        if (verbose || state->continuing)
            std::cerr << message;
    };
}
