#include "edge0/engine.h"
#include "edge0/diagnostics.h"
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

edge0::LogReceiver cli_log_receiver(bool verbose);
namespace {
const char *state(edge0::TriState value) {
    switch (value) {
    case edge0::TriState::yes:
        return "yes";
    case edge0::TriState::no:
        return "no";
    case edge0::TriState::unknown:
        return "unknown";
    }
    return "unknown";
}
} // namespace

int info_command(int argc, char **argv) {
    bool probe = false, verbose = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--probe" && !probe) probe = true;
        else if (arg == "--verbose" && !verbose) verbose = true;
        else {
            std::cerr << "Usage: edge0-infer info [--probe] [--verbose]\n";
            return 2;
        }
    }
    try {
        const auto caps = edge0::query_capabilities(probe, cli_log_receiver(verbose));
        std::cout << "engine.version: " << caps.engine_version << '\n'
                  << "engine.source_commit: " << caps.source_commit << '\n'
                  << "engine.source_modified: " << state(caps.source_modified);
        if (!caps.source_reason.empty()) std::cout << " reason=" << std::quoted(caps.source_reason);
        std::cout << '\n'
                  << "ggml.origin: llama.cpp " << caps.ggml_origin << " ggml/\n"
                  << "ggml.version: " << caps.ggml_version << '\n';
        std::cout << "runtime.cpu_threads: " << caps.cpu_threads.cpu_threads << " source=platform_default\n"
                  << "runtime.cpu_prefill_threads: " << caps.cpu_threads.cpu_prefill_threads << " source=platform_default\n";
        std::cout << "loading.prefill_device: device source=platform_default\n"
                  << "loading.stage_policy: fixed source=platform_default\n";
        for (std::size_t i = 0; i < caps.models.size(); ++i)
            std::cout << "model." << i << ": " << caps.models[i] << '\n';
        for (const auto &backend : caps.backends) {
            std::cout << "backend." << backend.name << ": compiled=" << state(backend.compiled)
                      << " initializable=" << state(backend.initializable)
                      << " verified=" << state(backend.verified);
            if (!backend.reason.empty())
                std::cout << " reason=" << std::quoted(backend.reason);
            std::cout << '\n';
        }
        for (std::size_t i = 0; i < caps.devices.size(); ++i) {
            const auto &device = caps.devices[i];
            std::cout << "device." << i << ": backend=" << device.backend << " name=" << device.name
                      << " description=" << std::quoted(device.description)
                      << " memory_total_bytes=" << device.memory_total_bytes
                      << " memory_free_bytes=" << device.memory_free_bytes << '\n';
        }
        const auto system = edge0::query_system_status();
        auto value = [](const char* name, const edge0::SystemValue& item) {
            std::cout << "system." << name << ": ";
            if (item.bytes) std::cout << *item.bytes; else std::cout << "unavailable";
            std::cout << " source=" << std::quoted(item.source) << " reason=" << std::quoted(item.reason) << '\n';
        };
        value("memory.total_bytes", system.memory.total);
        value("memory.available_bytes", system.memory.available);
        value("app_memory_limit", system.app_memory_limit);
        if (system.app_memory_limit.current_physical_bytes)
            value("app_memory_limit.current_physical_bytes", {*system.app_memory_limit.current_physical_bytes, "phys_footprint", {}});
        std::cout << "system.thermal: " << system.thermal.state.value_or("unavailable")
                  << " source=" << std::quoted(system.thermal.source) << " reason=" << std::quoted(system.thermal.reason) << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Capability query failed: " << error.what() << '\n';
        return 1;
    }
}
