#include "edge0/generation.h"
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

template<class T> T take(const edge0::Result<T>& result) {
    if (!result) throw std::runtime_error(result.error().message);
    return result.value();
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main() {
    const auto* assets = std::getenv("MOE_TINY");
    if (!assets || !std::filesystem::is_directory(std::filesystem::path(assets) / "tiny-a")) {
        std::cout << "SKIP: MOE_TINY/tiny-a unavailable\n";
        return 77;
    }
    try {
        std::string device;
        for (const auto& d : edge0::query_capabilities().devices)
            if (d.backend == "cpu") device = d.name;
        auto runtime = take(edge0::open_runtime({{device}, {}}));
        auto model = take(edge0::load_model(runtime, take(edge0::resolve_load_configuration(
            runtime, (std::filesystem::path(assets) / "tiny-a").string()))));
        require(take(edge0::query_model(model)).loading.model_options.at("routing") == "predicted",
                "Default predicted routing must exercise pending preparation");
        auto session = take(edge0::create_session(model));
        edge0::GenerationOverrides options;
        options.temperature = 0;
        options.presence_penalty = 0;
        options.max_tokens = 4096;
        options.logprobs = true;
        options.top_logprobs = 20;
        auto configuration = take(edge0::resolve_generation_configuration(model, options));
        const edge0::ChatInput input{{{"user", "hello"}}};
        unsigned events = 0;
        const auto first = take(edge0::generate(session, input, configuration, {},
            [&](const edge0::OutputEvent& event) -> std::optional<edge0::Error> {
                if (event.kind == edge0::OutputKind::answer && ++events == 200)
                    return edge0::Error{edge0::ErrorCategory::backend_error, "Injected delivery failure"};
                return {};
            }));
        require(events == 200 && first.stop_reason == edge0::StopReason::failed && first.error &&
                first.error_stage == edge0::GenerationStage::delivery && first.state_valid,
                "Delivery failure must retain valid state");
        options.max_tokens = 4;
        options.logprobs = false;
        options.top_logprobs = 0;
        configuration = take(edge0::resolve_generation_configuration(model, options));
        const auto second = take(edge0::generate(session, input, configuration));
        std::cout << "continued state_valid=" << second.state_valid << " stage=" << int(second.error_stage)
                  << " error=" << (second.error ? second.error->message : "") << '\n';
        require(!second.error && second.state_valid &&
                (second.stop_reason == edge0::StopReason::length || second.stop_reason == edge0::StopReason::end_of_generation),
                "Generation after delivery failure must finish normally");
        take(edge0::release_session(session));
        auto fresh_session = take(edge0::create_session(model));
        const auto fresh = take(edge0::generate(fresh_session, input, configuration));
        require(!fresh.error && fresh.state_valid && fresh.stop_reason == second.stop_reason &&
                !fresh.generated.empty() && fresh.generated == second.generated,
                "Continued generation must match fresh session token for token");
        take(edge0::release_session(fresh_session));
        take(edge0::unload_model(model));
        take(edge0::finish_runtime_use(runtime));
        std::cout << "PASS: delivery failure, valid state, normal continuation, fresh token equality\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
