#include "edge0/diagnostics.h"
#include "edge0/generation.h"
#include "edge0/generation_check.h"
#include "app/runtime_hooks.h"
#include "tests/support/runtime.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
using namespace edge0;
#undef CHECK
#define CHECK(expression) do { if (!(expression)) throw std::runtime_error(std::string("check failed: ") + #expression); } while (false)
template<class T> T take(const Result<T>& value) {
    if (!value) throw std::runtime_error(value.error().message);
    return value.value();
}
struct RuntimeCleanup {
    RuntimeReference runtime;
    ModelReference model;
    ~RuntimeCleanup() {
        if (model) { unload_model(model); model.reset(); }
        finish_runtime_use(runtime);
    }
};
void policy_replay(RuntimeReference runtime, const std::string& directory, const std::string& vulkan) {
    auto hooks = take(app::runtime_hooks(runtime));
    std::size_t choices = 0;
    hooks->policies["threshold"] = {
        [](const exec::Stages& stages) { return std::vector<std::shared_ptr<exec::Backend>>{stages.prefill, stages.decode}; },
        [&choices](const exec::Stages& stages, const exec::StageFacts& facts) {
            ++choices; return facts.tokens < 4 ? stages.decode : stages.prefill;
        }};
    LoadOverrides load;
    load.device = "CPU";
    load.prefill_device = vulkan;
    load.stage_policy = "threshold";
    load.context = 128;
    load.prefill_chunk = 128;
    load.model_options = {{"routing", "actual"}, {"prefill_ahead_min_tokens", "1"}};
    auto model = take(load_model(runtime, take(resolve_load_configuration(runtime, directory, load))));
    auto session = take(create_session(model));
    GenerationOverrides generation;
    generation.temperature = 0;
    generation.presence_penalty = 0;
    generation.max_tokens = 4;
    auto config = take(resolve_generation_configuration(model, generation));
    GenerationCheckOptions check;
    check.retain_logits = true;
    check.ignore_end_of_generation = true;
    bool fresh = true;
    for (const auto& input : std::vector<std::vector<std::int32_t>>{{1, 2, 3}, {1, 2, 3, 4, 5}}) {
        if (!fresh) take(reset_conversation(session));
        fresh = false;
        const auto original = take(generate_tokens_check(session, input, config, {}, check));
        CHECK(!original.result.error && original.result.stop_reason == StopReason::length);
        const bool short_input = input.size() < 4;
        CHECK(original.result.request_summary.prefill_device == (short_input ? "CPU" : vulkan));
        if (short_input) {
            CHECK(original.result.request_summary.handoffs == 0);
            CHECK(original.result.request_summary.handoff_bytes == 0);
            CHECK(original.result.request_summary.handoff_seconds == 0);
        } else {
            CHECK(original.result.request_summary.handoffs > 0 && original.result.request_summary.handoff_bytes > 0);
        }
        const auto calls = choices;
        take(reset_conversation(session));
        const auto replayed = take(replay_check(session, original.replay));
        CHECK(choices == calls);
        std::vector<float> sampled;
        for (std::size_t i = 0; i < original.replay.blocks.size(); ++i) if (original.replay.blocks[i].sample) {
            const auto start = replayed.logits.begin() + static_cast<std::ptrdiff_t>(i * original.vocabulary);
            sampled.insert(sampled.end(), start, start + static_cast<std::ptrdiff_t>(original.vocabulary));
        }
        CHECK(sampled.size() == original.logits.size());
        CHECK(std::memcmp(sampled.data(), original.logits.data(), original.logits.size() * sizeof(float)) == 0);
    }
    take(release_session(session));
    take(unload_model(model));
}
int main() try {
    const auto* tiny = std::getenv("MOE_TINY");
    if (!tiny) return 77;
    std::string vulkan;
    for (const auto& device : query_capabilities().devices)
        if (device.backend == "vulkan" && device.memory_total_bytes > 0) vulkan = device.name;
    if (vulkan.empty()) return 77;
    const std::string directory = std::string(tiny) + "/tiny-a";
    auto runtime = take(open_runtime({{"CPU", vulkan}, {}}));
    RuntimeCleanup cleanup{runtime, {}};
    policy_replay(runtime, directory, vulkan);
    LoadOverrides load;
    load.device = "CPU";
    load.prefill_device = vulkan;
    load.context = 128;
    load.prefill_chunk = 128;
    load.model_options = {{"routing", "actual"}, {"prefill_ahead_min_tokens", "1"}};
    cleanup.model = take(load_model(runtime, take(resolve_load_configuration(runtime, directory, load))));
    GenerationOverrides generation;
    generation.temperature = 0;
    generation.presence_penalty = 0;
    generation.max_tokens = 4;
    auto config = take(resolve_generation_configuration(cleanup.model, generation));
    GenerationCheckOptions check;
    check.retain_logits = true;
    check.ignore_end_of_generation = true;
    auto cold = [&](const std::vector<std::int32_t>& input) {
        auto session = take(create_session(cleanup.model));
        auto result = take(generate_tokens_check(session, input, config, {}, check));
        take(release_session(session));
        return result;
    };
    const std::vector<std::int32_t> input{1, 2, 3, 4, 5};
    const auto first = cold(input), second = cold(input);
    CHECK(first.result.generated == second.result.generated);
    CHECK(first.logits.size() == second.logits.size());
    CHECK(std::memcmp(first.logits.data(), second.logits.data(), first.logits.size() * sizeof(float)) == 0);
    CHECK(first.result.checkpoint_saved && first.result.checkpoint_boundary);
    auto session = take(create_session(cleanup.model));
    auto opened = take(generate_tokens_check(session, input, config, {}, check));
    CHECK(opened.result.checkpoint_saved);
    const auto stored = take(query_session_check(session));
    CHECK(stored.checkpoint_boundary && *stored.checkpoint_boundary > 0 && *stored.checkpoint_boundary < stored.e);
    auto restored = take(generate_tokens_check(session, stored.tokens, config, {}, check));
    CHECK(!restored.result.error && restored.result.checkpoint_restored && restored.result.reuse_method == ReuseMethod::restore);
    auto continued_input = take(query_session_check(session)).tokens;
    continued_input.push_back(6);
    continued_input.push_back(9);
    auto continued = take(generate_tokens_check(session, continued_input, config, {}, check));
    CHECK(!continued.result.error && continued.result.stop_reason == StopReason::length);
    auto cancelled = check;
    cancelled.cancel_probe = [](const CancelPoint& point) { return point.kind == CancelPointKind::before_step && point.token == 2; };
    auto interrupted_input = take(query_session_check(session)).tokens;
    interrupted_input.push_back(11);
    interrupted_input.push_back(12);
    auto interrupted = take(generate_tokens_check(session, interrupted_input, config, {}, cancelled));
    CHECK(interrupted.result.stop_reason == StopReason::cancelled && interrupted.result.state_valid);
    take(reset_conversation(session));
    auto reset = take(generate_tokens_check(session, input, config, {}, check));
    CHECK(!reset.result.error && reset.result.stop_reason == StopReason::length && !reset.result.text.empty());
    take(release_session(session));
    std::cout << "mixed Vulkan: replay, cold rerun, checkpoint, cancel, and reset matched\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
