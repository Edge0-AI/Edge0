#include "exec/execution.h"
#include "tests/support/check.h"
using namespace edge0;
using namespace edge0::exec;
int main() {
    auto a = platform::BackendFacility::create("CPU"), b = platform::BackendFacility::create("CPU");
    CHECK(a.first && b.first);
    std::map<std::string, std::shared_ptr<platform::BackendFacility>> facilities{
        {"first", std::make_shared<platform::BackendFacility>(std::move(a.first))},
        {"second", std::make_shared<platform::BackendFacility>(std::move(b.first))}};
    auto same = resolve_stages("first", "first", facilities);
    CHECK(same && !same.value().mixed() && !same.value().runs);
    auto mixed = resolve_stages("first", "second", facilities);
    CHECK(mixed && mixed.value().mixed() && mixed.value().runs);
    const auto& stages = mixed.value();
    CHECK(stages.of(Phase::prefill) == same.value().prefill);
    CHECK(stages.of(Phase::decode) != stages.of(Phase::prefill));
    auto missing = resolve_stages("first", "missing", facilities);
    CHECK(!missing && missing.error().category == ErrorCategory::unsupported);
    auto fixed = stage_policy("fixed"); CHECK(fixed);
    const auto candidates = fixed.value().candidates(stages);
    CHECK(candidates == std::vector<std::shared_ptr<Backend>>{stages.prefill});
    auto selected = choose_prefill_backend(stages, fixed.value(), candidates, {});
    CHECK(selected && selected.value() == stages.prefill);
    CHECK(!stage_policy("missing"));
    StagePolicy threshold{
        [](const Stages& s) { return std::vector<std::shared_ptr<Backend>>{s.prefill, s.decode}; },
        [](const Stages& s, const StageFacts& facts) { return facts.tokens < 4 ? s.decode : s.prefill; }};
    CHECK(choose_prefill_backend(stages, threshold, threshold.candidates(stages), {2, {}, 0}).value() == stages.decode);
    CHECK(choose_prefill_backend(stages, threshold, threshold.candidates(stages), {8, {}, 0}).value() == stages.prefill);
    auto outside = choose_prefill_backend(stages, threshold, candidates, {2, {}, 0});
    CHECK(!outside && outside.error().category == ErrorCategory::invalid_argument);
    CHECK(!choose_prefill_backend(stages, threshold, {nullptr}, {}));
    CHECK(!choose_prefill_backend(stages, {}, candidates, {}));
    std::cout << "stages: instance identity, runtime scope, fixed and injected policies passed\n";
}
