#include "exec/execution.h"
#include "tests/support/check.h"
#include <cstring>
#include <algorithm>
using namespace edge0;
using namespace edge0::exec;
namespace {
template<class T> T take(const Result<T>& result) {
    if (!result) { std::cerr << result.error().message << '\n'; std::exit(1); }
    return result.value();
}
std::shared_ptr<Backend> backend() {
    auto facility = platform::BackendFacility::create("CPU"); CHECK(facility.first);
    auto owner = std::make_shared<platform::BackendFacility>(std::move(facility.first));
    auto stages = take(resolve_stages("CPU", "CPU", owner));
    CHECK(stages.prefill == stages.decode);
    CHECK(take(resolve_stages("CPU", "CPU", owner)).prefill == stages.prefill);
    auto mixed = resolve_stages("CPU", "other-device", owner);
    CHECK(!mixed && mixed.error().category == ErrorCategory::unsupported);
    return stages.prefill;
}
struct CounterUse : Use {
    int& count; bool active = true;
    explicit CounterUse(int& value) : count(value) { ++count; }
    ~CounterUse() { if (active) release(); }
    Status release() override {
        if (!active) return Error{ErrorCategory::conflict, "Released"};
        active = false; --count; return {};
    }
    bool valid() const override { return active; }
    bool matches(ggml_tensor*, ggml_backend_t) const override { return active; }
    std::shared_ptr<void> keep_alive() const override { return {}; }
    std::string identity() const override { return "test"; }
};
struct Simple {
    std::shared_ptr<Graph> graph;
    std::shared_ptr<Plan> plan;
    ggml_tensor* output;
};
Simple simple(Budget& budget, const std::shared_ptr<Backend>& b) {
    auto graph = take(create_graph(budget, b, Phase::decode));
    auto* x = take(graph->input("x", {GGML_TYPE_F32, {4}}));
    auto* y = take(graph->input("y", {GGML_TYPE_F32, {4}}));
    auto* out = ggml_add(graph->context(), x, y);
    CHECK(!graph->output(out));
    auto plan = take(build_plan(graph));
    return {graph, plan, out};
}
}
int main() {
    auto b = backend();
    Budget budget(64 * 1024 * 1024);
    {
        auto state = take(allocate_state_storage(budget, b, {GGML_TYPE_F32, {32}}, edge0::exec::Purpose::session_state));
        CHECK(budget.current(edge0::exec::Purpose::session_state) == ggml_backend_buffer_get_size(state->allocation->buffer));
        CHECK(budget.current() == budget.current(edge0::exec::Purpose::session_state) + state->allocation->charges[1]->size());
        Budget small(1);
        auto rejected = allocate_state_storage(small, b, {GGML_TYPE_F32, {32}}, edge0::exec::Purpose::session_state);
        CHECK(!rejected && rejected.error().category == ErrorCategory::budget_exceeded);
        CHECK(small.current() == 0 && small.peak() == 0);
        auto other = backend();
        auto graph = take(create_graph(budget, other, Phase::prefill));
        const auto occupied = budget.current();
        auto mismatch = graph->bind_states(state);
        CHECK(!mismatch && mismatch.error().category == ErrorCategory::invalid_argument);
        CHECK(budget.current() == occupied);
    }
    CHECK(budget.current() == 0);
    {
        auto graph = take(create_graph(budget, b, Phase::decode));
        auto* x = take(graph->input("x", {GGML_TYPE_F32, {4}}));
        CHECK(!graph->output(ggml_silu(graph->context(), x)));
        auto rejected = build_plan(graph, [](ggml_backend_t, const ggml_tensor*) { return false; });
        CHECK(!rejected && rejected.error().category == ErrorCategory::unsupported);
        CHECK(rejected.error().message.find("f32") != std::string::npos);
        CHECK(rejected.error().message.find("UNARY") != std::string::npos);
    }
    {
        auto external = take(allocate_state_storage(budget, b, {GGML_TYPE_F32, {4}}, edge0::exec::Purpose::session_state));
        auto graph = take(create_graph(budget, b, Phase::decode));
        CHECK(!graph->output(ggml_scale(graph->context(), external->tensor(), 2)));
        auto rejected = build_plan(graph);
        CHECK(!rejected && rejected.error().category == ErrorCategory::invalid_argument);
    }
    // A computed ancestor can feed both the next state and the layer output.
    for (bool indirect : {false, true}) {
        auto state = take(allocate_state_storage(budget, b, {GGML_TYPE_F32, {4}}, edge0::exec::Purpose::session_state));
        float initial[4]{1, -2, 3, -4};
        ggml_backend_tensor_set(state->tensor(), initial, 0, sizeof(initial));
        auto graph = take(create_graph(budget, b, Phase::decode));
        auto* old = take(graph->bind_states(state));
        auto* input = take(graph->input("x", {GGML_TYPE_F32, {4}}));
        auto* mixed = ggml_add(graph->context(), ggml_reshape_1d(graph->context(), old, 4), input);
        auto* next = indirect ? ggml_scale(graph->context(), mixed, 3) : mixed;
        take(graph->bind_state_write(state, ggml_cpy(graph->context(), next, old)));
        auto* output = ggml_scale(graph->context(), ggml_view_1d(graph->context(), mixed, 4, 0), 2);
        CHECK(!graph->output(output));
        auto plan = take(build_plan(graph));
        double reference[4]{1, -2, 3, -4};
        float values[4]{0.125f, -0.25f, 0.375f, -0.5f};
        for (int step = 0; step < 4; ++step) {
            CHECK(!plan->set_inputs("x", values, sizeof(values))); CHECK(!plan->submit());
            auto completion = take(plan->collect_completion());
            CHECK(completion.success && completion.written.size() == 1 && completion.written[0] == state->allocation);
            float actual[4]{}, updated[4]{};
            CHECK(!plan->read_output(output, actual, 0, sizeof(actual)));
            ggml_backend_tensor_get(state->tensor(), updated, 0, sizeof(updated));
            for (int i = 0; i < 4; ++i) {
                const double mixed_reference = reference[i] + values[i];
                reference[i] = mixed_reference * (indirect ? 3 : 1);
                CHECK(actual[i] == mixed_reference * 2 && updated[i] == reference[i]);
            }
            CHECK(!plan->end_observation());
        }
    }
    CHECK(budget.current() == 0);
    // Sharing a pure alias with the write does not order an output's old-value read.
    for (int kind = 0; kind < 5; ++kind) {
        auto state = take(allocate_state_storage(budget, b, {GGML_TYPE_F32, {4, 4}}, edge0::exec::Purpose::session_state));
        auto graph = take(create_graph(budget, b, Phase::decode));
        auto* old = take(graph->bind_states(state));
        auto* alias = old;
        switch (kind) {
        case 1: alias = ggml_view_2d(graph->context(), old, 4, 4, old->nb[1], 0); break;
        case 2: alias = ggml_reshape_2d(graph->context(), old, 4, 4); break;
        case 3: alias = ggml_transpose(graph->context(), old); break;
        case 4: alias = ggml_permute(graph->context(), old, 1, 0, 2, 3); break;
        default: break;
        }
        take(graph->bind_state_write(state, ggml_cpy(graph->context(), ggml_cont(graph->context(), alias), old)));
        CHECK(!graph->output(alias));
        const auto occupied = budget.current();
        auto result = build_plan(graph);
        CHECK(!result && result.error().category == ErrorCategory::invalid_argument);
        CHECK(budget.current() == occupied);
    }
    CHECK(budget.current() == 0);
    for (int kind = 0; kind < 6; ++kind) for (bool dependent : {false, true}) {
        auto state = take(allocate_state_storage(budget, b, {GGML_TYPE_F32, {4, 4}}, edge0::exec::Purpose::session_state));
        auto graph = take(create_graph(budget, b, Phase::decode));
        auto* target = take(graph->bind_states(state));
        // Aliases in the update itself are legal old-value reads.
        auto* old = ggml_reshape_2d(graph->context(), target, 4, 4);
        auto* write = take(graph->bind_state_write(state,
            ggml_cpy(graph->context(), ggml_scale(graph->context(), old, 2), target)));
        auto* read = dependent ? write : target;
        switch (kind) {
        case 0: read = ggml_view_2d(graph->context(), read, 4, 4, read->nb[1], 0); break;
        case 1: read = ggml_reshape_2d(graph->context(), read, 4, 4); break;
        case 2: read = ggml_transpose(graph->context(), read); break;
        case 3: read = ggml_permute(graph->context(), read, 1, 0, 2, 3); break;
        case 4: read = ggml_reshape_2d(graph->context(), ggml_view_1d(graph->context(), read, 16, 0), 4, 4); break;
        default: break;
        }
        CHECK(!graph->output(ggml_scale(graph->context(), ggml_cont(graph->context(), read), 2)));
        const auto occupied = budget.current();
        auto result = build_plan(graph);
        if (dependent) CHECK(result);
        else {
            CHECK(!result && result.error().category == ErrorCategory::invalid_argument);
            CHECK(budget.current() == occupied);
        }
    }
    for (bool registered : {false, true}) {
        auto state = take(allocate_state_storage(budget, b, {GGML_TYPE_F32, {4}}, edge0::exec::Purpose::session_state));
        auto graph = take(create_graph(budget, b, Phase::decode));
        if (registered) take(graph->bind_states(state));
        CHECK(!graph->output(state->tensor()));
        const auto occupied = budget.current();
        auto result = build_plan(graph);
        if (registered) CHECK(result);
        else {
            CHECK(!result && result.error().category == ErrorCategory::invalid_argument);
            CHECK(budget.current() == occupied);
        }
    }
    CHECK(budget.current() == 0);
    {
        auto a = simple(budget, b); auto c = simple(budget, b);
        const auto occupied = budget.current();
        CHECK(check_plan_reuse(*a.plan, 1, {}).decision == ReuseDecision::reuse);
        CHECK(check_plan_reuse(*a.plan, 2, {}).reason == "token_count_changed");
        CHECK(budget.current() == occupied);
        float x[4]{1, 2, 3, 4}, y[4]{5, 6, 7, 8}, out[4]{};
        CHECK(!a.plan->set_inputs("x", x, sizeof(x)));
        auto missing = a.plan->submit();
        CHECK(missing && missing->category == ErrorCategory::invalid_argument && missing->message.find("y") != std::string::npos);
        CHECK(!a.plan->set_inputs("x", x, sizeof(x))); CHECK(!a.plan->set_inputs("y", y, sizeof(y)));
        CHECK(!a.plan->submit());
        CHECK(a.plan->read_output(a.output, out, 0, sizeof(out)));
        CHECK(a.plan->release_plan());
        CHECK(check_plan_reuse(*a.plan, 1, {}).reason == "plan_observed");
        CHECK(!c.plan->set_inputs("x", x, sizeof(x))); CHECK(!c.plan->set_inputs("y", y, sizeof(y)));
        auto busy = c.plan->submit(); CHECK(busy && busy->category == ErrorCategory::conflict);
        CHECK(take(a.plan->collect_completion()).success);
        CHECK(a.plan->submit());
        CHECK(!a.plan->read_output(a.output, out, 0, sizeof(out)));
        CHECK(out[0] == 6 && out[3] == 12);
        CHECK(!a.plan->end_observation());
        CHECK(a.plan->submit());
        CHECK(!a.plan->release_plan());
        CHECK(check_plan_reuse(*a.plan, 1, {}).reason == "plan_released");
    }
    CHECK(budget.current() == 0);
    {
        auto state = take(allocate_state_storage(budget, b, {GGML_TYPE_F32, {4}}, edge0::exec::Purpose::session_state));
        auto graph = take(create_graph(budget, b, Phase::decode));
        auto* input = take(graph->input("x", {GGML_TYPE_F32, {4}}));
        CHECK(!graph->retain_output(input, state));
        auto plan = take(build_plan(graph));
        float values[4]{2, 3, 4, 5}; CHECK(!plan->set_inputs("x", values, sizeof(values))); CHECK(!plan->submit());
        { auto completion = take(plan->collect_completion()); CHECK(completion.success && completion.written.size() == 1); }
        CHECK(!plan->end_observation());
        auto c = simple(budget, b);
        for (int i = 0; i < 5; ++i) {
            CHECK(!c.plan->set_inputs("x", values, sizeof(values))); CHECK(!c.plan->set_inputs("y", values, sizeof(values)));
            CHECK(!c.plan->submit()); CHECK(take(c.plan->collect_completion()).success); CHECK(!c.plan->end_observation());
        }
        float retained[4]{}; ggml_backend_tensor_get(state->tensor(), retained, 0, sizeof(retained));
        CHECK(std::memcmp(values, retained, sizeof(values)) == 0);
        const auto current = budget.current(); state.reset(); CHECK(budget.current() == current);
        CHECK(!plan->set_inputs("x", values, sizeof(values)));
        auto replaced = plan->submit(); CHECK(replaced && replaced->category == ErrorCategory::conflict);
        CHECK(!plan->release_plan()); graph.reset();
    }
    CHECK(budget.current() == 0);
    {
        auto graph = take(create_graph(budget, b, Phase::decode));
        auto* q = take(graph->input("q", {GGML_TYPE_F32, {64, 1, 1}}));
        auto* k = take(graph->input("k", {GGML_TYPE_F16, {64, 8, 1}}));
        auto* v = take(graph->input("v", {GGML_TYPE_F16, {64, 8, 1}}));
        auto* view = ggml_view_3d(graph->context(), k, 64, 8, 1, k->nb[1], k->nb[2], 0);
        auto* wrong = ggml_cont(graph->context(), view);
        auto unsupported = graph->bounded_view(view, 1, Strides::preserve, wrong);
        CHECK(unsupported && unsupported->category == ErrorCategory::unsupported);
        auto* attention = ggml_flash_attn_ext(graph->context(), q, view, v, nullptr, 0.125f, 0, 0);
        CHECK(!graph->bounded_view(view, 1, Strides::preserve, attention));
        int uses = 0;
        CHECK(!graph->bind_weights(k, b, std::make_shared<CounterUse>(uses)));
        CHECK(!graph->output(attention)); auto plan = take(build_plan(graph));
        const auto occupied = budget.current();
        float query[64]{}; ggml_fp16_t cache[512]{};
        CHECK(!plan->set_inputs("q", query, sizeof(query))); CHECK(!plan->set_inputs("k", cache, sizeof(cache))); CHECK(!plan->set_inputs("v", cache, sizeof(cache)));
        auto missing = plan->submit(); CHECK(missing && missing->category == ErrorCategory::invalid_argument);
        CHECK(uses == 0 && budget.current() == occupied);
        CHECK(!plan->bind_weights(k, std::make_shared<CounterUse>(uses)));
        CHECK(check_plan_reuse(*plan, 1, {{view, 8}}).decision == ReuseDecision::reuse);
        CHECK(check_plan_reuse(*plan, 1, {{view, 9}}).reason == "range_out_of_bounds");
        CHECK(view->ne[1] == 8 && budget.current() == occupied && uses == 1);
        auto range = plan->set_valid_ranges(view, 9);
        CHECK(range && range->category == ErrorCategory::invalid_argument);
        CHECK(uses == 0 && budget.current() == occupied);
    }
    CHECK(budget.current() == 0);
    {
        auto broken = backend(); int uses = 0;
        auto state = take(allocate_state_storage(budget, broken, {GGML_TYPE_F32, {4}}, edge0::exec::Purpose::session_state));
        float values[4]{1, 2, 3, 4}; ggml_backend_tensor_set(state->tensor(), values, 0, sizeof(values));
        auto graph = take(create_graph(budget, broken, Phase::decode));
        CHECK(!graph->bind_weights(state->tensor(), broken, std::make_shared<CounterUse>(uses)));
        auto* output = ggml_scale(graph->context(), state->tensor(), 2); CHECK(!graph->output(output));
        auto plan = take(build_plan(graph));
        broken->confirm = [](ggml_backend_t handle, ggml_cgraph* empty) {
            CHECK(platform::confirm_completion(handle, empty)); return false;
        };
        CHECK(!plan->submit());
        auto shared_use = std::make_shared<CounterUse>(uses);
        CHECK(plan->bind_weights(state->tensor(), shared_use));
        CHECK(uses == 1);
        auto completion = take(plan->collect_completion());
        CHECK(!completion.success && completion.writes_may_be_partial && uses == 1);
        CHECK(plan->read_output(output, values, 0, sizeof(values)));
        CHECK(!plan->end_observation() && uses == 0);
        CHECK(check_plan_reuse(*plan, 1, {}).decision == ReuseDecision::unsupported);
        auto refused = plan->submit(); CHECK(refused && refused->category == ErrorCategory::backend_error);
    }
    CHECK(budget.current() == 0);
    {
        int uses = 0;
        auto state = take(allocate_state_storage(budget, b, {GGML_TYPE_F32, {4}}, edge0::exec::Purpose::session_state));
        auto graph = take(create_graph(budget, b, Phase::decode));
        auto use = std::make_shared<CounterUse>(uses);
        CHECK(!graph->bind_weights(state->tensor(), b, use));
        auto* input = take(graph->input("required", {GGML_TYPE_F32, {4}}));
        CHECK(!graph->output(ggml_add(graph->context(), state->tensor(), input)));
        auto plan = take(build_plan(graph));
        CHECK(plan->submit()); CHECK(uses == 0);
        auto fresh = std::make_shared<CounterUse>(uses);
        CHECK(!plan->bind_weights(state->tensor(), fresh));
        float values[4]{}; ggml_backend_tensor_set(state->tensor(), values, 0, sizeof(values));
        CHECK(!plan->set_inputs("required", values, sizeof(values))); CHECK(!plan->submit());
        CHECK(plan->bind_weights(state->tensor(), fresh)); CHECK(uses == 1 && fresh->valid());
        CHECK(graph->bind_weights(state->tensor(), b, fresh)); CHECK(uses == 1 && fresh->valid());
        CHECK(take(plan->collect_completion()).success); CHECK(!plan->end_observation()); CHECK(uses == 0);
    }
    CHECK(budget.current() == 0);
    {
        auto state = take(allocate_state_storage(budget, b, {GGML_TYPE_F32, {4}}, edge0::exec::Purpose::session_state));
        auto graph = take(create_graph(budget, b, Phase::decode));
        auto* tensor = take(graph->bind_states(state));
        CHECK(!graph->output(ggml_scale(graph->context(), tensor, 2)));
        auto plan = take(build_plan(graph));
        auto replacement = take(allocate_state_storage(budget, b, {GGML_TYPE_F32, {4}}, edge0::exec::Purpose::session_state));
        state->allocation = replacement->allocation;
        CHECK(check_plan_reuse(*plan, 1, {}).reason == "state_storage_changed");
        auto changed = plan->submit(); CHECK(changed && changed->category == ErrorCategory::conflict);
        state.reset(); replacement.reset();
        CHECK(budget.current() > 0); CHECK(!plan->release_plan());
        CHECK(graph->context() == nullptr && budget.current() == 0);
    }
    {
        std::vector<std::shared_ptr<StateStorage>> states;
        auto graph = take(create_graph(budget, b, Phase::decode));
        auto* input = take(graph->input("x", {GGML_TYPE_F32, {4}}));
        for (int i = 0; i < 3; ++i) {
            states.push_back(take(allocate_state_storage(budget, b, {GGML_TYPE_F32, {4}}, edge0::exec::Purpose::session_state)));
            CHECK(!graph->retain_output(input, states.back()));
        }
        auto plan = take(build_plan(graph));
        float values[4]{1, 2, 3, 4};
        CHECK(!plan->set_inputs("x", values, sizeof(values))); CHECK(!plan->submit());
        auto completion = take(plan->collect_completion());
        CHECK(completion.success && !completion.writes_may_be_partial && completion.written.size() == 3);
        for (const auto& state : states) {
            CHECK(std::count(completion.written.begin(), completion.written.end(), state->allocation) == 1);
            float actual[4]{}; ggml_backend_tensor_get(state->tensor(), actual, 0, sizeof(actual));
            CHECK(std::memcmp(values, actual, sizeof(values)) == 0);
        }
        CHECK(!plan->end_observation());
    }
    CHECK(budget.current() == 0);
    std::cout << "execution: stages, storage accounting, binding guards, ranges, completion, observation, retention and failure sealing passed\n";
}
