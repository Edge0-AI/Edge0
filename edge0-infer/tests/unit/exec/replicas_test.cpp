#include "exec/execution.h"
#include "tests/support/check.h"
#include <cstring>
using namespace edge0;
using namespace edge0::exec;
namespace {
template<class T> T take(Result<T> result) {
    if (!result) { std::cerr << result.error().message << '\n'; std::exit(1); }
    return result.value();
}
Stages stages() {
    auto a = platform::BackendFacility::create("CPU"), b = platform::BackendFacility::create("CPU");
    CHECK(a.first && b.first);
    return take(resolve_stages("a", "b", std::map<std::string, std::shared_ptr<platform::BackendFacility>>{
        {"a", std::make_shared<platform::BackendFacility>(std::move(a.first))},
        {"b", std::make_shared<platform::BackendFacility>(std::move(b.first))}}));
}
struct Run {
    std::shared_ptr<Plan> plan;
    ggml_tensor* output;
};
Run writer(Budget& budget, const std::shared_ptr<StateStorage>& state) {
    auto graph = take(create_graph(budget, state->allocation->backend, Phase::prefill));
    auto* input = take(graph->input("x", {GGML_TYPE_F32, {4}}));
    CHECK(!graph->retain_output(input, state));
    return {take(build_plan(graph)), nullptr};
}
Run reader(Budget& budget, const std::shared_ptr<StateStorage>& state) {
    auto graph = take(create_graph(budget, state->allocation->backend, Phase::decode));
    auto* input = take(graph->bind_states(state));
    auto* output = ggml_scale(graph->context(), input, 2);
    CHECK(!graph->output(output));
    return {take(build_plan(graph)), output};
}
void start(Run& run, const std::array<float, 4>& values) {
    CHECK(!run.plan->set_inputs("x", values.data(), sizeof(values)));
    CHECK(!run.plan->submit());
}
void finish(Run& run) {
    CHECK(take(run.plan->collect_completion()).success);
    CHECK(!run.plan->end_observation());
}
std::array<float, 4> bytes(const std::shared_ptr<StateStorage>& state) {
    std::array<float, 4> data{};
    ggml_backend_tensor_get(state->tensor(), data.data(), 0, sizeof(data)); return data;
}
}
int main() {
    const auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    CHECK(cpu);
    CHECK(buffer_device(ggml_backend_cpu_buffer_type()) == ggml_backend_dev_name(cpu));
    CHECK(buffer_device(nullptr) == "host");
    const std::array<float, 4> values{1, 2, 3, 4};
    for (bool copy : {false, true}) {
        Budget budget(4 * 1024 * 1024);
        auto pair = stages();
        if (copy) pair.consumption = [](auto, auto) { return platform::Consumption::copy; };
        auto state = take(allocate_state_storage(budget, pair, {GGML_TYPE_F32, {4}}, edge0::exec::Purpose::session_state));
        auto a = take(state->on(pair.prefill)), b = take(state->on(pair.decode));
        CHECK(a->current() && b->current());
        CHECK(state->replicas->shared == !copy);
        CHECK((a->tensor()->data == b->tensor()->data) == !copy);
        CHECK(budget.current(Purpose::session_state) == ggml_backend_buffer_get_size(a->allocation->buffer) * (copy ? 2 : 1));
        CHECK(budget.snapshot().by_device.count("CPU") == 1);
        CHECK(budget.snapshot().by_device.at("CPU").current == budget.current(Purpose::session_state));
        auto write = writer(budget, a), read = reader(budget, b);
        start(write, values);
        auto busy = read.plan->submit(); CHECK(busy && busy->category == ErrorCategory::conflict);
        CHECK(transfer(state, pair.decode));
        // A different session's storage shares the same model-wide coordinator.
        auto other = take(allocate_state_storage(budget, pair, {GGML_TYPE_F32, {4}}, edge0::exec::Purpose::session_state));
        auto other_read = reader(budget, take(other->on(pair.decode)));
        CHECK(other_read.plan->submit());
        finish(write);
        CHECK(a->current() && b->current() == !copy);
        if (copy) {
            const auto unchanged = bytes(b);
            auto stale = read.plan->submit(); CHECK(stale && stale->category == ErrorCategory::conflict);
            CHECK(bytes(b) == unchanged && bytes(a) == values);
            state->replicas->stages.transfer = [](auto*, auto*, const auto&) -> Status {
                return Error{ErrorCategory::backend_error, "Injected transfer failure"};
            };
            CHECK(transfer(state, pair.decode));
            CHECK(!b->current() && a->current()); CHECK(bytes(a) == values && bytes(b) == unchanged);
            CHECK(state->replicas->stats.count == 0);
            state->replicas->stages.transfer = {};
            CHECK(!transfer(state, pair.decode, {{0, 2 * sizeof(float)}}, false));
            CHECK(!b->current());
            CHECK(!transfer(state, pair.decode, {{2 * sizeof(float), 2 * sizeof(float)}}));
            CHECK(b->current() && bytes(b) == values);
            CHECK(state->replicas->stats.bytes == sizeof(values));
            CHECK(state->replicas->stats.count == 2 && state->replicas->stats.seconds > 0);
        } else {
            CHECK(!transfer(state, pair.decode));
            CHECK(state->replicas->stats.count == 0 && state->replicas->stats.bytes == 0);
        }
        CHECK(!read.plan->submit());
        CHECK(take(read.plan->collect_completion()).success);
        std::array<float, 4> result{};
        CHECK(!read.plan->read_output(read.output, result.data(), 0, sizeof(result)));
        for (std::size_t i = 0; i < result.size(); ++i) CHECK(result[i] == 2 * values[i]);
        CHECK(!read.plan->end_observation());
        auto reverse = writer(budget, b); start(reverse, {5, 6, 7, 8}); finish(reverse);
        CHECK(!transfer(state, pair.prefill));
        CHECK(bytes(a) == bytes(b));
    }
    Budget budget(1024 * 1024);
    auto pair = stages(); pair.decode = pair.prefill; pair.runs.reset();
    auto single = take(allocate_state_storage(budget, pair, {GGML_TYPE_F32, {4}}, edge0::exec::Purpose::session_state));
    CHECK(!single->replicas && budget.current(Purpose::session_state) == ggml_backend_buffer_get_size(single->allocation->buffer));
    CHECK(take(single->on(pair.prefill)) == single);
    CHECK(!transfer(single, pair.prefill));
    std::cout << "replicas: shared ordering, stale rejection, partial and failed transfers, reverse transfer passed\n";
}
