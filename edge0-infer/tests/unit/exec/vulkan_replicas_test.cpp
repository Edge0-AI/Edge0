#include "exec/execution.h"
#include "tests/support/check.h"
#include <cstring>
#include <vector>
using namespace edge0;
using namespace edge0::exec;
template<class T> T take(Result<T> result) {
    if (!result) { std::cerr << result.error().message << '\n'; std::exit(1); }
    return result.value();
}
std::vector<unsigned char> read_tensor(ggml_tensor* tensor) {
    std::vector<unsigned char> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    return bytes;
}
int main() {
    std::string vulkan;
    for (const auto& device : platform::devices())
        if (device.backend == "vulkan" && device.memory_total_bytes > 0) vulkan = device.name;
    if (vulkan.empty()) return 77;
    auto cpu_created = platform::BackendFacility::create("CPU");
    auto vulkan_created = platform::BackendFacility::create(vulkan);
    if (!cpu_created.first || !vulkan_created.first) return 77;
    auto stages = take(resolve_stages("vulkan", "cpu", {
        {"vulkan", std::make_shared<platform::BackendFacility>(std::move(vulkan_created.first))},
        {"cpu", std::make_shared<platform::BackendFacility>(std::move(cpu_created.first))}}));
    stages.consumption = [](auto, auto) { return platform::Consumption::copy; };
    Budget budget(64 * 1024 * 1024);
    auto state = take(allocate_state_storage(budget, stages, {GGML_TYPE_I8, {64}}, edge0::exec::Purpose::session_state));
    auto gpu = take(state->on(stages.prefill)), host = take(state->on(stages.decode));
    CHECK(!state->replicas->shared);
    std::vector<unsigned char> source(64), other(64, 0x5A);
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = static_cast<unsigned char>(i + 1);
    ggml_backend_tensor_set(gpu->tensor(), source.data(), 0, source.size());
    gpu->mark_written();
    CHECK(!transfer(state, stages.decode));
    CHECK(read_tensor(host->tensor()) == source);
    CHECK(state->replicas->stats.count == 1 && state->replicas->stats.bytes == source.size());
    ggml_backend_tensor_set(gpu->tensor(), other.data(), 0, other.size());
    host->mark_written();
    const auto bytes_before = state->replicas->stats.bytes;
    const auto count_before = state->replicas->stats.count;
    CHECK(!transfer(state, stages.prefill, {{16, 16}}));
    auto moved = read_tensor(gpu->tensor());
    CHECK(std::vector<unsigned char>(moved.begin(), moved.begin() + 16) == std::vector<unsigned char>(16, 0x5A));
    CHECK(std::vector<unsigned char>(moved.begin() + 16, moved.begin() + 32) == std::vector<unsigned char>(source.begin() + 16, source.begin() + 32));
    CHECK(std::vector<unsigned char>(moved.begin() + 32, moved.end()) == std::vector<unsigned char>(32, 0x5A));
    CHECK(state->replicas->stats.count == count_before + 1 && state->replicas->stats.bytes == bytes_before + 16);
    gpu->mark_written();
    CHECK(!transfer(state, stages.decode, {{0, 8}}));
    auto back = read_tensor(host->tensor());
    CHECK(std::vector<unsigned char>(back.begin(), back.begin() + 8) == std::vector<unsigned char>(moved.begin(), moved.begin() + 8));
    CHECK(std::vector<unsigned char>(back.begin() + 8, back.end()) == std::vector<unsigned char>(source.begin() + 8, source.end()));
    CHECK(!state->reset_zero());
    const auto zero = std::vector<unsigned char>(64, 0);
    CHECK(read_tensor(gpu->tensor()) == zero && read_tensor(host->tensor()) == zero);
    auto graph = take(create_graph(budget, stages.decode, Phase::decode));
    CHECK(!graph->coordinate(stages));
    auto* input = take(graph->input("x", {GGML_TYPE_F32, {4}}));
    auto* output = ggml_scale(graph->context(), input, 2);
    CHECK(!graph->output(output));
    auto plan = take(build_plan(graph));
    const float values[4]{1, 2, 3, 4};
    CHECK(!plan->set_inputs("x", values, sizeof(values)));
    CHECK(!plan->submit());
    float early[4]{};
    CHECK(plan->read_output(output, early, 0, sizeof(early)));
    CHECK(take(plan->collect_completion()).success);
    float actual[4]{};
    CHECK(!plan->read_output(output, actual, 0, sizeof(actual)));
    CHECK(actual[0] == 2 && actual[3] == 8);
    CHECK(!plan->end_observation());
    std::cout << "exec: Vulkan and CPU replicas transferred, cleared, and refused an early read\n";
}
