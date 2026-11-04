#include "weights/storage.h"
#include "tests/support/check.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>
using namespace edge0;
using namespace edge0::weights;
template<class T> T take(const Result<T>& result) {
    if (!result) { std::cerr << result.error().message << '\n'; std::exit(1); }
    return result.value();
}
int main() {
    std::string name;
    for (const auto& device : platform::devices())
        if (device.backend == "vulkan" && device.memory_total_bytes > 0) name = device.name;
    if (name.empty()) return 77;
    auto created = platform::BackendFacility::create(name);
    if (!created.first) return 77;
    auto facility = std::make_shared<platform::BackendFacility>(std::move(created.first));
    auto backend = take(exec::resolve_stages(name, name, facility)).decode;
    const auto directory = std::filesystem::temp_directory_path() /
        ("moe-vulkan-concurrent-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{directory};
    constexpr std::size_t segment = 1024 * 1024;
    constexpr std::size_t gap = 4 * 1024;
    constexpr std::size_t stride = segment + gap;
    // Gaps keep the four regions from merging into one 4 MiB read.
    std::vector<unsigned char> file_bytes(4 * segment + 3 * gap);
    for (std::size_t i = 0; i < file_bytes.size(); ++i) file_bytes[i] = static_cast<unsigned char>(i * 17 + 9);
    const auto path = directory / "upload.bin";
    { std::ofstream out(path, std::ios::binary); out.write(reinterpret_cast<const char*>(file_bytes.data()), static_cast<std::streamsize>(file_bytes.size())); }
    exec::Budget budget(64 * 1024 * 1024);
    auto state = take(exec::allocate_state_storage(budget, backend, {GGML_TYPE_F32, {4}}, edge0::exec::Purpose::session_state));
    const float initial[4]{1.5f, -2.5f, 3.5f, -4.5f};
    ggml_backend_tensor_set(state->tensor(), initial, 0, sizeof(initial));
    auto run_scale = [&]() {
        auto graph = take(exec::create_graph(budget, backend, exec::Phase::prefill));
        auto* input = take(graph->bind_states(state));
        auto* output = ggml_scale(graph->context(), input, 2);
        CHECK(!graph->output(output));
        auto plan = take(exec::build_plan(graph));
        CHECK(!plan->submit());
        CHECK(take(plan->collect_completion()).success);
        float actual[4]{};
        CHECK(!plan->read_output(output, actual, 0, sizeof(actual)));
        CHECK(!plan->end_observation());
        return std::vector<float>(actual, actual + 4);
    };
    const auto baseline = run_scale();
    Weights weights(budget, 4);
    const Array array{{GGML_TYPE_I8, {static_cast<std::int64_t>(segment)}}, 4};
    auto storage = take(weights.create_storage(backend, {array}, edge0::exec::Purpose::resident_weights));
    std::vector<LoadItem> items;
    for (std::size_t region = 0; region < 4; ++region)
        items.push_back({region, "c" + std::to_string(region), {path.string(), region * stride, segment, array.layout}});
    std::atomic<int> hook_calls{0};
    std::atomic<int> entered{0};
    std::atomic<int> compute_started{0};
    std::atomic<int> release_last{0};
    // The last reader stays in before_read until the first compute has been collected,
    // so that completion is observed while an upload is still unfinished. The other
    // readers proceed as soon as compute has started and upload during that compute.
    weights.before_read = [&](std::size_t) {
        const int index = entered.fetch_add(1);
        hook_calls.fetch_add(1);
        while (compute_started.load() == 0) std::this_thread::yield();
        if (index == 3)
            while (release_last.load() == 0) std::this_thread::yield();
    };
    platform::file_testing::Hooks hooks;
    platform::file_testing::hooks = &hooks;
    auto write = take(weights.begin_write(storage, items));
    CHECK(!weights.fill_regions(write));
    std::vector<std::vector<float>> outputs;
    int overlapped = 0;
    compute_started.store(1);
    outputs.push_back(run_scale());
    if (!weights.ready(write)) ++overlapped;
    release_last.store(1);
    while (!weights.ready(write) || outputs.size() < 4) {
        const bool pending = !weights.ready(write);
        outputs.push_back(run_scale());
        if (pending && !weights.ready(write)) ++overlapped;
    }
    CHECK(overlapped > 0);
    CHECK(hook_calls.load() == 4);
    CHECK(take(weights.complete(write)).size() == 4);
    CHECK(hooks.vector_calls.load() == 4);
    platform::file_testing::hooks = nullptr;
    for (const auto& output : outputs) CHECK(output == baseline);
    for (std::size_t region = 0; region < 4; ++region) {
        std::vector<unsigned char> actual(segment);
        ggml_backend_tensor_get(storage->tensor(region), actual.data(), 0, actual.size());
        CHECK(std::memcmp(actual.data(), file_bytes.data() + region * stride, segment) == 0);
    }
    auto held = take(weights.acquire_views(storage, take(weights.query_state(storage, 0)).binding));
    auto rejected = weights.begin_write(storage, {items[0]});
    CHECK(!rejected && rejected.error().category == ErrorCategory::conflict);
    CHECK(!weights.release_use(held.use));
    CHECK(!weights.close(storage));
    std::cout << "weights: Vulkan upload overlapped compute and left the in-use region protected\n";
}
