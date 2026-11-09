#include "weights/storage.h"
#include "tests/support/check.h"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
using namespace edge0;
using namespace edge0::weights;
template<class T> T take(Result<T> result) {
    if (!result) { std::cerr << result.error().message << '\n'; std::exit(1); }
    return result.value();
}
int main() {
    auto a = platform::BackendFacility::create("CPU"), b = platform::BackendFacility::create("CPU");
    CHECK(a.first && b.first);
    const auto stages = take(exec::resolve_stages("a", "b", std::map<std::string, std::shared_ptr<platform::BackendFacility>>{
        {"a", std::make_shared<platform::BackendFacility>(std::move(a.first))},
        {"b", std::make_shared<platform::BackendFacility>(std::move(b.first))}}));
    exec::Budget budget(4 * 1024 * 1024);
    Weights weights(budget);
    const std::vector<std::shared_ptr<exec::Backend>> consumers{stages.prefill, stages.decode};
    const std::vector<Array> arrays{{{GGML_TYPE_F32, {4}}, 1}};
    const auto directory = std::filesystem::temp_directory_path() /
        ("moe-mixed-weights-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    CHECK(std::filesystem::create_directory(directory));
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{directory};
    const auto path = directory / "weights.bin";
    const float input[4]{1, 2, 3, 4};
    { std::ofstream file(path, std::ios::binary); file.write(reinterpret_cast<const char*>(input), sizeof(input)); }
    const LoadItem item{0, "weight", {path.string(), 0, sizeof(input), {GGML_TYPE_F32, {4}}}};
    for (auto protection : {Protection::region, Protection::group}) {
        auto storage = take(weights.create_storage(consumers, arrays, edge0::exec::Purpose::resident_weights, protection));
        CHECK(storage.size() == 1 && budget.current(exec::Purpose::resident_weights) == ggml_backend_buffer_get_size(storage.front()->allocation()->buffer));
        auto write = take(weights.begin_write(storage.front(), {item}));
        CHECK(!weights.fill_regions(write));
        const auto bindings = take(weights.complete(write));
        auto first = take(weights.acquire_views(storage.front(), bindings.front(), stages.prefill));
        auto second = take(weights.acquire_views(storage.front(), bindings.front(), stages.decode));
        CHECK(first.tensor->data == second.tensor->data);
        CHECK(first.use->matches(first.tensor, stages.prefill->handle()));
        CHECK(second.use->matches(second.tensor, stages.decode->handle()));
        CHECK(!second.use->matches(second.tensor, stages.prefill->handle()));
        CHECK(std::memcmp(first.tensor->data, input, sizeof(input)) == 0);
        CHECK(!weights.begin_write(storage.front(), {item}));
        CHECK(!weights.release_use(first.use));
        CHECK(!weights.begin_write(storage.front(), {item}));
        CHECK(!weights.release_use(second.use));
        if (protection == Protection::group) {
            auto group = take(weights.acquire_views(storage.front(), stages.decode, 0, bindings));
            CHECK(group.use->matches(group.tensor, stages.decode->handle()));
            CHECK(!weights.begin_write(storage.front(), {item}));
            CHECK(!weights.release_use(group.use));
        }
        CHECK(!weights.close(storage.front()));
    }
    {
        auto copies = take(weights.create_storage(consumers, arrays, edge0::exec::Purpose::resident_weights, Protection::region,
            [](auto, auto) { return platform::Consumption::copy; }));
        CHECK(copies.size() == 2);
        CHECK(copies[0]->tensor(0)->data != copies[1]->tensor(0)->data);
        CHECK(budget.current(exec::Purpose::resident_weights) == ggml_backend_buffer_get_size(copies[0]->allocation()->buffer) + ggml_backend_buffer_get_size(copies[1]->allocation()->buffer));
        for (auto& copy : copies) CHECK(!weights.close(copy));
    }
    const auto reads = weights.read_count();
    weights.direct = [](auto, auto) { return false; };
    auto admitted = take(weights.create_storage(consumers, arrays, edge0::exec::Purpose::resident_weights, Protection::region,
        [](auto, auto) { return platform::Consumption::copy; }));
    CHECK(admitted.size() == 2 && weights.read_count() == reads);
    for (auto& copy : admitted) CHECK(!weights.close(copy));
    std::cout << "mixed weights: shared views, group protection, independent copies, staged admission passed\n";
}
