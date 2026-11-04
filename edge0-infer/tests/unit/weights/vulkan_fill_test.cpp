#include "weights/storage.h"
#include "tests/support/check.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>
using namespace edge0;
using namespace edge0::weights;
template<class T> T take(const Result<T>& result) {
    if (!result) { std::cerr << result.error().message << '\n'; std::exit(1); }
    return result.value();
}
std::string vulkan_name() {
    for (const auto& device : platform::devices())
        if (device.backend == "vulkan" && device.memory_total_bytes > 0) return device.name;
    return {};
}
int main() {
    const auto name = vulkan_name();
    if (name.empty()) return 77;
    auto lcg_bytes = [](std::size_t count, std::uint32_t seed) {
        std::vector<unsigned char> payload(count);
        std::uint32_t state = seed;
        for (auto& byte : payload) {
            state = state * 1103515245u + 12345u;
            byte = static_cast<unsigned char>(state >> 16);
        }
        return payload;
    };
    auto read_bytes = [](ggml_tensor* tensor, std::size_t count) {
        std::vector<unsigned char> bytes(count);
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        return bytes;
    };
    auto created = platform::BackendFacility::create(name);
    if (!created.first) return 77;
    auto facility = std::make_shared<platform::BackendFacility>(std::move(created.first));
    auto backend = take(exec::resolve_stages(name, name, facility)).decode;
    const auto directory = std::filesystem::temp_directory_path() /
        ("moe-vulkan-fill-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{directory};
    std::vector<unsigned char> file_bytes(80);
    for (std::size_t i = 0; i < file_bytes.size(); ++i) file_bytes[i] = static_cast<unsigned char>(i + 3);
    const auto path = directory / "layer.bin";
    { std::ofstream out(path, std::ios::binary); out.write(reinterpret_cast<const char*>(file_bytes.data()), static_cast<std::streamsize>(file_bytes.size())); }
    const Array array{{GGML_TYPE_I8, {16}}, 5};
    auto item = [&](std::size_t region) {
        return LoadItem{region, "v" + std::to_string(region), {path.string(), region * 16, 16, {GGML_TYPE_I8, {16}}}};
    };
    auto paint = [](ggml_tensor* tensor) {
        std::vector<unsigned char> bytes(16, 0xA5);
        ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
    };
    auto readback = [](ggml_tensor* tensor) {
        std::vector<unsigned char> bytes(16);
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        return bytes;
    };
    const std::vector<unsigned char> untouched(16, 0xA5);
    exec::Budget budget(64 * 1024 * 1024);
    Weights weights(budget);
    {
        auto storage = take(weights.create_storage(backend, {array}, edge0::exec::Purpose::resident_weights));
        for (std::size_t region = 0; region < 5; ++region) paint(storage->tensor(region));
        auto write = take(weights.begin_write(storage, {item(1)}));
        CHECK(!weights.fill_regions(write));
        CHECK(take(weights.complete(write)).size() == 1);
        CHECK(readback(storage->tensor(0)) == untouched);
        CHECK(readback(storage->tensor(1)) == std::vector<unsigned char>(file_bytes.begin() + 16, file_bytes.begin() + 32));
        CHECK(readback(storage->tensor(2)) == untouched);
        CHECK(!weights.close(storage));
    }
    {
        platform::file_testing::Hooks hooks;
        struct HookScope {
            platform::file_testing::Hooks* hooks;
            HookScope(platform::file_testing::Hooks* value) : hooks(value) { platform::file_testing::hooks = hooks; }
            ~HookScope() { platform::file_testing::hooks = nullptr; }
        } scope{&hooks};
        auto storage = take(weights.create_storage(backend, {array}, edge0::exec::Purpose::resident_weights));
        for (std::size_t region = 0; region < 5; ++region) paint(storage->tensor(region));
        auto write = take(weights.begin_write(storage, {item(0), item(1), item(2), item(4)}));
        CHECK(!weights.fill_regions(write));
        CHECK(take(weights.complete(write)).size() == 4);
        CHECK(hooks.vector_calls.load() == 2);
        for (std::size_t region : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{4}})
            CHECK(readback(storage->tensor(region)) == std::vector<unsigned char>(file_bytes.begin() + region * 16, file_bytes.begin() + region * 16 + 16));
        CHECK(readback(storage->tensor(3)) == untouched);
        CHECK(!weights.close(storage));
    }
    {
        // One item larger than a read task: three staged uploads at tensor offsets 0, 4 MiB and 8 MiB.
        // Bytes come from a linear congruential generator so a wrong offset cannot alias every 4 MiB.
        constexpr std::size_t big = 10 * 1024 * 1024;
        const auto payload = lcg_bytes(big, 12345);
        const auto big_path = directory / "big.bin";
        { std::ofstream out(big_path, std::ios::binary); out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(big)); }
        exec::Budget big_budget(64 * 1024 * 1024);
        Weights big_weights(big_budget);
        const Array big_array{{GGML_TYPE_I8, {static_cast<std::int64_t>(big)}}, 2};
        auto storage = take(big_weights.create_storage(backend, {big_array}, edge0::exec::Purpose::resident_weights));
        const std::vector<unsigned char> painted(big, 0xA5);
        for (std::size_t region = 0; region < 2; ++region)
            ggml_backend_tensor_set(storage->tensor(region), painted.data(), 0, painted.size());
        platform::file_testing::Hooks hooks;
        platform::file_testing::hooks = &hooks;
        auto write = take(big_weights.begin_write(storage, {LoadItem{1, "big", {big_path.string(), 0, big, big_array.layout}}}));
        CHECK(!big_weights.fill_regions(write));
        CHECK(take(big_weights.complete(write)).size() == 1);
        CHECK(hooks.vector_calls.load() == 3);
        platform::file_testing::hooks = nullptr;
        CHECK(std::memcmp(read_bytes(storage->tensor(1), big).data(), payload.data(), big) == 0);
        CHECK(read_bytes(storage->tensor(0), big) == painted);
        CHECK(!big_weights.close(storage));
    }
    {
        // Three adjacent small items merge into one read. The neighbor region keeps its prior bytes.
        constexpr std::size_t piece = 16;
        const auto payload = lcg_bytes(piece * 3, 99);
        const auto merge_path = directory / "merge.bin";
        { std::ofstream out(merge_path, std::ios::binary); out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size())); }
        exec::Budget merge_budget(32 * 1024 * 1024);
        Weights merge_weights(merge_budget);
        const Array merge_array{{GGML_TYPE_I8, {static_cast<std::int64_t>(piece)}}, 4};
        auto storage = take(merge_weights.create_storage(backend, {merge_array}, edge0::exec::Purpose::resident_weights));
        const std::vector<unsigned char> painted(piece, 0xA5);
        for (std::size_t region = 0; region < 4; ++region)
            ggml_backend_tensor_set(storage->tensor(region), painted.data(), 0, painted.size());
        std::vector<LoadItem> items;
        for (std::size_t region = 0; region < 3; ++region)
            items.push_back({region, "m" + std::to_string(region), {merge_path.string(), region * piece, piece, merge_array.layout}});
        platform::file_testing::Hooks hooks;
        platform::file_testing::hooks = &hooks;
        auto write = take(merge_weights.begin_write(storage, items));
        CHECK(!merge_weights.fill_regions(write));
        CHECK(take(merge_weights.complete(write)).size() == 3);
        CHECK(hooks.vector_calls.load() == 1);
        platform::file_testing::hooks = nullptr;
        for (std::size_t region = 0; region < 3; ++region)
            CHECK(std::memcmp(read_bytes(storage->tensor(region), piece).data(), payload.data() + region * piece, piece) == 0);
        CHECK(read_bytes(storage->tensor(3), piece) == painted);
        CHECK(!merge_weights.close(storage));
    }
    std::cout << "weights: Vulkan partial and merged fills match the file\n";
}
