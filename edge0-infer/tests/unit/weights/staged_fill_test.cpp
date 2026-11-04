#include "weights/storage.h"
#include "tests/support/check.h"
#include <chrono>
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
int main() {
    auto created = platform::BackendFacility::create("CPU");
    CHECK(created.first);
    auto facility = std::make_shared<platform::BackendFacility>(std::move(created.first));
    auto backend = take(exec::resolve_stages("CPU", "CPU", facility)).decode;
    const auto directory = std::filesystem::temp_directory_path() /
        ("moe-staged-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{directory};
    std::vector<unsigned char> file_bytes(48);
    for (std::size_t i = 0; i < file_bytes.size(); ++i) file_bytes[i] = static_cast<unsigned char>(i + 1);
    const auto path = directory / "slice.bin";
    { std::ofstream out(path, std::ios::binary); out.write(reinterpret_cast<const char*>(file_bytes.data()), static_cast<std::streamsize>(file_bytes.size())); }
    const Array array{{GGML_TYPE_I8, {16}}, 3};
    auto item = [&](std::size_t region) {
        return LoadItem{region, "r" + std::to_string(region), {path.string(), region * 16, 16, {GGML_TYPE_I8, {16}}}};
    };
    {
        exec::Budget budget(32 * 1024 * 1024);
        Weights weights(budget);
        auto storage = take(weights.create_storage(backend, {array}, edge0::exec::Purpose::resident_weights));
        auto write = take(weights.begin_write(storage, {item(0), item(1), item(2)}));
        CHECK(!weights.fill_regions(write));
        CHECK(take(weights.complete(write)).size() == 3);
        CHECK(budget.current(edge0::exec::Purpose::staging) == 0);
        for (std::size_t region = 0; region < 3; ++region)
            CHECK(std::memcmp(storage->tensor(region)->data, file_bytes.data() + region * 16, 16) == 0);
        CHECK(!weights.close(storage));
    }
    {
        exec::Budget budget(32 * 1024 * 1024);
        Weights weights(budget);
        weights.direct = [](ggml_backend_t, ggml_backend_buffer_t) { return false; };
        auto storage = take(weights.create_storage(backend, {array}, edge0::exec::Purpose::resident_weights));
        for (std::size_t region = 0; region < 3; ++region)
            std::memset(storage->tensor(region)->data, 0xA5, 16);
        auto write = take(weights.begin_write(storage, {item(1)}));
        CHECK(!weights.fill_regions(write));
        CHECK(take(weights.complete(write)).size() == 1);
        CHECK(std::memcmp(storage->tensor(0)->data, std::vector<unsigned char>(16, 0xA5).data(), 16) == 0);
        CHECK(std::memcmp(storage->tensor(1)->data, file_bytes.data() + 16, 16) == 0);
        CHECK(std::memcmp(storage->tensor(2)->data, std::vector<unsigned char>(16, 0xA5).data(), 16) == 0);
        CHECK(budget.current(edge0::exec::Purpose::staging) == 16 * 1024 * 1024);
        auto writer = weights.writable_regions(take(weights.begin_write(storage, {item(0)})), 0, 0, 16);
        CHECK(!writer && writer.error().category == ErrorCategory::unsupported);
        CHECK(!weights.close(storage));
    }
    std::size_t storage_bytes = 0;
    {
        exec::Budget budget(32 * 1024 * 1024);
        Weights weights(budget, 4);
        auto storage = take(weights.create_storage(backend, {array}, edge0::exec::Purpose::resident_weights));
        storage_bytes = budget.current();
        CHECK(!weights.close(storage));
    }
    {
        exec::Budget budget(storage_bytes + 16 * 1024 * 1024 - 1);
        Weights weights(budget, 4);
        weights.direct = [](ggml_backend_t, ggml_backend_buffer_t) { return false; };
        auto storage = take(weights.create_storage(backend, {array}, edge0::exec::Purpose::resident_weights));
        const auto occupied = budget.current();
        auto write = take(weights.begin_write(storage, {item(0)}));
        auto rejected = weights.fill_regions(write);
        CHECK(rejected && rejected->category == ErrorCategory::budget_exceeded);
        CHECK(weights.read_count() == 0 && budget.current() == occupied && budget.current(edge0::exec::Purpose::staging) == 0);
        CHECK(!weights.close(storage));
    }
    {
        // One item larger than a read task: three staged uploads at tensor offsets 0, 4 MiB and 8 MiB.
        constexpr std::size_t big = 10 * 1024 * 1024;
        std::vector<unsigned char> payload(big);
        std::uint32_t state = 12345;
        for (auto& byte : payload) { state = state * 1103515245u + 12345u; byte = static_cast<unsigned char>(state >> 16); }
        const auto big_path = directory / "big.bin";
        { std::ofstream out(big_path, std::ios::binary); out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(big)); }
        exec::Budget budget(64 * 1024 * 1024);
        Weights weights(budget);
        weights.direct = [](ggml_backend_t, ggml_backend_buffer_t) { return false; };
        auto storage = take(weights.create_storage(backend, {{{GGML_TYPE_I8, {static_cast<std::int64_t>(big)}}, 2}}, edge0::exec::Purpose::resident_weights));
        for (std::size_t region = 0; region < 2; ++region) std::memset(storage->tensor(region)->data, 0xA5, big);
        auto write = take(weights.begin_write(storage, {LoadItem{1, "big", {big_path.string(), 0, big, {GGML_TYPE_I8, {static_cast<std::int64_t>(big)}}}}}));
        CHECK(!weights.fill_regions(write));
        CHECK(take(weights.complete(write)).size() == 1);
        CHECK(std::memcmp(storage->tensor(1)->data, payload.data(), big) == 0);
        CHECK(std::memcmp(storage->tensor(0)->data, std::vector<unsigned char>(big, 0xA5).data(), big) == 0);
        CHECK(!weights.close(storage));
    }
    {
        // Adjacent items merge into one read task; each segment lands in its own region.
        exec::Budget budget(32 * 1024 * 1024);
        Weights weights(budget);
        weights.direct = [](ggml_backend_t, ggml_backend_buffer_t) { return false; };
        auto storage = take(weights.create_storage(backend, {array}, edge0::exec::Purpose::resident_weights));
        auto write = take(weights.begin_write(storage, {item(0), item(1), item(2)}));
        CHECK(!weights.fill_regions(write));
        CHECK(take(weights.complete(write)).size() == 3);
        for (std::size_t region = 0; region < 3; ++region)
            CHECK(std::memcmp(storage->tensor(region)->data, file_bytes.data() + region * 16, 16) == 0);
        CHECK(!weights.close(storage));
    }
    {
        // A failed staged read is reported and nothing is published.
        platform::file_testing::Hooks hooks; hooks.fail_read_call = 1;
        platform::file_testing::hooks = &hooks;
        exec::Budget budget(32 * 1024 * 1024);
        {
            Weights weights(budget);
            weights.direct = [](ggml_backend_t, ggml_backend_buffer_t) { return false; };
            auto storage = take(weights.create_storage(backend, {array}, edge0::exec::Purpose::resident_weights));
            auto write = take(weights.begin_write(storage, {item(0)}));
            CHECK(!weights.fill_regions(write));
            auto completed = weights.complete(write);
            CHECK(!completed && completed.error().category == ErrorCategory::backend_error);
            CHECK(storage->state(0) != RegionState::published);
            CHECK(!weights.close(storage));
        }
        platform::file_testing::hooks = nullptr;
    }
    std::cout << "weights: staged fill keeps neighbors, uploads at tensor offsets, merges segments, reports read failure, charges staging once, and refuses writers\n";
}
