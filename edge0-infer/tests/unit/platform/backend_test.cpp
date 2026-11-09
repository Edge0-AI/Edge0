#include "platform/backend.h"
#include "tests/support/check.h"
#include <type_traits>
#include <utility>

int main() {
    using edge0::platform::BackendFacility;
    static_assert(!std::is_copy_constructible_v<BackendFacility>);
    CHECK(!edge0::platform::compiled_backends().empty());
    CHECK(!edge0::platform::devices().empty());
    for (int i = 0; i < 2; ++i) {
        auto created = BackendFacility::create("CPU");
        CHECK(created.first);
        CHECK(created.second.empty());
        BackendFacility moved(std::move(created.first));
        CHECK(moved);
        CHECK(!created.first);
        BackendFacility assigned;
        assigned = std::move(moved);
        CHECK(assigned);
        CHECK(!moved);
    }
    auto invalid = BackendFacility::create("nonexistent-device");
    CHECK(!invalid.first);
    CHECK(invalid.second.find("does not exist or backend was not compiled") != std::string::npos);
    CHECK(edge0::platform::probe("CPU").first);
    CHECK(!edge0::platform::probe("nonexistent-device").first);
    using namespace edge0::platform;
    CHECK(consumption({true, false, false}) == Consumption::direct);
    CHECK(consumption({true, true, true}) == Consumption::direct);
    CHECK(consumption({false, true, true}) == Consumption::wrap);
    CHECK(consumption({false, false, true}) == Consumption::copy);
    CHECK(consumption({false, true, false}) == Consumption::copy);
    auto first = BackendFacility::create("CPU"), second = BackendFacility::create("CPU");
    CHECK(first.first && second.first);
    CHECK(first.first.handle() != second.first.handle());
    const auto type = ggml_backend_get_default_buffer_type(first.first.handle());
    CHECK(consumption(type, second.first.handle()) == Consumption::direct);
    const std::vector<ggml_backend_t> consumers{first.first.handle(), second.first.handle()};
    CHECK(choose_storage_home(consumers) == type);
    CHECK(choose_storage_home(consumers) == choose_storage_home(consumers));
    std::vector<ggml_backend_t> visited;
    CHECK(choose_storage_home(consumers, [&](auto home, auto consumer) {
        CHECK(home == type); visited.push_back(consumer); return Consumption::wrap;
    }) == type);
    CHECK(visited == consumers);
    CHECK(!choose_storage_home(consumers, [](auto, auto) { return Consumption::copy; }));
    CHECK(!choose_storage_home({}));
    CHECK(!choose_storage_home({nullptr}));
    CHECK(!wrap_host_buffer(nullptr, first.first.handle(), 1));

    // Default CPU thread rules are pure functions of the logical-core count, so
    // every platform's rule is checkable on any host (device-independent).
    using edge0::platform::default_cpu_threads;
    const auto generic = default_cpu_threads(12, false);
    CHECK(generic.decode == 4 && generic.prefill == 4);
    const auto android = default_cpu_threads(8, true);
    CHECK(android.decode == 6 && android.prefill == 8);
    // Windows rule: physical = logical / 2; decode = max(1, physical - 2),
    // prefill = min(logical, physical + 2). i5-12400 (logical 12, physical 6).
    const auto windows = default_cpu_threads(12, false, true);
    CHECK(windows.decode == 4 && windows.prefill == 8);
    const auto windows8 = default_cpu_threads(8, false, true);
    CHECK(windows8.decode == 2 && windows8.prefill == 6);
    const auto windows4 = default_cpu_threads(4, false, true);
    CHECK(windows4.decode == 1 && windows4.prefill == 4);

    std::cout << "platform: 2 create/release cycles, move construction/assignment, invalid device, probes, "
                 "default CPU thread rules passed\n";
}
