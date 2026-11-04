#include "edge0/engine.h"
#include "edge0/forward_check.h"
#include "edge0/generation_check.h"
#include "src/app/runtime.h"
#include "src/app/adapters.h"
#include "src/platform/backend.h"
#include "ggml.h"
#include "ggml-backend-impl.h"
#include "tests/support/check.h"
#include <algorithm>
#include <string>
#include <map>
#include <thread>

namespace {
// Instrument a registered CPU device without changing the production library.
// The facility is a real CPU backend; only initialization logging and release
// counting are injected to observe rollback deterministically in Release builds.
int released = 0;
bool log_properties = false;
void emit_log(const char* text);
std::map<ggml_backend_t, void (*)(ggml_backend_t)> original_free;
ggml_backend_dev_t cpu_device;
void tracked_free(ggml_backend_t backend) {
    auto free = original_free.at(backend);
    original_free.erase(backend);
    ++released;
    free(backend);
}
const char* tracked_name(ggml_backend_dev_t) { return "TrackedCPU"; }
void tracked_props(ggml_backend_dev_t, ggml_backend_dev_props* props) {
    if (log_properties) emit_log("Tracked CPU properties\n");
    ggml_backend_dev_get_props(cpu_device, props);
    props->name = "TrackedCPU";
    props->memory_total /= 2;
    props->memory_free = std::min(props->memory_free, props->memory_total);
}
ggml_backend_t tracked_init(ggml_backend_dev_t, const char* params) {
    ggml_log_callback callback;
    void* data;
    ggml_log_get(&callback, &data);
    callback(GGML_LOG_LEVEL_INFO, "Tracked CPU initialization\n", data);
    auto backend = ggml_backend_dev_init(cpu_device, params);
    if (backend) {
        original_free.emplace(backend, backend->iface.free);
        backend->iface.free = tracked_free;
    }
    return backend;
}
void original_log(enum ggml_log_level, const char*, void* data) {
    ++*static_cast<int*>(data);
}
void emit_log(const char* text) {
    ggml_log_callback callback;
    void* data;
    ggml_log_get(&callback, &data);
    callback(GGML_LOG_LEVEL_INFO, text, data);
}
const edge0::BackendCapability& backend(const edge0::Capabilities& caps, const std::string& name) {
    auto found = std::find_if(caps.backends.begin(), caps.backends.end(), [&](const auto& item) { return item.name == name; });
    CHECK(found != caps.backends.end());
    return *found;
}
}

int main() {
    using edge0::ErrorCategory;
    auto absent = edge0::open_runtime();
    CHECK(!absent && absent.error().category == ErrorCategory::invalid_argument);
    auto empty = edge0::open_runtime({{}, {}});
    CHECK(!empty && empty.error().category == ErrorCategory::invalid_argument);
    auto duplicate = edge0::open_runtime({{"CPU", "CPU"}, {}});
    CHECK(!duplicate && duplicate.error().category == ErrorCategory::invalid_argument);
    cpu_device = ggml_backend_dev_by_name("CPU");
    CHECK(cpu_device);
    auto tracked_device = *cpu_device;
    tracked_device.iface.get_name = tracked_name;
    tracked_device.iface.init_backend = tracked_init;
    tracked_device.iface.get_props = tracked_props;
    ggml_backend_device_register(&tracked_device);
    int original_count = 0;
    ggml_log_callback saved;
    void* saved_data;
    ggml_log_get(&saved, &saved_data);
    ggml_log_set(original_log, &original_count);
    int received = 0;
    edge0::RuntimeOptions options{{"CPU"}, [&](edge0::LogLevel level, const std::string& text) {
        CHECK(level == edge0::LogLevel::info);
        CHECK(!text.empty());
        ++received;
    }};
    auto opened = edge0::open_runtime(options);
    CHECK(opened);
    CHECK(opened.value().available_devices() == options.devices);
    emit_log("Test GGML log while runtime is open");
    CHECK(received == 1);
    auto conflict = edge0::open_runtime(options);
    CHECK(!conflict && conflict.error().category == ErrorCategory::conflict);
    auto ended = edge0::finish_runtime_use(opened.value());
    CHECK(ended && ended.value() == "Safely ended");
    ggml_log_callback restored;
    void* restored_data;
    ggml_log_get(&restored, &restored_data);
    CHECK(restored == original_log && restored_data == &original_count);
    emit_log("After finish");
    CHECK(received == 1 && original_count == 1);
    auto repeated = edge0::finish_runtime_use(opened.value());
    CHECK(!repeated && repeated.error().category == ErrorCategory::runtime_ended);
    const int before_release = released;
    auto logged = edge0::open_runtime({{"TrackedCPU"}, options.log_receiver});
    CHECK(logged && received == 2);
    CHECK(edge0::finish_runtime_use(logged.value()));
    CHECK(released == before_release + 1);
    auto rollback = edge0::open_runtime({{"TrackedCPU", "missing"}, options.log_receiver});
    CHECK(!rollback && rollback.error().category == ErrorCategory::backend_error);
    CHECK(released == before_release + 2 && original_free.empty());
    const std::string nul_name("CPU\0not-a-device", 16);
    for (const auto& selection : {std::vector<std::string>{"missing"}, std::vector<std::string>{"CPU", "missing"},
                                 std::vector<std::string>{nul_name}, std::vector<std::string>{"CPU", nul_name}}) {
        auto invalid = edge0::open_runtime({selection, options.log_receiver});
        CHECK(!invalid && invalid.error().category == ErrorCategory::backend_error);
        CHECK(invalid.error().message.find("does not exist or backend was not compiled") != std::string::npos);
        ggml_log_get(&restored, &restored_data);
        CHECK(restored == original_log && restored_data == &original_count);
        auto retry = edge0::open_runtime({{"CPU"}, {}});
        CHECK(retry);
        CHECK(!edge0::finish_runtime_use(opened.value()));
        CHECK(edge0::finish_runtime_use(retry.value()));
    }
    ggml_log_set(saved, saved_data);
    const auto caps = edge0::query_capabilities();
    CHECK(backend(caps, "cpu").compiled == edge0::TriState::yes);
    CHECK(backend(caps, "cpu").initializable == edge0::TriState::unknown);
    CHECK(backend(caps, "cpu").verified == edge0::TriState::unknown);
#if !MOE_TEST_METAL
    CHECK(backend(caps, "metal").compiled == edge0::TriState::no);
#endif
    CHECK(caps.ggml_version == MOE_TEST_GGML_VERSION);
    CHECK(caps.ggml_version == ggml_version());
    std::vector<std::string> registered;
    for (const auto& entry : edge0::app::compiled_models()) registered.push_back(entry.adapter->id());
    CHECK(!registered.empty() && caps.models == registered);
    CHECK(std::find(caps.models.begin(), caps.models.end(), "edge0_35b") != caps.models.end());
    const auto probed = edge0::query_capabilities(true);
    CHECK(backend(probed, "cpu").initializable == edge0::TriState::yes);
    CHECK(backend(probed, "cpu").verified == edge0::TriState::unknown);
    std::size_t expected_budget = 0;
    for (const auto& device : caps.devices)
        if (device.name == "CPU") expected_budget = device.memory_total_bytes;
    CHECK(expected_budget > 0);
    auto default_runtime = edge0::open_runtime({{"CPU"}, {}});
    CHECK(default_runtime);
    const auto threads = edge0::query_runtime_cpu_threads(default_runtime.value());
    // The platform default is {4, 4} on macOS but core-count dependent on Windows
    // (physical - 2 decode, min(logical, physical + 2) prefill), so compare against
    // the platform rule rather than literals to keep this portable.
    const auto platform_default = edge0::platform::default_cpu_threads();
    CHECK(threads && threads.value().cpu_threads == platform_default.decode
          && threads.value().cpu_prefill_threads == platform_default.prefill);
    CHECK(threads.value().cpu_threads_source == edge0::CpuThreadSource::platform_default);
    {
        auto r = edge0::RuntimeAccess::acquire(default_runtime.value()); CHECK(r);
        edge0::RuntimeActivity active_call(r.value().registry); CHECK(!active_call.enter());
        edge0::ForwardCheckOptions forward; forward.input = {1};
        auto rejected = edge0::forward_check(default_runtime.value(), forward);
        CHECK(!rejected && rejected.error().category == ErrorCategory::busy);
    }
    for (unsigned count : {1u, 2u, 4u, 8u}) {
        auto defaults = edge0::platform::default_cpu_threads(count, true);
        CHECK(defaults.decode == static_cast<int>(count > 2 ? count - 2 : 1));
        CHECK(defaults.prefill == static_cast<int>(count));
    }
    auto default_budget = edge0::query_runtime_budget(default_runtime.value());
    CHECK(default_budget && default_budget.value().limit_bytes == expected_budget);
    CHECK(default_budget.value().source == edge0::BudgetSource::device_default);
    {
        auto resources = edge0::RuntimeAccess::acquire(default_runtime.value());
        CHECK(resources);
        auto busy = edge0::finish_runtime_use(default_runtime.value());
        CHECK(!busy && busy.error().category == edge0::ErrorCategory::conflict);
        CHECK(edge0::query_runtime_budget(default_runtime.value()));
        CHECK(!edge0::open_runtime({{"CPU"}, {}}));
    }
    CHECK(edge0::finish_runtime_use(default_runtime.value()));
    CHECK(!edge0::query_runtime_budget(default_runtime.value()));
    for (std::size_t invalid : {std::size_t(0), expected_budget + 1}) {
        auto rejected = edge0::open_runtime({{"CPU"}, {}, invalid});
        CHECK(!rejected && rejected.error().category == ErrorCategory::invalid_argument);
    }
    for (bool prefill : {false, true}) {
        for (int value : {0, static_cast<int>(std::thread::hardware_concurrency()) + 1}) {
            edge0::RuntimeOptions invalid_threads{{"CPU"}, {}};
            (prefill ? invalid_threads.cpu_prefill_threads : invalid_threads.cpu_threads) = value;
            auto rejected = edge0::open_runtime(invalid_threads);
            CHECK(!rejected && rejected.error().category == ErrorCategory::invalid_argument);
        }
        edge0::RuntimeOptions explicit_threads{{"CPU"}, {}};
        (prefill ? explicit_threads.cpu_prefill_threads : explicit_threads.cpu_threads) = 1;
        auto rt = edge0::open_runtime(explicit_threads); CHECK(rt);
        const auto actual = edge0::query_runtime_cpu_threads(rt.value()); CHECK(actual);
        CHECK(actual.value().cpu_threads == (prefill ? platform_default.decode : 1));
        CHECK(actual.value().cpu_prefill_threads == (prefill ? 1 : platform_default.prefill));
        CHECK(actual.value().cpu_threads_source == (prefill ? edge0::CpuThreadSource::platform_default : edge0::CpuThreadSource::explicit_value));
        CHECK(actual.value().cpu_prefill_threads_source == (prefill ? edge0::CpuThreadSource::explicit_value : edge0::CpuThreadSource::platform_default));
        {
            auto r = edge0::RuntimeAccess::acquire(rt.value()); CHECK(r);
            edge0::select_cpu_threads(r.value(), edge0::exec::Phase::prefill);
            CHECK(r.value().backends.at("CPU")->facility->cpu_use() == edge0::platform::CpuUse::multi_token);
            CHECK(edge0::measure_submission_overhead(rt.value(), "CPU", 1));
            CHECK(r.value().backends.at("CPU")->facility->cpu_use() == edge0::platform::CpuUse::single_token);
        }
        CHECK(edge0::finish_runtime_use(rt.value()));
        CHECK(!edge0::query_runtime_cpu_threads(rt.value()));
        explicit_threads.devices = {"MTL0"};
        auto no_cpu = edge0::open_runtime(explicit_threads);
        CHECK(!no_cpu && no_cpu.error().category == ErrorCategory::invalid_argument);
    }
    auto combined_runtime = edge0::open_runtime({{"CPU", "TrackedCPU"}, {}});
    CHECK(combined_runtime);
    auto combined_budget = edge0::query_runtime_budget(combined_runtime.value());
    CHECK(combined_budget && combined_budget.value().limit_bytes == expected_budget / 2);
    CHECK(combined_budget.value().source == edge0::BudgetSource::device_default);
    CHECK(edge0::finish_runtime_use(combined_runtime.value()));
    auto explicit_runtime = edge0::open_runtime({{"CPU"}, {}, expected_budget / 2});
    CHECK(explicit_runtime);
    auto explicit_budget = edge0::query_runtime_budget(explicit_runtime.value());
    CHECK(explicit_budget && explicit_budget.value().limit_bytes == expected_budget / 2);
    CHECK(explicit_budget.value().source == edge0::BudgetSource::explicit_value);
    CHECK(edge0::finish_runtime_use(explicit_runtime.value()));
    // Emit deterministically from device enumeration, not a synthetic query hook.
    ggml_log_set(original_log, &original_count);
    log_properties = true;
    const int prior_count = original_count;
    int query_count = 0;
    auto query_receiver = [&](edge0::LogLevel level, const std::string& text) {
        CHECK(level == edge0::LogLevel::info && text == "Tracked CPU properties\n");
        ++query_count;
    };
    auto check_restored = [&] {
        ggml_log_get(&restored, &restored_data);
        CHECK(restored == original_log && restored_data == &original_count);
    };
    edge0::query_capabilities(false, query_receiver);
    CHECK(query_count == 1 && original_count == prior_count);
    check_restored();
    edge0::query_capabilities();
    CHECK(query_count == 1 && original_count == prior_count + 1);
    check_restored();
    int first_count = 0, second_count = 0;
    std::thread first([&] { edge0::query_capabilities(false, [&](edge0::LogLevel, const std::string&) { ++first_count; }); });
    std::thread second([&] { edge0::query_capabilities(false, [&](edge0::LogLevel, const std::string&) { ++second_count; }); });
    first.join(); second.join();
    CHECK(first_count == 1 && second_count == 1);
    check_restored();
    auto active = edge0::open_runtime(options);
    CHECK(active);
    ggml_log_callback active_callback;
    void* active_data;
    ggml_log_get(&active_callback, &active_data);
    const int runtime_count = received;
    edge0::query_capabilities(false, query_receiver);
    CHECK(received == runtime_count + 1 && query_count == 1);
    ggml_log_get(&restored, &restored_data);
    CHECK(restored == active_callback && restored_data == active_data);
    CHECK(edge0::finish_runtime_use(active.value()));
    check_restored();
    auto no_receiver = edge0::open_runtime({{"CPU"}, {}});
    CHECK(no_receiver);
    const int before_empty_query = original_count;
    edge0::query_capabilities(false, query_receiver);
    CHECK(original_count == before_empty_query + 1 && query_count == 1);
    check_restored();
    CHECK(edge0::finish_runtime_use(no_receiver.value()));
    log_properties = false;
    ggml_log_set(saved, saved_data);
    std::cout << "app: runtime lifecycle, rollback, stale references, log forwarding/restoration and capabilities passed\n";
}
