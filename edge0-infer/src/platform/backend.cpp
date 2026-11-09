#include "platform/backend.h"
#include <algorithm>
#include <cctype>
#include <thread>

namespace edge0::platform {
namespace {
std::string normalize(const char* name) {
    std::string value(name);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value == "mtl" ? "metal" : value;
}
}

std::vector<std::string> compiled_backends() {
    std::vector<std::string> result;
    for (std::size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        result.push_back(normalize(ggml_backend_reg_name(ggml_backend_reg_get(i))));
    }
    return result;
}

std::vector<Device> devices() {
    std::vector<Device> result;
    for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const auto device = ggml_backend_dev_get(i);
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(device, &props);
        result.push_back({normalize(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device))),
            props.name, props.description, props.type, props.memory_total, props.memory_free});
    }
    return result;
}

CpuThreads default_cpu_threads(unsigned logical_cpus, bool android, bool windows) {
    if (android) {
        // Tuned on PKR110; asymmetric devices may need explicit thread counts.
        const int count = static_cast<int>(std::max(1u, std::min(logical_cpus, 512u)));
        return {std::max(1, count - 2), count};
    }
    if (windows) {
        // Desktop x86 Windows hyperthreads (logical cores are about twice the
        // physical count). Decode is bandwidth/IO-bound and extra hyperthreads
        // contend with the expert read threads, so decode stays below the
        // physical-core count; prefill is compute-parallel and gains from a
        // couple of extra lanes. Values measured on a 12-logical/6-physical part:
        // decode peaks below the physical count, prefill above it.
        // Measured only on an Intel Core i5-12400 (6 cores, 12 threads, no
        // efficiency cores) at 4, 6 and 8 threads. The logical / 2 estimate is
        // wrong without SMT and on hybrid parts; set the thread counts explicitly
        // there.
        const int logical = static_cast<int>(std::max(1u, std::min(logical_cpus, 512u)));
        const int physical = std::max(1, logical / 2);
        return {std::max(1, physical - 2), std::min(logical, physical + 2)};
    }
    return {4, 4};
}
CpuThreads default_cpu_threads() {
#if defined(__ANDROID__)
    return default_cpu_threads(std::thread::hardware_concurrency(), true);
#elif defined(_WIN32)
    return default_cpu_threads(std::thread::hardware_concurrency(), false, true);
#else
    return default_cpu_threads(std::thread::hardware_concurrency(), false);
#endif
}
void BackendFacility::release() {
    if (backend_) ggml_backend_free(backend_);
    backend_ = nullptr;
    if (prefill_pool_ && prefill_pool_ != decode_pool_) free_pool_(prefill_pool_);
    if (decode_pool_) free_pool_(decode_pool_);
    decode_pool_ = prefill_pool_ = nullptr;
}
void BackendFacility::swap(BackendFacility& other) noexcept {
    std::swap(backend_, other.backend_);
    std::swap(decode_pool_, other.decode_pool_); std::swap(prefill_pool_, other.prefill_pool_);
    std::swap(free_pool_, other.free_pool_); std::swap(set_pool_, other.set_pool_);
    std::swap(set_threads_, other.set_threads_); std::swap(threads_, other.threads_); std::swap(use_, other.use_);
}
BackendFacility::~BackendFacility() { release(); }
BackendFacility::BackendFacility(BackendFacility&& other) noexcept { swap(other); }
BackendFacility& BackendFacility::operator=(BackendFacility&& other) noexcept {
    if (this != &other) { release(); swap(other); }
    return *this;
}
void BackendFacility::select_cpu_use(CpuUse use) {
    if (!set_pool_) return;
    use_ = use;
    set_pool_(backend_, use == CpuUse::single_token ? decode_pool_ : prefill_pool_);
    set_threads_(backend_, cpu_threads());
}
std::pair<BackendFacility, std::string> BackendFacility::create(const std::string& name, CpuThreads threads) {
    const auto device = ggml_backend_dev_by_name(name.c_str());
    if (!device || name != ggml_backend_dev_name(device))
        return {BackendFacility{}, "Device does not exist or backend was not compiled: " + name};
    const auto backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) return {BackendFacility{}, "Backend initialization returned null for device: " + name};
    BackendFacility result{backend};
    if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        const auto reg = ggml_backend_dev_backend_reg(device);
        auto address = [&](const char* symbol) { return ggml_backend_reg_get_proc_address(reg, symbol); };
        auto make_pool = reinterpret_cast<ggml_threadpool_t (*)(ggml_threadpool_params*)>(address("ggml_threadpool_new"));
        result.free_pool_ = reinterpret_cast<void (*)(ggml_threadpool_t)>(address("ggml_threadpool_free"));
        result.set_pool_ = reinterpret_cast<void (*)(ggml_backend_t, ggml_threadpool_t)>(address("ggml_backend_cpu_set_threadpool"));
        result.set_threads_ = reinterpret_cast<ggml_backend_set_n_threads_t>(address("ggml_backend_set_n_threads"));
        if (!make_pool || !result.free_pool_ || !result.set_pool_ || !result.set_threads_)
            return {BackendFacility{}, "CPU threadpool registry symbols unavailable: " + name};
        if (threads.decode < 1 || threads.prefill < 1 || threads.decode > GGML_MAX_N_THREADS || threads.prefill > GGML_MAX_N_THREADS)
            return {BackendFacility{}, "CPU thread count exceeds threadpool capacity"};
        result.threads_ = threads;
        // Keep default poll=50: sleeping reduces decode throughput. Do not set a
        // CPU mask: GGML would permanently pin the calling thread as well.
        auto params = ggml_threadpool_params_default(threads.decode);
        result.decode_pool_ = make_pool(&params);
        if (!result.decode_pool_) return {BackendFacility{}, "CPU decode threadpool allocation failed"};
        result.prefill_pool_ = result.decode_pool_;
        if (threads.prefill != threads.decode) {
            params = ggml_threadpool_params_default(threads.prefill);
            params.paused = true;
            result.prefill_pool_ = make_pool(&params);
            if (!result.prefill_pool_) return {BackendFacility{}, "CPU prefill threadpool allocation failed"};
        }
        result.select_cpu_use(CpuUse::single_token);
    }
    return {std::move(result), std::string{}};
}

std::pair<bool, std::string> probe(const std::string& device) {
    auto result = BackendFacility::create(device);
    return {static_cast<bool>(result.first), result.second};
}
}

namespace edge0::platform {
bool directly_writable(ggml_backend_t backend, ggml_backend_buffer_t buffer) {
    if (!backend || !buffer) return false;
    if (ggml_backend_buffer_is_host(buffer)) return true;
    const auto device = ggml_backend_get_device(backend);
    const auto type = ggml_backend_buffer_get_type(buffer);
    if (type != ggml_backend_dev_buffer_type(device)) return false;
    const auto registration = ggml_backend_dev_backend_reg(device);
    const std::string name = ggml_backend_reg_name(registration);
    if (name == "CPU") return ggml_backend_buffer_is_host(buffer);
    // Pinned GGML: ggml-metal.cpp shared buffer type is MTL<N>; the
    // shared set_tensor path in ggml-metal-device.m uses memcpy to its base.
    return name == "MTL" && std::string(ggml_backend_buft_name(type)) == ggml_backend_dev_name(device);
}
bool confirm_completion(ggml_backend_t backend, ggml_cgraph* empty_graph) {
    if (!backend || !empty_graph || ggml_graph_n_nodes(empty_graph) != 0) return false;
    // Pinned GGML: ggml_metal_synchronize latches has_error; the next
    // ggml_metal_graph_compute reports it before encoding any commands.
    ggml_backend_synchronize(backend);
    const auto status = ggml_backend_graph_compute_async(backend, empty_graph);
    ggml_backend_synchronize(backend);
    return status == GGML_STATUS_SUCCESS;
}
}

namespace edge0::platform {
Consumption consumption(BufferCapabilities capabilities) {
    if (capabilities.supported) return Consumption::direct;
    if (capabilities.host && capabilities.host_pointer) return Consumption::wrap;
    return Consumption::copy;
}
Consumption consumption(ggml_backend_buffer_type_t home, ggml_backend_t consumer) {
    if (!home || !consumer) return Consumption::copy;
    const auto device = ggml_backend_get_device(consumer);
    if (!device) return Consumption::copy;
    ggml_backend_dev_props properties{};
    ggml_backend_dev_get_props(device, &properties);
    return consumption({ggml_backend_dev_supports_buft(device, home),
        ggml_backend_buft_is_host(home), properties.caps.buffer_from_host_ptr});
}
ggml_backend_buffer_type_t choose_storage_home(const std::vector<ggml_backend_t>& consumers,
    const ConsumptionQuery& query) {
    if (consumers.empty()) return nullptr;
    for (const auto consumer : consumers) if (!consumer) return nullptr;
    std::vector<ggml_backend_buffer_type_t> candidates;
    for (const auto consumer : consumers) {
        const auto type = ggml_backend_get_default_buffer_type(consumer);
        if (std::find(candidates.begin(), candidates.end(), type) == candidates.end()) candidates.push_back(type);
    }
    const auto host_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (host_device) {
        const auto host = ggml_backend_dev_buffer_type(host_device);
        if (std::find(candidates.begin(), candidates.end(), host) == candidates.end()) candidates.push_back(host);
    }
    for (const auto home : candidates) {
        const bool shared = std::all_of(consumers.begin(), consumers.end(), [&](auto consumer) {
            return (query ? query(home, consumer) : consumption(home, consumer)) != Consumption::copy;
        });
        if (shared) return home;
    }
    return nullptr;
}
ggml_backend_buffer_t wrap_host_buffer(ggml_backend_buffer_t home, ggml_backend_t consumer,
    std::size_t maximum_tensor_bytes) {
    if (!home || !consumer || !ggml_backend_buffer_is_host(home) || !maximum_tensor_bytes ||
        maximum_tensor_bytes > ggml_backend_buffer_get_size(home)) return nullptr;
    const auto device = ggml_backend_get_device(consumer);
    if (!device) return nullptr;
    ggml_backend_dev_props properties{};
    ggml_backend_dev_get_props(device, &properties);
    if (!properties.caps.buffer_from_host_ptr) return nullptr;
    return ggml_backend_dev_buffer_from_host_ptr(device, ggml_backend_buffer_get_base(home),
        ggml_backend_buffer_get_size(home), maximum_tensor_bytes);
}
}
