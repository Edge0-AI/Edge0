#pragma once

#include "ggml-backend.h"
#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace edge0::platform {
enum class CpuUse { single_token, multi_token };
struct CpuThreads { int decode = 4, prefill = 4; };
CpuThreads default_cpu_threads(unsigned logical_cpus, bool android, bool windows = false);
CpuThreads default_cpu_threads();

struct Device {
    std::string backend;
    std::string name;
    std::string description;
    enum ggml_backend_dev_type type;
    std::size_t memory_total_bytes;
    std::size_t memory_free_bytes;
};

std::vector<std::string> compiled_backends();
std::vector<Device> devices();

class BackendFacility {
public:
    BackendFacility() = default;
    ~BackendFacility();
    BackendFacility(const BackendFacility&) = delete;
    BackendFacility& operator=(const BackendFacility&) = delete;
    BackendFacility(BackendFacility&& other) noexcept;
    BackendFacility& operator=(BackendFacility&& other) noexcept;
    explicit operator bool() const noexcept { return backend_ != nullptr; }
    ggml_backend_t handle() const noexcept { return backend_; }
    ggml_backend_dev_t device() const noexcept { return backend_ ? ggml_backend_get_device(backend_) : nullptr; }
    static std::pair<BackendFacility, std::string> create(const std::string& device, CpuThreads threads = default_cpu_threads());
    void select_cpu_use(CpuUse use);
    CpuUse cpu_use() const noexcept { return use_; }
    int cpu_threads() const noexcept { return use_ == CpuUse::single_token ? threads_.decode : threads_.prefill; }
private:
    explicit BackendFacility(ggml_backend_t backend) : backend_(backend) {}
    void release();
    void swap(BackendFacility& other) noexcept;
    ggml_backend_t backend_ = nullptr;
    ggml_threadpool_t decode_pool_ = nullptr, prefill_pool_ = nullptr;
    void (*free_pool_)(ggml_threadpool_t) = nullptr;
    void (*set_pool_)(ggml_backend_t, ggml_threadpool_t) = nullptr;
    ggml_backend_set_n_threads_t set_threads_ = nullptr;
    CpuThreads threads_;
    CpuUse use_ = CpuUse::single_token;
};

bool directly_writable(ggml_backend_t backend, ggml_backend_buffer_t buffer);
enum class Consumption { direct, wrap, copy };
struct BufferCapabilities {
    bool supported = false, host = false, host_pointer = false;
};
Consumption consumption(BufferCapabilities capabilities);
Consumption consumption(ggml_backend_buffer_type_t home, ggml_backend_t consumer);
using ConsumptionQuery = std::function<Consumption(ggml_backend_buffer_type_t, ggml_backend_t)>;
// A null home means each consumer needs an independent allocation.
ggml_backend_buffer_type_t choose_storage_home(const std::vector<ggml_backend_t>& consumers,
    const ConsumptionQuery& query = {});
ggml_backend_buffer_t wrap_host_buffer(ggml_backend_buffer_t home, ggml_backend_t consumer,
    std::size_t maximum_tensor_bytes);
bool confirm_completion(ggml_backend_t backend, ggml_cgraph* empty_graph);

std::pair<bool, std::string> probe(const std::string& device);
}
