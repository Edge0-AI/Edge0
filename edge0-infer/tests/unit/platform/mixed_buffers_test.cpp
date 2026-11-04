#include "platform/backend.h"
#include "platform/memory.h"
#include "tests/support/check.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

namespace {
struct Store {
    ggml_context* context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor *a = nullptr, *b = nullptr;
    Store(ggml_backend_t backend, std::size_t count) {
        context = ggml_init({2 * ggml_tensor_overhead(), nullptr, true});
        CHECK(context);
        a = ggml_new_tensor_1d(context, GGML_TYPE_F32, static_cast<int64_t>(count));
        b = ggml_new_tensor_1d(context, GGML_TYPE_F32, static_cast<int64_t>(count));
        buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        CHECK(buffer);
        ggml_backend_buffer_clear(buffer, 0);
    }
    ~Store() { ggml_backend_buffer_free(buffer); ggml_free(context); }
    Store(const Store&) = delete;
    Store& operator=(const Store&) = delete;
};
struct Wrapped {
    ggml_context* context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor *a = nullptr, *b = nullptr;
    Wrapped(const Store& home, ggml_backend_dev_t device) {
        auto* base = static_cast<unsigned char*>(ggml_backend_buffer_get_base(home.buffer));
        buffer = ggml_backend_dev_buffer_from_host_ptr(device, base,
            ggml_backend_buffer_get_size(home.buffer), ggml_nbytes(home.a));
        CHECK(buffer);
        CHECK(ggml_backend_dev_supports_buft(device, ggml_backend_buffer_get_type(buffer)));
        context = ggml_init({2 * ggml_tensor_overhead(), nullptr, true});
        CHECK(context);
        a = ggml_dup_tensor(context, home.a); b = ggml_dup_tensor(context, home.b);
        CHECK(ggml_backend_tensor_alloc(buffer, a, home.a->data) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_tensor_alloc(buffer, b, home.b->data) == GGML_STATUS_SUCCESS);
        CHECK(a->data == home.a->data && b->data == home.b->data);
    }
    ~Wrapped() { ggml_backend_buffer_free(buffer); ggml_free(context); }
    Wrapped(const Wrapped&) = delete;
    Wrapped& operator=(const Wrapped&) = delete;
};
void scale(ggml_backend_t backend, ggml_tensor* source, ggml_tensor* destination, float factor) {
    auto* context = ggml_init({8 * ggml_tensor_overhead() + 2 * ggml_graph_overhead_custom(16, false), nullptr, true});
    CHECK(context);
    auto* graph = ggml_new_graph_custom(context, 16, false);
    auto* empty = ggml_new_graph_custom(context, 16, false);
    auto* value = ggml_scale(context, source, factor);
    auto* write = ggml_cpy(context, value, destination);
    ggml_build_forward_expand(graph, write);
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    CHECK(ggml_gallocr_alloc_graph(allocator, graph));
    CHECK(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS);
    CHECK(edge0::platform::confirm_completion(backend, empty));
    ggml_gallocr_free(allocator); ggml_free(context);
}
std::vector<float> read(ggml_tensor* tensor) {
    std::vector<float> values(static_cast<std::size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    return values;
}
void exact(ggml_tensor* a, ggml_tensor* b) {
    const auto x = read(a), y = read(b);
    CHECK(x.size() == y.size());
    CHECK(std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0);
}
std::uint64_t footprint() {
    const auto memory = edge0::platform::process_memory();
    CHECK(memory.physical.values && memory.physical.values->current_bytes);
    return *memory.physical.values->current_bytes;
}
}
int main() {
    auto cpu = edge0::platform::BackendFacility::create("CPU");
    const auto devices = edge0::platform::devices();
    const auto found = std::find_if(devices.begin(), devices.end(), [](const auto& device) {
        return device.backend == "metal";
    });
    if (found == devices.end()) { std::cout << "Metal unavailable\n"; return 77; }
    auto metal = edge0::platform::BackendFacility::create(found->name);
    if (!metal.first) { std::cout << "Metal unavailable: " << metal.second << '\n'; return 77; }
    CHECK(cpu.first);
    ggml_backend_dev_props properties{};
    ggml_backend_dev_get_props(metal.first.device(), &properties);
    CHECK(properties.caps.buffer_from_host_ptr);
    const auto cb = cpu.first.handle(), mb = metal.first.handle();
    // Include non-page-sized storage and multiple tensor offsets.
    for (const std::size_t count : {std::size_t(257), std::size_t(16384)}) {
        Store home(cb, count), reference(mb, count);
        Wrapped mapped(home, metal.first.device());
        std::vector<float> input(count);
        for (std::size_t i = 0; i < count; ++i) input[i] = static_cast<float>(int(i % 127) - 63) / 16;
        ggml_backend_tensor_set(home.a, input.data(), 0, ggml_nbytes(home.a));
        ggml_backend_tensor_set(reference.a, input.data(), 0, ggml_nbytes(reference.a));
        // CPU graph writes; Metal consumes the same physical bytes after completion.
        scale(cb, home.a, home.b, 2);
        scale(mb, reference.a, reference.b, 2);
        scale(mb, mapped.b, mapped.a, 3);
        scale(mb, reference.b, reference.a, 3);
        exact(home.a, reference.a);
        // Metal graph writes; CPU consumes without an intervening tensor copy.
        scale(cb, home.a, home.b, 4);
        scale(mb, reference.a, reference.b, 4);
        exact(home.b, reference.b);
        const auto output = read(home.b);
        for (std::size_t i = 0; i < count; ++i) CHECK(output[i] == input[i] * 24);
        std::cout << "wrap_exact count=" << count
                  << " host_alignment=" << ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(home.buffer))
                  << " mapped_alignment=" << ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(mapped.buffer))
                  << " mapped_max_size=" << ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(mapped.buffer)) << '\n';
    }
    constexpr std::size_t bytes = 64 * 1024 * 1024;
    Store host(cb, bytes / sizeof(float)), device(mb, bytes / sizeof(float));
    // Warm the memory query and mapping machinery before measuring only the wrapper.
    const auto before = footprint();
    {
        Wrapped mapped(host, metal.first.device());
        const auto after = footprint();
        const auto increase = after > before ? after - before : 0;
        const auto size = ggml_backend_buffer_get_size(host.buffer);
        std::cout << "wrap_footprint bytes=" << size << " before=" << before
                  << " after=" << after << " increase=" << increase << std::endl;
        CHECK(increase < size / 10);
    }
    std::vector<float> payload(bytes / sizeof(float), 1.25f);
    ggml_backend_tensor_set(host.a, payload.data(), 0, bytes);
    for (int direction = 0; direction < 2; ++direction) {
        auto* source = direction == 0 ? host.a : device.a;
        auto* target = direction == 0 ? device.a : host.b;
        ggml_backend_synchronize(cb); ggml_backend_synchronize(mb);
        const auto start = std::chrono::steady_clock::now();
        ggml_backend_tensor_copy(source, target);
        ggml_backend_synchronize(cb); ggml_backend_synchronize(mb);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        exact(source, target);
        std::cout << "copy direction=" << (direction == 0 ? "CPU_to_Metal" : "Metal_to_CPU")
                  << " bytes=" << bytes << " seconds=" << seconds << '\n';
    }
    std::cout << "mixed_buffers: wrap selected; bidirectional visibility and exact copy passed\n";
}
