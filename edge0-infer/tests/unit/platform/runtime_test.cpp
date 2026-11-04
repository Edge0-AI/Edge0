#include "platform/backend.h"
#include "platform/file.h"
#include "tests/support/check.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <chrono>
int main() {
    auto backend = edge0::platform::BackendFacility::create("CPU");
    CHECK(backend.first);
    CHECK(backend.first.device());
    auto buffer = ggml_backend_alloc_buffer(backend.first.handle(), 32);
    CHECK(buffer);
    CHECK(edge0::platform::directly_writable(backend.first.handle(), buffer));
    CHECK(!edge0::platform::directly_writable(nullptr, buffer));
    ggml_backend_buffer_free(buffer);
    auto* context = ggml_init({ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(8, false) * 2, nullptr, true});
    auto* graph = ggml_new_graph_custom(context, 8, false);
    auto* empty = ggml_new_graph_custom(context, 8, false);
    auto* x = ggml_new_tensor_1d(context, GGML_TYPE_F32, 1);
    auto* y = ggml_scale(context, x, 2.0f);
    ggml_build_forward_expand(graph, y);
    auto allocation = ggml_backend_alloc_ctx_tensors(context, backend.first.handle());
    float input = 3;
    ggml_backend_tensor_set(x, &input, 0, sizeof(input));
    CHECK(ggml_backend_graph_compute_async(backend.first.handle(), graph) == GGML_STATUS_SUCCESS);
    CHECK(edge0::platform::confirm_completion(backend.first.handle(), empty));
    float output = 0;
    ggml_backend_tensor_get(y, &output, 0, sizeof(output));
    CHECK(output == 6);
    ggml_backend_buffer_free(allocation);
    ggml_free(context);
    auto path = std::filesystem::current_path() / "build" / ("io-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    { std::ofstream file(path, std::ios::binary); file << "abcdef"; }
    auto file = edge0::platform::ReadFile::open(path.string());
    CHECK(file);
    if(edge0::platform::supports_direct_read()) {
        auto direct=edge0::platform::ReadFile::open(path.string(),edge0::platform::ReadMode::direct);CHECK(direct);
        std::array<char,6> cached_bytes{},direct_bytes{};
        CHECK(!file.value()->read_exact(0,cached_bytes.data(),cached_bytes.size()));
        CHECK(!direct.value()->read_exact(0,direct_bytes.data(),direct_bytes.size()));
        CHECK(cached_bytes==direct_bytes);
    }
    edge0::platform::file_testing::disable_direct_read=true;
    auto unsupported=edge0::platform::ReadFile::open("missing-no-open",edge0::platform::ReadMode::direct);
    CHECK(!unsupported && unsupported.error().category==edge0::ErrorCategory::unsupported);
    edge0::platform::file_testing::disable_direct_read=false;
    auto id = file.value()->identity();
    CHECK(id && id.value().size == 6 && id.value().inode != 0 && id.value().modified_seconds > 0);
    std::array<char, 3> data{};
    CHECK(!file.value()->read_exact(2, data.data(), data.size()));
    CHECK((data == std::array<char, 3>{'c', 'd', 'e'}));
    CHECK(file.value()->read_exact(5, data.data(), data.size()));
    { std::ofstream append(path, std::ios::app); append << 'g'; }
    CHECK(!(file.value()->identity().value() == id.value()));
    std::filesystem::remove(path);
    std::cout << "platform: direct buffer, completion probe, file identity and offset reads passed\n";
}
