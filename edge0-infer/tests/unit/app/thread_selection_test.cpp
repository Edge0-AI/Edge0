#include "app/runtime.h"
#include "edge0/forward_check.h"
#include "tests/support/check.h"
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <thread>

int main() {
    const char* root = std::getenv("MOE_TINY");
    if (!root) { std::cout << "SKIP: MOE_TINY is not set\n"; return 77; }
    // Prefill threads must differ from decode's one and stay within the logical CPU count.
    const auto logical = std::thread::hardware_concurrency();
    if (logical < 2) { std::cout << "SKIP: needs at least two logical CPUs\n"; return 77; }
    const int prefill_threads = int(std::min(4u, logical));
    auto runtime = edge0::open_runtime({{"CPU"}, {}, std::nullopt, 1, prefill_threads});
    CHECK(runtime);
    {
        auto resources = edge0::RuntimeAccess::acquire(runtime.value()); CHECK(resources);
        auto facility = resources.value().backends.at("CPU")->facility;
        edge0::ForwardCheckOptions options;
        options.model_directory = std::string(root) + "/tiny-a";
        options.device = "CPU"; options.input = {1, 2, 3};
        auto prefill = edge0::forward_check(runtime.value(), options); CHECK(prefill);
        CHECK(facility->cpu_use() == edge0::platform::CpuUse::multi_token);
        CHECK(facility->cpu_threads() == prefill_threads);
        options.generated = {4};
        auto decode = edge0::forward_check(runtime.value(), options); CHECK(decode);
        CHECK(facility->cpu_use() == edge0::platform::CpuUse::single_token);
        CHECK(facility->cpu_threads() == 1);
        CHECK(!resources.value().registry->active);
    }
    CHECK(edge0::finish_runtime_use(runtime.value()));
    std::cout << "Independent forward selects prefill and decode pools and releases admission\n";
}
