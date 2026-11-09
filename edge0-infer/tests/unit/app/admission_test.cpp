#include "app/runtime.h"
#include "model/loading.h"
#include "models/edge0_35b/assets.h"
#include "tests/support/check.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
int main() {
    const auto* path = std::getenv("MOE_ASSETS_35B");
    if (!path || !std::filesystem::exists(std::filesystem::path(path) / "assets.tsv")) {
        std::cout << "SKIP: missing MOE_ASSETS_35B\n"; return 77;
    }
    std::string device;
    for (const auto& item : edge0::query_capabilities().devices)
        if (item.backend == "metal") { device = item.name; break; }
    if (device.empty()) { std::cout << "SKIP: Metal device unavailable\n"; return 77; }
    auto runtime = edge0::open_runtime({{device}, {}});
    CHECK(runtime);
    {
        auto acquired = edge0::RuntimeAccess::acquire(runtime.value());
        CHECK(acquired);
        auto resources = acquired.value();
        edge0::model::LoadReport report;
        edge0::model::LoadRequest request;
        request.directory = path; request.device = device; request.report = &report;
        request.options = {{"routing", "actual"}, {"lora", "on"}, {"experts", "resident"}};
        const auto start = std::chrono::steady_clock::now();
        auto loaded = edge0::model::load(request, {&edge0::models::edge0_35b::adapter()}, resources.budget, resources.backends);
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        CHECK(!loaded && loaded.error().category == edge0::ErrorCategory::budget_exceeded);
        CHECK(report.weight_reads == 0 && resources.budget.current() == 0);
        std::cout << loaded.error().message << "; weight_reads=" << report.weight_reads
                  << "; budget_current=" << resources.budget.current() << "; seconds=" << elapsed << '\n';
    }
    CHECK(edge0::finish_runtime_use(runtime.value()));
}
