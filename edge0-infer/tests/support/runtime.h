#pragma once
#include "exec/execution.h"
#include "tests/support/check.h"
#include <cstring>
namespace test {
template<class T> T take(const edge0::Result<T>& result) {
    if (!result) { std::cerr << result.error().message << '\n'; std::exit(1); }
    return result.value();
}
inline std::shared_ptr<edge0::exec::Backend> backend(int argc, char** argv) {
    const bool device = argc == 3 && std::strcmp(argv[1], "--backend") == 0 && std::strcmp(argv[2], "metal") == 0;
    std::string name = "CPU";
    if (device) {
        name.clear();
        for (const auto& item : edge0::platform::devices()) if (item.backend == "metal") { name = item.name; break; }
        if (name.empty()) { std::cout << "not executed: no Metal device\n"; std::exit(77); }
    }
    auto facility = edge0::platform::BackendFacility::create(name);
    CHECK(facility.first);
    auto owner = std::make_shared<edge0::platform::BackendFacility>(std::move(facility.first));
    return take(edge0::exec::resolve_stages(name, name, owner)).decode;
}
}
