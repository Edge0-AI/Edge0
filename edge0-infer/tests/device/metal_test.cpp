#include "edge0/engine.h"
#include "tests/support/check.h"
#include <algorithm>
#include <iostream>

int main() {
    const auto caps = edge0::query_capabilities();
    auto found = std::find_if(caps.devices.begin(), caps.devices.end(), [](const auto& device) {
        return device.backend == "metal" && !device.name.empty();
    });
    if (found == caps.devices.end()) {
        std::cout << "Not executed: no Metal device\n";
        return 77;
    }
    for (int i = 0; i < 2; ++i) {
        auto runtime = edge0::open_runtime({{found->name}, {}});
        if (!runtime) std::cerr << runtime.error().message << '\n';
        CHECK(runtime);
        CHECK(runtime.value().available_devices() == std::vector<std::string>{found->name});
        CHECK(edge0::finish_runtime_use(runtime.value()));
    }
    std::cout << "metal: two open/finish cycles passed for " << found->name << '\n';
}
