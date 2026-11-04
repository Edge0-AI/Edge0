#include "edge0/generation.h"
#include "edge0/diagnostics.h"
#include "tests/support/runtime.h"
#include <cstdlib>
using namespace edge0;
using test::take;
int main() {
    const auto* reference=std::getenv("MOE_TINY");if(!reference)return 77;
    const std::string directory=std::string(reference)+"/tiny-a";
    std::size_t required=0;
    {
        auto runtime=take(open_runtime({{"CPU"},{}}));
        auto model=take(load_model(runtime,take(resolve_load_configuration(runtime,directory))));
        auto session=take(create_session(model));required=take(memory_snapshot(runtime)).ledger.current_bytes;
        take(release_session(session));take(unload_model(model));take(finish_runtime_use(runtime));
    }
    RuntimeOptions options;options.devices={"CPU"};options.memory_budget_bytes=required-1;
    auto runtime=take(open_runtime(options));auto model=take(load_model(runtime,take(resolve_load_configuration(runtime,directory))));
    const auto before=take(memory_snapshot(runtime)).ledger.current_bytes;
    auto rejected=create_session(model);CHECK(!rejected && rejected.error().category==ErrorCategory::budget_exceeded);
    CHECK(take(memory_snapshot(runtime)).ledger.current_bytes==before);
    take(unload_model(model));take(finish_runtime_use(runtime));
    std::cout<<"one-byte checkpoint admission failure rolls back ledger\n";
}
