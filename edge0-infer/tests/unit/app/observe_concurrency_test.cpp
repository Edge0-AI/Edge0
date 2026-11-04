#include "edge0/diagnostics.h"
#include "app/generation_internal.h"
#include "tests/support/runtime.h"
#include <atomic>
#include <cstdlib>
#include <thread>
using namespace edge0;
using test::take;
namespace {
std::atomic<std::size_t> memory_calls{0};
platform::ProcessMemory counted_memory() { ++memory_calls; return platform::process_memory(); }
}
int main() {
    const auto* tiny = std::getenv("MOE_TINY"); if (!tiny) return 77;
    auto runtime = take(open_runtime({{"CPU"},{}}));
    { auto resources = take(RuntimeAccess::acquire(runtime)); resources.registry->memory_query = counted_memory; }
    auto model = take(load_model(runtime, take(resolve_load_configuration(runtime, std::string(tiny) + "/tiny-a"))));
    auto config = take(resolve_generation_configuration(model));
    GenerationOverrides overrides; overrides.max_tokens = 16; overrides.temperature = 0;
    GenerationCheckOptions options; options.ignore_end_of_generation = true;
    auto session = take(create_session(model));
    memory_calls = 0;
    const std::vector<std::int32_t> input{1,2,3,4,5};
    const auto baseline = take(generate_tokens_check(session,input,config,overrides,options));
    CHECK(!baseline.result.error && memory_calls == 0);
    take(reset_conversation(session));
    std::atomic<bool> done{false}, active_seen{false};
    std::atomic<std::size_t> polls{0};
    std::thread observer([&] {
        while (!done) {
            const auto progress = take(query_generation_progress(runtime));
            if (progress.phase != GenerationPhase::idle) active_seen = true;
            const auto ledger = take(memory_snapshot(runtime)).ledger;
            std::size_t sum = 0;
            for (const auto& entry : ledger.by_device) sum += entry.second.current_bytes;
            CHECK(sum == ledger.current_bytes);
            const auto system = query_system_status();
            CHECK(system.memory.total.bytes || !system.memory.total.reason.empty());
            ++polls;
        }
    });
    options.cancel_probe = [&](const CancelPoint& point) {
        if (point.kind == CancelPointKind::after_reuse) while (!active_seen) std::this_thread::yield();
        return false;
    };
    const auto observed = take(generate_tokens_check(session,input,config,overrides,options));
    done = true; observer.join();
    CHECK(polls > 0 && active_seen && memory_calls > 0 && !observed.result.error);
    CHECK(observed.result.generated == baseline.result.generated);
    CHECK(take(query_generation_progress(runtime)).phase == GenerationPhase::idle);
    take(finish_runtime_use(runtime));
    std::cout << "concurrent queries: all three queries succeeded, tokens unchanged, baseline made no memory queries\n";
}
