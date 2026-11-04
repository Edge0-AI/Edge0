#include "edge0/diagnostics.h"
#include "edge0/generation.h"
#include "tests/support/runtime.h"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
using namespace edge0;
using test::take;
int main() {
    const auto* directory=std::getenv("MOE_ASSETS_35B");
    if(!directory || !std::filesystem::is_regular_file(std::filesystem::path(directory)/"assets.tsv")) {
        std::cout<<"SKIP: missing MOE_ASSETS_35B\n";return 77;
    }
    auto runtime=take(open_runtime({{"CPU"},{}}));
    for(const std::int64_t context:{4096,8192}) {
        LoadOverrides overrides;overrides.context=context;overrides.device="CPU";
        auto model=take(load_model(runtime,take(resolve_load_configuration(runtime,directory,overrides))));
        const auto before=take(memory_snapshot(runtime)).ledger.current_bytes;
        auto session=take(create_session(model));
        const auto after=take(memory_snapshot(runtime)).ledger.current_bytes;
        CHECK(after>=before);
        const auto actual=after-before;
        const auto expected=std::uint64_t((context+255)/256*256)*20*1024+3*65863680;
        const auto relative=std::abs(double(actual)-double(expected))/double(expected);
        std::cout<<std::setprecision(17)<<"{\"context\":"<<context<<",\"loaded_current_bytes\":"<<before
                 <<",\"session_current_bytes\":"<<after<<",\"state_bytes\":"<<actual
                 <<",\"expected_bytes\":"<<expected<<",\"relative_error\":"<<relative<<"}\n";
        CHECK(relative<=0.01);
        take(release_session(session));take(unload_model(model));
    }
    take(finish_runtime_use(runtime));
}
