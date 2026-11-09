#include "edge0/engine.h"
#include "tests/support/check.h"
#include <sstream>
edge0::LogReceiver cli_log_receiver(bool verbose);
int main() {
    using edge0::LogLevel;
    for (bool verbose : {false, true}) {
        std::ostringstream output;
        auto* previous = std::cerr.rdbuf(output.rdbuf());
        auto receive = cli_log_receiver(verbose);
        receive(LogLevel::continuation, "orphan ");
        receive(LogLevel::info, "info ");
        receive(LogLevel::continuation, "info-tail ");
        receive(LogLevel::warning, "warning ");
        receive(LogLevel::continuation, "warning-tail ");
        receive(LogLevel::continuation, "warning-tail2 ");
        receive(LogLevel::debug, "debug ");
        receive(LogLevel::continuation, "debug-tail ");
        receive(LogLevel::error, "error ");
        receive(LogLevel::continuation, "error-tail ");
        receive(LogLevel::none, "none ");
        receive(LogLevel::continuation, "none-tail ");
        std::cerr.rdbuf(previous);
        CHECK(output.str() == (verbose
            ? "orphan info info-tail warning warning-tail warning-tail2 debug debug-tail error error-tail none none-tail "
            : "warning warning-tail warning-tail2 error error-tail "));
    }
}
