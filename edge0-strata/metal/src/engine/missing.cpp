// metal/src/engine/missing.cpp - what a kernel entry point that has no Metal implementation yet does when called.
//
// Default: say which one and stop (exit code 86), so a run never continues on outputs nobody wrote.
// STRATA_METAL_STUBS=log: say it once per entry point and return (outputs untouched) - for listing which entry points a
// code path reaches, never for answers.
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>

namespace strata::metal_engine {

void missing(const char* name) {
    static const bool log_only = [] {
        const char* v = std::getenv("STRATA_METAL_STUBS");
        return v != nullptr && std::string(v) == "log";
    }();
    if (!log_only) {
        std::fprintf(stderr, "strata metal: %s has no Metal implementation yet\n", name);
        std::fflush(stderr);
        std::_Exit(86);
    }
    static std::mutex mu;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lk(mu);
    if (seen.insert(name).second) std::fprintf(stderr, "strata metal: STUB %s\n", name);
}

}  // namespace strata::metal_engine

// ---- entry points the shared host loop asks every request, with nothing to do on Metal yet
#include "strata/kernels/cvec.hpp"
namespace strata::kernels {
// no control vector on Metal (--control-vector-scaled is not ported): the empty one, `loaded()` false
const Cvec& cvec() {
    static const Cvec none{};
    return none;
}
}  // namespace strata::kernels

// an uncaught exception or abort on the Metal build prints where it came from (macOS gives no core dump by default and
// lldb needs a debugging permission a user's session may not have)
#include <csignal>
#include <signal.h>
#include <execinfo.h>
#include <exception>
#include <unistd.h>
namespace {
[[noreturn]] void metal_terminate() {
    void* fr[64];
    const int n = backtrace(fr, 64);
    const char msg[] = "strata metal: terminate - the call stack:\n";
    (void) !write(2, msg, sizeof msg - 1);
    backtrace_symbols_fd(fr, n, 2);
    std::abort();
}
const bool g_terminate_set = (std::set_terminate(metal_terminate), true);
// a crash (SIGSEGV / SIGBUS) prints its call stack too: macOS keeps no report for a command-line process by default
void metal_crash(int sig, siginfo_t* si, void*) {
    void* fr[64];
    const int n = backtrace(fr, 64);
    char msg[160];
    const int m = std::snprintf(msg, sizeof msg, "strata metal: crashed (signal %d at address %p) - the call stack:\n", sig,
                                si ? si->si_addr : nullptr);
    (void) !write(2, msg, (size_t) m);
    backtrace_symbols_fd(fr, n, 2);
    signal(sig, SIG_DFL);
    raise(sig);
}
bool set_crash() {
    struct sigaction sa {};
    sa.sa_sigaction = metal_crash;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
    return true;
}
const bool g_crash_set = set_crash();
}  // namespace
