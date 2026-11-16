// The window limits (include/strata/kernels/window_limits.hpp) and the pipelined-window probe
// (include/strata/program/pipe_probe.hpp): CPU only, header-only.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/program/pipe_probe.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
#if defined(_WIN32)
void setenv(const char* k, const char* v, int) { _putenv_s(k, v); }   // MSVC has _putenv_s only; an empty value unsets it
void unsetenv(const char* k) { _putenv_s(k, ""); }
#endif
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}  // namespace

int main() {
    using namespace strata::kernels;
    check(kVerifyNarrowT == 8, "the narrow window is the 0.2.7 one: 8 rows");
    check(kWindowMaxT == 8 || kWindowMaxT == 16, "a window holds 8 or 16 rows");
    check(kVerifyMaxT == kWindowMaxT, "kVerifyMaxT is the window limit");
    check(kVerifyMaxT >= kVerifyNarrowT, "the wide window is not narrower than the narrow one");
    check(cpu::MAXT == 8, "the CPU kernels' tokens per job stay 8 (a wider window is split into jobs)");
    check(kFusedGrMaxT == 8, "the hyper-connection kernels are built for 8 tokens (a wider window is read in slices)");
    check(kFusedGrWideT == kWindowMaxT, "the caller's token arrays follow the window");
    check(kVerifyMaxT * 10 <= 256, "the resident plan's block holds a full window of 10 routed experts a row");

    // the probe: off without the environment (no clock read), and a full cycle of events does not fail when it is on
    unsetenv("STRATA_P4_TIMING");
    unsetenv("STRATA_P4_TRACE");
    {
        strata::program::PipeProbe off(4, 4);
        check(!off.on() && !off.timing(), "the probe is off by default");
        check(off.now() == strata::program::PipeProbe::Time{}, "an off probe reads no clock");
        { strata::program::PipeProbe::PumpScope s(off); }
        off.launched(0, 1, off.now());
        off.done(0, 1);
        off.flush();
    }
    setenv("STRATA_P4_TIMING", "1", 1);
    {
        strata::program::PipeProbe on(4, 4);
        check(on.timing(), "STRATA_P4_TIMING turns the timing on");
        on.span_start(on.now());
        for (int w = 0; w < 3; ++w)
            for (int k = 0; k < 4; ++k) {
                strata::program::PipeProbe::PumpScope s(on);
                on.busy(2, 3);
                on.free_stage(1, k % 2 == 0);
                const auto before = on.now();
                on.launched(k, w, before);
                on.polled(before);
                on.done(k, w);
                strata::program::PipeProbe::VerdictScope v(on);
            }
        on.span_end(10.0);
        on.flush();
    }
    unsetenv("STRATA_P4_TIMING");
    std::printf("window_limits_test: %d checks passed (window %d rows)\n", checks, (int) kVerifyMaxT);
    return 0;
}
