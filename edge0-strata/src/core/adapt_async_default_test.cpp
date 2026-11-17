// adapt_async_default_test: the asynchronous adaptive tier as a measured choice (include/strata/core/adapt_async_default.hpp).
#include "strata/core/adapt_async_default.hpp"

#include <cstdio>
#include <cstring>

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

using strata::core::AdaptAsyncAuto;
using strata::core::AdaptAsyncSelf;
using Action = AdaptAsyncSelf::Action;
using Phase = AdaptAsyncSelf::Phase;

static AdaptAsyncAuto good() {
    AdaptAsyncAuto a;
    a.cuda = a.serve = a.adaptive = a.arena = true;
    return a;
}

static AdaptAsyncSelf::Config cfg0() {   // no warm-up: the cases below count their windows from the first
    AdaptAsyncSelf::Config c;
    c.warmup_windows = 0;
    return c;
}

// feed `n` windows of `ms` / `wait` / `tok`; returns the last non-None action
static Action feed(AdaptAsyncSelf& s, int n, double ms, double wait, int tok) {
    Action last = Action::None;
    for (int i = 0; i < n; ++i) {
        const Action a = s.window(ms, wait, tok);
        if (a != Action::None) last = a;
    }
    return last;
}

int main() {
    const char* why = nullptr;
    // ---- where the engine may choose by measurement
    CHECK(strata::core::adapt_async_self_decides(good(), &why));
    { AdaptAsyncAuto a = good(); a.flag = 0; CHECK(!strata::core::adapt_async_self_decides(a, &why)); CHECK(why && std::strstr(why, "given")); }
    { AdaptAsyncAuto a = good(); a.flag = 1; CHECK(!strata::core::adapt_async_self_decides(a)); }
    { AdaptAsyncAuto a = good(); a.env = 0; CHECK(!strata::core::adapt_async_self_decides(a, &why)); CHECK(why && std::strstr(why, "STRATA_ADAPT_ASYNC")); }
    { AdaptAsyncAuto a = good(); a.env = 1; CHECK(strata::core::adapt_async_self_decides(a)); }
    { AdaptAsyncAuto a = good(); a.env = -1; CHECK(strata::core::adapt_async_self_decides(a)); }
    { AdaptAsyncAuto a = good(); a.cuda = false; CHECK(!strata::core::adapt_async_self_decides(a)); }
    { AdaptAsyncAuto a = good(); a.serve = false; CHECK(!strata::core::adapt_async_self_decides(a)); }
    { AdaptAsyncAuto a = good(); a.adaptive = false; CHECK(!strata::core::adapt_async_self_decides(a)); }   // --adapt-every 100000 gates
    { AdaptAsyncAuto a = good(); a.arena = false; CHECK(!strata::core::adapt_async_self_decides(a)); }
    { AdaptAsyncAuto a = good(); a.multi_gpu = true; CHECK(!strata::core::adapt_async_self_decides(a)); }
    { AdaptAsyncAuto a = good(); a.pipeline_windows = true; CHECK(!strata::core::adapt_async_self_decides(a)); }
    { AdaptAsyncAuto a = good(); a.batch = true; CHECK(!strata::core::adapt_async_self_decides(a)); }
    { AdaptAsyncAuto a = good(); a.peer = true; CHECK(!strata::core::adapt_async_self_decides(a)); }
    { AdaptAsyncAuto a = good(); a.remote_opt = true; CHECK(!strata::core::adapt_async_self_decides(a)); }
    { AdaptAsyncAuto a = good(); a.adapt_overlap = true; CHECK(!strata::core::adapt_async_self_decides(a)); }
    CHECK(!strata::core::adapt_async_self_decides(AdaptAsyncAuto{}));

    // ---- the state machine
    {   // not armed: nothing happens
        AdaptAsyncSelf s;
        CHECK(s.phase() == Phase::Off);
        CHECK(feed(s, 500, 20.0, 5.0, 3) == Action::None);
    }
    {   // the wait is small (3% of a window): measure 64 windows, stay blocking, never switch
        AdaptAsyncSelf s(cfg0()); s.arm();
        CHECK(feed(s, 63, 20.0, 0.6, 3) == Action::None);
        CHECK(s.phase() == Phase::Measure);
        CHECK(feed(s, 1, 20.0, 0.6, 3) == Action::None);
        CHECK(s.phase() == Phase::Done && !s.kept());
        CHECK(feed(s, 500, 20.0, 0.6, 3) == Action::None);
    }
    {   // the wait is 7%: switch to async after the 64th window; the async windows are 10% faster per token: kept
        AdaptAsyncSelf s(cfg0()); s.arm();
        CHECK(feed(s, 63, 20.0, 1.4, 3) == Action::None);
        CHECK(s.window(20.0, 1.4, 3) == Action::ToAsync);
        CHECK(s.phase() == Phase::Trial && s.trialing());
        CHECK(s.wait_fraction() > 0.069 && s.wait_fraction() < 0.071);
        CHECK(feed(s, 4 + 63, 18.0, 0.1, 3) == Action::None);       // 4 warm-up windows are not counted
        CHECK(s.window(18.0, 0.1, 3) == Action::None);              // kept: no action
        CHECK(s.phase() == Phase::Done && s.kept());
        CHECK(s.trial_ms_per_token() < s.measure_ms_per_token());
    }
    {   // the wait is 7% and async is 3% SLOWER per token (> 2%): back to blocking, and it stays there
        AdaptAsyncSelf s(cfg0()); s.arm();
        CHECK(feed(s, 64, 20.0, 1.4, 3) == Action::ToAsync);
        CHECK(feed(s, 4 + 63, 20.6, 0.1, 3) == Action::None);
        CHECK(s.window(20.6, 0.1, 3) == Action::ToBlocking);
        CHECK(s.phase() == Phase::Done && !s.kept());
        CHECK(feed(s, 1000, 19.6, 0.1, 3) == Action::None);
    }
    {   // async slower (the CPU-bound box): back
        AdaptAsyncSelf s(cfg0()); s.arm();
        CHECK(feed(s, 64, 55.0, 4.0, 3) == Action::ToAsync);
        CHECK(feed(s, 4 + 63, 58.0, 0.2, 3) == Action::None);
        CHECK(s.window(58.0, 0.2, 3) == Action::ToBlocking);
    }
    {   // async 1% slower or equal: kept (the trial guards against harm, it does not demand a gain)
        AdaptAsyncSelf s(cfg0()); s.arm();
        CHECK(feed(s, 64, 20.0, 1.4, 3) == Action::ToAsync);
        CHECK(feed(s, 4 + 64, 20.2, 0.1, 3) == Action::None);
        CHECK(s.kept());
    }
    {   // 5% wait switches; a 3% gain is kept
        AdaptAsyncSelf s(cfg0()); s.arm();
        CHECK(feed(s, 64, 100.0, 5.0, 4) == Action::ToAsync);
        const double t = 100.0 / 1.0301;
        CHECK(feed(s, 4 + 64, t, 0.0, 4) == Action::None);
        CHECK(s.kept());
    }
    {   // windows with no token or no time are ignored (they neither count nor move the phase)
        AdaptAsyncSelf s(cfg0()); s.arm();
        CHECK(feed(s, 200, 20.0, 5.0, 0) == Action::None);
        CHECK(feed(s, 200, 0.0, 0.0, 3) == Action::None);
        CHECK(s.phase() == Phase::Measure);
    }
    {   // tokens per window matter: more tokens per window at the same time is faster per token
        AdaptAsyncSelf s(cfg0()); s.arm();
        CHECK(feed(s, 64, 20.0, 2.0, 2) == Action::ToAsync);       // 10 ms per token
        CHECK(feed(s, 4 + 64, 20.0, 0.1, 3) == Action::None);      // 6.7 ms per token: kept
        CHECK(s.kept());
    }
    {   // the default has a warm-up: the first 64 steady windows are not measured (a fresh engine's cache is still settling)
        AdaptAsyncSelf s; s.arm();
        CHECK(feed(s, 64, 20.0, 5.0, 3) == Action::None);          // warm-up
        CHECK(s.phase() == Phase::Measure);
        CHECK(feed(s, 63, 20.0, 1.4, 3) == Action::None);
        CHECK(s.window(20.0, 1.4, 3) == Action::ToAsync);          // 128th window: the measurement is the last 64 only
        CHECK(s.wait_fraction() > 0.069 && s.wait_fraction() < 0.071);
    }
    if (g_fail == 0) std::printf("adapt_async_default_test: OK\n");
    return g_fail == 0 ? 0 : 1;
}
