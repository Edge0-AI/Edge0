// src/core/nv_defaults_test.cpp - the NVIDIA per-architecture default tables (no GPU needed): the sm_121 (GB10, DGX Spark) table, its
// two off switches, and that no other card's table changed.
#include "strata/core/nv_defaults.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static void unset(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}
static void put(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

static const std::string* find(const std::vector<std::pair<std::string, std::string>>& t, const char* k) {
    for (const auto& kv : t)
        if (kv.first == k) return &kv.second;
    return nullptr;
}

int main() {
    using namespace strata::core;
    unset("STRATA_SM121_PERF");
    unset("STRATA_SM86_PERF");
    unset("STRATA_SM60_PERF");
    unset("STRATA_EMULATE_CC");
    // GB10 (12.1): the four measured switches, and nothing else
    {
        const auto t = nv_default_env(12, 1, false);
        CHECK(t.size() == 4);
        CHECK(find(t, "STRATA_PF_FUSED") && *find(t, "STRATA_PF_FUSED") == "1");
        CHECK(find(t, "STRATA_PREFILL_NVFP4") && *find(t, "STRATA_PREFILL_NVFP4") == "w4a4");
        CHECK(find(t, "STRATA_HC_UPMIX") && *find(t, "STRATA_HC_UPMIX") == "1");
        CHECK(find(t, "STRATA_SPEC_COST") && *find(t, "STRATA_SPEC_COST") == "1");
    }
    // STRATA_DEFAULTS=026 (the 0.2.7 master switch; the sm_121 table is newer than 0.2.6) empties it
    CHECK(nv_default_env(12, 1, true).empty());
    // STRATA_SM121_PERF=0 empties it; any other value keeps it
    put("STRATA_SM121_PERF", "0");
    CHECK(nv_default_env(12, 1, false).empty());
    put("STRATA_SM121_PERF", "1");
    CHECK(nv_default_env(12, 1, false).size() == 4);
    unset("STRATA_SM121_PERF");
    // the other cards: sm_120 (RTX 50) and sm_100 get nothing from this file; the sm_86 / sm_60 tables are unchanged by the flag
    CHECK(nv_default_env(12, 0, false).empty());
    CHECK(nv_default_env(10, 0, false).empty());
    CHECK(nv_default_env(8, 9, false).empty());
    CHECK(nv_default_env(7, 5, false).empty());
    {
        const auto a = nv_default_env(8, 6, false), b = nv_default_env(8, 6, true);
        CHECK(!a.empty() && a == b);               // STRATA_DEFAULTS=026 does not touch the 2026-10 sm_86 table
        const auto p = nv_default_env(6, 0, false), q = nv_default_env(6, 0, true);
        CHECK(!p.empty() && p == q);
        CHECK(!find(a, "STRATA_PF_FUSED") && !find(p, "STRATA_PF_FUSED"));
    }
    // the one-argument form reads STRATA_DEFAULTS
    put("STRATA_DEFAULTS", "026");
    CHECK(nv_default_env(12, 1).empty());
    unset("STRATA_DEFAULTS");
    if (fails == 0) std::printf("nv_defaults_test: ok\n");
    return fails == 0 ? 0 : 1;
}
