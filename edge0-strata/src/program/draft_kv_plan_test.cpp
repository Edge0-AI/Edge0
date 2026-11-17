// src/program/draft_kv_plan_test.cpp - which path computes the draft layer's prompt K/V (header-only, CPU).
//
//   1. STRATA_SPLIT_MTP_BATCH: unset, empty, 0 and text that is not a number are off; any other whole number is on;
//   2. one GPU: the batched pass is tried whatever the variable says (the default before this change);
//   3. a layer split: the default stays the drafter's own pass, silently; with the variable on the batched pass is
//      tried, unless the pack has no GGUF-form token table, which is reported instead;
//   4. no draft layer: nothing is tried and nothing is said.
#include "strata/program/draft_kv_plan.hpp"

#include <cstdio>
#include <initializer_list>

namespace dkv = strata::program::draft_kv;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-72s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("draft_kv_plan_test\n");
    check(!dkv::split_env_on(nullptr), "unset is off");
    check(!dkv::split_env_on(""), "empty is off");
    check(!dkv::split_env_on("0"), "0 is off");
    check(!dkv::split_env_on("off"), "text that is not a number is off");
    check(dkv::split_env_on("1"), "1 is on");
    check(dkv::split_env_on("2"), "any other whole number is on");

    for (const bool env : {false, true})
        for (const bool native : {false, true}) {
            const dkv::Plan p = dkv::plan(true, false, env, native);
            check(p.try_batched && p.why == nullptr, "one GPU: the batched pass is tried, nothing is reported");
        }

    {
        const dkv::Plan p = dkv::plan(true, true, false, true);
        check(!p.try_batched && p.why == nullptr, "split, variable off: the drafter's own pass, silently");
    }
    {
        const dkv::Plan p = dkv::plan(true, true, false, false);
        check(!p.try_batched && p.why == nullptr, "split, variable off, no token table: still silent");
    }
    {
        const dkv::Plan p = dkv::plan(true, true, true, true);
        check(p.try_batched && p.why == nullptr, "split, variable on: the batched pass is tried on the last stage");
    }
    {
        const dkv::Plan p = dkv::plan(true, true, true, false);
        check(!p.try_batched && p.why != nullptr, "split, variable on, no token table: declined with a reason");
    }
    for (const bool multi : {false, true})
        for (const bool env : {false, true}) {
            const dkv::Plan p = dkv::plan(false, multi, env, true);
            check(!p.try_batched && p.why == nullptr, "no draft layer: nothing tried, nothing said");
        }

    std::printf(g_fail == 0 ? "draft_kv_plan_test: all ok\n" : "draft_kv_plan_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
