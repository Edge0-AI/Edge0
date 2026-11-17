// warm_policy_test: when the HIP start-up warm-up of hipBLAS may run (include/strata/prefill/warm_policy.hpp). Pure.
#include "strata/prefill/warm_policy.hpp"

#include <cstdio>

using strata::prefill::warm_gemm_wanted;

static int g_fail = 0;
static void expect(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

int main() {
    expect(warm_gemm_wanted({}, nullptr), "no device list: the warm-up runs");
    expect(warm_gemm_wanted({"gfx1201"}, nullptr), "one R9700: on");
    expect(warm_gemm_wanted({"gfx1201", "gfx1201"}, nullptr), "2x R9700 (same architecture): on (021-iD: no loss)");
    expect(warm_gemm_wanted({"gfx1201:sramecc+:xnack-", "gfx1201"}, nullptr), "feature suffixes do not make a mixed pair");
    expect(!warm_gemm_wanted({"gfx1151", "gfx1201"}, nullptr), "8060S + R9700 (#3204): off");
    expect(!warm_gemm_wanted({"gfx1201", "gfx1151"}, nullptr), "the order does not matter");
    expect(!warm_gemm_wanted({"gfx1100", "gfx1100", "gfx1030"}, nullptr), "any second architecture turns it off");
    expect(!warm_gemm_wanted({"gfx1201"}, "0"), "STRATA_PREFILL_WARM=0: off");
    expect(!warm_gemm_wanted({"gfx1201", "gfx1201"}, "0"), "STRATA_PREFILL_WARM=0 on two cards: off");
    expect(warm_gemm_wanted({"gfx1151", "gfx1201"}, "1"), "STRATA_PREFILL_WARM=1 forces it on a mixed pair");
    expect(!warm_gemm_wanted({"gfx1151", "gfx1201"}, ""), "an empty value is unset: a mixed pair stays off");
    if (g_fail) { std::fprintf(stderr, "warm_policy_test: %d failure(s)\n", g_fail); return 1; }
    std::printf("warm_policy_test: OK\n");
    return 0;
}
