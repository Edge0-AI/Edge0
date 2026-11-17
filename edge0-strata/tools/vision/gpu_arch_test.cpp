// tools/vision/gpu_arch_test.cpp - the HIP picture reader's architecture check (gpu_arch.h) without a GPU.  Exit 0 = passed.
#include "gpu_arch.h"

#include <cstdio>

static int bad = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

int main() {
    CHECK(strata_gfx_in("  Device 0: AMD Radeon RX 9070 XT, gfx1201:sramecc-:xnack- (0x1201), VMM: no, Wave Size: 32") == "gfx1201");
    CHECK(strata_gfx_in("  Device 0: AMD Radeon Graphics, gfx1151 (0x1151)") == "gfx1151");
    CHECK(strata_gfx_in("  Device 0: gfx, nothing after it") == "");
    CHECK(strata_gfx_in("no architecture here") == "");
    CHECK(strata_arch_listed("gfx1200,gfx1201", "gfx1201"));
    CHECK(strata_arch_listed("gfx1200;gfx1201", "gfx1200"));
    CHECK(strata_arch_listed("gfx1100", "gfx1100"));
    CHECK(!strata_arch_listed("gfx1200,gfx1201", "gfx1100"));
    CHECK(!strata_arch_listed("gfx1100", "gfx110"));          // a prefix is not an architecture
    CHECK(!strata_arch_listed("gfx1100", "gfx11000"));
    CHECK(!strata_arch_listed("", "gfx1100"));
    CHECK(!strata_arch_listed("gfx1100", ""));
    std::printf(bad ? "gpu_arch_test: %d FAILED\n" : "gpu_arch_test: ok\n", bad);
    return bad ? 1 : 0;
}
