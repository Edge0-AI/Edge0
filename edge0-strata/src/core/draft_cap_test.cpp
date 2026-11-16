// src/core/draft_cap_test.cpp - the per-card window cap of --draft dflash2 (CPU only).
#include "strata/core/draft_cap.hpp"

#include <cstdio>

using namespace strata::core;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    CHECK(dflash2_auto_cap(true, 0) == 5);     // Strix Halo
    CHECK(dflash2_auto_cap(false, 12) == 6);   // Blackwell (sm_120)
    CHECK(dflash2_auto_cap(false, 8) == 0);    // Ampere / Ada: --spec as given
    CHECK(dflash2_auto_cap(false, 0) == 0);    // another AMD card
    CHECK(dflash2_small_nvidia(true, (size_t) 12 << 30));
    CHECK(!dflash2_small_nvidia(true, (size_t) 16 << 30));
    CHECK(!dflash2_small_nvidia(false, (size_t) 8 << 30));
    if (fails == 0) std::printf("draft_cap_test: ok\n");
    return fails == 0 ? 0 : 1;
}
