// pick_window_victim (include/strata/core/window_evict.hpp): CPU only (#3702).
#include "strata/core/window_evict.hpp"

#include <cstdio>
#include <cstdlib>

using strata::core::pick_window_victim;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}  // namespace

int main() {
    bool have[9] = {};
    check(pick_window_victim(have, 8, 5, 0) == 0, "nothing captured: nothing to free");
    have[2] = have[3] = have[5] = true;
    check(pick_window_victim(have, 8, 6, 0) == 5, "the largest captured size goes first");
    check(pick_window_victim(have, 8, 5, 0) == 3, "never the size being captured");
    check(pick_window_victim(have, 8, 5, 3) == 2, "never the prelaunched size");
    check(pick_window_victim(have, 4, 1, 0) == 3, "sizes above max_t are not considered");
    have[5] = false;
    check(pick_window_victim(have, 8, 3, 2) == 0, "only the kept and the pinned sizes are left: nothing");
    have[2] = have[3] = false;
    have[1] = true;
    check(pick_window_victim(have, 8, 7, 0) == 1, "size 1 can be freed");
    check(pick_window_victim(have, 8, 1, 0) == 0, "size 1 kept");
    std::printf("window_evict_test: %d checks passed\n", checks);
    return 0;
}
