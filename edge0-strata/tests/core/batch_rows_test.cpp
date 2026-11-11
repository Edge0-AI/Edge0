// The batch window's row layout (include/strata/core/batch_rows.hpp): plain one-row slots, a slot's contiguous group of
// proposed rows (--batch-mtp), and the hand-off capacity of a layer split's stages.  Header-only, CPU.
#include "strata/core/batch_rows.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
const char* run(const int* rows, const long long* p, int S, int n_slots, int max_t, int hbase, bool handoff) {
    int64_t pos[8];
    for (int t = 0; t < S; ++t) pos[t] = p[t];
    return strata::core::batch_rows_error(rows, S, pos, n_slots, max_t, hbase, handoff, 8);
}
}  // namespace

int main() {
    using strata::core::batch_rows_error;
    // plain batch: one row per slot, any positions
    {
        const int rows[] = {0, 1, 2};
        const long long pos[] = {10, 500, 3};
        check(run(rows, pos, 3, 3, 8, 0, false) == nullptr, "one row per slot");
        check(run(rows, pos, 3, 3, 8, 0, true) == nullptr, "one row per slot through the hand-off");
    }
    // --batch-mtp: a slot's two rows at consecutive positions; also on a layer split (S > slots)
    {
        const int rows[] = {1, 1, 0, 0};
        const long long pos[] = {40, 41, 7, 8};
        check(run(rows, pos, 4, 2, 8, 0, false) == nullptr, "two rows per slot, one stage");
        check(run(rows, pos, 4, 2, 8, 0, true) == nullptr, "two rows per slot across a layer split (S > slots)");
    }
    // eight rows = four slots of two: the whole hand-off
    {
        const int rows[] = {0, 0, 1, 1, 2, 2, 3, 3};
        const long long pos[] = {1, 2, 5, 6, 9, 10, 20, 21};
        check(run(rows, pos, 8, 4, 8, 0, true) == nullptr, "eight rows fill the hand-off");
        check(run(rows, pos, 8, 4, 8, 1, true) != nullptr, "a group past the hand-off's end");
        check(run(rows, pos, 8, 4, 8, 1, false) == nullptr, "no hand-off: the base does not bound the rows");
    }
    // refused layouts
    {
        const int rows[] = {0, 1, 0};
        const long long pos[] = {5, 9, 6};
        check(run(rows, pos, 3, 2, 8, 0, false) != nullptr, "a slot's rows must be contiguous");
    }
    {
        const int rows[] = {0, 0};
        const long long pos[] = {5, 7};
        check(run(rows, pos, 2, 1, 8, 0, false) != nullptr, "a slot's rows need consecutive positions");
    }
    {
        const int rows[] = {0, 2};
        const long long pos[] = {5, 7};
        check(run(rows, pos, 2, 2, 8, 0, false) != nullptr, "a slot past the slots held");
        const int neg[] = {-1};
        const long long p1[] = {1};
        check(run(neg, p1, 1, 2, 8, 0, false) != nullptr, "a negative slot");
    }
    {
        const int rows[] = {0, 0, 0};
        const long long pos[] = {1, 2, 3};
        check(run(rows, pos, 3, 1, 2, 0, false) != nullptr, "more rows than the window holds");
        check(run(rows, pos, 0, 1, 8, 0, false) != nullptr, "an empty window");
        check(run(rows, pos, 3, 1, 8, -1, false) != nullptr, "a negative hand-off base");
        check(run(rows, pos, 3, 1, 8, 0, false) == nullptr, "three rows of one slot are a group");
    }
    if (failures != 0) return EXIT_FAILURE;
    std::printf("batch_rows_test: ok\n");
    return EXIT_SUCCESS;
}
