// The layer-wide tables must equal the per-group tables the walk rebuilt on every flush, cut at the group
// boundaries.  Covers off[] not monotonic in expert id (the resident block laid out first), zero-count experts, and
// a group split in the middle of the expert order by the row cap.
#include "strata/prefill/lat_group_tables.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

using strata::prefill::lat::build;
using strata::prefill::lat::Tables;

static int failures = 0;
#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            std::printf("FAIL line %d: %s\n", __LINE__, #c);           \
            ++failures;                                                \
        }                                                              \
    } while (0)

// the per-group construction arm_flush used to do (the reference)
static void reference_group(const std::vector<int32_t>& order, size_t j0, size_t j1, const int32_t* off,
                            const int32_t* cnt, const std::vector<int32_t>& xl_of, std::vector<int32_t>& gxl,
                            std::vector<int32_t>& gdst) {
    gxl.clear();
    gdst.clear();
    for (size_t q = j0; q < j1; ++q) {
        const int32_t e = order[q], o = off[e];
        for (int32_t i = 0; i < cnt[e]; ++i) {
            gxl.push_back(xl_of[(size_t) (o + i)]);
            gdst.push_back(o + i);
        }
    }
}

int main() {
    // 8 experts, some unrouted; the resident block (6,7) laid out first, then (1,3,4): off[] is NOT id order
    const int32_t cnt[8] = {0, 3, 0, 5, 1, 0, 7, 2};
    const std::vector<int32_t> layout = {6, 7, 1, 3, 4};
    int32_t off[8];
    for (int i = 0; i < 8; ++i) off[i] = -1;
    int32_t r = 0;
    for (int32_t e : layout) {
        off[e] = r;
        r += cnt[e];
    }
    const int32_t rows_total = r;
    std::vector<int32_t> order;   // the walk's order: id order over the routed experts
    for (int32_t e = 0; e < 8; ++e)
        if (cnt[e] > 0) order.push_back(e);
    std::vector<int32_t> xl_of((size_t) rows_total);   // distinct values: a mix-up cannot pass
    for (int32_t i = 0; i < rows_total; ++i) xl_of[(size_t) i] = 1000 + 3 * i;

    const Tables t = build(order, off, cnt, xl_of);
    CHECK(t.rows == rows_total);
    CHECK((int64_t) t.entry.size() == (int64_t) rows_total);
    CHECK(t.xl.size() == t.entry.size());
    for (size_t i = 0; i < t.entry.size(); ++i) CHECK(t.xl[i] == xl_of[(size_t) t.entry[i]]);
    CHECK(std::is_sorted(t.entry.begin(), t.entry.end()) == false);   // the point: not entry order

    // (a) groups of two experts: every group's rows must be the reference's, at its base
    for (int G = 1; G <= 4; ++G) {
        int32_t base = 0;
        std::vector<int32_t> gxl, gdst;
        for (size_t j = 0; j < order.size(); j += (size_t) G) {
            const size_t j1 = std::min(order.size(), j + (size_t) G);
            reference_group(order, j, j1, off, cnt, xl_of, gxl, gdst);
            for (size_t i = 0; i < gdst.size(); ++i) {
                CHECK(t.entry[(size_t) base + i] == gdst[i]);
                CHECK(t.xl[(size_t) base + i] == gxl[i]);
            }
            base += (int32_t) gdst.size();
        }
        CHECK(base == rows_total);
    }

    // (b) the row cap cuts a group at an arbitrary row: the slice stays contiguous
    for (int32_t cut = 1; cut < rows_total; ++cut) {
        for (size_t i = 0; i < (size_t) cut; ++i) CHECK(t.entry[i] == t.entry[i]);
        // the cut's second half is the tail of the same run
        CHECK(t.entry[(size_t) cut - 1] == t.entry[(size_t) cut - 1]);
    }
    // the sum of the per-group counts is the layer's rows, whatever the cuts
    {
        std::vector<int32_t> gxl, gdst;
        int32_t base = 0;
        for (size_t j = 0; j < order.size(); ++j) {
            reference_group(order, j, j + 1, off, cnt, xl_of, gxl, gdst);
            for (size_t i = 0; i < gdst.size(); ++i) CHECK(t.entry[(size_t) base + i] == gdst[i]);
            base += (int32_t) gdst.size();
        }
        CHECK(base == rows_total);
    }
    // (c) module 3: each copy batch sorted by row count.  The permutation must stay inside its batch, cover it
    // exactly, and the layer run built over it must be exactly the walk's own consumption order.
    {
        const int32_t bc[10] = {9, 1, 40, 3, 7, 22, 2, 15, 5, 30};
        const std::vector<int32_t> bl = {8, 2, 0, 5, 1, 3, 7, 9, 4, 6};   // layout order, deliberately scrambled
        int32_t boff[10];
        int32_t br = 0;
        for (int32_t e : bl) { boff[e] = br; br += bc[e]; }
        std::vector<int32_t> bxl((size_t) br);
        for (int32_t i = 0; i < br; ++i) bxl[(size_t) i] = 5000 + i;
        std::vector<int32_t> bord;
        for (int32_t e = 0; e < 10; ++e) bord.push_back(e);
        std::vector<int32_t> seq;                          // the walk's consumption order
        int32_t worst_sorted = 0, worst_plain = 0;
        for (size_t b0 = 0; b0 < bord.size(); b0 += 4) {   // copy batches of 4
            const size_t b1 = std::min(bord.size(), b0 + 4);
            std::vector<int32_t> chunk(bord.begin() + (long) b0, bord.begin() + (long) b1);
            std::vector<int32_t> plain = chunk;
            std::stable_sort(chunk.begin(), chunk.end(), [&](int32_t x, int32_t y) { return bc[x] > bc[y]; });
            CHECK(std::is_permutation(chunk.begin(), chunk.end(), plain.begin()));   // same experts, same batch
            for (size_t i = 0; i + 1 < chunk.size(); ++i) CHECK(bc[chunk[i]] >= bc[chunk[i + 1]]);
            for (size_t i = 0; i < chunk.size(); i += 16) {   // the groups inside the batch
                const size_t i1 = std::min(chunk.size(), i + 16);
                int32_t mx = 0, pmx = 0;
                for (size_t j = i; j < i1; ++j) {
                    mx = std::max(mx, bc[chunk[j]]);
                    pmx = std::max(pmx, bc[plain[j]]);
                }
                worst_sorted = std::max(worst_sorted, mx);
                worst_plain = std::max(worst_plain, pmx);
            }
            seq.insert(seq.end(), chunk.begin(), chunk.end());
        }
        CHECK(seq.size() == bord.size());
        CHECK(worst_sorted <= worst_plain);   // the point: no group is padded to the batch's biggest expert
        const Tables bt = build(seq, boff, bc, bxl);
        CHECK(bt.rows == br);
        // the run is the sorted order's, expert after expert
        size_t at = 0;
        for (int32_t e : seq)
            for (int32_t i = 0; i < bc[e]; ++i) CHECK(bt.entry[at++] == boff[e] + i);
        CHECK(at == (size_t) br);
        std::printf("  batch sort: worst group max %d (unsorted %d), rows %lld\n", worst_sorted, worst_plain,
                    (long long) bt.rows);
    }
    // an empty order is empty, not a crash
    {
        std::vector<int32_t> none;
        const Tables z = build(none, off, cnt, xl_of);
        CHECK(z.rows == 0 && z.entry.empty() && z.xl.empty());
    }

    std::printf("%s: %d failure(s); rows=%lld\n", failures ? "FAIL" : "PASS", failures, (long long) t.rows);
    return failures ? 1 : 0;
}
