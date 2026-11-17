// The tile flatten, on its own: every input row covered exactly once, counts within [1, 64], absolute starts, and the
// same rows as the unflattened groups for random distributions (long tail, cnt 0, cnt 1, cnt = tile, cnt = tile + 1).
#include "strata/prefill/lat_pseudo_groups.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using strata::prefill::flatten_groups;
using strata::prefill::LatPseudoGroup;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static void one_case(const char* tag, const std::vector<int32_t>& counts, int32_t base_step, int tile) {
    const int n = (int) counts.size();
    std::vector<unsigned long long> blobs((size_t) n);
    std::vector<int32_t> starts((size_t) n);
    int32_t at = 0;
    for (int g = 0; g < n; ++g) { blobs[(size_t) g] = 0x1000ull + (unsigned long long) g; starts[(size_t) g] = at; at += counts[(size_t) g] + base_step; }
    const auto out = flatten_groups(blobs.data(), starts.data(), counts.data(), n, tile);

    // 1) counts within [1, tile]; 2) each piece lies inside its group and the pieces of a group tile it exactly
    std::vector<int> seen((size_t) at, 0);
    size_t q = 0;
    for (int g = 0; g < n; ++g) {
        int32_t covered = 0;
        while (q < out.size() && out[q].blob == blobs[(size_t) g]) {
            CHECK(out[q].count >= 1 && out[q].count <= tile);
            CHECK(out[q].start == starts[(size_t) g] + covered);
            for (int32_t r = 0; r < out[q].count; ++r) {
                const size_t row = (size_t) (out[q].start + r);
                CHECK(row < (size_t) at);
                ++seen[row];
            }
            covered += out[q].count;
            ++q;
        }
        CHECK(covered == counts[(size_t) g]);
    }
    CHECK(q == out.size());                       // nothing left over
    // 3) no row covered twice, and only the groups' own rows are touched
    for (int g = 0, i = 0; g < n; ++g)
        for (int32_t r = 0; r < counts[(size_t) g]; ++r, ++i) {
            const size_t row = (size_t) (starts[(size_t) g] + r);
            CHECK(seen[row] == 1);
        }
    std::printf("  %-28s groups %4d  rows %6d  pieces %6zu  ok\n", tag, n, at, out.size());
}

int main() {
    std::mt19937 rng(7);
    const int tile = 64;
    one_case("uniform 83", std::vector<int32_t>(64, 83), 0, tile);
    one_case("all one", std::vector<int32_t>(200, 1), 0, tile);
    one_case("all zero", std::vector<int32_t>(16, 0), 0, tile);
    one_case("tile boundary", {63, 64, 65, 128, 129, 3098}, 0, tile);
    {   // the engine's own long tail
        std::vector<int32_t> c((size_t) 512);
        std::uniform_int_distribution<int> d(1, 300);
        for (auto& v : c) v = d(rng);
        for (int i = 0; i < 6; ++i) c[(size_t) (rng() % 512)] = 3000 + (int) (rng() % 100);
        one_case("random + long tail", c, 0, tile);
    }
    {   // gaps between groups (starts are absolute and need not tile the row space)
        std::vector<int32_t> c((size_t) 40);
        std::uniform_int_distribution<int> d(0, 200);
        for (auto& v : c) v = d(rng);
        one_case("gaps between groups", c, 7, tile);
    }
    std::printf("%s: tile flatten (%d failure(s))\n", fails == 0 ? "PASS" : "FAIL", fails);
    return fails == 0 ? 0 : 1;
}
