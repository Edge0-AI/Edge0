// The auto layer split's stage-size rule (#1616): no GPU, no model.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "strata/program/split_rules.hpp"

using namespace strata::program::split_rules;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the reporter's 4-GPU pick on a 48-layer model: layers 0-1, 2, 3-19, 20-47 -> cuts 2,3,20
    CHECK(smallest_stage({2, 3, 20}, 48) == 1);
    CHECK(has_tiny_stage({2, 3, 20}, 48));
    // the manual split that fixed it: 4,8,24 is not a cut list of 4 stages; as cuts 4,8,24 -> 4,4,16,24
    CHECK(smallest_stage({4, 8, 24}, 48) == 4);
    CHECK(!has_tiny_stage({4, 8, 24}, 48));
    // identical cards: the balanced split has no tiny stage, so the rule never fires and the pick stays
    CHECK(!has_tiny_stage({12, 24, 36}, 48));
    CHECK(!has_tiny_stage({24}, 48));
    CHECK(has_tiny_stage({47}, 48));
    CHECK(has_tiny_stage({2}, 48));
    CHECK(!has_tiny_stage({3}, 48));
    CHECK(smallest_stage({}, 48) == 48);
    // the restriction needs the layers for it
    CHECK(restriction_possible(48, 4));
    CHECK(restriction_possible(12, 4));
    CHECK(!restriction_possible(11, 4));
    CHECK(!restriction_possible(48, 1));

    // #1440: two equal cards, 48 layers x 512 experts of ~2 MiB (IQ3_S-like).  Card 0 has more room than card 1 (the
    // last stage also carries the drafter).  Every K whose two caches hold all their pairs predicts the same time.
    {
        const int64_t L = 48, E = 512, blob = 2 << 20, MiB = 1 << 20;
        const std::vector<int64_t> room{29000 * MiB, 26200 * MiB};
        std::vector<int64_t> need(L + 1, 0);
        for (int64_t l = 0; l < L; ++l) need[(size_t) l + 1] = need[(size_t) l] + E * blob;
        auto session = [&](int64_t lb, int64_t le) { return (le - lb) * 40 * MiB; };
        std::vector<Placement> c;
        for (int64_t k = 2; k < L; ++k) {
            Placement p;
            p.cuts = {k};
            p.min_spare = min_spare(p.cuts, L, room, need, session);
            // the miss term: the pairs a stage's cache cannot hold (each costs the same here)
            double missing = 0;
            for (int i = 0; i < 2; ++i) {
                const int64_t lb = i == 0 ? 0 : k, le = i == 0 ? k : L;
                const int64_t fit = (room[(size_t) i] - session(lb, le)) / blob;
                const int64_t want = (le - lb) * E;
                if (fit < want) missing += (double) (want - (fit > 0 ? fit : 0));
            }
            p.ms = 48 * 0.5 + 190.0 * missing / (double) (L * E);
            c.push_back(p);
        }
        const size_t plain = fastest(c), head = pick_with_headroom(c, 0.002);
        // the plain pick is the first full placement, and it leaves card 1 (almost) nothing
        CHECK(c[plain].ms == c[head].ms);
        CHECK(c[plain].cuts[0] < c[head].cuts[0]);
        CHECK(c[plain].min_spare < 1024 * MiB);
        CHECK(c[head].min_spare > 1536 * MiB);
        // the pick is the most balanced full placement: no other full one keeps more on its tightest card
        for (const Placement& p : c)
            if (p.ms == c[plain].ms) CHECK(p.min_spare <= c[head].min_spare);
        std::printf("split_rules_test: #1440 plain K=%lld (spare %lld MiB), headroom K=%lld (spare %lld MiB)\n",
                    (long long) c[plain].cuts[0], (long long) (c[plain].min_spare / MiB), (long long) c[head].cuts[0],
                    (long long) (c[head].min_spare / MiB));
        // a slower placement is never taken for its headroom beyond the tolerance
        std::vector<Placement> two{{{10}, 100.0, 0}, {{20}, 101.0, 5000 * MiB}};
        CHECK(pick_with_headroom(two, 0.002) == 0);
        CHECK(pick_with_headroom(two, 0.02) == 1);
        // exact ties with equal spare keep the first (the old pick)
        std::vector<Placement> same{{{10}, 100.0, 7}, {{20}, 100.0, 7}};
        CHECK(pick_with_headroom(same, 0.0) == 0);
        CHECK(pick_with_headroom({}, 0.1) == 0);
    }
    // #2007: the miss scale per link
    {
        // the case 0.1.41's scale was made for: CUDA1 on a physical x1 link beside an x16 one
        const std::vector<double> x1 = link_miss_scale({24.0, 1.6});
        CHECK(x1[0] == 1.0 && x1[1] > 14.9 && x1[1] < 15.1);
        // two equal cards on gen3 x16 (2x Tesla T10, P100s): the fitted formula, whatever the link speed
        const std::vector<double> eq = link_miss_scale({12.0, 12.0});
        CHECK(eq[0] == 1.0 && eq[1] == 1.0);
        // a moderate gap (x8 beside x16, or the probe's noise): no scale
        const std::vector<double> mod = link_miss_scale({11.4, 5.9});
        CHECK(mod[0] == 1.0 && mod[1] == 1.0);
        // a 4x gap and more scales by the ratio, capped at 64; unprobed stages stay 1
        const std::vector<double> big = link_miss_scale({26.0, 6.0, 0.2, -1.0});
        CHECK(big[0] == 1.0 && big[1] > 4.3 && big[1] < 4.4 && big[2] == 64.0 && big[3] == 1.0);
        CHECK(link_miss_scale({-1.0, -1.0})[1] == 1.0);
    }
    // #2644: RTX 4060 Ti 16 GB + RTX 3060 12 GB, 48 layers.  The reporter's numbers: CUDA1 5.50 GiB free of 11.63 (the
    // 3060 side), CUDA0 the 16 GB card with ~14.2 GiB room.  The #1616 retry alone took K=3 (3 layers on the second card)
    // although the first pick K=2 was the same problem; the share of the room gives it 13 layers of 48.
    {
        const int64_t MiB = 1 << 20, L = 48;
        const std::vector<int64_t> room{14500 * MiB, 5632 * MiB};
        const std::vector<int64_t> f = share_floors(room, L);
        CHECK(f.size() == 2 && f[0] == 34 && f[1] == 13);
        CHECK(floors_exceed_minimum(f));
        // the cut K = the second stage's first layer.  With the big card first, the old rule's edge is K=45 (the second card
        // with 3 layers): below its floor.
        CHECK(below_floors({45}, L, f));
        CHECK(below_floors({47}, L, f));
        CHECK(!below_floors({35}, L, f));   // second card: 13 layers
        CHECK(!below_floors({34}, L, f));   // 14 layers
        CHECK(below_floors({36}, L, f));    // 12 layers: under its share
        CHECK(below_floors({33}, L, f));    // the big card under its share
        // the same pair the other way round: the small card first
        const std::vector<int64_t> r = share_floors({room[1], room[0]}, L);
        CHECK(r[0] == 13 && r[1] == 34);
        CHECK(below_floors({3}, L, r));     // the old retry's K=3 starves the small card
        CHECK(!below_floors({13}, L, r));
        CHECK(!below_floors({14}, L, r));
        // a predicted-time pick under the floors sits at the edge of the floor, not at the three-layer edge
        // (layer ms 0.82 on the 4060 Ti, 1.45 on the 3060 as the engine prices them: fewer layers on the 3060 predict faster)
        auto pick = [&](const std::vector<int64_t>& fl) {
            int64_t best_k = -1;
            double best = 1e30;
            for (int64_t k = 2; k < L; ++k) {
                if (!fl.empty() && below_floors({k}, L, fl)) continue;
                if (smallest_stage({k}, L) < (fl.empty() ? kMinStageLayers : 0)) continue;
                const double ms = (double) k * 1.45 + (double) (L - k) * 0.82;   // card 0 = the 3060 here
                if (ms < best) { best = ms; best_k = k; }
            }
            return best_k;
        };
        CHECK(pick({}) == 3);              // today: the 3060 gets 3 layers
        CHECK(pick(r) == 13);              // with the floors: its share
        std::printf("split_rules_test: #2644 floors %lld,%lld; 3060 first: old retry K=%lld, share retry K=%lld\n",
                    (long long) f[0], (long long) f[1], (long long) pick({}), (long long) pick(r));
        // identical cards: the even split, and the floors never exceed the layers
        const std::vector<int64_t> eq = share_floors({10000 * MiB, 10000 * MiB}, L);
        CHECK(eq[0] == 24 && eq[1] == 24);
        const std::vector<int64_t> three = share_floors({1 * MiB, 1 * MiB, 30000 * MiB}, 12);
        CHECK(three[0] == 3 && three[1] == 3 && three[2] == 6);   // lifted to 3, the big one trimmed to fit 12
        CHECK(share_floors({0, 0}, L)[0] == kMinStageLayers && share_floors({-5, 0}, L)[1] == kMinStageLayers);
        CHECK(!floors_exceed_minimum(share_floors({0, 0}, L)));
        CHECK(!below_floors({}, 48, {}));
    }
    if (failed == 0) std::printf("split_rules_test: ok\n");
    return failed ? 1 : 0;
}
