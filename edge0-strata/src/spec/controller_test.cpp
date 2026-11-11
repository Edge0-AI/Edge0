// src/spec/controller_test.cpp - plan v0.3 P6: the speculation controller's decisions (no model, no GPU).
#include "strata/spec/controller.hpp"

#include <cstdio>

using namespace strata::spec;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}
const char* name(Source s) { return s == Source::Mtp ? "mtp" : s == Source::Lookup ? "lookup" : "none"; }
}  // namespace

int main() {
    {   // priors (MTP 0.86 per draft): MTP with a short-to-medium window beats plain decoding
        Controller c;
        const Choice ch = c.choose(0, 0, true);
        check(ch.source == Source::Mtp && ch.k >= 2 && ch.k <= 5, "MTP chosen with k in 2..5 at the prior");
        std::printf("prior: %s k=%d expected %.2f tokens, %.1f tok/s (plain %.1f)\n", name(ch.source), ch.k,
                    ch.expected_tokens, 1000 * ch.tokens_per_ms, 1000.0 / c.cost().step_ms(0, false));
    }
    {   // a drafter that keeps missing its first token is switched off
        Controller c;
        Choice ch = c.choose(0, 0, true);
        for (int i = 0; i < 400; ++i) { c.observe(ch, 0, 0); ch = c.choose(0, 0, true); if (ch.source == Source::None) break; }
        check(ch.source == Source::None, "persistent rejection turns speculation off");
        std::printf("after rejections: %s, p0 %.2f\n", name(ch.source), c.mtp_accept(0));
    }
    {   // a long lookup match beats MTP; how LONG a window pays depends on what a wide verify costs
        Controller c;
        const Choice ch = c.choose(8, 20, true);
        check(ch.source == Source::Lookup, "long lookup match preferred over MTP");
        std::printf("long lookup match, measured costs: %s k=%d expected %.2f, %.1f tok/s\n", name(ch.source), ch.k,
                    ch.expected_tokens, 1000 * ch.tokens_per_ms);
        CostModel cheap;                                    // a verify pass whose cost barely grows with width
        cheap.dense_ratio = {1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16};
        cheap.hit_rate = 0.95;
        Controller c2(cheap);
        const Choice ch2 = c2.choose(8, 20, true);
        check(ch2.source == Source::Lookup && ch2.k == 8, "cheap wide verify takes the whole lookup window");
        std::printf("long lookup match, cheap wide verify: %s k=%d expected %.2f, %.1f tok/s\n", name(ch2.source),
                    ch2.k, ch2.expected_tokens, 1000 * ch2.tokens_per_ms);
    }
    {   // a bare trigram match, no MTP: not worth a verify window
        Controller c;
        const Choice ch = c.choose(8, 3, false);
        check(ch.source == Source::None, "short lookup match alone is not used");
    }
    {   // perfect acceptance is learned and widens the window
        Controller c;
        Choice ch = c.choose(0, 0, true);
        const int k0 = ch.k;
        for (int i = 0; i < 400; ++i) { c.observe(ch, ch.k, 0); ch = c.choose(0, 0, true); }
        check(ch.k > k0 && c.mtp_accept(0) > 0.99, "full acceptance learned and the window widens");
        std::printf("after full acceptance: %s k=%d (was %d), p0 %.3f\n", name(ch.source), ch.k, k0, c.mtp_accept(0));
    }
    std::printf("\nMTP acceptance -> choice (default cost model):\n");
    for (double p : {0.5, 0.6, 0.7, 0.8, 0.86, 0.9, 0.95}) {
        Controller c;
        Choice ch = c.choose(0, 0, true);
        for (int i = 0; i < 2000; ++i) {                    // feed a stream with iid acceptance p per position
            int acc = 0;
            unsigned r = 2654435761u * (unsigned) (i + 1);
            while (acc < ch.k && ((r = r * 1103515245u + 12345u) >> 8) % 1000 < p * 1000) ++acc;
            c.observe(ch, acc, 0);
            ch = c.choose(0, 0, true);
        }
        std::printf("  p=%.2f: %s k=%d, %.1f tok/s\n", p, name(ch.source), ch.k, 1000 * ch.tokens_per_ms);
    }
    std::printf("controller tests: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
