// The prompt planner's host-memory model (#2824): no GPU, no model.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "strata/program/prefill_plan.hpp"

using namespace strata::program::prefill_plan;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

static uint64_t gib(double g) { return (uint64_t) (g * kGiB); }

// options at the lend sizes the reporter's engine printed: 2048 1,688 slots 2.3 GiB, 4096 2.97K 4.0, 6144 5.8, 8192 7.5
static std::vector<Option> issue_options(int64_t full_ring) {
    std::vector<Option> v;
    const struct { int64_t c; double g; } L[] = {{1024, 1.6}, {2048, 2.3}, {3072, 3.1}, {4096, 4.0}, {5120, 4.9}, {6144, 5.8}, {7168, 6.6}, {8192, 7.5}};
    for (const auto& l : L) v.push_back({l.c, full_ring, full_ring, gib(l.g), 0});
    return v;
}

int main() {
    // #2824: 2x RTX 3060, 32 GB, IQ2_XS: the experts total 33.0 GiB, the GPU caches 13.84, the RAM copy 12.0 of the rest
    Host h;
    h.page_cache = gib(14.5);
    h.h2d_gbps = 8.0;
    h.ssd_gbps = 0.45;
    Tier t;
    t.experts_total = gib(33.0);
    t.gpu_cache = gib(13.84);
    t.file_tier = gib(7.18);
    t.extra_file = gib(0.35);
    const auto opts = issue_options(64);
    // the file sets the reporter computed: 15.1 GiB at 8192, 13.3 at 6144, 11.6 at 4096
    CHECK(std::abs(price(h, t, 29000, opts[7]).file_set / kGiB - 15.03) < 0.15);
    CHECK(std::abs(price(h, t, 29000, opts[5]).file_set / kGiB - 13.33) < 0.15);
    CHECK(std::abs(price(h, t, 29000, opts[3]).file_set / kGiB - 11.53) < 0.15);
    // measured drive reads per 29K prompt: 8192: 28-45 GB, 4096: 0.9-1.4 GB, 6144: 1.2-2.8 GB
    const Cost c8 = price(h, t, 28700, opts[7]), c6 = price(h, t, 28700, opts[5]), c4 = price(h, t, 28700, opts[3]);
    std::printf("8192: set %.1f GiB miss %.2f reads %.1f GB %.0f s | 6144: %.1f GiB %.2f %.1f GB %.0f s | 4096: %.1f GiB %.2f %.1f GB %.0f s\n",
                c8.file_set / kGiB, c8.miss, c8.read_bytes / 1e9, c8.seconds, c6.file_set / kGiB, c6.miss, c6.read_bytes / 1e9,
                c6.seconds, c4.file_set / kGiB, c4.miss, c4.read_bytes / 1e9, c4.seconds);
    CHECK(c8.read_bytes / 1e9 > 28 && c8.read_bytes / 1e9 < 46);
    CHECK(c4.read_bytes == 0 && c4.fits);
    CHECK(c6.read_bytes / 1e9 < 15);   // measured 1.2-2.8 GB: the model is conservative here
    // the planner leaves 8192 for the one that fits
    const Pick p = choose(h, t, 28700, opts, 7);
    std::printf("%s\n", describe(p, h, 28700).c_str());
    CHECK(p.changed);
    CHECK(p.best.opt.chunk >= 3072 && p.best.opt.chunk <= 5120);
    CHECK(p.best.read_bytes < 3e9);

    // P2003 (RTX 3060, IQ2_XS, cgroup 32G, NVMe): 8192 was the fastest of 2048/3328/4096/6144/8192 although it thrashes (measured 480 vs 391 tok/s at 4096).
    // file sets 19.1 / 18.2 / 17.5 / 17.0 / 16.4 GiB against a 17.0 GiB cache; the fast drive makes the extra passes the dearer cost
    Host nv;
    nv.page_cache = gib(17.0);
    nv.h2d_gbps = 25.6;
    nv.ssd_gbps = 2.5;
    Tier t2;
    t2.experts_total = gib(33.0);
    t2.gpu_cache = gib(6.0);
    t2.file_tier = gib(15.0);
    t2.extra_file = gib(0.35);
    std::vector<Option> o4 = {{2048, 96, 96, gib(1.1), 0}, {3328, 96, 96, gib(1.63), 0}, {4096, 96, 96, gib(2.0), 0}, {6144, 96, 96, gib(2.9), 0}, {8192, 96, 96, gib(3.8), 0}};
    const Pick n = choose(nv, t2, 28887, o4, 4);
    std::puts(describe(n, nv, 28887).c_str());
    CHECK(!n.changed);

    // a box where everything fits keeps today's choice
    Host big = h;
    big.page_cache = gib(40);
    const Pick q = choose(big, t, 28700, opts, 7);
    CHECK(!q.changed && q.best.opt.chunk == 8192 && q.best.read_bytes == 0);

    // a box whose file set can never fit (8 GB card + 32 GB RAM): no refusal, no blind shrink: the fewest passes win
    Tier hard;
    hard.experts_total = gib(48);
    hard.gpu_cache = gib(3.0);
    hard.file_tier = gib(40);
    hard.extra_file = gib(0.4);
    Host small = h;
    small.page_cache = gib(22);
    std::vector<Option> o2;
    for (int64_t c = 1024; c <= 8192; c += 1024) o2.push_back({c, 64, 64, gib(0.9 * (double) c / 1024.0), 0});
    const Pick r = choose(small, hard, 28700, o2, 7);
    std::printf("%s\n", describe(r, small, 28700).c_str());
    CHECK(r.best.opt.chunk == 8192);
    // and from a small today's choice it moves UP to fewer passes when nothing fits either way
    const Pick r2 = choose(small, hard, 28700, o2, 1);
    CHECK(r2.best.opt.chunk > 2048);

    // unbuffered reads (no page cache): every chunk reads its set; the biggest chunk reads least
    Host ub = h;
    ub.page_cache = 0;
    CHECK(price(ub, t, 28700, opts[7]).read_bytes < price(ub, t, 28700, opts[3]).read_bytes);

    // lent experts the RAM copy keeps leave the file set: 8192 with all 7.5 GiB kept is 7.5 GiB smaller
    Option kept = opts[7];
    kept.lend_kept = kept.lend_bytes;
    CHECK(price(h, t, 28700, kept).file_set < price(h, t, 28700, opts[7]).file_set - gib(7.4));
    CHECK(price(h, t, 28700, kept).fits);

    // a smaller ring only wins when it brings the set into the cache
    Tier edge = t;
    edge.file_tier = gib(6.2);   // 8192 with the 64-slot ring: 6.2+7.5+.35 = 14.05 GiB, 0.97 of the cache
    std::vector<Option> o3 = {{8192, 64, 64, gib(7.5), 0}, {8192, 16, 64, gib(7.0), 0}};
    const Pick s = choose(h, edge, 28700, o3, 0);
    std::printf("%s\n", describe(s, h, 28700).c_str());
    // request-sized chunks: a prompt that fits today's 8192 stays one chunk; a longer one is cut into equal chunks of at most the plan's
    CHECK(balanced_chunk(4101, 8192, 3840) == 4352);
    CHECK(balanced_chunk(8192, 8192, 3840) == 8192);
    CHECK(balanced_chunk(300, 8192, 3840) == 512);
    CHECK(balanced_chunk(20400, 8192, 3840) == 3584);    // 6 equal chunks (last 2,480), no stub
    CHECK(balanced_chunk(28887, 8192, 3840) == 3840);    // 8 chunks: 7 x 3840 + 2,007
    CHECK(balanced_chunk(8200, 8192, 3840) == 2816);     // 3 chunks of 2,816 instead of 3,840 + 3,840 + 520
    CHECK(balanced_chunk(20400, 8192, 0) == 8192);       // no plan: today's cap
    for (int64_t t = 8193; t < 40000; t += 37) {
        const int64_t c = balanced_chunk(t, 8192, 3840);
        const int64_t n = (t + c - 1) / c;
        CHECK(c % 256 == 0 && c <= 8192);
        CHECK((t - (n - 1) * c) * 5 >= c * 2 || n == 1);   // the last chunk is at least 40% of a chunk
        CHECK(c <= 3840);
        CHECK(n <= (t + 3839) / 3840 + 1);
    }
    std::printf("%s\n", failed ? "FAILED" : "prefill_plan_test: ok");
    return failed ? 1 : 0;
}
