#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// The prompt path's chunk, streamed ring and expert-cache loan, priced against the HOST as well as the GPU (#2824).
//
// The automatic chunk scan only asks whether the loan fits the GPU cache.  But a lent slot's expert is not in VRAM
// during the prompt: it comes from the RAM copy when the copy kept it, else from the model file, and a chunk of
// stream_all_min_tokens or more streams EVERY expert the GPU does not hold, once per chunk.  When the file set
// (the experts neither the GPU cache nor the RAM copy hold, plus the displaced ones with no RAM backup) is bigger than
// the RAM the OS can keep file pages in, the set cycles through the page cache and each chunk reads most of it again:
// 2x RTX 3060 / 32 GB / IQ2_XS read 28-45 GB for one 29K prompt at 8192-token chunks (file set 15.1 GiB against a
// 14.5 GiB page cache) and 0.9-1.4 GB at 4096 (11.6 GiB).
//
// This is the pure model (no engine, no GPU) so it can be tested on the CPU.  The engine builds the Options (a chunk x
// ring grid, each with the bytes it lends and the bytes of those the RAM copy keeps) and `choose` returns the one with
// the least predicted prompt time; a FlexGen-style joint choice of placement (what is lent) and I/O schedule (how many
// passes over the file set): when the set can never fit, f is ~1 for every option and the fewest passes (the biggest
// chunk) wins, so nothing is "rejected" and nothing is shrunk blindly.
//
// Time per prompt of N tokens in n = ceil(N / chunk) chunks:
//     n * (chunk-compute + stream(chunk) / h2d) + file_reads / ssd,   file_reads = n * f(W / page_cache) * W
// The streaming term is ADDITIVE with the compute term: the 12 GB-card 32K Q2_0 prompt measured 791 / 878 / 973 tok/s at
// 4096 / 6144 / 8192, which is 1,275 tok/s of compute plus ~1.9 s of streaming per chunk (a 20 GB stream at ~10 GB/s).
namespace strata::program::prefill_plan {

constexpr double kGiB = 1073741824.0;

struct Host {
    uint64_t page_cache = 0;     ///< bytes the OS can keep file pages in while a prompt runs (0 = reads are unbuffered)
    double h2d_gbps = 12.0;      ///< host -> device, GB/s (the engine's probe)
    double ssd_gbps = 0.45;      ///< file tier reads, GB/s (rotational ~0.15, SATA ~0.45, NVMe ~2.5)
    double compute_tok_s = 1200; ///< prompt compute without streaming, tokens/s (only scales the printed seconds)
    double ring_loss = 0.05;     ///< prompt time lost between a full ring and the smallest one (#583: 17 vs 199 slots)
};

struct Tier {
    uint64_t experts_total = 0;  ///< every routed expert of the pack
    uint64_t gpu_cache = 0;      ///< bytes of all GPU expert caches (lent slots included)
    uint64_t file_tier = 0;      ///< experts neither a GPU cache nor the RAM copy holds (the decode-time file set)
    uint64_t extra_file = 0;     ///< other file pages a prompt touches (PLE rows)
};

struct Option {
    int64_t chunk = 0;
    int64_t ring = 0;            ///< the ring slots the prompt path will really lay out
    int64_t ring_full = 0;       ///< the largest ring on offer (0 = unknown / not adjustable)
    uint64_t lend_bytes = 0;     ///< cache bytes the loan displaces
    uint64_t lend_kept = 0;      ///< ...of which the RAM copy keeps a copy
};

struct Cost {
    Option opt;
    double file_set = 0;         ///< bytes the prompt path reads through the file tier, one chunk
    double miss = 0;             ///< share of that set each chunk fetches from the drive
    double read_bytes = 0;       ///< drive bytes per prompt (steady state: the set already warm)
    double seconds = 0;          ///< predicted prompt time
    int64_t chunks = 0;
    bool fits = false;           ///< the set sits in the page cache with room to spare
};

/// The share of a chunk's file set that comes from the drive, from r = file_set / page_cache, with page_cache = the RAM
/// available after the copy less a slack.  Fitted to two rigs that disagree below r = 1 and agree above it:
///   #2824 (2x RTX 3060, 32 GB, SATA, 4 chunk sizes): r 0.80 read ~1 GB per prompt, r 0.92 1.2-2.8 GB, r 1.04 a miss share of 0.47-0.75;
///   P2003 (RTX 3060, IQ2_XS, cgroup 32G, 5 chunk sizes, 29K prompt): r 0.96 0.38, 1.00 0.44, 1.03 0.49, 1.07 0.61, 1.12 0.74.
/// Below 0.85 the set stays cached; from there the share climbs (the page cache is never exactly the size it looks: the
/// PLE shard and the startup reads keep pages in it), and past ~1.3 nearly every page misses.
inline double miss_share(double r) {
    if (r <= 0.85) return 0.0;
    if (r <= 1.00) return 0.40 * (r - 0.85) / 0.15;
    if (r <= 1.12) return 0.40 + 0.35 * (r - 1.00) / 0.12;
    if (r <= 1.30) return 0.75 + 0.25 * (r - 1.12) / 0.18;
    return 1.0;
}

inline Cost price(const Host& h, const Tier& t, int64_t tokens, const Option& o) {
    Cost c;
    c.opt = o;
    c.chunks = o.chunk > 0 ? (tokens + o.chunk - 1) / o.chunk : 1;
    const uint64_t unbacked = o.lend_bytes > o.lend_kept ? o.lend_bytes - o.lend_kept : 0;
    c.file_set = (double) t.file_tier + (double) unbacked + (double) t.extra_file;
    const double r = h.page_cache > 0 ? c.file_set / (double) h.page_cache : 1e9;   // unbuffered: nothing is cached
    c.miss = h.page_cache > 0 ? miss_share(r) : 1.0;
    c.fits = h.page_cache > 0 && r <= 0.85;
    c.read_bytes = (double) c.chunks * c.file_set * c.miss;
    // a chunk streams every expert the GPU does not hold: the loan returns to that set for the length of the prompt
    const double stream = std::max(0.0, (double) t.experts_total - (double) t.gpu_cache + (double) o.lend_bytes);
    double ring_pen = 1.0;
    if (o.ring_full > 16 && o.ring < o.ring_full)
        ring_pen = 1.0 + h.ring_loss * (1.0 - (double) (o.ring - 16) / (double) (o.ring_full - 16));
    const double compute = (double) tokens / h.compute_tok_s * ring_pen;
    c.seconds = compute + (double) c.chunks * stream / (h.h2d_gbps * 1e9) + c.read_bytes / (h.ssd_gbps * 1e9);
    return c;
}

struct Pick {
    Cost best, baseline;
    bool changed = false;
};

/// The option with the least predicted time.  `baseline` (today's choice) is kept unless the pick saves at least
/// `min_gain` of its time AND at least `min_save_s` seconds: a plan that fits the page cache is never touched, and
/// near-ties keep the established behaviour.
inline Pick choose(const Host& h, const Tier& t, int64_t tokens, const std::vector<Option>& options, size_t baseline,
                   double min_gain = 0.05, double min_save_s = 2.0) {
    Pick p;
    p.baseline = price(h, t, tokens, options.at(baseline));
    p.best = p.baseline;
    for (size_t i = 0; i < options.size(); ++i) {
        if (i == baseline) continue;
        const Cost c = price(h, t, tokens, options[i]);
        if (c.seconds < p.best.seconds - 1e-9 ||
            (std::abs(c.seconds - p.best.seconds) <= 1e-9 && c.opt.chunk > p.best.opt.chunk))
            p.best = c;
    }
    p.changed = p.best.opt.chunk != p.baseline.opt.chunk || p.best.opt.ring != p.baseline.opt.ring;
    if (p.changed && !(p.baseline.seconds - p.best.seconds >= min_gain * p.baseline.seconds &&
                       p.baseline.seconds - p.best.seconds >= min_save_s)) {
        p.best = p.baseline;
        p.changed = false;
    }
    return p;
}

/// The chunk a request of `tokens` reads in when the plan chose `plan` < `cap` (the buffers' chunk, what the VRAM scan picked): a
/// prompt that fits one chunk of `cap` keeps ONE chunk (rounded to 256) - one pass reads each expert once, so it cannot thrash the
/// page cache and splitting it only adds a pass - and a longer one reads in n = ceil(tokens / plan) chunks of equal size (rounded up
/// to 256, at most `cap`), so there is never a stub of a tail.
inline int64_t balanced_chunk(int64_t tokens, int64_t cap, int64_t plan) {
    if (tokens <= 0 || cap <= 0) return 0;
    const int64_t one = (tokens + 255) / 256 * 256;
    if (plan <= 0 || plan >= cap || one <= cap) return std::min(one, cap);
    // n chunks of the plan's size or a little less (never more: the loan the plan priced stays the loan), the first size whose last
    // chunk is at least 40% of a chunk; failing that, the one with the longest last chunk
    const int64_t n = (tokens + plan - 1) / plan;
    int64_t c = std::min((tokens + n - 1) / n + 255, plan + 255) / 256 * 256, best = c;
    double best_tail = -1;
    for (int k = 0; k < 8 && c >= 256; ++k, c -= 256) {
        const int64_t nn = (tokens + c - 1) / c, tail = tokens - (nn - 1) * c;
        if (tail * 5 >= c * 2) return c;
        if ((double) tail / (double) c > best_tail) { best_tail = (double) tail / (double) c; best = c; }
    }
    return best;
}

/// One log line: what was chosen, why, and the reads it expects per prompt.
inline std::string describe(const Pick& p, const Host& h, int64_t tokens) {
    char b[768];
    const Cost& c = p.best;
    if (!p.changed) {
        std::snprintf(b, sizeof b,
                      "strata: prefill plan: chunk %lld, ring %lld kept: file set %.1f GiB %s page cache %.1f GiB -> about %.1f GB "
                      "read from the drive per %lld-token prompt (%lld chunks)%s",
                      (long long) c.opt.chunk, (long long) c.opt.ring, c.file_set / kGiB,
                      c.fits ? "fits the" : "does not fit the", (double) h.page_cache / kGiB, c.read_bytes / 1e9,
                      (long long) tokens, (long long) c.chunks,
                      c.fits ? "" : "; no option does better (the set cannot fit: fewest passes win)");
    } else {
        const Cost& o = p.baseline;
        std::snprintf(b, sizeof b,
                      "strata: prefill plan: chunk %lld -> %lld, ring %lld -> %lld: today's choice has a %.1f GiB file set "
                      "against a %.1f GiB page cache (about %.1f GB read from the drive per %lld-token prompt, %.0f s); this one "
                      "%.1f GiB, about %.1f GB read, %.0f s. --prefill N keeps your own chunk",
                      (long long) o.opt.chunk, (long long) c.opt.chunk, (long long) o.opt.ring, (long long) c.opt.ring,
                      o.file_set / kGiB, (double) h.page_cache / kGiB, o.read_bytes / 1e9, (long long) tokens, o.seconds,
                      c.file_set / kGiB, c.read_bytes / 1e9, c.seconds);
    }
    return b;
}

}  // namespace strata::program::prefill_plan
