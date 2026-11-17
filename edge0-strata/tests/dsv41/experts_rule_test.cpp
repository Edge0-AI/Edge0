// Strata. The experts-per-word rule of the DeepSeek V4.1 engine (src/dsv41/ds4_experts_rule.hpp): 5 on a card with 16 GiB or less, 6
// elsewhere, --experts overrides, DS4_TOPK over both.  No GPU needed.
#include "ds4_experts_rule.hpp"

#include <cstdio>
#include <cstdlib>

using strata::ds4::experts_per_word;
using strata::ds4::experts_test_override;
using strata::ds4::copy_all_fits;
using strata::ds4::attn_slab_rows;
using strata::ds4::staging_ring_mb;
using strata::ds4::store_gpu_shares_ram;

static int fails = 0;
#define CHECK_EQ(a, b)                                                                              \
    do {                                                                                            \
        const long long x = (a), y = (b);                                                           \
        if (x != y) {                                                                               \
            std::printf("FAIL %s:%d: %s = %lld, expected %lld\n", __FILE__, __LINE__, #a, x, y);    \
            ++fails;                                                                                \
        }                                                                                           \
    } while (0)

int main() {
    const uint64_t GiB = 1ull << 30;
    // the owner's 12 GB-class cards (RTX 5070 12227 MiB, RTX 3060 12 GB), 16 GiB cards: faster mode by default
    CHECK_EQ(experts_per_word(-1, 12227ull << 20, 6), 5);
    CHECK_EQ(experts_per_word(-1, 12 * GiB, 6), 5);
    CHECK_EQ(experts_per_word(-1, 16 * GiB, 6), 5);
    CHECK_EQ(experts_per_word(-1, 8 * GiB, 6), 5);
    // 24 GB and bigger, and several cards: the model's own 6
    CHECK_EQ(experts_per_word(-1, 16 * GiB + 1, 6), 6);
    CHECK_EQ(experts_per_word(-1, 24 * GiB, 6), 6);
    CHECK_EQ(experts_per_word(-1, 4 * 32 * GiB, 6), 6);
    CHECK_EQ(experts_per_word(-1, 2 * 12 * GiB, 6), 6);   // two 12 GB cards hold 24 GiB in all
    // a GPU that shares the PC's RAM (Strix Halo, ~91 GiB visible): faster mode whatever the size; discrete cards by size only
    CHECK_EQ(experts_per_word(-1, 91 * GiB, 6, true), 5);
    CHECK_EQ(experts_per_word(-1, 24 * GiB, 6, false), 6);   // a discrete 24 GB card
    CHECK_EQ(experts_per_word(-1, 12 * GiB, 6, false), 5);   // a discrete 12 GB card
    CHECK_EQ(experts_per_word(6, 91 * GiB, 6, true), 6);     // --experts 6 wins on the unified-memory GPU too
    // two or more RDNA4 cards that hold at least half of the experts (4 x R9700, 128 GiB in all): faster mode by default (the 5th argument)
    CHECK_EQ(experts_per_word(-1, 4 * 32 * GiB, 6, false, true), 5);
    CHECK_EQ(experts_per_word(-1, 3 * 32 * GiB, 6, false, true), 5);
    CHECK_EQ(experts_per_word(-1, 4 * 32 * GiB, 6, false, false), 6);   // the same cards when they hold under half (or are not RDNA4): the model's 6
    CHECK_EQ(experts_per_word(6, 4 * 32 * GiB, 6, false, true), 6);     // --experts 6 (the app's / Terminal's "Full quality") wins
    CHECK_EQ(experts_per_word(4, 4 * 32 * GiB, 6, false, true), 4);     // so does any other --experts N
    CHECK_EQ(experts_per_word(5, 4 * 32 * GiB, 6, false, false), 5);
    CHECK_EQ(experts_per_word(-1, 4 * 32 * GiB, 1, false, true), 1);    // a model of one expert a word has nothing to drop
    CHECK_EQ(experts_test_override(experts_per_word(-1, 4 * 32 * GiB, 6, false, true), 6), 5);
    CHECK_EQ(experts_per_word(-1, 91 * GiB, 4, true), 4);    // never raised
    // unknown memory (0): the model's count, never a silent cut
    CHECK_EQ(experts_per_word(-1, 0, 6), 6);
    // the flag decides, both ways
    CHECK_EQ(experts_per_word(6, 12 * GiB, 6), 6);
    CHECK_EQ(experts_per_word(4, 12 * GiB, 6), 4);
    CHECK_EQ(experts_per_word(5, 96 * GiB, 6), 5);
    CHECK_EQ(experts_per_word(9, 12 * GiB, 6), 6);   // never more than the model has
    CHECK_EQ(experts_per_word(1, 12 * GiB, 6), 1);
    // a model that routes to 1 (or no) expert: left alone
    CHECK_EQ(experts_per_word(-1, 12 * GiB, 1), 1);
    CHECK_EQ(experts_per_word(-1, 12 * GiB, 0), 0);
    // a model that routes to fewer than 5: the rule never raises it
    CHECK_EQ(experts_per_word(-1, 12 * GiB, 4), 4);
    // the test override wins over the rule and the flag
#if defined(_WIN32)
    _putenv("DS4_TOPK=4");
#else
    setenv("DS4_TOPK", "4", 1);
#endif
    CHECK_EQ(experts_test_override(experts_per_word(-1, 96 * GiB, 6), 6), 4);
    CHECK_EQ(experts_test_override(experts_per_word(6, 12 * GiB, 6), 6), 4);
#if defined(_WIN32)
    _putenv("DS4_TOPK=9");
#else
    setenv("DS4_TOPK", "9", 1);
#endif
    CHECK_EQ(experts_test_override(5, 6), 5);   // out of range: ignored
#if defined(_WIN32)
    _putenv("DS4_TOPK=");
#else
    unsetenv("DS4_TOPK");
#endif
    CHECK_EQ(experts_test_override(5, 6), 5);
    // attention slab rows: attn_slab_rows(env, chunk, base, spare_bytes, bytes_per_row).  bpr of this model ~ 32768 * 10 + 8 * 1024 * 6 = 377 KB.
    {
        const double bpr = 377000.0;
        CHECK_EQ(attn_slab_rows(-1, 8192, 256, 1e18, bpr), 2048);         // unset, the expert tier is off (--vram-experts 0): automatic 2048
        CHECK_EQ(attn_slab_rows(-1, 8192, 256, 4.0e9, bpr), 2048);        // 4 GB of VRAM to spare: automatic
        CHECK_EQ(attn_slab_rows(-1, 8192, 256, -5.0e9, bpr), 256);        // the 5070: every spare byte is an expert's: the 0.2.6 slab
        CHECK_EQ(attn_slab_rows(-1, 8192, 256, 0.0, bpr), 256);
        CHECK_EQ(attn_slab_rows(-1, 8192, 256, 0.9e9, bpr), 256);         // 1792 rows x 377 KB = 676 MB; needs 1.25 x + 256 MB = 1.1 GB
        CHECK_EQ(attn_slab_rows(-1, 8192, 256, 1.2e9, bpr), 2048);
        CHECK_EQ(attn_slab_rows(-1, 1024, 256, 1e18, bpr), 1024);         // a 1024 chunk caps the slab
        CHECK_EQ(attn_slab_rows(-1, 4096, 256, 1e18, bpr), 2048);
        CHECK_EQ(attn_slab_rows(-1, 128, 64, 1e18, bpr), 128);
        CHECK_EQ(attn_slab_rows(-1, 8192, 2560, 1e18, bpr), 2560);        // a base already above the automatic size is kept
        CHECK_EQ(attn_slab_rows(0, 8192, 256, 1e18, bpr), 256);           // DS4_ATTN_SLAB=0: the 0.2.6 rule, whatever the spare
        CHECK_EQ(attn_slab_rows(1024, 8192, 256, -5.0e9, bpr), 1024);     // N: that many rows, the spare is the user's call
        CHECK_EQ(attn_slab_rows(100, 8192, 256, 1e18, bpr), 64);          // rounded down to 64, at least 64
        CHECK_EQ(attn_slab_rows(99999, 8192, 256, 1e18, bpr), 8192);      // at most the chunk
        CHECK_EQ(attn_slab_rows(300, 8192, 256, 1e18, bpr), 256);
        // the cards hold half the experts or more (expert_share, the 6th argument): the whole small chunk's slab, min(chunk, 1024) rows, whatever
        // the spare (the 4 x R9700 cell: chunk 1024, share 0.77, spare negative: 256 -> 1024 rows)
        CHECK_EQ(attn_slab_rows(-1, 1024, 256, -5.0e9, bpr, 0.77), 1024);
        CHECK_EQ(attn_slab_rows(-1, 1024, 256, -5.0e9, bpr, 0.50), 1024);
        CHECK_EQ(attn_slab_rows(-1, 1024, 256, -5.0e9, bpr, 0.49), 256);   // under half: the base, as before
        CHECK_EQ(attn_slab_rows(-1, 512, 256, -5.0e9, bpr, 0.77), 512);   // a chunk under 1024 caps it
        CHECK_EQ(attn_slab_rows(-1, 8192, 256, -5.0e9, bpr, 0.77), 1024);   // a big chunk: still 1024 rows without the room for more
        CHECK_EQ(attn_slab_rows(-1, 8192, 256, 4.0e9, bpr, 0.77), 2048);   // the room is there too: the automatic growth wins
        CHECK_EQ(attn_slab_rows(-1, 8192, 256, 1e18, bpr, 1.0), 2048);     // everything fits (a big card): 2048 as without the share
        CHECK_EQ(attn_slab_rows(-1, 8192, 2560, -5.0e9, bpr, 0.77), 2560);   // never below the base
        CHECK_EQ(attn_slab_rows(0, 1024, 256, 1e18, bpr, 0.77), 256);      // DS4_ATTN_SLAB=0: the 0.2.6 rule whatever the share
        CHECK_EQ(attn_slab_rows(512, 1024, 256, 1e18, bpr, 0.77), 512);    // DS4_ATTN_SLAB=N: that many rows whatever the share
        CHECK_EQ(attn_slab_rows(-1, 1024, 1024, -5.0e9, bpr, 0.77), 1024);   // a unified-memory base (32 MB of q = 1024 rows) is kept
    }
    // the staging ring (MB): staging_ring_mb(share, room_bytes, n_dev, integrated, hip)
    {
        const double GB = 1e9;
        // the cards hold the experts: 16 MB, discrete (4 x R9700 at 0.86 and a card that holds everything)
        CHECK_EQ(staging_ring_mb(0.86, 4 * 30 * GB, 4, false, true), 16);
        CHECK_EQ(staging_ring_mb(1.00, 70 * GB, 1, false, true), 16);
        CHECK_EQ(staging_ring_mb(0.82, 4 * 30 * GB, 4, false, true), 16);
        // they hold most of them (4 x R9700, share 0.72-0.77, 31 GB of room a device): 8 MB a GB of one device's room = 248 MB
        CHECK_EQ(staging_ring_mb(0.77, 4 * 31 * GB, 4, false, true), 248);
        CHECK_EQ(staging_ring_mb(0.72, 4 * 31 * GB, 4, false, true), 248);
        CHECK_EQ(staging_ring_mb(0.81, 4 * 31 * GB, 4, false, true), 248);
        CHECK_EQ(staging_ring_mb(0.77, 4 * 10 * GB, 4, false, true), 128);       // small devices: at least 128
        CHECK_EQ(staging_ring_mb(0.60, 3 * 90 * GB, 3, false, true), 512);       // big devices: at most 512
        CHECK_EQ(staging_ring_mb(0.50, 2 * 31 * GB, 2, false, true), 248);       // the per-device room, not the summed one (1 card: 31 GB -> 248 too)
        CHECK_EQ(staging_ring_mb(0.60, 31 * GB, 1, false, true), 248);
        // they hold under half (discrete): 64 MB a GB of the summed room, 400..1200 (streamed; not measured on a discrete card)
        CHECK_EQ(staging_ring_mb(0.19, 31 * GB, 1, false, true), 1200);
        CHECK_EQ(staging_ring_mb(0.06, 10 * GB, 1, false, true), 640);           // a 12 GB card
        CHECK_EQ(staging_ring_mb(0.06, 4 * GB, 1, false, true), 400);            // a small pool still gets 400
        CHECK_EQ(staging_ring_mb(0.40, 2 * 31 * GB, 2, false, true), 1200);      // capped
        CHECK_EQ(staging_ring_mb(0.49, 4 * 31 * GB, 4, false, true), 1200);
        // a unified-memory GPU (Strix Halo): never the small ring; the streamed 400..1200 MB under 1/1.3, 16 above
        CHECK_EQ(staging_ring_mb(0.60, 40 * GB, 1, true, true), 1200);           // share 0.6 would be the small ring on a discrete card
        CHECK_EQ(staging_ring_mb(0.30, 18.75 * GB, 1, true, true), 1200);
        CHECK_EQ(staging_ring_mb(0.30, 12 * GB, 1, true, true), 768);
        CHECK_EQ(staging_ring_mb(0.30, 5 * GB, 1, true, true), 400);
        CHECK_EQ(staging_ring_mb(0.77, 40 * GB, 1, true, true), 16);             // main's "fit" threshold, 1/1.3 = 0.769
        CHECK_EQ(staging_ring_mb(0.76, 40 * GB, 1, true, true), 1200);
        // nothing known: no devices
        CHECK_EQ(staging_ring_mb(0.0, 0.0, 0, false, true), 400);
        // a system that is not RDNA4 keeps main's rule everywhere: 16 MB from 1/1.3, the streamed ring below it (no small band, no 0.82)
        CHECK_EQ(staging_ring_mb(0.77, 4 * 31 * GB, 4, false, false), 16);
        CHECK_EQ(staging_ring_mb(0.60, 4 * 31 * GB, 4, false, false), 1200);
        CHECK_EQ(staging_ring_mb(0.80, 4 * 31 * GB, 4, false, false), 16);
        CHECK_EQ(staging_ring_mb(0.06, 10 * GB, 1, false, false), 640);
        CHECK_EQ(staging_ring_mb(0.30, 12 * GB, 1, true, false), 768);
    }
    // prompt chunk: prefill_chunk_default(own_default, gpu_share, gguf_copy, file_experts, hip, min_vram, big=8192, multi_tp=false).  What `strata launch` runs on one card
    // passes neither --tp nor --threads, but the engine's default tp is 2, so own_default is false there; a launch with several cards passes --tp 1.
    using strata::ds4::prefill_chunk_default;
    // RTX 5070 / RTX 3060 (12 GB), launch default: experts in place on a card under 16 GiB -> 4096 (their logs said 8192 before this rule)
    CHECK_EQ(prefill_chunk_default(false, 0.06, false, true, false, 12227ull << 20), 4096);
    CHECK_EQ(prefill_chunk_default(false, 0.06, false, true, false, 12 * GiB), 4096);
    // the same cards with --tp 1 on one card: share 0.06 < 0.5 -> the same 4096
    CHECK_EQ(prefill_chunk_default(true, 0.06, false, true, false, 12 * GiB), 4096);
    // 16 GB cards (15.9 GiB total) 4096; a card with exactly 16 GiB or more keeps the big chunk; short VRAM (8 GB) 4096; unknown VRAM the big one
    CHECK_EQ(prefill_chunk_default(false, 0.10, false, true, false, 16 * GiB - 1), 4096);
    CHECK_EQ(prefill_chunk_default(false, 0.10, false, true, false, 16 * GiB), 8192);
    CHECK_EQ(prefill_chunk_default(false, 0.10, false, true, false, 24 * GiB), 8192);
    CHECK_EQ(prefill_chunk_default(false, 0.0, false, true, false, 8 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(false, 0.0, false, true, false, 0), 8192);
    // a 12 GB HIP card is not measured: the big chunk
    CHECK_EQ(prefill_chunk_default(false, 0.06, false, true, true, 12 * GiB), 8192);
    // a 5090-class card (32 GB, share 0.19): in place 8192; with the experts copied into RAM 4096
    CHECK_EQ(prefill_chunk_default(false, 0.19, false, true, false, 32 * GiB), 8192);
    CHECK_EQ(prefill_chunk_default(false, 0.19, true, true, false, 32 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(true, 0.19, true, true, false, 32 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(false, 0.10, true, true, false, 16 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(false, 0.05, true, true, false, 24 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(false, 0.06, true, true, false, 12227ull << 20), 4096);
    // copied into RAM on a card under 10 GiB (short VRAM): not the copy rule; the in-place rule for a small card gives 4096, unknown VRAM 8192
    CHECK_EQ(prefill_chunk_default(false, 0.0, true, true, false, 8 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(false, 0.0, true, true, false, 0), 8192);
    // 1 x R9700 (HIP, 32 GB, share 0.19), single-card default run (tp 1 passed by a multi-card box): 8192, also when copied (HIP is not measured at 4096)
    CHECK_EQ(prefill_chunk_default(true, 0.19, false, true, true, 32 * GiB), 8192);
    CHECK_EQ(prefill_chunk_default(true, 0.19, true, true, true, 32 * GiB), 8192);
    // 4 x R9700 (share 0.86), single-card default run (--tp 1 --threads off): the small chunk
    CHECK_EQ(prefill_chunk_default(true, 0.86, false, true, true, 32 * GiB), 1024);
    CHECK_EQ(prefill_chunk_default(true, 0.50, false, true, true, 32 * GiB), 1024);
    CHECK_EQ(prefill_chunk_default(true, 0.49, false, true, true, 32 * GiB), 8192);
    // a multi-card tp run (own_default false, multi_tp: RDNA4, 2+ cards, no --threads on; the engine's default tp 2, or --tp 2 from the launcher) takes
    // the small chunk too: the 4 x R9700 default-tp datapoint (8K warm 1247 ppts at 1024 against 690)
    CHECK_EQ(prefill_chunk_default(false, 0.86, false, true, true, 32 * GiB, 8192, true), 1024);
    CHECK_EQ(prefill_chunk_default(false, 0.50, false, true, true, 32 * GiB, 8192, true), 1024);
    CHECK_EQ(prefill_chunk_default(false, 0.49, false, true, true, 32 * GiB, 8192, true), 8192);
    // one card with the default tp, and `--threads on` (neither own_default nor multi_tp) keep their defaults whatever the share: 8192
    // streamed, 2048 for a pack that is not a GGUF, 4096 copied into RAM on a CUDA card
    CHECK_EQ(prefill_chunk_default(false, 0.86, false, true, true, 32 * GiB), 8192);
    CHECK_EQ(prefill_chunk_default(false, 1.00, false, true, true, 80 * GiB), 8192);
    CHECK_EQ(prefill_chunk_default(false, 1.00, true, true, false, 80 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(false, 0.86, false, false, true, 32 * GiB), 2048);
    CHECK_EQ(prefill_chunk_default(false, 0.30, false, false, false, 24 * GiB), 2048);
    // the rule takes no --max-context: a 128K session on a 12 GB card gets the same chunk, because the chunk's buffers and the KV are
    // allocated whole at load for that context and a chunk that does not fit there is halved before the first prompt (4096 -> 2048 -> 1024 ...)
    // copy_all_fits: the 251 GB R9700 box (108 GB lane) copies; a 64 GB PC, a 125 GB box with the 2.25 pack's lane, an unknown RAM do not
    {
        const double GB = 1e9, margin = 8.0 * (1ull << 30);
        CHECK_EQ(copy_all_fits(108 * GB, (uint64_t) (236 * GB), margin, 1e300), 1);
        CHECK_EQ(copy_all_fits(108 * GB, (uint64_t) (110 * GB), margin, 1e300), 0);   // inside the margin
        CHECK_EQ(copy_all_fits(108 * GB, (uint64_t) (116 * GB), margin, 1e300), 0);   // 108 + 8.6 > 116
        CHECK_EQ(copy_all_fits(108 * GB, (uint64_t) (118 * GB), margin, 1e300), 1);
        CHECK_EQ(copy_all_fits(108 * GB, (uint64_t) (101 * GB), margin, 1e300), 0);   // the 125 GB box, free RAM 101 GB
        CHECK_EQ(copy_all_fits(92 * GB, ~uint64_t{0}, margin, 1e300), 0);             // RAM not known
        CHECK_EQ(copy_all_fits(0, (uint64_t) (236 * GB), margin, 1e300), 0);          // nothing on the CPU lane
        // Windows: the commit room (RAM + page file beside its margin) must hold the lane too, whatever the free RAM says
        CHECK_EQ(copy_all_fits(108 * GB, (uint64_t) (236 * GB), margin, 90 * GB), 0);
        CHECK_EQ(copy_all_fits(108 * GB, (uint64_t) (236 * GB), margin, 108 * GB), 1);
        // the owner-PC class: 64 GB RAM, a 12 GB card (a lane of ~120 GB), 40 GB free, a 90 GB commit limit
        CHECK_EQ(copy_all_fits(120 * GB, (uint64_t) (40 * GB), margin, 60 * GB), 0);
        // a 192 GB Windows PC with a 4 GB page file: 150 GB free RAM but only 100 GB of commit room: no
        CHECK_EQ(copy_all_fits(108 * GB, (uint64_t) (150 * GB), margin, 100 * GB), 0);
    }
    // ---- DS4_DEFAULTS=026: every rule gives v0.2.6's answer (the last argument v026 = true); an explicit variable / flag still wins; the six old
    // DS4_*=0 switches keep working with v026 = false
    {
        using namespace strata::ds4;
        const double GB = 1e9;
        CHECK_EQ(is_026_token("026"), 1);
        CHECK_EQ(is_026_token("0.2.6"), 1);
        CHECK_EQ(is_026_token("v0.2.6"), 1);
        CHECK_EQ(is_026_token("0"), 0);
        CHECK_EQ(is_026_token(""), 0);
        CHECK_EQ(is_026_token("027"), 0);
        CHECK_EQ(is_026_token(nullptr), 0);
        // the 5-experts default: 4 x R9700 goes back to 6; a small card and an integrated GPU were Faster in 0.2.6 already; --experts wins
        CHECK_EQ(experts_per_word(-1, 4 * 32 * GiB, 6, false, true, true), 6);
        CHECK_EQ(experts_per_word(-1, 4 * 32 * GiB, 6, false, true, false), 5);
        CHECK_EQ(experts_per_word(-1, 12 * GiB, 6, false, false, true), 5);
        CHECK_EQ(experts_per_word(-1, 91 * GiB, 6, true, false, true), 5);
        CHECK_EQ(experts_per_word(5, 4 * 32 * GiB, 6, false, true, true), 5);
        // the chunk rule: the multi-card tp run loses the 1024; a --tp 1 run keeps it (0.2.6 had it); an explicit --prefill-chunk is not this rule
        CHECK_EQ(prefill_chunk_default(false, 0.86, false, true, true, 32 * GiB, 8192, true, true), 8192);
        CHECK_EQ(prefill_chunk_default(false, 0.86, false, true, true, 32 * GiB, 8192, true, false), 1024);
        CHECK_EQ(prefill_chunk_default(true, 0.86, false, true, true, 32 * GiB, 8192, true, true), 1024);
        CHECK_EQ(prefill_chunk_default(false, 0.30, false, false, false, 24 * GiB, 8192, true, true), 2048);
        // the slab: the base 8 MB everywhere, no growth, no whole-chunk slab; DS4_ATTN_SLAB=N (env N) and =0 still do what they say
        CHECK_EQ(attn_slab_base_mb(true, 0, true), 8);
        CHECK_EQ(attn_slab_base_mb(true, 0, false), 32);
        CHECK_EQ(attn_slab_base_mb(false, 0, false), 8);
        CHECK_EQ(attn_slab_base_mb(false, 0, true), 8);
        CHECK_EQ(attn_slab_base_mb(true, 16, true), 16);                   // STRATA_DS4_A_MB wins
        CHECK_EQ(attn_slab_rows(-1, 8192, 256, 1e18, 377000.0, 1.0, true), 256);      // spare room: the 0.2.6 slab
        CHECK_EQ(attn_slab_rows(-1, 1024, 256, -5.0e9, 377000.0, 0.77, true), 256);   // the 4 x R9700 whole-chunk slab: off
        CHECK_EQ(attn_slab_rows(1024, 1024, 256, -5.0e9, 377000.0, 0.77, true), 1024);   // DS4_ATTN_SLAB=1024 wins
        CHECK_EQ(attn_slab_rows(0, 8192, 256, 1e18, 377000.0, 1.0, true), 256);
        CHECK_EQ(attn_slab_rows(0, 8192, 256, 1e18, 377000.0, 1.0, false), 256);       // the old DS4_ATTN_SLAB=0 switch
        // the staging ring: 16 MB where the cards hold the experts (>= 1/1.3), a fixed 400 MB below, whatever the platform
        CHECK_EQ(staging_ring_mb(0.86, 4 * 30 * GB, 4, false, true, true), 16);
        CHECK_EQ(staging_ring_mb(0.77, 4 * 31 * GB, 4, false, true, true), 16);
        CHECK_EQ(staging_ring_mb(0.72, 4 * 31 * GB, 4, false, true, true), 400);
        CHECK_EQ(staging_ring_mb(0.06, 10 * GB, 1, false, false, true), 400);
        CHECK_EQ(staging_ring_mb(0.30, 18.75 * GB, 1, true, false, true), 400);        // Strix Halo: not 1200
        CHECK_EQ(staging_ring_mb(0.0, 0.0, 0, false, false, true), 400);
        // the pinned copy and the huge-page arena: opt-in again; =1 / =0 win
        CHECK_EQ(copy_pin_default(-1, true, true, 0.8, true, true), 0);
        CHECK_EQ(copy_pin_default(-1, true, true, 0.8, true, false), 1);
        CHECK_EQ(copy_pin_default(1, false, false, 0.0, false, true), 1);
        CHECK_EQ(copy_pin_default(0, true, true, 0.8, true, false), 0);               // the old DS4_COPY_PIN=0
        CHECK_EQ(copy_pin_default(-1, false, true, 0.5, true, false), 1);
        CHECK_EQ(copy_pin_default(-1, false, true, 0.49, true, false), 0);
        CHECK_EQ(copy_pin_default(-1, false, false, 0.9, true, false), 0);
        CHECK_EQ(copy_pin_default(-1, true, false, 0.0, false, false), 0);            // nothing to pin
        CHECK_EQ(copy_hugetlb_default(-1, 120 * GB, 100 * GB, true), 0);
        CHECK_EQ(copy_hugetlb_default(-1, 120 * GB, 100 * GB, false), 1);
        CHECK_EQ(copy_hugetlb_default(-1, 90 * GB, 100 * GB, false), 0);
        CHECK_EQ(copy_hugetlb_default(1, 0.0, 100 * GB, true), 1);
        CHECK_EQ(copy_hugetlb_default(0, 120 * GB, 100 * GB, false), 0);              // the old DS4_COPY_HUGETLB=0
        // the matrix-core attention: opt-in again, except on gfx12 and (since 2026-10-28) on gfx1151 on Windows
        CHECK_EQ(fast_attn_default(-1, true, false, true), 0);
        CHECK_EQ(fast_attn_default(-1, true, false, false), 1);
        CHECK_EQ(fast_attn_default(-1, false, false, false), 0);
        CHECK_EQ(fast_attn_default(-1, false, true, false), 1);                       // gfx1151 on Windows
        CHECK_EQ(fast_attn_default(-1, false, true, true), 0);                        // ... but never for 0.2.6's arithmetic
        CHECK_EQ(fast_attn_default(1, false, false, true), 1);
        CHECK_EQ(fast_attn_default(0, true, true, false), 0);                         // the old DS4_FAST_ATTN=0
        // the drafter's cumulative cut: 0 again; STRATA_DS4_DSPARK_CUM wins
        CHECK_EQ((int) (dspark_cum_value(false, 0.0f, true, true) * 100), 0);
        CHECK_EQ((int) (dspark_cum_value(false, 0.0f, true, false) * 100), 50);
        CHECK_EQ((int) (dspark_cum_value(false, 0.0f, false, false) * 100), 0);
        CHECK_EQ((int) (dspark_cum_value(true, 0.7f, true, true) * 100), 70);
        CHECK_EQ((int) (dspark_cum_value(true, 0.0f, true, false) * 100), 0);
        // the store: spare slots and put off, the post-prompt settle back on (and without the copy-mode skip); the old =0 switches keep working
        CHECK_EQ(store_feature_default(-1, true), 0);
        CHECK_EQ(store_feature_default(-1, false), 1);
        CHECK_EQ(store_feature_default(1, true), 1);
        CHECK_EQ(store_feature_default(0, false), 0);                                  // the old DS4_STORE_SPARE=0 / DS4_STORE_PUT=0
        CHECK_EQ((int) settle_store_default(true), 1);
        CHECK_EQ((int) settle_store_default(false), 0);
        CHECK_EQ(settle_skips_copy_experts(true), 0);
        CHECK_EQ(settle_skips_copy_experts(false), 1);
        // the dense dequantizer, the head, the drafter's markov head, the F16 operand, the RDNA3 GEMM
        CHECK_EQ(dq_old_default(-1, false, true), 1);
        CHECK_EQ(dq_old_default(-1, false, false), 0);
        CHECK_EQ(dq_old_default(-1, true, false), 1);
        CHECK_EQ(dq_old_default(0, true, true), 0);                                    // DS4_DQ_OLD=0 wins
        CHECK_EQ(head_per_default(true), 4);
        CHECK_EQ(head_per_default(false), 8);
        CHECK_EQ(markov_head_q8_default(true), 0);
        CHECK_EQ(markov_head_q8_default(false), 1);
        CHECK_EQ(gate_up_x16_default(true), 0);
        CHECK_EQ(gate_up_x16_default(false), 1);
        CHECK_EQ(dense_pfgemm_default(-1, true), 0);
        CHECK_EQ(dense_pfgemm_default(-1, false), 1);
        CHECK_EQ(dense_pfgemm_default(1, true), 1);
        CHECK_EQ(dense_pfgemm_default(0, false), 0);
    }
    // #4147: the auto store's "the GPU shares the RAM" decision.  A discrete 2 x RTX 3090 whose pinned buffers took 7.5 GB of the free RAM is NOT
    // unified memory (the 256K store stayed 48 GB); a Strix Halo / GB10 (integrated) whose weights took RAM is, as before.
    {
        const uint64_t G = 1000000000ull;
        CHECK_EQ(store_gpu_shares_ram(false, 62 * G, 54 * G), 0);          // discrete, free RAM fell 8 GB: no
        CHECK_EQ(store_gpu_shares_ram(false, 62 * G, 20 * G), 0);          // discrete, even a 42 GB drop: no
        CHECK_EQ(store_gpu_shares_ram(true, 62 * G, 54 * G), 1);           // integrated, 8 GB drop: yes (unchanged)
        CHECK_EQ(store_gpu_shares_ram(true, 62 * G, 58 * G), 1);           // exactly the 4 GB threshold
        CHECK_EQ(store_gpu_shares_ram(true, 62 * G, 58 * G + 1), 0);       // just under it
        CHECK_EQ(store_gpu_shares_ram(true, 54 * G, 62 * G), 0);           // RAM grew: no
        CHECK_EQ(store_gpu_shares_ram(true, ~uint64_t{0}, 40 * G), 0);     // the first estimate unknown
        CHECK_EQ(store_gpu_shares_ram(true, 62 * G, ~uint64_t{0}), 0);
    }
    if (fails == 0) std::printf("experts rule: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
