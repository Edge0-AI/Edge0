// Strata. Experts per word: the "faster mode" rule of the DeepSeek V4.1 engine.  Header only, no GPU code (tests/dsv41/experts_rule_test.cpp).
//
// The model routes every word to its top 6 experts.  On a PC with a small graphics card the CPU lane and the SSD set the speed, and
// each expert dropped saves their bytes: the router's top N are kept and their weights renormalised (the same selection the full
// model makes, a shorter list).  The default is 5 where the graphics memory is 16 GiB or less, or the GPU shares the PC's RAM, the model's own count elsewhere;
// --experts N says it outright (--experts 6: full quality), DS4_TOPK=N is the test override over everything.
#pragma once

#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace strata::ds4 {

// DS4_DEFAULTS=026: the master switch.  Every default the DeepSeek engine changed since v0.2.6 goes back to what v0.2.6 did, so "off" in the app's
// "DeepSeek speed-ups" really is version 0.2.6.  Each rule below takes `v026` (the engine passes defaults_026()); an explicit per-feature variable or
// flag (DS4_ATTN_SLAB=N, DS4_COPY_PIN=1, DS4_FAST_ATTN=1, --experts N, --prefill-chunk N, ...) still wins over it.  The list, from
// `git diff v0.2.6..f-027-int2 -- src/dsv41`: the staging ring, the prompt chunk rule, the attention slab (base and growth), the pinned copy,
// the huge-page arena, the matrix-core attention, the drafter's cumulative cut, the 5-experts default, the store's spare slots and put, the
// store settle after a prompt, the dense dequantizer, the F16 gate/up operand, the markov head in Q8, the head's column count, the RDNA3 WMMA GEMM.
inline bool is_026_token(const char* v) {
    return v != nullptr && (std::strcmp(v, "026") == 0 || std::strcmp(v, "0.2.6") == 0 || std::strcmp(v, "v0.2.6") == 0);
}
inline bool defaults_026() {
    static const bool on = is_026_token(std::getenv("DS4_DEFAULTS"));
    return on;
}

constexpr uint64_t kFasterModeVramMax = 16ull << 30;   // GiB of graphics memory in all, at most
constexpr int kFasterModeExperts = 5;

// requested: --experts (-1: not given, the rule decides); vram_bytes: the total memory of the cards the engine uses;
// model_k: the experts a word takes in the model; integrated: a GPU that shares the PC's RAM (Strix Halo: part of the experts live only on
// the SSD, each expert dropped is less read - measured 5 against 6 on a Radeon 8065S: decode +11..13%, prompts +5..7%).  The result is in 1 .. model_k.
// multi_rdna4: two or more RDNA4 cards (gfx1200 / gfx1201: R9700, RX 9000) that together hold at least half of the routed experts - the 4 x R9700 box
// the Marlin stream measured: 5 against 6 experts, 2.25 bpw, 8K prompt 1339 -> 1419 ppts (+6%) and decode 78.1 -> 81.2 tok/s (+4%), reference KL
// 0.10987 against 0.10833 and top-1 87.59% against 87.53% (the gate is 0.006).  Faster mode there too, by default.
inline int experts_per_word(int requested, uint64_t vram_bytes, int model_k, bool integrated = false, bool multi_rdna4 = false,
                            bool v026 = false) {
    if (model_k <= 1) return model_k;
    if (v026) multi_rdna4 = false;   // 0.2.6: Faster by default on a small card or an integrated GPU only
    int n = requested;
    if (n < 1) n = integrated || multi_rdna4 || (vram_bytes != 0 && vram_bytes <= kFasterModeVramMax) ? kFasterModeExperts : model_k;
    if (n > model_k) n = model_k;
    return n < 1 ? 1 : n;
}

// Prompt chunk (--prefill-chunk) default, one rule decided once at load from bytes (no timing).  The chunk's buffers come out of the VRAM the
// expert tier would have, and every chunk streams the experts the cards do not hold once, so a bigger chunk is fewer passes.  Measured (8K prompt):
//   - the GPUs hold most of the experts (4 x R9700, share 0.86): 1024 370 tok/s, 8192 214  -> small chunk;
//   - the experts are read in place and the CPU lane streams most of them (RTX 5070 / 3060 12 GB, 1 x R9700): 8192 best (5070 244 against 186
//     at 1024, 3060 228 against 175, 1 x R9700 431 against 200) -> the big chunk (the GGUF pack's long-standing default);
//   - the experts copied into RAM (--mmap-experts off): H100 32 GiB 2048 502, 4096 538, 8192 495 -> 4096 (not measured on HIP: no).
// Reference KL against the original model does not move with the chunk (1024 0.1705, 2048 0.1692, 4096 0.1688, 8192 0.1695).
// Alloc failure halves the chunk at load (4096 -> 2048 -> 1024 ...), KV and score buffers are allocated whole at the full --max-context.
constexpr uint64_t kChunk4096MinVram = 10ull << 30;
constexpr uint64_t kChunkInPlace4096Below = 16ull << 30;
// own_default: a run with tp 1 and no --threads (`strata launch` writes --tp 1 for 4 or more AMD cards); multi_tp: two or more cards, tp above 1
// and no `--threads on` (the engine's own default tp is 2, and `strata launch` writes --tp 2 for 2-3 cards); gpu_share: the share of the
// routed experts the cards can hold (1 = all);
// gguf_copy: a GGUF pack with --mmap-experts off; file_experts: the experts stay in the files; hip: a HIP build; min_vram_bytes: the
// smallest card; big: the chunk of the streamed case (the engine's kVlRows, 8192)
inline int prefill_chunk_default(bool own_default, double gpu_share, bool gguf_copy, bool file_experts, bool hip, uint64_t min_vram_bytes,
                                 int big = 8192, bool multi_tp = false, bool v026 = false) {
    if (v026) multi_tp = false;   // 0.2.6: the 1024 chunk for a --tp 1 run only
    // The cards hold most of the experts: the small chunk is the measured one.  Two datapoints, both on 4 x R9700: with `--tp 1` (what
    // `strata launch` writes for 4 or more AMD cards), share 0.86: 1024 370 against 214 tok/s at 8192; and with the engine's own default tp
    // (2, pairs of cards, nothing written), where the rule used to fall through to 2048/8192: 2.25 bpw, --dspark auto, ids 5705e1164a, 8K
    // warm prompts 1024 -> 1247 ppts against 690 at 8192, decode unchanged.  So a multi-card tp run takes it too.  One card with the default
    // tp, `--threads on` (it stalled at the first 8K prompt in the 4-card runs) and a system that is not RDNA4 keep their old defaults: not measured with 1024.
    // An explicit --prefill-chunk N wins over all of this (the call site takes the rule only when none was given).
    if ((own_default || multi_tp) && gpu_share >= 0.5) return 1024;
    if (gguf_copy && !hip && min_vram_bytes >= kChunk4096MinVram) return 4096;
    // the experts read in place on a card under 16 GiB (the 12 GB PCs; not measured on HIP): 4096 keeps ~120 more experts resident than 8192 (RTX 5070
    // 287 against 165, RTX 3060 386 against 277) - 5 interleaved pairs, 4096 against 8192, decode: 5070 agent x1.020 (5/5) code x1.004 (3/5), 3060
    // agent x1.018 (4/5) code x1.091 (5/5); an agent turn of 3K new tokens on a 16K context takes the same time (5070 18.9-19.2 s against 19.1-19.4 s,
    // 3060 21.4 s against 20.9-21.4 s) - for 9% less speed on a cold 8K prompt (5070 223 against 244 tok/s)
    if (file_experts && !hip && min_vram_bytes != 0 && min_vram_bytes < kChunkInPlace4096Below) return 4096;
    return file_experts ? big : 2048;
}

// --mmap-experts auto copies all the CPU lane's experts into RAM at load (no demand store) only when they fit: lane_bytes of experts,
// ram_avail the RAM this process can still get (~0: unknown -> no), ram_margin left to the rest, commit_room what Windows still lets it
// commit beside its margin (1e300: no such limit, Linux).  All or none: a lane that does not fit both leaves the demand store as it was.
// Measured 1 x R9700 + 251 GB, 2.00 pack: decode +10-25% (the copies sit on huge pages, the store's page-locked arena does not).
inline bool copy_all_fits(double lane_bytes, uint64_t ram_avail, double ram_margin, double commit_room) {
    if (!(lane_bytes > 0.0) || ram_avail == ~uint64_t{0}) return false;
    return lane_bytes + ram_margin <= (double) ram_avail && lane_bytes <= commit_room;
}

// Whether the GPU's allocations come out of the same RAM the auto store would use (Strix Halo, DGX Spark's GB10): only a device the driver reports as
// integrated, AND the free RAM fell by `uma_drop_gb` or more between the first store estimate (before the weights went to the GPU) and now.
// A discrete card (2 x RTX 3090, 2 x A6000) never counts, however much RAM the pinned staging buffers, the page cache and the other programs took
// meanwhile: v0.2.5..0.2.7 read every drop of 4 GB as unified memory and shrank a 256K context's store by 27 GB (issue #4147, 48 -> 17.8 GB).
// avail_before / avail_now: free RAM in bytes (~0: unknown -> no).
inline bool store_gpu_shares_ram(bool any_integrated_device, uint64_t avail_before, uint64_t avail_now, double uma_drop_gb = 4.0) {
    if (!any_integrated_device || avail_before == ~uint64_t{0} || avail_now == ~uint64_t{0}) return false;
    return avail_before > avail_now && (double) (avail_before - avail_now) / 1e9 >= uma_drop_gb;
}

// The chunked prompt's attention slab (rows a pass: the dense projections q_b / wo_a / wo_b are dequantized and multiplied once a slab, so
// a 256-row slab dequantizes them 32 times for an 8K prompt).  `base`: the 0.2.6 rule (8 MB of rows / heads width, 256 on this model).
// DS4_ATTN_SLAB=N (`env` >= 0): 0 gives the 0.2.6 rule back, N > 0 asks for that many rows (64..chunk, a multiple of 64).  Unset (`env` < 0):
// automatic - up to 2048 rows, but only where the bigger buffers take VRAM no expert would have used: `spare_bytes` is the card's room
// beyond what its expert tier can use (huge when the tier is off, e.g. --vram-experts 0), `bytes_per_row` the buffers' cost of one row.
// The answers do not change with the slab (a row's arithmetic is its own), but the experts a card holds do: measured 2026-10-26 on the RTX
// 5070 (12 GB, 2.25 bpw, 8K prompt): 1024 rows +2.0% prompt, 2048 rows +2.2%, at 29 / 83 fewer experts in VRAM, a -3% decode and different
// greedy ids (placement); so a card whose room is all experts keeps the 0.2.6 slab, and the automatic size is for cards with VRAM to spare.
// `expert_share` (the share of the routed experts the cards hold, 0 when unknown): the second measured case.  Where the cards hold at least
// half the experts the whole small chunk's slab is worth its VRAM: 4 x R9700 (2.25 bpw, share 0.77, chunk 1024), 8K warm 1204 / 1236 / 1250
// ppts with the 256-row rule against 1324 / 1353 / 1363 with a 1024-row slab (+9%), decode 77.7 against 78.1 tok/s, 268 MB a device (1% of
// the card).  That branch asks for min(chunk, 1024) rows whatever the spare room (the spare of such a box is negative: its room is all
// experts), and never less than the base; the automatic growth on top of it still applies where the room is there.  A unified-memory GPU
// (the caller passes env 0 there) and DS4_ATTN_SLAB=0 never take either branch.  The caller passes 0 for `expert_share` in a build that is not
// HIP: that branch is the 4 x R9700 measurement.
inline int attn_slab_rows(int env, int chunk, int base, double spare_bytes, double bytes_per_row, double expert_share = 0.0,
                          bool v026 = false) {
    auto clampn = [&](int a) { return std::max(64, std::min(chunk, a) / 64 * 64); };
    if (v026 && env < 0) env = 0;   // 0.2.6: DS4_ATTN_SLAB unset gave the base; DS4_ATTN_SLAB=N still asks for N rows
    if (env == 0) return base;
    if (env > 0) return clampn(env);
    int rows = base;
    if (expert_share >= 0.5) rows = std::max(rows, clampn(1024));
    const int target = clampn(2048);
    if (target > rows) {
        const double extra = (double) (target - base) * bytes_per_row;
        if (spare_bytes >= extra * 1.25 + (256.0 * 1048576.0)) rows = target;
    }
    return rows;
}

// The prompt path's expert staging ring (MB).  The prompt streams the experts the cards do not hold through it.  Three measured cases and
// the rest left as it was:
//   - the cards hold the experts (share >= 0.82): 16 MB.  4 x R9700, share 0.86 (the 2.00 pack): a bigger ring costs decode after a prompt
//     (-1.8% at 16 MB against none, -4.4% at 64, -4.6% at 400); everything fitting needs no ring at all;
//   - they hold most of them (0.5 <= share < 0.82, discrete cards): a small ring from each DEVICE's room, 8 MB a GB of it, 128..512 MB
//     (4 x R9700, 2.25 bpw, share 0.72-0.77, chunk 1024, 8K prompt + six 128-token decodes a load: 2000 MB 1243 ppts / 73.2 tok/s, 1000 MB
//     1287 / 77.5, 500 MB 1314 / 78.5, 256 MB 1324 / 78.4, 128 MB 1338 / 78.3; the room per device there is 31 GB -> 248 MB);
//   - a unified-memory GPU (Strix Halo) that holds under 1/1.3 of them, and every discrete card that holds under half: a share of the
//     summed room, 64 MB a GB, 400..1200 MB.  Measured on Strix Halo only (400 -> 1200: the queue 3,200 -> 8,087 experts, 4K prompt 27.3
//     -> 21.3 s, decode unchanged); a discrete card in this band is not measured and takes the same, VRAM-bounded, rule.
// A unified-memory GPU never takes the small ring (its datapoint is the 1200 MB one): it keeps the 16 MB "fit" rule at 1/1.3.
// `hip`: the 0.5..0.82 band and the 0.82 threshold are the 4 x R9700 measurements, so only an RDNA4 system (every device a gfx1200 or gfx1201; the parameter is named hip) takes them; any other build keeps main's
// rule there (16 MB from 1/1.3, the streamed ring below it) until a CUDA multi-card rig is measured.
inline int staging_ring_mb(double share, double room_bytes, int n_dev, bool integrated, bool hip, bool v026 = false) {
    if (v026) return share >= 1.0 / 1.3 ? 16 : 400;   // 0.2.6: 16 MB where the cards hold the experts, a fixed 400 MB otherwise
    const double GB = 1e9;
    const double per_dev = n_dev > 0 ? room_bytes / (double) n_dev : 0.0;
    const double streamed = std::min(1200.0, std::max(400.0, room_bytes / GB * 64.0));
    if (integrated || !hip) return share >= 1.0 / 1.3 ? 16 : (int) streamed;
    if (share >= 0.82) return 16;
    if (share >= 0.5) return (int) std::min(512.0, std::max(128.0, per_dev / GB * 8.0));
    return (int) streamed;
}

// DS4_TOPK=N (a test, not for users): over the flag and the rule; returns the count to use
inline int experts_test_override(int chosen, int model_k) {
    if (const char* v = std::getenv("DS4_TOPK")) {
        const int n = std::atoi(v);
        if (n >= 1 && n <= model_k) return n;
    }
    return chosen;
}


// The base of the attention slab, MB of q (rows = MB / the heads' width, rounded to 64): STRATA_DS4_A_MB when given; 0.2.6 and a discrete card 8 MB;
// a unified-memory GPU 32 MB (main's 3133160ed, Strix Halo).
inline size_t attn_slab_base_mb(bool unified, size_t env_mb, bool v026 = false) {
    if (env_mb != 0) return env_mb;
    return !v026 && unified ? 32 : 8;
}

// DS4_COPY_PIN (pin the CPU lane's RAM copy so the prompt DMAs it in place): env -1 unset, else 0 / 1.  Explicit wins; 0.2.6: opt-in, so unset = off;
// 0.2.7: on when the whole arena is on explicit huge pages OR (RDNA4 and the cards hold at least half the experts).
inline bool copy_pin_default(int env, bool arena_all_huge, bool rdna4, double share, bool have_chunks, bool v026 = false) {
    if (env >= 0) return env != 0;
    if (v026) return false;
    return (arena_all_huge || (rdna4 && share >= 0.5)) && have_chunks;
}

// DS4_COPY_HUGETLB (the copy arena on explicit 2 MB pages): env -1 unset, else 0 / 1; 0.2.6: opt-in; 0.2.7: on when the reserved pool already holds the whole copy.
inline bool copy_hugetlb_default(int env, double pool_free_bytes, double copy_bytes, bool v026 = false) {
    if (env >= 0) return env != 0;
    if (v026) return false;
    return pool_free_bytes > 0.0 && pool_free_bytes >= copy_bytes;
}

// DS4_FAST_ATTN (the chunked prompt's attention on the matrix cores): env -1 unset, else 0 / 1; 0.2.6: opt-in; 0.2.7: on for
// gfx1200 / gfx1201, and (this change) on for gfx1151 on Windows, where it measured +43.6% on a 32K prompt (101.6 -> 70.8 s,
// sparse attention 38.3 -> 7.0 s), the same greedy first token and a first-token KL far under the 0.0228 bar.
inline bool fast_attn_default(int env, bool gfx12, bool gfx1151win = false, bool v026 = false) {
    if (env >= 0) return env != 0;
    return !v026 && (gfx12 || gfx1151win);
}

// The drafter's cumulative confidence cut (STRATA_DS4_DSPARK_CUM, a request's dspark_cum): the variable when set; 0.2.6: 0; 0.2.7: 0.5 on gfx1200 / gfx1201.
inline float dspark_cum_value(bool has_env, float env_value, bool gfx12, bool v026 = false) {
    if (has_env) return env_value;
    return !v026 && gfx12 ? 0.5f : 0.0f;
}

// DS4_STORE_SPARE (the RAM store takes the spare slots) and DS4_STORE_PUT (an expert the VRAM cache gives back is put into the store from memory):
// env -1 unset, else 0 / 1; 0.2.6: neither existed (off); 0.2.7: on.
inline bool store_feature_default(int env, bool v026 = false) {
    if (env >= 0) return env != 0;
    return !v026;
}

// DS4_SETTLE_STORE (the RAM store's post-prompt preload) when the variable is unset: 0.2.6 ran it (1), 0.2.7 does not (0: it cost a 4K prompt 4.7 s).
inline float settle_store_default(bool v026 = false) { return v026 ? 1.0f : 0.0f; }
// The preload skips the experts the CPU lane already holds as a RAM copy (0.2.7); 0.2.6 did not.
inline bool settle_skips_copy_experts(bool v026 = false) { return !v026; }

// The chunked prompt's dense dequantizer: the row-per-lane dq_rows_kernel (0.2.6's values, bit for bit) or dq_rows_kernel2 (the coalesced transpose).
// DS4_DQ_OLD (-1 unset, else 0 / 1) wins; 0.2.6: the old one; 0.2.7: the old one on gfx11 only.
inline bool dq_old_default(int env, bool gfx11, bool v026 = false) {
    if (env >= 0) return env != 0;
    return v026 || gfx11;
}

// The head's columns a pass for a GGUF head (DS4_HEAD_PER overrides): 0.2.6 4, 0.2.7 8 (the same logits per column).
inline int head_per_default(bool v026 = false) { return v026 ? 4 : 8; }
// The drafter's markov head quantised to Q8_0 as it loads (0.2.7) or read as F16 (0.2.6).
inline bool markov_head_q8_default(bool v026 = false) { return !v026; }
// The gate/up tiles read the layer's F16 copy of the activations (0.2.7, bit-identical) or the F32 rows (0.2.6).
inline bool gate_up_x16_default(bool v026 = false) { return !v026; }
// STRATA_DENSE_PFGEMM (the RDNA3 WMMA dense GEMM before hipBLAS; -1 unset, else 0 / 1): 0.2.6 did not have it.
inline bool dense_pfgemm_default(int env, bool v026 = false) {
    if (env >= 0) return env != 0;
    return !v026;
}

}  // namespace strata::ds4
