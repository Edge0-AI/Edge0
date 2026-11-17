// src/core/coupled_draft_test.cpp - coupled draft sampling's host-side arithmetic (include/strata/core/coupled_draft.hpp).
//
// No GPU.  A simulated decode - windows of random size, random accepted counts, the serve loop's first window of one
// token and the CLI's `draft_first` - checks that
//   1. every draft the MTP chain makes is drawn with the Philox counter of the verify row that checks it: the verify
//      window at pos0 draws row t with pos0 + t (verify.cpp run()), and row t checks the draft in row t + 1;
//   2. the MTP cell of each draft predicts exactly the position that draft occupies in the next window;
//   3. the penalty window each draft sees in the device ring (the staged base, then the chain's earlier drafts) is
//      token for token the history row `penalty_rows` gives the verify row that checks it;
//   4. the penalty window length matches the target's (serve caps the history at 4,096).
#include "strata/core/coupled_draft.hpp"

#include <cstdio>
#include <random>
#include <vector>

using namespace strata::core;

namespace {
int g_fail = 0;
int64_t g_checks = 0;
void check(bool ok, const char* what, long long a = 0, long long b = 0) {
    ++g_checks;
    if (!ok) {
        if (++g_fail <= 20) std::printf("  FAIL %s (%lld vs %lld)\n", what, a, b);
    }
}

// verify.cpp run(): sp.counter = pos0, and the kernel draws row t with sp.counter + t
uint64_t verify_row_counter(int64_t pos0, int t) { return (uint64_t) pos0 + (uint64_t) t; }

// The counters and positions of a chain of `n` drafts made after a window at `p` with `a` accepted (draft()), or for
// draft_first (cell0 = p - 1, a = 0), checked against the next window at `p_next`.
void check_chain(int64_t p, int a, int n, int64_t p_next) {
    for (int j = 0; j < n; ++j) {
        const int64_t cell = coupled_draft_cell(p, a, j);
        // mtp.hpp: cell c predicts the token at c + 2; the next window's row j + 1 holds position p_next + j + 1
        check(cell + 2 == p_next + j + 1, "draft position = its row in the next window", cell + 2, p_next + j + 1);
        // row j + 1 is kept iff it equals row j's pick, drawn at p_next + j
        check(coupled_draft_counter(cell) == verify_row_counter(p_next, j), "draft counter = verifying row's counter",
              (long long) coupled_draft_counter(cell), (long long) verify_row_counter(p_next, j));
    }
}

// The device ring for one chain: base staged at [cap - h, cap) by the host, draft i at cap + i; draft j's window is
// [coupled_hist_start(cap, j, h), + h).  Compared with penalty_rows over (consumed, [next, drafts...]).
void check_history(std::mt19937& rng, int cap, int h, int n_consumed, int n_drafts) {
    std::uniform_int_distribution<int> tok(0, 999);
    std::vector<int32_t> consumed((size_t) n_consumed), drafts((size_t) n_drafts);
    for (auto& t : consumed) t = tok(rng);
    for (auto& t : drafts) t = tok(rng);
    const int32_t next = tok(rng);
    std::vector<int32_t> mapped((size_t) cap, 777777), ring((size_t) (cap + n_drafts), 888888);
    coupled_hist_base(consumed.data(), (int64_t) consumed.size(), next, h, mapped.data() + (cap - h));   // set_draft_history
    for (int i = cap - h; i < cap; ++i) ring[(size_t) i] = mapped[(size_t) i];                           // coupled_stage_kernel
    // the next window: [next, drafts...], T = n_drafts + 1 rows; row j checks draft j (row j + 1)
    std::vector<int32_t> window;
    window.push_back(next);
    window.insert(window.end(), drafts.begin(), drafts.end());
    const int T = (int) window.size();
    std::vector<int32_t> rows((size_t) T * (size_t) h);
    strata::kernels::penalty_rows(consumed.data(), (int64_t) consumed.size(), window.data(), T, h, rows.data());
    for (int j = 0; j < n_drafts; ++j) {
        const int s = coupled_hist_start(cap, j, h);
        check(s >= 0 && s + h <= cap + n_drafts, "the window lies inside the ring", s, cap + n_drafts);
        for (int i = 0; i < h; ++i)
            check(ring[(size_t) (s + i)] == rows[(size_t) j * (size_t) h + (size_t) i], "draft history = verify row history",
                  ring[(size_t) (s + i)], rows[(size_t) j * (size_t) h + (size_t) i]);
        ring[(size_t) (cap + j)] = drafts[(size_t) j];   // coupled_merge_kernel appends the draft after sampling it
    }
}
}  // namespace

int main() {
    std::printf("coupled_draft_test\n");
    std::mt19937 rng(20260930);
    // ---- 1 + 2: a serve-style decode (the first window is the last prompt token alone, then MTP windows)
    for (int run = 0; run < 200; ++run) {
        std::uniform_int_distribution<int> tdist(1, 8);
        int64_t p = 100 + run * 37;   // n - 1
        int T = 1;
        for (int round = 0; round < 300; ++round) {
            std::uniform_int_distribution<int> adist(0, T - 1);
            const int a = adist(rng);
            const int64_t p_next = p + a + 1;
            const int n = 7;   // the chain makes up to max_t - 1 drafts, whatever T the next window takes
            check_chain(p, a, n, p_next);
            p = p_next;
            T = tdist(rng);    // --spec-min-p / the draft policy choose the next window's size
        }
    }
    // the CLI's draft_first: the first window at p, drafts from cell p - 1 with a = 0
    for (int64_t p = 1; p < 5000; p += 13) check_chain(p - 1, 0, 7, p);
    // ---- 3: the penalty ring against penalty_rows
    for (int cap : {8, 64, kCoupledHistCap}) {
        for (int h : {1, 3, 8}) {
            if (h > cap) continue;
            for (int n_consumed : {0, 1, 2, 5, 7, 20, 100})
                for (int n_drafts : {1, 3, 7}) check_history(rng, cap, h, n_consumed, n_drafts);
        }
        check_history(rng, cap, cap, 3, 7);        // the window is the whole ring base
        check_history(rng, cap, cap, 5000, 7);
    }
    // ---- 4: the window length the target uses: min(penalty_last_n, history_len) with history_len =
    // min(penalty_last_n, 4096) in serve (0 = off)
    for (int pln : {-1, 0, 1, 64, 4095, 4096, 4097, 100000}) {
        const int hist_n = std::min(std::max(pln, 0), 4096);
        const int target = hist_n > 0 ? std::min(pln, hist_n) : 0;
        check(coupled_hist_len(pln, kCoupledHistCap) == target, "penalty window length = the target's",
              coupled_hist_len(pln, kCoupledHistCap), target);
    }
    std::printf("  %lld checks, %d failed\n", (long long) g_checks, g_fail);
    std::printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
