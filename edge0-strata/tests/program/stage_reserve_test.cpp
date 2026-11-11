// tests/program/stage_reserve_test.cpp - #3702: the later-stage reserve arithmetic of the layer split search.
// reserve_mib / room_bytes must equal generate.cpp's `stage_room` expression as it was written there; the proposal
// functions are pinned too (they are not used by the default placement).
#include "strata/program/stage_reserve.hpp"

#include <cstdint>
#include <cstdio>
#include <string>

using namespace strata::program::stage_reserve;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_fail;
}

// the expression as it stood in generate.cpp before the move (kWindowMib, kDrafterMib, kPipeSnapMib, kPipeWindowMib)
int64_t legacy_room(int64_t fb, int64_t vram_reserve_mib, int64_t vram_reserve_later_mib, int64_t pf, bool later, bool drafter,
                    int64_t pipeline_windows) {
    const int64_t kWin = 96, kDrafter = 1000;
    const int64_t kPipeWindow = pipeline_windows > 0 ? 160 : 0;
    const int64_t kPipeSnap = pipeline_windows >= 2 ? 96 : 0;
    const int64_t base_reserve = later && vram_reserve_later_mib >= 0 ? vram_reserve_later_mib : vram_reserve_mib;
    const int64_t reserve = (base_reserve + pf + (later ? kWin : kPipeSnap) + kPipeWindow + (drafter ? kDrafter : 0)) << 20;
    return std::max<int64_t>(fb - reserve, 0);
}
}  // namespace

int main() {
    // the same numbers as the old expression over a grid of settings
    long cases = 0;
    for (int64_t fb : {int64_t(0), int64_t(500) << 20, int64_t(3000) << 20, int64_t(15) << 30})
        for (int64_t vr : {int64_t(0), int64_t(700), int64_t(1280)})
            for (int64_t vl : {int64_t(-1), int64_t(0), int64_t(1400)})
                for (int64_t pf : {int64_t(0), int64_t(160 + 4096 * 680 / 1024)})
                    for (int later = 0; later < 2; ++later)
                        for (int dr = 0; dr < 2; ++dr)
                            for (int64_t pw : {int64_t(0), int64_t(1), int64_t(2)}) {
                                Inputs in;
                                in.base_reserve_mib = later && vl >= 0 ? vl : vr;
                                in.prompt_buffers_mib = pf;
                                in.later = later != 0;
                                in.drafter = dr != 0;
                                in.pipe_window_mib = pw > 0 ? 160 : 0;
                                in.pipe_snap_mib = pw >= 2 ? 96 : 0;
                                ++cases;
                                check(room_bytes(fb, in) == legacy_room(fb, vr, vl, pf, later != 0, dr != 0, pw),
                                      "room differs from the old expression (fb " + std::to_string(fb >> 20) + " MiB, reserve " +
                                          std::to_string(vr) + "/" + std::to_string(vl) + ", later " + std::to_string(later) + ")");
                            }
    // the configuration of #3702: default reserves (700), borrowed prompt buffers, CUDA1 = the last card with the drafter
    {
        Inputs in;
        in.base_reserve_mib = 700;
        in.later = true;
        in.drafter = true;
        check(reserve_mib(in) == 700 + 96 + 1000, "the last card of #3702 keeps 1796 MiB planned");
        in.drafter = false;
        check(reserve_mib(in) == 796, "a later card without the drafter keeps 796 MiB planned");
        in.later = false;
        check(reserve_mib(in) == 700, "the first card keeps the base reserve alone");
        in.pipe_snap_mib = 96;
        check(reserve_mib(in) == 796, "the first card books the pipeline's second GDN state");
    }
    // the same arithmetic with --vram-reserve-later-mib 1400 (the workaround that worked in #3702)
    {
        Inputs in;
        in.base_reserve_mib = 1400;
        in.later = true;
        check(reserve_mib(in) == 1496, "the workaround reserve: 1400 + 96");
    }
    // the proposal: the window term grows with the number of window sizes (--spec 4 -> windows of 1..6 tokens)
    check(proposed_window_mib(1) == 96 + 48, "proposal: one window size = the arena and the burst");
    check(proposed_window_mib(6) == 96 + 5 * 6 + 48, "proposal: six sizes = 174 MiB");
    check(proposed_window_mib(0) == proposed_window_mib(1), "proposal: no negative sizes");
    {
        Inputs in;
        in.base_reserve_mib = 700;
        in.later = true;
        check(proposed_reserve_mib(in, 6) == 700 + 174, "proposed reserve on a later card, six sizes");
        in.later = false;
        check(proposed_reserve_mib(in, 6) == reserve_mib(in), "the first card is not changed by the proposal");
    }
    std::printf("stage_reserve_test: %ld cases, %s\n", cases, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
