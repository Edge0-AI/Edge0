// strata/core/sm120_series.hpp - bitwise kernel switches that are ON by default on a compute capability 12.0 card
// (RTX 50 series / RTX PRO Blackwell workstation cards).
//
// 147-verify measured these exact switches on an RTX PRO 5000 Blackwell 72 GB (IQ3_S, every expert in VRAM): together
// a verify round of 4 tokens 13.70 -> 12.23 ms with the same text (notes-147-l620.md).  Each one is bitwise equal to
// the kernels it replaces (their own parity tests and the identity gate), so only the speed changes:
//   STRATA_SIDE_MIXER=2  the mixer's input-only work on branches of the verify graph
//   STRATA_ATTN_DAILY=1  the Daily's decode attention (attn_chunk_kernel_pre75 + merge v2)
//   STRATA_GR_UP75=1     the hyper-connection up projection with its rows' loads hoisted
//   STRATA_GR_DOWN_TG=1  the hyper-connection down projection with one token a block row (T times the blocks); =2 (token
//                        pairs) on a 12.0 card with fewer than 80 SMs
//   STRATA_GU_SWIGLU=1   gate/up, SwiGLU and the q8_1 blocks of the routed experts in one launch
//   STRATA_GR_UP_WIDE=1  the hyper-connection up projection with one warp a row (1,280 blocks instead of 160; +1.9%
//                        story / +0.7% code decode on the RTX PRO 5000, 5 pairs, token ids identical; bitwise)
// and one that rounds differently (KL-checked, the AMD gfx12 rule): STRATA_PF_FUSED (the native packs' prompt experts on the
// fused int8 kernels, IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS gate/up only, 12.0 cards with 80+ SMs only; =0 keeps MMQ, =1 takes every
// covered format on any card).
//
//   STRATA_SM120_PERF=0  turns the whole set off on a 12.0 card (the defaults every other card has).
// Each switch's own variable still wins in both directions (=0 off, =1 / =2 on), on any card, as before.
// Nothing here changes a card that is not a 12.0 (an sm_121 / sm_100 card keeps its defaults).
#pragma once

#include "strata/core/emulate.hpp"

#include <cstdlib>

namespace strata {

/// STRATA_SM120_PERF (read once): unset or anything but "0" = on
inline bool sm120_series() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_SM120_PERF");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return on;
}

/// the device (raw cudaDevAttrComputeCapabilityMajor / Minor) is a 12.0 card, or emulates one (STRATA_EMULATE_CC=120)
inline bool is_sm120(int major, int minor) { return cc_major_of(major) == 12 && cc_minor_of(minor) == 0; }

/// the default of a bitwise switch of the set: on for a 12.0 card unless STRATA_SM120_PERF=0
inline bool sm120_default(int major, int minor) { return is_sm120(major, minor) && sm120_series(); }

// ---- 0.2.7 defaults of the CUDA speed work (pure rules; the engine reads the device and calls these; tests in arch_defaults_test)
// `hip`: a HIP build, where a gfx1200 / gfx1201 reports compute capability 12.0 as well and none of these is measured or wanted.
// `v026`: STRATA_DEFAULTS=026 (defaults_026.hpp qwen_defaults_026()): every one of these is then off, as in v0.2.6.  An explicit variable
// (STRATA_ROUTE_PRIOR=2, STRATA_GR_V3=1, ...) is decided by the caller before it asks the rule, so it still wins.
// `total_bytes`: the card's memory (cudaDeviceProp::totalGlobalMem); `integrated`: a GPU that shares the PC's memory.
constexpr double kSmallCardGiB = 13.0;   // "a 12 GB card": 12 GB cards report 11.6 - 12.0 GiB, 16 GB ones 15.6 - 16.0

/// The residency-bias bonus the user can ask for on a discrete CUDA card of 13 GiB or less (STRATA_ROUTE_PRIOR=1.5; the app's Expert-settings row
/// "Faster answers on 12 GB NVIDIA cards" writes it): THE one place the value lives.  1.5 since the owner decision of 2026-10-26 (2.0 had real defects in
/// the copy and code check on the RTX 3060 12 GB with IQ3_XXS; none in 19 tasks at 1.5 or 1.0).  Speed against off: story +15%, code +37%.
/// It was the default on those cards in 0.2.7 only.  Owner decision of 2026-10-29: OFF by default again (#4402: repetition loops and weaker results in long
/// agent sessions; our own 0.2.7 agent check passed 4.5 of 10 tasks with it on against 5.6 with it off).  It stays an opt-in.
constexpr float kRoutePriorSmallCard = 1.5f;

/// STRATA_ROUTE_PRIOR when the user set nothing: 0 on every card (residency-biased routing changes which experts run, so it is only on when asked for:
/// STRATA_ROUTE_PRIOR=1.5, or the Expert-settings row).  The same answer as v0.2.6 and as 0.2.7 with STRATA_ROUTE_PRIOR=0 or STRATA_DEFAULTS=026.
/// The arguments are kept so a caller and the table test (arch_defaults_test) still ask the same question per card; none of them changes the answer.
inline float route_prior_default(bool /*hip*/, bool /*integrated*/, double /*total_bytes*/, bool /*v026*/ = false) { return 0.0f; }

/// The lambda the engine uses given the text of STRATA_ROUTE_PRIOR (nullptr = unset): any set value is the user's own and wins on every card
/// (atof; "0" and text that reads as 0 are off; "1.5" is what the Expert-settings row writes), unset is route_prior_default (0).
inline float route_prior_resolve(const char* env, bool hip, bool integrated, double total_bytes, bool v026 = false) {
    return env != nullptr ? (float) std::atof(env) : route_prior_default(hip, integrated, total_bytes, v026);
}

/// STRATA_GR_V3 (the stream-split hyper-connection read, another summation order) when the user set nothing: ON for a discrete compute capability
/// 12.0 card (STRATA_SM120_PERF=0 turns the set off) with more than 13 GiB, CUDA builds only.  Measured: RTX PRO 5000 72 GB +1.2% decode clean,
/// -3.0% / -5.9% on 2x RTX A4000 (sm_86), +1.0% / -3.4% on an RTX 3060 (sm_86): so not those; a 12 GB sm_120 card is unmeasured.
inline bool gr_v3_default(bool hip, int major, int minor, bool integrated, double total_bytes, bool v026 = false) {
    return !v026 && !hip && !integrated && sm120_default(major, minor) && total_bytes / 1073741824.0 > kSmallCardGiB;
}

/// STRATA_PROMPT_ATTN_IMMA (the prompt attention on int8 tensor cores, another summation order) when the user set nothing: ON for every CUDA card of
/// compute capability 8.0 or newer (the int8 m16n8k32 MMA), not Turing / Volta / Pascal, not HIP.  `cc` = 10 * major + minor as the engine
/// resolves it (an emulated or forced Turing counts as 75).  Measured on sm_120 (RTX PRO 5000) and sm_86 (2x RTX A4000): +1.1 .. +1.8% prompt.
inline bool prompt_attn_imma_default(bool hip, int cc, bool v026 = false) { return !v026 && !hip && cc >= 80; }

/// STRATA_MMVQ_IL_89 (#3453: the decode kernel row tables of an Ada card, the cells where an RTX 4070 and an RTX 4080 agreed; bitwise the same
/// output, only the rows-a-warp split differs) when the user set nothing: ON for compute capability 8.9 only (owner decision, 26 October: not yet
/// measured on an Ada card by us), off under STRATA_DEFAULTS=026 and in HIP builds.  `cc` = 10 * major + minor as the engine resolves it.
inline bool mmvq_il_89_default(bool hip, int cc, bool v026 = false) { return !v026 && !hip && cc == 89; }

/// The two sm_80-only defaults of 0.2.7 (#3779: the wide dense dequant and the MMVQ-IL sm_80 tables, both bitwise the old output, measured on a
/// CMP 170HX only) given the text of their variable (nullptr = unset): "1" on and "0" off win, unset is on unless STRATA_DEFAULTS=026.
inline bool sm80_default_on(const char* env, bool v026 = false) {
    if (env != nullptr && env[0] == '1') return true;
    if (env != nullptr && env[0] == '0') return false;
    return !v026;
}

/// The answer for STRATA_MMVQ_IL_89 given the variable's text (nullptr = unset): "1" on and "0" off win over everything, anything else (unset,
/// empty, other text) is the rule above.
inline bool mmvq_il_89_resolve(const char* env, bool hip, int cc, bool v026 = false) {
    if (env != nullptr && env[0] == '1') return true;
    if (env != nullptr && env[0] == '0') return false;
    return mmvq_il_89_default(hip, cc, v026);
}

/// STRATA_GR_UP_WIDE (the hyper-connection up projection, one warp a row; bitwise the old kernel) when the user set nothing: ON for a compute
/// capability 12.0 card (STRATA_SM120_PERF=0 turns the set off), CUDA builds only.
inline bool gr_up_wide_default(bool hip, int major, int minor, bool v026 = false) { return !v026 && !hip && sm120_default(major, minor); }

/// STRATA_PREFILL_LAYER_MAJOR (#3976: the layer-major prompt walk, which reads each layer's streamed experts once per super-chunk instead of once
/// per chunk; another summation order of the routed experts, so KL-checked, not bitwise) when the user set nothing: ON for a CUDA layer split
/// (`stages` = the number of GPUs the prompt path runs on, 2 or more), OFF on one GPU, in a HIP build and under STRATA_DEFAULTS=026.  Measured on
/// 2x RTX A4000 (sm_86) with Qwen3.8-Flash-Next IQ3_XXS and streamed experts only.  The engine adds its own per-prompt declines (image prompts,
/// DFlash2 taps, KVA, a batch read that yields, a peer GPU, no VRAM for a super-chunk): those read the chunk-major path as before.
inline bool prefill_layer_major_default(bool hip, int stages, bool v026 = false) { return !v026 && !hip && stages >= 2; }

/// The answer for STRATA_PREFILL_LAYER_MAJOR given the variable's text (nullptr = unset): any text that starts a number other than 0 is on, "0" (or any
/// text that reads as 0) is off, both win over everything; unset or empty is the rule above.  An explicit "1" is honoured on one GPU too (the
/// opt-in of the first ports); a HIP build still declines at run time.
inline bool prefill_layer_major_resolve(const char* env, bool hip, int stages, bool v026 = false) {
    if (env != nullptr && env[0] != '\0') return std::atoi(env) != 0;
    return prefill_layer_major_default(hip, stages, v026);
}

}  // namespace strata
