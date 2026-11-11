// src/core/nv_defaults.cpp - see include/strata/core/nv_defaults.hpp.
#include "strata/core/nv_defaults.hpp"

#include "strata/core/defaults_026.hpp"
#include "strata/core/emulate.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::core {

namespace {
bool off(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '0';
}
}  // namespace

std::vector<std::pair<std::string, std::string>> nv_default_env(int major, int minor) {
    return nv_default_env(major, minor, strata::qwen_defaults_026());
}

std::vector<std::pair<std::string, std::string>> nv_default_env(int major, int minor, bool defaults026) {
    std::vector<std::pair<std::string, std::string>> t;
    const int mj = strata::cc_major_of(major), mn = strata::cc_minor_of(minor);
    if (mj == 8 && mn == 6) {
        if (off("STRATA_SM86_PERF")) return t;
        t = {
            // 148-nv, RTX 3060 + RTX A4000 (IQ3_XXS, default flags, 8 pairs): the stack +9.6% / +14.6% (8/8 each), identity gate IDENTICAL.
            // Each is in because removing it from the full stack cost speed on a card (drop-one, 4 rounds) and saved none on the other.
            {"STRATA_ADAPT_NOWAIT", "1"},        // A4000 -6.3 points when removed (with ADAPT_LAG), 3060 -0.8
            {"STRATA_ADAPT_LAG", "2"},
            {"STRATA_CPU_IQ_ALL_EXACT1", "1"},   // A4000 -3.3, 3060 -0.6
            {"STRATA_SIDE_MIXER", "2"},          // A4000 -1.2, 3060 -2.3
            {"STRATA_ATTN_DAILY", "1"},          // -1.2, -1.3
            {"STRATA_GR_UP75", "1"},             // -0.9, -1.8
            {"STRATA_GU_SWIGLU", "1"},           // -0.8, -1.4
            {"STRATA_QSA_ROWS", "1"},            // -0.5, -1.2
            {"STRATA_B6_MMVQ_ROWS", "1"},        // A4000 0, 3060 -1.7
        };
    } else if (mj == 6 && mn == 0) {
        if (off("STRATA_SM60_PERF")) return t;
        t = {
            // 148-nv, Tesla P100 (IQ3_XXS, default flags, 8 pairs): the stack +17.8% (8/8), identity gate IDENTICAL.
            {"STRATA_ADAPT_NOWAIT", "1"},        // -10 points when removed (with ADAPT_LAG)
            {"STRATA_ADAPT_LAG", "2"},
            {"STRATA_SIDE_MIXER", "2"},          // -1.8
            {"STRATA_FUSE_WAIT", "2"},           // -1.6
            {"STRATA_B6_MMVQ_ROWS", "1"},        // -1.3
            {"STRATA_FUSE_DECODE", "1"},         // -1.3
            {"STRATA_GR_UP75", "1"},             // -1.2
        };
    } else if (mj == 12 && mn == 1) {
        // GB10 (NVIDIA DGX Spark, 48 SMs, unified memory): only what was measured on that part goes in.  New in 0.2.7.1, so
        // STRATA_DEFAULTS=026 (the master switch of everything since 0.2.6) leaves the table out, like STRATA_SM121_PERF=0.
        if (off("STRATA_SM121_PERF") || defaults026) return t;
        t = {
            // the fused IQ prompt kernels.  IQ3_S, 64K context, fresh 4K/16K/32K prompts at T=0, 3 paired rounds against
            // the same tree without it: prompt 1.139 / 1.149 / 1.121, decode unchanged; 5 of 5 greedy answers byte-
            // identical with it on and off (docs/DGX_SPARK_LINUX.md; HANDOVER §6.2).  It rounds differently from MMQ,
            // so it is a default with an off switch, like the sm_86 table.
            {"STRATA_PF_FUSED", "1"},
            // NVFP4 prompt MMQ with FP4 activations (W4A4), the Blackwell path the engine keeps opt-in.  On this part, the
            // NVFP4 pack measured prompt 1.106x at 4K and 1.229x at 16K against the default W4A8 MMQ, decode unchanged
            // (1.015x), and 24 of 24 greedy answers byte-identical with it on and off.  Opt-in upstream because it is only
            // checked on Blackwell; switched on here for the same reason as the row above, with the same off switch.
            {"STRATA_PREFILL_NVFP4", "w4a4"},
            // the hyper-connection read's fused up projection + mix (gr_upmix).  Measured on IQ3_S against the unfused read:
            // decode 1.054x [1.007, 1.089], prompt unchanged, and 24 of 24 greedy answers byte-identical (the parity test
            // src/prefill/gr_upmix_parity.cu covers it upstream).  The HC read is ~13% of a decode round here, which is
            // why a fusion on it moves the number at all.
            {"STRATA_HC_UPMIX", "1"},
            // the cost-aware MTP window: pick the verify window from what a draft costs against what it has been paying.
            // Measured on IQ3_S: decode 1.047x [1.019, 1.079], prompt unchanged, 24 of 24 greedy answers byte-identical.
            // It only changes which drafts are attempted, and every draft is verified, so the answers cannot drift.
            {"STRATA_SPEC_COST", "1"},
        };
    }
    return t;
}

std::vector<std::string> apply_nv_defaults(int major, int minor) {
    const int mj = strata::cc_major_of(major), mn = strata::cc_minor_of(minor);
    std::vector<std::string> set;
    for (const auto& kv : nv_default_env(major, minor)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, whatever it is
#if defined(_WIN32)
        _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
        setenv(kv.first.c_str(), kv.second.c_str(), 0);
#endif
        set.push_back(kv.first);
    }
    if (!set.empty()) {
        const bool sm86 = mj == 8;
        const bool sm121 = mj == 12 && mn == 1;
        // sm_86 / sm_60: bit-exact switches.  sm_121: the fused prompt path and the FP4-activation NVFP4 prompt round differently from the
        // paths they replace (24 of 24 greedy answers were identical on the Spark; no KL run), so the line does not say "exact".
        std::fprintf(stderr, "strata: %s: %zu %sspeed switches on by default (%s=0 turns them off%s; a switch you set is kept): ",
                     sm86 ? "sm_86 (Ampere)" : sm121 ? "sm_121 (GB10)" : "sm_60 (Pascal)", set.size(), sm121 ? "" : "exact ",
                     sm86 ? "STRATA_SM86_PERF" : sm121 ? "STRATA_SM121_PERF" : "STRATA_SM60_PERF",
                     sm121 ? ", as does STRATA_DEFAULTS=026" : "");
        for (size_t i = 0; i < set.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", set[i].c_str() + 7);
        std::fprintf(stderr, "\n");
    }
    return set;
}

void apply_spec_adapt_default(int major, int minor) {
    const int mj = strata::cc_major_of(major), mn = strata::cc_minor_of(minor);
    // sm_86 only: the P100 (sm_60) gained 4-16% on JSON/code/chat/tool calls but lost 3% on the 4K agent transcript (5/5 pairs), so there it stays opt-in
    // Owner round 9 (2026-10-22): on the z440 A4000s the 5-pair A/B against 0.2.2 was 0.995 (code 0.996, json 1.019, tool 0.973,
    // agent 0.991) and the 3060 was length-confounded, so the earlier +9..17% did not reproduce: opt-in (STRATA_SPEC_ADAPT_SM86=1)
    // until a clean A/B shows it.
    const bool sm86 = mj == 8 && mn == 6;
    if (!sm86) return;
    if (const char* v = std::getenv("STRATA_SPEC_ADAPT_SM86"); v == nullptr || v[0] != '1') return;
    if (off("STRATA_SM86_PERF")) return;
    const std::pair<const char*, const char*> kv[2] = {{"STRATA_SPEC_ADAPT_EVERY", "2"}, {"STRATA_SPEC_ADAPT_WARM", "200"}};
    for (const auto& p : kv) {
        if (std::getenv(p.first) != nullptr) continue;
#if defined(_WIN32)
        _putenv_s(p.first, p.second);
#else
        setenv(p.first, p.second, 0);
#endif
    }
}

}  // namespace strata::core
