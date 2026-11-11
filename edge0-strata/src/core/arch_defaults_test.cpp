// src/core/arch_defaults_test.cpp - the per-architecture default table (no GPU needed).
#include "strata/core/arch_defaults.hpp"
#include "strata/core/gfx12_platform_defaults.hpp"
#include "strata/core/sm120_series.hpp"
#include "strata/core/defaults_026.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static void unset(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}
static void put(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

int main() {
    using namespace strata::core;
    unset("STRATA_GFX1151_DEFAULTS");
    // other architectures (and CUDA, which passes an empty name): nothing
    CHECK(arch_default_env("gfx1100").empty());
    CHECK(arch_default_env("gfx1150").empty());
    CHECK(arch_default_env("gfx1152").empty());
    CHECK(arch_default_env("gfx11510").empty());   // not a prefix match
    CHECK(arch_default_env("").empty());
    CHECK(arch_default_env(nullptr).empty());
    // gfx1151, plain and with its feature suffix: the table, and no bit-changing switch in it
    for (const char* a : {"gfx1151", "gfx1151:sramecc-:xnack-"}) {
        const auto t = arch_default_env(a);
        CHECK(t.size() >= 10);
        for (const auto& kv : t) {
            CHECK(kv.first.rfind("STRATA_", 0) == 0);
            for (const char* bits : {"STRATA_PF_GEMM", "STRATA_PF_FUSED", "STRATA_HC_UPMIX", "STRATA_PA_FAST", "STRATA_HIP_WMMA",
                                     "STRATA_SELECT_WMMA", "STRATA_HC_Q8", "STRATA_PF_HCDOWN", "STRATA_WMMA_GEMM"})
                CHECK(kv.first != bits);
        }
    }
    // 0.2.7: STRATA_SH_STREAM.  Windows gfx1151 leaves the shared-expert fork off (it measured slower there, same ids); Linux gfx1151,
    // the 026 master switch and every other architecture are unchanged.  Pure rule: the platform is an argument.
    {
        auto find = [](const std::vector<std::pair<std::string, std::string>>& t, const char* k) -> const std::string* {
            for (const auto& kv : t) if (kv.first == k) return &kv.second;
            return static_cast<const std::string*>(nullptr);
        };
        for (const char* a : {"gfx1151", "gfx1151:sramecc-:xnack-"}) {
            const auto lin = arch_default_env(a, false, false);
            const auto win = arch_default_env(a, false, true);
            const auto lin26 = arch_default_env(a, true, false);
            const auto win26 = arch_default_env(a, true, true);
            const std::string* l = find(lin, "STRATA_SH_STREAM");
            CHECK(l != nullptr && *l == "1");                      // Linux gfx1151: unchanged, the fork is on
            CHECK(find(win, "STRATA_SH_STREAM") == nullptr);       // Windows gfx1151: not in the table, so the HIP default (off) applies
            CHECK(find(win26, "STRATA_SH_STREAM") != nullptr);     // STRATA_DEFAULTS=026: v0.2.6's table on Windows too
            CHECK(find(lin26, "STRATA_SH_STREAM") != nullptr);
            CHECK(win26 == lin26 && win26 == lin);                 // under 026 and on Linux the whole table is the same
            CHECK(win.size() + 1 == lin.size());                   // Windows loses that one entry and nothing else
            for (const auto& kv : win) CHECK(find(lin, kv.first.c_str()) != nullptr && *find(lin, kv.first.c_str()) == kv.second);
        }
        // other architectures: the platform changes nothing
        for (const char* a : {"gfx1100", "gfx1030", "gfx1103", "gfx1150", "gfx1152", "gfx1201", "gfx1200", ""}) {
            CHECK(arch_default_env(a, false, true) == arch_default_env(a, false, false));
            CHECK(arch_default_env(a, true, true) == arch_default_env(a, true, false));
        }
        CHECK(arch_default_env(nullptr, false, true).empty());
        // the build's own platform answers like the explicit form (the one-argument form reads STRATA_DEFAULTS, unset here)
        unset("STRATA_DEFAULTS");
#if defined(_WIN32)
        CHECK(arch_default_env("gfx1151") == arch_default_env("gfx1151", false, true));
        CHECK(find(arch_default_env("gfx1151"), "STRATA_SH_STREAM") == nullptr);
#else
        CHECK(arch_default_env("gfx1151") == arch_default_env("gfx1151", false, false));
        CHECK(find(arch_default_env("gfx1151"), "STRATA_SH_STREAM") != nullptr);
#endif
        // applying the table: an explicit STRATA_SH_STREAM=1 stays 1, an explicit 0 stays 0, and the Windows table does not set it
        unset("STRATA_QFUSE");
        put("STRATA_SH_STREAM", "1");
        apply_arch_defaults("gfx1151");
        CHECK(std::getenv("STRATA_SH_STREAM") != nullptr && std::string(std::getenv("STRATA_SH_STREAM")) == "1");
        put("STRATA_SH_STREAM", "0");
        apply_arch_defaults("gfx1151");
        CHECK(std::string(std::getenv("STRATA_SH_STREAM")) == "0");
        unset("STRATA_SH_STREAM");
        unset("STRATA_QFUSE");
        const auto applied = apply_arch_defaults("gfx1151");
        bool sh_set = false;
        for (const auto& n : applied) sh_set = sh_set || n == "STRATA_SH_STREAM";
#if defined(_WIN32)
        CHECK(!sh_set);
        CHECK(std::getenv("STRATA_SH_STREAM") == nullptr || std::string(std::getenv("STRATA_SH_STREAM")).empty());
#else
        CHECK(sh_set);
        CHECK(std::getenv("STRATA_SH_STREAM") != nullptr && std::string(std::getenv("STRATA_SH_STREAM")) == "1");
#endif
        unset("STRATA_SH_STREAM");
        unset("STRATA_QFUSE");
        unset("STRATA_TSUM");
    }
    // gfx12: the five prompt switches and four bit-identical decode switches, no bit-changing one
    unset("STRATA_GFX12_DEFAULTS");
    for (const char* a : {"gfx1200", "gfx1201", "gfx1201:sramecc-:xnack-"}) {
        const auto t = arch_default_env(a);
        CHECK(t.size() == 9);
        bool attn_lane = false, attn_merge = false, tsum = false, expdec = false;
        for (const auto& kv : t) {
            if (kv.first == "STRATA_ATTN_LANECELL") attn_lane = kv.second == "1";
            if (kv.first == "STRATA_ATTN_MERGE_V2") attn_merge = kv.second == "1";
            if (kv.first == "STRATA_TSUM") tsum = kv.second == "1";
            if (kv.first == "STRATA_EXPDEC") expdec = kv.second == "1";
        }
        CHECK(attn_lane && attn_merge && tsum && expdec);   // the 0.2.7 R9700 decode measurement, bit-identical
        for (const auto& kv : t) CHECK(kv.first != "STRATA_HCD_EXACT");
        for (const auto& kv : t) {
            CHECK(kv.first != "STRATA_Q8_PACKED" && kv.first != "STRATA_SH_STREAM" && kv.first != "STRATA_PF_GEMM" &&
                  kv.first != "STRATA_PF_FUSED");
        }
    }
    CHECK(arch_default_env("gfx12010").empty());
    put("STRATA_GFX12_DEFAULTS", "0");
    CHECK(arch_default_env("gfx1201").empty());
    unset("STRATA_GFX12_DEFAULTS");
    // a user's setting wins, whatever it is; the rest are set
    put("STRATA_TSUM", "0");
    unset("STRATA_QFUSE");
    const auto set = apply_arch_defaults("gfx1151");
    CHECK(std::getenv("STRATA_TSUM") != nullptr && std::string(std::getenv("STRATA_TSUM")) == "0");
    CHECK(std::getenv("STRATA_QFUSE") != nullptr && std::string(std::getenv("STRATA_QFUSE")) == "1");
    for (const auto& n : set) CHECK(n != "STRATA_TSUM");
    // the opt-out
    put("STRATA_GFX1151_DEFAULTS", "0");
    CHECK(arch_default_env("gfx1151").empty());
    CHECK(apply_arch_defaults("gfx1151").empty());
    // the two approved gfx12 prompt defaults (WMMA attention, Q2_0 fused experts): on for Linux, off for Windows
    CHECK(gfx12_prompt_defaults_on(false));
    CHECK(!gfx12_prompt_defaults_on(true));
#if defined(_WIN32)
    CHECK(!kGfx12PromptDefaultsOn);
#else
    CHECK(kGfx12PromptDefaultsOn);
#endif
    // 153-igpu: the matrix-core prompt set of the integrated Radeons, chosen by the UMA flag AND the architecture.
    // Mock device properties: {gcnArchName, integrated}.
    struct Mock { const char* name; const char* gcn; bool integrated; bool expect_on; };
    const Mock mocks[] = {
        {"Radeon 780M (Phoenix, UMA)",          "gfx1103",                 true,  true},
        {"Radeon 780M with feature suffix",     "gfx1103:sramecc-:xnack-", true,  true},
        {"Radeon 890M (Strix Point, UMA)",      "gfx1150",                 true,  true},
        {"Radeon 8060S (Strix Halo, UMA)",      "gfx1151",                 true,  true},
        {"RX 7900 XTX (dGPU)",                  "gfx1100",                 false, false},
        {"RX 7800 XT (dGPU)",                   "gfx1101",                 false, false},
        {"R9700 (dGPU)",                        "gfx1201",                 false, false},
        {"RX 9060 (dGPU)",                      "gfx1200",                 false, false},
        {"RX 6800 (dGPU)",                      "gfx1030",                 false, false},
        {"a gfx1103 reported as a discrete card", "gfx1103",               false, false},
        {"a gfx1151 reported as a discrete card", "gfx1151",               false, false},
        {"Radeon 860M (Krackan, no kernels)",   "gfx1152",                 true,  false},
        {"older iGPU (gfx1036, no kernels)",    "gfx1036",                 true,  false},
        {"an integrated gfx1201 would not get it", "gfx1201",              true,  false},
        {"CUDA (no gcn name)",                  "",                        true,  false},
    };
    unset("STRATA_IGPU_DEFAULTS");
    unset("STRATA_IGPU_PROMPT_SET");
    // 021-minipc: with nothing set the prompt set is the default of the Radeon 780M only (gfx1103, integrated, not Windows)
    CHECK(kIgpuPromptSetDefaultOn);
    CHECK(igpu_prompt_set_enabled("gfx1103", false));
    CHECK(igpu_prompt_set_enabled("gfx1103:sramecc-:xnack-", false));
    CHECK(!igpu_prompt_set_enabled("gfx1103", true));        // Windows: not tested
    for (const char* g : {"gfx1150", "gfx1151", "gfx1100", "gfx1201", "gfx1152", "", "gfx11030"}) CHECK(!igpu_prompt_set_enabled(g, false));
    CHECK(!igpu_prompt_set_enabled(nullptr, false));
    for (const Mock& m : mocks) {
        const bool want = m.expect_on && std::string(m.gcn).rfind("gfx1103", 0) == 0;
#if defined(_WIN32)
        // 0.2.8: Strix Halo (gfx1151, integrated) gets the matrix-core prompt set on Windows; the 780M and the 890M are
        // not measured there and stay off
        const bool win_on = m.expect_on && m.integrated && std::string(m.gcn).rfind("gfx1151", 0) == 0;
        CHECK(igpu_default_env(m.gcn, m.integrated).size() == (win_on ? 7u : 0u));
#else
        CHECK(igpu_default_env(m.gcn, m.integrated).size() == (want ? 5u : 0u));
#endif
    }
#if !defined(_WIN32)
    for (const char* k : {"STRATA_HIP_WMMA", "STRATA_PF_GEMM", "STRATA_PA_FAST", "STRATA_SELECT_WMMA", "STRATA_HC_UPMIX"}) unset(k);
    CHECK(apply_arch_defaults("gfx1103", true).size() == 5);       // a 780M gets the set with nothing set
    for (const char* k : {"STRATA_HIP_WMMA", "STRATA_PF_GEMM", "STRATA_PA_FAST", "STRATA_SELECT_WMMA", "STRATA_HC_UPMIX"}) unset(k);
    CHECK(apply_arch_defaults("gfx1150", true).empty());           // 880M / 890M: no hardware, stays as it was
    for (const char* k : {"STRATA_HIP_WMMA", "STRATA_PF_GEMM", "STRATA_PA_FAST", "STRATA_SELECT_WMMA", "STRATA_HC_UPMIX"}) unset(k);
#endif
    // the off switches
    put("STRATA_IGPU_PROMPT_SET", "0");
    CHECK(!igpu_prompt_set_enabled("gfx1103", false));
    CHECK(igpu_default_env("gfx1103", true).empty());
    unset("STRATA_IGPU_PROMPT_SET");
    put("STRATA_IGPU_DEFAULTS", "0");
    CHECK(!igpu_prompt_set_enabled("gfx1103", false));
    unset("STRATA_IGPU_DEFAULTS");
    // opt-in per run for the other arches: STRATA_IGPU_PROMPT_SET=1; the checks below run with it on (every mock arch of the list)
    put("STRATA_IGPU_PROMPT_SET", "1");
    CHECK(igpu_prompt_set_enabled("gfx1150", false));
    put("STRATA_IGPU_DEFAULTS", "0");
    CHECK(!igpu_prompt_set_enabled("gfx1150", false));   // the whole-set off switch wins
    unset("STRATA_IGPU_DEFAULTS");
    for (const Mock& m : mocks) {
        const auto t = igpu_default_env(m.gcn, m.integrated);
        CHECK(t.empty() == !m.expect_on);
        if (m.expect_on) {
            bool halo_win = false;
#if defined(_WIN32)
            halo_win = std::string(m.gcn).rfind("gfx1151", 0) == 0;
#endif
            CHECK(t.size() == (halo_win ? 7u : 5u));
            std::vector<const char*> want = {"STRATA_HIP_WMMA", "STRATA_PF_GEMM", "STRATA_PA_FAST", "STRATA_HC_UPMIX"};
            want.push_back("STRATA_SELECT_WMMA");
            if (halo_win) { want.push_back("STRATA_PF_FUSED"); want.push_back("STRATA_PF_F16"); }
            for (const char* w : want) {
                bool found = false;
                for (const auto& kv : t) found = found || (kv.first == w && kv.second == "1");
                CHECK(found);
            }
            for (const auto& kv : t) CHECK(kv.first != "STRATA_HC_Q8" && kv.first != "STRATA_PF_HCDOWN");
            if (!halo_win) { for (const auto& kv : t) CHECK(kv.first != "STRATA_PF_FUSED"); }
        }
        if (!m.expect_on) std::printf("  (not on) %s\n", m.name);
    }
    CHECK(igpu_default_env(nullptr, true).empty());
    CHECK(igpu_prompt_defaults_arch("gfx1103") && igpu_prompt_defaults_arch("gfx1150") && igpu_prompt_defaults_arch("gfx1151"));
    CHECK(!igpu_prompt_defaults_arch("gfx11030") && !igpu_prompt_defaults_arch("gfx1100") && !igpu_prompt_defaults_arch("gfx1201"));
    // the exact tables are untouched: gfx1103 / gfx1150 stay out of arch_default_env, gfx1151 keeps no bit-changing switch
    CHECK(arch_default_env("gfx1103").empty());
    // the opt-out of the whole set
    put("STRATA_IGPU_DEFAULTS", "0");
    CHECK(igpu_default_env("gfx1103", true).empty());
    unset("STRATA_IGPU_DEFAULTS");
    put("STRATA_IGPU_PROMPT_SET", "0");
    CHECK(igpu_default_env("gfx1103", true).empty());
    put("STRATA_IGPU_PROMPT_SET", "1");
    // apply: a discrete card sets nothing, an iGPU sets the five, a switch the user set (even to 0) is kept
    const char* five[] = {"STRATA_HIP_WMMA", "STRATA_PF_GEMM", "STRATA_PA_FAST", "STRATA_SELECT_WMMA", "STRATA_HC_UPMIX"};
    for (const char* k : five) unset(k);
    CHECK(apply_arch_defaults("gfx1100", false).empty());
    // gfx12: a discrete RDNA4 card gets its own nine switches (five prompt, four decode); a second call has nothing to do
    const char* g12[] = {"STRATA_GDN_HEAD", "STRATA_GDN_PP",  "STRATA_GDN_CONVL2",    "STRATA_GDN_NOY", "STRATA_CVEC_FUSE",
                         "STRATA_ATTN_LANECELL", "STRATA_ATTN_MERGE_V2", "STRATA_TSUM", "STRATA_EXPDEC"};
    for (const char* k : g12) unset(k);
    const auto g12set = apply_arch_defaults("gfx1201", false);
    CHECK(g12set.size() == 9);
    for (const char* k : g12) {
        CHECK(std::getenv(k) != nullptr);
        CHECK(std::string(std::getenv(k)) == (std::strcmp(k, "STRATA_GDN_PP") == 0 ? "2" : "1"));
    }
    CHECK(apply_arch_defaults("gfx1201", false).empty());
    CHECK(apply_arch_defaults("gfx1103", false).empty());   // integrated == false: a mixed box (iGPU + dGPU) keeps the old defaults
    for (const char* k : five) CHECK(std::getenv(k) == nullptr);
    put("STRATA_HC_UPMIX", "0");
    const auto on = apply_arch_defaults("gfx1103", true);
    CHECK(on.size() == 4);
    for (const char* k : {"STRATA_HIP_WMMA", "STRATA_PF_GEMM", "STRATA_PA_FAST", "STRATA_SELECT_WMMA"})
        CHECK(std::getenv(k) != nullptr && std::string(std::getenv(k)) == "1");
    CHECK(std::getenv("STRATA_HC_UPMIX") != nullptr && std::string(std::getenv("STRATA_HC_UPMIX")) == "0");
    for (const char* k : five) unset(k);
    unset("STRATA_IGPU_PROMPT_SET");
#if defined(_WIN32)
    // 0.2.8: Strix Halo (gfx1151) on Windows gets the matrix-core prompt set with nothing asked.  Measured on the strix
    // box (Qwen3.8-Flash-Next IQ3_S, engine 0.1.52, 4K / 8K ids prompts): 4K 690 -> 1,090, 8K 785 -> 1,183 (x1.58 /
    // x1.51), KL 0.0088 mean against the HC_Q8 bar of 0.0228.  The set is HIP_WMMA + PF_GEMM + PA_FAST + HC_UPMIX +
    // PF_FUSED (0.2.9 adds SELECT_WMMA and PF_F16 to the Windows gfx1151 set), and the 780M / 890M keep exactly what was measured on them.
    unset("STRATA_IGPU_DEFAULTS");
    unset("STRATA_GFX1151_DEFAULTS");   // the 19-switch gfx1151 table is live in this block (the checks above turned it off)
    CHECK(igpu_prompt_set_enabled("gfx1151", true));
    CHECK(igpu_prompt_set_enabled("gfx1151:sramecc-:xnack-", true));
    CHECK(!igpu_prompt_set_enabled("gfx1151", false));      // Linux gfx1151: still opt-in
    CHECK(!igpu_prompt_set_enabled("gfx1150", true));
    CHECK(!igpu_prompt_set_enabled("gfx1103", true));
    {
        const auto t = igpu_default_env("gfx1151", true);
        CHECK(t.size() == 7);
        auto has = [&t](const char* k) {
            for (const auto& kv : t) if (kv.first == k) return kv.second == "1";
            return false;
        };
        CHECK(has("STRATA_HIP_WMMA") && has("STRATA_PF_GEMM") && has("STRATA_PA_FAST") && has("STRATA_HC_UPMIX") &&
              has("STRATA_PF_FUSED") && has("STRATA_SELECT_WMMA") && has("STRATA_PF_F16"));
        CHECK(igpu_default_env("gfx1151", false).empty());   // a gfx1151 reported as a discrete card
        CHECK(igpu_default_env("gfx1150", true).empty());    // 890M: unchanged
        CHECK(igpu_default_env("gfx1103", true).empty());    // 780M on Windows: unchanged
    }
    {
        const char* halo[] = {"STRATA_HIP_WMMA", "STRATA_PF_GEMM", "STRATA_PA_FAST", "STRATA_HC_UPMIX", "STRATA_PF_FUSED",
                              "STRATA_SELECT_WMMA", "STRATA_PF_F16"};
        for (const char* k : halo) unset(k);
        const auto set = apply_arch_defaults("gfx1151", true);
        for (const char* k : halo) {
            bool in = false;
            for (const auto& n : set) in = in || n == k;
            CHECK(in);
            CHECK(std::getenv(k) != nullptr && std::string(std::getenv(k)) == "1");
        }
        // a switch the user set (even to 0) is kept, and the whole set turns off with either switch
        for (const char* k : halo) unset(k);
        put("STRATA_HC_UPMIX", "0");
        const auto set2 = apply_arch_defaults("gfx1151", true);
        CHECK(std::getenv("STRATA_HC_UPMIX") != nullptr && std::string(std::getenv("STRATA_HC_UPMIX")) == "0");
        for (const auto& n : set2) CHECK(n != "STRATA_HC_UPMIX");
        CHECK(std::getenv("STRATA_PF_FUSED") != nullptr);
        unset("STRATA_HC_UPMIX");
        for (const char* k : halo) unset(k);
        put("STRATA_IGPU_PROMPT_SET", "0");
        CHECK(igpu_default_env("gfx1151", true).empty());
        unset("STRATA_IGPU_PROMPT_SET");
        put("STRATA_IGPU_DEFAULTS", "0");
        CHECK(igpu_default_env("gfx1151", true).empty());
        unset("STRATA_IGPU_DEFAULTS");
        put("STRATA_GFX1151_DEFAULTS", "0");   // back to the state this block found (the 026 checks below rely on it)
    }
#endif
    // 021-minipc: UMA mode is NOT a default (slower, resets not fixed): off with nothing asked; the logic below runs with on=true
    CHECK(!kIgpuUmaArenaDefaultOn);
    CHECK(!igpu_uma_arena_default("gfx1103", true, false));
    // 021-minipc: UMA mode (the expert arena stays unregistered) is chosen for the Radeon 780M only: gfx1103 + integrated + not Windows
    unset("STRATA_UMA_REGISTER");
    CHECK(igpu_uma_arena_default("gfx1103", true, false, true));
    CHECK(igpu_uma_arena_default("gfx1103:sramecc-:xnack-", true, false, true));
    CHECK(!igpu_uma_arena_default("gfx1103", true, true, true));      // Windows: not measured
    CHECK(!igpu_uma_arena_default("gfx1103", false, false, true));    // a gfx1103 that is not integrated
    for (const char* g : {"gfx1150", "gfx1151", "gfx1152", "gfx1100", "gfx1101", "gfx1200", "gfx1201", "gfx1030", "gfx11030", "", "gfx90a"})
        CHECK(!igpu_uma_arena_default(g, true, false, true));
    // 0.2.1.2: Strix Halo on Windows is unregistered by default (on or off), nothing else on Windows; Linux gfx1151 unchanged
    CHECK(igpu_uma_arena_default("gfx1151", true, true));
    CHECK(igpu_uma_arena_default("gfx1151:sramecc+:xnack-", true, true));
    CHECK(!igpu_uma_arena_default("gfx1151", false, true));           // not integrated
    CHECK(!igpu_uma_arena_default("gfx1151", true, false));           // Linux
    for (const char* g : {"gfx1150", "gfx1152", "gfx1103", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201", "gfx1030", "", "gfx90a"})
        CHECK(!igpu_uma_arena_default(g, true, true));
    CHECK(!igpu_uma_arena_default(nullptr, true, true));
    put("STRATA_UMA_REGISTER", "1");
    CHECK(!igpu_uma_arena_default("gfx1151", true, true));
    unset("STRATA_UMA_REGISTER");
    CHECK(!igpu_uma_arena_default(nullptr, true, false, true));
    put("STRATA_UMA_REGISTER", "1");
    CHECK(!igpu_uma_arena_default("gfx1103", true, false, true));     // the user registers the arena as before
    unset("STRATA_UMA_REGISTER");
    // both platforms: no Windows guard on the iGPU set (the owner wants it there too; untested on Windows is only said in the log)
    // ---- STRATA_DEFAULTS=026 (defaults_026.hpp): every 0.2.7 Qwen default gives v0.2.6's answer; an explicit variable is decided by the caller first
    {
        const double GiB = 1073741824.0;
        using namespace strata;
        CHECK(is_026_token("026") && is_026_token("0.2.6") && is_026_token("v0.2.6"));
        CHECK(!is_026_token(nullptr) && !is_026_token("") && !is_026_token("0") && !is_026_token("027"));
        // residency-biased routing: 0 with the master switch on or off (0.2.8: off by default on every card, 026 or not)
        CHECK(route_prior_default(false, false, 12.0 * GiB, false) == 0.0f);
        CHECK(route_prior_default(false, false, 12.0 * GiB, true) == 0.0f);
        CHECK(route_prior_default(false, false, 8.0 * GiB, true) == 0.0f);
        // an explicit STRATA_ROUTE_PRIOR=1.5 wins over 026 as well
        CHECK(route_prior_resolve("1.5", false, false, 12.0 * GiB, true) == 1.5f);
        // the wide HC up read: on for a 12.0 card on CUDA, off under 026, off in HIP, off on sm_86 / sm_121
        CHECK(gr_up_wide_default(false, 12, 0));
        CHECK(!gr_up_wide_default(false, 12, 0, true));
        CHECK(!gr_up_wide_default(true, 12, 0));
        CHECK(!gr_up_wide_default(false, 8, 6));
        CHECK(!gr_up_wide_default(false, 12, 1));
        // #3453: the sm_89 MMVQ-IL tables: on by default for compute capability 8.9 only, off under 026 and in HIP; "1" and "0" win
        CHECK(mmvq_il_89_default(false, 89));
        CHECK(!mmvq_il_89_default(false, 86) && !mmvq_il_89_default(false, 80) && !mmvq_il_89_default(false, 90) &&
              !mmvq_il_89_default(false, 120) && !mmvq_il_89_default(false, 75) && !mmvq_il_89_default(false, 70) &&
              !mmvq_il_89_default(false, 60) && !mmvq_il_89_default(false, 0));
        CHECK(!mmvq_il_89_default(true, 89));
        CHECK(!mmvq_il_89_default(false, 89, true));                       // STRATA_DEFAULTS=026
        CHECK(mmvq_il_89_resolve(nullptr, false, 89, false));              // unset: the rule
        CHECK(!mmvq_il_89_resolve(nullptr, false, 89, true));              // unset + 026: off
        CHECK(!mmvq_il_89_resolve(nullptr, false, 86, false));             // another card: off
        CHECK(!mmvq_il_89_resolve("0", false, 89, false));                 // STRATA_MMVQ_IL_89=0 turns it off
        CHECK(mmvq_il_89_resolve("1", false, 89, true));                   // an explicit =1 wins over the master switch
        CHECK(mmvq_il_89_resolve("1", false, 86, false));                  // (the tables are only looked up for cc 89 anyway)
        CHECK(mmvq_il_89_resolve("", false, 89, false) && !mmvq_il_89_resolve("", false, 89, true));   // empty text: the rule
        // #3779: the two sm_80-only defaults (wide dequant, MMVQ-IL sm_80 tables): on unless STRATA_DEFAULTS=026; "1" and "0" win
        CHECK(sm80_default_on(nullptr, false) && !sm80_default_on(nullptr, true));
        CHECK(!sm80_default_on("0", false) && sm80_default_on("1", true));
        CHECK(sm80_default_on("", false) && !sm80_default_on("", true));
        // GR_V3 and IMMA
        CHECK(gr_v3_default(false, 12, 0, false, 72.0 * GiB, false));
        CHECK(!gr_v3_default(false, 12, 0, false, 72.0 * GiB, true));
        CHECK(prompt_attn_imma_default(false, 86, false));
        CHECK(!prompt_attn_imma_default(false, 86, true));
        CHECK(!prompt_attn_imma_default(false, 120, true));
        // the gfx12 decode switches: the arch table is v0.2.6's five prompt switches under 026, nine otherwise; gfx1151 and the others do not move
        {
            const auto t26 = strata::core::arch_default_env("gfx1201", true);
            const auto t27 = strata::core::arch_default_env("gfx1201", false);
            auto has = [](const std::vector<std::pair<std::string, std::string>>& t, const char* k) {
                for (const auto& kv : t) if (kv.first == k) return true;
                return false;
            };
            CHECK(t26.size() == 5 && t27.size() == 9);
            CHECK(has(t26, "STRATA_GDN_HEAD") && has(t26, "STRATA_CVEC_FUSE"));
            CHECK(!has(t26, "STRATA_ATTN_LANECELL") && !has(t26, "STRATA_ATTN_MERGE_V2") && !has(t26, "STRATA_TSUM") && !has(t26, "STRATA_EXPDEC"));
            CHECK(has(t27, "STRATA_ATTN_LANECELL") && has(t27, "STRATA_ATTN_MERGE_V2") && has(t27, "STRATA_TSUM") && has(t27, "STRATA_EXPDEC"));
            CHECK(strata::core::arch_default_env("gfx1200", true).size() == 5);
            CHECK(strata::core::arch_default_env("gfx1151", true) == strata::core::arch_default_env("gfx1151", false));
            CHECK(strata::core::arch_default_env("gfx1100", true).empty() && strata::core::arch_default_env("gfx1100", false).empty());
        }
        // the decode miss split start value: 0.15 on gfx12 native packs, 0.55 under 026 or STRATA_PCIE_FRAC_DEFAULT=old; the others do not move
        CHECK(pcie_start_frac(false, true, true, false, false) == 0.15);
        CHECK(pcie_start_frac(false, true, true, false, true) == 0.55);
        CHECK(pcie_start_frac(false, true, true, true, false) == 0.55);
        CHECK(pcie_start_frac(false, true, false, false, false) == 0.55);
        CHECK(pcie_start_frac(true, true, true, false, true) == 0.25);
        CHECK(pcie_start_frac(true, true, true, false, false) == 0.25);
        CHECK(pcie_start_frac(false, false, true, false, true) == 0.2);
        CHECK(pcie_start_frac(false, false, true, false, false) == 0.2);
    }
    std::printf(fails ? "arch_defaults_test: %d FAILED\n" : "arch_defaults_test: OK\n", fails);
    // ---- 0.2.7 CUDA defaults (sm120_series.hpp): which cards get which, and which never do
    {
        const double GiB = 1073741824.0;
        using namespace strata;
        // residency-biased routing (owner decision of 2026-10-29, #4402): OFF by default on every card, the 0.2.6 / STRATA_ROUTE_PRIOR=0 answer.
        // 0.2.7 gave 1.5 to a discrete CUDA card of 13 GiB or less; the same cards are listed so the table says where the default went.
        const double cards[] = {0.0, 4.0 * GiB, 8.0 * GiB, 11.9 * GiB, 12.0 * GiB, 13.0 * GiB, 13.5 * GiB, 15.7 * GiB, 24.0 * GiB, 31.4 * GiB, 48.0 * GiB, 72.0 * GiB};
        for (const double bytes : cards)
            for (const bool hip : {false, true})
                for (const bool integrated : {false, true})
                    for (const bool v026 : {false, true}) {
                        CHECK(route_prior_default(hip, integrated, bytes, v026) == 0.0f);
                        CHECK(route_prior_resolve(nullptr, hip, integrated, bytes, v026) == 0.0f);   // nothing set: off, = STRATA_ROUTE_PRIOR=0
                        CHECK(route_prior_resolve(nullptr, hip, integrated, bytes, v026) == route_prior_resolve("0", hip, integrated, bytes, v026));
                    }
        CHECK(route_prior_default(false, false, 11.9 * GiB) == 0.0f);       // RTX 3060 / 5070 12 GB: was 1.5 in 0.2.7
        CHECK(route_prior_default(false, false, 13.0 * GiB) == 0.0f);       // the old boundary
        // the explicit opt-in still works: any set value wins on every card (the Expert-settings row writes 1.5)
        CHECK(strata::kRoutePriorSmallCard == 1.5f);                          // the one place the opt-in value lives
        CHECK(route_prior_resolve("1.5", false, false, 12.0 * GiB) == 1.5f);
        CHECK(route_prior_resolve("1.5", false, false, 12.0 * GiB) == strata::kRoutePriorSmallCard);
        CHECK(route_prior_resolve("1.5", false, false, 8.0 * GiB, true) == 1.5f);   // also under STRATA_DEFAULTS=026
        CHECK(route_prior_resolve("1.5", false, false, 72.0 * GiB) == 1.5f);        // and on a card the 0.2.7 default never touched
        CHECK(route_prior_resolve("1.5", true, false, 12.0 * GiB) == 1.5f);         // the variable is the user's own on any build
        CHECK(route_prior_resolve("2.0", false, false, 12.0 * GiB) == 2.0f);
        CHECK(route_prior_resolve("0.5", false, false, 12.0 * GiB) == 0.5f);
        CHECK(route_prior_resolve("0", false, false, 12.0 * GiB) == 0.0f);
        CHECK(route_prior_resolve("", false, false, 12.0 * GiB) == 0.0f);           // set but empty reads as 0, as atof always did here
        CHECK(route_prior_resolve("off", false, false, 12.0 * GiB) == 0.0f);
        // the stream-split hyper-connection read: a discrete 12.0 card above 13 GiB (RTX PRO 5000 72 GB, RTX 5090 32 GB), nothing else
        CHECK(gr_v3_default(false, 12, 0, false, 72.0 * GiB));
        CHECK(gr_v3_default(false, 12, 0, false, 31.4 * GiB));              // RTX 5090
        CHECK(!gr_v3_default(false, 12, 0, false, 11.9 * GiB));             // RTX 5070 12 GB: streams most experts, unmeasured
        CHECK(gr_v3_default(false, 12, 0, false, 15.6 * GiB));               // a 16 GB sm_120 card (RTX 5060 Ti / 5070 Ti): above 13 GiB, on (unmeasured size, same arch)
        CHECK(!gr_v3_default(false, 8, 6, false, 15.7 * GiB));              // 2x RTX A4000 (sm_86): measured -3.0% / -5.9%
        CHECK(!gr_v3_default(false, 8, 6, false, 72.0 * GiB));              // any sm_86
        CHECK(!gr_v3_default(false, 8, 9, false, 24.0 * GiB));              // sm_89
        CHECK(!gr_v3_default(false, 12, 1, false, 128.0 * GiB));            // sm_121 (DGX Spark class): not 12.0
        CHECK(!gr_v3_default(false, 12, 0, true, 128.0 * GiB));             // integrated
        CHECK(!gr_v3_default(true, 12, 0, false, 32.0 * GiB));              // a HIP build: gfx1201 reports 12.0 too
        CHECK(!gr_v3_default(false, 6, 0, false, 16.0 * GiB));              // P100
    }
    {
        using namespace strata;
        // the prompt attention on int8 tensor cores: every CUDA card from sm_80 on, not Turing / Volta / Pascal, not HIP
        CHECK(prompt_attn_imma_default(false, 120));                        // RTX PRO 5000 / RTX 50: measured +1.6-1.8%
        CHECK(prompt_attn_imma_default(false, 86));                         // 2x RTX A4000 / RTX 3060: measured +1.1-1.3%
        CHECK(prompt_attn_imma_default(false, 80));
        CHECK(prompt_attn_imma_default(false, 89));
        CHECK(!prompt_attn_imma_default(false, 75));                        // Turing (also a forced STRATA_QSA_WARP=1 card)
        CHECK(!prompt_attn_imma_default(false, 70));                        // Volta
        CHECK(!prompt_attn_imma_default(false, 60));                        // Pascal (P100)
        CHECK(!prompt_attn_imma_default(true, 120));                        // HIP
    }
    {
        using namespace strata;
        // the layer-major prompt walk (#3976): on by default for a CUDA layer split only
        CHECK(prefill_layer_major_default(false, 2));                       // 2x RTX A4000 / A6000: measured 5x time to first token
        CHECK(prefill_layer_major_default(false, 4));
        CHECK(!prefill_layer_major_default(false, 1));                      // one GPU: opt-in only
        CHECK(!prefill_layer_major_default(false, 0));
        CHECK(!prefill_layer_major_default(true, 2));                       // HIP builds
        CHECK(!prefill_layer_major_default(false, 2, true));                // STRATA_DEFAULTS=026
        CHECK(!prefill_layer_major_default(true, 1, true));
        // the variable wins in both directions, whatever the rule says
        CHECK(!prefill_layer_major_resolve("0", false, 2));                 // the off switch
        CHECK(prefill_layer_major_resolve("1", false, 1));                  // the opt-in of the first ports: one GPU too
        CHECK(prefill_layer_major_resolve("1", false, 2, true));            // explicit beats STRATA_DEFAULTS=026
        CHECK(!prefill_layer_major_resolve("0", false, 2, false));
        CHECK(!prefill_layer_major_resolve("0", true, 2));
        CHECK(prefill_layer_major_resolve("2", false, 1));                  // any non-zero number is on
        CHECK(!prefill_layer_major_resolve("off", false, 2));               // text that reads as 0 is off (atoi), as before this rule existed
        // unset or empty: the rule
        CHECK(prefill_layer_major_resolve(nullptr, false, 2));
        CHECK(prefill_layer_major_resolve("", false, 2));
        CHECK(!prefill_layer_major_resolve(nullptr, false, 1));
        CHECK(!prefill_layer_major_resolve(nullptr, false, 2, true));
        CHECK(!prefill_layer_major_resolve(nullptr, true, 2));
    }
    return fails ? 1 : 0;
}
