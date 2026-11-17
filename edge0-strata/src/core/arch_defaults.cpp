// src/core/arch_defaults.cpp - see include/strata/core/arch_defaults.hpp.
#include "strata/core/arch_defaults.hpp"
#include "strata/core/defaults_026.hpp"

#include "strata/kernels/gfx_arch.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch) {
    return arch_default_env(gcn_arch, strata::qwen_defaults_026());
}

std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch, bool defaults026) {
#if defined(_WIN32)
    return arch_default_env(gcn_arch, defaults026, true);
#else
    return arch_default_env(gcn_arch, defaults026, false);
#endif
}

std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch, bool defaults026, bool windows) {
    std::vector<std::pair<std::string, std::string>> t;
    if (strata::kernels::gfx_arch_is_gfx12(gcn_arch)) {
        // gfx1200 / gfx1201 (RDNA4): the PROMPT half of the gfx1151 table (reported exact on an R9700 in #1478, re-checked
        // bitwise in notes-142-amd.md) plus four decode switches measured one at a time on an R9700 in 0.2.7 (p620, IQ3_S,
        // engine 0.1.51, 8K prompt, rotated pairs): ATTN_LANECELL + ATTN_MERGE_V2 +1.9%, TSUM +1.9%, EXPDEC +0.6%, GDN_SPLIT
        // 0.0%; the four together +3.5% (3/3 pairs) with the answer md5 identical to the default.  The rest of the gfx1151
        // decode half stays out (PF_GEMM/PF_FUSED are bit-changing; Q8_PACKED, SH_STREAM and QFUSE were measured on other
        // cards only, and the whole half measured slower here when it was tried as one block).
        // STRATA_HCD_EXACT is left out: it needs the gfx1151 hipBLASLt tuning table, finds none on gfx12, says so on every start
        // and does nothing (measured: with and without it identical, 0.0 to +0.1%).
        // STRATA_GDN_CONVL2 is AMD-only on purpose: it is not bitwise on CUDA 13.3 sm_120 (#1522), and this table is never
        // consulted by a CUDA build.
        const char* off12 = std::getenv("STRATA_GFX12_DEFAULTS");
        if (off12 != nullptr && off12[0] == '0') return t;
        t = {
            {"STRATA_GDN_HEAD", "1"},
            {"STRATA_GDN_PP", "2"},
            {"STRATA_GDN_CONVL2", "1"},
            {"STRATA_GDN_NOY", "1"},
            {"STRATA_CVEC_FUSE", "1"},
        };
        if (defaults026) return t;   // STRATA_DEFAULTS=026: v0.2.6's table, the prompt half only
        // decode (bit-identical on the R9700 measurement above)
        t.push_back({"STRATA_ATTN_LANECELL", "1"});
        t.push_back({"STRATA_ATTN_MERGE_V2", "1"});
        t.push_back({"STRATA_TSUM", "1"});
        t.push_back({"STRATA_EXPDEC", "1"});
        return t;
    }
    if (!strata::kernels::gfx_arch_is_gfx1151(gcn_arch)) return t;
    const char* off = std::getenv("STRATA_GFX1151_DEFAULTS");
    if (off != nullptr && off[0] == '0') return t;
    // Exact (bitwise) on Aurora, each with its measured gain in aurora_s23.md:
    t = {
        // prompt
        {"STRATA_GDN_HEAD", "1"},        // the GDN recurrence, four lanes a column + the grid-stride norm (S23)
        {"STRATA_GDN_PP", "2"},          //   two columns per lane: 64K prompt +2.9% (stream A, round 2)
        {"STRATA_GDN_CONVL2", "1"},      // conv + q/k L2 norm in one kernel: 64K -0.3 s (stream A)
        {"STRATA_GDN_NOY", "1"},         // no dead FP32 store of the norm (stream A)
        {"STRATA_CVEC_FUSE", "1"},       // a steered layer's write, control vector and next norm in one pass (S23)
        {"STRATA_HCD_EXACT", "1"},       // the HC down GEMM in hipBLASLt solution 1176's k order: 64K +1.8% (stream S)
        {"STRATA_PF_PAD", "1"},          // padded GEMM row strides (acts with STRATA_PF_GEMM; inert without it) (S23)
        {"STRATA_TOPK_REG_MIN_BLOCKS", "4096"},   // the register top-k from 16K cells, not 28K: same ids; per 2,048 queries the ref takes 1.0 / 2.4 / 3.9 ms at 16K / 20K / 24K cells, the scan 0.87 / 0.93 / 1.00
        // decode
        {"STRATA_Q8_PACKED", "1"},       // the packed Q8_0 decode layout: +2.3% (S26)
        {"STRATA_Q6_PACKED", "1"},       // the packed Q6_K heads (S26)
        {"STRATA_MMVF_ROWS", "1"},       // bf16 multi-row GEMV, 4 rows per block (S25)
        {"STRATA_ATTN_LANECELL", "1"},   // decode attention scores, one cell per thread (S25)
        {"STRATA_EXPERT_V2", "1"},       // the grouped decode experts, IQ3_S gate/up + IQ4_NL down (S26)
        {"STRATA_TSUM", "1"},            // several warp sums as one transposed butterfly (S26)
        {"STRATA_LFUSE", "1"},           // fewer launches around the shared expert and the KV append (S26)
        {"STRATA_GDN_SPLIT", "1"},       // the GDN step over 4 blocks per head (S25/S26)
        {"STRATA_QFUSE", "1"},           // activation q8_1 images written by their producers (S26)
        {"STRATA_PLE_BATCH", "1"},       // the verify window's PLE key / value projections at once (S25)
        {"STRATA_SH_STREAM", "1"},       // the shared expert on its own stream: decode +1.8% / +6.7% (UD-Q4_K_XL) (140-m)
    };
    // 0.2.7: Windows gfx1151 leaves the shared-expert fork off (the HIP default).  Interleaved A/B on a Strix Halo Windows PC (Qwen
    // IQ3_S, engine 0.1.51, 9 chat answers each, 4K prompt): decode 43.9 / 43.6 / 43.2 tok/s with it, 48.3 / 46.0 / 46.6 without
    // (+6.9% median), prompts unchanged, the token ids identical either way (docs/STRIX_HALO_WINDOWS.md).  Linux gfx1151 keeps the
    // fork on until it is measured there.  STRATA_SH_STREAM=1 turns it back on; STRATA_DEFAULTS=026 keeps v0.2.6's table.
    if (windows && !defaults026) {
        for (auto it = t.begin(); it != t.end(); ++it) {
            if (it->first == "STRATA_SH_STREAM") { t.erase(it); break; }
        }
    }
    return t;
}

bool igpu_prompt_defaults_arch(const char* gcn_arch) {
    return strata::kernels::gfx_arch_is(gcn_arch, "gfx1103") || strata::kernels::gfx_arch_is(gcn_arch, "gfx1150") ||
           strata::kernels::gfx_arch_is(gcn_arch, "gfx1151");
}

bool igpu_prompt_set_enabled(const char* gcn_arch, bool windows) {
    const char* off = std::getenv("STRATA_IGPU_DEFAULTS");
    if (off != nullptr && off[0] == '0') return false;
    const char* e = std::getenv("STRATA_IGPU_PROMPT_SET");
    if (e != nullptr && e[0] == '1') return true;
    if (e != nullptr && e[0] == '0') return false;
    // 0.2.8: Strix Halo (gfx1151) gets the matrix-core prompt set on Windows by default.  Measured on the strix box
    // (Qwen3.8-Flash-Next IQ3_S, engine 0.1.52, ids prompts of 4,095 / 8,191 tokens, interleaved pairs, one engine at a
    // time): 4K 690 -> 1,090 and 8K 785 -> 1,183 on the f-024-strix-pp branch (0.2.3 engine, the same switches as
    // environment variables), i.e. x1.58 / x1.51; the first-token KL against the accepted HC_Q8 bar was 0.0088 mean
    // (0.0228 allowed), top-1 the same on every prompt.  Linux gfx1151 stays opt-in until it is measured there; gfx1103
    // keeps its own rule below.
    if (windows && strata::kernels::gfx_arch_is(gcn_arch, "gfx1151")) return true;
    return kIgpuPromptSetDefaultOn && !windows && strata::kernels::gfx_arch_is(gcn_arch, "gfx1103");
}

bool igpu_uma_arena_default(const char* gcn_arch, bool integrated, bool windows, bool on) {
    if (!integrated) return false;
    const char* reg = std::getenv("STRATA_UMA_REGISTER");
    if (reg != nullptr && reg[0] == '1') return false;
    // 0.2.1.2: a Strix Halo (gfx1151) on Windows.  Registering the 47 GiB arena with the driver took ALL the free VRAM (91 GiB before,
    // 0 after; BIOS carve-out 64 GB + 32 GB shared): the expert cache got 0 slots and the engine exited ("the embedding buffer
    // failed"), 3 of 3 starts (notes-022-strix.md).  Unregistered, the arena is held by the working-set lock and the cache gets
    // 24,576 slots.  Independent of `on` (that is the 780M's opt-in).  STRATA_ARENA_PIN_GIB=N / 0 still registers it.
    if (windows) return strata::kernels::gfx_arch_is(gcn_arch, "gfx1151");
    return on && strata::kernels::gfx_arch_is(gcn_arch, "gfx1103");
}

std::vector<std::pair<std::string, std::string>> igpu_default_env(const char* gcn_arch, bool integrated) {
    std::vector<std::pair<std::string, std::string>> t;
    if (!integrated || !igpu_prompt_defaults_arch(gcn_arch)) return t;
#if defined(_WIN32)
    const bool win = true;
#else
    const bool win = false;
#endif
    if (!igpu_prompt_set_enabled(gcn_arch, win)) return t;
    // Measured on a Radeon 780M (gfx1103, P2006, Q2_0, notes-153-igpu.md / notes-148-amd.md): prompts 109.5 -> 214.7 tok/s
    // with the first four, 233 with HC_UPMIX, decode unchanged.  Dense projections are 63% of a prompt's GPU time there.
    t = {
        {"STRATA_HIP_WMMA", "1"},      // the prompt attention on matrix cores
        {"STRATA_PF_GEMM", "1"},       // the prompt's FP16 projections on the 128 x 256 WMMA GEMM
        {"STRATA_PA_FAST", "1"},       // the prompt attention with single FP16 q and p
        {"STRATA_SELECT_WMMA", "1"},   // the block scorer on matrix cores
        {"STRATA_HC_UPMIX", "1"},      // the hyper-connection up projection with its epilogue
    };
    // gfx1151 (Strix Halo) on Windows: SELECT_WMMA is in since 0.2.9 (x1.008 at 8K, x1.037 at 32K, x1.077 at 64K), and PF_FUSED (the
    // fused int8 gate/up prompt experts, #136) measured x1.12 there and joins the set.  The 780M and the 890M keep the
    // table above exactly as measured on them.
    if (win && strata::kernels::gfx_arch_is(gcn_arch, "gfx1151")) {
        t = {
            {"STRATA_HIP_WMMA", "1"},
            {"STRATA_PF_GEMM", "1"},
            {"STRATA_PA_FAST", "1"},
            {"STRATA_HC_UPMIX", "1"},
            {"STRATA_PF_FUSED", "1"},
            {"STRATA_SELECT_WMMA", "1"},   // 0.2.9: the block scorer on matrix cores; not bitwise (about 10 of 2,051 selected cells flip at the cutoff
                                           // for queries past ~5K tokens), x1.008 at 8K, x1.037 at 32K, x1.077 at 64K (STRIX-Q-RESULTS.md)
            // 0.2.9 (owner, 10-30: a consistent Strix Halo speed gain goes on by default): the prompt experts of every covered
            // format on the F16 matrix-core kernels (the Q8_0 down layers of the Unsloth files included).  Windows gfx1151,
            // interleaved x3, same 16 greedy ids: IQ3_S 4K 1,187 -> 1,296 and 8K 1,285 -> 1,399 tok/s (x1.09), UD-IQ4_XS 4K 1,119 ->
            // 1,221 and 8K 1,180 -> 1,270; decode unchanged.  The rounding differs from the int8 kernels (KL is information only).
            {"STRATA_PF_F16", "1"},
        };
    }
    return t;
}

static void set_default(const std::pair<std::string, std::string>& kv) {
#if defined(_WIN32)
    _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
    setenv(kv.first.c_str(), kv.second.c_str(), 0);
#endif
}

std::vector<std::string> apply_arch_defaults(const char* gcn_arch, bool integrated) {
    std::vector<std::string> set;
    for (const auto& kv : arch_default_env(gcn_arch)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, whatever it is
        set_default(kv);
        set.push_back(kv.first);
    }
    if (!set.empty()) {
        const bool g12 = strata::kernels::gfx_arch_is_gfx12(gcn_arch);
        std::fprintf(stderr, "strata: %s: %zu exact speed switches on by default (%s=0 turns "
                             "them off; a switch you set is kept): ",
                     g12 ? "gfx12 (RDNA4)" : "gfx1151 (Strix Halo)", set.size(),
                     g12 ? "STRATA_GFX12_DEFAULTS" : "STRATA_GFX1151_DEFAULTS");
        for (size_t i = 0; i < set.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", set[i].c_str() + 7);
        std::fprintf(stderr, "\n");
    }
    std::vector<std::string> igpu;
    for (const auto& kv : igpu_default_env(gcn_arch, integrated)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;
        set_default(kv);
        igpu.push_back(kv.first);
    }
    if (!igpu.empty()) {
        std::fprintf(stderr, "strata: integrated GPU (%s): the matrix-core prompt set is on (1.6x to 2x faster prompts on a "
                             "Radeon 780M, the answers differ in the last bits): ", gcn_arch);
        for (size_t i = 0; i < igpu.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", igpu[i].c_str() + 7);
        std::fprintf(stderr, ".  Turn one off with NAME=0 (for example STRATA_HC_UPMIX=0), all of them with STRATA_IGPU_PROMPT_SET=0");
#if defined(_WIN32)
        std::fprintf(stderr, ".  Not yet tested on Windows");
#endif
        std::fprintf(stderr, "\n");
        set.insert(set.end(), igpu.begin(), igpu.end());
    }
    return set;
}

}  // namespace strata::core
