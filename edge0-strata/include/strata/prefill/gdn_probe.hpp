// include/strata/prefill/gdn_probe.hpp - the GDN recurrence dispatch, made observable and forceable by tests.
//
// strata::prefill::gdn_recurrence picks its kernel from environment switches that are read once per process.  A
// test or benchmark that links the real strata_prefill library installs a GdnProbe on its thread to
//   * read back which path ran for the last call (`selected`) and why a path that was asked for did not run
//     (`reason`), instead of guessing from timings, and
//   * force one path for the call (`force`): a forced path that is not eligible launches NOTHING and reports
//     kGdnNone plus the reason, so a harness can never time a silent fallback as if it were the candidate.
// Without a probe (the engine) the dispatch is byte-for-byte what it was: one thread-local pointer read.
// Header only; CUDA builds fill the probe (src/prefill/kernels.cu), other backends may adopt the same struct.
#pragma once

namespace strata::prefill {

enum GdnPath : int {
    kGdnNone = -1,     // out: nothing was launched
    kGdnAuto = 0,      // in: the production choice (the environment switches decide)
    kGdnKeyHead = 1,   // gdn_rec_kh_kernel: the three value heads of a key head in one thread (sm_80+, 64 blocks resident)
    kGdnPipe = 2,      // gdn_rec_cols_pipe_kernel: the software-pipelined column kernel
    kGdnCols = 3,      // gdn_rec_cols_kernel (STRATA_GDN_PIPELINE=0)
    kGdnChunked = 4,   // STRATA_GDN_CHUNKED: FP32 chunked prep + scan (CUDA, C = 32)
    kGdnTc = 5,        // STRATA_GDN_TC=1: chunked, scan on the tensor cores (CUDA sm_80+, 125 KB of shared memory a block)
    kGdnTcSplit = 6,   // the same scan with v / T / P single-buffered: 98 KB, fits the 99-KB cards (sm_86, sm_89, sm_120)
    kGdnTcS2 = 7,      // experiment: the split scan with 2 passes a product (lo.hi + hi.hi), not for production
    kGdnTcS1 = 8,      // experiment: the split scan with plain f16 operands (1 pass), not for production
    kGdnPathCount = 9
};

struct GdnProbe {
    // in
    int force = kGdnAuto;      // kGdnAuto: dispatch as production does; otherwise only this path may run
    bool any_sms = false;      // chunked: skip the 128-SM test, as STRATA_GDN_CHUNKED=2 does
    int tc_tiles = 0;          // tcs: value tiles a head (1, 2; 0 = STRATA_GDN_TC_TILES, else automatic: 2 on cards with >= 96 SMs)
    int gsb = 0;               // chunked: tokens per scratch super-block (a multiple of 32, at most 2048; 0 = 2048)
    // out (set by every gdn_recurrence call made on the thread that installed the probe)
    int selected = kGdnNone;   // the path that launched
    int wanted = kGdnAuto;     // the path that was asked for: `force`, or the first the environment enabled
    const char* reason = "";   // when selected != wanted: why the wanted path did not run
    long calls = 0;
};

inline thread_local GdnProbe* g_gdn_probe = nullptr;
inline void gdn_probe_install(GdnProbe* p) { g_gdn_probe = p; }
inline GdnProbe* gdn_probe_get() { return g_gdn_probe; }

inline const char* gdn_path_name(int p) {
    switch (p) {
        case kGdnKeyHead: return "keyhead";
        case kGdnPipe: return "pipe";
        case kGdnCols: return "cols";
        case kGdnChunked: return "chunked";
        case kGdnTc: return "tc";
        case kGdnTcSplit: return "tcs";
        case kGdnTcS2: return "tcs2";
        case kGdnTcS1: return "tcs1";
        case kGdnAuto: return "auto";
        default: return "none";
    }
}

}  // namespace strata::prefill
