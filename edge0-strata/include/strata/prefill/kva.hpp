// include/strata/prefill/kva.hpp - STRATA_KVA: the approximate late-layer prompt read (opt-in since 0.2.7).
//
// After layer `split` the prompt's late layers need only their recurrent state (GDN recurrence + conv history)
// and their KV/indexer caches; the attention output, the output projection, the MoE half and its expert
// transfers have no remaining consumer for the approximate positions.  The plan is:
//
//   positions [0, n - suffix)          approximated: late layers run state-only on a predicted block input
//   positions [n - suffix, n)          fully computed (the suffix), request-global
//
// v1 keeps the suffix on a CHUNK boundary (a chunk is approximated only when its last position is before the
// frontier), so no layer ever has to split one chunk into two regimes.  The late layers' block input is the
// hyper-connection residual R (10,240 floats = hc * n_embd) that the chunk's layer loop carries; it comes from
//   identity  the residual entering layer `split` (no projector at all, the zero-cost baseline),
//   oracle    a captured real input per late layer (STRATA_KVA_ORACLE=<dir>; the plumbing/parity mode),
//   linear    a fitted low-rank projector (STRATA_KVA_PROJECTOR=<file>; not implemented in this step).
//
// 0.2.7: the read is OFF unless STRATA_KVA=1 is set (the quality runs found lost records and invented rows in long
// extraction and agent tasks, docs/KVA.md).  Unset, nothing here runs and the engine is 0.2.6's.  Switched on, it applies
// to a cold prompt of STRATA_KVA_MIN_PROMPT tokens or more on one CUDA/HIP device, approximating only the layers from
// 36 on, with a 2048-token exact tail (the configuration the 0.2.7 quality runs found cleanest; STRATA_KVA_SPLIT and
// STRATA_KVA_SUFFIX override).  Everything is read from the environment once per process.
//
// What the engine tells this header at start-up (kva_engine()): whether its configuration rules the read out for the
// whole process (batch slots, a layer split, a peer device).
//
// The draft layer (MTP) builds its K/V from the final rows of the prompt, which an approximated chunk does not have.
// Its window is NOT changed, and by default neither is anything else: the draft cells of an approximated chunk are
// built from the rows that chunk has and the draft layer attends to them (measured: draft acceptance after a KVA
// read stays within noise on the P100).  The research option STRATA_KVA_DRAFT_FLOOR=1 gives the draft layer a "cell
// floor" per session (MtpDrafter::set_cell_floor; kva_floor.hpp keeps it with the tokens of the session,
// window_start.hpp is the one rule the kernel and the tests share): the first cell built from a real row, below
// which no draft cell is built or read.  The target model verifies every token whatever the draft read.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::prefill {

/// Does this build have the state-only late-layer path?  CUDA and HIP: yes.  The Metal engine compiles prefill.cpp
/// against its own kernel library, which has none.
#if defined(STRATA_GPU_METAL)
inline constexpr bool kKvaBackend = false;
#else
inline constexpr bool kKvaBackend = true;
#endif

/// The reason decline() gives for a model whose shape KVA was not measured on (prefill.cpp prints it with the numbers).
inline constexpr const char* kKvaShapeReason = "this model's shape is not one KVA was measured on";

/// The fully computed tail of every approximated read.
inline constexpr int64_t kKvaSuffixDefault = 2048;
/// The first approximated layer when switched on: 36 of 48 (the cleanest measured; 24 lost records and agent loops).
inline constexpr int64_t kKvaSplitDefault = 36;

/// What the running engine knows and the environment cannot say.
struct KvaEngine {
    std::string blocked;           ///< non-empty: this process never reads approximately, and why
};

inline KvaEngine& kva_engine() {
    static KvaEngine e;
    return e;
}

struct KvaConfig {
    enum class Mode { Identity, Oracle, Linear };
    enum class Only { All, Gdn, Qsa };
    /// What a late layer of an approximated chunk still does: StateOnly (the default) advances only its state (GDN
    /// recurrence, K/V and indexer pages); Moe runs the whole attention half (outputs into the residual) and skips only
    /// the MoE half and its expert transfers.
    enum class Skip { StateOnly, Moe };

    bool on = false;             ///< STRATA_KVA=1 (and this build has a path); unset or 0: off
    bool explicit_on = false;    ///< STRATA_KVA was set to something other than 0: every declined read says why
    bool capture = false;        ///< STRATA_KVA_CAPTURE=<dir>: dump every late layer's real input rows
    bool capture_fp32 = false;   ///< STRATA_KVA_CAPTURE_FP32=1: fp32 records (else bf16; fp32 is for parity runs)
    bool oracle_fp32 = false;    ///< STRATA_KVA_ORACLE_FP32=1: the oracle files are fp32 (else bf16)
    bool parity = false;         ///< STRATA_KVA_PARITY=1: report the state-only path against the full path
    bool any_model = false;      ///< STRATA_KVA_ANY_MODEL=1 (research): also run on a shape KVA was not measured on, with a warning
    bool draft_floor = false;    ///< STRATA_KVA_DRAFT_FLOOR=1: the draft layer skips the cells of approximated chunks (research; default: it reads them)
    int64_t split = kKvaSplitDefault;   ///< STRATA_KVA_SPLIT: first approximated layer (zero-based)
    int64_t suffix_env = 0;      ///< STRATA_KVA_SUFFIX: fully computed tail positions (0 = not set: kKvaSuffixDefault)
    int64_t min_prompt = 4096;   ///< STRATA_KVA_MIN_PROMPT: shorter prompts stay on the full path
    Mode mode = Mode::Identity;  ///< STRATA_KVA_MODE=identity|oracle|linear
    Only only = Only::All;       ///< STRATA_KVA_ONLY=all|gdn|qsa: which late layers take the approximate path
    Skip skip = Skip::StateOnly; ///< STRATA_KVA_SKIP=moe: skip only the MoE half of the late layers (opt-in)
    std::string oracle_dir;      ///< STRATA_KVA_ORACLE
    std::string capture_dir;     ///< STRATA_KVA_CAPTURE
    std::string projector;       ///< STRATA_KVA_PROJECTOR

    /// The fully computed tail of a read: STRATA_KVA_SUFFIX when set, otherwise 2048.
    int64_t effective_suffix() const { return suffix_env > 0 ? suffix_env : kKvaSuffixDefault; }

    /// Does the draft layer need the cell floor at all?  Only when a read can approximate something: KVA on, not ruled
    /// out for the process, and a draft layer loaded.  Otherwise the draft layer's attention is 0.2.6's own code.
    bool wants_cell_floor(const KvaEngine& e, bool has_drafter) const {
        return kKvaBackend && on && draft_floor && e.blocked.empty() && has_drafter;
    }

    /// Why this read cannot take the approximate path, or nullptr when it can: a cold read of one prompt (not a
    /// continuation, no layer split, no pictures, no DFlash2 taps), on an engine that allows it, long enough for a
    /// suffix to leave approximated positions behind it.  `routine` (optional) is set when the reason is an
    /// everyday one (a continuation, a short prompt) that is not worth a log line.
    const char* decline(const KvaEngine& e, int64_t pos0, int64_t n, int64_t lb, int64_t le, int64_t n_layers,
                        bool pictures = false, bool taps = false, bool* routine = nullptr,
                        bool unmeasured_shape = false) const {
        if (routine) *routine = false;
        if (!on) return "off (STRATA_KVA is not 1)";
        if (capture) return "capture mode";
        if (!e.blocked.empty()) return e.blocked.c_str();
        if (pos0 != 0) {
            if (routine) *routine = true;
            return "a continuation read (pos0 != 0)";
        }
        if (lb != 0 || (le >= 0 && le != n_layers)) return "a layer split (not the whole trunk)";
        if (pictures) return "a prompt with pictures";
        if (taps) return "the DFlash2 drafter reads the late layers' hidden states";
        if (n < min_prompt) {
            if (routine) *routine = true;
            return "the prompt is shorter than STRATA_KVA_MIN_PROMPT";
        }
        if (n <= effective_suffix() + 64) {
            if (routine) *routine = true;
            return "the suffix leaves too few positions to approximate";
        }
        // the split layer, the state-only layers and the tail were measured on one geometry (Qwen3.8-Flash-Next, 48
        // layers, 512 experts); another model reads exactly unless the research override asks for it
        if (unmeasured_shape && !any_model) return kKvaShapeReason;
        return nullptr;
    }
};

inline int64_t kva_env_i64(const char* name, int64_t dflt) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return dflt;
    return (int64_t) std::atoll(v);
}

inline bool kva_env_flag(const char* name, bool dflt) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return dflt;
    return v[0] != '0';
}

/// The settings as the environment says now (uncached: the tests call it after changing the environment).
inline KvaConfig kva_config_from_env() {
    KvaConfig k;
    const char* sw = std::getenv("STRATA_KVA");
    const bool set = sw != nullptr && *sw != '\0';
    k.on = set && sw[0] != '0';
    k.explicit_on = k.on;
    if (!kKvaBackend) k.on = false;
    k.split = kva_env_i64("STRATA_KVA_SPLIT", kKvaSplitDefault);
    k.suffix_env = std::max<int64_t>(0, kva_env_i64("STRATA_KVA_SUFFIX", 0));
    k.min_prompt = kva_env_i64("STRATA_KVA_MIN_PROMPT", 4096);
    k.parity = kva_env_flag("STRATA_KVA_PARITY", false);
    k.draft_floor = kva_env_flag("STRATA_KVA_DRAFT_FLOOR", false);
    k.any_model = kva_env_flag("STRATA_KVA_ANY_MODEL", false);
    if (const char* d = std::getenv("STRATA_KVA_CAPTURE"); d != nullptr && *d != '\0') {
        k.capture = true;
        k.capture_dir = d;
        k.capture_fp32 = kva_env_flag("STRATA_KVA_CAPTURE_FP32", false);
    }
    if (const char* d = std::getenv("STRATA_KVA_ORACLE"); d != nullptr && *d != '\0') {
        k.oracle_dir = d;
        k.mode = KvaConfig::Mode::Oracle;
        k.oracle_fp32 = kva_env_flag("STRATA_KVA_ORACLE_FP32", false);
    }
    if (const char* d = std::getenv("STRATA_KVA_PROJECTOR"); d != nullptr && *d != '\0') {
        k.projector = d;
        k.mode = KvaConfig::Mode::Linear;
    }
    if (const char* m = std::getenv("STRATA_KVA_MODE"); m != nullptr && *m != '\0') {
        if (std::string(m) == "oracle") k.mode = KvaConfig::Mode::Oracle;
        else if (std::string(m) == "linear") k.mode = KvaConfig::Mode::Linear;
        else k.mode = KvaConfig::Mode::Identity;
    }
    if (const char* sk = std::getenv("STRATA_KVA_SKIP"); sk != nullptr && std::string(sk) == "moe") k.skip = KvaConfig::Skip::Moe;
    if (const char* o = std::getenv("STRATA_KVA_ONLY"); o != nullptr && *o != '\0') {
        if (std::string(o) == "gdn") k.only = KvaConfig::Only::Gdn;
        else if (std::string(o) == "qsa") k.only = KvaConfig::Only::Qsa;
        else k.only = KvaConfig::Only::All;
    }
    return k;
}

/// The process's KVA settings, parsed once (the engine's own options are parsed once too).
inline const KvaConfig& kva_config() {
    static const KvaConfig c = kva_config_from_env();
    return c;
}

}  // namespace strata::prefill
