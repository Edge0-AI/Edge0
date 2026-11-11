// include/strata/program/draft_kv_plan.hpp - which code computes the draft layer's prompt K/V for a chunk.
//
// One GPU: the prompt path's batched pass (E-9, `Prefill::draft_kv`) tries first and the drafter's own
// per-8-row pass (`MtpDrafter::prefill`) takes what it declines.  A layer split: only the drafter's own pass
// ran, until `STRATA_SPLIT_MTP_BATCH=1` (opt-in) lets the last stage's prompt path try E-9 as well.  The
// drafter's K/V then come from Q8_1 x Q8_0 MMQ instead of its own mmvq, so its drafts can differ a little; the
// target's tokens are decided by the verify window and do not.
//
// Pure (no GPU, no environment read): generate.cpp passes in what it knows.
#pragma once

#include <cstdlib>

namespace strata::program::draft_kv {

/// `STRATA_SPLIT_MTP_BATCH`'s value: any whole number other than 0 turns it on; unset, empty, `0` and text
/// that is not a number leave it off.
inline bool split_env_on(const char* value) { return value != nullptr && std::atoi(value) != 0; }

struct Plan {
    bool try_batched = false;      ///< call Prefill::draft_kv for this chunk
    const char* why = nullptr;     ///< when `try_batched` is false only because of a missing prerequisite: what is missing
};

/// `use_mtp`: the engine runs a draft layer. `multi_gpu`: a layer split. `split_env`: split_env_on(...) of the
/// variable. `has_native_embed`: the GGUF-form token embedding table exists (a --native pack): the split's
/// last stage reads it, a single GPU's path can also gather from its own weights.
inline Plan plan(bool use_mtp, bool multi_gpu, bool split_env, bool has_native_embed) {
    Plan p;
    if (!use_mtp) return p;
    if (!multi_gpu) {
        p.try_batched = true;
        return p;
    }
    if (!split_env) return p;   // the default on a split: the drafter's own pass, nothing to say
    if (!has_native_embed) {
        p.why = "the split's last stage has no GGUF-form token embedding table (not --native)";
        return p;
    }
    p.try_batched = true;
    return p;
}

}  // namespace strata::program::draft_kv
