// include/strata/program/batch_groups.hpp - how many pipelined slot groups `--batch` runs with.
//
// `--batch-groups N` pipelines N groups of slots through the stages of a layer split; `--batch-groups auto` picks the
// most groups (at most one per stage) that divide the slots.  Since 0.1.41 a layer split with `--batch` >= 2 and no
// `--batch-groups` at all means `auto`.  The pipelined path does not run `--batch-mtp`'s drafts or the batch
// adaptive tier (`--adapt-async 1` between batch windows).  `--adapt-async 1` with the number of groups NOT given still
// makes `auto` resolve to one group (and says why).  `--batch-mtp` does not any more (0.1.43.1): on 2x A4000 and 2x RTX
// 3090 four concurrent streams lost 25-30% with the serial MTP windows against the pipelined default, so the default
// pipelines and `--batch-mtp` is left off there (said at start); STRATA_BATCH_MTP_SERIAL=1 or `--batch-groups 1` keeps
// the serial windows with the slots' drafts (the 0.1.42/0.1.43 behaviour).  A number or `auto` given on the command line
// is honoured as before.
//
// The pipelined path starts no round of the adaptive VRAM tier on its own (the serial batch windows start one every
// `--adapt-every` windows): opt-in, STRATA_PIPE_ADAPT_EVERY=N lets it run one after every N group windows, once the groups
// in flight have finished (pipe_adapt_every / pipe_adapt_due below).
//
// Pure (no GPU, no environment read): generate.cpp passes in what it knows.
#pragma once

#include <cstdlib>

namespace strata::program::batch_groups {

/// STRATA_PIPE_ADAPT_EVERY's value -> the interval in group windows; 0 (off: the pipelined path adapts nothing, as before)
/// when unset, empty, not a number, 0 or negative.
inline int pipe_adapt_every(const char* value) {
    if (value == nullptr) return 0;
    const int n = std::atoi(value);
    return n > 0 ? n : 0;
}

/// What the pipelined loop knows when it decides on a round.
struct PipeAdapt {
    int every = 0;               ///< pipe_adapt_every()
    bool tier_on = false;        ///< the adaptive tier counts the routing here (it is on and not the whole model)
    bool async_tier = false;     ///< --adapt-async 1 runs its own rounds
    bool prompt_read = false;    ///< a prompt is read beside the slots (it may hold a loan of cache slots)
    bool swaps_pending = false;  ///< the last round's copies have not landed yet
    long long windows = 0;       ///< group windows completed so far
    long long last = 0;          ///< `windows` when the last round ran
};

/// A round is due: the loop starts no new group, and runs the round once nothing is in flight.
inline bool pipe_adapt_due(const PipeAdapt& p) {
    return p.every > 0 && p.tier_on && !p.async_tier && !p.prompt_read && !p.swaps_pending && p.windows - p.last >= p.every;
}

/// What kept `auto` from pipelining.
enum class Serial { None, BatchMtp, AdaptAsync, TwoCardsSerialFaster };

struct Input {
    bool set = false;          ///< --batch-groups was given (a number or `auto`)
    bool auto_given = false;   ///< ... as `auto`
    int given = 1;             ///< the number given (1 when none)
    int batch = 0;             ///< the slots
    int later_stages = 0;      ///< the layer split's stages after CUDA0 (0: no layer split)
    bool batch_mtp = false;    ///< --batch-mtp is on (it passed the gates that do not concern the groups)
    bool adapt_async = false;  ///< --adapt-async 1 was asked for
    bool mtp_serial = false;   ///< STRATA_BATCH_MTP_SERIAL=1: --batch-mtp with no --batch-groups runs one serial group (0.1.43)
    bool nv_serial_faster = false;   ///< opt-in (see nv_serial_wanted): NVIDIA build, two cards, STRATA_BATCH_GROUPS_NV_SERIAL=1 or STRATA_DEFAULTS=026 - one serial group, the 0.2.1-0.2.6 default
    bool mtp_piped = false;    ///< STRATA_PIPED_MTP=1: the pipelined groups draft too (each group carries its slots' rows)
};

struct Result {
    int groups = 1;
    bool from_auto = false;            ///< the number came from `auto` (given or the 0.1.41 default)
    Serial serial = Serial::None;      ///< `auto` would have pipelined, but this feature needs the serial windows
    int would_be = 1;                  ///< what `auto` would have picked without it (== groups unless `serial`)
    bool mtp_dropped = false;          ///< --batch-mtp met the default pipeline: the pipeline wins, --batch-mtp is off
    bool mtp_piped = false;            ///< STRATA_PIPED_MTP: --batch-mtp's drafts run in the pipelined groups
};

/// Whether two NVIDIA cards run one serial group when --batch-groups is not given.  Not by default since 0.2.7: the pipelined groups
/// measured faster on 2x RTX A6000 (IQ3_S, every expert resident) and on 2x RTX A4000 (Q2_0 pack); the serial default of 0.2.1-0.2.6
/// had measured faster on 2x A4000 with IQ3_XXS.  STRATA_BATCH_GROUPS_NV_SERIAL asks for it with any value but 0 (an explicit 0 or
/// non-zero value wins over the master switch); with the variable unset or empty, STRATA_DEFAULTS=026 (v0.2.6's behaviour) asks for
/// it.  `env_value` is the variable's value (null when unset).
inline bool nv_serial_wanted(const char* env_value, bool defaults_026) {
    if (env_value != nullptr && env_value[0] != '\0') return env_value[0] != '0';
    return defaults_026;
}

/// The most groups, at most one per stage of the layer split (`later_stages` + 1 GPUs), that divide `batch`.
inline int auto_groups(int later_stages, int batch) {
    int best = 1;
    if (later_stages <= 0) return best;
    for (int d = 2; d <= later_stages + 1 && d <= batch; ++d)
        if (batch % d == 0) best = d;
    return best;
}

inline Result resolve(const Input& in) {
    Result r;
    r.groups = in.given;
    r.would_be = in.given;
    const bool defaulted = !in.set && in.later_stages > 0 && in.batch > 1;   // the 0.1.41 default
    if (in.auto_given || defaulted) {
        r.from_auto = true;
        r.would_be = auto_groups(in.later_stages, in.batch);
        r.groups = r.would_be;
        if (!in.set && r.would_be > 1) {   // only the default gives way; `--batch-groups auto` asked for the pipeline
            if (in.batch_mtp && in.mtp_serial) r.serial = Serial::BatchMtp;
            else if (in.nv_serial_faster && in.later_stages == 1 && !in.batch_mtp && !in.adapt_async) r.serial = Serial::TwoCardsSerialFaster;
            else if (in.adapt_async) r.serial = Serial::AdaptAsync;
            if (r.serial != Serial::None) r.groups = 1;
            r.mtp_dropped = in.batch_mtp && r.serial == Serial::None;
        }
    }
    // STRATA_PIPED_MTP: the pipelined groups draft too.  A group's window carries each of its slots' rows (the slot's
    // token and that slot's own proposals), so any number of slots per group drafts.  This sits outside the block
    // above on purpose: with --batch-groups given (a number or `auto`) the operator asked for those groups, and the
    // drafts run in them too.  Unset (mtp_piped false) nothing here changes a result.
    r.mtp_piped = in.batch_mtp && in.mtp_piped && r.serial == Serial::None && r.groups > 1;
    if (r.mtp_piped) r.mtp_dropped = false;
    return r;
}

inline const char* serial_what(Serial s) {
    return s == Serial::BatchMtp ? "--batch-mtp" : s == Serial::AdaptAsync ? "--adapt-async 1"
         : s == Serial::TwoCardsSerialFaster ? "serial windows (STRATA_BATCH_GROUPS_NV_SERIAL=1 or STRATA_DEFAULTS=026 asked for them on two NVIDIA cards; --batch-groups auto pipelines)" : "";
}

}  // namespace strata::program::batch_groups
