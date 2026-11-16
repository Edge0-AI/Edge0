// include/strata/prefill/copy_events.hpp - which streamed expert copies get a "copied" event (#4005).
//
// The streamed prompt walk issues one host-to-device copy per expert entry on the copy stream and the compute stream
// waits for it.  By default every entry records its own event (every == 1) and the compute stream waits on the event of
// the entry it needs.  STRATA_PREFILL_COPY_EVENT_EVERY=N (tester switch) records every N-th entry only, and the
// compute stream waits on the first recorded entry at or after the one it needs: the copy stream is in order, so that
// event covers it.  The two rules are pure functions so a host test can check them without a GPU.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill {

/// The value of STRATA_PREFILL_COPY_EVENT_EVERY: 1 (every entry) unless the text is a number above 1; at most 64.
inline int copy_event_every_from(int parsed) { return parsed > 1 ? (parsed < 64 ? parsed : 64) : 1; }

/// Entry `idx` of `total` records its copied event right after its copy: always with every == 1; else every N-th entry
/// and the last one.
inline bool copy_event_due(size_t idx, size_t total, int every) {
    return every <= 1 || (idx + 1) % (size_t) every == 0 || idx + 1 == total;
}

/// The entry whose recorded event a consumer waits on for entry `k`, when the entries [0, recorded) have had their
/// events decided (`recorded` = 1 + the last recorded entry) and `is_recorded(r)` says whether entry r has one.  The
/// first recorded entry at or after k; the last recorded entry (recorded - 1) when none is found, which covers k as
/// well.  Precondition: k < recorded.
template <class IsRecorded>
inline size_t copy_event_entry(size_t k, size_t recorded, IsRecorded is_recorded) {
    for (size_t r = k; r < recorded; ++r)
        if (is_recorded(r)) return r;
    return recorded - 1;
}

/// STRATA_PREFILL_COPY_WAIT_DEDUPE: the copy stream may skip its wait on the release event of entry `idx`'s slot (`rel`)
/// when the wait it made for the previous entry was on the same event (`prev_rel`).  That holds only when both slots were
/// released in the current walk (idx - 1 >= ring), because the event of an earlier walk can be re-recorded by this walk's
/// first flush, and when the ring is longer than one MMQ group of `group` experts, because a flush of a whole ring would
/// re-record the event right after the previous slot's wait.
inline bool copy_wait_dedupe_ok(size_t idx, int ring, int group, int rel, int prev_rel) {
    return ring > group && idx > (size_t) ring && rel == prev_rel;
}

}  // namespace strata::prefill
