#pragma once
// The row layout of a batch window (Verifier::stage_batch), as pure logic so a CPU test can pin it down.
//
// A batch window carries S rows, each a token of one slot.  A slot owns one row (plain --batch) or a contiguous group
// of rows at consecutive positions (--batch-mtp: the slot's current token and its MTP proposal; a layer split's every
// stage sees the same layout).  The rows travel between the stages of a layer split in hand-off buffers of
// `handoff_rows` rows, the window's group starting at row `hbase`.

#include <cstdint>

namespace strata::core {

/// nullptr when the layout is valid, else why not.  `rows[t]` is row t's slot, `pos[t]` its position.  `n_slots`: the
/// slots the verifier holds; `max_t`: its window capacity; `via_handoff`: the stage hands its rows on or takes them from
/// another stage (a layer split), so [hbase, hbase + S) must fit the `handoff_rows` rows of the hand-off buffers.
inline const char* batch_rows_error(const int* rows, int S, const int64_t* pos, int n_slots, int max_t, int hbase,
                                    bool via_handoff, int handoff_rows) {
    if (S < 1 || S > max_t || hbase < 0 || (via_handoff && hbase + S > handoff_rows))
        return "verify: batch rows out of range (init_slots)";
    for (int t = 0; t < S; ++t) {
        if (rows[t] < 0 || rows[t] >= n_slots) return "verify: a batch row's slot is out of range";
        for (int u = 0; u < t - 1; ++u)
            if (rows[u] == rows[t] && rows[t - 1] != rows[t]) return "verify: a slot's proposed rows must be contiguous";
        if (t > 0 && rows[t] == rows[t - 1] && pos[t] != pos[t - 1] + 1)
            return "verify: proposed rows must have consecutive positions";
    }
    return nullptr;
}

}  // namespace strata::core
