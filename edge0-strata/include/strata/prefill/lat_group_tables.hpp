// include/strata/prefill/lat_group_tables.hpp - the latent A stage's entry-major row maps, built once a layer
// instead of once per MMQ group.
//
// A group of experts is one contiguous run of buffer rows, and its rows are the experts' entries in the order the
// walk visits `order`: expert after expert, each expert's [off[e], off[e] + cnt[e]).  The two tables are, per buffer
// row, that entry's xl row (what `quantize` reads) and the entry index itself (the down product's dst row).  Because
// the run is contiguous and the walk's group boundaries are just cuts of it, a group uses
// `ids = entry + base` / `quantize(xl, xl + base, ...)` with base = the rows before it - no per-group rebuild and no
// per-group upload of two row tables.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::prefill::lat {

struct Tables {
    std::vector<int32_t> xl;      // buffer row -> its entry's xl row
    std::vector<int32_t> entry;   // buffer row -> its (absolute) entry index
    int64_t rows = 0;             // the layer's rows
};

/// `order` the walk's experts in walk order, `off`/`cnt` the layer's per-expert entry ranges (off NOT monotonic in
/// expert id: the resident block is laid out first), `xl_row_of_entry` the layer's entry -> xl row map.
inline Tables build(const std::vector<int32_t>& order, const int32_t* off, const int32_t* cnt,
                    const std::vector<int32_t>& xl_row_of_entry) {
    Tables t;
    size_t n = 0;
    for (int32_t e : order) n += (size_t) cnt[e];
    t.xl.reserve(n);
    t.entry.reserve(n);
    for (int32_t e : order) {
        const int32_t o = off[e];
        for (int32_t i = 0; i < cnt[e]; ++i) {
            const int32_t v = o + i;
            t.xl.push_back(xl_row_of_entry[(size_t) v]);
            t.entry.push_back(v);
        }
    }
    t.rows = (int64_t) t.entry.size();
    return t;
}

}  // namespace strata::prefill::lat
