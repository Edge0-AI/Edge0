// include/strata/ngram/ple_reader.hpp - plan v0.3 P2: the n-gram table read straight from the SSD.
//
// The table is 320,001,536 rows of 90 bytes (IQ4_NL, 26.8 GiB) or 160 (FP8 E4M3, 51.2 GB) and is never held in RAM: every row comes from an
// unbuffered 4 KiB read (platform::DirectFile). A token needs 16 rows on 16 different pages, and all 16 depend
// on the token itself, so the only time to hide them is the embedding plus layer 0. Hence the split API:
//
//     Ticket t = reader.issue(rows, n, out_raw);    // as soon as the token id is known
//     ...                                           // embedding, layer 0
//     reader.collect(t);                            // before layer 1 reads the rows
//
// `issue` also serves prefill: pass all 16 x N rows of a chunk; pages are deduplicated, sorted by offset and
// kept at most `max_inflight` deep, so a chunk's reads can run while the previous chunk computes.
//
// THE ROW CACHE IS NOT THE TABLE. It keeps rows this process has already fetched (90 bytes each, bounded,
// clock eviction). Measured on the frozen corpus (bench/results/2026-09-23-ngram-io): 1M rows (~95 MB)
// would serve up to ~82% of reads; within one long prompt 20-34% of rows recur. Capacity 0 disables it.
#pragma once

#include "strata/platform/direct_file.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::ngram {

inline constexpr uint32_t ROW_BYTES = 90;          ///< an IQ4_NL row; `open` takes the table's own size
inline constexpr uint32_t PAGE = 4096;

struct ReaderStats {
    uint64_t requests = 0;        ///< rows asked for
    uint64_t cache_hits = 0;      ///< rows served from the row cache
    uint64_t dedup_rows = 0;      ///< rows that shared a page already being read in the same ticket
    uint64_t reads = 0;           ///< SSD read requests issued
    uint64_t bytes = 0;           ///< bytes read from the SSD
    double wait_us = 0;           ///< time `collect` spent blocked
    double submit_us = 0;         ///< time spent inside the read submission call (non-zero = it blocks)
    double read_us_sum = 0;       ///< sum of per-read latencies (issue to completion)
    uint64_t late_injected = 0;   ///< reads delayed by fault injection
    uint64_t keepalive_reads = 0; ///< pages read only to keep the SSD awake (not in `reads`, `bytes` or latencies)
    double keepalive_us_max = 0;  ///< the slowest of them
    std::vector<float> read_us;   ///< last <= 65,536 read latencies, for percentiles
    double percentile(double q) const;
};

class PleReader {
public:
    struct Ticket {
        uint32_t id = 0;
    };

    PleReader();
    ~PleReader();
    PleReader(const PleReader&) = delete;
    PleReader& operator=(const PleReader&) = delete;

    /// `table_offset` is the byte offset of row 0 in the file and `n_rows` the row count; both come from a
    /// validated GGUF parse (PleTable::open checks the table exactly fills the file from there).
    /// `io_thread` (default): a worker thread submits and reaps reads, so `issue` costs the caller no ReadFile
    /// calls. false: the caller's thread does it (A/B arm).
    /// `row_bytes`: one row's size in the file (90 IQ4_NL, 160 FP8), at most one page.
    bool open(const std::string& path, uint64_t table_offset, uint64_t n_rows, uint32_t max_inflight,
              uint64_t cache_rows, std::string& err, bool io_thread = true, uint32_t row_bytes = ROW_BYTES);
    uint32_t row_bytes() const;
    void close();
    bool is_open() const;

    /// Start fetching `n` rows; row i's raw bytes land at `out_raw + row_bytes * i`. `out_raw` must stay valid
    /// until `collect` returns. Out-of-range rows produce zero bytes (the mmap path's behaviour).
    Ticket issue(const uint32_t* rows, size_t n, uint8_t* out_raw);

    /// Block until every row of the ticket is in `out_raw`. Returns false on an I/O error (message in `err`).
    bool collect(Ticket t, std::string& err);

    /// Keep the SSD awake while the table is in use (io_thread mode only; call after `open`): when no read has
    /// gone out for `period_ms`, the worker reads one page of the table, until `window_s` after the last `issue`.
    /// Some SSDs drop into a power state after ~250 ms without a command and stall the next reads 50-150 ms
    /// (a WD_BLACK SN7100 on Windows 11; the NVMe idle timeouts of the power plan did not change it), which in
    /// decode happens after a few rounds whose rows all came from the row cache. 0 turns it off (the reader's
    /// default; generate turns it on, see STRATA_SSD_KEEPALIVE). A keep-alive read that fails turns it off.
    void set_keepalive(double period_ms, double window_s);

    /// Fault injection for tests and for the plan's P2 exit check: every read completes no earlier than
    /// `delay_us` after it was issued. 0 disables.
    void set_injected_delay_us(double delay_us);

    /// Read with no ticket in flight (the worker updates these while reads are outstanding; with the keep-alive
    /// on it may also be counting a keep-alive read - `snapshot` takes a consistent copy).
    const ReaderStats& stats() const;
    ReaderStats snapshot() const;
    void reset_stats();
    uint64_t cache_capacity() const;
    uint64_t cache_size() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace strata::ngram
