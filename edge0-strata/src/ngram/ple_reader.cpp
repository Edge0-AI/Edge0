// src/ngram/ple_reader.cpp - see include/strata/ngram/ple_reader.hpp.
#include "strata/ngram/ple_reader.hpp"
#include "strata/platform/aux_cpus.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace strata::ngram {

using platform::Completion;
using platform::DirectFile;
using platform::now_us;

double ReaderStats::percentile(double q) const {
    if (read_us.empty()) return 0.0;
    std::vector<float> v(read_us);
    const size_t k = std::min(v.size() - 1, (size_t) (q * (double) (v.size() - 1) + 0.5));
    std::nth_element(v.begin(), v.begin() + (ptrdiff_t) k, v.end());
    return v[k];
}

namespace {

constexpr size_t LATENCY_RING = 65536;
constexpr uint32_t WAYS = 8;
constexpr uint32_t EMPTY = 0xFFFFFFFFu;

/// Set-associative row cache: 8 ways per set, round-robin replacement inside a set. Bounded by construction.
struct RowCache {
    uint64_t sets = 0;
    uint32_t rb = ROW_BYTES;        // bytes per row
    std::vector<uint32_t> keys;     // sets * WAYS
    std::vector<uint8_t> data;      // sets * WAYS * rb
    std::vector<uint8_t> next;      // per-set replacement pointer
    uint64_t used = 0;

    void init(uint64_t rows, uint32_t row_bytes = ROW_BYTES) {
        sets = rows / WAYS;
        rb = row_bytes;
        keys.assign(sets * WAYS, EMPTY);
        data.assign(sets * WAYS * rb, 0);
        next.assign(sets, 0);
        used = 0;
    }
    static uint64_t mix(uint32_t r) {
        uint64_t x = r * 0x9E3779B97F4A7C15ull;
        return x ^ (x >> 29);
    }
    const uint8_t* find(uint32_t row) const {
        if (sets == 0) return nullptr;
        const uint64_t s = mix(row) % sets;
        for (uint32_t w = 0; w < WAYS; ++w)
            if (keys[s * WAYS + w] == row) return &data[(s * WAYS + w) * rb];
        return nullptr;
    }
    void insert(uint32_t row, const uint8_t* bytes) {
        if (sets == 0 || find(row) != nullptr) return;
        const uint64_t s = mix(row) % sets;
        uint32_t w = next[s];
        next[s] = (uint8_t) ((w + 1) % WAYS);
        if (keys[s * WAYS + w] == EMPTY) ++used;
        keys[s * WAYS + w] = row;
        std::memcpy(&data[(s * WAYS + w) * rb], bytes, rb);
    }
};

struct Use {
    uint32_t row;
    uint32_t in_page;      // byte offset of the row inside the read buffer
    uint8_t* dst;
};

struct Job {
    uint64_t offset = 0;   // aligned file offset
    uint32_t length = 0;   // PAGE or 2 * PAGE (a row that straddles a page boundary)
    uint32_t ticket = 0;
    std::vector<Use> uses;
    double issued_us = 0;
    bool keepalive = false;// no rows and no ticket: it only keeps the SSD awake (`set_keepalive`)
};

struct TicketState {
    uint32_t pending = 0;  // jobs not yet completed
};

}  // namespace

// THREADING. With `io_thread` (the default) one worker thread owns every DirectFile call: it submits queued jobs
// and reaps completions, so `issue` on the caller's thread only builds jobs and wakes it (~11 us per ReadFile
// no longer lands on the host loop). `mu` guards everything below except `file`, which only the worker touches
// (plus `wake`, which is thread-safe). Without `io_thread` the caller does all of it, as before.
struct PleReader::Impl {
    DirectFile file;
    uint64_t table_offset = 0;
    uint64_t n_rows = 0;
    uint32_t row_bytes = ROW_BYTES;
    uint32_t max_inflight = 0;
    uint8_t* slab = nullptr;              // max_inflight slots of 2 pages
    std::vector<uint32_t> free_slots;
    std::vector<Job> inflight;            // indexed by slot
    std::vector<Completion> delayed;      // completed but held back by fault injection
    std::deque<Job> queue;                // not yet submitted
    std::unordered_map<uint32_t, TicketState> tickets;
    uint32_t next_ticket = 1;
    RowCache cache;
    ReaderStats stats;
    size_t ring_pos = 0;
    double delay_us = 0;
    std::string error;
    // the keep-alive (`set_keepalive`), steady-clock microseconds; `keep_us` 0 = off
    double keep_us = 0;
    double keep_window_us = 0;
    double last_issue_us = 0;             // the last `issue`
    double last_read_us = 0;              // the last read that went out, rows or keep-alive
    uint64_t rng = 0x9E3779B97F4A7C15ull;

    bool threaded = false;
    bool stop = false;
    std::mutex mu;
    std::condition_variable cv_work;      // worker: there is something to submit
    std::condition_variable cv_done;      // collectors: a ticket may have completed
    std::thread worker;

    uint8_t* slot_buf(uint32_t s) { return slab + (size_t) s * 2 * PAGE; }
    bool busy() const { return free_slots.size() < max_inflight || !delayed.empty(); }

    void cancel_queued() {
        for (const Job& job : queue) {
            auto it = tickets.find(job.ticket);
            if (it != tickets.end() && it->second.pending > 0) --it->second.pending;
        }
        queue.clear();
    }

    void record_latency(double us) {
        stats.read_us_sum += us;
        if (stats.read_us.size() < LATENCY_RING) stats.read_us.push_back((float) us);
        else stats.read_us[ring_pos++ % LATENCY_RING] = (float) us;
    }

    bool pump() {
        while (!queue.empty() && !free_slots.empty()) {
            const uint32_t s = free_slots.back();
            free_slots.pop_back();
            inflight[s] = std::move(queue.front());
            queue.pop_front();
            Job& j = inflight[s];
            j.issued_us = now_us();
            std::string kerr;             // a keep-alive read that cannot go out must not fail the reader
            if (!file.submit(j.offset, slot_buf(s), j.length, s, j.keepalive ? kerr : error)) {
                if (j.keepalive) {                 // no more of them: the SSD may sleep as before
                    keep_us = 0;
                    j.keepalive = false;
                    free_slots.push_back(s);
                    continue;
                }
                j.uses.clear();
                auto it = tickets.find(j.ticket);
                if (it != tickets.end() && it->second.pending > 0) --it->second.pending;
                free_slots.push_back(s);
                cancel_queued();
                return false;
            }
            last_read_us = j.issued_us;
            if (!j.keepalive) {                    // a keep-alive read counts when it completes (`finish`)
                stats.submit_us += now_us() - j.issued_us;
                ++stats.reads;
            }
        }
        return true;
    }

    /// When the next keep-alive read is due, or < 0 when none is: it is off, the reader failed, or no rows
    /// were asked for within the window (then the SSD may sleep; the next `issue` re-arms it).
    double keepalive_due(double now) const {
        if (keep_us <= 0 || !error.empty() || last_issue_us <= 0 || now - last_issue_us > keep_window_us) return -1;
        return last_read_us + keep_us;
    }

    /// One page of the table, a different one each time, so the SSD really reads (not its controller's buffer).
    void queue_keepalive() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        const uint64_t first = table_offset / PAGE, end = (table_offset + n_rows * (uint64_t) row_bytes) / PAGE;
        Job j;
        j.offset = (first + (end > first ? rng % (end - first) : 0)) * PAGE;
        j.length = PAGE;
        j.keepalive = true;
        queue.push_back(std::move(j));
    }

    bool finish(const Completion& c) {
        const uint32_t s = (uint32_t) c.tag;
        if (s >= inflight.size()) {
            error = "PleReader: invalid table-read completion";
            cancel_queued();
            return false;
        }
        Job& j = inflight[s];
        if (j.keepalive) {                         // no rows: it counts once it is done, with how long it took;
            if (c.ok) {                            // a failed one turns the keep-alive off, not the reader
                ++stats.keepalive_reads;
                stats.keepalive_us_max = std::max(stats.keepalive_us_max, now_us() - j.issued_us);
            } else {
                keep_us = 0;
            }
            j.keepalive = false;
            free_slots.push_back(s);
            return error.empty() ? pump() : true;
        }
        if (!c.ok) {
            error = "PleReader: a table read failed";
            cancel_queued();
            j.uses.clear();
            auto it = tickets.find(j.ticket);
            if (it != tickets.end() && it->second.pending > 0) --it->second.pending;
            free_slots.push_back(s);
            return false;
        }
        record_latency(now_us() - j.issued_us);
        stats.bytes += c.bytes;
        const uint8_t* buf = slot_buf(s);
        for (const Use& u : j.uses) {
            if (u.in_page + row_bytes > c.bytes) {
                error = "PleReader: short read inside the table";
                cancel_queued();
                j.uses.clear();
                auto it = tickets.find(j.ticket);
                if (it != tickets.end() && it->second.pending > 0) --it->second.pending;
                free_slots.push_back(s);
                return false;
            }
        }
        for (const Use& u : j.uses) {
            std::memcpy(u.dst, buf + u.in_page, row_bytes);
            cache.insert(u.row, buf + u.in_page);
        }
        auto it = tickets.find(j.ticket);
        if (it != tickets.end() && it->second.pending > 0) --it->second.pending;
        j.uses.clear();
        free_slots.push_back(s);
        return error.empty() ? pump() : true;
    }

    /// Completions (or wake packets) just returned by `file.wait`, applying fault injection.
    bool process(const Completion* got, int n) {
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (got[i].tag == DirectFile::WAKE_TAG) continue;
            if (got[i].tag >= inflight.size()) {
                error = "PleReader: invalid table-read completion";
                cancel_queued();
                ok = false;
                continue;
            }
            if (delay_us > 0 && now_us() - inflight[(uint32_t) got[i].tag].issued_us < delay_us) {
                delayed.push_back(got[i]);
                ++stats.late_injected;
                continue;
            }
            if (!finish(got[i])) ok = false;   // drain the rest of this batch so no completed slot is stranded
        }
        return ok;
    }

    bool release_delayed() {
        const double now = now_us();
        for (size_t i = 0; i < delayed.size();) {
            if (now - inflight[(uint32_t) delayed[i].tag].issued_us >= delay_us) {
                const Completion c = delayed[i];
                delayed.erase(delayed.begin() + (ptrdiff_t) i);
                if (!finish(c)) return false;
            } else {
                ++i;
            }
        }
        return true;
    }

    /// Caller-thread mode: process whatever has completed; blocks up to `timeout_ms` for the first completion.
    bool drain(int timeout_ms) {
        Completion got[64];
        if (!delayed.empty()) {
            if (!release_delayed()) return false;
            timeout_ms = 0;                        // keep polling the held completions
        }
        const int n = file.wait(got, 64, timeout_ms);
        return process(got, n);
    }

    void worker_loop() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            // Idle: wait for work. While rows are being asked for, the wait ends in time for a keep-alive read.
            while (!(stop || !queue.empty() || busy())) {
                const double now = now_us();
                const double due = keepalive_due(now);
                if (due < 0) cv_work.wait(lk);
                else if (now >= due) queue_keepalive();
                else cv_work.wait_for(lk, std::chrono::microseconds(std::max<int64_t>(1000, (int64_t) (due - now))));
            }
            if (stop && !busy()) break;
            if (error.empty() && !pump() && error.empty()) error = "PleReader: submit failed";
            if (!delayed.empty() && !release_delayed() && error.empty()) error = "PleReader: read failed";
            if (!error.empty()) {
                cv_done.notify_all();
                if (!busy()) { cv_work.wait(lk, [&] { return stop; }); break; }
            }
            if (!busy()) { cv_done.notify_all(); continue; }
            // Block in the port WITHOUT the lock, so `issue` can queue work; `issue` wakes us with a packet.
            const int timeout = delayed.empty() ? -1 : 0;
            lk.unlock();
            Completion got[64];
            const int n = file.wait(got, 64, timeout);
            lk.lock();
            if (!process(got, n) && error.empty()) error = "PleReader: read failed";
            cv_done.notify_all();
        }
    }
};

PleReader::PleReader() : impl_(new Impl) {}
PleReader::~PleReader() {
    close();
    delete impl_;
}

bool PleReader::open(const std::string& path, uint64_t table_offset, uint64_t n_rows, uint32_t max_inflight,
                     uint64_t cache_rows, std::string& err, bool io_thread, uint32_t row_bytes) {
    close();
    if (max_inflight == 0 || max_inflight > 1024) { err = "PleReader: max_inflight must be 1..1024"; return false; }
    if (row_bytes == 0 || row_bytes > PAGE) { err = "PleReader: row_bytes must be 1..4096"; return false; }
    if (!impl_->file.open(path, err)) return false;
    impl_->row_bytes = row_bytes;
    if (table_offset + n_rows * (uint64_t) row_bytes > impl_->file.size()) {
        err = "PleReader: the table extends past the end of " + path;
        close();
        return false;
    }
    impl_->table_offset = table_offset;
    impl_->n_rows = n_rows;
    impl_->max_inflight = max_inflight;
    impl_->slab = (uint8_t*) DirectFile::alloc_aligned((size_t) max_inflight * 2 * PAGE);
    if (impl_->slab == nullptr) { err = "PleReader: cannot allocate read buffers"; close(); return false; }
    impl_->inflight.assign(max_inflight, Job{});
    impl_->free_slots.clear();
    for (uint32_t s = max_inflight; s-- > 0;) impl_->free_slots.push_back(s);
    impl_->cache.init(cache_rows, row_bytes);
    impl_->error.clear();
    impl_->keep_us = 0;
    impl_->last_issue_us = impl_->last_read_us = 0;
    impl_->rng = 0x9E3779B97F4A7C15ull ^ (uint64_t) now_us();
    if (impl_->rng == 0) impl_->rng = 1;
    reset_stats();
    impl_->stop = false;
    impl_->threaded = io_thread;
    if (io_thread) {
        try {
            impl_->worker = std::thread([this] { strata::aux_cpus::pin_current_thread(); impl_->worker_loop(); });
        } catch (const std::exception& e) {
            err = std::string("PleReader: cannot create I/O worker: ") + e.what();
            close();
            return false;
        } catch (...) {
            err = "PleReader: cannot create I/O worker";
            close();
            return false;
        }
    }
    return true;
}

void PleReader::close() {
    Impl& m = *impl_;
    if (m.worker.joinable()) {
        {
            std::lock_guard<std::mutex> lk(m.mu);
            m.stop = true;
            m.queue.clear();                       // unsubmitted work is dropped; in-flight reads still drain
        }
        m.cv_work.notify_all();
        m.file.wake();
        m.worker.join();
    }
    // Outstanding reads must finish before their buffers are released (caller-thread mode, or a worker that
    // stopped on an error).
    if (m.file.is_open()) {
        while (m.free_slots.size() < m.max_inflight) {
            Completion c[64];
            const int n = m.file.wait(c, 64, -1);
            if (n == 0) break;
            for (int i = 0; i < n; ++i)
                if (c[i].tag != DirectFile::WAKE_TAG) m.free_slots.push_back((uint32_t) c[i].tag);
        }
    }
    m.file.close();
    DirectFile::free_aligned(m.slab);
    m.slab = nullptr;
    m.queue.clear();
    m.tickets.clear();
    m.delayed.clear();
    m.inflight.clear();
    m.free_slots.clear();
    m.cache.init(0);
    m.threaded = false;
    m.stop = false;
    m.keep_us = 0;
    m.last_issue_us = m.last_read_us = 0;
}

bool PleReader::is_open() const { return impl_->file.is_open(); }
uint32_t PleReader::row_bytes() const { return impl_->row_bytes; }

PleReader::Ticket PleReader::issue(const uint32_t* rows, size_t n, uint8_t* out_raw) {
    Impl& m = *impl_;
    std::unique_lock<std::mutex> lk(m.mu, std::defer_lock);
    if (m.threaded) lk.lock();
    const uint32_t id = m.next_ticket++;
    if (m.next_ticket == 0) m.next_ticket = 1;
    const double now = now_us();
    const bool rearm = m.keep_us > 0 && (m.last_issue_us <= 0 || now - m.last_issue_us > m.keep_window_us);
    m.last_issue_us = now;
    TicketState& ts = m.tickets[id];
    // After a failed read the reader stays failed (`collect` reports `error` until the next `open`) and its worker has
    // parked (worker_loop). A ticket queued now would never be read and `collect` would wait for it for ever, so it
    // is not queued: it has no jobs, and `collect` returns the error at once.
    if (!m.error.empty()) {
        ts.pending = 0;
        return Ticket{id};
    }
    std::unordered_map<uint64_t, size_t> by_page;     // aligned offset -> index in `jobs`
    std::vector<Job> jobs;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t rb = m.row_bytes;
        uint8_t* dst = out_raw + i * rb;
        ++m.stats.requests;
        if (rows[i] >= m.n_rows) {
            std::memset(dst, 0, rb);
            continue;
        }
        if (const uint8_t* hit = m.cache.find(rows[i])) {
            std::memcpy(dst, hit, rb);
            ++m.stats.cache_hits;
            continue;
        }
        const uint64_t at = m.table_offset + (uint64_t) rows[i] * rb;
        const uint64_t first = at / PAGE * PAGE;
        const uint32_t length = (uint32_t) ((at + rb - 1) / PAGE * PAGE - first + PAGE);
        auto f = by_page.find(first);
        if (f != by_page.end()) {
            Job& j = jobs[f->second];
            j.length = std::max(j.length, length);
            j.uses.push_back(Use{rows[i], (uint32_t) (at - first), dst});
            ++m.stats.dedup_rows;
            continue;
        }
        by_page.emplace(first, jobs.size());
        Job j;
        j.offset = first;
        j.length = length;
        j.ticket = id;
        j.uses.push_back(Use{rows[i], (uint32_t) (at - first), dst});
        jobs.push_back(std::move(j));
    }
    // Sorted by offset: prefill chunks then read the SSD in near-sequential order.
    std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) { return a.offset < b.offset; });
    ts.pending = (uint32_t) jobs.size();
    const bool has_jobs = ts.pending > 0;
    for (Job& j : jobs) m.queue.push_back(std::move(j));
    if (m.threaded) {
        lk.unlock();
        if (has_jobs) {
            m.cv_work.notify_one();
            m.file.wake();                         // in case the worker is blocked in the port
        } else if (rearm) {
            m.cv_work.notify_one();                // the keep-alive had lapsed: start it again
        }
    } else if (!m.pump() && m.error.empty()) {
        m.error = "PleReader: submit failed";
    }
    return Ticket{id};
}

bool PleReader::collect(Ticket t, std::string& err) {
    Impl& m = *impl_;
    const double start = now_us();
    if (m.threaded) {
        std::unique_lock<std::mutex> lk(m.mu);
        auto it = m.tickets.find(t.id);
        if (it == m.tickets.end()) { err = "PleReader: unknown ticket"; return false; }
        m.cv_done.wait(lk, [&] {
            const auto current = m.tickets.find(t.id);
            return current == m.tickets.end() || current->second.pending == 0;
        });
        it = m.tickets.find(t.id);
        if (it == m.tickets.end()) { err = "PleReader: unknown ticket"; return false; }
        if (!m.error.empty()) { err = m.error; m.tickets.erase(it); return false; }
        m.stats.wait_us += now_us() - start;
        m.tickets.erase(it);
        return true;
    }
    auto it = m.tickets.find(t.id);
    if (it == m.tickets.end()) { err = "PleReader: unknown ticket"; return false; }
    while (it->second.pending > 0) {
        const bool drained = m.drain(-1);
        if (!drained && m.error.empty()) m.error = "PleReader: read failed";
        it = m.tickets.find(t.id);
        if (it == m.tickets.end()) { err = "PleReader: unknown ticket"; return false; }
    }
    if (!m.error.empty()) { err = m.error; m.tickets.erase(it); return false; }
    m.stats.wait_us += now_us() - start;
    m.tickets.erase(it);
    return true;
}

void PleReader::set_keepalive(double period_ms, double window_s) {
    Impl& m = *impl_;
    {
        std::lock_guard<std::mutex> lk(m.mu);
        m.keep_us = m.threaded && period_ms > 0 ? period_ms * 1000.0 : 0.0;
        m.keep_window_us = window_s > 0 ? window_s * 1e6 : 0.0;
    }
    m.cv_work.notify_all();
}

void PleReader::set_injected_delay_us(double delay_us) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->delay_us = delay_us < 0 ? 0 : delay_us;
}
const ReaderStats& PleReader::stats() const { return impl_->stats; }
ReaderStats PleReader::snapshot() const {
    Impl& m = *impl_;
    std::unique_lock<std::mutex> lk(m.mu, std::defer_lock);
    if (m.threaded) lk.lock();
    return m.stats;
}
void PleReader::reset_stats() {
    Impl& m = *impl_;
    std::unique_lock<std::mutex> lk(m.mu, std::defer_lock);
    if (m.threaded) lk.lock();                     // the worker may be counting a keep-alive read
    m.stats = ReaderStats{};
    m.ring_pos = 0;
}
uint64_t PleReader::cache_capacity() const { return impl_->cache.sets * WAYS; }
uint64_t PleReader::cache_size() const { return impl_->cache.used; }

}  // namespace strata::ngram
