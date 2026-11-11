// src/ngram/ple_reader_stress_test.cpp - #3602: the direct table reader under load, every delivered row compared byte
// for byte with a plain buffered read of the same file.  CPU and a temp file only, no model, no GPU.
//
//   ple_reader_stress_test [--dir D] [--seconds S] [--big] [--no-sparse-check]
//
// What it covers that ple_reader_test --selftest does not:
//   * several client threads issuing and collecting at once (the io thread, the row cache and the slot buffers are
//     shared), several tickets outstanding per client, collected in any order;
//   * a row cache so small that nearly every insert evicts, next to one that holds everything;
//   * max_inflight 1 and 2 (a slot is reused at once), randomly injected completion delays, the keep-alive reads;
//   * rows that straddle a page boundary at every phase (header offsets 0, 192, 4051, 4000 and row sizes 90, 110, 160,
//     4096), the first and last rows of the table, a file with and without bytes after the table, out-of-range rows;
//   * "late writers": every collected buffer is kept and compared again after later tickets ran, so a read that
//     lands in a buffer after `collect` returned shows up;
//   * the reader after a failed read: the next issue/collect must fail, not wait for ever (Linux: the file is cut
//     short under the open reader);
//   * --big: a sparse table whose rows sit at file offsets around 2^31 and 2^32 and at the end of a 5 GiB file.
#include "strata/ngram/ple_reader.hpp"
#include "strata/platform/direct_file.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winioctl.h>
#else
#include <unistd.h>
#endif

namespace ng = strata::ngram;

namespace {

std::atomic<int> g_fail{0};
std::mutex g_print;
#define CHECK(c, ...)                                                                          \
    do {                                                                                       \
        if (!(c)) {                                                                            \
            std::lock_guard<std::mutex> lk_(g_print);                                          \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                          \
            std::fprintf(stderr, __VA_ARGS__);                                                 \
            std::fprintf(stderr, "\n");                                                        \
            ++g_fail;                                                                          \
        }                                                                                      \
    } while (0)

constexpr uint32_t PAGE = ng::PAGE;

/// The bytes of row `row` (rb bytes): a xorshift stream seeded by the row, so no two rows alike and a row read from
/// the wrong place never looks like another row's.
void fill_row(uint32_t row, uint32_t rb, uint8_t* out) {
    uint64_t x = 0x9E3779B97F4A7C15ull * (uint64_t) (row + 1) + 0x1234567ull;
    for (uint32_t b = 0; b < rb; ++b) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out[b] = (uint8_t) (x >> 24);
    }
}

struct Spec {
    const char* name;
    uint32_t rb;
    uint64_t header;
    uint32_t n_rows;
    uint64_t trailing;   // bytes after the table
};

bool make_table(const std::string& path, const Spec& s) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    std::vector<uint8_t> head(s.header, 0xAB);
    if (!head.empty()) f.write((const char*) head.data(), (std::streamsize) head.size());
    std::vector<uint8_t> r(s.rb);
    for (uint32_t i = 0; i < s.n_rows; ++i) {
        fill_row(i, s.rb, r.data());
        f.write((const char*) r.data(), s.rb);
    }
    std::vector<uint8_t> tail(s.trailing, 0xEE);
    if (!tail.empty()) f.write((const char*) tail.data(), (std::streamsize) tail.size());
    return (bool) f;
}

/// A buffered read of the file: the reference.
struct Ref {
    std::ifstream f;
    uint64_t header;
    uint32_t rb;
    uint64_t n_rows;
    Ref(const std::string& path, uint64_t h, uint32_t r, uint64_t n) : f(path, std::ios::binary), header(h), rb(r), n_rows(n) {}
    void row(uint32_t r, uint8_t* out) {
        if (r >= n_rows) {
            std::memset(out, 0, rb);
            return;
        }
        f.clear();
        f.seekg((std::streamoff) (header + (uint64_t) r * rb));
        f.read((char*) out, rb);
        if ((uint32_t) f.gcount() != rb) std::memset(out, 0x77, rb);   // the reference itself failed: never equal
    }
};

struct Delivered {
    std::vector<uint32_t> rows;
    std::vector<uint8_t> buf;
};

/// 0 = delivered bytes equal the reference (and, where `expect` is set, the generated row).
bool verify(Ref& ref, const Delivered& d, uint32_t rb, const char* what, bool generated) {
    std::vector<uint8_t> want(rb), gen(rb);
    for (size_t i = 0; i < d.rows.size(); ++i) {
        ref.row(d.rows[i], want.data());
        if (std::memcmp(want.data(), &d.buf[i * rb], rb) != 0) {
            size_t b = 0;
            while (b < rb && want[b] == d.buf[i * rb + b]) ++b;
            CHECK(false, "%s: row %u (index %zu of %zu) differs from the file at byte %zu", what, d.rows[i], i, d.rows.size(), b);
            return false;
        }
        if (generated && d.rows[i] < ref.n_rows) {
            fill_row(d.rows[i], rb, gen.data());
            if (std::memcmp(gen.data(), want.data(), rb) != 0) {
                CHECK(false, "%s: row %u: the file is not what the writer wrote", what, d.rows[i]);
                return false;
            }
        }
    }
    return true;
}

struct Picker {
    const Spec& s;
    std::vector<uint32_t> straddlers, tail_rows, hot;
    explicit Picker(const Spec& sp, std::mt19937& rng) : s(sp) {
        for (uint32_t r = 0; r < s.n_rows; ++r) {
            const uint64_t a = s.header + (uint64_t) r * s.rb;
            if (a / PAGE != (a + s.rb - 1) / PAGE) straddlers.push_back(r);
        }
        for (uint32_t k = 0; k < 200 && k < s.n_rows; ++k) tail_rows.push_back(s.n_rows - 1 - k);
        for (int k = 0; k < 48; ++k) hot.push_back(rng() % s.n_rows);
    }
    uint32_t one(std::mt19937& rng) const {
        switch (rng() % 9) {
            case 0: return straddlers.empty() ? rng() % s.n_rows : straddlers[rng() % straddlers.size()];
            case 1: return tail_rows[rng() % tail_rows.size()];
            case 2: return hot[rng() % hot.size()];
            case 3: return (uint32_t) (rng() % 3) + (uint32_t) (rng() % 4);                 // the first rows
            case 4: return s.n_rows + (uint32_t) (rng() % 4);                                 // just out of range
            case 5: return rng() % 40 == 0 ? 0xFFFFFFFFu : rng() % s.n_rows;
            default: return rng() % s.n_rows;
        }
    }
    std::vector<uint32_t> make(std::mt19937& rng) const {
        static const size_t sizes[] = {1, 2, 5, 16, 16, 17, 64, 130, 700, 2500};
        const size_t n = sizes[rng() % (sizeof sizes / sizeof sizes[0])];
        std::vector<uint32_t> r;
        r.reserve(n);
        switch (rng() % 5) {
            case 0: {   // a run: neighbours share pages
                const uint32_t at = rng() % s.n_rows;
                for (size_t i = 0; i < n; ++i) r.push_back(at + (uint32_t) i);
                break;
            }
            case 1:     // duplicates
                for (size_t i = 0; i < n; ++i) r.push_back(hot[rng() % 4]);
                break;
            default:
                for (size_t i = 0; i < n; ++i) r.push_back(one(rng));
        }
        return r;
    }
};

struct Config {
    bool threaded;
    uint32_t inflight;
    uint64_t cache_rows;
    bool delay;
    bool keepalive;
    int clients;
};

void client(ng::PleReader& rd, const Spec& s, const Picker& pk, const std::string& path, uint32_t seed,
            std::chrono::steady_clock::time_point until, long min_tickets, std::atomic<long>& done) {
    std::mt19937 rng(seed);
    Ref ref(path, s.header, s.rb, s.n_rows);
    std::deque<Delivered> keep;
    long tickets = 0;
    while (std::chrono::steady_clock::now() < until || tickets < min_tickets) {
        const int k = 1 + (int) (rng() % 3);
        std::vector<Delivered> ds((size_t) k);
        std::vector<ng::PleReader::Ticket> ts((size_t) k);
        for (int j = 0; j < k; ++j) {
            ds[(size_t) j].rows = pk.make(rng);
            ds[(size_t) j].buf.assign(ds[(size_t) j].rows.size() * s.rb, 0xCC);
            ts[(size_t) j] = rd.issue(ds[(size_t) j].rows.data(), ds[(size_t) j].rows.size(), ds[(size_t) j].buf.data());
        }
        std::vector<int> order((size_t) k);
        for (int j = 0; j < k; ++j) order[(size_t) j] = j;
        std::shuffle(order.begin(), order.end(), rng);
        for (int j : order) {
            std::string err;
            if (!rd.collect(ts[(size_t) j], err)) {
                CHECK(false, "collect failed: %s", err.c_str());
                return;
            }
            verify(ref, ds[(size_t) j], s.rb, "ticket", true);
            ++tickets;
            keep.push_back(std::move(ds[(size_t) j]));
            if (keep.size() > 6) keep.pop_front();
        }
        if (rng() % 4 == 0)   // late writers: what was collected earlier must still be what it was
            for (const Delivered& d : keep) verify(ref, d, s.rb, "buffer changed after collect", false);
        if (g_fail.load() > 20) return;
    }
    for (const Delivered& d : keep) verify(ref, d, s.rb, "buffer changed after collect (end)", false);
    done += tickets;
}

bool run_config(const std::string& dir, const Spec& s, const Config& c, double seconds, uint32_t seed) {
    const std::string path = dir + "/ple_stress_" + std::to_string(s.rb) + "_" + std::to_string(s.header) + ".bin";
    std::mt19937 rng(seed);
    Picker pk(s, rng);
    ng::PleReader rd;
    std::string err;
    const uint64_t cache_rows = std::min<uint64_t>(c.cache_rows, (64ull << 20) / s.rb);   // at most 64 MiB of cache
    if (!rd.open(path, s.header, s.n_rows, c.inflight, cache_rows, err, c.threaded, s.rb)) {
        std::fprintf(stderr, "cannot open %s: %s\n", path.c_str(), err.c_str());
        return false;
    }
    if (c.keepalive) rd.set_keepalive(1.0, 2.0);
    std::atomic<bool> stop{false};
    std::thread chaos;
    // (the caller-thread reader has no lock: its delay is set once, from this thread, before any client starts)
    if (c.delay && !c.threaded) rd.set_injected_delay_us(300);
    if (c.delay && c.threaded)
        chaos = std::thread([&] {
            std::mt19937 r(seed ^ 77u);
            while (!stop.load()) {
                static const double d[] = {0, 0, 150, 600, 2500};
                rd.set_injected_delay_us(d[r() % 5]);
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            }
            rd.set_injected_delay_us(0);
        });
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds((long) (seconds * 1000));
    std::atomic<long> done{0};
    std::vector<std::thread> th;
    const int clients = c.threaded ? c.clients : 1;
    // a run that is slow (delays) still makes at least this many tickets per client
    for (int i = 0; i < clients; ++i)
        th.emplace_back(client, std::ref(rd), std::cref(s), std::cref(pk), std::cref(path), seed + 1000u * (uint32_t) i, until, 12L,
                        std::ref(done));
    for (auto& t : th) t.join();
    stop = true;
    if (chaos.joinable()) chaos.join();
    CHECK(rd.cache_size() <= rd.cache_capacity(), "row cache %llu over its capacity %llu", (unsigned long long) rd.cache_size(),
          (unsigned long long) rd.cache_capacity());
    const ng::ReaderStats st = rd.snapshot();
    std::printf("  %-14s rb %3u hdr %4llu | %s inflight %2u cache %5llu%s%s clients %d: %ld tickets, %llu rows, %.1f%% cache hits, %llu reads\n",
                s.name, s.rb, (unsigned long long) s.header, c.threaded ? "thread" : "caller", c.inflight,
                (unsigned long long) c.cache_rows, c.delay ? " delay" : "", c.keepalive ? " keepalive" : "", clients,
                done.load(), (unsigned long long) st.requests, st.requests ? 100.0 * (double) st.cache_hits / (double) st.requests : 0.0,
                (unsigned long long) st.reads);
    rd.close();
    return true;
}

#if !defined(_WIN32)
/// After a failed read the reader must keep failing (collect returns false) and must not hang.
void error_does_not_hang(const std::string& dir) {
    const Spec s{"error", 90, 192, 20000, 0};
    const std::string path = dir + "/ple_stress_error.bin";
    if (!make_table(path, s)) { CHECK(false, "cannot write %s", path.c_str()); return; }
    for (bool threaded : {true, false}) {
        if (!make_table(path, s)) { CHECK(false, "cannot rewrite %s", path.c_str()); return; }
        ng::PleReader rd;
        std::string err;
        if (!rd.open(path, s.header, s.n_rows, 8, 0, err, threaded, s.rb)) { CHECK(false, "open: %s", err.c_str()); return; }
        std::vector<uint32_t> rows = {0, 1, 2, 3};
        std::vector<uint8_t> out(rows.size() * s.rb);
        CHECK(rd.collect(rd.issue(rows.data(), rows.size(), out.data()), err), "first ticket: %s", err.c_str());
        // cut the file short under the open reader: every page past the cut now reads as a short read
        if (truncate(path.c_str(), (off_t) (s.header + 100 * s.rb)) != 0) { CHECK(false, "truncate failed"); return; }
        std::vector<uint32_t> beyond = {15000, 15001, 19999};
        std::vector<uint8_t> out2(beyond.size() * s.rb);
        auto go = std::async(std::launch::async, [&]() -> int {
            std::string e1, e2, e3;
            const bool a = rd.collect(rd.issue(beyond.data(), beyond.size(), out2.data()), e1);
            // a ticket after the failure: nothing to read (or something the cache holds) - it must still return
            std::vector<uint32_t> again = {5, 15500};
            std::vector<uint8_t> out3(again.size() * s.rb);
            const bool b = rd.collect(rd.issue(again.data(), again.size(), out3.data()), e2);
            const bool c = rd.collect(rd.issue(beyond.data(), beyond.size(), out2.data()), e3);
            return (a ? 1 : 0) | (b ? 2 : 0) | (c ? 4 : 0);
        });
        if (go.wait_for(std::chrono::seconds(20)) != std::future_status::ready) {
            std::fprintf(stderr, "FAIL: a ticket issued after a failed read never returned (%s reader)\n", threaded ? "thread" : "caller");
            ++g_fail;
            std::_Exit(3);   // the reader's destructor would wait for the same thing
        }
        const int r = go.get();
        CHECK((r & 1) == 0, "a read past the cut file did not fail (%s reader)", threaded ? "thread" : "caller");
        CHECK((r & 2) == 0 && (r & 4) == 0, "a ticket after the failure succeeded (mask %d, %s reader)", r, threaded ? "thread" : "caller");
        rd.close();
    }
    std::filesystem::remove(path);
    std::printf("  error path: a failed read fails the next tickets too, none hangs\n");
}
#endif

// ----------------------------------------------------------------------------------------------- the big sparse table
/// A sparse file of `size` bytes with the given (offset, bytes) pieces written.
bool make_sparse(const std::string& path, uint64_t size, const std::vector<std::pair<uint64_t, std::vector<uint8_t>>>& pieces) {
#if defined(_WIN32)
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    DWORD ret = 0;
    DeviceIoControl(h, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &ret, nullptr);   // if it is refused the file is just big
    bool ok = true;
    for (const auto& p : pieces) {
        LARGE_INTEGER at;
        at.QuadPart = (LONGLONG) p.first;
        DWORD w = 0;
        ok = ok && SetFilePointerEx(h, at, nullptr, FILE_BEGIN) && WriteFile(h, p.second.data(), (DWORD) p.second.size(), &w, nullptr) &&
             w == p.second.size();
    }
    LARGE_INTEGER end;
    end.QuadPart = (LONGLONG) size;
    ok = ok && SetFilePointerEx(h, end, nullptr, FILE_BEGIN) && SetEndOfFile(h);
    CloseHandle(h);
    return ok;
#else
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        for (const auto& p : pieces) {
            f.seekp((std::streamoff) p.first);
            f.write((const char*) p.second.data(), (std::streamsize) p.second.size());
        }
        if (!f) return false;
    }
    std::error_code ec;
    std::filesystem::resize_file(path, size, ec);
    return !ec;
#endif
}

void big_offsets(const std::string& dir) {
    constexpr uint32_t RB = 90;
    constexpr uint64_t HEADER = 192;
    constexpr uint32_t N = 61000000u;                       // 5.49e9 bytes: past 2^32 (4.29e9), below 2^33
    const uint64_t size = HEADER + (uint64_t) N * RB;
    std::set<uint32_t> marks;
    auto around = [&](uint64_t byte_offset) {
        const uint32_t r = (uint32_t) ((byte_offset - HEADER) / RB);
        for (int d = -3; d <= 3; ++d)
            if ((int64_t) r + d >= 0 && (uint64_t) ((int64_t) r + d) < N) marks.insert((uint32_t) ((int64_t) r + d));
    };
    around(1ull << 31);
    around((1ull << 32) - 4096);
    around(1ull << 32);
    around((1ull << 32) + 4096);
    around(3ull << 30);
    around(5ull << 30);
    for (uint32_t k = 0; k < 6; ++k) marks.insert(N - 1 - k);
    for (uint32_t k = 0; k < 4; ++k) marks.insert(k);
    // rows that cross a page boundary in the neighbourhood of 2^32
    for (uint32_t r = (uint32_t) (((1ull << 32) - HEADER) / RB) - 200; r < (uint32_t) (((1ull << 32) - HEADER) / RB) + 200; ++r) {
        const uint64_t a = HEADER + (uint64_t) r * RB;
        if (a / PAGE != (a + RB - 1) / PAGE) marks.insert(r);
    }
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> pieces;
    for (uint32_t r : marks) {
        std::vector<uint8_t> b(RB);
        fill_row(r, RB, b.data());
        pieces.emplace_back(HEADER + (uint64_t) r * RB, std::move(b));
    }
    const std::string path = dir + "/ple_stress_big.bin";
    if (!make_sparse(path, size, pieces)) {
        std::fprintf(stderr, "  big table: cannot create the sparse file here, skipped\n");
        return;
    }
    Ref ref(path, HEADER, RB, N);
    std::mt19937 rng(99);
    for (bool threaded : {true, false}) {
        ng::PleReader rd;
        std::string err;
        if (!rd.open(path, HEADER, N, 16, 1024, err, threaded, RB)) { CHECK(false, "big open: %s", err.c_str()); break; }
        std::vector<uint32_t> mk(marks.begin(), marks.end());
        for (int round = 0; round < 40; ++round) {
            Delivered d;
            for (int i = 0; i < 64; ++i) d.rows.push_back(rng() % 3 == 0 ? (uint32_t) (rng() % N) : mk[rng() % mk.size()]);
            d.buf.assign(d.rows.size() * RB, 0xCC);
            CHECK(rd.collect(rd.issue(d.rows.data(), d.rows.size(), d.buf.data()), err), "big collect: %s", err.c_str());
            verify(ref, d, RB, "big table", false);
            // and against what was written: a marked row is its generated bytes, any other row is zeros
            std::vector<uint8_t> want(RB);
            for (size_t i = 0; i < d.rows.size(); ++i) {
                if (marks.count(d.rows[i])) fill_row(d.rows[i], RB, want.data());
                else std::memset(want.data(), 0, RB);
                CHECK(std::memcmp(want.data(), &d.buf[i * RB], RB) == 0, "big table: row %u is not what was written", d.rows[i]);
            }
        }
        rd.close();
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::printf("  big table: %zu marked rows at offsets around 2^31, 2^32 and the end of a %.2f GiB file, both readers: done\n",
                marks.size(), (double) size / (1024.0 * 1024.0 * 1024.0));
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = std::filesystem::temp_directory_path().string();
    double seconds = 0.3;
    bool big = false, only_error = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dir" && i + 1 < argc) dir = argv[++i];
        else if (a == "--seconds" && i + 1 < argc) seconds = std::atof(argv[++i]);
        else if (a == "--big") big = true;
        else if (a == "--only-error-path") only_error = true;
        else { std::fprintf(stderr, "usage: ple_reader_stress_test [--dir D] [--seconds S] [--big] [--only-error-path]\n"); return 2; }
    }
    // a file system without direct reads (tmpfs on an older kernel): say so and skip
    {
        const std::string probe = dir + "/ple_stress_probe.bin";
        const Spec s{"probe", 90, 192, 1000, 0};
        if (!make_table(probe, s)) { std::fprintf(stderr, "cannot write in %s\n", dir.c_str()); return 2; }
        ng::PleReader rd;
        std::string err;
        const bool ok = rd.open(probe, s.header, s.n_rows, 4, 0, err, true, s.rb);
        rd.close();
        std::filesystem::remove(probe);
        if (!ok) {
            std::printf("SKIP: direct reads are not possible in %s (%s)\n", dir.c_str(), err.c_str());
            return 77;
        }
    }
    const Spec specs[] = {
        {"iq4nl", 90, 192, 150000, 0},          // the real shard's shape: header 192, 90-byte rows, the table ends the file
        {"iq4nl+tail", 90, 192, 150000, 777},   // bytes after the table
        {"odd phase", 90, 4051, 120000, 0},     // the first row straddles
        {"q5_0", 110, 4000, 100000, 0},
        {"fp8", 160, 0, 90000, 123},
        {"page row", 4096, 192, 3000, 0},       // every row crosses a page boundary
        {"tiny", 1, 5, 30000, 0},
    };
    std::vector<Config> cfgs;
    for (uint32_t inflight : {1u, 2u, 3u, 64u})
        for (uint64_t cache : {0ull, 16ull, 100000ull})
            cfgs.push_back({true, inflight, cache, (inflight + cache) % 2 == 0, inflight == 2, 4});
    cfgs.push_back({false, 1, 16, false, false, 1});
    cfgs.push_back({false, 8, 100000, false, false, 1});
    cfgs.push_back({false, 64, 0, true, false, 1});
    std::printf("ple_reader stress: %zu tables x %zu configurations, %.1f s each (dir %s)\n", sizeof specs / sizeof specs[0],
                cfgs.size(), seconds, dir.c_str());
    uint32_t seed = 1;
    for (const Spec& s : specs) {
        if (only_error) break;
        const std::string path = dir + "/ple_stress_" + std::to_string(s.rb) + "_" + std::to_string(s.header) + ".bin";
        if (!make_table(path, s)) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return 2; }
        for (const Config& c : cfgs) {
            if (!run_config(dir, s, c, seconds, ++seed)) return 2;
            if (g_fail.load() > 0) break;
        }
        std::filesystem::remove(path);
        if (g_fail.load() > 0) break;
    }
#if !defined(_WIN32)
    if (g_fail.load() == 0) error_does_not_hang(dir);
#endif
    if (big && g_fail.load() == 0) big_offsets(dir);
    std::printf("ple_reader stress: %s\n", g_fail.load() ? "FAILED" : "OK");
    return g_fail.load() ? 1 : 0;
}
