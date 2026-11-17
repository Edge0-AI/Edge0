// src/platform/direct_file.cpp - see include/strata/platform/direct_file.hpp.
#include "strata/platform/direct_file.hpp"
#include "strata/platform/aux_cpus.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::platform {

double now_us() {
    using namespace std::chrono;
    return (double) duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

void* DirectFile::alloc_aligned(size_t bytes) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* p = nullptr;
    return posix_memalign(&p, alignment(), bytes) == 0 ? p : nullptr;
#endif
}

void DirectFile::free_aligned(void* p) {
    if (p == nullptr) return;
#if defined(_WIN32)
    VirtualFree(p, 0, MEM_RELEASE);
#else
    std::free(p);
#endif
}

namespace {
/// A queued read: `submit` only queues it; the pool's threads issue it (perf-review C-4 / F-2).  One thread
/// issuing every read was the limit of the n-gram table's prompt reads: ~12 us of kernel time per read, ~80K
/// reads/s, while an NVMe drive serves several times that at depth.
struct Pending {
    uint64_t offset;
    void* buffer;
    uint32_t length;
    uint64_t tag;
};

/// The number of issuing threads: STRATA_IO_THREADS, else 4 (Windows: overlapped submits) / 16 (Linux: each
/// thread does one blocking pread, so the thread count is the queue depth).
int io_threads(int dflt) {
    const char* v = std::getenv("STRATA_IO_THREADS");
    const int n = v ? std::atoi(v) : dflt;
    return std::clamp(n, 1, 64);
}
}  // namespace

#if defined(_WIN32)
// ------------------------------------------------------------------------------------------------ Windows
namespace {
/// One in-flight request. The OVERLAPPED must be the first member so a completion packet's OVERLAPPED*
/// converts back to the request.
struct Req {
    OVERLAPPED ov;
    uint64_t tag;
};
}  // namespace

struct DirectFile::Impl {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE port = nullptr;
    uint64_t size = 0;
    std::mutex mu;                      // everything below
    std::deque<Req*> free_reqs;
    std::vector<Req*> all_reqs;
    std::deque<Completion> immediate;   // requests that completed without a port packet (EOF) or failed
    std::deque<Pending> queue;          // submitted, not yet issued
    std::condition_variable cv;
    bool stop = false;
    std::vector<std::thread> pool;

    Req* take() {
        if (free_reqs.empty()) {
            Req* r = new Req();
            all_reqs.push_back(r);
            return r;
        }
        Req* r = free_reqs.front();
        free_reqs.pop_front();
        return r;
    }

    void issue(const Pending& p) {
        Req* r;
        {
            std::lock_guard<std::mutex> lk(mu);
            r = take();
        }
        std::memset(&r->ov, 0, sizeof r->ov);
        r->ov.Offset = (DWORD) (p.offset & 0xFFFFFFFFull);
        r->ov.OffsetHigh = (DWORD) (p.offset >> 32);
        r->tag = p.tag;
        if (!ReadFile(file, p.buffer, p.length, nullptr, &r->ov)) {
            const DWORD e = GetLastError();
            if (e != ERROR_IO_PENDING) {
                std::lock_guard<std::mutex> lk(mu);
                free_reqs.push_back(r);
                // at or past end of file: a zero-byte completion; anything else: a failed one
                immediate.push_back(Completion{p.tag, 0, e == ERROR_HANDLE_EOF});
                PostQueuedCompletionStatus(port, 0, 0, nullptr);   // wake a waiter to collect it
            }
        }
    }

    void worker() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv.wait(lk, [&] { return stop || !queue.empty(); });
            if (stop && queue.empty()) return;
            const Pending p = queue.front();
            queue.pop_front();
            lk.unlock();
            issue(p);
            lk.lock();
        }
    }

    void stop_pool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        for (std::thread& t : pool) t.join();
        pool.clear();
        stop = false;
    }
};

DirectFile::DirectFile() : impl_(new Impl) {}
DirectFile::~DirectFile() {
    close();
    for (Req* r : impl_->all_reqs) delete r;
    delete impl_;
}

bool DirectFile::open(const std::string& path, std::string& err) {
    close();
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath((size_t) (wlen > 0 ? wlen : 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
    // NO_BUFFERING: the cache manager never holds these pages, so the table cannot grow into RAM.
    // RANDOM_ACCESS: no read-ahead. OVERLAPPED: many reads in flight, completed through the port below.
    impl_->file = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (impl_->file == INVALID_HANDLE_VALUE) {
        err = "DirectFile: cannot open " + path + " (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(impl_->file, &sz)) {
        err = "DirectFile: cannot size " + path;
        close();
        return false;
    }
    impl_->size = (uint64_t) sz.QuadPart;
    impl_->port = CreateIoCompletionPort(impl_->file, nullptr, 0, 1);
    if (impl_->port == nullptr) {
        err = "DirectFile: CreateIoCompletionPort failed (error " + std::to_string(GetLastError()) + ")";
        close();
        return false;
    }
    // Completions of reads that finish synchronously are still queued to the port, so every issued read
    // produces exactly one packet; `wait` is the only completion path.
    const int n = io_threads(4);
    for (int i = 0; i < n; ++i) impl_->pool.emplace_back([this] { strata::aux_cpus::pin_current_thread(); impl_->worker(); });
    return true;
}

void DirectFile::close() {
    impl_->stop_pool();
    if (impl_->port != nullptr) CloseHandle(impl_->port);
    if (impl_->file != INVALID_HANDLE_VALUE) CloseHandle(impl_->file);
    impl_->port = nullptr;
    impl_->file = INVALID_HANDLE_VALUE;
    impl_->size = 0;
    impl_->immediate.clear();
    impl_->queue.clear();
}

bool DirectFile::is_open() const { return impl_->file != INVALID_HANDLE_VALUE; }
uint64_t DirectFile::size() const { return impl_->size; }

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    if (!is_open()) { err = "DirectFile: not open"; return false; }
    if (offset % alignment() || length % alignment() || ((uintptr_t) buffer) % alignment() || length == 0) {
        err = "DirectFile: unaligned request";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->queue.push_back(Pending{offset, buffer, length, tag});
    }
    impl_->cv.notify_one();
    return true;
}

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
    int n = 0;
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        while (n < max && !impl_->immediate.empty()) {
            out[n++] = impl_->immediate.front();
            impl_->immediate.pop_front();
        }
    }
    if (n == max || !is_open()) return n;
    OVERLAPPED_ENTRY entries[64];
    const ULONG want = (ULONG) (max - n < 64 ? max - n : 64);
    ULONG got = 0;
    const DWORD t = timeout_ms < 0 ? INFINITE : (DWORD) timeout_ms;
    if (!GetQueuedCompletionStatusEx(impl_->port, entries, want, &got, n > 0 ? 0 : t, FALSE)) return n;
    std::lock_guard<std::mutex> lk(impl_->mu);
    for (ULONG i = 0; i < got; ++i) {
        if (entries[i].lpOverlapped == nullptr) {          // a wake() packet (or an immediate completion's nudge)
            while (n < max && !impl_->immediate.empty()) {
                out[n++] = impl_->immediate.front();
                impl_->immediate.pop_front();
            }
            if (n < max) out[n++] = Completion{WAKE_TAG, 0, true};
            continue;
        }
        Req* r = (Req*) entries[i].lpOverlapped;
        const uint32_t status = (uint32_t) r->ov.Internal;   // an NTSTATUS
        Completion c;
        c.tag = r->tag;
        c.bytes = entries[i].dwNumberOfBytesTransferred;
        // STATUS_END_OF_FILE (0xC0000011) is a legal short read at the table's last page.
        c.ok = status == 0 || status == 0xC0000011u;
        out[n++] = c;
        impl_->free_reqs.push_back(r);
    }
    return n;
}

void DirectFile::wake() {
    if (impl_->port != nullptr) PostQueuedCompletionStatus(impl_->port, 0, 0, nullptr);
}

#else
// ------------------------------------------------------------------------------------------------ POSIX
// perf-review F-2: a pool of threads, each doing blocking O_DIRECT preads, so reads run in parallel (the thread
// count is the queue depth).  It replaced one synchronous pread inside `submit`, which read the n-gram table's
// rows one at a time.
struct DirectFile::Impl {
    int fd = -1;
    uint64_t size = 0;
    std::mutex mu;
    std::condition_variable cv_work, cv_done;
    std::deque<Pending> queue;
    std::deque<Completion> done;
    bool stop = false;
    std::vector<std::thread> pool;

    void worker() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv_work.wait(lk, [&] { return stop || !queue.empty(); });
            if (stop && queue.empty()) return;
            const Pending p = queue.front();
            queue.pop_front();
            lk.unlock();
            const ssize_t got = pread(fd, p.buffer, p.length, (off_t) p.offset);
            lk.lock();
            done.push_back(Completion{p.tag, got < 0 ? 0u : (uint32_t) got, got >= 0});
            cv_done.notify_all();
        }
    }

    void stop_pool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv_work.notify_all();
        for (std::thread& t : pool) t.join();
        pool.clear();
        stop = false;
    }
};

DirectFile::DirectFile() : impl_(new Impl) {}
DirectFile::~DirectFile() { close(); delete impl_; }

bool DirectFile::open(const std::string& path, std::string& err) {
    close();
#if defined(__APPLE__)   // macOS has no O_DIRECT: F_NOCACHE keeps the reads out of the unified buffer cache
    impl_->fd = ::open(path.c_str(), O_RDONLY);
    if (impl_->fd >= 0 && fcntl(impl_->fd, F_NOCACHE, 1) != 0) { ::close(impl_->fd); impl_->fd = -1; }
#else
    impl_->fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
#endif
    if (impl_->fd < 0) { err = "DirectFile: cannot open " + path; return false; }
    struct stat st;
    if (fstat(impl_->fd, &st) != 0) { err = "DirectFile: cannot size " + path; close(); return false; }
    impl_->size = (uint64_t) st.st_size;
    const int n = io_threads(16);
    for (int i = 0; i < n; ++i) impl_->pool.emplace_back([this] { strata::aux_cpus::pin_current_thread(); impl_->worker(); });
    return true;
}

void DirectFile::close() {
    impl_->stop_pool();
    if (impl_->fd >= 0) ::close(impl_->fd);
    impl_->fd = -1;
    impl_->size = 0;
    impl_->done.clear();
    impl_->queue.clear();
}

bool DirectFile::is_open() const { return impl_->fd >= 0; }
uint64_t DirectFile::size() const { return impl_->size; }

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    if (offset % alignment() || length % alignment() || ((uintptr_t) buffer) % alignment() || length == 0) {
        err = "DirectFile: unaligned request";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->queue.push_back(Pending{offset, buffer, length, tag});
    }
    impl_->cv_work.notify_one();
    return true;
}

void DirectFile::wake() {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->done.push_back(Completion{WAKE_TAG, 0, true});
    impl_->cv_done.notify_all();
}

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
    std::unique_lock<std::mutex> lk(impl_->mu);
    if (impl_->done.empty() && timeout_ms != 0) {
        auto ready = [&] { return !impl_->done.empty(); };
        if (timeout_ms < 0) impl_->cv_done.wait(lk, ready);
        else impl_->cv_done.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready);
    }
    int n = 0;
    while (n < max && !impl_->done.empty()) {
        out[n++] = impl_->done.front();
        impl_->done.pop_front();
    }
    return n;
}
#endif

}  // namespace strata::platform
