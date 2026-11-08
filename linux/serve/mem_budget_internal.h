#pragma once
#include "posix_io.h"

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace edge0 {

struct BudgetOps {
    std::function<bool(size_t &)> rss = read_rss;
    std::function<int(void *, size_t)> discard;
    std::function<std::chrono::steady_clock::time_point()> now = std::chrono::steady_clock::now;
};

class MemoryBudget {
public:
    explicit MemoryBudget(BudgetOps ops = {});
    ~MemoryBudget();
    bool start(size_t bytes, bool background = true);
    void stop();
    bool enabled();
    bool check();
    void mapped(void * address, size_t length);
    int unmap(void * address, size_t length);
private:
    bool check_locked();
    bool reclaim_locked(size_t & rss, size_t target);
    void failure(const char * reason, size_t rss);
    BudgetOps ops_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread worker_;
    size_t limit_ = 0, cursor_ = 0;
    size_t stalled_rss_ = 0;  // RSS after the last pass that missed its target; 0 when the last pass reached it
    std::chrono::steady_clock::time_point retry_at_{};  // earliest repeat of a pass that missed its target
    bool quit_ = false, reported_ = false;
    struct Region { uintptr_t begin, end; };
    std::vector<Region> regions_;
};

} // namespace edge0
