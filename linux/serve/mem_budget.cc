#include "mem_budget.h"
#include "mem_budget_internal.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>

namespace edge0 {

MemoryBudget::MemoryBudget(BudgetOps ops) : ops_(std::move(ops)) {
    if (!ops_.discard) ops_.discard = [](void * p, size_t n) { return madvise(p, n, MADV_DONTNEED); };
}

MemoryBudget::~MemoryBudget() { stop(); }

bool MemoryBudget::start(size_t bytes, bool background) {
    stop();
    if (!bytes) return false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        limit_ = bytes;
        quit_ = false;
        reported_ = false;
        size_t rss = 0;
        if (!ops_.rss(rss)) { failure("cannot read process RSS", 0); limit_ = 0; return false; }
    }
    if (background) {
        try {
            worker_ = std::thread([this] {
                std::unique_lock<std::mutex> lock(mutex_);
                while (!wake_.wait_for(lock, std::chrono::milliseconds(25), [this] { return quit_; })) {
                    check_locked();
                }
            });
        } catch (const std::exception & error) {
            std::lock_guard<std::mutex> lock(mutex_);
            failure(error.what(), 0);
            limit_ = 0;
            return false;
        }
    }
    std::fprintf(stderr, "[mem-budget] enabled: %zu MiB, cooperative RSS reclamation (25 ms); no hard kernel cap\n",
                 bytes / (1024 * 1024));
    return true;
}

void MemoryBudget::stop() {
    { std::lock_guard<std::mutex> lock(mutex_); quit_ = true; }
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
    std::lock_guard<std::mutex> lock(mutex_);
    limit_ = 0;
    cursor_ = 0;
    stalled_rss_ = 0;
    regions_.clear();
}

bool MemoryBudget::enabled() {
    std::lock_guard<std::mutex> lock(mutex_);
    return limit_ != 0;
}

void MemoryBudget::failure(const char * reason, size_t rss) {
    if (!reported_) {
        std::fprintf(stderr, "[mem-budget] enforcement failed: %s (RSS=%zu MiB budget=%zu MiB)\n",
                     reason, rss / (1024 * 1024), limit_ / (1024 * 1024));
        reported_ = true;
    }
}

bool MemoryBudget::check_locked() {
    if (!limit_) return true;
    size_t rss = 0;
    if (!ops_.rss(rss)) { failure("cannot read process RSS", 0); return false; }
    // Small budgets need room for prefill/checkpoint bursts between polls.
    // Keep the original 5% trigger/10% target for budgets of 10 GiB or more.
    const size_t reserve = std::max(limit_ / 20, std::min(limit_ / 5, size_t(512) << 20));
    if (rss <= limit_ - reserve) { reported_ = false; stalled_rss_ = 0; return true; }
    // A pass that missed its target is repeated only after 64 MiB of RSS growth or a 2 s pause.
    if (stalled_rss_ && rss < stalled_rss_ + (size_t(64) << 20) && ops_.now() < retry_at_) return rss <= limit_;
    const size_t target = limit_ - limit_ / 10 - (reserve - limit_ / 20);
    if (!reclaim_locked(rss, target)) return false;
    stalled_rss_ = rss > target ? rss : 0;
    retry_at_ = ops_.now() + std::chrono::seconds(2);
    if (rss > limit_) { failure("non-reclaimable allocations or active reads exceed the budget", rss); return false; }
    reported_ = false;
    return true;
}

bool MemoryBudget::reclaim_locked(size_t & rss, size_t target) {
    // Rotate over model regions; anonymous/KV/LoRA/head/pool memory is never discarded.
    for (size_t visited = 0; visited < regions_.size() && rss > target; ++visited) {
        const Region & region = regions_[cursor_++ % regions_.size()];
        for (uintptr_t address = region.begin; address < region.end && rss > target;) {
            const size_t length = std::min(size_t(16) << 20, region.end - address);
            if (ops_.discard(reinterpret_cast<void *>(address), length)) {
                failure(std::strerror(errno), rss);
                return false;
            }
            address += length;
            if (!ops_.rss(rss)) { failure("cannot read process RSS after reclamation", 0); return false; }
        }
    }
    return true;
}

bool MemoryBudget::check() {
    std::lock_guard<std::mutex> lock(mutex_);
    return check_locked();
}

void MemoryBudget::mapped(void * address, size_t length) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!limit_ || !length) return;
    const uintptr_t begin = reinterpret_cast<uintptr_t>(address);
    if (begin % page_size() || length > UINTPTR_MAX - begin) {
        failure("invalid model mapping", 0);
        return;
    }
    regions_.push_back({begin, begin + length});
}

int MemoryBudget::unmap(void * address, size_t length) {
    std::lock_guard<std::mutex> lock(mutex_);
    const int result = munmap(address, length);
    if (result) return result;
    const uintptr_t begin = reinterpret_cast<uintptr_t>(address);
    // The kernel rounds the end of an unmapping to a whole page.
    const size_t padding = (page_size() - length % page_size()) % page_size();
    const uintptr_t end = begin + length + padding;
    std::vector<Region> remaining;
    for (const Region & region : regions_) {
        if (region.begin >= end || region.end <= begin) remaining.push_back(region);
        else {
            if (region.begin < begin) remaining.push_back({region.begin, begin});
            if (region.end > end) remaining.push_back({end, region.end});
        }
    }
    regions_ = std::move(remaining);
    return 0;
}

} // namespace edge0

namespace {
// Never destroyed: a forced exit may run static destructors while another thread holds the budget lock.
edge0::MemoryBudget & process_budget() { static auto & instance = *new edge0::MemoryBudget; return instance; }
}

bool edge0_apply_mem_budget(int mib) {
    size_t bytes = 0;
    if (!edge0::resolve_mib(mib, std::getenv("E0_MEM_BUDGET_MB"), bytes)) {
        std::fprintf(stderr, "[mem-budget] invalid MiB budget; expected a nonnegative decimal integer\n");
        return false;
    }
    if (!bytes) { process_budget().stop(); return true; }  // no budget requested: dormant, not a failure
    return process_budget().start(bytes);
}

bool edge0_mem_budget_enabled() { return process_budget().enabled(); }
bool edge0_mem_budget_check() { return process_budget().check(); }
void edge0_mem_budget_shutdown() { process_budget().stop(); }
void edge0_mem_budget_mapped(void * address, size_t length) { process_budget().mapped(address, length); }
int edge0_mem_budget_munmap(void * address, size_t length) { return process_budget().unmap(address, length); }
