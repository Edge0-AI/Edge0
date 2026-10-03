#include "hotstack.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>

namespace e0n {

HotStack::HotStack(std::shared_ptr<const ModelFiles> files,
                   std::shared_ptr<const ExpertMap> map, uint64_t budget_bytes)
    : files_(std::move(files)), map_(std::move(map)), budget_(budget_bytes) {}

HotStack::~HotStack() {
  std::lock_guard<std::mutex> g(mu_);
  for (auto &[s, fd] : fds_) ::close(fd);
}

bool HotStack::load_entry(int layer, int64_t expert, EntryPtr &out, std::string &err) {
  auto rs = map_->regions(layer, expert);
  if (rs.size() != 9) {
    err = "region count != 9 (L" + std::to_string(layer) + " E" + std::to_string(expert) + ")";
    return false;
  }
  auto e = std::make_shared<Entry>();
  e->layer = layer;
  e->expert = expert;
  e->regions = rs;
  uint64_t total = 0;
  for (auto &r : rs) total += r.len;
  e->bytes = total;
  e->blob.resize(total);
  e->blob_offsets.resize(rs.size());
  uint64_t done_logical = 0;
  for (size_t i = 0; i < rs.size(); i++) {
    const Region &r = rs[i];
    e->blob_offsets[i] = done_logical;
    int fd;
    {
      // fds_ is used outside the caller's lock (this runs in acquire's unlocked
      // section); opening a new fd still needs a short lock on fds_.
      std::lock_guard<std::mutex> g(mu_);
      auto it = fds_.find(r.shard);
      if (it == fds_.end()) {
        fd = ::open(files_->shard_path(r.shard).c_str(), O_RDONLY);
        if (fd < 0) {
          err = "open shard failed: " + r.shard;
          return false;
        }
        // Load fds are *not* F_NOCACHE: an evicted reload should reuse the OS page cache.
        fds_[r.shard] = fd;
      } else {
        fd = it->second;
      }
    }
    uint8_t *dst = e->blob.data() + done_logical;
    uint64_t got_total = 0;
    while (got_total < r.len) {
      uint64_t want = std::min<uint64_t>(r.len - got_total, 1u << 20);
      ssize_t got = ::pread(fd, dst + got_total, want, r.off + got_total);
      if (got <= 0) {
        err = "pread failed " + r.shard;
        return false;
      }
      got_total += static_cast<uint64_t>(got);
    }
    done_logical += r.len;
  }
  out = std::move(e);
  return true;
}

HotStack::Lease HotStack::acquire(int layer, int64_t expert, std::string &err) {
  const uint64_t k = key_of(layer, expert);
  // 1) hit (fast path).
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = items_.find(k);
    if (it != items_.end()) {
      lru_.splice(lru_.begin(), lru_, it->second.lru);  // MRU
      metrics_.hits++;
      return it->second.entry;
    }
    metrics_.misses++;
  }
  // 2) disk read outside the lock.
  EntryPtr entry;
  if (!load_entry(layer, expert, entry, err)) return nullptr;
  // 3) insert + evict (double-check: a concurrent load of the same key discards ours).
  {
    std::lock_guard<std::mutex> g(mu_);
    if (auto it = items_.find(k); it != items_.end()) {
      lru_.splice(lru_.begin(), lru_, it->second.lru);
      return it->second.entry;  // drop this thread's duplicate load (bytes not counted)
    }
    lru_.push_front(k);
    metrics_.heap_bytes += entry->bytes;
    metrics_.heap_peak = std::max(metrics_.heap_peak, metrics_.heap_bytes);
    metrics_.loaded++;
    items_.emplace(k, Item{entry, lru_.begin()});
    Lease ret = entry;  // hold a lease first so eviction cannot drop the entry we just inserted
    // LRU eviction from the tail; skip entries still referenced by a lease (use_count>1).
    if (metrics_.heap_bytes > budget_) {
      auto it = lru_.end();
      while (metrics_.heap_bytes > budget_ && it != lru_.begin()) {
        --it;
        auto iit = items_.find(*it);
        if (iit == items_.end()) { it = lru_.begin(); continue; }
        if (iit->second.entry.use_count() > 1) continue;  // in use: skip this round
        metrics_.heap_bytes -= iit->second.entry->bytes;
        metrics_.evictions++;
        lru_.erase(iit->second.lru);
        items_.erase(iit);
        it = lru_.end();  // iterator invalid after erase; restart from the tail
      }
    }
    return ret;
  }
}

CacheMetrics HotStack::snapshot() const {
  std::lock_guard<std::mutex> g(mu_);
  return metrics_;
}

size_t HotStack::size() const {
  std::lock_guard<std::mutex> g(mu_);
  return items_.size();
}

}  // namespace e0n
